"""FastAPI service around the saved exercise classifier.

Predictions are served from the precomputed feature table (features.csv) rather
than from raw recordings. The raw CSVs are far too large to deploy, but every one
of them is already represented in features.csv by the feature vector the model
actually consumes — so nothing is lost at inference time.

Consequences of that choice, all deliberate:
  * no filesystem access to data/ and no dependency on src/features.py
  * no /signal endpoint, because the raw waveform is not available
  * only recordings present in features.csv can be classified
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
FEATURES_CSV = ROOT / "features.csv"

# Columns in features.csv that describe the recording rather than the movement.
ID_COLUMN = "recording"
LABEL_COLUMN = "activity"
META_COLUMNS = (ID_COLUMN, LABEL_COLUMN, "n_samples")

# Comma-separated origins, or "*" for any. Defaults open so a frontend on a
# different port or domain can call this during development.
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")


class Artifacts:
    """The loaded model and everything needed to feed it consistently."""

    def __init__(self, model_dir: Path) -> None:
        missing = [
            f.name
            for f in (
                model_dir / "model.pkl",
                model_dir / "scaler.pkl",
                model_dir / "feature_columns.json",
                model_dir / "labels.json",
            )
            if not f.exists()
        ]
        if missing:
            raise RuntimeError(
                f"missing artifacts in {model_dir}: {', '.join(missing)} "
                "- run python src/train_final.py first"
            )

        self.model = joblib.load(model_dir / "model.pkl")
        self.scaler = joblib.load(model_dir / "scaler.pkl")
        self.feature_columns: list[str] = json.loads(
            (model_dir / "feature_columns.json").read_text()
        )
        self.labels: list[str] = json.loads((model_dir / "labels.json").read_text())

        # Fail loudly at startup rather than silently mis-predicting later.
        if self.scaler.n_features_in_ != len(self.feature_columns):
            raise RuntimeError(
                f"scaler expects {self.scaler.n_features_in_} features but "
                f"feature_columns.json lists {len(self.feature_columns)}"
            )

    def predict(self, vector: pd.DataFrame) -> tuple[str, float, dict[str, float]]:
        """Scale and classify one already-ordered single-row feature frame."""
        proba = self.model.predict_proba(self.scaler.transform(vector))[0]
        distribution = {str(label): float(p) for label, p in zip(self.model.classes_, proba)}
        predicted = max(distribution, key=distribution.get)
        return predicted, distribution[predicted], distribution


def load_feature_table(path: Path, feature_columns: list[str]) -> pd.DataFrame:
    """Read features.csv and index it by recording id.

    Every column the model expects must be present, and ids must be unique — both
    are checked here so a bad table fails at startup rather than at request time.
    """
    if not path.exists():
        raise RuntimeError(f"{path} not found - run python src/features.py first")

    table = pd.read_csv(path)

    for column in (ID_COLUMN, LABEL_COLUMN):
        if column not in table.columns:
            raise RuntimeError(f"{path} has no '{column}' column")

    absent = [c for c in feature_columns if c not in table.columns]
    if absent:
        raise RuntimeError(
            f"{path} is missing {len(absent)} feature column(s) the model expects: "
            f"{', '.join(absent[:5])}{' ...' if len(absent) > 5 else ''}"
        )

    duplicated = table[ID_COLUMN].duplicated()
    if duplicated.any():
        raise RuntimeError(
            f"{path} has duplicate ids: {', '.join(table.loc[duplicated, ID_COLUMN].head())}"
        )

    return table.set_index(ID_COLUMN, drop=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifacts = Artifacts(MODEL_DIR)
    table = load_feature_table(FEATURES_CSV, artifacts.feature_columns)

    app.state.artifacts = artifacts
    app.state.table = table
    print(
        f"Loaded {len(artifacts.labels)} classes, "
        f"{len(artifacts.feature_columns)} features, "
        f"{len(table)} recordings from {FEATURES_CSV.name}"
    )
    yield


app = FastAPI(
    title="Exercise Detection from Wrist Motion",
    description="Classify which strength exercise a wrist-IMU recording contains.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOWED_ORIGINS == "*" else [o.strip() for o in ALLOWED_ORIGINS.split(",")],
    allow_credentials=False,  # cannot be combined with a wildcard origin
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class Recording(BaseModel):
    filename: str = Field(description="Recording id, as stored in features.csv")
    activity: str = Field(description="True label")


class RecordingsResponse(BaseModel):
    count: int
    classes: list[str]
    recordings: list[Recording]


class PredictRequest(BaseModel):
    filename: str = Field(description="An id returned by GET /recordings")


class PredictResponse(BaseModel):
    filename: str
    predicted: str
    confidence: float
    probabilities: dict[str, float]
    true_label: str
    correct: bool


@app.get("/recordings", response_model=RecordingsResponse)
def list_recordings() -> dict[str, Any]:
    """Every recording in the feature table, with its ground-truth label."""
    table = app.state.table
    return {
        "count": len(table),
        "classes": app.state.artifacts.labels,
        "recordings": [
            {"filename": str(row[ID_COLUMN]), "activity": str(row[LABEL_COLUMN])}
            for _, row in table.iterrows()
        ],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> dict[str, Any]:
    """Look up a precomputed feature row, scale it, and classify."""
    table = app.state.table
    artifacts = app.state.artifacts

    name = request.filename
    if name not in table.index:
        raise HTTPException(status_code=404, detail=f"unknown recording: {name}")

    # Single-row frame keeps the column names, so sklearn sees the same feature
    # order it was fit with instead of a bare array.
    vector = table.loc[[name], artifacts.feature_columns]
    if len(vector) != 1:
        raise HTTPException(status_code=500, detail=f"ambiguous id: {name}")

    if vector.isna().any().any():
        blank = vector.columns[vector.isna().iloc[0]].tolist()
        raise HTTPException(
            status_code=500,
            detail=f"feature row has missing values: {', '.join(blank[:5])}",
        )

    predicted, confidence, distribution = artifacts.predict(vector)
    true_label = str(table.loc[name, LABEL_COLUMN])

    return {
        "filename": name,
        "predicted": predicted,
        "confidence": confidence,
        "probabilities": distribution,
        "true_label": true_label,
        "correct": predicted == true_label,
    }
