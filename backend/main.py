"""FastAPI service around the saved exercise classifier.

Predictions are served from the precomputed feature table (features.csv) rather
than from raw recordings. The raw CSVs are far too large to deploy, but every one
of them is already represented in features.csv by the feature vector the model
actually consumes — so nothing is lost at inference time.

Consequences of that choice, all deliberate:
  * no filesystem access to data/
  * no /signal endpoint, because the raw waveform is not available
  * only recordings present in features.csv can be classified

/predict-upload is the exception: an uploaded recording has no precomputed row, so
that path does run the real feature extraction from src/features.py.
"""

from __future__ import annotations

import io
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
FEATURES_CSV = ROOT / "features.csv"

# Uploads need the real extractor (src/) and the upload validator (backend/).
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

from features import features_from_frame  # noqa: E402
from prep import REQUIRED, PrepError, prep  # noqa: E402

# Refuse absurd uploads before pandas tries to parse them.
MAX_UPLOAD_BYTES = 64 * 1024 * 1024

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
    n_samples: int | None = Field(
        default=None, description="Rows in the source recording, when the table records it"
    )


class UploadPredictResponse(BaseModel):
    """No true_label or correct here - an uploaded recording has no known label."""

    filename: str
    predicted: str
    confidence: float
    probabilities: dict[str, float]
    report: dict[str, Any] = Field(description="What prep() validated and trimmed")


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


@app.post("/predict-upload", response_model=UploadPredictResponse)
async def predict_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    """Classify an uploaded Apple-format wrist CSV.

    Unlike /predict, there is no precomputed row to look up, so this runs the same
    feature extraction the training pipeline used - prep() first, to validate and
    trim the upload, then features_from_frame() on the cleaned frame.
    """
    artifacts = app.state.artifacts

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="uploaded file is empty")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"file is {len(raw) / 1e6:.1f} MB; limit is {MAX_UPLOAD_BYTES / 1e6:.0f} MB",
        )

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"could not parse CSV: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        clean, report = prep(df)
    except PrepError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # prep() drops rows where every sensor column is blank; rows blank in only some
    # columns survive and would poison the statistics, so clear them the same way
    # the training loader does before extracting.
    before = len(clean)
    clean = clean.dropna(subset=REQUIRED).reset_index(drop=True)
    report = {**report, "partial_rows_dropped": before - len(clean)}
    if len(clean) < 100:
        raise HTTPException(
            status_code=400,
            detail=f"only {len(clean)} complete rows after cleaning; too short to classify",
        )

    try:
        features = features_from_frame(clean)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"feature extraction failed: {type(exc).__name__}: {exc}",
        ) from exc

    absent = [c for c in artifacts.feature_columns if c not in features]
    if absent:
        raise HTTPException(
            status_code=500,
            detail=f"extractor did not produce {len(absent)} expected feature(s): "
            f"{', '.join(absent[:5])}",
        )

    vector = pd.DataFrame(
        [[features[c] for c in artifacts.feature_columns]], columns=artifacts.feature_columns
    )
    if vector.isna().any().any():
        blank = vector.columns[vector.isna().iloc[0]].tolist()
        raise HTTPException(
            status_code=400,
            detail=f"extracted features contain missing values: {', '.join(blank[:5])}",
        )

    predicted, confidence, distribution = artifacts.predict(vector)
    return {
        "filename": file.filename or "upload.csv",
        "predicted": predicted,
        "confidence": confidence,
        "probabilities": distribution,
        "report": report,
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

    n_samples = None
    if "n_samples" in table.columns:
        n_samples = int(table.loc[name, "n_samples"])

    return {
        "filename": name,
        "predicted": predicted,
        "confidence": confidence,
        "probabilities": distribution,
        "true_label": true_label,
        "correct": predicted == true_label,
        "n_samples": n_samples,
    }
