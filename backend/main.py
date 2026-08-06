"""FastAPI service around the saved exercise classifier.

Loads the four artifacts written by src/train_final.py once at startup, then serves
two endpoints: one listing the recordings available to try, one running a chosen
recording through the full feature-extraction -> scale -> predict path.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import (  # noqa: E402
    ACCEL_AXES,
    DATA_DIR,
    KEPT_CLASSES,
    SAMPLE_RATE_HZ,
    extract_features,
    load_recording,
    parse_activity,
)

# Enough resolution to see individual reps without shipping 46k points to a browser.
DEFAULT_SIGNAL_POINTS = 400

MODEL_DIR = ROOT / "models"

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

    def predict(self, features: dict[str, float]) -> tuple[str, float, dict[str, float]]:
        """Order, scale, and classify one feature dict."""
        vector = pd.DataFrame(
            [[features[c] for c in self.feature_columns]], columns=self.feature_columns
        )
        proba = self.model.predict_proba(self.scaler.transform(vector))[0]
        distribution = {
            str(label): float(p) for label, p in zip(self.model.classes_, proba)
        }
        predicted = max(distribution, key=distribution.get)
        return predicted, distribution[predicted], distribution


def discover_recordings(data_dir: Path) -> dict[str, str]:
    """Map filename -> true label for the kept classes.

    Doubles as the allow-list for /predict: a filename that is not a key here is
    never opened, so the endpoint cannot be walked out of the data directory.
    """
    kept = set(KEPT_CLASSES)
    found: dict[str, str] = {}
    for path in sorted(data_dir.glob("*.csv")):
        try:
            activity = parse_activity(path.name)
        except ValueError:
            continue
        if activity in kept:
            found[path.name] = activity
    return found


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.artifacts = Artifacts(MODEL_DIR)
    app.state.recordings = discover_recordings(DATA_DIR)
    print(
        f"Loaded {len(app.state.artifacts.labels)} classes, "
        f"{len(app.state.artifacts.feature_columns)} features, "
        f"{len(app.state.recordings)} recordings"
    )
    yield


app = FastAPI(
    title="Exercise Detection from Wrist Motion",
    description="Classify which strength exercise a wrist-IMU recording contains.",
    version="1.0.0",
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
    filename: str
    activity: str = Field(description="True label parsed from the filename")


class RecordingsResponse(BaseModel):
    count: int
    classes: list[str]
    recordings: list[Recording]


class SignalPoint(BaseModel):
    t: float = Field(description="Seconds from the start of the recording")
    value: float = Field(description="Acceleration magnitude in g")


class SignalResponse(BaseModel):
    filename: str
    activity: str
    duration_s: float
    n_samples: int = Field(description="Samples in the source recording")
    n_points: int = Field(description="Points after downsampling")
    peak: float
    series: list[SignalPoint]


class PredictRequest(BaseModel):
    filename: str = Field(description="A filename returned by GET /recordings")


class PredictResponse(BaseModel):
    filename: str
    predicted: str
    confidence: float
    probabilities: dict[str, float]
    true_label: str
    correct: bool


@app.get("/recordings", response_model=RecordingsResponse)
def list_recordings() -> dict[str, Any]:
    """Every recording the service can classify, with its ground-truth label."""
    recordings = app.state.recordings
    return {
        "count": len(recordings),
        "classes": app.state.artifacts.labels,
        "recordings": [
            {"filename": name, "activity": activity} for name, activity in recordings.items()
        ],
    }


@app.get("/signal", response_model=SignalResponse)
def signal(
    filename: str = Query(description="A filename returned by GET /recordings"),
    points: int = Query(DEFAULT_SIGNAL_POINTS, ge=50, le=4000),
) -> dict[str, Any]:
    """Acceleration magnitude over time, downsampled for plotting.

    Downsampling averages within equal-width blocks rather than taking every Nth
    sample. Stride sampling at this ratio would alias the rep cadence and could drop
    peaks entirely; block means keep the envelope the reps actually trace.
    """
    name = Path(filename).name
    activity = app.state.recordings.get(name)
    if activity is None:
        raise HTTPException(status_code=404, detail=f"unknown recording: {filename}")

    try:
        df = load_recording(DATA_DIR / name)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"could not read recording: {type(exc).__name__}: {exc}"
        ) from exc

    magnitude = np.sqrt((df[ACCEL_AXES].to_numpy(float) ** 2).sum(axis=1))
    n = len(magnitude)
    duration = n / SAMPLE_RATE_HZ

    n_points = min(points, n)
    edges = np.linspace(0, n, n_points + 1).astype(int)
    series = [
        {
            "t": round(float(edges[i] / SAMPLE_RATE_HZ), 3),
            "value": round(float(magnitude[edges[i] : max(edges[i + 1], edges[i] + 1)].mean()), 5),
        }
        for i in range(n_points)
    ]

    return {
        "filename": name,
        "activity": activity,
        "duration_s": round(duration, 2),
        "n_samples": n,
        "n_points": len(series),
        "peak": round(float(magnitude.max()), 5),
        "series": series,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> dict[str, Any]:
    """Run one recording end to end: raw CSV -> features -> scale -> class."""
    filename = Path(request.filename).name  # strip any directory component
    true_label = app.state.recordings.get(filename)
    if true_label is None:
        raise HTTPException(status_code=404, detail=f"unknown recording: {request.filename}")

    try:
        features = extract_features(DATA_DIR / filename)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"feature extraction failed: {type(exc).__name__}: {exc}"
        ) from exc

    predicted, confidence, distribution = app.state.artifacts.predict(features)
    return {
        "filename": filename,
        "predicted": predicted,
        "confidence": confidence,
        "probabilities": distribution,
        "true_label": true_label,
        "correct": predicted == true_label,
    }
