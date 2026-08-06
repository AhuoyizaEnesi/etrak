"""In-memory augmentation of raw wrist-IMU recordings.

Augmented copies exist only inside a training fold. They are never written to
features.csv and never enter a validation fold — the reported accuracy is always
measured on real recordings only.

The four transforms each target a nuisance factor the classifier should ignore:

  rotation   the watch sits at a slightly different angle on the wrist
  amplitude  the same movement performed with more or less force
  time scale the same movement performed at a different tempo
  jitter     sensor noise

Rotation is the most valuable of the four here. The mean-gravity features are the
single strongest block in the model, and gravity encodes wrist attitude — which is
partly a property of *how the watch was strapped on that day* rather than of the
exercise. Rotating the sensor frame forces the forest to tolerate that variation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from features import (
    ACCEL_AXES,
    DATA_DIR,
    GRAVITY_AXES,
    MOTION_COLS,
    ROT_AXES,
    load_recording,
    parse_activity,
)

# Vector triplets that must rotate together to stay physically consistent.
VECTOR_TRIPLETS = (ACCEL_AXES, ROT_AXES, GRAVITY_AXES)

MAX_ROTATION_DEG = 15.0
AMPLITUDE_RANGE = (0.90, 1.10)
TIME_SCALE_RANGE = (0.90, 1.10)
JITTER_SIGMA = 0.02  # as a fraction of each axis's own standard deviation


def load_raw_cache(names: list[str], data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load every recording's motion columns once, so folds re-augment from memory."""
    cache: dict[str, pd.DataFrame] = {}
    for name in names:
        cache[name] = load_recording(data_dir / name)[MOTION_COLS].astype(float)
    return cache


def _rotate(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Rotate the whole sensor frame by a small random 3D rotation.

    The same rotation is applied to acceleration, rotation rate and gravity, because
    all three are expressed in the device frame; rotating them independently would
    describe a physically impossible sensor.
    """
    axis = rng.normal(size=3)
    norm = np.linalg.norm(axis)
    if norm == 0:
        return df
    angle = np.deg2rad(rng.uniform(-MAX_ROTATION_DEG, MAX_ROTATION_DEG))
    matrix = Rotation.from_rotvec(angle * axis / norm).as_matrix()

    out = df.copy()
    for triplet in VECTOR_TRIPLETS:
        out[triplet] = df[triplet].to_numpy() @ matrix.T
    return out


def _scale_amplitude(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Scale motion intensity, leaving gravity alone — gravity is a unit direction."""
    factor = rng.uniform(*AMPLITUDE_RANGE)
    out = df.copy()
    out[ACCEL_AXES + ROT_AXES] = df[ACCEL_AXES + ROT_AXES].to_numpy() * factor
    return out


def _time_scale(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Resample to a different duration, i.e. perform the same reps faster or slower.

    The sample rate is still read as 100Hz downstream, so stretching the series is
    what shifts the cadence the FFT and autocorrelation features see.
    """
    factor = rng.uniform(*TIME_SCALE_RANGE)
    n = len(df)
    new_n = max(16, int(round(n / factor)))
    src = np.arange(n)
    dst = np.linspace(0, n - 1, new_n)
    return pd.DataFrame(
        {col: np.interp(dst, src, df[col].to_numpy()) for col in df.columns},
        columns=df.columns,
    )


def _jitter(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add per-axis Gaussian noise proportional to that axis's own spread."""
    out = df.copy()
    values = df.to_numpy()
    sigma = values.std(axis=0, keepdims=True) * JITTER_SIGMA
    return pd.DataFrame(values + rng.normal(scale=np.maximum(sigma, 1e-12), size=values.shape),
                        columns=df.columns)


def augment_frame(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Apply the full transform chain to one recording, with fresh random draws."""
    out = _rotate(df, rng)
    out = _scale_amplitude(out, rng)
    out = _time_scale(out, rng)
    return _jitter(out, rng)


__all__ = ["load_raw_cache", "augment_frame", "parse_activity"]
