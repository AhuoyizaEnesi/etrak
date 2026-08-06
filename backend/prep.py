"""
prep.py — validate and prepare an uploaded Apple-format wrist CSV
so it can go straight into the existing feature extraction.

Apple data is already in real units, gravity-split, 100Hz — so no unit
conversion needed. This just makes an uploaded file SAFE to run:
  1. confirm required sensor columns exist (clear error if not)
  2. drop leading warm-up rows where sensor columns are empty
  3. coerce sensor columns to numeric, drop fully-empty rows
  4. return a clean dataframe + a status report
"""

import pandas as pd
import numpy as np

REQUIRED = [
    "wristMotion_accelerationX", "wristMotion_accelerationY", "wristMotion_accelerationZ",
    "wristMotion_rotationRateX", "wristMotion_rotationRateY", "wristMotion_rotationRateZ",
    "wristMotion_gravityX", "wristMotion_gravityY", "wristMotion_gravityZ",
]


class PrepError(Exception):
    pass


def prep(df):
    """Returns (clean_df, report dict). Raises PrepError with a clear message if unusable."""
    report = {}

    # 1. required columns present?
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise PrepError(
            f"Missing required columns: {missing}. "
            f"Expected Apple Watch wrist-motion format."
        )

    n0 = len(df)

    # 2. coerce sensor cols numeric (blank strings -> NaN)
    for c in REQUIRED:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 3. drop leading warm-up rows where all sensor cols are NaN
    sensor_all_nan = df[REQUIRED].isna().all(axis=1)
    first_valid = (~sensor_all_nan).idxmax() if (~sensor_all_nan).any() else None
    if first_valid is None:
        raise PrepError("No rows contain sensor data.")
    df = df.loc[first_valid:].reset_index(drop=True)
    dropped_lead = n0 - len(df)

    # 4. drop any remaining rows that are still fully empty in sensors
    still_nan = df[REQUIRED].isna().all(axis=1)
    df = df[~still_nan].reset_index(drop=True)

    # 5. must have enough rows to be a real set
    if len(df) < 100:  # <1s at 100Hz
        raise PrepError(f"Only {len(df)} usable rows after cleaning; too short to classify.")

    report["rows_in"] = n0
    report["rows_out"] = len(df)
    report["leading_warmup_dropped"] = int(dropped_lead)
    report["has_secondsElapsed"] = "secondsElapsed" in df.columns

    return df, report
