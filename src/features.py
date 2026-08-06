"""Feature extraction for wrist-IMU exercise recordings.

Each recording (one CSV = one set) is reduced to a single fixed-length vector of
descriptive motion statistics, so a whole set can be classified as one sample.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.signal import find_peaks

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "Whales1and2_Raw_Labelled"
OUTPUT_CSV = Path(__file__).resolve().parents[1] / "features.csv"

# Classes with enough sets to train on.
KEPT_CLASSES = ["APULL", "SACLR", "SAOCTE", "MTE", "IDBC", "30DBP", "NGCR"]

ACCEL_AXES = ["wristMotion_accelerationX", "wristMotion_accelerationY", "wristMotion_accelerationZ"]
ROT_AXES = ["wristMotion_rotationRateX", "wristMotion_rotationRateY", "wristMotion_rotationRateZ"]
GRAVITY_AXES = ["wristMotion_gravityX", "wristMotion_gravityY", "wristMotion_gravityZ"]
MOTION_COLS = ACCEL_AXES + ROT_AXES + GRAVITY_AXES

SAMPLE_RATE_HZ = 100.0

# Reps in strength work land around 0.25-0.7 Hz (a 1.5-4s rep), so the dominant
# frequency is searched in a band wide enough to hold the cadence and its first
# harmonic without letting sensor noise win the argmax.
REP_BAND_HZ = (0.1, 5.0)

# Fractions of spectral energy per band: cadence, first harmonic, sharp direction
# changes, then tremor/impact content.
FREQ_BANDS: dict[str, tuple[float, float]] = {
    "cadence": (0.1, 0.5),
    "harmonic": (0.5, 1.5),
    "transition": (1.5, 4.0),
    "tremor": (4.0, 15.0),
}

# Plausible rep periods, used to bound the autocorrelation peak search.
PERIOD_RANGE_S = (0.5, 15.0)


def parse_activity(filename: str) -> str:
    """Pull the activity code out of a filename like 011224_APULL_W61_S1_R12-....csv."""
    parts = filename.split("_")
    if len(parts) < 2:
        raise ValueError(f"cannot parse activity code from {filename!r}")
    return parts[1]


def load_recording(path: Path) -> pd.DataFrame:
    """Load one recording and drop the leading rows where the IMU has not reported yet.

    The sensor stream starts ~0.24s after logging begins, leaving ~23 all-NaN motion
    rows at the top of every file. Anything still missing after that point is dropped
    too, so the statistics below never see a NaN.
    """
    df = pd.read_csv(path)

    missing = [c for c in MOTION_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")

    motion = df[MOTION_COLS]
    first_valid = motion.notna().all(axis=1).idxmax() if motion.notna().all(axis=1).any() else None
    if first_valid is None:
        raise ValueError("no rows with complete motion data")

    return df.loc[first_valid:].dropna(subset=MOTION_COLS).reset_index(drop=True)


def _magnitude(df: pd.DataFrame, axes: list[str]) -> np.ndarray:
    """Euclidean norm across the three axes, per sample."""
    return np.sqrt((df[axes] ** 2).sum(axis=1)).to_numpy()


def _spectral_features(sig: np.ndarray, prefix: str) -> dict[str, float]:
    """Dominant frequency and per-band energy share of one magnitude signal.

    The mean is removed first so DC does not swamp the spectrum, and band energies
    are reported as fractions of total AC power — that keeps them comparable between
    a heavy set that shakes the wrist hard and a light one that does not.
    """
    centred = sig - sig.mean()
    if len(centred) < 4 or not np.any(centred):
        empty = {f"{prefix}_dom_freq": 0.0, f"{prefix}_dom_freq_share": 0.0}
        empty.update({f"{prefix}_band_{name}": 0.0 for name in FREQ_BANDS})
        return empty

    freqs = np.fft.rfftfreq(len(centred), d=1.0 / SAMPLE_RATE_HZ)
    power = np.abs(np.fft.rfft(centred)) ** 2
    total = power[1:].sum()  # drop the DC bin

    in_band = (freqs >= REP_BAND_HZ[0]) & (freqs <= REP_BAND_HZ[1])
    if in_band.any() and power[in_band].sum() > 0:
        peak = int(np.flatnonzero(in_band)[np.argmax(power[in_band])])
        dom_freq = float(freqs[peak])
        dom_share = float(power[peak] / total) if total > 0 else 0.0
    else:
        dom_freq, dom_share = 0.0, 0.0

    out = {f"{prefix}_dom_freq": dom_freq, f"{prefix}_dom_freq_share": dom_share}
    for name, (lo, hi) in FREQ_BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        out[f"{prefix}_band_{name}"] = float(power[mask].sum() / total) if total > 0 else 0.0
    return out


def _autocorrelation_features(sig: np.ndarray, prefix: str, dom_freq: float) -> dict[str, float]:
    """How strongly the signal repeats, and at what period.

    A clean set of reps gives a tall autocorrelation bump at the rep period; a ragged
    or drifting movement does not. Two readings are taken: the ACF evaluated at the
    period the FFT called dominant, and the tallest genuine local maximum in the
    plausible rep-period window.

    The peak search looks for local maxima rather than the plain argmax — the ACF of
    a 100Hz magnitude signal decays monotonically from lag 0, so an argmax over the
    window just returns the window's left edge and encodes nothing.
    """
    empty = {f"{prefix}_ac_peak": 0.0, f"{prefix}_ac_period_s": 0.0, f"{prefix}_ac_at_dom": 0.0}

    centred = sig - sig.mean()
    n = len(centred)
    lo_lag = int(PERIOD_RANGE_S[0] * SAMPLE_RATE_HZ)
    hi_lag = min(int(PERIOD_RANGE_S[1] * SAMPLE_RATE_HZ), n // 2)
    if n < 4 or hi_lag <= lo_lag or not np.any(centred):
        return empty

    # FFT-based autocorrelation, zero-padded to avoid circular wraparound.
    padded = int(2 ** np.ceil(np.log2(2 * n)))
    spectrum = np.fft.rfft(centred, n=padded)
    acf = np.fft.irfft(np.abs(spectrum) ** 2, n=padded)[:n]
    if acf[0] <= 0:
        return empty
    acf = acf / acf[0]

    out = dict(empty)
    if dom_freq > 0:
        lag = int(round(SAMPLE_RATE_HZ / dom_freq))
        if 0 < lag < n:
            out[f"{prefix}_ac_at_dom"] = float(acf[lag])

    window = acf[lo_lag:hi_lag]
    peaks, _ = find_peaks(window)
    if peaks.size:
        best = int(peaks[np.argmax(window[peaks])])
        out[f"{prefix}_ac_peak"] = float(window[best])
        out[f"{prefix}_ac_period_s"] = float((lo_lag + best) / SAMPLE_RATE_HZ)
    return out


def _shape_features(sig: np.ndarray, prefix: str) -> dict[str, float]:
    """Asymmetry and peakedness of the magnitude distribution.

    Skew separates movements with a hard concentric burst and a slow eccentric from
    ones that are symmetric in effort; kurtosis picks out spiky versus smooth motion.
    """
    return {
        f"{prefix}_skew": float(sp_stats.skew(sig)),
        f"{prefix}_kurtosis": float(sp_stats.kurtosis(sig)),
    }


def _cross_axis_correlations(df: pd.DataFrame, axes: list[str], prefix: str) -> dict[str, float]:
    """Pearson correlation between each pair of axes.

    This is what tells a movement in one plane from a movement that sweeps two axes
    together — pure summary statistics cannot see it, because each axis is described
    in isolation.
    """
    out: dict[str, float] = {}
    for (i, a), (j, b) in ((p, q) for p in enumerate(axes) for q in enumerate(axes) if p[0] < q[0]):
        pair = "xyz"[i] + "xyz"[j]
        x, y = df[a].to_numpy(), df[b].to_numpy()
        if x.std() == 0 or y.std() == 0:
            out[f"{prefix}_corr_{pair}"] = 0.0
        else:
            out[f"{prefix}_corr_{pair}"] = float(np.corrcoef(x, y)[0, 1])
    return out


def features_from_frame(df: pd.DataFrame) -> dict[str, float]:
    """Compute the feature vector for an already-loaded, already-cleaned recording.

    Split out from extract_features so augmented copies of a signal can be featurised
    in memory without round-tripping through a file.
    """
    accel_mag = _magnitude(df, ACCEL_AXES)
    rot_mag = _magnitude(df, ROT_AXES)

    features: dict[str, float | str | int] = {
        "accel_mag_mean": float(accel_mag.mean()),
        "accel_mag_std": float(accel_mag.std(ddof=1)),
        "accel_mag_max": float(accel_mag.max()),
        "rot_mag_mean": float(rot_mag.mean()),
        "rot_mag_std": float(rot_mag.std(ddof=1)),
        "rot_mag_max": float(rot_mag.max()),
    }

    # Per-axis location and spread for acceleration and rotation rate.
    for prefix, axes in (("accel", ACCEL_AXES), ("rot", ROT_AXES)):
        for axis_name, col in zip("xyz", axes):
            values = df[col].to_numpy()
            features[f"{prefix}_{axis_name}_mean"] = float(values.mean())
            features[f"{prefix}_{axis_name}_std"] = float(values.std(ddof=1))

    # Mean gravity vector: where the wrist is held, averaged over the set.
    for axis_name, col in zip("xyz", GRAVITY_AXES):
        features[f"gravity_{axis_name}_mean"] = float(df[col].mean())

    # Per-axis motion energy.
    for prefix, axes in (("accel", ACCEL_AXES), ("rot", ROT_AXES)):
        for axis_name, col in zip("xyz", axes):
            features[f"{prefix}_{axis_name}_energy"] = float(df[col].var(ddof=1))

    # Temporal and shape descriptors: what the movement does over time, rather than
    # what it averages out to.
    for prefix, mag in (("accel_mag", accel_mag), ("rot_mag", rot_mag)):
        spectral = _spectral_features(mag, prefix)
        features.update(spectral)
        features.update(_autocorrelation_features(mag, prefix, spectral[f"{prefix}_dom_freq"]))
        features.update(_shape_features(mag, prefix))

    features.update(_cross_axis_correlations(df, ACCEL_AXES, "accel"))
    features.update(_cross_axis_correlations(df, ROT_AXES, "rot"))
    return features


def extract_features(path: Path) -> dict[str, float | str | int]:
    """Reduce one recording to a fixed feature vector plus its label."""
    df = load_recording(path)

    features: dict[str, float | str | int] = dict(features_from_frame(df))
    features["n_samples"] = int(len(df))
    features["recording"] = path.name
    features["activity"] = parse_activity(path.name)
    return features


def build_feature_table(
    data_dir: Path = DATA_DIR,
    classes: list[str] | None = None,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Extract features for every recording in the kept classes.

    Returns the feature table and a list of (filename, error) for anything that failed.
    """
    classes = KEPT_CLASSES if classes is None else classes
    kept = set(classes)

    rows: list[dict[str, float | str | int]] = []
    failures: list[tuple[str, str]] = []

    for path in sorted(data_dir.glob("*.csv")):
        try:
            if parse_activity(path.name) not in kept:
                continue
            rows.append(extract_features(path))
        except Exception as exc:  # keep going; report at the end
            failures.append((path.name, f"{type(exc).__name__}: {exc}"))

    table = pd.DataFrame(rows)
    if not table.empty:
        meta = ["recording", "activity", "n_samples"]
        table = table[[c for c in table.columns if c not in meta] + meta]
        table = table.sort_values(["activity", "recording"]).reset_index(drop=True)
    return table, failures


def main() -> int:
    table, failures = build_feature_table()

    if table.empty:
        print(f"No recordings processed from {DATA_DIR}")
        return 1

    table.to_csv(OUTPUT_CSV, index=False)

    feature_cols = [c for c in table.columns if c not in ("recording", "activity", "n_samples")]
    print(f"Recordings processed : {len(table)}")
    print(f"Features per recording: {len(feature_cols)}")
    print(f"Saved to             : {OUTPUT_CSV}")

    print("\nClass distribution:")
    for activity, count in table["activity"].value_counts().items():
        print(f"  {activity:<8} {count}")

    print("\nFirst rows:")
    with pd.option_context("display.width", 200, "display.max_columns", 12):
        print(table[["recording", "activity", "n_samples", *feature_cols[:6]]].head())

    if failures:
        print(f"\nFailed to process ({len(failures)}):")
        for name, err in failures:
            print(f"  {name}: {err}")
    else:
        print("\nFailed to process: none")

    return 0


if __name__ == "__main__":
    sys.exit(main())
