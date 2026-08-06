"""Fit the deployment model on every real recording and persist it.

Unlike src/train.py, which only ever fits models inside cross-validation folds and
throws them away, this script trains one final model on all 108 real recordings
(plus in-memory augmented copies) and writes the artifacts an inference service
needs: the forest, the scaler, the feature order, and the class labels.

There is deliberately no accuracy number printed here. A model scored on the data
it was fit on tells you nothing; the honest estimate comes from src/train.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from augment import augment_frame, load_raw_cache
from features import extract_features, features_from_frame

ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = ROOT / "features.csv"
MODEL_DIR = ROOT / "models"

META_COLS = ["recording", "activity", "n_samples"]
LABEL_COL = "activity"
N_AUGMENT = 3
RANDOM_STATE = 42


def build_training_matrix(
    table: pd.DataFrame, n_augment: int, seed: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.Series]:
    """All real recordings, plus n_augment augmented copies of each.

    Augmented rows exist only in memory and only here — they are never written back
    to features.csv, which stays a table of real recordings.
    """
    feature_cols = [c for c in table.columns if c not in META_COLS]
    X_real = table[feature_cols]
    y_real = table[LABEL_COL]

    if n_augment <= 0:
        return X_real, y_real

    cache = load_raw_cache(table["recording"].tolist())
    rng = np.random.default_rng(seed)

    rows, labels = [], []
    for name, label in zip(table["recording"], y_real):
        raw = cache[name]
        for _ in range(n_augment):
            rows.append(features_from_frame(augment_frame(raw, rng)))
            labels.append(label)

    X = pd.concat([X_real, pd.DataFrame(rows, columns=feature_cols)], ignore_index=True)
    y = pd.concat([y_real, pd.Series(labels)], ignore_index=True)
    return X, y


def fit_and_save(X: pd.DataFrame, y: pd.Series, out_dir: Path) -> dict[str, Path]:
    """Standardize, fit the forest, and write the four deployment artifacts.

    The scaler is fit on the same matrix the forest trains on, so the two stay
    consistent. A forest is scale invariant, so this does not change predictions
    today — it matters the moment the estimator is swapped for one that is not.
    """
    scaler = StandardScaler().fit(X)
    forest = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ).fit(scaler.transform(X), y)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "model": out_dir / "model.pkl",
        "scaler": out_dir / "scaler.pkl",
        "feature_columns": out_dir / "feature_columns.json",
        "labels": out_dir / "labels.json",
    }

    joblib.dump(forest, paths["model"])
    joblib.dump(scaler, paths["scaler"])
    paths["feature_columns"].write_text(json.dumps(list(X.columns), indent=2))
    paths["labels"].write_text(json.dumps(sorted(forest.classes_.tolist()), indent=2))
    return paths


def verify(out_dir: Path, table: pd.DataFrame, data_dir: Path | None = None) -> bool:
    """Reload the artifacts from disk and run one recording end to end.

    Features are recomputed from the raw CSV rather than read out of features.csv,
    so this exercises the same path a deployed service would take.
    """
    from features import DATA_DIR

    data_dir = data_dir or DATA_DIR

    forest = joblib.load(out_dir / "model.pkl")
    scaler = joblib.load(out_dir / "scaler.pkl")
    feature_columns = json.loads((out_dir / "feature_columns.json").read_text())
    labels = json.loads((out_dir / "labels.json").read_text())

    row = table.iloc[0]
    name, truth = row["recording"], row[LABEL_COL]

    computed = extract_features(data_dir / name)
    vector = pd.DataFrame([[computed[c] for c in feature_columns]], columns=feature_columns)

    prediction = forest.predict(scaler.transform(vector))[0]
    proba = forest.predict_proba(scaler.transform(vector))[0]
    confidence = float(proba[list(forest.classes_).index(prediction)])

    print(f"  recording  : {name}")
    print(f"  true label : {truth}")
    print(f"  predicted  : {prediction}  (confidence {confidence:.1%})")
    print(f"  features   : {len(feature_columns)} columns, order matches training")
    print(f"  labels     : {len(labels)} classes -> {', '.join(labels)}")

    ok = prediction == truth
    print(f"  result     : {'OK' if ok else 'MISMATCH'}")
    return ok


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURES_CSV)
    parser.add_argument("--out", type=Path, default=MODEL_DIR)
    parser.add_argument("--augment", type=int, default=N_AUGMENT, help="copies per recording")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.features.exists():
        print(f"{args.features} not found - run src/features.py first")
        return 1

    table = pd.read_csv(args.features)
    X, y = build_training_matrix(table, args.augment)

    print(f"Real recordings   : {len(table)}")
    print(f"Augmented copies  : {len(X) - len(table)}  ({args.augment} per recording)")
    print(f"Training rows     : {len(X)}")
    print(f"Features          : {X.shape[1]}")
    print(f"Classes           : {y.nunique()}\n")

    paths = fit_and_save(X, y, args.out)
    print("Saved:")
    for role, path in paths.items():
        print(f"  {role:<16} {path.relative_to(ROOT)}  ({path.stat().st_size:,} bytes)")

    print("\nReload-and-predict check:")
    return 0 if verify(args.out, table) else 1


if __name__ == "__main__":
    sys.exit(main())
