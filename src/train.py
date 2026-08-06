"""Train and cross-validate the wrist-IMU exercise classifier.

The dataset is small (about a hundred sets) and imbalanced, so there is no single
held-out test split: every recording is scored by a model that never saw it, via
stratified k-fold cross-validation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless: never open a GUI window when saving the figure

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = ROOT / "features.csv"
PLOT_PATH = ROOT / "notebooks" / "confusion_matrix.png"

# Columns that describe the recording rather than the movement.
META_COLS = ["recording", "activity", "n_samples"]

LABEL_COL = "activity"
N_SPLITS = 5
RANDOM_STATE = 42

# Sequential blue ramp (steps 100 -> 700), light to dark.
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"


def load_dataset(path: Path = FEATURES_CSV) -> tuple[pd.DataFrame, pd.Series]:
    """Split the feature table into a feature matrix and a label vector."""
    table = pd.read_csv(path)
    if LABEL_COL not in table.columns:
        raise ValueError(f"{path} has no '{LABEL_COL}' column")

    y = table[LABEL_COL]
    X = table.drop(columns=[c for c in META_COLS if c in table.columns])
    return X, y


def build_model() -> Pipeline:
    """Standardize, then fit a random forest.

    The scaler lives inside the pipeline so it is fit on training folds only — the
    validation fold's statistics never leak into the transform. A forest is scale
    invariant so this does not change its predictions, but it keeps the pipeline
    honest if the estimator is ever swapped for a distance- or coefficient-based one.
    """
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "forest",
                RandomForestClassifier(
                    n_estimators=500,
                    class_weight="balanced",  # smallest class has ~40% of the largest's sets
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def augmented_cv_predict(
    X: pd.DataFrame,
    y: pd.Series,
    recordings: pd.Series,
    cv: StratifiedKFold,
    n_aug: int,
    seed: int = RANDOM_STATE,
) -> np.ndarray:
    """Cross-validated predictions with augmented copies added to training folds only.

    For each fold the training recordings are re-featurised from their raw signals
    n_aug extra times under random transforms; the validation fold keeps its real
    feature rows untouched. Nothing augmented is ever scored, so the accuracy this
    produces is still measured on real recordings.
    """
    from augment import augment_frame, load_raw_cache
    from features import features_from_frame

    cache = load_raw_cache(recordings.tolist())
    y_pred = np.empty(len(y), dtype=object)

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        rng = np.random.default_rng(seed + fold)
        rows, labels = [], []
        for idx in train_idx:
            raw = cache[recordings.iloc[idx]]
            for _ in range(n_aug):
                rows.append(features_from_frame(augment_frame(raw, rng)))
                labels.append(y.iloc[idx])

        X_train = pd.concat(
            [X.iloc[train_idx], pd.DataFrame(rows, columns=X.columns)], ignore_index=True
        )
        y_train = pd.concat([y.iloc[train_idx], pd.Series(labels)], ignore_index=True)

        model = build_model()
        model.fit(X_train, y_train)
        y_pred[val_idx] = model.predict(X.iloc[val_idx])

    return y_pred


def fold_importances(X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold) -> pd.DataFrame:
    """Mean +/- std impurity importance across folds, rather than from one lucky fit."""
    per_fold = []
    for train_idx, _ in cv.split(X, y):
        model = build_model()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        per_fold.append(model.named_steps["forest"].feature_importances_)

    stacked = np.vstack(per_fold)
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance": stacked.mean(axis=0),
                "std": stacked.std(axis=0),
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], path: Path) -> None:
    """Counts as cell labels, colour by share of the true class.

    Colouring by row share rather than raw count keeps a 26-set class from washing
    out a 10-set one; the printed number stays the honest count.
    """
    row_totals = cm.sum(axis=1, keepdims=True)
    shares = np.divide(cm, row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals > 0)

    cmap = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)

    fig, ax = plt.subplots(figsize=(8.4, 6.8))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    n = len(labels)
    mesh = ax.pcolormesh(
        np.arange(n + 1),
        np.arange(n + 1),
        shares,
        cmap=cmap,
        vmin=0,
        vmax=1,
        edgecolors=SURFACE,  # 2px surface gap between cells
        linewidth=2,
    )

    for i in range(n):
        for j in range(n):
            if cm[i, j] == 0:
                continue
            # White ink only once the fill is dark enough to need it.
            colour = "#ffffff" if shares[i, j] > 0.55 else INK
            ax.text(
                j + 0.5,
                i + 0.5,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=11,
                color=colour,
            )

    ax.set_xticks(np.arange(n) + 0.5, labels, color=INK, fontsize=10)
    ax.set_yticks(np.arange(n) + 0.5, labels, color=INK, fontsize=10)
    ax.set_xlabel("Predicted exercise", color=INK_MUTED, fontsize=10, labelpad=10)
    ax.set_ylabel("Actual exercise", color=INK_MUTED, fontsize=10, labelpad=10)
    ax.set_title(
        "Where the wrist signal confuses exercises",
        color=INK,
        fontsize=13,
        pad=14,
        loc="left",
    )
    ax.invert_yaxis()
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    bar = fig.colorbar(mesh, ax=ax, fraction=0.045, pad=0.03)
    bar.set_label("Share of that exercise's sets", color=INK_MUTED, fontsize=9)
    bar.ax.tick_params(colors=INK_MUTED, length=0, labelsize=9)
    bar.outline.set_visible(False)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURES_CSV, help="feature table CSV")
    parser.add_argument("--plot", type=Path, default=PLOT_PATH, help="confusion matrix output")
    parser.add_argument("--top", type=int, default=12, help="how many features to list")
    parser.add_argument(
        "--augment",
        type=int,
        default=0,
        metavar="N",
        help="add N augmented copies of each training recording (training folds only)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.features.exists():
        print(f"{args.features} not found - run src/features.py first")
        return 1

    X, y = load_dataset(args.features)
    labels = sorted(y.unique())
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    model = build_model()

    print(f"Recordings : {len(X)}")
    print(f"Features   : {X.shape[1]}")
    print(f"Classes    : {len(labels)}  ({', '.join(labels)})")
    print(f"Validation : stratified {N_SPLITS}-fold cross-validation")
    if args.augment:
        print(f"Augmentation: {args.augment} copies per training recording (training folds only)")
    print()

    if args.augment:
        recordings = pd.read_csv(args.features)["recording"]
        y_pred = augmented_cv_predict(X, y, recordings, cv, args.augment)
        fold_scores = np.array(
            [(y_pred[idx] == y.iloc[idx]).mean() for _, idx in cv.split(X, y)]
        )
    else:
        fold_scores = cross_val_score(model, X, y, cv=cv, n_jobs=-1)
        y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)

    accuracy = float((y_pred == y).mean())
    majority = y.value_counts().iloc[0] / len(y)

    print(f"Cross-validated accuracy : {accuracy:.1%}")
    print(f"  per fold               : {', '.join(f'{s:.1%}' for s in fold_scores)}")
    print(f"  fold mean +/- std      : {fold_scores.mean():.1%} +/- {fold_scores.std():.1%}")
    print(f"Majority-class baseline  : {majority:.1%}  ({y.value_counts().index[0]})")
    print(f"Lift over baseline       : {accuracy - majority:+.1%}")
    print(f"Random-chance baseline   : {1 / len(labels):.1%}\n")

    print("Per-class precision / recall:")
    print(classification_report(y, y_pred, labels=labels, digits=3, zero_division=0))

    cm = confusion_matrix(y, y_pred, labels=labels)
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(pd.DataFrame(cm, index=labels, columns=labels).to_string())

    plot_confusion_matrix(cm, labels, args.plot)
    print(f"\nConfusion matrix saved to {args.plot}")

    importances = fold_importances(X, y, cv)
    print(f"\nTop {args.top} features by mean impurity importance across folds:")
    for _, row in importances.head(args.top).iterrows():
        bar = "#" * max(1, round(row["importance"] * 200))
        print(f"  {row['feature']:<20} {row['importance']:.4f} +/- {row['std']:.4f}  {bar}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
