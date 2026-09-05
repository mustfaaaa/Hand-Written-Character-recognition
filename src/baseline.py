"""Classical machine-learning baselines on flattened pixels (Phase 8).

The point of this module is to answer one question honestly: *how far do
traditional models get without learning any spatial structure?*  The
baselines are therefore deliberately not tuned - they are a reference line
for the CNN, not competitors.

All three baselines are fitted on the same stratified subsample of the
training split (a CPU-budget decision, recorded in the saved metrics) and
evaluated on the full, untouched official test set.
"""

from __future__ import annotations

import argparse
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier

from data_loader import load_emnist, load_mnist, stratified_split
from utils import (
    BASELINE_MODELS_DIR,
    METRICS_DIR,
    SEED,
    ensure_dirs,
    header,
    save_json,
    set_seed,
)

#: Training images used to fit each baseline (kept small so KNN/logreg finish).
DEFAULT_SUBSAMPLE = 30_000


def flatten(x: np.ndarray) -> np.ndarray:
    """``(N, 28, 28)`` uint8 -> ``(N, 784)`` float32 scaled to [0, 1]."""
    return x.reshape(len(x), -1).astype(np.float32) / 255.0


def subsample(x: np.ndarray, y: np.ndarray, n: int, seed: int = SEED):
    """Class-proportional subsample of at most ``n`` rows."""
    if n >= len(x):
        return x, y
    rng = np.random.default_rng(seed)
    per_class = max(1, n // len(np.unique(y)))
    idx = np.concatenate(
        [
            rng.choice(np.flatnonzero(y == c), size=min(per_class, int((y == c).sum())), replace=False)
            for c in np.unique(y)
        ]
    )
    rng.shuffle(idx)
    return x[idx], y[idx]


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy plus macro/weighted precision, recall and F1."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def build_baselines(seed: int = SEED) -> dict:
    """The three untuned reference models."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=200, n_jobs=-1, random_state=seed
        ),
        "knn_k3": KNeighborsClassifier(n_neighbors=3, n_jobs=-1),
        "random_forest": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=seed
        ),
    }


def run_baselines(
    dataset: str = "emnist_balanced",
    n_subsample: int = DEFAULT_SUBSAMPLE,
    val_fraction: float = 0.1,
) -> dict:
    """Fit every baseline and evaluate on validation and test splits."""
    ensure_dirs()
    set_seed()

    data = load_mnist() if dataset == "mnist" else load_emnist("balanced")
    x_tr, y_tr, x_val, y_val = stratified_split(
        data["x_train"], data["y_train"], val_fraction=val_fraction
    )
    x_fit, y_fit = subsample(x_tr, y_tr, n_subsample)

    x_fit_f = flatten(x_fit)
    x_val_f = flatten(x_val)
    x_test_f = flatten(data["x_test"])
    y_test = data["y_test"]

    header(f"BASELINES on {dataset} (fit on {len(x_fit):,} images, {len(x_test_f):,} test)")

    results: dict[str, dict] = {}
    for name, model in build_baselines().items():
        start = time.perf_counter()
        model.fit(x_fit_f, y_fit)
        fit_seconds = time.perf_counter() - start

        start = time.perf_counter()
        test_pred = model.predict(x_test_f)
        predict_seconds = time.perf_counter() - start
        val_pred = model.predict(x_val_f)

        entry = {
            "model": name,
            "dataset": dataset,
            "n_fit": int(len(x_fit)),
            "fit_seconds": round(fit_seconds, 2),
            "predict_seconds_test": round(predict_seconds, 2),
            "val": score_predictions(y_val, val_pred),
            "test": score_predictions(y_test, test_pred),
        }
        results[name] = entry
        joblib.dump(model, BASELINE_MODELS_DIR / f"{dataset}_{name}.joblib")
        print(
            f"{name:22s} val_acc={entry['val']['accuracy']:.4f}  "
            f"test_acc={entry['test']['accuracy']:.4f}  "
            f"test_macroF1={entry['test']['f1_macro']:.4f}  "
            f"fit={fit_seconds:.1f}s",
            flush=True,
        )

    save_json(results, METRICS_DIR / f"baselines_{dataset}.json")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train classical ML baselines.")
    parser.add_argument("--dataset", default="emnist_balanced", choices=["emnist_balanced", "mnist"])
    parser.add_argument("--n-subsample", type=int, default=DEFAULT_SUBSAMPLE)
    args = parser.parse_args()
    run_baselines(dataset=args.dataset, n_subsample=args.n_subsample)


if __name__ == "__main__":
    main()
