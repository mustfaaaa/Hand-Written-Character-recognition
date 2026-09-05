"""Final evaluation on the untouched test set (Phases 12-15).

This is the only module that reads the official test split.  It produces
every number and figure quoted in the report:

* accuracy, macro/weighted precision, recall and F1
* per-class classification report
* raw and row-normalised confusion matrices
* the confusion pairs the model *actually* makes (never assumed)
* error analysis with true/predicted labels and confidences
* confidence calibration: correct vs incorrect prediction confidence
* training/validation accuracy and loss curves
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: figures are written, never shown
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from data_loader import load_emnist, load_mnist
from preprocessing import prepare_arrays, to_uint8_images
from utils import (
    FIGURES_DIR,
    METRICS_DIR,
    ensure_dirs,
    header,
    save_json,
    set_seed,
)

EXPERIMENTS_DIR = METRICS_DIR / "experiments"


def load_test_data(dataset: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load the official, untouched test split."""
    data = load_mnist() if dataset == "mnist" else load_emnist("balanced")
    return data["x_test"], data["y_test"], data["class_names"]


def predict_probabilities(
    model: tf.keras.Model, x: np.ndarray, batch_size: int = 512
) -> np.ndarray:
    """Softmax probabilities for every image, shape ``(N, n_classes)``."""
    return model.predict(prepare_arrays(x), batch_size=batch_size, verbose=0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    """Headline metrics plus the full per-class report."""
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
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
        "per_class": report,
    }


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def plot_confusion_matrix(
    cm: np.ndarray, class_names: list[str], path: Path, normalise: bool = True
) -> Path:
    """Heatmap of the confusion matrix."""
    matrix = cm.astype(np.float64)
    if normalise:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, np.maximum(row_sums, 1))

    n = len(class_names)
    size = max(8, min(22, n * 0.34))
    fig, ax = plt.subplots(figsize=(size, size * 0.88))
    sns.heatmap(
        matrix,
        cmap="viridis",
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        cbar_kws={"shrink": 0.6, "label": "fraction of true class" if normalise else "count"},
        vmin=0,
        vmax=1 if normalise else None,
        ax=ax,
    )
    ax.set_xlabel("Predicted character")
    ax.set_ylabel("True character")
    ax.set_title(
        f"Confusion matrix ({'row-normalised' if normalise else 'counts'}) - {n} classes"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_confusion_offdiagonal(
    cm: np.ndarray, class_names: list[str], path: Path, top_k: int = 20
) -> Path:
    """Bar chart of the most frequent *actual* confusions."""
    pairs = top_confusions(cm, class_names, top_k=top_k)
    labels = [f"{p['true']} → {p['predicted']}" for p in pairs][::-1]
    counts = [p["count"] for p in pairs][::-1]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(labels))))
    ax.barh(labels, counts, color="#c1442e")
    ax.set_xlabel("misclassified test images")
    ax.set_title(f"Top {len(labels)} confusion pairs actually made by the model")
    for i, c in enumerate(counts):
        ax.text(c + max(counts) * 0.01, i, str(c), va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_training_curves(history: dict, path: Path, title: str) -> Path:
    """Accuracy and loss curves for train vs validation."""
    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["accuracy"], "-o", ms=3, label="train")
    axes[0].plot(epochs, history["val_accuracy"], "-o", ms=3, label="validation")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["loss"], "-o", ms=3, label="train")
    axes[1].plot(epochs, history["val_loss"], "-o", ms=3, label="validation")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[1].set_title("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_error_grid(
    x_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    class_names: list[str],
    path: Path,
    indices: np.ndarray,
    title: str,
) -> Path:
    """Grid of misclassified images annotated with true/pred/confidence."""
    n = len(indices)
    cols = 8
    rows = int(np.ceil(n / cols))
    images = to_uint8_images(x_test)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.85))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, idx in zip(axes, indices):
        ax.imshow(images[idx], cmap="gray")
        ax.set_title(
            f"T:{class_names[y_true[idx]]}  P:{class_names[y_pred[idx]]}\n{confidence[idx] * 100:.0f}%",
            fontsize=8,
            color="#b00020",
        )
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_confidence_analysis(
    probabilities: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, path: Path
) -> tuple[Path, dict]:
    """Compare prediction confidence for correct vs incorrect predictions."""
    confidence = probabilities.max(axis=1)
    correct = y_pred == y_true

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    bins = np.linspace(0, 1, 41)
    axes[0].hist(confidence[correct], bins=bins, alpha=0.75, label="correct", color="#2a7d54")
    axes[0].hist(confidence[~correct], bins=bins, alpha=0.75, label="incorrect", color="#c1442e")
    axes[0].set_xlabel("max softmax probability")
    axes[0].set_ylabel("test images")
    axes[0].set_yscale("log")
    axes[0].set_title("Prediction confidence distribution")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Reliability: accuracy within each confidence bin.
    edges = np.linspace(0, 1, 11)
    centres, accuracies, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lo) & (confidence < hi if hi < 1 else confidence <= hi)
        if mask.sum() > 0:
            centres.append((lo + hi) / 2)
            accuracies.append(float(correct[mask].mean()))
            counts.append(int(mask.sum()))
    axes[1].plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    axes[1].plot(centres, accuracies, "-o", color="#1f4e79", label="model")
    axes[1].set_xlabel("confidence bin")
    axes[1].set_ylabel("accuracy in bin")
    axes[1].set_title("Reliability diagram")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)

    stats = {
        "mean_confidence_correct": float(confidence[correct].mean()),
        "mean_confidence_incorrect": float(confidence[~correct].mean()) if (~correct).any() else None,
        "median_confidence_correct": float(np.median(confidence[correct])),
        "median_confidence_incorrect": float(np.median(confidence[~correct]))
        if (~correct).any()
        else None,
        "fraction_below_50pct_confidence": float((confidence < 0.5).mean()),
        "accuracy_on_high_confidence_ge_99pct": float(correct[confidence >= 0.99].mean())
        if (confidence >= 0.99).any()
        else None,
        "coverage_high_confidence_ge_99pct": float((confidence >= 0.99).mean()),
        "reliability_bins": [
            {"confidence_bin_center": c, "accuracy": a, "n": n}
            for c, a, n in zip(centres, accuracies, counts)
        ],
    }
    return path, stats


# --------------------------------------------------------------------------
# Analysis helpers
# --------------------------------------------------------------------------
def top_confusions(cm: np.ndarray, class_names: list[str], top_k: int = 20) -> list[dict]:
    """The ``top_k`` most frequent off-diagonal cells, measured from ``cm``."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(off.ravel())[::-1][:top_k]
    pairs = []
    for position in flat:
        i, j = np.unravel_index(position, off.shape)
        count = int(off[i, j])
        if count == 0:
            continue
        support = int(cm[i].sum())
        pairs.append(
            {
                "true": class_names[i],
                "predicted": class_names[j],
                "count": count,
                "true_class_support": support,
                "rate_of_true_class": round(count / support, 4) if support else 0.0,
            }
        )
    return pairs


def hardest_classes(metrics: dict, class_names: list[str], k: int = 10) -> list[dict]:
    """Classes with the lowest per-class F1, taken from the real report."""
    rows = [
        {
            "character": name,
            "precision": metrics["per_class"][name]["precision"],
            "recall": metrics["per_class"][name]["recall"],
            "f1": metrics["per_class"][name]["f1-score"],
            "support": int(metrics["per_class"][name]["support"]),
        }
        for name in class_names
    ]
    return sorted(rows, key=lambda r: r["f1"])[:k]


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def evaluate_model(
    model_path: str | Path,
    dataset: str = "emnist_balanced",
    tag: str = "final",
    history_path: str | Path | None = None,
    n_error_examples: int = 32,
) -> dict:
    """Run the full evaluation suite and write metrics + figures."""
    ensure_dirs()
    set_seed()

    model_path = Path(model_path)
    header(f"EVALUATION | model={model_path.name} | dataset={dataset}")
    model = tf.keras.models.load_model(model_path)

    x_test, y_test, class_names = load_test_data(dataset)
    probabilities = predict_probabilities(model, x_test)
    y_pred = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)

    metrics = compute_metrics(y_test, y_pred, class_names)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))

    print(
        f"test accuracy   : {metrics['accuracy']:.4f}\n"
        f"macro  F1       : {metrics['f1_macro']:.4f}\n"
        f"weighted F1     : {metrics['f1_weighted']:.4f}\n"
        f"macro precision : {metrics['precision_macro']:.4f}\n"
        f"macro recall    : {metrics['recall_macro']:.4f}",
        flush=True,
    )

    # ---- figures -------------------------------------------------------
    figures = {}
    figures["confusion_matrix_normalised"] = str(
        plot_confusion_matrix(cm, class_names, FIGURES_DIR / f"{tag}_confusion_matrix_norm.png", True)
    )
    figures["confusion_matrix_counts"] = str(
        plot_confusion_matrix(cm, class_names, FIGURES_DIR / f"{tag}_confusion_matrix_counts.png", False)
    )
    figures["top_confusions"] = str(
        plot_confusion_offdiagonal(cm, class_names, FIGURES_DIR / f"{tag}_top_confusions.png")
    )

    errors = np.flatnonzero(y_pred != y_test)
    rng = np.random.default_rng(0)
    if len(errors) > 0:
        # Two complementary views: the most confident mistakes (systematic
        # failures) and a random sample (typical failures).
        confident_errors = errors[np.argsort(confidence[errors])[::-1][:n_error_examples]]
        figures["errors_most_confident"] = str(
            plot_error_grid(
                x_test, y_test, y_pred, confidence, class_names,
                FIGURES_DIR / f"{tag}_errors_most_confident.png",
                confident_errors,
                "Most confident mistakes (model was sure and wrong)",
            )
        )
        sample = rng.choice(errors, size=min(n_error_examples, len(errors)), replace=False)
        figures["errors_random_sample"] = str(
            plot_error_grid(
                x_test, y_test, y_pred, confidence, class_names,
                FIGURES_DIR / f"{tag}_errors_random.png",
                np.sort(sample),
                "Random sample of misclassified test images",
            )
        )

    conf_path, confidence_stats = plot_confidence_analysis(
        probabilities, y_test, y_pred, FIGURES_DIR / f"{tag}_confidence_analysis.png"
    )
    figures["confidence_analysis"] = str(conf_path)

    if history_path and Path(history_path).exists():
        record = json.loads(Path(history_path).read_text(encoding="utf-8"))
        figures["training_curves"] = str(
            plot_training_curves(
                record["history"],
                FIGURES_DIR / f"{tag}_training_curves.png",
                f"Training curves - {record['name']}",
            )
        )

    # ---- error records -------------------------------------------------
    error_records = [
        {
            "test_index": int(i),
            "true": class_names[y_test[i]],
            "predicted": class_names[y_pred[i]],
            "confidence": round(float(confidence[i]), 4),
            "true_class_probability": round(float(probabilities[i, y_test[i]]), 4),
        }
        for i in errors[np.argsort(confidence[errors])[::-1][:50]]
    ]

    results = {
        "model_path": str(model_path),
        "dataset": dataset,
        "n_test": int(len(y_test)),
        "n_classes": len(class_names),
        "class_names": class_names,
        "metrics": metrics,
        "n_errors": int(len(errors)),
        "error_rate": float(len(errors) / len(y_test)),
        "top_confusions": top_confusions(cm, class_names, top_k=25),
        "hardest_classes": hardest_classes(metrics, class_names, k=10),
        "confidence": confidence_stats,
        "most_confident_errors": error_records,
        "figures": figures,
    }
    save_json(results, METRICS_DIR / f"evaluation_{tag}.json")
    np.savetxt(METRICS_DIR / f"confusion_matrix_{tag}.csv", cm, fmt="%d", delimiter=",")

    print("\nTop 10 confusions actually observed:")
    for pair in results["top_confusions"][:10]:
        print(
            f"  {pair['true']} -> {pair['predicted']:2s}  "
            f"{pair['count']:4d} / {pair['true_class_support']}  ({pair['rate_of_true_class'] * 100:.1f}% of class)"
        )
    print("\nHardest classes by F1:")
    for row in results["hardest_classes"]:
        print(f"  {row['character']:2s}  F1={row['f1']:.3f}  P={row['precision']:.3f}  R={row['recall']:.3f}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test set.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="emnist_balanced", choices=["emnist_balanced", "mnist"])
    parser.add_argument("--tag", default="final")
    parser.add_argument("--history", default=None, help="path to the experiment JSON record")
    args = parser.parse_args()
    evaluate_model(args.model, dataset=args.dataset, tag=args.tag, history_path=args.history)


if __name__ == "__main__":
    main()
