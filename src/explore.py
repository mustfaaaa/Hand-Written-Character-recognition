"""Dataset inspection, orientation verification and EDA (Phases 2-4).

Every statistic printed here is computed from the loaded arrays; nothing is
hard-coded.  Running this module writes a JSON summary to
``outputs/metrics/`` and all exploratory figures to ``outputs/figures/``.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from augmentation import build_augmenter
from data_loader import (
    dataset_summary,
    load_emnist,
    load_mnist,
    stratified_split,
    verify_orientation,
)
from preprocessing import prepare_arrays, to_uint8_images
from utils import FIGURES_DIR, METRICS_DIR, SEED, ensure_dirs, header, save_json, set_seed


# --------------------------------------------------------------------------
# Phase 3 - orientation
# --------------------------------------------------------------------------
def figure_orientation_check(variant: str = "balanced", n: int = 10) -> dict:
    """Show raw vs transposed EMNIST images side by side, plus the metric."""
    raw = load_emnist(variant, orientation="none")
    fixed = load_emnist(variant, orientation="transpose")
    names = raw["class_names"]

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(raw["x_train"]), size=n, replace=False)

    fig, axes = plt.subplots(2, n, figsize=(n * 1.25, 3.4))
    for col, i in enumerate(idx):
        axes[0, col].imshow(raw["x_train"][i], cmap="gray")
        axes[0, col].set_title(names[raw["y_train"][i]], fontsize=11)
        axes[1, col].imshow(fixed["x_train"][i], cmap="gray")
        axes[1, col].set_title(names[fixed["y_train"][i]], fontsize=11)
        for row in (0, 1):
            axes[row, col].axis("off")
    axes[0, 0].set_ylabel("as stored")
    fig.text(0.005, 0.72, "AS STORED\n(wrong)", fontsize=9, color="#b00020", weight="bold")
    fig.text(0.005, 0.26, "TRANSPOSED\n(correct)", fontsize=9, color="#2a7d54", weight="bold")
    fig.suptitle("EMNIST orientation check: raw IDX bytes vs transposed", y=0.99)
    fig.tight_layout(rect=(0.06, 0, 1, 0.95))
    path = FIGURES_DIR / "phase3_orientation_check.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)

    scores = verify_orientation(variant)

    # Bar chart of the quantitative evidence.
    fig, ax = plt.subplots(figsize=(7, 3.6))
    items = sorted(scores["scores"].items(), key=lambda kv: kv[1])
    ax.barh(
        [k for k, _ in items],
        [v for _, v in items],
        color=["#2a7d54" if k == "transpose" else "#9aa0a6" for k, _ in items],
    )
    ax.set_xlabel("mean correlation of EMNIST digit means vs MNIST digit means")
    ax.set_title("Orientation chosen by measurement, not by eye")
    for i, (_, v) in enumerate(items):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    path2 = FIGURES_DIR / "phase3_orientation_scores.png"
    fig.savefig(path2, dpi=140)
    plt.close(fig)

    return {**scores, "figures": [str(path), str(path2)]}


# --------------------------------------------------------------------------
# Phase 4 - EDA figures
# --------------------------------------------------------------------------
def figure_sample_grid(data: dict, rows: int = 6, cols: int = 12) -> str:
    """Random image grid with labels."""
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(data["x_train"]), size=rows * cols, replace=False)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.95, rows * 1.12))
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(data["x_train"][i], cmap="gray")
        ax.set_title(data["class_names"][data["y_train"][i]], fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Random {data['variant']} training samples (orientation corrected)")
    fig.tight_layout()
    path = FIGURES_DIR / f"eda_{data['variant']}_sample_grid.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def figure_one_per_class(data: dict) -> str:
    """One representative example of every class."""
    names = data["class_names"]
    n = len(names)
    cols = 12 if n > 20 else 10
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.95, rows * 1.15))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for cls in range(n):
        pool = np.flatnonzero(data["y_train"] == cls)
        axes[cls].imshow(data["x_train"][pool[0]], cmap="gray")
        axes[cls].set_title(f"{cls}: {names[cls]}", fontsize=9)
    fig.suptitle(f"One sample per class - {data['variant']} ({n} classes)")
    fig.tight_layout()
    path = FIGURES_DIR / f"eda_{data['variant']}_one_per_class.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def figure_class_distribution(data: dict, summary: dict) -> str:
    """Train and test counts per class."""
    names = data["class_names"]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.28), 4.2))
    ax.bar(x - 0.2, summary["train_class_counts"], width=0.4, label="train", color="#1f4e79")
    ax.bar(x + 0.2, summary["test_class_counts"], width=0.4, label="test", color="#c1442e")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_xlabel("class")
    ax.set_ylabel("images")
    ax.set_title(
        f"Class distribution - {data['variant']} "
        f"(max/min train ratio = {summary['imbalance_ratio']:.2f})"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = FIGURES_DIR / f"eda_{data['variant']}_class_distribution.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def figure_pixel_intensity(data: dict) -> str:
    """Pixel intensity histogram and per-image ink coverage."""
    rng = np.random.default_rng(SEED)
    sample = data["x_train"][rng.choice(len(data["x_train"]), size=8000, replace=False)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(sample.ravel(), bins=64, color="#1f4e79")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("pixel value (0-255)")
    axes[0].set_ylabel("count (log)")
    axes[0].set_title("Pixel intensity distribution")
    axes[0].grid(alpha=0.3)

    coverage = (sample > 0).mean(axis=(1, 2))
    axes[1].hist(coverage, bins=50, color="#2a7d54")
    axes[1].set_xlabel("fraction of non-zero (ink) pixels per image")
    axes[1].set_ylabel("images")
    axes[1].set_title(f"Ink coverage (mean {coverage.mean() * 100:.1f}% of the frame)")
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"{data['variant']}: foreground is bright ink on a black background")
    fig.tight_layout()
    path = FIGURES_DIR / f"eda_{data['variant']}_pixel_intensity.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def figure_average_per_class(data: dict) -> str:
    """Mean image per class - blurry means high within-class variability."""
    names = data["class_names"]
    n = len(names)
    cols = 12 if n > 20 else 10
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.95, rows * 1.15))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for cls in range(n):
        mean_image = data["x_train"][data["y_train"] == cls].mean(axis=0)
        axes[cls].imshow(mean_image, cmap="magma")
        axes[cls].set_title(names[cls], fontsize=9)
    fig.suptitle(f"Average image per class - {data['variant']}")
    fig.tight_layout()
    path = FIGURES_DIR / f"eda_{data['variant']}_average_per_class.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


#: Groups of characters that are plausibly confusable *before* seeing any
#: model output.  These are hypotheses for EDA only - the confusions that
#: get reported in the results come from the real confusion matrix.
SIMILAR_GROUPS = [
    ["0", "O", "Q", "D"],
    ["1", "I", "l"],
    ["5", "S", "s"],
    ["2", "Z"],
    ["9", "g", "q"],
    ["6", "b", "G"],
    ["U", "V"],
    ["C", "G"],
]


def figure_similar_characters(data: dict, n_examples: int = 6) -> str:
    """Side-by-side samples of visually similar character groups."""
    names = data["class_names"]
    available = [[c for c in group if c in names] for group in SIMILAR_GROUPS]
    available = [g for g in available if len(g) >= 2]

    rows = sum(len(g) for g in available)
    fig, axes = plt.subplots(rows, n_examples + 1, figsize=((n_examples + 1) * 1.05, rows * 1.05))
    rng = np.random.default_rng(SEED)

    row = 0
    for group in available:
        for char in group:
            cls = names.index(char)
            pool = np.flatnonzero(data["y_train"] == cls)
            picks = rng.choice(pool, size=n_examples, replace=False)
            axes[row, 0].text(0.5, 0.5, char, fontsize=15, ha="center", va="center", weight="bold")
            axes[row, 0].axis("off")
            for col, i in enumerate(picks, start=1):
                axes[row, col].imshow(data["x_train"][i], cmap="gray")
                axes[row, col].axis("off")
            row += 1

    fig.suptitle("Visually similar character groups (EDA hypotheses)", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    path = FIGURES_DIR / f"eda_{data['variant']}_similar_characters.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def figure_augmentation_examples(data: dict, n: int = 8) -> str:
    """Original vs augmented versions of the same images (Phase 6 check)."""
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(data["x_train"]), size=n, replace=False)
    originals = prepare_arrays(data["x_train"][idx])
    augmenter = build_augmenter()

    rows = 4  # 1 original + 3 augmented draws
    fig, axes = plt.subplots(rows, n, figsize=(n * 1.15, rows * 1.25))
    for col in range(n):
        axes[0, col].imshow(to_uint8_images(originals)[col], cmap="gray")
        axes[0, col].set_title(data["class_names"][data["y_train"][idx[col]]], fontsize=9)
    for r in range(1, rows):
        augmented = augmenter(originals, training=True).numpy()
        for col in range(n):
            axes[r, col].imshow(to_uint8_images(augmented)[col], cmap="gray")
    for ax in axes.ravel():
        ax.axis("off")
    fig.text(0.005, 0.80, "original", fontsize=9, weight="bold")
    fig.text(0.005, 0.45, "augmented", fontsize=9, weight="bold")
    fig.suptitle("Augmentation preserves character identity (±10° rotation, ±10% shift/zoom)")
    fig.tight_layout(rect=(0.05, 0, 1, 0.95))
    path = FIGURES_DIR / f"eda_{data['variant']}_augmentation.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def figure_mnist_vs_emnist(mnist: dict, emnist: dict) -> str:
    """Visual contrast between the baseline dataset and the main dataset."""
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(2, 12, figsize=(12 * 0.95, 2 * 1.2))
    for col in range(12):
        i = rng.integers(len(mnist["x_train"]))
        axes[0, col].imshow(mnist["x_train"][i], cmap="gray")
        axes[0, col].set_title(mnist["class_names"][mnist["y_train"][i]], fontsize=9)
        j = rng.integers(len(emnist["x_train"]))
        axes[1, col].imshow(emnist["x_train"][j], cmap="gray")
        axes[1, col].set_title(emnist["class_names"][emnist["y_train"][j]], fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    fig.text(0.005, 0.72, "MNIST\n10 classes", fontsize=9, weight="bold")
    fig.text(0.005, 0.25, "EMNIST\n47 classes", fontsize=9, weight="bold")
    fig.suptitle("MNIST (digits only) vs EMNIST Balanced (digits + letters)")
    fig.tight_layout(rect=(0.07, 0, 1, 0.93))
    path = FIGURES_DIR / "eda_mnist_vs_emnist.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


# --------------------------------------------------------------------------
def run(variant: str = "balanced") -> dict:
    """Run every inspection step and save the summary."""
    ensure_dirs()
    set_seed()

    header("PHASE 3 - EMNIST ORIENTATION VERIFICATION")
    orientation = figure_orientation_check(variant)
    for name, score in sorted(orientation["scores"].items(), key=lambda kv: -kv[1]):
        print(f"  {name:12s} correlation vs MNIST = {score:+.4f}")
    print(f"  => best transform: {orientation['best']!r}; 'transpose' is correct: {orientation['correct']}")

    header("PHASE 2 - DATASET INSPECTION")
    emnist = load_emnist(variant)
    mnist = load_mnist()
    summary = dataset_summary(emnist)
    mnist_summary = dataset_summary(mnist)

    for label, s in (("EMNIST " + variant, summary), ("MNIST", mnist_summary)):
        print(
            f"\n{label}\n"
            f"  train images : {s['n_train']:,}\n"
            f"  test images  : {s['n_test']:,}\n"
            f"  image shape  : {s['image_shape']}  dtype={s['dtype']}\n"
            f"  classes      : {s['n_classes']}\n"
            f"  pixel range  : [{s['pixel_min']}, {s['pixel_max']}]  "
            f"mean={s['pixel_mean']:.2f} std={s['pixel_std']:.2f}\n"
            f"  per-class train counts: min={s['train_count_min']} max={s['train_count_max']} "
            f"(imbalance ratio {s['imbalance_ratio']:.2f})"
        )
    print(f"\n  EMNIST classes: {' '.join(summary['class_names'])}")

    # Confirm the split logic keeps the test set untouched and stratified.
    x_tr, y_tr, x_val, y_val = stratified_split(emnist["x_train"], emnist["y_train"], 0.1)
    print(
        f"\n  stratified split -> train {len(x_tr):,} / val {len(x_val):,} "
        f"(val per-class min={np.bincount(y_val).min()}, max={np.bincount(y_val).max()}); "
        f"test {summary['n_test']:,} untouched"
    )

    header("PHASE 4 - EXPLORATORY DATA ANALYSIS")
    figures = {
        "sample_grid": figure_sample_grid(emnist),
        "one_per_class": figure_one_per_class(emnist),
        "class_distribution": figure_class_distribution(emnist, summary),
        "pixel_intensity": figure_pixel_intensity(emnist),
        "average_per_class": figure_average_per_class(emnist),
        "similar_characters": figure_similar_characters(emnist),
        "augmentation": figure_augmentation_examples(emnist),
        "mnist_vs_emnist": figure_mnist_vs_emnist(mnist, emnist),
        **{f"orientation_{i}": p for i, p in enumerate(orientation["figures"])},
    }
    for name, path in figures.items():
        print(f"  wrote {name}: {path}")

    result = {
        "orientation": {k: v for k, v in orientation.items() if k != "figures"},
        "emnist_summary": summary,
        "mnist_summary": mnist_summary,
        "figures": figures,
    }
    save_json(result, METRICS_DIR / "dataset_summary.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset inspection and EDA.")
    parser.add_argument("--variant", default="balanced")
    args = parser.parse_args()
    run(args.variant)


if __name__ == "__main__":
    main()
