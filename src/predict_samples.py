"""Run the exported model over the simulated user images (Phases 16-18).

Reports real accuracy per image family, writes a JSON record of every
prediction with its Top-K, and renders a contact sheet.  Nothing here is
hard-coded: every number comes from ``CharacterRecognizer``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from inference import CharacterRecognizer
from utils import (
    FIGURES_DIR,
    PREDICTIONS_DIR,
    SAMPLES_DIR,
    ensure_dirs,
    header,
    load_json,
    save_json,
)


def contact_sheet(rows: list[dict], path: Path, title: str, cols: int = 6) -> Path:
    """Grid of sample images annotated with true/predicted/confidence."""
    n = len(rows)
    grid_rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(grid_rows, cols, figsize=(cols * 2.05, grid_rows * 2.45))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")

    for ax, row in zip(axes, rows):
        image = Image.open(row["path"])
        if image.mode == "RGBA":  # show transparency against white
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image)
        ax.imshow(np.array(image.convert("L")), cmap="gray")
        correct = row["correct"]
        ax.set_title(
            f"true {row['true_label']} → {row['predicted']}\n{row['confidence'] * 100:.1f}%",
            fontsize=9,
            color="#2a7d54" if correct else "#b00020",
        )
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def run(top_k: int = 5, explain_n: int = 4) -> dict:
    """Predict every sample in the manifest and summarise the results."""
    ensure_dirs()
    manifest_path = SAMPLES_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("No samples/manifest.json - run src/make_samples.py first.")
    manifest = load_json(manifest_path)

    recognizer = CharacterRecognizer()
    header(f"PREDICTING {len(manifest)} SIMULATED USER IMAGES")

    rows: list[dict] = []
    for entry in manifest:
        prediction = recognizer.predict_image(entry["path"], top_k=top_k)
        top_chars = [t["character"] for t in prediction.top_k]
        rows.append(
            {
                "path": entry["path"],
                "file": Path(entry["path"]).name,
                "family": entry["family"],
                "source": entry["source"],
                "true_label": entry["true_label"],
                "predicted": prediction.character,
                "confidence": prediction.confidence,
                "correct": prediction.character == entry["true_label"],
                "in_top_3": entry["true_label"] in top_chars[:3],
                "in_top_5": entry["true_label"] in top_chars[:5],
                "top_k": prediction.top_k,
            }
        )

    # ---- per-family accuracy ------------------------------------------
    families: dict[str, dict] = {}
    for family in sorted({r["family"] for r in rows}):
        subset = [r for r in rows if r["family"] == family]
        families[family] = {
            "n": len(subset),
            "top1_accuracy": float(np.mean([r["correct"] for r in subset])),
            "top3_accuracy": float(np.mean([r["in_top_3"] for r in subset])),
            "top5_accuracy": float(np.mean([r["in_top_5"] for r in subset])),
            "mean_confidence": float(np.mean([r["confidence"] for r in subset])),
        }

    print(f"\n{'family':12s} {'n':>3s} {'top1':>7s} {'top3':>7s} {'top5':>7s} {'meanconf':>9s}")
    for family, stats in families.items():
        print(
            f"{family:12s} {stats['n']:>3d} {stats['top1_accuracy']:>7.3f} "
            f"{stats['top3_accuracy']:>7.3f} {stats['top5_accuracy']:>7.3f} "
            f"{stats['mean_confidence']:>9.3f}"
        )

    print("\nPer-image results:")
    for row in rows:
        mark = "OK  " if row["correct"] else "MISS"
        alternatives = ", ".join(
            f"{t['character']}:{t['probability'] * 100:.1f}%" for t in row["top_k"][:3]
        )
        print(
            f"  [{mark}] {row['file']:34s} true={row['true_label']:2s} "
            f"pred={row['predicted']:2s} conf={row['confidence'] * 100:5.1f}%  top3=({alternatives})"
        )

    # ---- figures -------------------------------------------------------
    figures = {}
    for family in families:
        subset = [r for r in rows if r["family"] == family]
        figures[family] = str(
            contact_sheet(
                subset,
                FIGURES_DIR / f"samples_{family}.png",
                f"Predictions on simulated user images - {family} "
                f"(top-1 {families[family]['top1_accuracy'] * 100:.0f}%)",
            )
        )

    # Detailed step-by-step explanations for a few images.
    explained = []
    for row in rows[:explain_n]:
        _, figure = recognizer.explain(
            row["path"],
            top_k=5,
            save_path=FIGURES_DIR / f"inference_steps_{Path(row['path']).stem}.png",
        )
        explained.append(str(figure))

    overall = {
        "n_images": len(rows),
        "top1_accuracy": float(np.mean([r["correct"] for r in rows])),
        "top3_accuracy": float(np.mean([r["in_top_3"] for r in rows])),
        "top5_accuracy": float(np.mean([r["in_top_5"] for r in rows])),
    }
    print(
        f"\nOverall: top-1 {overall['top1_accuracy'] * 100:.1f}%  "
        f"top-3 {overall['top3_accuracy'] * 100:.1f}%  "
        f"top-5 {overall['top5_accuracy'] * 100:.1f}%"
    )

    results = {
        "overall": overall,
        "by_family": families,
        "predictions": rows,
        "figures": figures,
        "explanation_figures": explained,
        "note": (
            "These are simulated camera/scan renderings of unseen EMNIST test "
            "glyphs and printed-font glyphs, not photographs of a real person's "
            "handwriting. The 'font' family is deliberately out of distribution."
        ),
    }
    save_json(results, PREDICTIONS_DIR / "sample_predictions.json")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict all simulated user images.")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    run(top_k=args.top_k)


if __name__ == "__main__":
    main()
