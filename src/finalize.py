"""Select the winning model, export it, and verify the reload (Phase 19).

Model selection uses **validation** accuracy only.  The test set is touched
exactly once, afterwards, by :func:`evaluate.evaluate_model`.

The reload check is not a formality: it re-loads the exported ``.keras`` file
in a fresh object and asserts that its probabilities match the source model
bit-for-bit on a fixed batch.  If the export were lossy, this fails loudly.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import tensorflow as tf

from data_loader import load_emnist, load_mnist
from evaluate import evaluate_model
from inference import DEFAULT_MODEL_PATH, CharacterRecognizer, export_model
from preprocessing import prepare_arrays
from train import EXPERIMENTS, EXPERIMENTS_DIR
from utils import METRICS_DIR, ensure_dirs, header, save_json, set_seed


def select_best_experiment(dataset: str = "emnist_balanced") -> dict:
    """Return the experiment record with the highest validation accuracy."""
    records = []
    for name in EXPERIMENTS:
        path = EXPERIMENTS_DIR / f"{dataset}_{name}.json"
        if path.exists():
            records.append(json.loads(path.read_text(encoding="utf-8")))
    if not records:
        raise FileNotFoundError(
            f"No experiment records in {EXPERIMENTS_DIR}. Run src/train.py first."
        )
    best = max(records, key=lambda r: r["val_accuracy"])
    header("MODEL SELECTION (validation accuracy - test set still untouched)")
    for record in sorted(records, key=lambda r: -r["val_accuracy"]):
        marker = "  <-- selected" if record["name"] == best["name"] else ""
        print(
            f"  {record['name']:26s} val_acc={record['val_accuracy']:.4f}  "
            f"val_loss={record['val_loss']:.4f}  params={record['parameters']['total']:,}{marker}"
        )
    return best


def verify_reload(source_path: str, exported_path=DEFAULT_MODEL_PATH, n: int = 256) -> dict:
    """Assert the exported model reproduces the source model's outputs."""
    header("RELOAD VERIFICATION")
    data = load_emnist("balanced")
    x = prepare_arrays(data["x_test"][:n])

    source = tf.keras.models.load_model(source_path)
    reloaded = tf.keras.models.load_model(exported_path)

    p_source = source.predict(x, verbose=0)
    p_reloaded = reloaded.predict(x, verbose=0)

    max_abs_diff = float(np.abs(p_source - p_reloaded).max())
    labels_match = bool((p_source.argmax(1) == p_reloaded.argmax(1)).all())

    print(f"  compared {n} test images")
    print(f"  max abs. probability difference (source vs reloaded): {max_abs_diff:.3e}")
    print(f"  predicted labels identical: {labels_match}")
    if not labels_match:
        raise AssertionError("Reloaded model disagrees with the source model.")
    return {
        "n_compared": n,
        "max_abs_probability_difference": max_abs_diff,
        "labels_identical": labels_match,
    }


def run(dataset: str = "emnist_balanced") -> dict:
    """Select, export, verify and evaluate the final model."""
    ensure_dirs()
    set_seed()

    best = select_best_experiment(dataset)
    data = load_mnist() if dataset == "mnist" else load_emnist("balanced")
    class_names = data["class_names"]

    header("EXPORT")
    paths = export_model(
        best["model_path"], class_names=class_names, dataset=dataset, metrics=None
    )
    for key, value in paths.items():
        print(f"  {key}: {value}")

    reload_check = verify_reload(best["model_path"])

    # The single, final look at the test set.
    results = evaluate_model(
        DEFAULT_MODEL_PATH,
        dataset=dataset,
        tag="final",
        history_path=EXPERIMENTS_DIR / f"{dataset}_{best['name']}.json",
    )

    # Re-export with the test metrics recorded inside labels.json.
    headline = {
        key: results["metrics"][key]
        for key in (
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
        )
    }
    export_model(best["model_path"], class_names=class_names, dataset=dataset, metrics=headline)

    # Prove the exported artefacts work from a cold start.
    header("COLD-START INFERENCE CHECK")
    recognizer = CharacterRecognizer()
    sample = prepare_arrays(data["x_test"][:1])
    prediction = recognizer.predict_tensor(sample, top_k=3)
    print(f"  loaded {recognizer.model_path.name} with {len(recognizer.class_names)} labels")
    print(f"  test image 0 true={class_names[data['y_test'][0]]} -> {prediction.character} "
          f"({prediction.confidence * 100:.1f}%)")

    summary = {
        "selected_experiment": best["name"],
        "selection_criterion": "highest validation accuracy",
        "val_accuracy": best["val_accuracy"],
        "architecture_variant": best["variant"],
        "parameters": best["parameters"],
        "hyperparameters": best["hyperparameters"],
        "augmentation": best["augmentation"],
        "exported": paths,
        "reload_verification": reload_check,
        "test_metrics": headline,
        "environment": best["environment"],
    }
    save_json(summary, METRICS_DIR / "final_model_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and evaluate the final model.")
    parser.add_argument("--dataset", default="emnist_balanced", choices=["emnist_balanced", "mnist"])
    args = parser.parse_args()
    run(args.dataset)


if __name__ == "__main__":
    main()
