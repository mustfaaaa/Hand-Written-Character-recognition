"""Regenerate the Results block in ``README.md`` from the metrics files.

The README's headline numbers are machine-written for the same reason
``report.md`` is: a hand-typed accuracy figure silently goes stale the moment
a model is retrained.  Everything between the ``RESULTS:START`` and
``RESULTS:END`` markers is replaced; the rest of the README is untouched.

Run after ``finalize.py`` (and, optionally, ``predict_samples.py``):
    python src/update_readme.py
"""

from __future__ import annotations

import re

from utils import METRICS_DIR, PREDICTIONS_DIR, PROJECT_ROOT, header, load_json

README_PATH = PROJECT_ROOT / "README.md"
EXPERIMENTS_DIR = METRICS_DIR / "experiments"

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"

BASELINE_LABELS = {
    "logistic_regression": "Logistic regression",
    "knn_k3": "K-nearest neighbours (k=3)",
    "random_forest": "Random forest (200 trees)",
}

EXPERIMENT_LABELS = {
    "exp2_cnn_simple": "Simple CNN",
    "exp3_cnn_bn": "+ BatchNorm",
    "exp4_cnn_bn_drop": "+ Dropout",
    "exp5_cnn_bn_drop_aug": "+ Augmentation",
    "exp6_cnn_deep_aug_lr": "+ Depth & LR schedule",
}

EXPERIMENT_QUESTIONS = {
    "exp2_cnn_simple": "How far does a plain CNN get?",
    "exp3_cnn_bn": "Does batch normalisation help?",
    "exp4_cnn_bn_drop": "Does dropout curb overfitting?",
    "exp5_cnn_bn_drop_aug": "Does augmentation generalise better?",
    "exp6_cnn_deep_aug_lr": "Does more depth + LR scheduling win?",
}


def _optional(path):
    return load_json(path) if path.exists() else None


def pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def build_block() -> str:
    evaluation = _optional(METRICS_DIR / "evaluation_final.json")
    final = _optional(METRICS_DIR / "final_model_summary.json")
    baselines = _optional(METRICS_DIR / "baselines_emnist_balanced.json")
    table = _optional(METRICS_DIR / "experiment_table_emnist_balanced.json")
    samples = _optional(PREDICTIONS_DIR / "sample_predictions.json")

    if not (evaluation and final):
        return (
            "*Not generated yet. Run the pipeline (see **Quick start** below), then "
            "`python src/update_readme.py` to populate this section.*"
        )

    metrics = evaluation["metrics"]
    lines: list[str] = []
    add = lines.append

    # ---- headline cards ------------------------------------------------
    add("<div align=\"center\">")
    add("")
    add(
        f"### 🎯 {pct(metrics['accuracy'])} accuracy &nbsp;·&nbsp; "
        f"{pct(metrics['f1_macro'])} macro F1"
    )
    add("")
    add(
        f"on **{evaluation['n_test']:,} held-out test images** across "
        f"**{evaluation['n_classes']} classes** — a test set the model never saw during "
        "training or model selection."
    )
    add("")
    add("</div>")
    add("")

    # ---- model vs baselines --------------------------------------------
    add("| Model | Accuracy | Precision | Recall | Macro F1 | Weighted F1 |")
    add("|:---|---:|---:|---:|---:|---:|")
    if baselines:
        for name, record in baselines.items():
            t = record["test"]
            add(
                f"| {BASELINE_LABELS.get(name, name)} | {pct(t['accuracy'])} | "
                f"{pct(t['precision_macro'])} | {pct(t['recall_macro'])} | "
                f"{pct(t['f1_macro'])} | {pct(t['f1_weighted'])} |"
            )
    add(
        f"| **🏆 CNN (final)** | **{pct(metrics['accuracy'])}** | "
        f"**{pct(metrics['precision_macro'])}** | **{pct(metrics['recall_macro'])}** | "
        f"**{pct(metrics['f1_macro'])}** | **{pct(metrics['f1_weighted'])}** |"
    )
    add("")

    if baselines:
        best = max(baselines.items(), key=lambda kv: kv[1]["test"]["accuracy"])
        gain = (metrics["accuracy"] - best[1]["test"]["accuracy"]) * 100
        reduction = (
            1 - (1 - metrics["accuracy"]) / (1 - best[1]["test"]["accuracy"])
        ) * 100
        add(
            f"The CNN beats the strongest classical baseline "
            f"({BASELINE_LABELS[best[0]]}, {pct(best[1]['test']['accuracy'])}) by "
            f"**{gain:.2f} percentage points** — an error reduction of **{reduction:.1f}%**. "
            "The baselines flatten each image into 784 independent pixels; the CNN's "
            "convolutions exploit the spatial structure they discard."
        )
        add("")

    # ---- experiment ladder ---------------------------------------------
    if table:
        add("<details>")
        add("<summary><b>The experiment ladder</b> — each step changes exactly one thing</summary>")
        add("")
        add("| Experiment | Question | Val accuracy | Train−val gap |")
        add("|:---|:---|---:|---:|")
        for row in table:
            name = row["experiment"]
            add(
                f"| {EXPERIMENT_LABELS.get(name, name)} | "
                f"{EXPERIMENT_QUESTIONS.get(name, row['change'])} | "
                f"{pct(row['val_accuracy'])} | {row['train_val_gap'] * 100:+.2f} pp |"
            )
        add("")
        add(
            "`Train−val gap` is training accuracy minus validation accuracy at the best "
            "epoch — the direct measure of overfitting. These are **validation** figures; "
            "the test set was still untouched at this stage."
        )
        add("")
        add("</details>")
        add("")

    # ---- hardest classes / confusions -----------------------------------
    confusions = evaluation.get("top_confusions") or []
    if confusions:
        add("<details>")
        add("<summary><b>What it gets wrong</b> — read from the real confusion matrix</summary>")
        add("")
        add("| True | Predicted | Count | % of that class |")
        add("|:---:|:---:|---:|---:|")
        for pair in confusions[:8]:
            add(
                f"| `{pair['true']}` | `{pair['predicted']}` | {pair['count']} | "
                f"{pair['rate_of_true_class'] * 100:.1f}% |"
            )
        add("")
        add(
            "These are character pairs whose handwritten forms genuinely overlap when "
            "stripped of surrounding context — not arbitrary failures."
        )
        add("")
        add("![Confusion matrix](outputs/figures/final_confusion_matrix_norm.png)")
        add("")
        add("</details>")
        add("")

    # ---- confidence ------------------------------------------------------
    confidence = evaluation.get("confidence")
    if confidence and confidence.get("mean_confidence_incorrect") is not None:
        add("<details>")
        add("<summary><b>Is the confidence trustworthy?</b></summary>")
        add("")
        add("| Statistic | Value |")
        add("|:---|---:|")
        add(f"| Mean confidence when **correct** | {pct(confidence['mean_confidence_correct'])} |")
        add(f"| Mean confidence when **incorrect** | {pct(confidence['mean_confidence_incorrect'])} |")
        if confidence.get("accuracy_on_high_confidence_ge_99pct") is not None:
            add(
                f"| Accuracy when confidence ≥ 99% | "
                f"{pct(confidence['accuracy_on_high_confidence_ge_99pct'])} "
                f"(covers {pct(confidence['coverage_high_confidence_ge_99pct'])} of the test set) |"
            )
        add("")
        gap = (
            confidence["mean_confidence_correct"] - confidence["mean_confidence_incorrect"]
        ) * 100
        add(
            f"The model is **{gap:.1f} percentage points** less confident when it is wrong, "
            "so a threshold can route uncertain characters to a human instead of silently "
            "guessing. The web app flags anything below 60% as uncertain."
        )
        add("")
        add("</details>")
        add("")

    # ---- real-world images ----------------------------------------------
    if samples:
        add("<details>")
        add("<summary><b>On realistic photo-style images</b></summary>")
        add("")
        add("| Image family | Count | Top-1 | Top-3 | Top-5 |")
        add("|:---|---:|---:|---:|---:|")
        for family, stats in samples["by_family"].items():
            add(
                f"| {family} | {stats['n']} | {pct(stats['top1_accuracy'], 1)} | "
                f"{pct(stats['top3_accuracy'], 1)} | {pct(stats['top5_accuracy'], 1)} |"
            )
        overall = samples["overall"]
        add(
            f"| **Overall** | **{overall['n_images']}** | "
            f"**{pct(overall['top1_accuracy'], 1)}** | "
            f"**{pct(overall['top3_accuracy'], 1)}** | "
            f"**{pct(overall['top5_accuracy'], 1)}** |"
        )
        add("")
        add(f"> {samples['note']}")
        add("")
        add("![Sample predictions](outputs/figures/samples_photo.png)")
        add("")
        add("</details>")
        add("")

    # ---- provenance ------------------------------------------------------
    env = final.get("environment", {})
    selected = final.get("selected_experiment", "?")
    params = final.get("parameters", {}).get("total")
    detail = [f"Selected on validation accuracy: `{selected}`"]
    if params:
        detail.append(f"{params:,} parameters")
    if env.get("tensorflow"):
        detail.append(f"TensorFlow {env['tensorflow']}")
    if env.get("cpu_count"):
        gpu = "GPU" if env.get("gpus") else "CPU only"
        detail.append(f"{env['cpu_count']} threads, {gpu}")
    add(f"<sub>{' · '.join(detail)} · numbers generated by `src/update_readme.py`</sub>")

    return "\n".join(lines)


def main() -> None:
    header("UPDATING README RESULTS BLOCK")
    text = README_PATH.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise SystemExit(f"Markers {START} / {END} not found in {README_PATH}")

    block = build_block()
    updated = re.sub(
        re.escape(START) + r".*?" + re.escape(END),
        f"{START}\n{block}\n{END}",
        text,
        flags=re.DOTALL,
    )
    README_PATH.write_text(updated, encoding="utf-8")
    print(f"  wrote {README_PATH} ({len(block):,} characters in the results block)")


if __name__ == "__main__":
    main()
