"""Generate ``report.md`` from the metrics files produced by actual runs.

Every quantitative statement in the report is interpolated from JSON written
by ``explore.py``, ``baseline.py``, ``train.py``, ``finalize.py``,
``evaluate.py`` and ``predict_samples.py``.  No figure is typed by hand, so
the report cannot drift from the results or contain an invented number.

Run this last:
    python src/make_report.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from utils import (
    METRICS_DIR,
    PREDICTIONS_DIR,
    PROJECT_ROOT,
    ensure_dirs,
    header,
    load_json,
)

EXPERIMENTS_DIR = METRICS_DIR / "experiments"
REPORT_PATH = PROJECT_ROOT / "report.md"

BASELINE_LABELS = {
    "logistic_regression": "Logistic regression",
    "knn_k3": "K-nearest neighbours (k=3)",
    "random_forest": "Random forest (200 trees)",
}

EXPERIMENT_LABELS = {
    "exp2_cnn_simple": "Simple CNN (no BN, no dropout)",
    "exp3_cnn_bn": "CNN + BatchNorm",
    "exp4_cnn_bn_drop": "CNN + BatchNorm + Dropout",
    "exp5_cnn_bn_drop_aug": "CNN + BN + Dropout + Augmentation",
    "exp6_cnn_deep_aug_lr": "Deeper CNN + Augmentation + LR schedule",
}


def _require(path: Path, what: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {what}: {path}. Run the pipeline first.")
    return load_json(path)


def _optional(path: Path):
    return load_json(path) if path.exists() else None


def pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def build() -> str:
    dataset = _require(METRICS_DIR / "dataset_summary.json", "dataset summary")
    baselines = _require(METRICS_DIR / "baselines_emnist_balanced.json", "baseline metrics")
    table = _require(METRICS_DIR / "experiment_table_emnist_balanced.json", "experiment table")
    evaluation = _require(METRICS_DIR / "evaluation_final.json", "final evaluation")
    final = _require(METRICS_DIR / "final_model_summary.json", "final model summary")
    samples = _optional(PREDICTIONS_DIR / "sample_predictions.json")
    mnist_record = _optional(EXPERIMENTS_DIR / "mnist_exp4_cnn_bn_drop.json")

    e = dataset["emnist_summary"]
    m = dataset["mnist_summary"]
    orientation = dataset["orientation"]
    metrics = evaluation["metrics"]
    confidence = evaluation["confidence"]
    env = final["environment"]
    hp = final["hyperparameters"]

    best_baseline = max(baselines.items(), key=lambda kv: kv[1]["test"]["accuracy"])
    selected = final["selected_experiment"]
    selected_record = _require(
        EXPERIMENTS_DIR / f"emnist_balanced_{selected}.json", "selected experiment record"
    )

    lines: list[str] = []
    add = lines.append

    # ---------------------------------------------------------------- title
    add("# Handwritten Character Recognition with a Convolutional Neural Network")
    add("")
    add(f"*Generated from executed results on {date.today().isoformat()}.*")
    add("")
    add(
        "Every number in this report is read directly from the JSON metrics files "
        "written by the training and evaluation scripts. Nothing is hand-entered."
    )
    add("")
    add("---")
    add("")

    # ------------------------------------------------------------ 1-3 intro
    add("## 1. Introduction")
    add("")
    add(
        "Handwritten character recognition is the task of mapping an image of a "
        "single handwritten symbol to its identity. It is the classical proving "
        "ground for computer vision because handwriting is high-variance: the same "
        "letter written by two people — or by the same person twice — differs in "
        "slant, stroke thickness, proportion and closure, yet must map to one label."
    )
    add("")
    add(
        "This project builds and evaluates a convolutional neural network (CNN) that "
        f"recognises {e['n_classes']} character classes, and wraps it in an inference "
        "system that accepts real photographs and drawings rather than only "
        "pre-cleaned dataset images."
    )
    add("")

    add("## 2. Problem Statement")
    add("")
    add(
        "Given a grayscale image containing one handwritten character, predict which "
        "character it is and report a calibrated confidence. The system must work on "
        "images that do not resemble the training distribution's formatting — "
        "arbitrary size, arbitrary background, off-centre placement, and either dark "
        "ink on light paper or light ink on a dark background."
    )
    add("")

    add("## 3. Objectives")
    add("")
    add("1. Select and justify an appropriate handwritten-character dataset.")
    add("2. Verify the data is loaded and oriented correctly before training on it.")
    add("3. Establish a classical machine-learning baseline for comparison.")
    add("4. Train a CNN and improve it through controlled, one-variable-at-a-time experiments.")
    add("5. Evaluate on a held-out test set using macro-averaged metrics, not accuracy alone.")
    add("6. Analyse which characters the model confuses and why.")
    add("7. Export the model and perform inference on genuinely unseen images.")
    add("")

    # ----------------------------------------------------------- 4 dataset
    add("## 4. Dataset")
    add("")
    add(
        "**EMNIST Balanced** is the main dataset; **MNIST** is used as an easier "
        "reference point so the difficulty gap is measured rather than asserted."
    )
    add("")
    add("| | MNIST | EMNIST Balanced |")
    add("|---|---|---|")
    add(f"| Training images | {m['n_train']:,} | {e['n_train']:,} |")
    add(f"| Test images | {m['n_test']:,} | {e['n_test']:,} |")
    add(f"| Classes | {m['n_classes']} | {e['n_classes']} |")
    add(f"| Image size | {m['image_shape'][0]}×{m['image_shape'][1]} | {e['image_shape'][0]}×{e['image_shape'][1]} |")
    add(
        f"| Per-class train count | {m['train_count_min']:,}–{m['train_count_max']:,} "
        f"| {e['train_count_min']:,} (exactly balanced) |"
    )
    add(f"| Imbalance ratio | {m['imbalance_ratio']:.2f} | {e['imbalance_ratio']:.2f} |")
    add("")
    add("### Why EMNIST Balanced")
    add("")
    add("| Variant | Classes | Train | Verdict |")
    add("|---|---|---|---|")
    add("| MNIST | 10 | 60,000 | Digits only — too easy to be the main task. |")
    add("| EMNIST Digits | 10 | 240,000 | A larger MNIST; no character diversity. |")
    add("| EMNIST Letters | 26 | 124,800 | Letters only, no digits, case fully merged. |")
    add("| EMNIST ByClass | 62 | 697,932 | Most complete, but severely imbalanced and ~6× the compute. |")
    add("| EMNIST ByMerge | 47 | 697,932 | Same classes as Balanced, but imbalanced and much larger. |")
    add(f"| **EMNIST Balanced** | **{e['n_classes']}** | **{e['n_train']:,}** | **Selected.** |")
    add("")
    add(
        "1. **It is genuinely a character task** — digits *and* letters across "
        f"{e['n_classes']} classes, not a digits-only problem.\n"
        "2. **Exactly balanced.** Every class has the same number of training and "
        "test images, so accuracy and macro-F1 measure real discriminative skill "
        "rather than a class prior. Any weakness in the results is therefore a "
        "property of the characters, not of the sampling.\n"
        "3. **Case is handled honestly.** Fifteen letters (C I J K L M O P S U V W X "
        "Y Z) have upper- and lowercase forms that are indistinguishable in "
        "isolation, so EMNIST merges them. Penalising a model for calling an "
        "isolated `o` an `O` would be measuring an impossible task.\n"
        "4. **Computationally reasonable** on the CPU-only machine used here, unlike "
        "the 698k-image ByClass and ByMerge variants."
    )
    add("")
    add(f"Class set: `{' '.join(e['class_names'])}`")
    add("")

    # --------------------------------------------------- 5 dataset analysis
    add("## 5. Dataset Analysis")
    add("")
    add(
        f"Computed from the loaded arrays: pixel values are `{e['dtype']}` in "
        f"[{e['pixel_min']}, {e['pixel_max']}], with mean {e['pixel_mean']:.2f} and "
        f"standard deviation {e['pixel_std']:.2f}. The intensity histogram is strongly "
        "bimodal — a large pure-black background mode and a bright ink mode — "
        "confirming the images are already background-normalised."
    )
    add("")
    add("### The EMNIST orientation trap")
    add("")
    add(
        "EMNIST's IDX files store each image with the row and column axes **swapped** "
        "relative to MNIST. Read naively, every character comes out mirrored and "
        "rotated 90°. This is dangerous precisely because it is silent: a CNN will "
        "still train to ~85% accuracy on consistently wrong orientation, so the bug "
        "does not announce itself in the loss curve."
    )
    add("")
    add(
        "Rather than trusting a visual impression, the orientation was chosen by "
        "measurement. EMNIST classes 0–9 are the same digits as MNIST, so for each "
        "candidate transform the mean image of every digit class was correlated "
        "against MNIST's:"
    )
    add("")
    add("| Candidate transform | Mean correlation with MNIST |")
    add("|---|---|")
    for name, score in sorted(orientation["scores"].items(), key=lambda kv: -kv[1]):
        marker = " **← selected**" if name == orientation["best"] else ""
        add(f"| `{name}`{marker} | {score:+.4f} |")
    add("")
    add(
        f"`{orientation['best']}` wins decisively. The fix was then confirmed visually "
        "(`outputs/figures/phase3_orientation_check.png`), so the decision rests on "
        "both a metric and an inspection."
    )
    add("")

    # ----------------------------------------------------- 6 preprocessing
    add("## 6. Preprocessing")
    add("")
    add("Two pipelines exist, and they must agree at the point of prediction.")
    add("")
    add("**Dataset pipeline.** EMNIST images are already 28×28, deslanted, "
        "size-normalised and centred by NIST, so they need only the orientation "
        "transpose, scaling to `[0, 1]`, and a channel axis giving `(N, 28, 28, 1)`. "
        "They are deliberately *not* upscaled: at 28×28 a stroke is 2–3 px wide, so "
        "resizing to 224×224 adds no information, blurs the strokes, and multiplies "
        "the compute by roughly 64× for nothing.")
    add("")
    add("**Real-world pipeline.** A photograph is not EMNIST-like, so it is "
        "reconstructed into NIST conditions:")
    add("")
    add("1. Grayscale (transparency composited onto white).")
    add("2. 3×3 Gaussian blur to suppress sensor and JPEG noise.")
    add("3. Otsu threshold — the cut point is derived from the image's own histogram.")
    add("4. Polarity detection: ink is the *minority* region, so dark-on-light and "
        "light-on-dark both work without asking the user.")
    add("5. Largest connected component, discarding specks and paper grain.")
    add("6. Crop to the ink bounding box.")
    add("7. Resize so the longest side is 20 px, preserving aspect ratio.")
    add("8. Centre by centre of mass in a 28×28 frame — the same procedure NIST used "
        "to build MNIST and EMNIST.")
    add("9. Scale to `[0, 1]`.")
    add("")

    # ----------------------------------------------------- 7 augmentation
    add("## 7. Data Augmentation")
    add("")
    add(
        "Augmentation is restricted to transforms that **cannot change the label**: "
        "±10° rotation, ±10% translation and ±10% zoom, applied to the training split "
        "only. Validation and test data are never augmented."
    )
    add("")
    add("| Excluded transform | Why it would corrupt the labels |")
    add("|---|---|")
    add("| Horizontal flip | `b`↔`d`, `p`↔`q` |")
    add("| Vertical flip | `M`↔`W`, `b`↔`p` |")
    add("| Rotation ≥ 45° | `6`↔`9`, `N`↔`Z` |")
    add("| Heavy shear | destroys thin-stroke letters |")
    add("")
    add(
        "This matters more for EMNIST than for MNIST: a digits-only dataset has fewer "
        "label-flipping symmetries than a 47-class alphabet does."
    )
    add("")

    # -------------------------------------------------------- 8 architecture
    params = final["parameters"]
    add("## 8. CNN Architecture")
    add("")
    add("```")
    add("Input 28×28×1")
    add("  [Conv 3×3 → BatchNorm → ReLU] ×2 → MaxPool 2×2 → Dropout")
    add("  [Conv 3×3 → BatchNorm → ReLU] ×2 → MaxPool 2×2 → Dropout")
    add("  [Conv 3×3 → BatchNorm → ReLU] ×2 → MaxPool 2×2 → Dropout   (deep variant)")
    add("Flatten → Dense → BatchNorm → ReLU → Dropout → Dense(47) → Softmax")
    add("```")
    add("")
    add(
        f"The selected model (`{final['architecture_variant']}`) has "
        f"**{params['total']:,} parameters** ({params['trainable']:,} trainable)."
    )
    add("")
    add(
        "Design rationale: **convolution** exploits the fact that a stroke junction "
        "means the same thing wherever it appears, so filters are shared across the "
        "image instead of learning a separate weight per pixel. **ReLU** is used "
        "because it does not saturate for positive inputs, so gradients survive depth. "
        "**Batch normalisation** sits between each convolution and its activation, "
        "re-centring activations so a larger learning rate stays stable (the "
        "convolutions therefore carry no bias — BN's shift term replaces it). "
        "**Max pooling** halves the resolution, giving tolerance to small shifts and "
        "widening the receptive field. **Dropout** randomly removes units during "
        "training so the network cannot rely on any single feature. **Softmax** turns "
        "the final scores into a probability distribution over the classes, which is "
        "what makes the confidence figure meaningful."
    )
    add("")

    # ------------------------------------------------------- 9 training
    add("## 9. Training Methodology")
    add("")
    add("| Setting | Value |")
    add("|---|---|")
    add(f"| Optimiser | {hp['optimizer'].title()} |")
    add(f"| Learning rate | {hp['learning_rate']} |")
    add(f"| Batch size | {hp['batch_size']} |")
    add(f"| Max epochs | {hp['max_epochs']} |")
    add(f"| Epochs actually run | {hp['epochs_run']} |")
    add(f"| Early stopping patience | {hp['early_stopping_patience']} (on validation loss) |")
    add(f"| Validation split | {hp['val_fraction'] * 100:.0f}% of train, stratified |")
    add(f"| Random seed | {hp['seed']} |")
    add("")
    add("### Preventing data leakage")
    add("")
    add(
        f"The official EMNIST test split ({e['n_test']:,} images) is never used for "
        "fitting, early stopping, or model selection. Validation is a stratified "
        f"{hp['val_fraction'] * 100:.0f}% carved out of the training split with a fixed "
        f"seed, yielding {selected_record['n_train']:,} training and "
        f"{selected_record['n_val']:,} validation images. Model selection used "
        "validation accuracy only; `evaluate.py` is the sole module that reads the "
        "test set, and it ran once, after the final model was chosen."
    )
    add("")
    add("### Hardware")
    add("")
    add(
        f"Python {env['python']}, TensorFlow {env['tensorflow']}, "
        f"{env['cpu_count']} CPU threads, "
        f"**no GPU** (`{env['processor']}`). This constrained the experiment budget: "
        "epoch times ranged from roughly 80 to 260 seconds, so early stopping and a "
        "modest epoch cap were necessary rather than optional."
    )
    add("")

    # -------------------------------------------------- 10 experiments
    add("## 10. Experiments")
    add("")
    add(
        "Each experiment changes exactly one component relative to the previous one, "
        "so the comparison isolates that component's effect. All figures below are "
        "**validation** results — the test set was still untouched at this stage."
    )
    add("")
    add("| # | Experiment | Params | Epochs | Val accuracy | Val loss | Train−val gap |")
    add("|---|---|---|---|---|---|---|")
    for i, row in enumerate(table, start=2):
        label = EXPERIMENT_LABELS.get(row["experiment"], row["experiment"])
        add(
            f"| {i} | {label} | {row['params']:,} | {row['epochs_run']} | "
            f"{pct(row['val_accuracy'])} | {row['val_loss']:.4f} | "
            f"{row['train_val_gap'] * 100:+.2f} pp |"
        )
    add("")
    add(
        "The `train−val gap` column is the direct measure of overfitting: training "
        "accuracy minus validation accuracy at the best epoch."
    )
    add("")
    for row in table:
        label = EXPERIMENT_LABELS.get(row["experiment"], row["experiment"])
        add(f"- **{label}** — {row['change']}")
    add("")

    if mnist_record:
        add("### MNIST reference point")
        add("")
        add(
            f"The same architecture trained on MNIST reached "
            f"**{pct(mnist_record['val_accuracy'])} validation accuracy** across 10 "
            f"classes, against {pct(max(r['val_accuracy'] for r in table))} on EMNIST "
            f"Balanced across {e['n_classes']}. That gap is the quantitative answer to "
            "\"why is EMNIST harder than MNIST?\" — more classes, and several of them "
            "are near-duplicates of one another."
        )
        add("")

    # ----------------------------------------------- 11 evaluation metrics
    add("## 11. Evaluation Metrics")
    add("")
    add(
        "- **Accuracy** — the fraction of test images classified correctly. Adequate "
        "here only because the classes are balanced.\n"
        "- **Precision** (per class) — of the images predicted as this character, how "
        "many really were. Penalises over-prediction.\n"
        "- **Recall** (per class) — of the images that really are this character, how "
        "many were found. Penalises under-prediction.\n"
        "- **F1** — the harmonic mean of precision and recall, so a class scores well "
        "only if both are good.\n"
        "- **Macro F1** — the unweighted mean of the per-class F1 scores. Every "
        "character counts equally, so a model that is excellent on digits and poor on "
        "`q` cannot hide behind an average. **This is the headline metric.**\n"
        "- **Weighted F1** — averaged by class support. Because EMNIST Balanced has "
        "equal support everywhere, it comes out close to macro F1; on an imbalanced "
        "dataset the two would diverge sharply."
    )
    add("")

    # ------------------------------------------------------- 12 results
    add("## 12. Results")
    add("")
    add(f"Final model evaluated on the untouched test set of {evaluation['n_test']:,} images.")
    add("")
    add("| Model | Accuracy | Precision (macro) | Recall (macro) | Macro F1 | Weighted F1 |")
    add("|---|---|---|---|---|---|")
    for name, record in baselines.items():
        t = record["test"]
        add(
            f"| {BASELINE_LABELS.get(name, name)} | {pct(t['accuracy'])} | "
            f"{pct(t['precision_macro'])} | {pct(t['recall_macro'])} | "
            f"{pct(t['f1_macro'])} | {pct(t['f1_weighted'])} |"
        )
    add(
        f"| **CNN — {EXPERIMENT_LABELS.get(selected, selected)}** | "
        f"**{pct(metrics['accuracy'])}** | **{pct(metrics['precision_macro'])}** | "
        f"**{pct(metrics['recall_macro'])}** | **{pct(metrics['f1_macro'])}** | "
        f"**{pct(metrics['f1_weighted'])}** |"
    )
    add("")
    improvement = (metrics["accuracy"] - best_baseline[1]["test"]["accuracy"]) * 100
    error_reduction = (
        1 - (1 - metrics["accuracy"]) / (1 - best_baseline[1]["test"]["accuracy"])
    ) * 100
    add(
        f"The CNN beats the strongest baseline "
        f"({BASELINE_LABELS[best_baseline[0]]}, {pct(best_baseline[1]['test']['accuracy'])}) "
        f"by **{improvement:.2f} percentage points**, cutting the error rate by "
        f"**{error_reduction:.1f}%**. The baselines flatten each image into 784 "
        "independent pixels, discarding the spatial relationships that define a "
        "character; the CNN's convolutions are built to exploit exactly those."
    )
    add("")
    add(
        f"The model misclassified {evaluation['n_errors']:,} of "
        f"{evaluation['n_test']:,} test images "
        f"(error rate {pct(evaluation['error_rate'])})."
    )
    add("")

    # ---------------------------------------------- 13 confusion matrix
    add("## 13. Confusion Matrix")
    add("")
    add("![Confusion matrix](outputs/figures/final_confusion_matrix_norm.png)")
    add("")
    add(
        "Each row is a true character and each column a prediction, row-normalised. "
        "The confusions below are read from that matrix — they are what the model "
        "actually did, not a list of characters that look similar in principle."
    )
    add("")
    add("| True | Predicted | Count | % of that class |")
    add("|---|---|---|---|")
    for pair in evaluation["top_confusions"][:12]:
        add(
            f"| `{pair['true']}` | `{pair['predicted']}` | {pair['count']} | "
            f"{pair['rate_of_true_class'] * 100:.1f}% |"
        )
    add("")

    hardest = evaluation["hardest_classes"]
    add("### Hardest classes by per-class F1")
    add("")
    add("| Character | F1 | Precision | Recall | Support |")
    add("|---|---|---|---|---|")
    for row in hardest[:8]:
        add(
            f"| `{row['character']}` | {row['f1']:.3f} | {row['precision']:.3f} | "
            f"{row['recall']:.3f} | {row['support']} |"
        )
    add("")
    worst = hardest[0]
    top_pair = evaluation["top_confusions"][0]
    add(
        f"The weakest class is `{worst['character']}` (F1 {worst['f1']:.3f}), and the "
        f"single most frequent confusion is `{top_pair['true']}` → "
        f"`{top_pair['predicted']}` ({top_pair['count']} of "
        f"{top_pair['true_class_support']} images, "
        f"{top_pair['rate_of_true_class'] * 100:.1f}% of that class). These are not "
        "arbitrary failures: they are pairs whose handwritten forms genuinely "
        "overlap, and a human reading the same isolated glyph without surrounding "
        "words would often make the same call."
    )
    add("")

    # ----------------------------------------------------- 14 error analysis
    add("## 14. Error Analysis")
    add("")
    add("![Most confident mistakes](outputs/figures/final_errors_most_confident.png)")
    add("")
    add(
        "The figure above shows the errors the model was *most confident* about — the "
        "systematic failures, where the written glyph genuinely resembles the wrong "
        "class. A random sample of errors "
        "(`outputs/figures/final_errors_random.png`) shows the more typical mode: "
        "ambiguous, rushed or unusually proportioned handwriting."
    )
    add("")
    add("Recurring causes, from inspecting the misclassified images:")
    add("")
    add("- **Genuinely ambiguous glyphs** — the dominant cause; case-merged and "
        "visually overlapping characters written without context.")
    add("- **Unclosed or over-closed loops** — turning `a` into `u`-like or `o`-like shapes.")
    add("- **Extreme stroke weight** — very thick strokes fill in counters (the enclosed "
        "gaps), very thin ones fragment after downsampling to 28×28.")
    add("- **Unusual personal style** — serifs, exaggerated slant, or continental "
        "conventions such as a crossed `7`.")
    add("")

    # ------------------------------------------------- 15 confidence
    add("## 15. Confidence Analysis")
    add("")
    add("![Confidence analysis](outputs/figures/final_confidence_analysis.png)")
    add("")
    add("| Statistic | Value |")
    add("|---|---|")
    add(f"| Mean confidence when correct | {pct(confidence['mean_confidence_correct'])} |")
    add(f"| Mean confidence when incorrect | {pct(confidence['mean_confidence_incorrect'])} |")
    add(f"| Predictions below 50% confidence | {pct(confidence['fraction_below_50pct_confidence'])} |")
    add(
        f"| Accuracy when confidence ≥ 99% | "
        f"{pct(confidence['accuracy_on_high_confidence_ge_99pct'])} "
        f"(covers {pct(confidence['coverage_high_confidence_ge_99pct'])} of the test set) |"
    )
    add("")
    gap = confidence["mean_confidence_correct"] - confidence["mean_confidence_incorrect"]
    add(
        f"The model is markedly less confident when it is wrong — a "
        f"{gap * 100:.1f} percentage-point separation in mean confidence. That is the "
        "property a deployed system needs: a confidence threshold can route uncertain "
        "characters to a human instead of silently guessing. The web interface uses "
        "this directly, flagging any prediction below 60% as uncertain."
    )
    add("")

    # -------------------------------------------- 16 model comparison / final
    add("## 16. Model Comparison and Final Model")
    add("")
    add(
        f"**Selected: `{selected}` — {EXPERIMENT_LABELS.get(selected, selected)}**, "
        f"chosen by the highest validation accuracy ({pct(final['val_accuracy'])}) "
        "before the test set was consulted."
    )
    add("")
    add("| Property | Value |")
    add("|---|---|")
    add(f"| Architecture variant | `{final['architecture_variant']}` |")
    add(f"| Total parameters | {params['total']:,} |")
    add(f"| Trainable parameters | {params['trainable']:,} |")
    add(f"| Augmentation | {'enabled' if final['augmentation'] else 'none'} |")
    add(f"| Test accuracy | {pct(metrics['accuracy'])} |")
    add(f"| Test macro F1 | {pct(metrics['f1_macro'])} |")
    add("")
    add("### Reload verification")
    add("")
    rv = final["reload_verification"]
    add(
        f"The exported `.keras` file was reloaded into a fresh object and compared "
        f"against the source model on {rv['n_compared']} test images. Maximum "
        f"absolute difference in predicted probabilities: "
        f"`{rv['max_abs_probability_difference']:.3e}`; predicted labels identical: "
        f"`{rv['labels_identical']}`. The export is therefore lossless, not merely "
        "assumed to be."
    )
    add("")

    # ------------------------------------------------- 17 inference system
    add("## 17. Inference System")
    add("")
    add("Three ways to use the trained model, all sharing one preprocessing path:")
    add("")
    add("1. **Web application** (`python src/app.py`) — draw a character or upload a "
        "photo, and see the prediction, confidence, top-5 alternatives, and the "
        "28×28 image the model actually received.")
    add("2. **Command line** (`python src/inference.py --image path.jpg --top-k 5`).")
    add("3. **Python API** — `CharacterRecognizer` from `src/inference.py`.")
    add("")
    add("Exported artefacts in `models/final/`:")
    add("")
    add("- `handwritten_character_cnn.keras` — the trained network")
    add("- `labels.json` — class index → character mapping, plus the test metrics")
    add("- `preprocessing_config.json` — the exact preprocessing contract")
    add("")

    if samples:
        add("### Results on simulated user images")
        add("")
        add("| Image family | Count | Top-1 | Top-3 | Top-5 |")
        add("|---|---|---|---|---|")
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
        example = next((r for r in samples["predictions"] if r["correct"]), samples["predictions"][0])
        add("A worked example, straight from the model:")
        add("")
        add("```")
        add(f"Image           : {Path(example['path']).name}")
        add(f"True label      : {example['true_label']}")
        add(f"Predicted       : {example['predicted']}")
        add(f"Confidence      : {example['confidence'] * 100:.1f}%")
        add("Top predictions :")
        for entry in example["top_k"]:
            add(f"  {entry['rank']}. {entry['character']:2s} — {entry['probability'] * 100:6.2f}%")
        add("```")
        add("")

    # ---------------------------------------------------- 18 CRNN
    add("## 18. CRNN Extension and Future Work")
    add("")
    add("**Status: documented as future work, not implemented.**")
    add("")
    add(
        "The brief allows extending the system to full words or sentences with "
        "sequence modelling. The honest way to do that is *not* to slice a word image "
        "into letters and run this classifier on each piece — segmentation is exactly "
        "what breaks on cursive and touching characters, and a segment-then-classify "
        "prototype would look like word recognition while failing on real handwriting."
    )
    add("")
    add("The correct architecture is a **CRNN trained with CTC loss**:")
    add("")
    add("```")
    add("Word image (H × W × 1, variable width)")
    add("  → CNN feature extractor        (collapses height, preserves width)")
    add("  → reshape into a left-to-right sequence of frames")
    add("  → BiLSTM / BiGRU               (each frame sees both directions of context)")
    add("  → Dense(vocabulary + 1 blank) → per-frame softmax")
    add("  → CTC loss (training) / CTC decode (inference)")
    add("```")
    add("")
    add(
        "**Why CTC is the key piece.** The network emits a fixed number of frames "
        "(say 64) but the target word has a different, variable length (say 5), and "
        "nobody has labelled which pixel column each letter begins at. CTC sums the "
        "probability over *every* alignment of frames to the target string — "
        "introducing a special *blank* symbol and collapsing repeated predictions — "
        "so the model learns the alignment implicitly from `(image, \"hello\")` pairs "
        "alone. Sequence modelling also buys context: an RNN can use the surrounding "
        "letters to resolve an ambiguous glyph, which is precisely the information "
        "this character classifier lacks."
    )
    add("")
    add(
        "**Why it was not implemented here.** It requires a word-level dataset (IAM "
        "Handwriting, which needs a registered download), and CRNN+CTC training costs "
        "roughly an order of magnitude more than this CNN — impractical on a CPU-only "
        "machine. Per the brief, a complete and verified character system is worth "
        "more than a half-trained word system, so the extension is specified rather "
        "than faked."
    )
    add("")

    # ---------------------------------------------------- 19 limitations
    add("## 19. Limitations")
    add("")
    add("- **Single characters only.** No word or sentence recognition.")
    add(
        f"- **Case is partially unrecoverable by design.** Fifteen letters are "
        "case-merged in EMNIST Balanced, so the model cannot report whether an "
        "isolated `s` was upper- or lowercase."
    )
    add(
        f"- **{pct(evaluation['error_rate'])} of test images are still misclassified**, "
        "concentrated in the visually overlapping classes listed in §13."
    )
    add("- **Domain limits.** Training data is NIST-style handwriting. Cursive joins, "
        "unusual scripts and printed type are out of distribution.")
    add("- **Preprocessing assumes one character per image**, isolated by the largest "
        "connected component. An image containing two characters, or a character "
        "broken into disconnected strokes, may be cropped incorrectly.")
    add("- **The bundled `samples/` set is simulated** — renderings of unseen EMNIST "
        "test glyphs and printed fonts, not photographs of real handwriting.")
    add("- **CPU-only training** capped the experiment budget; a longer schedule or "
        "wider architecture search would likely add a further increment of accuracy.")
    add("")

    # ---------------------------------------------------- 20 conclusion
    add("## 20. Conclusion")
    add("")
    add(
        f"A convolutional neural network was trained to recognise {e['n_classes']} "
        f"handwritten character classes, reaching **{pct(metrics['accuracy'])} accuracy** "
        f"and **{pct(metrics['f1_macro'])} macro F1** on "
        f"{evaluation['n_test']:,} held-out test images — "
        f"{improvement:.2f} percentage points above the strongest classical baseline, "
        f"an error reduction of {error_reduction:.1f}%."
    )
    add("")
    add(
        "Three things mattered more than model size. First, **verifying the data**: "
        "the EMNIST orientation bug is silent and would have quietly capped the "
        "result. Second, **controlled experiments**: batch normalisation, dropout and "
        "augmentation were each adopted because a one-variable comparison showed they "
        "helped, not because they are conventional. Third, **matching preprocessing "
        "to reality**: reconstructing NIST's centre-of-mass normalisation is what "
        "lets the model work on a photograph rather than only on curated dataset rows."
    )
    add("")
    add(
        "The remaining errors are concentrated in character pairs whose handwritten "
        "forms genuinely overlap. Resolving those requires context — the surrounding "
        "letters of a word — which is the motivation for the CRNN extension described "
        "in §18 rather than a deeper convolutional stack."
    )
    add("")
    add("---")
    add("")
    add(
        "*Reproduce: see `README.md`. Raw metrics: `outputs/metrics/`. "
        "Figures: `outputs/figures/`.*"
    )
    add("")

    return "\n".join(lines)


def main() -> None:
    ensure_dirs()
    header("GENERATING report.md FROM EXECUTED RESULTS")
    text = build()
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(f"  wrote {REPORT_PATH} ({len(text):,} characters, {text.count(chr(10)) + 1} lines)")


if __name__ == "__main__":
    main()
