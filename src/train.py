"""Experiment runner: trains one CNN variant and records its results.

Data-leakage policy
-------------------
The official EMNIST test split is loaded here only to report its size; it is
never fed to ``fit``, never used for early stopping, and never used to pick
a checkpoint.  Validation data is carved out of the *training* split with a
stratified, seeded split.  All model selection happens on validation.
Final test numbers are produced exclusively by ``evaluate.py``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from augmentation import augmentation_config, make_dataset
from data_loader import load_emnist, load_mnist, stratified_split
from models import build_variant, count_parameters
from preprocessing import prepare_arrays
from utils import (
    FINAL_MODELS_DIR,
    METRICS_DIR,
    SEED,
    describe_environment,
    ensure_dirs,
    header,
    save_json,
    set_seed,
)

EXPERIMENTS_DIR = METRICS_DIR / "experiments"

#: The experiment grid.  Each entry changes exactly one thing relative to
#: the previous one, so the comparison isolates that component.
EXPERIMENTS: dict[str, dict] = {
    "exp2_cnn_simple": {
        "variant": "simple",
        "augment": False,
        "question": "How far does a plain CNN (no BN, no dropout) get?",
    },
    "exp3_cnn_bn": {
        "variant": "bn",
        "augment": False,
        "question": "Does batch normalisation help optimisation/accuracy?",
    },
    "exp4_cnn_bn_drop": {
        "variant": "bn_drop",
        "augment": False,
        "question": "Does dropout reduce the train/validation gap?",
    },
    "exp5_cnn_bn_drop_aug": {
        "variant": "bn_drop",
        "augment": True,
        "epochs": 18,
        "question": "Does label-preserving augmentation improve generalisation?",
    },
    "exp6_cnn_deep_aug_lr": {
        "variant": "deep",
        "augment": True,
        "lr_schedule": True,
        "epochs": 25,
        "question": "Does a deeper model with LR scheduling beat exp5?",
    },
}


def load_dataset(dataset: str) -> dict:
    """Load a dataset by short name."""
    if dataset == "mnist":
        return load_mnist()
    if dataset == "emnist_balanced":
        return load_emnist("balanced")
    raise ValueError(f"Unknown dataset {dataset!r}")


def build_splits(dataset: str, val_fraction: float = 0.1, batch_size: int = 128, augment: bool = False):
    """Return tf.data pipelines plus the raw arrays and metadata."""
    data = load_dataset(dataset)
    x_tr_raw, y_tr, x_val_raw, y_val = stratified_split(
        data["x_train"], data["y_train"], val_fraction=val_fraction
    )
    x_tr = prepare_arrays(x_tr_raw)
    x_val = prepare_arrays(x_val_raw)

    train_ds = make_dataset(x_tr, y_tr, batch_size=batch_size, shuffle=True, augment=augment)
    val_ds = make_dataset(x_val, y_val, batch_size=batch_size, shuffle=False, augment=False)
    return train_ds, val_ds, (x_tr, y_tr, x_val, y_val), data


def make_callbacks(checkpoint_path: Path, lr_schedule: bool, patience: int = 5) -> list:
    """Early stopping, best-model checkpointing and optional LR reduction."""
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=0,
        ),
    ]
    if lr_schedule:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1
            )
        )
    return callbacks


def run_experiment(
    name: str,
    dataset: str = "emnist_balanced",
    epochs: int | None = None,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    val_fraction: float = 0.1,
    default_epochs: int = 12,
) -> dict:
    """Train one experiment and persist its model, history and metrics."""
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {name!r}; choose from {list(EXPERIMENTS)}")
    spec = EXPERIMENTS[name]
    epochs = epochs or spec.get("epochs", default_epochs)

    ensure_dirs()
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    header(f"{name}  |  dataset={dataset}  |  {spec['question']}")

    train_ds, val_ds, arrays, data = build_splits(
        dataset, val_fraction=val_fraction, batch_size=batch_size, augment=spec["augment"]
    )
    x_tr, y_tr, x_val, y_val = arrays
    n_classes = len(data["class_names"])

    model = build_variant(spec["variant"], n_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    params = count_parameters(model)
    print(
        f"train={len(x_tr):,}  val={len(x_val):,}  test(untouched)={len(data['x_test']):,}  "
        f"classes={n_classes}  params={params['total']:,}",
        flush=True,
    )

    checkpoint_path = FINAL_MODELS_DIR / f"{dataset}_{name}.keras"
    callbacks = make_callbacks(checkpoint_path, spec.get("lr_schedule", False))

    start = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )
    train_seconds = time.perf_counter() - start

    # EarlyStopping restored the best weights; persist that exact model.
    model.save(checkpoint_path)

    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    best_epoch = int(np.argmin(hist["val_loss"]))
    record = {
        "name": name,
        "dataset": dataset,
        "question": spec["question"],
        "variant": spec["variant"],
        "augmentation": augmentation_config() if spec["augment"] else None,
        "lr_schedule": bool(spec.get("lr_schedule", False)),
        "hyperparameters": {
            "optimizer": "adam",
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "max_epochs": epochs,
            "epochs_run": len(hist["loss"]),
            "val_fraction": val_fraction,
            "seed": SEED,
            "early_stopping_patience": 5,
        },
        "parameters": params,
        "n_train": int(len(x_tr)),
        "n_val": int(len(x_val)),
        "n_test_untouched": int(len(data["x_test"])),
        "n_classes": n_classes,
        "train_seconds": round(train_seconds, 1),
        "seconds_per_epoch": round(train_seconds / max(len(hist["loss"]), 1), 1),
        "best_epoch_1based": best_epoch + 1,
        "val_accuracy": hist["val_accuracy"][best_epoch],
        "val_loss": hist["val_loss"][best_epoch],
        "train_accuracy_at_best": hist["accuracy"][best_epoch],
        "train_val_gap": hist["accuracy"][best_epoch] - hist["val_accuracy"][best_epoch],
        "history": hist,
        "model_path": str(checkpoint_path),
        "environment": describe_environment(),
    }
    save_json(record, EXPERIMENTS_DIR / f"{dataset}_{name}.json")

    print(
        f"\n>> {name}: best epoch {best_epoch + 1}/{len(hist['loss'])}  "
        f"val_acc={record['val_accuracy']:.4f}  val_loss={record['val_loss']:.4f}  "
        f"gap={record['train_val_gap']:+.4f}  time={train_seconds / 60:.1f} min",
        flush=True,
    )
    return record


def collect_experiment_table(dataset: str = "emnist_balanced") -> list[dict]:
    """Build the experiment comparison table from saved records."""
    rows = []
    for name in EXPERIMENTS:
        path = EXPERIMENTS_DIR / f"{dataset}_{name}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "experiment": name,
                "change": record["question"],
                "params": record["parameters"]["total"],
                "epochs_run": record["hyperparameters"]["epochs_run"],
                "val_accuracy": record["val_accuracy"],
                "val_loss": record["val_loss"],
                "train_val_gap": record["train_val_gap"],
                "minutes": round(record["train_seconds"] / 60, 1),
            }
        )
    save_json(rows, METRICS_DIR / f"experiment_table_{dataset}.json")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CNN experiment.")
    parser.add_argument("--experiment", default="all", help="experiment name or 'all'")
    parser.add_argument("--dataset", default="emnist_balanced", choices=["emnist_balanced", "mnist"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--default-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    names = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    for name in names:
        run_experiment(
            name,
            dataset=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            default_epochs=args.default_epochs,
        )

    table = collect_experiment_table(args.dataset)
    header("EXPERIMENT TABLE (validation only - test set untouched)")
    print(f"{'experiment':26s} {'params':>9s} {'ep':>3s} {'val_acc':>8s} {'val_loss':>9s} {'gap':>8s} {'min':>6s}")
    for row in table:
        print(
            f"{row['experiment']:26s} {row['params']:>9,} {row['epochs_run']:>3d} "
            f"{row['val_accuracy']:>8.4f} {row['val_loss']:>9.4f} {row['train_val_gap']:>+8.4f} {row['minutes']:>6.1f}"
        )


if __name__ == "__main__":
    main()
