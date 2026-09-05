"""Shared utilities: paths, seeding, timing and JSON/artifact helpers.

Every other module imports its paths from here so that the project has a
single source of truth for where data, models and outputs live.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np


def _make_stdout_safe() -> None:
    """Never let a stray non-ASCII character crash a script on Windows.

    Windows consoles default to cp1252, which cannot encode characters such
    as arrows or Greek letters.  Printing one raises UnicodeEncodeError and
    kills an otherwise-successful run, so switch the streams to UTF-8 and
    degrade gracefully if the terminal cannot render a glyph.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="backslashreplace")
            except (ValueError, OSError):  # pragma: no cover - detached stream
                pass


_make_stdout_safe()


# --------------------------------------------------------------------------
# Project paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
BASELINE_MODELS_DIR = MODELS_DIR / "baseline"
FINAL_MODELS_DIR = MODELS_DIR / "final"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

SAMPLES_DIR = PROJECT_ROOT / "samples"

#: Single global seed used for every stochastic component of the project.
SEED = 42


def ensure_dirs() -> None:
    """Create every project directory if it does not already exist."""
    for path in (
        RAW_DIR,
        PROCESSED_DIR,
        BASELINE_MODELS_DIR,
        FINAL_MODELS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        PREDICTIONS_DIR,
        SAMPLES_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and (if importable) TensorFlow.

    Note: full bit-for-bit determinism is not guaranteed on CPU because
    oneDNN kernels may reduce in a non-deterministic order.  Seeding still
    removes the dominant sources of run-to-run variation (weight init,
    shuffling, augmentation and dropout masks).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # TensorFlow is optional for the pure-NumPy parts of the project.
        import tensorflow as tf

        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
    except Exception:  # pragma: no cover - only hit when TF is absent
        pass


def save_json(obj: Any, path: str | Path, indent: int = 2) -> Path:
    """Write ``obj`` to ``path`` as UTF-8 JSON, creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=indent, ensure_ascii=False, default=_json_default)
    return path


def load_json(path: str | Path) -> Any:
    """Read UTF-8 JSON from ``path``."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(obj: Any) -> Any:
    """Make NumPy scalars/arrays JSON serialisable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serialisable")


@contextmanager
def timer(label: str) -> Iterator[dict]:
    """Context manager that measures and prints wall-clock duration."""
    record: dict = {"label": label}
    start = time.perf_counter()
    try:
        yield record
    finally:
        record["seconds"] = time.perf_counter() - start
        print(f"[timer] {label}: {record['seconds']:.2f}s", flush=True)


def header(title: str, width: int = 74) -> None:
    """Print a consistent section banner (keeps long logs readable)."""
    print("\n" + "=" * width)
    print(title)
    print("=" * width, flush=True)


def describe_environment() -> dict:
    """Collect versions/hardware so results can be reproduced later."""
    import platform

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "seed": SEED,
    }
    try:
        import tensorflow as tf

        info["tensorflow"] = tf.__version__
        info["gpus"] = [d.name for d in tf.config.list_physical_devices("GPU")]
    except Exception:
        info["tensorflow"] = None
        info["gpus"] = []
    try:
        import sklearn

        info["scikit_learn"] = sklearn.__version__
    except Exception:
        info["scikit_learn"] = None
    return info
