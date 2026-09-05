"""Dataset loading for MNIST and EMNIST.

The EMNIST archive published by NIST (``gzip.zip``) contains every EMNIST
variant in the classic IDX format used by MNIST.  This module reads the
requested variant straight out of the archive (no full extraction of the
536 MB zip) and caches the decoded arrays as ``.npz`` under ``data/processed``
so later runs start instantly.

IMPORTANT — orientation
-----------------------
EMNIST images are stored with the row/column axes swapped relative to the
MNIST convention, so a naive ``reshape(28, 28)`` yields characters that are
mirrored and rotated by 90 degrees.  ``load_emnist`` applies a transpose by
default; ``verify_orientation`` in this module checks that decision against
MNIST empirically instead of trusting it.
"""

from __future__ import annotations

import gzip
import io
import struct
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from utils import PROCESSED_DIR, RAW_DIR, SEED

EMNIST_ZIP_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
EMNIST_ZIP_PATH = RAW_DIR / "emnist_gzip.zip"

MNIST_NPZ_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
MNIST_NPZ_PATH = RAW_DIR / "mnist.npz"

#: EMNIST variants and the number of classes each one defines.
EMNIST_VARIANTS = {
    "balanced": 47,
    "byclass": 62,
    "bymerge": 47,
    "letters": 26,
    "digits": 10,
    "mnist": 10,
}


# --------------------------------------------------------------------------
# Low-level IDX decoding
# --------------------------------------------------------------------------
def _read_idx(buffer: bytes) -> np.ndarray:
    """Decode an IDX (MNIST-format) byte buffer into a NumPy array."""
    magic, count = struct.unpack(">II", buffer[:8])
    if magic == 2051:  # 3-D tensor: images
        rows, cols = struct.unpack(">II", buffer[8:16])
        data = np.frombuffer(buffer, dtype=np.uint8, offset=16)
        return data.reshape(count, rows, cols)
    if magic == 2049:  # 1-D tensor: labels
        data = np.frombuffer(buffer, dtype=np.uint8, offset=8)
        return data.reshape(count)
    raise ValueError(f"Unrecognised IDX magic number: {magic}")


def _zip_member(archive: zipfile.ZipFile, suffix: str) -> str:
    """Find the single archive member whose name ends with ``suffix``."""
    matches = [n for n in archive.namelist() if n.endswith(suffix)]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one archive member ending in {suffix!r}, found {matches}"
        )
    return matches[0]


def _read_gz_member(archive: zipfile.ZipFile, suffix: str) -> np.ndarray:
    """Read a gzipped IDX member out of the EMNIST zip without extracting it."""
    with archive.open(_zip_member(archive, suffix)) as member:
        raw = gzip.decompress(member.read())
    return _read_idx(raw)


# --------------------------------------------------------------------------
# Downloads
# --------------------------------------------------------------------------
def _download(url: str, destination: Path) -> Path:
    """Download ``url`` to ``destination`` unless the file already exists."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    print(f"Downloading {url} -> {destination} ...", flush=True)
    urllib.request.urlretrieve(url, destination)
    print(f"Downloaded {destination.stat().st_size / 1e6:.1f} MB", flush=True)
    return destination


# --------------------------------------------------------------------------
# EMNIST
# --------------------------------------------------------------------------
def load_emnist_mapping(variant: str = "balanced") -> dict[int, str]:
    """Return ``{class_index: character}`` from the variant's mapping file."""
    _download(EMNIST_ZIP_URL, EMNIST_ZIP_PATH)
    with zipfile.ZipFile(EMNIST_ZIP_PATH) as archive:
        name = _zip_member(archive, f"emnist-{variant}-mapping.txt")
        text = archive.read(name).decode("utf-8")
    mapping: dict[int, str] = {}
    for line in text.strip().splitlines():
        index, ascii_code = line.split()
        mapping[int(index)] = chr(int(ascii_code))
    return mapping


def load_emnist(
    variant: str = "balanced",
    orientation: str = "transpose",
    use_cache: bool = True,
) -> dict:
    """Load an EMNIST variant as raw ``uint8`` arrays.

    Parameters
    ----------
    variant:
        One of :data:`EMNIST_VARIANTS`.
    orientation:
        ``"transpose"`` swaps the row/column axes (the correct fix for
        EMNIST, verified in :func:`verify_orientation`).  ``"none"`` keeps
        the bytes exactly as stored and exists so the orientation check can
        compare the two.
    use_cache:
        Reuse/write the decoded ``.npz`` cache under ``data/processed``.

    Returns
    -------
    dict with ``x_train``, ``y_train``, ``x_test``, ``y_test``, ``mapping``
    and ``class_names``.
    """
    if variant not in EMNIST_VARIANTS:
        raise ValueError(f"Unknown EMNIST variant {variant!r}; choose from {list(EMNIST_VARIANTS)}")
    if orientation not in {"transpose", "none"}:
        raise ValueError("orientation must be 'transpose' or 'none'")

    cache = PROCESSED_DIR / f"emnist_{variant}_{orientation}.npz"
    mapping = load_emnist_mapping(variant)

    if use_cache and cache.exists():
        with np.load(cache) as data:
            arrays = {key: data[key] for key in ("x_train", "y_train", "x_test", "y_test")}
    else:
        _download(EMNIST_ZIP_URL, EMNIST_ZIP_PATH)
        with zipfile.ZipFile(EMNIST_ZIP_PATH) as archive:
            arrays = {
                "x_train": _read_gz_member(archive, f"emnist-{variant}-train-images-idx3-ubyte.gz"),
                "y_train": _read_gz_member(archive, f"emnist-{variant}-train-labels-idx1-ubyte.gz"),
                "x_test": _read_gz_member(archive, f"emnist-{variant}-test-images-idx3-ubyte.gz"),
                "y_test": _read_gz_member(archive, f"emnist-{variant}-test-labels-idx1-ubyte.gz"),
            }
        if orientation == "transpose":
            # Swap the H and W axes of every image (axis 0 is the batch).
            arrays["x_train"] = np.ascontiguousarray(arrays["x_train"].transpose(0, 2, 1))
            arrays["x_test"] = np.ascontiguousarray(arrays["x_test"].transpose(0, 2, 1))
        if variant == "letters":
            # The 'letters' variant labels classes 1..26; shift to 0..25.
            arrays["y_train"] = arrays["y_train"] - 1
            arrays["y_test"] = arrays["y_test"] - 1
        if use_cache:
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, **arrays)

    class_names = [mapping[i] for i in sorted(mapping)]
    if variant == "letters":
        class_names = [mapping[i] for i in sorted(mapping)]
    return {**arrays, "mapping": mapping, "class_names": class_names, "variant": variant}


# --------------------------------------------------------------------------
# MNIST
# --------------------------------------------------------------------------
def load_mnist() -> dict:
    """Load MNIST as raw ``uint8`` arrays from the official ``.npz``."""
    _download(MNIST_NPZ_URL, MNIST_NPZ_PATH)
    with np.load(MNIST_NPZ_PATH, allow_pickle=True) as data:
        arrays = {
            "x_train": data["x_train"],
            "y_train": data["y_train"],
            "x_test": data["x_test"],
            "y_test": data["y_test"],
        }
    class_names = [str(d) for d in range(10)]
    return {
        **arrays,
        "mapping": {i: str(i) for i in range(10)},
        "class_names": class_names,
        "variant": "mnist",
    }


# --------------------------------------------------------------------------
# Orientation verification (Phase 3)
# --------------------------------------------------------------------------
def _class_mean_images(x: np.ndarray, y: np.ndarray, classes: range) -> np.ndarray:
    """Mean image per class, as float32 in [0, 1]."""
    return np.stack([x[y == c].mean(axis=0) for c in classes]).astype(np.float32) / 255.0


def verify_orientation(variant: str = "balanced", n_per_class: int = 400) -> dict:
    """Decide EMNIST's correct orientation by comparing digits with MNIST.

    EMNIST classes 0-9 are the same ten digits as MNIST, drawn from the same
    NIST source.  For each candidate transform we compute the mean image of
    every digit class and correlate it with the corresponding MNIST mean
    image.  The transform with the highest mean correlation is the correct
    one.  This replaces "it looks right" with a measurement.
    """
    mnist = load_mnist()
    mnist_means = _class_mean_images(mnist["x_train"], mnist["y_train"], range(10))

    raw = load_emnist(variant=variant, orientation="none", use_cache=True)
    x, y = raw["x_train"], raw["y_train"]

    # Subsample per digit class to keep the check fast.
    rng = np.random.default_rng(SEED)
    keep = np.concatenate(
        [rng.choice(np.flatnonzero(y == c), size=n_per_class, replace=False) for c in range(10)]
    )
    x, y = x[keep], y[keep]

    candidates = {
        "as_stored": lambda a: a,
        "transpose": lambda a: a.transpose(0, 2, 1),
        "rot90_ccw": lambda a: np.rot90(a, k=1, axes=(1, 2)),
        "rot90_cw": lambda a: np.rot90(a, k=-1, axes=(1, 2)),
        "flip_lr": lambda a: a[:, :, ::-1],
        "flip_ud": lambda a: a[:, ::-1, :],
    }

    scores: dict[str, float] = {}
    for name, transform in candidates.items():
        means = _class_mean_images(transform(x), y, range(10))
        correlations = [
            float(np.corrcoef(means[c].ravel(), mnist_means[c].ravel())[0, 1]) for c in range(10)
        ]
        scores[name] = float(np.mean(correlations))

    best = max(scores, key=scores.get)
    return {"scores": scores, "best": best, "correct": best == "transpose"}


# --------------------------------------------------------------------------
# Splitting (Phase 7)
# --------------------------------------------------------------------------
def stratified_split(
    x: np.ndarray,
    y: np.ndarray,
    val_fraction: float = 0.1,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split ``(x, y)`` into train/validation preserving class proportions.

    The official EMNIST test set is never touched here — validation data is
    carved out of the training set only.
    """
    rng = np.random.default_rng(seed)
    train_idx: list[np.ndarray] = []
    val_idx: list[np.ndarray] = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_fraction))
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])

    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def dataset_summary(data: dict) -> dict:
    """Compute the dataset statistics required by Phase 2 from real arrays."""
    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]
    train_counts = np.bincount(y_train, minlength=len(data["class_names"]))
    test_counts = np.bincount(y_test, minlength=len(data["class_names"]))
    return {
        "variant": data["variant"],
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "image_shape": list(x_train.shape[1:]),
        "n_classes": int(len(data["class_names"])),
        "class_names": data["class_names"],
        "dtype": str(x_train.dtype),
        "pixel_min": int(x_train.min()),
        "pixel_max": int(x_train.max()),
        "pixel_mean": float(x_train.mean()),
        "pixel_std": float(x_train.std()),
        "train_class_counts": train_counts.tolist(),
        "test_class_counts": test_counts.tolist(),
        "train_count_min": int(train_counts.min()),
        "train_count_max": int(train_counts.max()),
        "imbalance_ratio": float(train_counts.max() / max(train_counts.min(), 1)),
    }
