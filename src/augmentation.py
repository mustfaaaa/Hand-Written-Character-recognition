"""Label-preserving augmentation and the ``tf.data`` input pipelines.

Design rule
-----------
A handwritten character's *identity* is encoded in its shape **and** its
orientation.  Any transform that can turn one class into another is banned:

* no horizontal flip  (b <-> d, p <-> q)
* no vertical flip    (M <-> W, b <-> p, 6 <-> 9-ish)
* no 90/180 degree rotation (6 <-> 9, N <-> Z)
* rotations kept to +/-10 degrees, which covers natural slant variation
  without approaching the angle where 6 becomes 9.

Augmentation is applied by :func:`make_dataset` **only** when
``augment=True``, which the training script sets for the training split and
never for validation or test.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from utils import SEED

AUTOTUNE = tf.data.AUTOTUNE

#: Augmentation strengths, expressed in the units Keras expects.
#: RandomRotation's factor is a fraction of a full turn: 10 deg = 10/360.
ROTATION_FACTOR = 10.0 / 360.0
TRANSLATION_FACTOR = 0.10  # +/-10% of 28 px ~ +/-2.8 px
ZOOM_FACTOR = 0.10  # +/-10% scale


def build_augmenter(seed: int = SEED) -> tf.keras.Sequential:
    """Return the augmentation stack used for training batches."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(
                factor=ROTATION_FACTOR,
                fill_mode="constant",
                fill_value=0.0,
                seed=seed,
            ),
            tf.keras.layers.RandomTranslation(
                height_factor=TRANSLATION_FACTOR,
                width_factor=TRANSLATION_FACTOR,
                fill_mode="constant",
                fill_value=0.0,
                seed=seed,
            ),
            tf.keras.layers.RandomZoom(
                height_factor=ZOOM_FACTOR,
                width_factor=ZOOM_FACTOR,
                fill_mode="constant",
                fill_value=0.0,
                seed=seed,
            ),
        ],
        name="augmentation",
    )


def make_dataset(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int = 128,
    shuffle: bool = False,
    augment: bool = False,
    seed: int = SEED,
) -> tf.data.Dataset:
    """Build a ``tf.data`` pipeline over in-memory arrays.

    Parameters
    ----------
    x, y:
        Preprocessed images ``(N, 28, 28, 1)`` float32 and integer labels.
    shuffle:
        Shuffle with a full-size buffer (train split only).
    augment:
        Apply :func:`build_augmenter`.  Never enable this for validation or
        test data.
    """
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(x), 20000), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)

    if augment:
        augmenter = build_augmenter(seed=seed)
        dataset = dataset.map(
            lambda images, labels: (augmenter(images, training=True), labels),
            num_parallel_calls=AUTOTUNE,
        )
    return dataset.prefetch(AUTOTUNE)


def augmentation_config() -> dict:
    """Serialisable record of the augmentation settings (for the report)."""
    return {
        "rotation_degrees": round(ROTATION_FACTOR * 360, 2),
        "translation_fraction": TRANSLATION_FACTOR,
        "zoom_fraction": ZOOM_FACTOR,
        "fill_mode": "constant (0 = background)",
        "excluded": [
            "horizontal flip (b<->d, p<->q)",
            "vertical flip (M<->W, b<->p)",
            "rotations >= 45 degrees (6<->9, N<->Z)",
            "shear beyond natural slant",
        ],
        "applied_to": "training split only",
    }
