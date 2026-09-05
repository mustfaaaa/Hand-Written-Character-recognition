"""CNN architectures for handwritten character recognition.

One flexible builder (:func:`build_cnn`) covers every experiment so that the
comparisons in the report differ by exactly the component under test
(batch normalisation, dropout, depth, head type) and nothing else.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, regularizers

from preprocessing import INPUT_SHAPE


def _conv_block(
    x: tf.Tensor,
    filters: int,
    n_convs: int,
    use_bn: bool,
    l2: float,
    dropout: float,
    block_id: int,
) -> tf.Tensor:
    """Conv -> [BN] -> ReLU (xN) -> MaxPool -> [Dropout].

    BatchNorm sits between the convolution and its activation, which is why
    the convolution itself carries no bias (BN's beta term replaces it).
    """
    regulariser = regularizers.l2(l2) if l2 else None
    for i in range(n_convs):
        x = layers.Conv2D(
            filters,
            kernel_size=3,
            padding="same",
            use_bias=not use_bn,
            kernel_regularizer=regulariser,
            name=f"block{block_id}_conv{i + 1}",
        )(x)
        if use_bn:
            x = layers.BatchNormalization(name=f"block{block_id}_bn{i + 1}")(x)
        x = layers.Activation("relu", name=f"block{block_id}_relu{i + 1}")(x)
    x = layers.MaxPooling2D(pool_size=2, name=f"block{block_id}_pool")(x)
    if dropout:
        x = layers.Dropout(dropout, name=f"block{block_id}_drop")(x)
    return x


def build_cnn(
    n_classes: int,
    filters: tuple[int, ...] = (32, 64),
    convs_per_block: int = 2,
    use_bn: bool = True,
    block_dropout: tuple[float, ...] | float = 0.25,
    dense_units: int = 128,
    head_dropout: float = 0.5,
    head: str = "flatten",
    l2: float = 0.0,
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    name: str = "cnn",
) -> tf.keras.Model:
    """Build a configurable VGG-style CNN.

    Parameters
    ----------
    filters:
        One entry per convolutional block.
    convs_per_block:
        Convolutions stacked before each pooling layer.
    use_bn:
        Insert BatchNormalization after every convolution.
    block_dropout:
        Spatial dropout rate after each pooling layer (scalar or per-block).
    head:
        ``"flatten"`` or ``"gap"`` (global average pooling).
    l2:
        L2 weight-decay coefficient on convolution/dense kernels; 0 disables.
    """
    if isinstance(block_dropout, (int, float)):
        block_dropout = tuple([float(block_dropout)] * len(filters))
    if len(block_dropout) != len(filters):
        raise ValueError("block_dropout must be a scalar or match len(filters)")

    regulariser = regularizers.l2(l2) if l2 else None
    inputs = layers.Input(shape=input_shape, name="image")

    x = inputs
    for block_id, (f, drop) in enumerate(zip(filters, block_dropout), start=1):
        x = _conv_block(x, f, convs_per_block, use_bn, l2, drop, block_id)

    if head == "gap":
        x = layers.GlobalAveragePooling2D(name="gap")(x)
    elif head == "flatten":
        x = layers.Flatten(name="flatten")(x)
    else:
        raise ValueError("head must be 'flatten' or 'gap'")

    if dense_units:
        x = layers.Dense(
            dense_units, use_bias=not use_bn, kernel_regularizer=regulariser, name="fc1"
        )(x)
        if use_bn:
            x = layers.BatchNormalization(name="fc1_bn")(x)
        x = layers.Activation("relu", name="fc1_relu")(x)
        if head_dropout:
            x = layers.Dropout(head_dropout, name="fc1_drop")(x)

    outputs = layers.Dense(
        n_classes, activation="softmax", kernel_regularizer=regulariser, name="predictions"
    )(x)
    return tf.keras.Model(inputs, outputs, name=name)


# --------------------------------------------------------------------------
# Named experiment variants
# --------------------------------------------------------------------------
def build_variant(variant: str, n_classes: int) -> tf.keras.Model:
    """Return the model for a named experiment.

    ``simple``  - plain CNN, no BN, no dropout: the CNN reference point.
    ``bn``      - the same network plus batch normalisation.
    ``bn_drop`` - BN plus dropout (this is also the architecture trained
                  with augmentation in experiment 5).
    ``deep``    - three blocks, wider, GAP head, small L2: the candidate
                  final architecture.
    """
    if variant == "simple":
        return build_cnn(
            n_classes,
            filters=(32, 64),
            convs_per_block=2,
            use_bn=False,
            block_dropout=0.0,
            head_dropout=0.0,
            name="cnn_simple",
        )
    if variant == "bn":
        return build_cnn(
            n_classes,
            filters=(32, 64),
            convs_per_block=2,
            use_bn=True,
            block_dropout=0.0,
            head_dropout=0.0,
            name="cnn_bn",
        )
    if variant == "bn_drop":
        return build_cnn(
            n_classes,
            filters=(32, 64),
            convs_per_block=2,
            use_bn=True,
            block_dropout=0.25,
            head_dropout=0.5,
            name="cnn_bn_drop",
        )
    if variant == "deep":
        return build_cnn(
            n_classes,
            filters=(32, 64, 128),
            convs_per_block=2,
            use_bn=True,
            block_dropout=(0.20, 0.25, 0.30),
            dense_units=256,
            head_dropout=0.5,
            head="flatten",
            l2=1e-4,
            name="cnn_deep",
        )
    raise ValueError(f"Unknown variant {variant!r}")


def count_parameters(model: tf.keras.Model) -> dict:
    """Trainable / non-trainable parameter counts."""
    trainable = int(sum(int(tf.size(w)) for w in model.trainable_weights))
    non_trainable = int(sum(int(tf.size(w)) for w in model.non_trainable_weights))
    return {
        "trainable": trainable,
        "non_trainable": non_trainable,
        "total": trainable + non_trainable,
    }
