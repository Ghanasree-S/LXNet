"""LXNet architecture and the transfer-learning baselines it is compared against.

LXNet: 7x7 stem (32 filters) then three conv blocks (48/72/128 filters, Swish
activations), the last with no pooling or dropout to preserve fine spatial
detail, followed by global average pooling and a softmax head -- the paper's
exact architecture. GAP is what keeps the model small: flattening the final
feature map into a dense layer would cost tens of millions of weights on its
own, dozens of times the whole network.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models

NUM_CLASSES = 9
INPUT_SHAPE = (224, 224, 3)

BASELINES = {
    "DenseNet201": tf.keras.applications.DenseNet201,
    "ResNet50V2": tf.keras.applications.ResNet50V2,
    "InceptionV3": tf.keras.applications.InceptionV3,
}


def _conv_block(x, filters: int, block: int, pool: bool, dropout: float | None):
    """Paper Eqs. 6-10: two 3x3 convs (BN + Swish each), then optional pool/dropout.

    Block 3 passes ``pool=False, dropout=None`` -- the paper's "no-pooling final
    block" that keeps spatial detail for subtle pathologies.
    """
    for i in (1, 2):
        name = f"conv{block}_{i}"
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=name)(x)
        x = layers.BatchNormalization(name=f"bn{block}_{i}")(x)
        # Named on the last block's second activation so CAM methods have a
        # stable, post-activation attachment point (pre-BN is signed and its
        # per-channel scale is arbitrary -- not what Grad-CAM is defined over).
        act_name = "final_conv" if (block == 3 and i == 2) else f"swish{block}_{i}"
        x = layers.Activation("swish", name=act_name)(x)
    if pool:
        x = layers.MaxPooling2D(2, name=f"pool{block}")(x)
    if dropout is not None:
        x = layers.SpatialDropout2D(dropout, name=f"spatial_drop{block}")(x)
    return x


def build_lxnet(
    num_classes: int = NUM_CLASSES,
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    learning_rate: float = 3e-4,
) -> tf.keras.Model:
    """Build and compile LXNet, matching the paper's architecture exactly.

    7x7 stem (32 filters) -> Block1 (48, pool+SpatialDropout2D keep-prob 0.7)
    -> Block2 (72, pool+SpatialDropout2D) -> Block3 (128, no pool/dropout) ->
    GlobalAveragePooling2D -> Dropout(0.3) -> Softmax. Optimizer Nadam,
    lr 0.0003, per the paper's ablation-selected configuration (Table 8).
    """
    inputs = layers.Input(shape=input_shape, name="input")

    x = layers.Conv2D(32, 7, padding="same", use_bias=False, name="stem_conv")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("swish", name="stem_swish")(x)

    x = _conv_block(x, 48, block=1, pool=True, dropout=0.3)
    x = _conv_block(x, 72, block=2, pool=True, dropout=0.3)
    x = _conv_block(x, 128, block=3, pool=False, dropout=None)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, name="drop_head")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="LXNet")
    model.compile(
        optimizer=tf.keras.optimizers.Nadam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_baseline(
    name: str,
    num_classes: int = NUM_CLASSES,
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    learning_rate: float = 3e-4,
    weights: str | None = "imagenet",
    trainable_backbone: bool = False,
) -> tf.keras.Model:
    """Build a transfer-learning baseline with a matching classifier head.

    The backbone is frozen by default: the comparison of interest is LXNet
    trained from scratch against the standard "pretrained backbone + new head"
    recipe. The paper benchmarks all baselines "under identical settings", so
    optimizer, learning rate and head configuration mirror LXNet's.
    """
    if name not in BASELINES:
        raise ValueError(f"unknown baseline {name!r}; expected one of {sorted(BASELINES)}")

    backbone = BASELINES[name](include_top=False, weights=weights, input_shape=input_shape)
    backbone.trainable = trainable_backbone

    inputs = layers.Input(shape=input_shape, name="input")
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, name="drop_head")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name=name)
    model.compile(
        optimizer=tf.keras.optimizers.Nadam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model(name: str, **kwargs) -> tf.keras.Model:
    """Dispatch by model name; ``"LXNet"`` or any key of :data:`BASELINES`."""
    if name == "LXNet":
        return build_lxnet(**kwargs)
    return build_baseline(name, **kwargs)
