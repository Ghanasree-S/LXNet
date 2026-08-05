"""LXNet architecture and the transfer-learning baselines it is compared against.

LXNet is three VGG-style conv blocks (32/64/128) followed by global average
pooling and a small classifier head. GAP rather than Flatten is what keeps the
model at ~0.35M parameters: flattening a 28x28x128 feature map into a 256-unit
dense layer would alone cost 25.7M weights, i.e. 70x the whole network.
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


def _conv_block(x, filters: int, dropout: float, block: int):
    for i in (1, 2):
        name = f"conv{block}_{i}"
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=name)(x)
        x = layers.BatchNormalization(name=f"bn{block}_{i}")(x)
        x = layers.Activation("relu", name=f"relu{block}_{i}")(x)
    x = layers.MaxPooling2D(2, name=f"pool{block}")(x)
    return layers.Dropout(dropout, name=f"drop{block}")(x)


def build_lxnet(
    num_classes: int = NUM_CLASSES,
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Build and compile LXNet."""
    inputs = layers.Input(shape=input_shape, name="input")

    x = _conv_block(inputs, 32, dropout=0.1, block=1)
    x = _conv_block(x, 64, dropout=0.1, block=2)

    # Final block is named so the CAM methods have a stable attachment point.
    # The name sits on the ReLU, not the Conv2D: CAM methods weight *feature
    # maps*, and the raw convolution output is pre-BatchNorm (so its per-channel
    # scale is arbitrary) and signed (so a channel's negative half survives into
    # the weighted sum). Post-activation is what Grad-CAM is defined over.
    x = layers.Conv2D(128, 3, padding="same", use_bias=False, name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.Activation("relu", name="relu3_1")(x)
    x = layers.Conv2D(128, 3, padding="same", use_bias=False, name="conv3_2")(x)
    x = layers.BatchNormalization(name="bn3_2")(x)
    x = layers.Activation("relu", name="final_conv")(x)
    x = layers.MaxPooling2D(2, name="pool3")(x)
    x = layers.Dropout(0.2, name="drop3")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)

    x = layers.Dense(256, use_bias=False, name="fc1")(x)
    x = layers.BatchNormalization(name="bn_fc1")(x)
    x = layers.Activation("relu", name="relu_fc1")(x)
    x = layers.Dropout(0.4, name="drop_fc1")(x)

    x = layers.Dense(128, use_bias=False, name="fc2")(x)
    x = layers.BatchNormalization(name="bn_fc2")(x)
    x = layers.Activation("relu", name="relu_fc2")(x)
    x = layers.Dropout(0.3, name="drop_fc2")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="LXNet")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_baseline(
    name: str,
    num_classes: int = NUM_CLASSES,
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    learning_rate: float = 1e-4,
    weights: str | None = "imagenet",
    trainable_backbone: bool = False,
) -> tf.keras.Model:
    """Build a transfer-learning baseline with a matching classifier head.

    The backbone is frozen by default: the comparison of interest is LXNet
    trained from scratch against the standard "pretrained backbone + new head"
    recipe, and a fair comparison needs an identical head on all four models.
    """
    if name not in BASELINES:
        raise ValueError(f"unknown baseline {name!r}; expected one of {sorted(BASELINES)}")

    backbone = BASELINES[name](include_top=False, weights=weights, input_shape=input_shape)
    backbone.trainable = trainable_backbone

    inputs = layers.Input(shape=input_shape, name="input")
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.4, name="drop_fc1")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name=name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model(name: str, **kwargs) -> tf.keras.Model:
    """Dispatch by model name; ``"LXNet"`` or any key of :data:`BASELINES`."""
    if name == "LXNet":
        return build_lxnet(**kwargs)
    return build_baseline(name, **kwargs)
