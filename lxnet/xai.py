"""Interpretability: Grad-CAM, Score-CAM and LIME.

All three answer "which pixels drove this prediction", by different means:
Grad-CAM weights conv feature maps by their gradients, Score-CAM weights them
by how much each map alone raises the class score (no gradients, slower), and
LIME fits a sparse linear model to perturbed superpixels.
"""

from __future__ import annotations

import cv2
import numpy as np
import tensorflow as tf

DEFAULT_LAYER = "final_conv"


def _feature_layer(model: tf.keras.Model, layer_name: str | None) -> str:
    name = layer_name or DEFAULT_LAYER
    if name not in {layer.name for layer in model.layers}:
        # The default was previously returned unchecked, so a model without the
        # attachment point failed later inside get_layer with no hint of why.
        raise ValueError(f"layer {name!r} not found in model {model.name!r}")
    return name


def _normalise(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.maximum(heatmap, 0)
    peak = heatmap.max()
    if peak <= 0:
        return np.zeros_like(heatmap)
    return heatmap / peak


def _resize(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(heatmap, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)


def grad_cam(
    model: tf.keras.Model,
    image: np.ndarray,
    class_index: int | None = None,
    layer_name: str | None = None,
) -> np.ndarray:
    """Gradient-weighted class activation map, normalised to [0, 1]."""
    layer_name = _feature_layer(model, layer_name)
    grad_model = tf.keras.Model(model.inputs, [model.get_layer(layer_name).output, model.output])

    batch = tf.convert_to_tensor(image[None], dtype=tf.float32)
    with tf.GradientTape() as tape:
        features, predictions = grad_model(batch)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        score = predictions[:, class_index]

    gradients = tape.gradient(score, features)
    # Global-average-pool the gradients into one weight per feature map.
    weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(features[0] * weights, axis=-1).numpy()

    return _normalise(_resize(heatmap, image.shape[:2]))


def score_cam(
    model: tf.keras.Model,
    image: np.ndarray,
    class_index: int | None = None,
    layer_name: str | None = None,
    max_channels: int = 32,
    batch_size: int = 32,
) -> np.ndarray:
    """Gradient-free CAM: weight each feature map by the score it alone produces.

    The full method evaluates every channel, which is hundreds of forward passes.
    ``max_channels`` keeps the highest-activation channels only -- the rest
    contribute almost nothing to the final map and dominate the runtime.
    """
    layer_name = _feature_layer(model, layer_name)
    feature_model = tf.keras.Model(model.inputs, model.get_layer(layer_name).output)

    batch = tf.convert_to_tensor(image[None], dtype=tf.float32)
    features = feature_model(batch)[0].numpy()

    if class_index is None:
        class_index = int(model.predict(batch, verbose=0).argmax())

    # Rank channels by peak activation and keep the strongest few.
    strength = features.max(axis=(0, 1))
    keep = np.argsort(strength)[::-1][:max_channels]

    masks = []
    for channel in keep:
        activation = _normalise(_resize(features[..., channel], image.shape[:2]))
        masks.append(image * activation[..., None])

    scores = []
    masks_array = np.asarray(masks, dtype=np.float32)
    for start in range(0, len(masks_array), batch_size):
        chunk = masks_array[start : start + batch_size]
        scores.append(model.predict(chunk, verbose=0)[:, class_index])
    weights = np.concatenate(scores)

    # Softmax the weights so a single channel cannot dominate the map.
    weights = np.exp(weights - weights.max())
    weights /= weights.sum()

    heatmap = np.zeros(features.shape[:2], dtype=np.float32)
    for weight, channel in zip(weights, keep, strict=True):
        heatmap += weight * features[..., channel]

    return _normalise(_resize(heatmap, image.shape[:2]))


def lime_explanation(
    model: tf.keras.Model,
    image: np.ndarray,
    class_index: int | None = None,
    num_samples: int = 1000,
    num_features: int = 8,
):
    """LIME superpixel explanation. Returns ``(image_with_boundaries, mask)``.

    Imported lazily: ``lime`` and ``scikit-image`` are only needed for this one
    figure, so the rest of the pipeline does not depend on them.
    """
    from lime import lime_image
    from skimage.segmentation import mark_boundaries

    explainer = lime_image.LimeImageExplainer()

    def predict(batch: np.ndarray) -> np.ndarray:
        return model.predict(batch.astype(np.float32), verbose=0)

    explanation = explainer.explain_instance(
        image.astype(np.double),
        predict,
        top_labels=5,
        hide_color=0,
        num_samples=num_samples,
    )
    label = class_index if class_index is not None else explanation.top_labels[0]
    surface, mask = explanation.get_image_and_mask(
        label, positive_only=True, num_features=num_features, hide_rest=False
    )
    return mark_boundaries(surface, mask), mask


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Blend a heatmap over the X-ray for display. Returns uint8 RGB."""
    if heatmap.shape != image.shape[:2]:
        heatmap = _resize(heatmap, image.shape[:2])

    coloured = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)
    coloured = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)

    base = np.uint8(255 * np.clip(image, 0, 1))
    if base.ndim == 2:
        base = np.repeat(base[..., None], 3, axis=-1)

    return np.uint8(np.clip(coloured * alpha + base * (1 - alpha), 0, 255))
