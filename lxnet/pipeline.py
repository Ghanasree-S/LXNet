"""Cached tf.data input pipeline.

CLAHE runs on CPU via OpenCV and is far too slow to repeat every epoch, so the
whole dataset is decoded, CLAHE'd and resized exactly once into a uint8 array.
Stored as grayscale that is 338 MB for 6.7k images at 224x224 -- the same data
as float32 RGB would be 4 GB. Channels are replicated on device instead.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import tensorflow as tf

from .data import Sample
from .preprocess import IMAGE_SIZE, apply_clahe, apply_clahe_grid_search

log = logging.getLogger(__name__)

AUTOTUNE = tf.data.AUTOTUNE


def build_cache(
    samples: Sequence[Sample],
    size: tuple[int, int] = IMAGE_SIZE,
    clahe: bool = True,
    cache_path: str | Path | None = None,
    clahe_grid_search: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode, CLAHE and resize every sample once into a uint8 array.

    Returns ``(images, labels)`` where images is ``(N, H, W)`` uint8 grayscale.
    If ``cache_path`` exists it is loaded instead of recomputed.
    """
    import cv2  # local import keeps module import cheap for tests that skip cv2

    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            with np.load(cache_path) as data:
                log.info("loaded preprocessed cache from %s", cache_path)
                return data["images"], data["labels"]

    images = np.empty((len(samples), *size), dtype=np.uint8)
    labels = np.empty(len(samples), dtype=np.int32)

    for i, sample in enumerate(samples):
        # cv2.imread cannot handle non-ASCII paths on Windows, and the class
        # directories are Portuguese; read the bytes ourselves and decode.
        buffer = np.fromfile(sample.path, dtype=np.uint8)
        gray = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"could not decode image: {sample.path}")
        if clahe:
            gray = apply_clahe_grid_search(gray) if clahe_grid_search else apply_clahe(gray)
        images[i] = cv2.resize(gray, (size[1], size[0]), interpolation=cv2.INTER_AREA)
        labels[i] = sample.label

        if (i + 1) % 1000 == 0:
            log.info("preprocessed %d/%d images", i + 1, len(samples))

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, images=images, labels=labels)
        log.info("wrote preprocessed cache to %s", cache_path)

    return images, labels


def _augment(x: tf.Tensor, seed: int) -> tf.Tensor:
    """Mild geometric/photometric jitter.

    Deliberately conservative: no vertical flip and no large rotation, because
    a mirrored or upside-down chest X-ray is not a plausible clinical image and
    situs inversus is itself a finding.
    """
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_brightness(x, max_delta=0.10)
    x = tf.image.random_contrast(x, lower=0.90, upper=1.10)
    return tf.clip_by_value(x, 0.0, 1.0)


def make_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 32,
    training: bool = False,
    seed: int = 42,
    augment: bool = True,
) -> tf.data.Dataset:
    """Build a tf.data pipeline over a preprocessed cache.

    Evaluation datasets are never shuffled or augmented: predictions are matched
    to labels by position downstream.
    """
    ds = tf.data.Dataset.from_tensor_slices((images, labels))

    if training:
        ds = ds.shuffle(min(len(images), 4096), seed=seed, reshuffle_each_iteration=True)

    def _prepare(img, label):
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.expand_dims(img, -1)
        img = tf.repeat(img, 3, axis=-1)
        return img, label

    ds = ds.map(_prepare, num_parallel_calls=AUTOTUNE)

    if training and augment:
        ds = ds.map(lambda x, y: (_augment(x, seed), y), num_parallel_calls=AUTOTUNE)

    return ds.batch(batch_size).prefetch(AUTOTUNE)
