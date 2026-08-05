"""CLAHE preprocessing and image loading.

The paper's preprocessing is CLAHE (Contrast Limited Adaptive Histogram
Equalisation) on the grayscale X-ray, then resize to the network input and
replicate to three channels so ImageNet-pretrained baselines can be compared
against LXNet on identical pixels.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMAGE_SIZE = (224, 224)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid: tuple[int, int] = CLAHE_TILE_GRID,
) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation on a grayscale image.

    Args:
        image: 2-D uint8 array.
        clip_limit: contrast ceiling; higher amplifies noise in flat regions.
        tile_grid: number of tiles the image is equalised over, (rows, cols).
    """
    if image.ndim != 2:
        raise ValueError(f"apply_clahe expects a 2-D grayscale image, got shape {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(image)


def load_image(
    path: str | Path,
    size: tuple[int, int] = IMAGE_SIZE,
    clahe: bool = True,
) -> np.ndarray:
    """Load one X-ray as a float32 ``(H, W, 3)`` array scaled to [0, 1].

    Source images are a mix of L/RGB/RGBA at 40 different resolutions, so
    everything is flattened to single-channel first: CLAHE is only meaningful
    on one channel, and applying it per-RGB-channel would shift colour balance.
    """
    with Image.open(path) as im:
        gray = np.asarray(im.convert("L"), dtype=np.uint8)

    if clahe:
        gray = apply_clahe(gray)

    # cv2.resize takes (width, height); size is (height, width).
    resized = cv2.resize(gray, (size[1], size[0]), interpolation=cv2.INTER_AREA)

    scaled = resized.astype(np.float32) / 255.0
    return np.repeat(scaled[..., None], 3, axis=-1)


def preprocessing_demo(path: str | Path, size: tuple[int, int] = IMAGE_SIZE):
    """Return (original, clahe) uint8 pair for the before/after figure."""
    with Image.open(path) as im:
        gray = np.asarray(im.convert("L"), dtype=np.uint8)
    resized = cv2.resize(gray, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    return resized, apply_clahe(resized)
