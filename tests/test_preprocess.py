"""Tests for CLAHE preprocessing and image loading.

CLAHE is the paper's headline preprocessing step. These tests pin down the
properties that matter: it must actually increase local contrast, must not
change geometry, and must produce network-ready tensors regardless of whether
the source file was grayscale, RGB or RGBA.
"""

import numpy as np
import pytest
from PIL import Image

from lxnet.preprocess import IMAGE_SIZE, apply_clahe, load_image


@pytest.fixture
def low_contrast():
    """A smooth gradient squeezed into a narrow intensity band."""
    grad = np.linspace(110, 145, 64 * 64).reshape(64, 64)
    return grad.astype(np.uint8)


class TestApplyClahe:
    def test_preserves_shape_and_dtype(self, low_contrast):
        out = apply_clahe(low_contrast)
        assert out.shape == low_contrast.shape
        assert out.dtype == np.uint8

    def test_increases_contrast(self, low_contrast):
        out = apply_clahe(low_contrast)
        assert out.std() > low_contrast.std()

    def test_output_spans_a_wider_intensity_range(self, low_contrast):
        out = apply_clahe(low_contrast)
        assert out.ptp() >= low_contrast.ptp()

    def test_stays_within_uint8_bounds(self, low_contrast):
        out = apply_clahe(low_contrast)
        assert out.min() >= 0 and out.max() <= 255

    def test_is_deterministic(self, low_contrast):
        assert np.array_equal(apply_clahe(low_contrast), apply_clahe(low_contrast))

    def test_rejects_colour_input(self):
        """CLAHE here is defined on single-channel X-rays; colour must be explicit."""
        with pytest.raises(ValueError):
            apply_clahe(np.zeros((32, 32, 3), dtype=np.uint8))

    def test_handles_a_flat_image_without_blowing_up(self):
        flat = np.full((32, 32), 128, dtype=np.uint8)
        out = apply_clahe(flat)
        assert out.shape == flat.shape
        assert np.isfinite(out).all()


class TestLoadImage:
    @pytest.mark.parametrize(
        "mode,value", [("L", 120), ("RGB", (120, 120, 120)), ("RGBA", (120, 120, 120, 255))]
    )
    def test_normalises_any_source_mode_to_three_channels(self, tmp_path, mode, value):
        p = tmp_path / f"{mode}.png"
        Image.new(mode, (80, 60), value).save(p)

        out = load_image(p)

        assert out.shape == (*IMAGE_SIZE, 3)

    def test_resizes_to_the_configured_input_size(self, tmp_path):
        p = tmp_path / "odd.png"
        Image.new("L", (137, 411), 90).save(p)
        assert load_image(p).shape[:2] == IMAGE_SIZE

    def test_returns_float32_scaled_to_unit_range(self, tmp_path):
        p = tmp_path / "x.png"
        Image.new("L", (64, 64), 200).save(p)

        out = load_image(p)

        assert out.dtype == np.float32
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_channels_are_replicated_not_garbage(self, tmp_path):
        """A grayscale X-ray widened to 3 channels must have identical channels."""
        p = tmp_path / "g.png"
        Image.new("L", (64, 64), 77).save(p)

        out = load_image(p)

        assert np.allclose(out[..., 0], out[..., 1])
        assert np.allclose(out[..., 1], out[..., 2])

    def test_clahe_can_be_disabled(self, tmp_path):
        p = tmp_path / "g.png"
        grad = np.linspace(110, 145, 64 * 64).reshape(64, 64).astype(np.uint8)
        Image.fromarray(grad, mode="L").save(p)

        with_clahe = load_image(p, clahe=True)
        without = load_image(p, clahe=False)

        assert not np.allclose(with_clahe, without)
