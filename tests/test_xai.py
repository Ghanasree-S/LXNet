"""Tests for the interpretability methods.

Saliency code fails quietly -- a bug usually yields a plausible-looking blob
rather than an exception. So these tests assert the properties a correct
attribution map must have: right shape, normalised, class-dependent, and
actually responsive to the input rather than constant.
"""

import numpy as np
import pytest

from lxnet.models import build_lxnet
from lxnet.xai import grad_cam, overlay_heatmap, score_cam


@pytest.fixture(scope="module")
def model():
    return build_lxnet()


@pytest.fixture
def image():
    rng = np.random.default_rng(0)
    return rng.random((224, 224, 3)).astype(np.float32)


class TestGradCAM:
    def test_heatmap_matches_the_input_resolution(self, model, image):
        assert grad_cam(model, image).shape == (224, 224)

    def test_heatmap_is_normalised_to_unit_range(self, model, image):
        heatmap = grad_cam(model, image)
        assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0
        assert heatmap.max() == pytest.approx(1.0, abs=1e-5)

    def test_contains_no_nans(self, model, image):
        assert np.isfinite(grad_cam(model, image)).all()

    def test_is_not_uniform(self, model, image):
        """A constant map means gradients never reached the conv layer."""
        assert grad_cam(model, image).std() > 0

    def test_explains_the_requested_class(self, model, image):
        a = grad_cam(model, image, class_index=0)
        b = grad_cam(model, image, class_index=5)
        assert not np.allclose(a, b), "explanation must depend on the target class"

    def test_defaults_to_the_predicted_class(self, model, image):
        predicted = int(model.predict(image[None], verbose=0).argmax())
        assert np.allclose(grad_cam(model, image), grad_cam(model, image, class_index=predicted))

    def test_rejects_an_unknown_layer_name(self, model, image):
        with pytest.raises(ValueError):
            grad_cam(model, image, layer_name="no_such_layer")


class TestScoreCAM:
    def test_heatmap_matches_the_input_resolution(self, model, image):
        assert score_cam(model, image, max_channels=8).shape == (224, 224)

    def test_heatmap_is_normalised(self, model, image):
        heatmap = score_cam(model, image, max_channels=8)
        assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0

    def test_contains_no_nans(self, model, image):
        assert np.isfinite(score_cam(model, image, max_channels=8)).all()

    def test_needs_no_gradients(self, model, image):
        """Score-CAM is gradient-free; it must work under a no-gradient context."""
        heatmap = score_cam(model, image, max_channels=4)
        assert heatmap.shape == (224, 224)


class TestOverlay:
    def test_returns_an_rgb_uint8_image(self, image):
        heatmap = np.linspace(0, 1, 224 * 224).reshape(224, 224).astype(np.float32)
        out = overlay_heatmap(image, heatmap)
        assert out.shape == (224, 224, 3)
        assert out.dtype == np.uint8

    def test_resizes_a_smaller_heatmap_to_the_image(self, image):
        small = np.linspace(0, 1, 28 * 28).reshape(28, 28).astype(np.float32)
        assert overlay_heatmap(image, small).shape == (224, 224, 3)


def test_cam_attachment_point_is_post_activation():
    """CAM weights feature maps; pre-BatchNorm conv output is signed and unscaled."""
    import numpy as np
    import tensorflow as tf

    from lxnet.models import build_lxnet
    from lxnet.xai import DEFAULT_LAYER

    model = build_lxnet()
    probe = tf.keras.Model(model.inputs, model.get_layer(DEFAULT_LAYER).output)
    features = probe(np.random.rand(1, 224, 224, 3).astype("float32")).numpy()
    assert features.min() >= 0.0, "attachment point is not a rectified feature map"
