"""Tests for the LXNet architecture and the transfer-learning baselines.

LXNet's entire claim is "competitive accuracy at ~0.35M parameters, 50-70x
smaller than the standard backbones". Parameter count is therefore not an
incidental detail -- it is the result being replicated, so it is asserted here.
"""

import numpy as np
import pytest

from lxnet.models import BASELINES, NUM_CLASSES, build_baseline, build_lxnet

# Paper: "approximately 0.35 million parameters".
LXNET_PARAM_BUDGET = 500_000


@pytest.fixture(scope="module")
def lxnet():
    return build_lxnet()


class TestLXNetArchitecture:
    def test_accepts_224x224x3_input(self, lxnet):
        assert lxnet.input_shape == (None, 224, 224, 3)

    def test_outputs_one_probability_per_class(self, lxnet):
        assert lxnet.output_shape == (None, NUM_CLASSES)

    def test_stays_within_the_lightweight_parameter_budget(self, lxnet):
        assert lxnet.count_params() < LXNET_PARAM_BUDGET

    def test_is_approximately_the_paper_s_035m_parameters(self, lxnet):
        assert 0.30e6 < lxnet.count_params() < 0.40e6

    def test_predictions_are_a_probability_distribution(self, lxnet):
        out = lxnet.predict(np.zeros((2, 224, 224, 3), dtype=np.float32), verbose=0)
        assert out.shape == (2, NUM_CLASSES)
        np.testing.assert_allclose(out.sum(axis=1), 1.0, rtol=1e-5)
        assert (out >= 0).all()

    def test_uses_batch_normalisation(self, lxnet):
        kinds = {type(layer).__name__ for layer in lxnet.layers}
        assert "BatchNormalization" in kinds

    def test_uses_dropout_regularisation(self, lxnet):
        kinds = {type(layer).__name__ for layer in lxnet.layers}
        assert "Dropout" in kinds

    def test_has_a_named_final_conv_layer_for_cam_methods(self, lxnet):
        """Grad-CAM/Score-CAM need a stable handle on the last conv feature map."""
        layer = lxnet.get_layer("final_conv")
        assert len(layer.output_shape) == 4

    def test_is_compiled_and_trainable(self, lxnet):
        assert lxnet.optimizer is not None
        assert lxnet.loss is not None

    def test_can_overfit_a_tiny_batch(self):
        """A model that cannot memorise 8 images has a wiring bug."""
        model = build_lxnet(learning_rate=1e-3)
        x = np.random.default_rng(0).random((8, 224, 224, 3)).astype(np.float32)
        y = np.array([0, 1, 2, 3, 4, 5, 6, 7])

        history = model.fit(x, y, epochs=30, batch_size=8, verbose=0)

        assert history.history["loss"][-1] < history.history["loss"][0]


class TestBaselines:
    def test_every_baseline_is_registered(self):
        assert set(BASELINES) == {"DenseNet201", "ResNet50V2", "InceptionV3"}

    @pytest.mark.slow
    @pytest.mark.parametrize("name", ["DenseNet201", "ResNet50V2", "InceptionV3"])
    def test_baseline_outputs_match_the_class_count(self, name):
        model = build_baseline(name, weights=None)
        assert model.output_shape == (None, NUM_CLASSES)

    @pytest.mark.slow
    def test_baselines_are_far_larger_than_lxnet(self, lxnet):
        """The 50-70x size gap is the paper's headline comparison."""
        baseline = build_baseline("DenseNet201", weights=None)
        assert baseline.count_params() > 20 * lxnet.count_params()

    def test_rejects_an_unknown_baseline_name(self):
        with pytest.raises(ValueError, match="unknown"):
            build_baseline("NotARealNet", weights=None)
