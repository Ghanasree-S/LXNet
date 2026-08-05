"""Tests for the cached tf.data input pipeline.

The cache stores CLAHE'd grayscale uint8 (338 MB for the full dataset) instead
of float32 RGB (4 GB), and channels are replicated inside the graph. These
tests guard the two things that break silently: image/label misalignment, and
augmentation bleeding into evaluation data.
"""

import numpy as np
import pytest
from PIL import Image

from lxnet.data import Sample
from lxnet.pipeline import build_cache, make_dataset


@pytest.fixture
def samples(tmp_path):
    """12 samples across 3 classes, each a solid shade keyed to its index."""
    out = []
    for i in range(12):
        p = tmp_path / f"img{i}.png"
        Image.fromarray(np.full((40, 40), 10 + i * 10, dtype=np.uint8), mode="L").save(p)
        out.append(Sample(path=p, label=i % 3, class_name=f"c{i % 3}", digest=f"d{i}"))
    return out


class TestBuildCache:
    def test_returns_one_grayscale_entry_per_sample(self, samples):
        images, labels = build_cache(samples, size=(32, 32))
        assert images.shape == (12, 32, 32)
        assert labels.shape == (12,)

    def test_cache_is_uint8_to_keep_memory_down(self, samples):
        images, _ = build_cache(samples, size=(32, 32))
        assert images.dtype == np.uint8

    def test_labels_stay_aligned_with_their_images(self, samples):
        images, labels = build_cache(samples, size=(32, 32))
        assert list(labels) == [s.label for s in samples]
        # brightness increases with index, so image order must be preserved too
        means = images.reshape(12, -1).mean(axis=1)
        assert np.all(np.diff(means) > 0)

    def test_persists_and_reloads_from_disk(self, samples, tmp_path):
        cache = tmp_path / "cache.npz"
        build_cache(samples, size=(32, 32), cache_path=cache)
        assert cache.exists()

        images, labels = build_cache(samples, size=(32, 32), cache_path=cache)

        assert images.shape == (12, 32, 32)
        assert list(labels) == [s.label for s in samples]


class TestMakeDataset:
    def test_yields_batches_of_the_requested_size(self, samples):
        images, labels = build_cache(samples, size=(32, 32))
        ds = make_dataset(images, labels, batch_size=4, training=False)

        x, y = next(iter(ds))

        assert x.shape == (4, 32, 32, 3)
        assert y.shape == (4,)

    def test_expands_grayscale_to_three_identical_channels(self, samples):
        images, labels = build_cache(samples, size=(32, 32))
        x, _ = next(iter(make_dataset(images, labels, batch_size=2, training=False)))
        x = x.numpy()
        assert np.allclose(x[..., 0], x[..., 1])
        assert np.allclose(x[..., 1], x[..., 2])

    def test_scales_pixels_to_unit_range(self, samples):
        images, labels = build_cache(samples, size=(32, 32))
        x, _ = next(iter(make_dataset(images, labels, batch_size=4, training=False)))
        assert float(np.min(x)) >= 0.0 and float(np.max(x)) <= 1.0

    def test_evaluation_order_is_deterministic(self, samples):
        """Predictions are matched to labels by position, so eval must not shuffle."""
        images, labels = build_cache(samples, size=(32, 32))

        first = np.concatenate(
            [y.numpy() for _, y in make_dataset(images, labels, 4, training=False)]
        )
        second = np.concatenate(
            [y.numpy() for _, y in make_dataset(images, labels, 4, training=False)]
        )

        assert list(first) == list(second) == [s.label for s in samples]

    def test_training_data_is_shuffled(self, samples):
        images, labels = build_cache(samples, size=(32, 32))
        ds = make_dataset(images, labels, batch_size=12, training=True, seed=0)
        orders = {tuple(np.concatenate([y.numpy() for _, y in ds]).tolist()) for _ in range(4)}
        assert len(orders) > 1

    def test_augmentation_is_applied_only_when_training(self, samples):
        """Same batch twice with training=False must be pixel-identical."""
        images, labels = build_cache(samples, size=(32, 32))
        ds = make_dataset(images, labels, batch_size=4, training=False)

        a, _ = next(iter(ds))
        b, _ = next(iter(ds))

        assert np.allclose(a.numpy(), b.numpy())
