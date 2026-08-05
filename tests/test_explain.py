"""Contract for the interpretability panel driver."""

import numpy as np
import pytest
import tensorflow as tf

from lxnet import explain
from lxnet.data import Sample


def _samples():
    return [
        Sample(path=f"img_{i}.png", label=i % 3, class_name=f"0{i % 3} c", digest=f"d{i}")
        for i in range(12)
    ]


def test_pick_examples_returns_one_per_requested_class():
    rng = np.random.default_rng(0)
    chosen = explain.pick_examples(_samples(), [0, 1, 2], rng)
    assert [s.label for s in chosen] == [0, 1, 2]


def test_pick_examples_skips_classes_with_no_samples():
    rng = np.random.default_rng(0)
    chosen = explain.pick_examples(_samples(), [0, 7], rng)
    assert [s.label for s in chosen] == [0]


def test_pick_examples_is_deterministic_for_a_seed():
    a = explain.pick_examples(_samples(), [0, 1, 2], np.random.default_rng(5))
    b = explain.pick_examples(_samples(), [0, 1, 2], np.random.default_rng(5))
    assert [s.digest for s in a] == [s.digest for s in b]


@pytest.mark.slow
def test_build_panel_writes_a_figure(tmp_path):
    """Drives the real CAM code paths against a tiny model on grayscale cache rows."""
    size = 32
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((size, size, 3)),
            tf.keras.layers.Conv2D(4, 3, padding="same", activation="relu", name="final_conv"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )
    rng = np.random.default_rng(0)
    images = rng.integers(0, 255, (6, size, size), dtype=np.uint8)  # (N,H,W) like the cache
    out = explain.build_panel(
        model,
        images,
        rows=[(0, 0), (1, 1)],
        class_names=["a", "b", "c"],
        out=tmp_path / "panel.png",
        lime_samples=20,
    )
    assert out.exists() and out.stat().st_size > 5000
