"""CLI-level guarantees: the GPU guard and the CV early-stopping separation."""

import numpy as np
import pytest
import tensorflow as tf

from lxnet import train


@pytest.fixture
def no_gpu(monkeypatch):
    monkeypatch.setattr(tf.config, "list_physical_devices", lambda kind: [])


def test_refuses_to_train_on_cpu_by_default(no_gpu, tmp_path):
    with pytest.raises(SystemExit, match="refusing to train on CPU"):
        train.main(["--out-dir", str(tmp_path), "--data-root", str(tmp_path)])


def test_allow_cpu_opts_back_in(no_gpu, tmp_path):
    """Reaching dataset indexing proves the guard let it through."""
    with pytest.raises(ValueError, match="no class directories"):
        train.main(["--allow-cpu", "--out-dir", str(tmp_path), "--data-root", str(tmp_path)])


# --- the early-stopping monitor must never be the rows being reported ---

LABELS = np.repeat(np.arange(9), 40)
POOL = np.arange(len(LABELS))


def test_carve_is_disjoint_and_covers_every_row():
    train_rows, monitor = train._carve_validation(POOL, LABELS, seed=0)
    assert set(train_rows).isdisjoint(monitor)
    assert set(train_rows) | set(monitor) == set(POOL.tolist())


def test_carve_keeps_every_class_in_the_monitor():
    """A monitor missing a class makes val_loss blind to it."""
    _, monitor = train._carve_validation(POOL, LABELS, seed=0)
    assert set(LABELS[monitor].tolist()) == set(range(9))


def test_carve_never_empties_the_training_side():
    tiny = np.array([0, 1])
    train_rows, monitor = train._carve_validation(tiny, np.array([3, 3]), seed=0)
    assert len(train_rows) >= 1 and len(monitor) >= 1


def test_cross_validate_never_trains_on_the_scored_fold(monkeypatch):
    """The regression guard: fold rows must reach train_once only as the test set."""
    seen = []

    def spy(model_name, images, labels, train_idx, val_idx, test_idx, out_dir, **kw):
        seen.append((set(train_idx.tolist()), set(val_idx.tolist()), set(test_idx.tolist())))
        return {"accuracy": 1.0}, {}

    monkeypatch.setattr(train, "train_once", spy)
    held = POOL[:45]
    rest = POOL[45:]
    train.cross_validate("LXNet", None, LABELS, [(rest, held)], out_dir=None, seed=1)

    trained_on, monitored_on, scored_on = seen[0]
    assert scored_on == set(held.tolist())
    assert trained_on.isdisjoint(scored_on), "trained on the fold it reports"
    assert monitored_on.isdisjoint(scored_on), "early stopping peeked at the reported fold"
