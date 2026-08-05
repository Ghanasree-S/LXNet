"""Contract for near-duplicate detection and group-aware splitting.

SHA-256 only catches byte-identical files. This dataset is 31% near-duplicates:
the same X-ray re-saved, resized or re-compressed hashes differently but is
visually the same image. Split those naively and the model is tested on
pictures it memorised in training, which is how a chest X-ray classifier
reports 98% and means none of it.

These tests pin the fix: perceptual grouping, then split whole groups.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lxnet import data


def _write(path: Path, arr: np.ndarray, **kw) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, **kw)
    return path


@pytest.fixture
def base_image():
    """Structure in BOTH axes: a difference hash compares horizontal neighbours,
    so an image constant along rows would hash pure noise and prove nothing."""
    rng = np.random.default_rng(0)
    x = np.linspace(0, 255, 64)
    X, Y = np.meshgrid(x, x)
    # the sin((X+Y)) term is non-separable, so rows differ from one another and
    # a flipped copy is genuinely a different picture to a difference hash
    img = np.sin(X / 11) * 50 + np.cos(Y / 17) * 40 + np.sin((X + Y) / 7) * 35 + 128
    return (img + rng.normal(0, 2, (64, 64))).clip(0, 255).astype(np.uint8)


@pytest.fixture
def other_image():
    """A structurally unrelated X-ray, not a transform of ``base_image``."""
    rng = np.random.default_rng(99)
    x = np.linspace(0, 255, 64)
    X, Y = np.meshgrid(x, x)
    img = np.cos(X / 4) * 70 + np.sin(Y / 6) * 55 + np.cos((X - Y) / 9) * 30 + 110
    return (img + rng.normal(0, 2, (64, 64))).clip(0, 255).astype(np.uint8)


def test_perceptual_digest_is_stable_for_identical_content(tmp_path, base_image):
    a = _write(tmp_path / "a.png", base_image)
    b = _write(tmp_path / "b.png", base_image)
    assert data.perceptual_digest(a) == data.perceptual_digest(b)


def test_perceptual_digest_survives_recompression_and_resize(tmp_path, base_image):
    a = _write(tmp_path / "a.png", base_image)
    _write(tmp_path / "b.jpg", base_image, quality=60)  # re-compressed
    big = np.asarray(Image.fromarray(base_image).resize((128, 128), Image.LANCZOS))
    _write(tmp_path / "c.png", big)  # resized
    d = data.perceptual_digest
    assert d(a) == d(tmp_path / "b.jpg"), "JPEG recompression must not change the group"
    assert d(a) == d(tmp_path / "c.png"), "resizing must not change the group"


def test_perceptual_digest_differs_for_different_images(tmp_path, base_image, other_image):
    a = _write(tmp_path / "a.png", base_image)
    b = _write(tmp_path / "b.png", other_image)
    assert data.perceptual_digest(a) != data.perceptual_digest(b)


def test_group_samples_collapses_near_duplicates(tmp_path, base_image, other_image):
    d = tmp_path / "00 Anatomia Normal"
    _write(d / "a.png", base_image)
    _write(d / "b.jpg", base_image, quality=50)
    _write(d / "c.png", other_image)
    samples = data.index_dataset(tmp_path)
    groups = data.group_samples(samples)
    assert len(groups) == 2, "the two encodings of one X-ray form a single group"
    assert sorted(len(g) for g in groups.values()) == [1, 2]


def test_group_aware_split_keeps_near_duplicates_together(tmp_path, base_image):
    """The core guarantee: no perceptual group may span two splits."""
    rng = np.random.default_rng(1)
    for ci, folder in enumerate(sorted(data.CLASS_LABELS)):
        d = tmp_path / f"{folder} Class"
        for i in range(12):
            img = (
                (base_image.astype(int) + rng.integers(-90, 90) + ci * 7)
                .clip(0, 255)
                .astype(np.uint8)
            )
            _write(d / f"img_{i}.png", img)
            if i % 3 == 0:  # every third image gets a re-compressed twin
                _write(d / f"img_{i}_copy.jpg", img, quality=55)

    samples, _ = data.dedupe(data.index_dataset(tmp_path))
    parts = data.group_aware_split(samples, seed=42)

    seen: dict[str, str] = {}
    for split_name, split in parts.items():
        for s in split:
            g = data.perceptual_digest(s.path)
            if g in seen:
                assert seen[g] == split_name, (
                    f"perceptual group {g[:8]} spans {seen[g]} and {split_name} -- leakage"
                )
            seen[g] = split_name


def test_group_aware_split_is_disjoint_and_total(tmp_path, base_image):
    rng = np.random.default_rng(2)
    for ci, folder in enumerate(sorted(data.CLASS_LABELS)):
        d = tmp_path / f"{folder} Class"
        for i in range(10):
            img = (
                (base_image.astype(int) + rng.integers(-90, 90) + ci * 5)
                .clip(0, 255)
                .astype(np.uint8)
            )
            _write(d / f"img_{i}.png", img)
    samples, _ = data.dedupe(data.index_dataset(tmp_path))
    parts = data.group_aware_split(samples, seed=3)
    paths = {k: {s.path for s in v} for k, v in parts.items()}
    assert paths["train"].isdisjoint(paths["val"])
    assert paths["train"].isdisjoint(paths["test"])
    assert paths["val"].isdisjoint(paths["test"])
    assert sum(len(v) for v in parts.values()) == len(samples)


def test_group_aware_split_is_deterministic(tmp_path, base_image):
    rng = np.random.default_rng(4)
    for ci, folder in enumerate(sorted(data.CLASS_LABELS)):
        d = tmp_path / f"{folder} Class"
        for i in range(10):
            img = (
                (base_image.astype(int) + rng.integers(-90, 90) + ci * 5)
                .clip(0, 255)
                .astype(np.uint8)
            )
            _write(d / f"img_{i}.png", img)
    samples, _ = data.dedupe(data.index_dataset(tmp_path))
    a = data.group_aware_split(samples, seed=11)["test"]
    b = data.group_aware_split(samples, seed=11)["test"]
    assert [s.path for s in a] == [s.path for s in b]


def test_group_aware_folds_never_split_a_group(tmp_path, base_image):
    """Cross-validation must respect groups too, or every fold leaks."""
    rng = np.random.default_rng(5)
    for ci, folder in enumerate(sorted(data.CLASS_LABELS)):
        d = tmp_path / f"{folder} Class"
        for i in range(12):
            img = (
                (base_image.astype(int) + rng.integers(-90, 90) + ci * 6)
                .clip(0, 255)
                .astype(np.uint8)
            )
            _write(d / f"img_{i}.png", img)
            if i % 4 == 0:
                _write(d / f"img_{i}_twin.jpg", img, quality=55)

    samples, _ = data.dedupe(data.index_dataset(tmp_path))
    folds = data.group_aware_folds(samples, n_splits=5, seed=42)
    assert len(folds) == 5
    for train, val in folds:
        gtrain = {data.perceptual_digest(s.path) for s in train}
        gval = {data.perceptual_digest(s.path) for s in val}
        assert gtrain.isdisjoint(gval), "a perceptual group appears in both fold halves"
        assert len(train) + len(val) == len(samples)


def test_build_folds_grouped_is_leak_free_at_index_level(tmp_path, base_image):
    """End-to-end wiring: train.build_folds must carry the guarantee to row indices."""
    import numpy as np

    from lxnet.train import build_folds

    rng = np.random.default_rng(7)
    for ci, folder in enumerate(sorted(data.CLASS_LABELS)):
        d = tmp_path / f"{folder} Class"
        for i in range(12):
            img = (
                (base_image.astype(int) + rng.integers(-90, 90) + ci * 6)
                .clip(0, 255)
                .astype(np.uint8)
            )
            _write(d / f"img_{i}.png", img)
            if i % 4 == 0:
                _write(d / f"img_{i}_twin.jpg", img, quality=55)

    samples, _ = data.dedupe(data.index_dataset(tmp_path))
    digest_to_row = {s.digest: i for i, s in enumerate(samples)}
    labels = np.array([s.label for s in samples])
    pool = np.arange(len(samples))

    group_of = {s.path: g for g, members in data.group_samples(samples).items() for s in members}

    for mode, must_be_clean in [("grouped", True), ("random", False)]:
        pairs = build_folds(samples, pool, labels, digest_to_row, folds=5, seed=42, split_mode=mode)
        assert len(pairs) == 5
        leaked = 0
        for tr, va in pairs:
            assert set(tr).isdisjoint(set(va))
            gtr = {group_of[samples[i].path] for i in tr}
            gva = {group_of[samples[i].path] for i in va}
            leaked += len(gtr & gva)
        if must_be_clean:
            assert leaked == 0, "grouped folds leaked a near-duplicate group"
