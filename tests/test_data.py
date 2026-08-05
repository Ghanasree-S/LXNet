"""Tests for dataset indexing, deduplication and leak-free splitting.

The whole point of the replication is that the reported accuracy is earned, so
these tests are mostly about the ways a chest-X-ray pipeline silently cheats:
duplicate images spanning splits, augmented copies leaking into validation, and
class balancing applied before the split instead of after.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lxnet.data import (
    Sample,
    class_weights,
    dedupe,
    index_dataset,
    oversample_to_balance,
    stratified_split,
)


def _write_image(path: Path, value: int, size=(32, 32)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full(size, value, dtype=np.uint8), mode="L").save(path)


@pytest.fixture
def toy_root(tmp_path):
    """Two classes; class_b has a byte-identical duplicate of one of its images."""
    root = tmp_path / "dataset"
    for i in range(4):
        _write_image(root / "00 Class A" / f"a{i}.jpeg", value=10 + i)
    for i in range(3):
        _write_image(root / "01 Class B" / f"b{i}.jpeg", value=100 + i)
    # exact duplicate of b0 within the same class
    (root / "01 Class B" / "b0_copy.jpeg").write_bytes(
        (root / "01 Class B" / "b0.jpeg").read_bytes()
    )
    return root


class TestIndexDataset:
    def test_finds_every_image_and_assigns_contiguous_labels(self, toy_root):
        samples = index_dataset(toy_root)
        assert len(samples) == 8
        assert {s.label for s in samples} == {0, 1}

    def test_label_order_follows_numeric_directory_prefix(self, toy_root):
        samples = index_dataset(toy_root)
        by_label = {s.label: s.class_name for s in samples}
        assert by_label[0].startswith("00")
        assert by_label[1].startswith("01")

    def test_is_case_insensitive_about_extensions(self, tmp_path):
        root = tmp_path / "ds"
        _write_image(root / "00 A" / "x.jpeg", 5)
        _write_image(root / "00 A" / "y.JPG", 6)
        _write_image(root / "00 A" / "z.png", 7)
        assert len(index_dataset(root)) == 3

    def test_computes_a_content_digest_per_sample(self, toy_root):
        samples = index_dataset(toy_root)
        dupes = [s for s in samples if s.path.name in ("b0.jpeg", "b0_copy.jpeg")]
        assert dupes[0].digest == dupes[1].digest


class TestDedupe:
    def test_removes_byte_identical_copies(self, toy_root):
        kept, report = dedupe(index_dataset(toy_root))
        assert len(kept) == 7
        assert report.exact_duplicates == 1
        assert len({s.digest for s in kept}) == 7

    def test_drops_all_copies_when_duplicate_spans_two_classes(self, tmp_path):
        """Same pixels filed under two labels is contradictory: trust neither."""
        root = tmp_path / "ds"
        _write_image(root / "00 A" / "a.jpeg", 42)
        _write_image(root / "01 B" / "b.jpeg", 99)
        (root / "01 B" / "conflict.jpeg").write_bytes((root / "00 A" / "a.jpeg").read_bytes())

        kept, report = dedupe(index_dataset(root))

        assert report.cross_class_conflicts == 1
        assert {s.path.name for s in kept} == {"b.jpeg"}

    def test_is_a_no_op_on_already_unique_data(self, tmp_path):
        root = tmp_path / "ds"
        for i in range(5):
            _write_image(root / "00 A" / f"a{i}.jpeg", i)
        kept, report = dedupe(index_dataset(root))
        assert len(kept) == 5
        assert report.exact_duplicates == 0


def _samples(counts):
    """Build synthetic Samples: {label: n}."""
    out = []
    for label, n in counts.items():
        for i in range(n):
            out.append(
                Sample(
                    path=Path(f"/fake/c{label}/img{i}.jpg"),
                    label=label,
                    class_name=f"class{label}",
                    digest=f"{label}-{i}",
                )
            )
    return out


class TestStratifiedSplit:
    def test_partitions_every_sample_exactly_once(self):
        samples = _samples({0: 50, 1: 30, 2: 20})
        parts = stratified_split(samples, (0.7, 0.15, 0.15), seed=0)
        total = sum(len(v) for v in parts.values())
        assert total == 100
        digests = [s.digest for v in parts.values() for s in v]
        assert len(set(digests)) == 100

    def test_no_digest_appears_in_two_splits(self):
        samples = _samples({0: 40, 1: 40})
        parts = stratified_split(samples, (0.6, 0.2, 0.2), seed=1)
        train = {s.digest for s in parts["train"]}
        val = {s.digest for s in parts["val"]}
        test = {s.digest for s in parts["test"]}
        assert train & val == set()
        assert train & test == set()
        assert val & test == set()

    def test_preserves_class_proportions(self):
        samples = _samples({0: 100, 1: 50})
        parts = stratified_split(samples, (0.8, 0.1, 0.1), seed=2)
        for part in parts.values():
            labels = [s.label for s in part]
            ratio = labels.count(0) / max(labels.count(1), 1)
            assert 1.5 < ratio < 2.5, "2:1 class ratio should survive the split"

    def test_every_class_present_in_every_split(self):
        samples = _samples({0: 30, 1: 30, 2: 30})
        parts = stratified_split(samples, (0.7, 0.15, 0.15), seed=3)
        for name, part in parts.items():
            assert {s.label for s in part} == {0, 1, 2}, f"{name} is missing a class"

    def test_is_deterministic_for_a_fixed_seed(self):
        samples = _samples({0: 30, 1: 20})
        a = stratified_split(samples, (0.7, 0.15, 0.15), seed=7)
        b = stratified_split(samples, (0.7, 0.15, 0.15), seed=7)
        assert [s.digest for s in a["train"]] == [s.digest for s in b["train"]]

    def test_different_seeds_give_different_partitions(self):
        samples = _samples({0: 50, 1: 50})
        a = stratified_split(samples, (0.7, 0.15, 0.15), seed=1)
        b = stratified_split(samples, (0.7, 0.15, 0.15), seed=2)
        assert [s.digest for s in a["train"]] != [s.digest for s in b["train"]]

    def test_rejects_fractions_that_do_not_sum_to_one(self):
        with pytest.raises(ValueError):
            stratified_split(_samples({0: 10}), (0.7, 0.7, 0.7), seed=0)


class TestOversampleToBalance:
    def test_equalises_class_counts(self):
        balanced = oversample_to_balance(_samples({0: 100, 1: 25, 2: 50}), seed=0)
        counts = {}
        for s in balanced:
            counts[s.label] = counts.get(s.label, 0) + 1
        assert set(counts.values()) == {100}

    def test_introduces_no_new_images(self):
        """Balancing may repeat existing samples but must never invent a digest."""
        original = _samples({0: 20, 1: 5})
        balanced = oversample_to_balance(original, seed=0)
        assert {s.digest for s in balanced} <= {s.digest for s in original}

    def test_keeps_every_original_sample(self):
        original = _samples({0: 20, 1: 5})
        balanced = oversample_to_balance(original, seed=0)
        assert {s.digest for s in original} <= {s.digest for s in balanced}

    def test_is_deterministic_for_a_fixed_seed(self):
        original = _samples({0: 30, 1: 7})
        a = oversample_to_balance(original, seed=5)
        b = oversample_to_balance(original, seed=5)
        assert [s.digest for s in a] == [s.digest for s in b]


class TestClassWeights:
    def test_rare_classes_weigh_more_than_common_ones(self):
        w = class_weights(_samples({0: 100, 1: 25}))
        assert w[1] > w[0]

    def test_weights_average_to_about_one(self):
        w = class_weights(_samples({0: 100, 1: 50, 2: 50}))
        assert 0.9 < float(np.mean(list(w.values()))) < 1.6
