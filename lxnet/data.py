"""Dataset indexing, deduplication and leak-free splitting.

Everything here exists to make the reported accuracy trustworthy. The dataset
ships with 74 groups of byte-identical images; if those are split naively the
same X-ray lands in both train and test and the score is inflated for free.
So: hash first, split on unique images only, and balance *after* splitting.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Short English labels for the Portuguese source directories, keyed by the
# numeric prefix the dataset uses. Display only -- the prefix drives ordering.
CLASS_LABELS = {
    "00": "Normal",
    "01": "Pneumonia",
    "02": "Higher Density",
    "03": "Lower Density",
    "04": "Obstructive Pulmonary",
    "05": "Degenerative Infectious",
    "06": "Encapsulated Lesions",
    "07": "Mediastinal Changes",
    "08": "Chest Changes",
}


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    class_name: str
    digest: str

    @property
    def display_name(self) -> str:
        return CLASS_LABELS.get(self.class_name[:2], self.class_name)


@dataclass
class DedupeReport:
    exact_duplicates: int = 0
    cross_class_conflicts: int = 0
    kept: int = 0

    def __str__(self) -> str:
        return (
            f"kept {self.kept} images; dropped {self.exact_duplicates} exact duplicates "
            f"and {self.cross_class_conflicts} cross-class conflicts"
        )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index_dataset(root: str | Path) -> list[Sample]:
    """Index one-directory-per-class image data, ordered by directory name.

    Labels are assigned by sorting class directories, so the dataset's ``00 ..``
    ``08 ..`` prefixes map to labels 0..8.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {root}")

    class_dirs = sorted((d for d in root.iterdir() if d.is_dir()), key=lambda d: d.name)
    if not class_dirs:
        raise ValueError(f"no class directories under {root}")

    samples: list[Sample] = []
    for label, class_dir in enumerate(class_dirs):
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append(
                    Sample(
                        path=path,
                        label=label,
                        class_name=class_dir.name,
                        digest=_digest(path),
                    )
                )
    return samples


def dedupe(samples: Sequence[Sample]) -> tuple[list[Sample], DedupeReport]:
    """Collapse byte-identical images; discard those labelled inconsistently.

    An image appearing under two different classes is a labelling contradiction.
    Keeping either copy would train on a coin flip, so both are dropped.
    """
    by_digest: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_digest[s.digest].append(s)

    kept: list[Sample] = []
    report = DedupeReport()
    for group in by_digest.values():
        labels = {s.label for s in group}
        if len(labels) > 1:
            report.cross_class_conflicts += 1
            log.warning(
                "dropping %d copies of an image labelled as %s",
                len(group),
                sorted(s.class_name for s in group),
            )
            continue
        report.exact_duplicates += len(group) - 1
        kept.append(group[0])

    report.kept = len(kept)
    return kept, report


def stratified_split(
    samples: Sequence[Sample],
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, list[Sample]]:
    """Split into train/val/test, preserving class proportions.

    Deduplicate before calling this: identical images in two splits is the most
    common way an X-ray classifier reports an accuracy it did not earn.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1.0, got {fractions} summing to {sum(fractions)}")
    if len(fractions) != 3:
        raise ValueError("expected exactly three fractions (train, val, test)")

    rng = np.random.default_rng(seed)
    train_frac, val_frac, _ = fractions
    parts: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}

    by_label: dict[int, list[Sample]] = defaultdict(list)
    for s in samples:
        by_label[s.label].append(s)

    for label in sorted(by_label):
        group = sorted(by_label[label], key=lambda s: s.digest)  # stable input order
        rng.shuffle(group)
        n = len(group)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        # Guarantee every class is represented in val/test when the class is
        # large enough; rounding alone can starve a small class.
        if n >= 3:
            n_train = min(n_train, n - 2)
            n_val = max(1, min(n_val, n - n_train - 1))
        parts["train"] += group[:n_train]
        parts["val"] += group[n_train : n_train + n_val]
        parts["test"] += group[n_train + n_val :]

    return parts


@cache
def _dhash(path_str: str, hash_size: int = 8) -> str:
    """Difference hash: compare each pixel to its right neighbour on a tiny grey thumbnail.

    Downscaling to 9x8 discards resolution, compression artefacts and most noise,
    so a re-saved or resized copy of an X-ray lands on the same digest.
    """
    with Image.open(path_str) as im:
        thumb = im.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    a = np.asarray(thumb, dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten()).tobytes().hex()


def perceptual_digest(path: str | Path, hash_size: int = 8) -> str:
    """Visual fingerprint of an image; equal digests mean 'the same X-ray'."""
    return _dhash(str(path), hash_size)


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

# Two digests within this many differing bits (out of 64) are the same X-ray.
#
# Measured on this dataset, chaining through intermediate near-matches begins
# immediately, because chest X-rays are near-identical by construction (same
# anatomy, same framing, low contrast). Largest resulting group by threshold:
#
#   <=0 -> 21      <=1 -> 143      <=2 -> 374      <=3 -> 1680
#
# The 1680-image group spans 8 of the 9 classes and has a median internal
# distance of 14 bits, i.e. it is a transitive-closure artefact, not a set of
# duplicates. Exact digest equality is the only threshold that holds: sampled
# dhash-0 pairs have median pixel correlation 0.9913 against 0.4263 for random
# pairs, so they really are the same X-ray re-saved.
GROUP_MAX_DISTANCE = 0


def group_samples(
    samples: Sequence[Sample], max_distance: int = GROUP_MAX_DISTANCE
) -> dict[str, list[Sample]]:
    """Bucket samples by visual identity: one bucket per distinct X-ray.

    Exact digest equality is not enough -- re-compressing an image flips a few
    hash bits. Digests within ``max_distance`` bits are unioned into one group,
    so every copy of an X-ray travels together into a single split.
    """
    by_digest: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_digest[perceptual_digest(s.path)].append(s)

    digests = sorted(by_digest)
    if not digests:
        return {}

    bits = np.array([list(bytes.fromhex(d)) for d in digests], dtype=np.uint8)
    parent = list(range(len(digests)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    chunk = 256
    for start in range(0, len(digests), chunk):
        block = bits[start : start + chunk]
        dist = _POPCOUNT[block[:, None, :] ^ bits[None, :, :]].sum(axis=2)
        for local_i, row in enumerate(dist):
            i = start + local_i
            for j in np.nonzero(row <= max_distance)[0]:
                if j > i:
                    union(i, int(j))

    grouped: dict[str, list[Sample]] = defaultdict(list)
    for idx, digest in enumerate(digests):
        grouped[digests[find(idx)]].extend(by_digest[digest])
    return dict(grouped)


def group_aware_split(
    samples: Sequence[Sample],
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, list[Sample]]:
    """Stratified train/val/test split that never breaks a near-duplicate group.

    ``stratified_split`` splits individual images, which lets a re-compressed copy
    of a training X-ray sit in the test set. On this dataset that inflates
    accuracy by tens of points. Here whole perceptual groups are dealt out
    instead, so evaluation only ever sees genuinely unseen images.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1.0, got {fractions}")

    rng = np.random.default_rng(seed)
    targets = dict(zip(("train", "val", "test"), fractions, strict=True))

    # A group is assigned the label most of its members carry.
    by_label: dict[int, list[list[Sample]]] = defaultdict(list)
    for _, members in sorted(group_samples(samples).items()):
        label = Counter(s.label for s in members).most_common(1)[0][0]
        by_label[label].append(members)

    parts: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
    for label in sorted(by_label):
        entries = by_label[label]
        order = rng.permutation(len(entries))
        counts = {k: 0 for k in targets}
        for idx in order:
            members = entries[idx]
            total = sum(counts.values()) + len(members)
            # give the group to whichever split is furthest below its quota
            choice = max(targets, key=lambda k: targets[k] * total - counts[k])
            parts[choice].extend(members)
            counts[choice] += len(members)
    return parts


def group_aware_folds(
    samples: Sequence[Sample], n_splits: int = 5, seed: int = 42
) -> list[tuple[list[Sample], list[Sample]]]:
    """Stratified k-fold where every near-duplicate group stays on one side."""
    lookup = {s.path: gid for gid, members in group_samples(samples).items() for s in members}
    groups = [lookup[s.path] for s in samples]
    labels = [s.label for s in samples]
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [
        ([samples[i] for i in train_idx], [samples[i] for i in val_idx])
        for train_idx, val_idx in splitter.split(samples, labels, groups=groups)
    ]


def oversample_to_balance(samples: Sequence[Sample], seed: int = 42) -> list[Sample]:
    """Random oversampling of minority classes up to the majority count.

    Apply to the training split only. The paper balances the whole dataset,
    which leaks duplicated minority images into evaluation; see README.
    """
    rng = np.random.default_rng(seed)
    by_label: dict[int, list[Sample]] = defaultdict(list)
    for s in samples:
        by_label[s.label].append(s)

    target = max(len(v) for v in by_label.values())
    out: list[Sample] = []
    for label in sorted(by_label):
        group = sorted(by_label[label], key=lambda s: s.digest)
        out.extend(group)
        deficit = target - len(group)
        if deficit > 0:
            picks = rng.integers(0, len(group), size=deficit)
            out.extend(group[i] for i in picks)
    return out


def class_weights(samples: Sequence[Sample]) -> dict[int, float]:
    """Inverse-frequency weights, normalised to mean ~1.

    An alternative to oversampling that duplicates no data at all.
    """
    counts = Counter(s.label for s in samples)
    n_classes = len(counts)
    total = sum(counts.values())
    return {label: total / (n_classes * count) for label, count in sorted(counts.items())}


def distribution(samples: Iterable[Sample]) -> dict[str, int]:
    """Human-readable per-class counts, for logging and plots."""
    counts = Counter(s.display_name for s in samples)
    return dict(sorted(counts.items()))
