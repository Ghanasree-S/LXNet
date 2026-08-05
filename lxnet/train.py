"""Training: single hold-out runs and stratified k-fold cross-validation.

Ordering is the whole ballgame. Deduplicate, then split, then balance only the
training fold. The paper balances before splitting, which places augmented
copies of training images into the evaluation set; see README for the effect.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

from .data import (
    DedupeReport,
    build_paper_balanced_dataset,
    dedupe,
    distribution,
    group_aware_folds,
    group_aware_split,
    index_dataset,
    stratified_split,
)
from .evaluate import compute_metrics, summarise_folds, wilcoxon_compare
from .models import build_model
from .pipeline import build_cache, make_dataset
from .preprocess import IMAGE_SIZE

log = logging.getLogger(__name__)

DEFAULT_MODELS = ["LXNet", "DenseNet201", "ResNet50V2", "InceptionV3"]


def _callbacks(checkpoint: Path, patience: int = 8):
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(2, patience // 3), min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint), monitor="val_loss", save_best_only=True, save_weights_only=True
        ),
    ]


def _index_of(samples, digest_to_row):
    return np.array([digest_to_row[s.digest] for s in samples], dtype=np.int64)


def load_and_prepare(
    data_root: str | Path, cache_dir: Path, seed: int = 42, paper_mode: bool = False
):
    """Index, deduplicate and cache the dataset. Returns samples plus the cache.

    ``paper_mode`` reproduces the paper's protocol exactly: no dedupe (its
    protocol never removes near-duplicates), balancing to 1,000 images/class
    (500 original + 500 augmented) *before* any split, and per-image CLAHE
    grid search. This is the leaky pipeline the rest of the module exists to
    avoid; it is opt-in only.
    """
    samples = index_dataset(data_root)
    log.info("indexed %d images", len(samples))

    if paper_mode:
        report = DedupeReport(exact_duplicates=0, cross_class_conflicts=0, kept=len(samples))
        samples = build_paper_balanced_dataset(
            samples, cache_dir=cache_dir / "paper_augmented", target_per_class=1000, seed=seed
        )
        log.info("paper-mode balancing: %s", distribution(samples))
        images, labels = build_cache(
            samples,
            size=IMAGE_SIZE,
            cache_path=cache_dir / "preprocessed.npz",
            clahe_grid_search=True,
        )
    else:
        samples, report = dedupe(samples)
        log.info("dedupe: %s", report)
        images, labels = build_cache(
            samples, size=IMAGE_SIZE, cache_path=cache_dir / "preprocessed.npz"
        )

    digest_to_row = {s.digest: i for i, s in enumerate(samples)}
    return samples, images, labels, digest_to_row, report


def train_once(
    model_name: str,
    images: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    out_dir: Path,
    epochs: int = 40,
    batch_size: int = 32,
    tag: str = "holdout",
    weights: dict[int, float] | None = None,
) -> dict:
    """Train one model on one split and evaluate it on the test indices."""
    tf.keras.backend.clear_session()
    model = build_model(model_name)

    train_ds = make_dataset(images[train_idx], labels[train_idx], batch_size, training=True)
    val_ds = make_dataset(images[val_idx], labels[val_idx], batch_size, training=False)
    test_ds = make_dataset(images[test_idx], labels[test_idx], batch_size, training=False)

    checkpoint = out_dir / "checkpoints" / f"{model_name}_{tag}.weights.h5"
    started = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=_callbacks(checkpoint),
        class_weight=weights,
        verbose=2,
    )
    elapsed = time.time() - started

    probabilities = model.predict(test_ds, verbose=0)
    predictions = probabilities.argmax(axis=1)

    metrics = compute_metrics(labels[test_idx], predictions, num_classes=9)
    metrics.update(
        model=model_name,
        tag=tag,
        params=int(model.count_params()),
        epochs_run=len(history.history["loss"]),
        train_seconds=round(elapsed, 1),
    )
    log.info(
        "%s [%s] accuracy=%.4f f1_macro=%.4f (%.0fs)",
        model_name,
        tag,
        metrics["accuracy"],
        metrics["f1_macro"],
        elapsed,
    )
    return metrics, history.history


def build_folds(
    samples: list,
    pool_idx: np.ndarray,
    labels: np.ndarray,
    digest_to_row: dict[str, int],
    folds: int = 5,
    seed: int = 42,
    split_mode: str = "grouped",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fold index pairs, either plain stratified or near-duplicate aware.

    ``grouped`` keeps every perceptual copy of an X-ray inside one fold half.
    ``random`` is the naive scheme the paper uses; it is retained only so the
    two can be compared honestly in the report.
    """
    if split_mode == "grouped":
        pool_samples = [samples[i] for i in pool_idx]
        pairs = group_aware_folds(pool_samples, n_splits=folds, seed=seed)
        return [
            (
                np.array([digest_to_row[s.digest] for s in tr], dtype=np.int64),
                np.array([digest_to_row[s.digest] for s in va], dtype=np.int64),
            )
            for tr, va in pairs
        ]

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return [(pool_idx[tr], pool_idx[va]) for tr, va in splitter.split(pool_idx, labels[pool_idx])]


def cross_validate(
    model_name: str,
    images: np.ndarray,
    labels: np.ndarray,
    fold_pairs: list[tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
    epochs: int = 40,
    batch_size: int = 32,
    seed: int = 42,
    balance: bool = True,
) -> list[dict]:
    """Run k-fold CV over pre-computed fold index pairs.

    Balancing happens inside each fold, after the validation split is carved
    out, so a duplicated minority image can never appear on both sides.
    """
    results = []

    for fold, (train_idx, val_idx) in enumerate(fold_pairs, start=1):
        # Early stopping needs a monitor set, and it must not be the fold being
        # scored: restore_best_weights picks the best of ~40 epochs, so scoring
        # on the monitor reports the maximum over 40 draws rather than the
        # model's accuracy. Carve the monitor out of the training rows instead.
        inner_train, inner_val = _carve_validation(train_idx, labels, seed=seed + fold)
        if balance:
            # Carve first, then balance, so an oversampled copy cannot straddle
            # the two -- the same ordering the hold-out arm uses.
            inner_train = _oversample_indices(inner_train, labels, seed=seed + fold)

        metrics, _ = train_once(
            model_name,
            images,
            labels,
            inner_train,
            inner_val,  # early stopping monitor
            val_idx,  # the held-out fold, scored once
            out_dir,
            epochs=epochs,
            batch_size=batch_size,
            tag=f"fold{fold}",
        )
        metrics["fold"] = fold
        results.append(metrics)

    return results


def _carve_validation(
    idx: np.ndarray, labels: np.ndarray, seed: int, fraction: float = 0.15
) -> tuple[np.ndarray, np.ndarray]:
    """Split fold-training rows into (train, early-stopping monitor), stratified.

    ponytail: carves by row, not by perceptual group, so a near-duplicate may
    straddle the inner boundary. That only blunts the stopping-epoch choice --
    the reported score comes from the outer fold, which stays group-clean. Carve
    by group if the stopping epoch ever looks systematically late.
    """
    rng = np.random.default_rng(seed)
    by_label: dict[int, list[int]] = {}
    for i in idx:
        by_label.setdefault(int(labels[i]), []).append(int(i))

    monitor: list[int] = []
    for label in sorted(by_label):
        rows = np.array(by_label[label], dtype=np.int64)
        rng.shuffle(rows)
        n = min(len(rows) - 1, max(1, int(round(len(rows) * fraction))))
        monitor.extend(rows[:n].tolist())

    held = set(monitor)
    train = np.array(sorted(int(i) for i in idx if int(i) not in held), dtype=np.int64)
    return train, np.array(sorted(monitor), dtype=np.int64)


def _oversample_indices(idx: np.ndarray, labels: np.ndarray, seed: int) -> np.ndarray:
    """Random oversampling of minority classes, operating on cache row indices."""
    rng = np.random.default_rng(seed)
    by_label: dict[int, list[int]] = {}
    for i in idx:
        by_label.setdefault(int(labels[i]), []).append(int(i))

    target = max(len(v) for v in by_label.values())
    out: list[int] = []
    for label in sorted(by_label):
        rows = by_label[label]
        out.extend(rows)
        deficit = target - len(rows)
        if deficit > 0:
            out.extend(int(rows[j]) for j in rng.integers(0, len(rows), size=deficit))
    return np.array(sorted(out), dtype=np.int64)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate LXNet and baselines.")
    parser.add_argument("--data-root", default="dataset", type=Path)
    parser.add_argument("--out-dir", default="runs/latest", type=Path)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-cv", action="store_true", help="hold-out evaluation only")
    parser.add_argument(
        "--allow-cpu", action="store_true", help="train without a GPU (slow; off by default)"
    )
    parser.add_argument("--limit", type=int, help="use only N images (smoke test)")
    parser.add_argument(
        "--split-mode",
        choices=["grouped", "random"],
        default="grouped",
        help="grouped keeps near-duplicate images in one split (honest); "
        "random reproduces the paper's leaky protocol for comparison",
    )
    parser.add_argument(
        "--cv-models",
        nargs="+",
        help="models to cross-validate (default: all --models). "
        "The heavy backbones are usually hold-out only for time reasons.",
    )
    parser.add_argument(
        "--paper-mode",
        action="store_true",
        help="reproduce the paper's exact protocol: no dedupe, balance to "
        "1000 images/class (500 original + 500 augmented) before an 80/10/10 "
        "random split, per-image CLAHE grid search. Implies --split-mode random "
        "and disables post-split oversampling (the data is already balanced).",
    )
    args = parser.parse_args(argv)
    if args.paper_mode:
        args.split_mode = "random"

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    tf.keras.utils.set_random_seed(args.seed)

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    log.info("GPUs visible: %s", [g.name for g in gpus] or "none")
    if not gpus and not args.allow_cpu:
        # TF 2.10 falls back to CPU silently when the conda env's Library/bin is
        # off PATH, which turns a 40-minute run into an overnight one that looks
        # fine in the log. Refuse rather than discover it tomorrow.
        raise SystemExit(
            "no GPU visible to TensorFlow -- refusing to train on CPU.\n"
            "Activate the env (`conda activate lxnet`) so CUDA is on PATH, or pass --allow-cpu."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples, images, labels, digest_to_row, report = load_and_prepare(
        args.data_root, args.out_dir, seed=args.seed, paper_mode=args.paper_mode
    )

    if args.limit:
        keep = np.linspace(0, len(samples) - 1, args.limit).astype(int)
        samples = [samples[i] for i in keep]
        images, labels = images[keep], labels[keep]
        digest_to_row = {s.digest: i for i, s in enumerate(samples)}

    log.info("class distribution: %s", distribution(samples))

    split_fn = group_aware_split if args.split_mode == "grouped" else stratified_split
    fractions = (0.80, 0.10, 0.10) if args.paper_mode else (0.70, 0.15, 0.15)
    log.info("split mode: %s (paper_mode=%s)", args.split_mode, args.paper_mode)
    parts = split_fn(samples, fractions, seed=args.seed)
    train_idx = _index_of(parts["train"], digest_to_row)
    val_idx = _index_of(parts["val"], digest_to_row)
    test_idx = _index_of(parts["test"], digest_to_row)
    log.info("split: %d train / %d val / %d test", len(train_idx), len(val_idx), len(test_idx))

    if args.paper_mode:
        # Already balanced to 1000/class before the split -- oversampling again
        # would duplicate rows on top of the paper's own balancing step.
        balanced_train = train_idx
    else:
        balanced_train = _oversample_indices(train_idx, labels, seed=args.seed)
    log.info("training set balanced: %d -> %d rows", len(train_idx), len(balanced_train))

    summary: dict = {
        "split_mode": args.split_mode,
        "dedupe": {
            "kept": report.kept,
            "exact_duplicates": report.exact_duplicates,
            "cross_class_conflicts": report.cross_class_conflicts,
        },
        "split": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "holdout": {},
        "cross_validation": {},
    }

    for name in args.models:
        metrics, history = train_once(
            name,
            images,
            labels,
            balanced_train,
            val_idx,
            test_idx,
            args.out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            tag="holdout",
            weights=None,
        )
        summary["holdout"][name] = metrics
        _write(args.out_dir / f"history_{name}.json", history)
        _write(args.out_dir / "results.json", summary)

    if not args.skip_cv:
        pool_idx = np.concatenate([train_idx, val_idx])
        fold_pairs = build_folds(
            samples,
            pool_idx,
            labels,
            digest_to_row,
            folds=args.folds,
            seed=args.seed,
            split_mode=args.split_mode,
        )
        for name in args.cv_models or args.models:
            folds = cross_validate(
                name,
                images,
                labels,
                fold_pairs,
                args.out_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=args.seed,
                balance=not args.paper_mode,
            )
            summary["cross_validation"][name] = {
                "folds": folds,
                "summary": summarise_folds(folds),
            }
            _write(args.out_dir / "results.json", summary)

        cv = summary["cross_validation"]
        if "LXNet" in cv:
            lx = [f["accuracy"] for f in cv["LXNet"]["folds"]]
            summary["wilcoxon"] = {
                other: wilcoxon_compare(lx, [f["accuracy"] for f in cv[other]["folds"]])
                for other in cv
                if other != "LXNet"
            }

    _write(args.out_dir / "results.json", summary)
    log.info("wrote %s", args.out_dir / "results.json")
    return 0


def _jsonable(obj):
    """Coerce numpy scalars/arrays that Keras histories and metrics are full of."""
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_jsonable))


if __name__ == "__main__":
    raise SystemExit(main())
