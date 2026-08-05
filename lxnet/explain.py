"""Generate the interpretability panel from a trained checkpoint.

One row per class: the CLAHE'd X-ray, then Grad-CAM, Score-CAM and LIME for the
class the model actually predicted. Correct and incorrect predictions are both
shown and labelled -- a panel of successes only would say nothing about whether
the model looks at lung tissue when it gets things wrong.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import CLASS_LABELS, dedupe, group_aware_split, index_dataset, stratified_split
from .models import build_model
from .report import INK, INK_MUTED, SURFACE
from .xai import grad_cam, lime_explanation, overlay_heatmap, score_cam

log = logging.getLogger(__name__)


def pick_examples(samples, labels_wanted, rng) -> list:
    """One sample per class, chosen deterministically."""
    chosen = []
    for label in labels_wanted:
        pool = [s for s in samples if s.label == label]
        if pool:
            chosen.append(pool[int(rng.integers(len(pool)))])
    return chosen


def build_panel(
    model,
    images: np.ndarray,
    rows: list[tuple[int, int]],
    class_names: list[str],
    out: Path,
    lime_samples: int = 400,
) -> Path:
    """Render the four-column panel. ``rows`` is a list of (cache_row, true_label)."""
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(11.5, 2.9 * n), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_2d(axes)

    for r, (row_idx, true_label) in enumerate(rows):
        # The cache holds (H, W) uint8 grayscale; the model takes (H, W, 3) in [0,1],
        # matching pipeline.make_dataset exactly.
        gray = images[row_idx]
        image = np.repeat((gray.astype(np.float32) / 255.0)[..., None], 3, axis=-1)
        probs = model.predict(image[None], verbose=0)[0]
        predicted = int(probs.argmax())

        display = np.repeat(gray[..., None], 3, axis=-1).astype(np.uint8)

        gc = overlay_heatmap(display, grad_cam(model, image, predicted))
        sc = overlay_heatmap(display, score_cam(model, image, predicted))
        try:
            lime_img, _ = lime_explanation(model, image, predicted, num_samples=lime_samples)
            lime_img = (lime_img * 255).astype(np.uint8) if lime_img.max() <= 1.5 else lime_img
        except Exception as exc:  # LIME is the one optional dependency
            log.warning("LIME failed on row %d: %s", r, exc)
            lime_img = display

        ok = "correct" if predicted == true_label else f"predicted {class_names[predicted]}"
        panels = [
            (display, f"{class_names[true_label]}\n{ok} · p={probs[predicted]:.2f}"),
            (gc, "Grad-CAM"),
            (sc, "Score-CAM"),
            (lime_img, "LIME"),
        ]
        for c, (img, title) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(
                title,
                fontsize=8.5,
                color=INK if c == 0 else INK_MUTED,
                loc="left",
                pad=4,
            )

    fig.suptitle(
        "LXNet interpretability — group-aware test set",
        fontsize=12,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Write the XAI panel for a trained model.")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/grouped"))
    parser.add_argument("--model", default="LXNet")
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--out", type=Path, default=Path("docs/fig_xai.png"))
    parser.add_argument("--split-mode", choices=["grouped", "random"], default="grouped")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lime-samples", type=int, default=400)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cache = np.load(args.run_dir / "preprocessed.npz")
    images = cache["images"]

    samples, _ = dedupe(index_dataset(args.data_root))
    row_of = {s.digest: i for i, s in enumerate(samples)}
    split_fn = group_aware_split if args.split_mode == "grouped" else stratified_split
    test = split_fn(samples, (0.70, 0.15, 0.15), seed=args.seed)["test"]

    rng = np.random.default_rng(args.seed)
    chosen = pick_examples(test, sorted({s.label for s in test}), rng)
    rows = [(row_of[s.digest], s.label) for s in chosen]

    weights = args.run_dir / "checkpoints" / f"{args.model}_holdout.weights.h5"
    if not weights.exists():
        raise SystemExit(f"no checkpoint at {weights}; train first")
    model = build_model(args.model)
    model.load_weights(weights)
    log.info(
        "loaded %s (%d params); rendering %d rows",
        args.model,
        model.count_params(),
        len(rows),
    )

    out = build_panel(
        model, images, rows, list(CLASS_LABELS.values()), args.out, lime_samples=args.lime_samples
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
