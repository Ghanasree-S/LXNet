"""Inference: turn a chest X-ray file into a ranked class distribution.

This is the trust boundary. Everything upstream of here is our own cached,
audited tensor; everything arriving here is a file someone handed us. So the
input is validated before it reaches the model, and the failure modes are
explicit exceptions rather than a confident prediction over garbage.

Preprocessing deliberately calls the same ``preprocess.load_image`` the training
cache is built from. Inference preprocessing drifting from training preprocessing
is the classic silent production bug: nothing errors, the numbers just quietly
get worse.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import CLASS_LABELS
from .models import build_model
from .preprocess import IMAGE_SIZE, load_image

log = logging.getLogger(__name__)

# Below this, an image carries no diagnostic content at 224x224 and is almost
# certainly a thumbnail, an icon, or a decode that went wrong.
MIN_DIMENSION = 32

DEFAULT_CHECKPOINT = Path("runs/grouped/checkpoints/LXNet_holdout.weights.h5")


class InvalidImageError(ValueError):
    """The input is not a usable image. Distinct from 'the model is unsure'."""


@dataclass(frozen=True)
class Prediction:
    label: int
    class_name: str
    probability: float

    def __str__(self) -> str:
        return f"{self.class_name:<26} {self.probability:.3f}"


def validate_image_path(path: str | Path) -> Path:
    """Check a caller-supplied path before anything expensive touches it.

    Raises:
        FileNotFoundError: no such file, or it is a directory.
        InvalidImageError: empty, undecodable, or too small to carry signal.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"not a file: {path}")
    if path.stat().st_size == 0:
        raise InvalidImageError(f"file is empty: {path}")

    # Verify decodability here rather than letting it surface as a stack trace
    # from inside the preprocessing pipeline.
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as im:
            im.verify()  # cheap: parses headers, does not decode pixels
        with Image.open(path) as im:
            width, height = im.size
    except UnidentifiedImageError as exc:
        raise InvalidImageError(f"not a readable image: {path}") from exc
    except OSError as exc:
        raise InvalidImageError(f"corrupt image file: {path} ({exc})") from exc

    if min(width, height) < MIN_DIMENSION:
        raise InvalidImageError(
            f"image is {width}x{height}; smaller than {MIN_DIMENSION}px carries no "
            f"diagnostic content once resized to {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}"
        )
    return path


class Classifier:
    """A loaded model, held open so weights are read once, not once per image."""

    def __init__(self, checkpoint: str | Path = DEFAULT_CHECKPOINT, model_name: str = "LXNet"):
        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"no checkpoint at {checkpoint}. Train one with `lxnet-train`, or pass "
                f"--checkpoint."
            )
        self.model_name = model_name
        # index_dataset assigns label i to the i-th class directory in sorted
        # order, so names must be ordered by sorted key -- not by dict insertion
        # order, which only coincides with it today.
        self.class_names = [CLASS_LABELS[k] for k in sorted(CLASS_LABELS)]
        self.model = build_model(model_name)
        self.model.load_weights(str(checkpoint))
        log.info("loaded %s (%d params) from %s", model_name, self.model.count_params(), checkpoint)

    def predict(self, path: str | Path, top_k: int | None = None) -> list[Prediction]:
        """Rank the classes for one image, most probable first."""
        path = validate_image_path(path)
        image = load_image(path)
        probabilities = self.model.predict(image[None], verbose=0)[0]

        ranked = [
            Prediction(int(i), self.class_names[i], float(probabilities[i]))
            for i in np.argsort(probabilities)[::-1]
        ]
        return ranked[:top_k] if top_k else ranked

    def predict_many(self, paths) -> dict[str, list[Prediction] | str]:
        """Predict over several files. One bad file does not abort the batch."""
        results: dict[str, list[Prediction] | str] = {}
        for p in paths:
            try:
                results[str(p)] = self.predict(p)
            except (FileNotFoundError, InvalidImageError) as exc:
                log.warning("skipping %s: %s", p, exc)
                results[str(p)] = f"error: {exc}"
        return results


def _as_json(results: dict) -> str:
    payload = {}
    for path, value in results.items():
        if isinstance(value, str):
            payload[path] = {"error": value}
        else:
            payload[path] = {
                "prediction": value[0].class_name,
                "confidence": round(value[0].probability, 6),
                "probabilities": {p.class_name: round(p.probability, 6) for p in value},
            }
    return json.dumps(payload, indent=2)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="lxnet-predict",
        description="Classify a chest X-ray into one of nine categories.",
        epilog="Research use only. Not a medical device; see docs/MODEL_CARD.md.",
    )
    parser.add_argument("images", nargs="+", type=Path, help="image file(s) to classify")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model", default="LXNet")
    parser.add_argument("--top-k", type=int, default=3, help="classes to show (0 = all)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--quiet", action="store_true", help="suppress the disclaimer")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    try:
        classifier = Classifier(args.checkpoint, args.model)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    results = classifier.predict_many(args.images)

    if args.json:
        print(_as_json(results))
    else:
        for path, value in results.items():
            print(f"\n{path}")
            if isinstance(value, str):
                print(f"  {value}")
                continue
            shown = value if args.top_k == 0 else value[: args.top_k]
            for prediction in shown:
                print(f"  {prediction}")
        if not args.quiet:
            # ASCII only: this lands on a Windows console defaulting to cp1252,
            # where an em dash renders as a replacement character.
            print(
                "\nResearch use only - not a medical device. Pneumonia recall is 73%.",
                file=sys.stderr,
            )

    # Non-zero if every input failed, so a caller can branch on it.
    return 0 if any(not isinstance(v, str) for v in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
