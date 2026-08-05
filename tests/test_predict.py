"""Contract for the inference path — the trust boundary and the preprocessing bond.

The failures worth pinning here are the quiet ones: a corrupt upload that reaches
the model anyway, and inference preprocessing drifting away from the preprocessing
the weights were trained on.
"""

import json

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from lxnet import predict
from lxnet.predict import Classifier, InvalidImageError, Prediction, validate_image_path


@pytest.fixture
def xray(tmp_path):
    """A plausible grayscale radiograph-shaped file."""
    rng = np.random.default_rng(0)
    path = tmp_path / "chest.png"
    Image.fromarray(rng.integers(0, 255, (450, 450), dtype=np.uint8), mode="L").save(path)
    return path


# --- input validation: everything below must fail before touching the model ---


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="no such file"):
        validate_image_path(tmp_path / "absent.png")


def test_directory_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a file"):
        validate_image_path(tmp_path)


def test_empty_file_is_rejected(tmp_path):
    empty = tmp_path / "empty.png"
    empty.touch()
    with pytest.raises(InvalidImageError, match="empty"):
        validate_image_path(empty)


def test_non_image_bytes_are_rejected(tmp_path):
    """A .png that is actually text — the shape of a bad upload."""
    fake = tmp_path / "notreally.png"
    fake.write_bytes(b"this is not a PNG, it is prose")
    with pytest.raises(InvalidImageError, match="not a readable image"):
        validate_image_path(fake)


def test_truncated_image_is_rejected(tmp_path, xray):
    truncated = tmp_path / "truncated.png"
    truncated.write_bytes(xray.read_bytes()[:60])
    with pytest.raises(InvalidImageError):
        validate_image_path(truncated)


def test_thumbnail_sized_image_is_rejected(tmp_path):
    """Decodable, but nothing survives the upscale to 224x224."""
    tiny = tmp_path / "tiny.png"
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="L").save(tiny)
    with pytest.raises(InvalidImageError, match="carries no"):
        validate_image_path(tiny)


def test_a_real_image_passes(xray):
    assert validate_image_path(xray) == xray


# --- the preprocessing bond between training and inference ---


def test_inference_preprocessing_matches_the_training_cache(xray):
    """If these ever diverge, accuracy silently degrades and nothing errors."""
    from lxnet.data import Sample
    from lxnet.pipeline import build_cache
    from lxnet.preprocess import load_image

    cached, _ = build_cache([Sample(path=str(xray), label=0, class_name="00 x", digest="d")])
    # build_cache stores uint8 (H, W); make_dataset later scales and replicates.
    from_cache = np.repeat((cached[0].astype(np.float32) / 255.0)[..., None], 3, axis=-1)
    from_inference = load_image(xray)

    assert from_inference.shape == from_cache.shape
    np.testing.assert_allclose(from_inference, from_cache, atol=1 / 255)


# --- prediction shape and ordering ---


@pytest.fixture
def classifier(tmp_path, monkeypatch):
    """A real Classifier over a tiny stand-in model, so no checkpoint is needed."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((224, 224, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(9, activation="softmax"),
        ]
    )
    weights = tmp_path / "stub.weights.h5"
    model.save_weights(str(weights))
    monkeypatch.setattr(predict, "build_model", lambda name: model)
    return Classifier(weights, "LXNet")


def test_missing_checkpoint_names_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="lxnet-train"):
        Classifier(tmp_path / "nope.weights.h5")


def test_predict_returns_one_entry_per_class(classifier, xray):
    out = classifier.predict(xray)
    assert len(out) == 9
    assert all(isinstance(p, Prediction) for p in out)


def test_probabilities_form_a_distribution(classifier, xray):
    out = classifier.predict(xray)
    assert all(0.0 <= p.probability <= 1.0 for p in out)
    assert sum(p.probability for p in out) == pytest.approx(1.0, abs=1e-5)


def test_predictions_are_ranked_most_probable_first(classifier, xray):
    probabilities = [p.probability for p in classifier.predict(xray)]
    assert probabilities == sorted(probabilities, reverse=True)


def test_labels_and_names_stay_aligned(classifier, xray):
    """A shuffled name lookup would produce confident, wrong, plausible output.

    Label i is the i-th class directory in sorted order -- the rule
    ``data.index_dataset`` uses -- so inference must resolve names the same way.
    """
    from lxnet.data import CLASS_LABELS

    expected = [CLASS_LABELS[k] for k in sorted(CLASS_LABELS)]
    assert classifier.class_names == expected
    for p in classifier.predict(xray):
        assert expected[p.label] == p.class_name


def test_top_k_truncates(classifier, xray):
    assert len(classifier.predict(xray, top_k=3)) == 3


def test_one_bad_file_does_not_abort_the_batch(classifier, xray, tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"nope")
    results = classifier.predict_many([xray, bad])
    assert isinstance(results[str(xray)], list)
    assert str(results[str(bad)]).startswith("error:")


# --- CLI ---


def test_cli_emits_valid_json(classifier, xray, monkeypatch, capsys):
    monkeypatch.setattr(predict, "Classifier", lambda *a, **k: classifier)
    from lxnet.data import CLASS_LABELS

    code = predict.main([str(xray), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload[str(xray)]["prediction"] in CLASS_LABELS.values()
    assert len(payload[str(xray)]["probabilities"]) == 9


def test_cli_reports_failure_when_every_input_is_bad(classifier, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(predict, "Classifier", lambda *a, **k: classifier)
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"nope")
    assert predict.main([str(bad)]) == 1


def test_cli_exits_2_without_a_checkpoint(tmp_path, capsys):
    code = predict.main([str(tmp_path / "x.png"), "--checkpoint", str(tmp_path / "absent.h5")])
    assert code == 2
    assert "error" in capsys.readouterr().err
