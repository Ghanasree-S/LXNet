"""Tests for metrics and the Wilcoxon signed-rank comparison.

Macro-averaged metrics matter here because the dataset is imbalanced 1340:544 --
a model that ignores the smallest class can still post a decent micro accuracy,
so the tests pin the behaviour that exposes that.
"""

import numpy as np
import pytest

from lxnet.evaluate import compute_metrics, summarise_folds, wilcoxon_compare


class TestComputeMetrics:
    def test_perfect_predictions_score_one(self):
        y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        m = compute_metrics(y, y)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["f1_macro"] == pytest.approx(1.0)

    def test_reports_accuracy_precision_recall_and_f1(self):
        y = np.array([0, 0, 1, 1])
        m = compute_metrics(y, np.array([0, 1, 1, 1]))
        assert set(m) >= {"accuracy", "precision_macro", "recall_macro", "f1_macro"}

    def test_accuracy_counts_exact_matches(self):
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 8])
        assert compute_metrics(y_true, y_pred)["accuracy"] == pytest.approx(0.75)

    def test_macro_f1_punishes_ignoring_a_minority_class(self):
        """9:1 imbalance, model always predicts the majority."""
        y_true = np.array([0] * 90 + [1] * 10)
        y_pred = np.zeros(100, dtype=int)

        m = compute_metrics(y_true, y_pred)

        assert m["accuracy"] == pytest.approx(0.90)
        assert m["f1_macro"] < 0.5, "macro F1 must expose the ignored class"

    def test_includes_a_confusion_matrix_of_the_right_shape(self):
        y = np.arange(9)
        m = compute_metrics(y, y, num_classes=9)
        assert np.asarray(m["confusion_matrix"]).shape == (9, 9)

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            compute_metrics(np.array([0, 1]), np.array([0]))


class TestWilcoxonCompare:
    def test_identical_scores_are_not_significant(self):
        a = [0.90, 0.91, 0.89, 0.92, 0.90]
        result = wilcoxon_compare(a, a)
        assert result["p_value"] > 0.05
        assert result["significant"] is False

    def test_detects_a_consistent_difference(self):
        a = [0.95, 0.96, 0.94, 0.97, 0.95, 0.96]
        b = [0.80, 0.81, 0.79, 0.82, 0.80, 0.81]
        result = wilcoxon_compare(a, b)
        assert result["p_value"] < 0.05
        assert result["significant"] is True

    def test_reports_the_median_difference(self):
        a = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        b = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
        assert wilcoxon_compare(a, b)["median_difference"] == pytest.approx(0.1)

    def test_requires_equal_length_samples(self):
        with pytest.raises(ValueError):
            wilcoxon_compare([0.9, 0.8], [0.9])


class TestSummariseFolds:
    def test_reports_mean_and_standard_deviation(self):
        s = summarise_folds([{"accuracy": 0.9}, {"accuracy": 0.8}, {"accuracy": 0.85}])
        assert s["accuracy_mean"] == pytest.approx(0.85)
        assert s["accuracy_std"] == pytest.approx(np.std([0.9, 0.8, 0.85]))

    def test_summarises_every_numeric_metric(self):
        folds = [
            {"accuracy": 0.9, "f1_macro": 0.8},
            {"accuracy": 0.8, "f1_macro": 0.7},
        ]
        s = summarise_folds(folds)
        assert "f1_macro_mean" in s and "accuracy_mean" in s

    def test_ignores_non_scalar_entries(self):
        folds = [
            {"accuracy": 0.9, "confusion_matrix": [[1, 0], [0, 1]]},
            {"accuracy": 0.8, "confusion_matrix": [[1, 0], [0, 1]]},
        ]
        s = summarise_folds(folds)
        assert "confusion_matrix_mean" not in s

    def test_rejects_an_empty_fold_list(self):
        with pytest.raises(ValueError):
            summarise_folds([])
