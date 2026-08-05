"""Contract for turning results.json into tables and figures."""

import pytest

from lxnet import report


@pytest.fixture
def grouped_results():
    return {
        "split_mode": "grouped",
        "split": {"train": 4660, "val": 1000, "test": 997},
        "dedupe": {"kept": 6657, "exact_duplicates": 84, "cross_class_conflicts": 1},
        "holdout": {
            "LXNet": {"accuracy": 0.71, "f1_macro": 0.70, "params": 356585},
            "DenseNet201": {"accuracy": 0.78, "f1_macro": 0.77, "params": 18816073},
        },
        "cross_validation": {
            "LXNet": {
                "folds": [
                    {"accuracy": a, "f1_macro": a - 0.01} for a in (0.70, 0.72, 0.69, 0.71, 0.73)
                ],
                "summary": {"accuracy_mean": 0.71, "accuracy_std": 0.014},
            }
        },
    }


@pytest.fixture
def random_results():
    return {
        "split_mode": "random",
        "split": {"train": 4660, "val": 999, "test": 998},
        "holdout": {"LXNet": {"accuracy": 0.98, "f1_macro": 0.98, "params": 356585}},
        "cross_validation": {
            "LXNet": {
                "folds": [{"accuracy": a, "f1_macro": a} for a in (0.97, 0.98, 0.98, 0.97, 0.98)],
                "summary": {"accuracy_mean": 0.976, "accuracy_std": 0.005},
            }
        },
    }


def test_comparison_table_pairs_the_two_protocols(grouped_results, random_results):
    rows = report.build_comparison_table(grouped_results, random_results)
    lx = next(r for r in rows if r["model"] == "LXNet")
    assert lx["grouped_accuracy"] == pytest.approx(0.71)
    assert lx["random_accuracy"] == pytest.approx(0.98)
    assert lx["inflation"] == pytest.approx(0.27, abs=1e-9)


def test_comparison_table_handles_models_missing_from_one_arm(grouped_results, random_results):
    rows = report.build_comparison_table(grouped_results, random_results)
    dn = next(r for r in rows if r["model"] == "DenseNet201")
    assert dn["random_accuracy"] is None
    assert dn["inflation"] is None
    assert dn["grouped_accuracy"] == pytest.approx(0.78)


def test_comparison_table_is_sorted_by_honest_accuracy(grouped_results, random_results):
    rows = report.build_comparison_table(grouped_results, random_results)
    accs = [r["grouped_accuracy"] for r in rows]
    assert accs == sorted(accs, reverse=True)


def test_markdown_table_renders_every_model_and_marks_missing(grouped_results, random_results):
    rows = report.build_comparison_table(grouped_results, random_results)
    md = report.markdown_table(rows)
    assert "LXNet" in md and "DenseNet201" in md
    assert md.count("\n") >= len(rows) + 1
    assert "—" in md, "models absent from an arm must render a dash, not a crash"


def test_markdown_table_reports_percentages_not_fractions(grouped_results, random_results):
    md = report.markdown_table(report.build_comparison_table(grouped_results, random_results))
    assert "71.0" in md and "98.0" in md


def test_figures_are_written(tmp_path, grouped_results, random_results):
    written = report.write_figures(grouped_results, random_results, out_dir=tmp_path)
    assert written, "expected at least one figure"
    for path in written:
        assert path.exists() and path.stat().st_size > 1000, f"{path} looks empty"


def test_write_figures_survives_absent_cross_validation(tmp_path, grouped_results):
    trimmed = dict(grouped_results)
    trimmed["cross_validation"] = {}
    written = report.write_figures(trimmed, None, out_dir=tmp_path)
    assert all(p.exists() for p in written)


def test_summary_paragraph_quotes_the_leakage_rate(grouped_results, random_results):
    text = report.summarise(grouped_results, random_results, leaked_fraction=0.404)
    assert "40.4" in text
    assert "LXNet" in text
