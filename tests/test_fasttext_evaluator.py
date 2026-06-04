"""Tests for FastText multi-label evaluation semantics."""

from __future__ import annotations

from pathlib import Path

from safety_classifier.fasttext_layer.evaluator import _read_test, evaluate_model


class _MultiLabelModel:
    def get_labels(self):
        return ["__label__hate", "__label__toxicity", "__label__safe"]

    def predict(self, text, k=1, threshold=0.0, on_unicode_error="strict"):
        if k == -1:
            return (
                ["__label__toxicity", "__label__hate", "__label__safe"],
                [0.8, 0.7, 0.1],
            )
        return (["__label__toxicity"], [0.8])


def test_read_test_groups_duplicate_multilabel_rows(tmp_path: Path):
    path = tmp_path / "abuse_test.txt"
    path.write_text(
        "__label__hate same text\n"
        "__label__toxicity same text\n"
        "__label__safe other text\n",
        encoding="utf-8",
    )

    examples = _read_test(path)

    assert examples[0] == ({"hate", "toxicity"}, "same text")
    assert examples[1] == ({"safe"}, "other text")


def test_evaluate_model_accepts_any_true_label_for_top1(tmp_path: Path):
    path = tmp_path / "abuse_test.txt"
    path.write_text(
        "__label__hate same text\n"
        "__label__toxicity same text\n",
        encoding="utf-8",
    )

    metrics = evaluate_model(_MultiLabelModel(), path, threshold=0.5)

    assert metrics["top1_accuracy"] == 1.0
    assert metrics["per_label"]["hate"]["recall"] == 1.0
    assert metrics["per_label"]["toxicity"]["recall"] == 1.0


def test_evaluate_model_reports_action_recall_metrics(tmp_path: Path):
    path = tmp_path / "abuse_test.txt"
    path.write_text(
        "__label__hate same text\n"
        "__label__toxicity same text\n",
        encoding="utf-8",
    )

    metrics = evaluate_model(_MultiLabelModel(), path, threshold=0.5)

    assert metrics["macro_route_floor_recall"] == 1.0
    assert metrics["macro_best_f1"] == 1.0
    assert metrics["action_metrics"]["hate"]["route_floor_recall"] == 1.0
    assert metrics["action_metrics"]["toxicity"]["route_floor_recall"] == 1.0
