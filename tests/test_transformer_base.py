"""Tests for transformer wrapper scoring behavior."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from safety_classifier.transformers_layer.base import BaseHFClassifier


class _FakeClassifier(BaseHFClassifier):
    def __init__(self, activation: str):
        self.score_activation = activation
        self.id2label = {0: "a", 1: "b"}

    def _logits(self, texts: list[str]):
        return torch.tensor([[2.0, 2.0]])

    def map_scores(self, raw):
        return raw


def test_softmax_activation_for_mutually_exclusive_labels():
    probs = _FakeClassifier("softmax")._probabilities(["x"])[0]

    assert probs == {"a": 0.5, "b": 0.5}


def test_sigmoid_activation_for_multilabel_categories():
    probs = _FakeClassifier("sigmoid")._probabilities(["x"])[0]

    assert round(probs["a"], 4) == 0.8808
    assert round(probs["b"], 4) == 0.8808
