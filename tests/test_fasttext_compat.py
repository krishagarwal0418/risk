"""Tests for FastText wrapper compatibility helpers."""

from __future__ import annotations

import pytest

from safety_classifier.fasttext_layer.compat import predict_fasttext


class _RawFastText:
    def predict(self, text, k, threshold, on_unicode_error):
        assert text == "hello"
        assert k == -1
        assert threshold == 0.0
        assert on_unicode_error == "strict"
        return [(0.9, "__label__safe"), (0.1, "__label__risk")]


class _Numpy2BrokenModel:
    f = _RawFastText()

    def predict(self, *args, **kwargs):
        raise ValueError("Unable to avoid copy while creating an array as requested.")


class _OtherBrokenModel:
    f = _RawFastText()

    def predict(self, *args, **kwargs):
        raise ValueError("something else")


def test_predict_fasttext_falls_back_for_numpy2_copy_error():
    labels, probs = predict_fasttext(_Numpy2BrokenModel(), "hello", k=-1)

    assert labels == ("__label__safe", "__label__risk")
    assert probs.tolist() == [0.9, 0.1]


def test_predict_fasttext_reraises_unrelated_value_error():
    with pytest.raises(ValueError, match="something else"):
        predict_fasttext(_OtherBrokenModel(), "hello", k=-1)
