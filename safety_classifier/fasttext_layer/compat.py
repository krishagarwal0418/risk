"""Compatibility helpers for the FastText Python wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np


def predict_fasttext(
    model: Any,
    text: str,
    k: int = 1,
    threshold: float = 0.0,
    on_unicode_error: str = "strict",
):
    """Call ``model.predict`` with a NumPy 2 fallback.

    Some FastText wheels still return probabilities with
    ``np.array(probs, copy=False)``, which raises under NumPy 2. The underlying
    binding exposes ``model.f.predict`` with the same predictions, so we can
    reconstruct the public return shape with ``np.asarray``.
    """
    try:
        return model.predict(
            text,
            k=k,
            threshold=threshold,
            on_unicode_error=on_unicode_error,
        )
    except ValueError as exc:
        if "Unable to avoid copy" not in str(exc):
            raise
        predictions = model.f.predict(text, k, threshold, on_unicode_error)
        if predictions:
            probs, labels = zip(*predictions)
        else:
            probs, labels = [], ()
        return labels, np.asarray(probs, dtype=float)
