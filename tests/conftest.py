"""Make the repo importable and provide shared test fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest

from safety_classifier import constants as C


class MockModel:
    """A transformer-confirmer stand-in: returns fixed canonical scores."""

    def __init__(self, name: str, scores: dict[str, float]) -> None:
        self.name = name
        self._scores = scores

    def classify(self, text: str) -> dict:
        full = {lab: 0.0 for lab in C.SCORED_LABELS}
        full.update(self._scores)
        return {"model_name": self.name, "scores": full, "raw": {}, "latency_ms": 0.1}


class MockFastText:
    """A FastText router stand-in with controllable scores + suggestions."""

    loaded = True

    def __init__(self, scores: dict[str, float], suggested: list[str]) -> None:
        self._scores = scores
        self._suggested = suggested

    def predict(self, text: str) -> dict:
        full = {lab: 0.0 for lab in C.SCORED_LABELS}
        full.update(self._scores)
        return {"scores": full, "suggested_models": self._suggested, "latency_ms": 0.2}


@pytest.fixture
def mock_model_factory():
    return MockModel


@pytest.fixture
def mock_fasttext_factory():
    return MockFastText
