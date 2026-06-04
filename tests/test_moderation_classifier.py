"""Tests for broad moderation score mapping."""

from __future__ import annotations

from safety_classifier import constants as C
from safety_classifier.transformers_layer.moderation import ModerationClassifier


def test_moderation_maps_moderationbert_self_harm_aliases():
    clf = object.__new__(ModerationClassifier)

    scores = clf.map_scores(
        {
            "self_harm": 0.3,
            "self_harm_instructions": 0.8,
            "self_harm_intent": 0.7,
        }
    )

    assert scores[C.SELF_HARM] == 0.8


def test_moderation_maps_moderationbert_threatening_aliases():
    clf = object.__new__(ModerationClassifier)

    scores = clf.map_scores(
        {
            "hate_threatening": 0.6,
            "violence_graphic": 0.9,
            "harassment_threatening": 0.4,
        }
    )

    assert scores[C.HATE] == 0.6
    assert scores[C.VIOLENCE] == 0.9
    assert scores[C.HARASSMENT] == 0.4
