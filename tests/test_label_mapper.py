"""Tests for the label mapper."""

from __future__ import annotations

from safety_classifier import constants as C
from safety_classifier.data.label_mapper import map_label, map_labels


def test_prompt_injection_aliases():
    for alias in ("injection", "prompt-injection", "malicious_instruction", "indirect_injection"):
        assert map_label(alias) == C.PROMPT_INJECTION


def test_jailbreak_aliases():
    for alias in ("jailbreak", "jailbreak_attempt", "dan", "bypass"):
        assert map_label(alias) == C.JAILBREAK


def test_toxicity_aliases():
    for alias in ("toxic", "severe_toxic", "insult", "obscene"):
        assert map_label(alias) == C.TOXICITY


def test_hate_aliases():
    for alias in ("hate", "identity_hate", "hate_speech", "discrimination"):
        assert map_label(alias) == C.HATE


def test_self_harm_aliases():
    for alias in ("self_harm", "self-harm", "suicide", "self_harm_instructions"):
        assert map_label(alias) == C.SELF_HARM


def test_violence_aliases():
    for alias in ("violence", "threat", "hate/threatening", "harassment_threatening"):
        assert map_label(alias) == C.VIOLENCE


def test_obscene_promoted_to_sexual_in_sexual_context():
    assert map_label("obscene") == C.TOXICITY
    assert map_label("obscene", context="explicit sexual content") == C.SEXUAL


def test_unknown_label_maps_to_unknown_not_dropped():
    assert map_label("some_brand_new_category") == C.UNKNOWN


def test_map_labels_drops_safe_when_risk_present():
    out = map_labels(["safe", "toxic"])
    assert C.TOXICITY in out
    assert C.SAFE not in out


def test_map_labels_multilabel():
    out = map_labels(["hate", "violence"])
    assert set(out) == {C.HATE, C.VIOLENCE}
