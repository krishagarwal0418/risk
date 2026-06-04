"""Tests for the score merger and decision logic."""

from __future__ import annotations

from safety_classifier import constants as C
from safety_classifier.routing.merger import decide, merge_scores
from safety_classifier.routing.thresholds import Thresholds


def test_merge_keeps_max_score():
    merged = merge_scores([
        {C.PROMPT_INJECTION: 0.3, C.HATE: 0.1},
        {C.PROMPT_INJECTION: 0.9, C.HATE: 0.05},
    ])
    assert merged[C.PROMPT_INJECTION] == 0.9
    assert merged[C.HATE] == 0.1


def test_safe_does_not_erase_unsafe():
    # A "safe" source contributes only zeros; it must not lower an unsafe score.
    merged = merge_scores([
        {C.TOXICITY: 0.8},
        {lab: 0.0 for lab in C.SCORED_LABELS},  # safe signal
    ])
    assert merged[C.TOXICITY] == 0.8


def test_all_canonical_scores_exist():
    merged = merge_scores([{C.HATE: 0.5}])
    for label in C.SCORED_LABELS:
        assert label in merged


def test_block_decision():
    thresholds = Thresholds()
    merged = merge_scores([{C.PROMPT_INJECTION: 0.95}])
    result = decide(merged, thresholds)
    assert result["decision"] == C.DECISION_BLOCK
    assert C.PROMPT_INJECTION in result["labels"]
    assert result["risk_level"] in (C.RISK_HIGH, C.RISK_CRITICAL)


def test_review_decision():
    thresholds = Thresholds()
    # prompt_injection review=0.50, block=0.80 -> 0.6 should be review.
    merged = merge_scores([{C.PROMPT_INJECTION: 0.6}])
    result = decide(merged, thresholds)
    assert result["decision"] == C.DECISION_REVIEW
    assert result["risk_level"] == C.RISK_MEDIUM


def test_allow_decision():
    thresholds = Thresholds()
    merged = merge_scores([{C.TOXICITY: 0.05}])
    result = decide(merged, thresholds)
    assert result["decision"] == C.DECISION_ALLOW
    assert result["risk_level"] == C.RISK_NONE


def test_critical_on_self_harm():
    thresholds = Thresholds()
    merged = merge_scores([{C.SELF_HARM: 0.9}])
    result = decide(merged, thresholds)
    assert result["risk_level"] == C.RISK_CRITICAL


def test_critical_on_multiple_blocks():
    thresholds = Thresholds()
    merged = merge_scores([{C.PROMPT_INJECTION: 0.95, C.HATE: 0.9}])
    result = decide(merged, thresholds)
    assert result["risk_level"] == C.RISK_CRITICAL
