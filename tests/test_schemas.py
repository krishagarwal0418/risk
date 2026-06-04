"""Tests for the Pydantic schemas."""

from __future__ import annotations

from safety_classifier import constants as C
from safety_classifier.schemas import (
    FastTextResult,
    ModelResult,
    SafetyRequest,
    SafetyResult,
)


def test_safety_request_defaults():
    req = SafetyRequest(text="hello")
    assert req.full_scan is False
    assert req.include_raw is False
    assert req.request_id is None


def test_model_result_roundtrip():
    mr = ModelResult(model_name="m", scores={C.HATE: 0.5}, raw=None, latency_ms=1.2)
    assert mr.scores[C.HATE] == 0.5
    assert mr.raw is None


def test_fasttext_result():
    fr = FastTextResult(
        head_name="attack", scores={C.PROMPT_INJECTION: 0.3},
        suggested_models=["prompt_injection"], latency_ms=0.5,
    )
    assert fr.suggested_models == ["prompt_injection"]


def test_safety_result_full():
    result = SafetyResult(
        decision=C.DECISION_BLOCK,
        risk_level=C.RISK_HIGH,
        labels=[C.PROMPT_INJECTION],
        scores={lab: 0.0 for lab in C.PUBLIC_SCORE_LABELS},
        internal_scores={lab: 0.0 for lab in C.INTERNAL_SCORE_LABELS},
        triggered_models=["fasttext_heads"],
        skipped_models=[],
        latency_ms=12.3,
        reasons=["test"],
    )
    dumped = result.model_dump()
    for label in C.PUBLIC_SCORE_LABELS:
        assert label in dumped["scores"]
    assert dumped["decision"] == C.DECISION_BLOCK


def test_safety_result_constructed_from_router_output():
    # The router returns a plain dict; SafetyResult(**dict) must accept it.
    router_output = {
        "decision": "allow",
        "risk_level": "none",
        "labels": ["safe"],
        "scores": {lab: 0.0 for lab in C.PUBLIC_SCORE_LABELS},
        "internal_scores": {lab: 0.0 for lab in C.INTERNAL_SCORE_LABELS},
        "triggered_models": [],
        "skipped_models": [],
        "latency_ms": 1.0,
        "reasons": [],
        "raw_model_outputs": None,
    }
    result = SafetyResult(**router_output)
    assert result.decision == "allow"
