"""Tests for FastText file formatting and head projection."""

from __future__ import annotations

from safety_classifier import constants as C
from collections import Counter

from safety_classifier.data.splitter import (
    _fasttext_line,
    _head_labels_for,
    _label_cap_for,
    _labels_under_cap,
)


def test_fasttext_line_single_label():
    lines = _fasttext_line([C.PROMPT_INJECTION], "ignore previous instructions")
    assert lines == ["__label__prompt_injection ignore previous instructions"]


def test_fasttext_line_multilabel_single_line():
    # Multi-label -> single line with all label prefixes (FastText native format).
    # Old behavior (one line per label) caused contradictory OVA gradients.
    lines = _fasttext_line([C.HATE, C.VIOLENCE], "text here")
    assert lines == ["__label__hate __label__violence text here"]
    assert len(lines) == 1  # always one line regardless of label count


def test_fasttext_line_flattens_newlines():
    lines = _fasttext_line([C.SAFE], "line one\nline two")
    assert "\n" not in lines[0]
    assert lines[0] == "__label__safe line one line two"


def test_label_prefix_constant():
    assert C.FASTTEXT_LABEL_PREFIX == "__label__"


def test_head_projection_attack():
    # prompt_injection belongs to the attack head.
    assert _head_labels_for([C.PROMPT_INJECTION], C.ATTACK_HEAD_LABELS) == [C.PROMPT_INJECTION]
    # sexual is out of scope for the attack head -> skipped.
    assert _head_labels_for([C.SEXUAL], C.ATTACK_HEAD_LABELS) == []


def test_head_projection_safe_used_in_all_heads():
    for head_labels in (C.ATTACK_HEAD_LABELS, C.ABUSE_HEAD_LABELS, C.HIGH_RISK_HEAD_LABELS):
        assert _head_labels_for([C.SAFE], head_labels) == [C.SAFE]


def test_head_projection_multilabel_within_head():
    out = _head_labels_for([C.HATE, C.TOXICITY], C.ABUSE_HEAD_LABELS)
    assert set(out) == {C.HATE, C.TOXICITY}


def test_labels_under_cap_filters_only_capped_labels():
    labels = [C.HATE, C.TOXICITY]
    # hate and toxicity share a budget (group key = hate).
    # With hate written=2 at cap=2, the shared group is full -> both filtered.
    written = Counter({C.HATE: 2, C.TOXICITY: 1})
    assert _labels_under_cap(labels, written, cap=2) == []
    assert _labels_under_cap(labels, written, cap=0) == labels


def test_labels_under_cap_shared_budget_hate_toxicity():
    # A text with only toxicity still consumes the shared hate/toxicity budget.
    written = Counter({C.HATE: 1})  # 1 against the shared key
    assert _labels_under_cap([C.TOXICITY], written, cap=2) == [C.TOXICITY]
    written[C.HATE] = 2
    assert _labels_under_cap([C.TOXICITY], written, cap=2) == []


def test_label_cap_for_prefers_head_specific_env(monkeypatch):
    monkeypatch.setenv("SC_MAX_PER_LABEL", "22000")
    monkeypatch.setenv("SC_MAX_PER_LABEL_HIGH_RISK", "3000")

    assert _label_cap_for("train", "attack") == 22000
    assert _label_cap_for("train", "high_risk") == 3000
    assert _label_cap_for("val", "high_risk") == 375
