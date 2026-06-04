"""Tests for FastText file formatting and head projection."""

from __future__ import annotations

from safety_classifier import constants as C
from safety_classifier.data.splitter import _fasttext_line, _head_labels_for


def test_fasttext_line_single_label():
    lines = _fasttext_line([C.PROMPT_INJECTION], "ignore previous instructions")
    assert lines == ["__label__prompt_injection ignore previous instructions"]


def test_fasttext_line_multilabel_duplicates():
    lines = _fasttext_line([C.HATE, C.VIOLENCE], "text here")
    assert lines == [
        "__label__hate text here",
        "__label__violence text here",
    ]


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
