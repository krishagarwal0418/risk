"""Long-text windowing.

Prompt injections often hide at the *end* of long inputs, so we always classify
the first and last windows. Middle windows are added in full-scan mode. Window
scores are merged by taking the max per label, and the winning window is tracked
for explainability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class Window:
    name: str          # "first" | "last" | "middle_{i}"
    text: str


def _word_windows(words: list[str], budget: int) -> tuple[str, str, list[str]]:
    """Return (first_chunk, last_chunk, middle_chunks) as space-joined strings.

    ``budget`` is an approximate token budget; we use a word count as a cheap proxy
    (a real tokenizer is applied downstream with truncation=max_length anyway).
    """
    first = " ".join(words[:budget])
    last = " ".join(words[-budget:])
    middles: list[str] = []
    start = budget
    while start < len(words) - budget:
        middles.append(" ".join(words[start : start + budget]))
        start += budget
    return first, last, middles


def build_windows(
    text: str,
    token_budget: int = 128,
    long_text_threshold: int = 200,
    full_scan: bool = False,
) -> list[Window]:
    """Build the windows to classify for ``text``."""
    words = text.split()
    if len(words) <= long_text_threshold:
        return [Window("full", text)]
    first, last, middles = _word_windows(words, token_budget)
    windows = [Window("first", first), Window("last", last)]
    if full_scan:
        windows.extend(Window(f"middle_{i}", m) for i, m in enumerate(middles))
    return windows


def merge_window_scores(
    scorer: Callable[[str], dict[str, float]],
    windows: list[Window],
) -> tuple[dict[str, float], dict[str, str]]:
    """Run ``scorer`` over each window; return (max scores, winning window per label)."""
    merged: dict[str, float] = {}
    winning: dict[str, str] = {}
    for window in windows:
        scores = scorer(window.text)
        for label, score in scores.items():
            if score > merged.get(label, -1.0):
                merged[label] = score
                winning[label] = window.name
    return merged, winning
