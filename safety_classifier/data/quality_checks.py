"""Lightweight quality checks for processed records."""

from __future__ import annotations

from typing import Any

MIN_CHARS = 3
MAX_CHARS = 20000


def check_record(rec: dict[str, Any]) -> tuple[bool, str | None]:
    """Return ``(ok, reason_if_skipped)`` for a processed record."""
    text = rec.get("text")
    if not isinstance(text, str):
        return False, "non_string_text"
    stripped = text.strip()
    if len(stripped) < MIN_CHARS:
        return False, "too_short"
    if len(stripped) > MAX_CHARS:
        return False, "too_long"
    labels = rec.get("labels")
    if not labels or not isinstance(labels, list):
        return False, "missing_labels"
    return True, None
