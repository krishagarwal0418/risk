"""Merge model scores into a final decision + risk level.

Rules:
  * Max score per canonical label across all contributing models.
  * A "safe" signal never suppresses an unsafe label (we only ever take max).
  * Specialist models win naturally because their score is higher.
  * Agreement between FastText and a transformer on the same label adds a reason.
  * Decision: allow / review / block based on per-label thresholds.
"""

from __future__ import annotations

from typing import Any

from .. import constants as C
from .thresholds import Thresholds


def merge_scores(score_sources: list[dict[str, float]]) -> dict[str, float]:
    """Take the max per canonical label across all score dicts."""
    merged: dict[str, float] = {lab: 0.0 for lab in C.SCORED_LABELS}
    for source in score_sources:
        for label, value in source.items():
            if label in merged:
                merged[label] = max(merged[label], float(value))
    return merged


def _risk_level(
    merged: dict[str, float],
    thresholds: Thresholds,
    blocked_labels: list[str],
    reviewed_labels: list[str],
) -> str:
    max_score = max(merged.values()) if merged else 0.0
    # Critical conditions.
    if (
        merged.get(C.SELF_HARM, 0.0) >= 0.85
        or merged.get(C.VIOLENCE, 0.0) >= 0.90
        or len(blocked_labels) >= 2
    ):
        return C.RISK_CRITICAL
    if blocked_labels:
        return C.RISK_HIGH
    if reviewed_labels:
        return C.RISK_MEDIUM
    if max_score >= 0.20:
        return C.RISK_LOW
    return C.RISK_NONE


def decide(
    merged: dict[str, float],
    thresholds: Thresholds,
    fasttext_scores: dict[str, float] | None = None,
    extra_reasons: list[str] | None = None,
) -> dict[str, Any]:
    """Produce decision, risk_level, labels and reasons from merged scores."""
    reasons: list[str] = list(extra_reasons or [])
    blocked: list[str] = []
    reviewed: list[str] = []
    triggered_labels: list[str] = []

    for label, score in merged.items():
        t = thresholds.for_label(label)
        if score >= t.block:
            blocked.append(label)
            triggered_labels.append(label)
            reasons.append(f"{label} score {score:.2f} crossed block threshold {t.block:.2f}")
        elif score >= t.review:
            reviewed.append(label)
            triggered_labels.append(label)
            reasons.append(f"{label} score {score:.2f} crossed review threshold {t.review:.2f}")
        elif score >= t.route:
            triggered_labels.append(label)

    # Agreement signal between FastText and merged transformer score.
    if fasttext_scores:
        for label in C.SCORED_LABELS:
            ft = fasttext_scores.get(label, 0.0)
            tr = merged.get(label, 0.0)
            if ft >= 0.30 and tr >= 0.30:
                reasons.append(
                    f"FastText and transformer agree on {label} "
                    f"(ft={ft:.2f}, model={tr:.2f}) — elevated confidence"
                )

    if blocked:
        decision = C.DECISION_BLOCK
    elif reviewed:
        decision = C.DECISION_REVIEW
    else:
        decision = C.DECISION_ALLOW

    risk_level = _risk_level(merged, thresholds, blocked, reviewed)

    # Public labels = those that crossed at least the review threshold, ranked.
    final_labels = sorted(
        set(blocked + reviewed), key=lambda l: merged.get(l, 0.0), reverse=True
    )
    if not final_labels:
        final_labels = [C.SAFE] if max(merged.values(), default=0.0) < 0.20 else []
        if not final_labels:
            # low-risk: surface the top routed label if any
            routed = sorted(triggered_labels, key=lambda l: merged.get(l, 0.0), reverse=True)
            final_labels = routed[:1] if routed else [C.SAFE]

    return {
        "decision": decision,
        "risk_level": risk_level,
        "labels": final_labels,
        "reasons": reasons,
    }
