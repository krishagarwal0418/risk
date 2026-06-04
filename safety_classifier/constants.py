"""Canonical taxonomy and shared constants for the safety classifier."""

from __future__ import annotations

# Canonical labels used throughout the system.
SAFE = "safe"
PROMPT_INJECTION = "prompt_injection"
JAILBREAK = "jailbreak"
TOXICITY = "toxicity"
HATE = "hate"
HARASSMENT = "harassment"
SEXUAL = "sexual"
VIOLENCE = "violence"
SELF_HARM = "self_harm"
DANGEROUS_INFORMATION = "dangerous_information"
ILLEGAL_ACTIVITY = "illegal_activity"
UNKNOWN = "unknown"

# Full canonical label set (includes safe + unknown).
CANONICAL_LABELS: tuple[str, ...] = (
    SAFE,
    PROMPT_INJECTION,
    JAILBREAK,
    TOXICITY,
    HATE,
    HARASSMENT,
    SEXUAL,
    VIOLENCE,
    SELF_HARM,
    DANGEROUS_INFORMATION,
    ILLEGAL_ACTIVITY,
    UNKNOWN,
)

# Risk labels (everything that is not "safe" or "unknown").
RISK_LABELS: tuple[str, ...] = (
    PROMPT_INJECTION,
    JAILBREAK,
    TOXICITY,
    HATE,
    HARASSMENT,
    SEXUAL,
    VIOLENCE,
    SELF_HARM,
    DANGEROUS_INFORMATION,
    ILLEGAL_ACTIVITY,
)

# Labels that must always appear in the public API `scores` output.
PUBLIC_SCORE_LABELS: tuple[str, ...] = (
    PROMPT_INJECTION,
    JAILBREAK,
    TOXICITY,
    HATE,
    SEXUAL,
    VIOLENCE,
    SELF_HARM,
)

# Labels tracked internally but reported under `internal_scores`.
INTERNAL_SCORE_LABELS: tuple[str, ...] = (
    HARASSMENT,
    DANGEROUS_INFORMATION,
    ILLEGAL_ACTIVITY,
)

# All labels for which we maintain numeric scores (public + internal).
SCORED_LABELS: tuple[str, ...] = PUBLIC_SCORE_LABELS + INTERNAL_SCORE_LABELS

# FastText head -> the canonical labels each head predicts.
ATTACK_HEAD_LABELS: tuple[str, ...] = (SAFE, PROMPT_INJECTION, JAILBREAK)
ABUSE_HEAD_LABELS: tuple[str, ...] = (SAFE, TOXICITY, HATE, HARASSMENT)
HIGH_RISK_HEAD_LABELS: tuple[str, ...] = (
    SAFE,
    SEXUAL,
    VIOLENCE,
    SELF_HARM,
    DANGEROUS_INFORMATION,
    ILLEGAL_ACTIVITY,
)

HEAD_LABELS: dict[str, tuple[str, ...]] = {
    "attack": ATTACK_HEAD_LABELS,
    "abuse": ABUSE_HEAD_LABELS,
    "high_risk": HIGH_RISK_HEAD_LABELS,
}

# Decision and risk-level vocabularies.
DECISION_ALLOW = "allow"
DECISION_REVIEW = "review"
DECISION_BLOCK = "block"

RISK_NONE = "none"
RISK_LOW = "low"
RISK_MEDIUM = "medium"
RISK_HIGH = "high"
RISK_CRITICAL = "critical"

FASTTEXT_LABEL_PREFIX = "__label__"
