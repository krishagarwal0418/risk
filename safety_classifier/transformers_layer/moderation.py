"""Broad moderation wrapper.

The primary config uses ifmain/ModerationBERT-En-02; the fallback config uses
oxyapi/albert-moderation-001. Both expose OpenAI-style moderation categories once
generic ``LABEL_0`` names are mapped for ModerationBERT. A single raw category may
map to several canonical labels, e.g. ``hate/threatening`` -> hate + violence.
"""

from __future__ import annotations

from .. import constants as C
from .base import BaseHFClassifier

# Normalized raw category -> list of canonical labels it contributes to.
_CATEGORY_MAP: dict[str, list[str]] = {
    "harassment": [C.HARASSMENT, C.TOXICITY],
    "harassment/threatening": [C.HARASSMENT, C.TOXICITY, C.VIOLENCE],
    "harassment_threatening": [C.HARASSMENT, C.TOXICITY, C.VIOLENCE],
    "hate": [C.HATE],
    "hate/threatening": [C.HATE, C.VIOLENCE],
    "hate_threatening": [C.HATE, C.VIOLENCE],
    "self-harm": [C.SELF_HARM],
    "self_harm": [C.SELF_HARM],
    "self-harm/intent": [C.SELF_HARM],
    "self_harm_intent": [C.SELF_HARM],
    "self-harm/instructions": [C.SELF_HARM],
    "self_harm_instructions": [C.SELF_HARM],
    "sexual": [C.SEXUAL],
    "sexual/minors": [C.SEXUAL],
    "sexual_minors": [C.SEXUAL],
    "violence": [C.VIOLENCE],
    "violence/graphic": [C.VIOLENCE],
    "violence_graphic": [C.VIOLENCE],
    "dangerous": [C.DANGEROUS_INFORMATION],
    "dangerous_information": [C.DANGEROUS_INFORMATION],
    "weapons": [C.DANGEROUS_INFORMATION],
    "illegal": [C.ILLEGAL_ACTIVITY],
    "illegal_activity": [C.ILLEGAL_ACTIVITY],
    "toxic": [C.TOXICITY],
    "toxicity": [C.TOXICITY],
}

_SAFE_TOKENS = ("safe", "benign", "ok", "none", "clean", "neutral")

_MODERATION_BERT_LABELS = (
    "harassment",
    "harassment_threatening",
    "hate",
    "hate_threatening",
    "self_harm",
    "self_harm_instructions",
    "self_harm_intent",
    "sexual",
    "sexual_minors",
    "violence",
    "violence_graphic",
    "self-harm",
    "sexual/minors",
    "hate/threatening",
    "violence/graphic",
    "self-harm/intent",
    "self-harm/instructions",
    "harassment/threatening",
)


def _normalize_label(label: str) -> str:
    return label.strip().lower().replace(" ", "_")


class ModerationClassifier(BaseHFClassifier):
    score_activation = "sigmoid"

    def _load(self) -> None:
        super()._load()
        labels = tuple(self.id2label.get(i, "") for i in range(len(_MODERATION_BERT_LABELS)))
        generic = tuple(f"LABEL_{i}" for i in range(len(_MODERATION_BERT_LABELS)))
        if labels == generic:
            self.id2label = {i: label for i, label in enumerate(_MODERATION_BERT_LABELS)}

    def map_scores(self, raw: dict[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for label, prob in raw.items():
            norm = _normalize_label(label)
            if any(tok == norm or tok in norm for tok in _SAFE_TOKENS):
                continue
            # Try exact, then slash/underscore variants.
            targets = (
                _CATEGORY_MAP.get(norm)
                or _CATEGORY_MAP.get(norm.replace("_", "/"))
                or _CATEGORY_MAP.get(norm.replace("/", "_"))
            )
            if not targets:
                # Unknown category: preserved in raw payload, not mapped to risk.
                continue
            for canonical in targets:
                out[canonical] = max(out.get(canonical, 0.0), prob)
        return out
