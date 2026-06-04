"""Optional toxicity specialist fallback (Intel/toxic-prompt-roberta or unitary/toxic-bert).

Used only when enabled and the broad moderation model returns medium-confidence
toxicity/hate. Maps Jigsaw-style multi-label categories to canonical labels.
"""

from __future__ import annotations

from .. import constants as C
from .base import BaseHFClassifier

_TOKEN_MAP: dict[str, list[str]] = {
    "toxic": [C.TOXICITY],
    "toxicity": [C.TOXICITY],
    "severe_toxic": [C.TOXICITY],
    "severe_toxicity": [C.TOXICITY],
    "insult": [C.TOXICITY],
    "obscene": [C.TOXICITY],  # promoted to sexual upstream if sexual context
    "threat": [C.VIOLENCE],
    "identity_hate": [C.HATE],
    "identity_attack": [C.HATE],
    "hate": [C.HATE],
}

_SAFE_TOKENS = ("non_toxic", "non-toxic", "safe", "neutral", "clean", "label_0")


def _norm(label: str) -> str:
    return label.strip().lower().replace(" ", "_").replace("-", "_")


class ToxicFallbackClassifier(BaseHFClassifier):
    def map_scores(self, raw: dict[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for label, prob in raw.items():
            norm = _norm(label)
            if any(tok in norm for tok in _SAFE_TOKENS):
                continue
            targets = _TOKEN_MAP.get(norm)
            if not targets:
                continue
            for canonical in targets:
                out[canonical] = max(out.get(canonical, 0.0), prob)
        return out
