"""Jailbreak detector wrapper (madhurjindal/Jailbreak-Detector).

id2label is typically {0: "benign", 1: "jailbreak"}. Mapped against the actual
runtime id2label.
"""

from __future__ import annotations

from .. import constants as C
from .base import BaseHFClassifier

_JAILBREAK_TOKENS = ("jailbreak", "attack", "unsafe", "malicious")
_SAFE_TOKENS = ("benign", "safe", "clean", "legit")


class JailbreakClassifier(BaseHFClassifier):
    def map_scores(self, raw: dict[str, float]) -> dict[str, float]:
        score = 0.0
        for label, prob in raw.items():
            low = label.lower()
            if any(tok in low for tok in _JAILBREAK_TOKENS):
                score = max(score, prob)
            elif any(tok in low for tok in _SAFE_TOKENS):
                continue
        return {C.JAILBREAK: score}
