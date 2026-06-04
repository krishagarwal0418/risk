"""Prompt-injection detector wrapper (protectai/deberta-v3-small-prompt-injection-v2).

The model's id2label is typically {0: "SAFE", 1: "INJECTION"}. We map injection-ish
labels to ``prompt_injection`` and benign-ish labels to safe (contributing nothing
to risk scores). Mapping is resolved against the *actual* id2label at runtime.
"""

from __future__ import annotations

from .. import constants as C
from .base import BaseHFClassifier

_INJECTION_TOKENS = ("injection", "prompt_injection", "malicious", "attack", "unsafe")
_SAFE_TOKENS = ("safe", "benign", "legit", "clean")


class PromptInjectionClassifier(BaseHFClassifier):
    def map_scores(self, raw: dict[str, float]) -> dict[str, float]:
        score = 0.0
        for label, prob in raw.items():
            low = label.lower()
            if any(tok in low for tok in _INJECTION_TOKENS):
                score = max(score, prob)
            elif any(tok in low for tok in _SAFE_TOKENS):
                continue  # safe contributes nothing
            # Unknown labels are preserved in `raw`; do not map to a risk label.
        return {C.PROMPT_INJECTION: score}
