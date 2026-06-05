"""Prompt-injection detector wrapper.

The model's id2label is typically {0: "SAFE", 1: "INJECTION"} but some models
(e.g. fmops/distilbert-prompt-injection) ship with generic LABEL_0/LABEL_1 names.
For binary models with generic labels we remap at load time: LABEL_0=safe,
LABEL_1=injection (by HuggingFace convention the positive class is always last).
"""

from __future__ import annotations

from .. import constants as C
from .base import BaseHFClassifier

_INJECTION_TOKENS = ("injection", "prompt_injection", "malicious", "attack", "unsafe")
_SAFE_TOKENS = ("safe", "benign", "legit", "clean")


class PromptInjectionClassifier(BaseHFClassifier):
    def _load(self) -> None:
        super()._load()
        # Fix generic LABEL_0/LABEL_1 for binary prompt-injection models.
        # Convention: lower index = negative class (safe), higher = positive (injection).
        vals = list(self.id2label.values())
        if len(vals) == 2 and all(v.startswith("LABEL_") for v in vals):
            self.id2label = {0: "safe", 1: "injection"}

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
