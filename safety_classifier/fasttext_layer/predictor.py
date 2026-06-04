"""FastText routing layer: loads the three heads and produces canonical scores."""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any, Optional

from .. import constants as C
from ..config import load_models_config, resolve_path

# Canonical label -> the transformer model key that confirms it.
_LABEL_TO_MODEL: dict[str, str] = {
    C.PROMPT_INJECTION: "prompt_injection",
    C.JAILBREAK: "jailbreak",
    C.TOXICITY: "moderation",
    C.HATE: "moderation",
    C.HARASSMENT: "moderation",
    C.SEXUAL: "moderation",
    C.VIOLENCE: "moderation",
    C.SELF_HARM: "moderation",
    C.DANGEROUS_INFORMATION: "moderation",
    C.ILLEGAL_ACTIVITY: "moderation",
}


class FastTextSafetyRouter:
    """Loads attack / abuse / high_risk heads and scores text against them."""

    def __init__(
        self,
        attack_path: Optional[str] = None,
        abuse_path: Optional[str] = None,
        high_risk_path: Optional[str] = None,
        route_threshold: float = 0.20,
    ) -> None:
        cfg = load_models_config().get("fasttext", {})
        self._paths = {
            "attack": Path(attack_path or resolve_path(cfg.get("attack_head_path", ""))),
            "abuse": Path(abuse_path or resolve_path(cfg.get("abuse_head_path", ""))),
            "high_risk": Path(
                high_risk_path or resolve_path(cfg.get("high_risk_head_path", ""))
            ),
        }
        self.route_threshold = route_threshold
        self._models: dict[str, Any] = {}
        self._load()

    @property
    def loaded(self) -> bool:
        return len(self._models) > 0

    def _load(self) -> None:
        try:
            import fasttext
        except ImportError:
            warnings.warn(
                "fasttext is not installed; FastTextSafetyRouter is disabled. "
                "Install fasttext-wheel and train the heads via the Colab pipeline.",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        # fastText prints a deprecation warning to stderr on load; silence it.
        fasttext.FastText.eprint = lambda *a, **k: None  # type: ignore[attr-defined]
        for head, path in self._paths.items():
            if path and path.exists():
                self._models[head] = fasttext.load_model(str(path))
            else:
                warnings.warn(
                    f"FastText head '{head}' not found at {path}. Train it with the "
                    "Colab pipeline (scripts/03_train_fasttext_heads.py).",
                    RuntimeWarning,
                    stacklevel=2,
                )

    @staticmethod
    def _predict_head(model: Any, text: str) -> dict[str, float]:
        """Return label->prob for a single head using one-vs-all output."""
        labels, probs = model.predict(text.replace("\n", " "), k=-1)
        out: dict[str, float] = {}
        for lab, prob in zip(labels, probs):
            out[lab[len("__label__"):]] = float(prob)
        return out

    def predict(self, text: str) -> dict[str, Any]:
        """Score ``text`` across all canonical labels and suggest models to run."""
        started = time.time()
        scores: dict[str, float] = {lab: 0.0 for lab in C.SCORED_LABELS}

        if not self.loaded:
            return {
                "scores": scores,
                "suggested_models": [],
                "latency_ms": round((time.time() - started) * 1000, 3),
                "head_scores": {},
            }

        head_scores: dict[str, dict[str, float]] = {}
        for head, model in self._models.items():
            preds = self._predict_head(model, text)
            head_scores[head] = preds
            for lab, prob in preds.items():
                if lab in scores:
                    scores[lab] = max(scores[lab], prob)

        suggested: list[str] = []
        for lab, score in scores.items():
            if score >= self.route_threshold:
                model_key = _LABEL_TO_MODEL.get(lab)
                if model_key and model_key not in suggested:
                    suggested.append(model_key)

        return {
            "scores": scores,
            "suggested_models": suggested,
            "latency_ms": round((time.time() - started) * 1000, 3),
            "head_scores": head_scores,
        }
