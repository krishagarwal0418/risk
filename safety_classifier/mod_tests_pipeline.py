"""Local mod_tests safety pipeline.

Flow:
  validation -> normalization -> deterministic rules -> FastText heads ->
  local prompt-guard + toxic-bert transformer confirmers.

This is intentionally separate from the main ``SafetyClassifier`` so candidate
models can be evaluated without changing production config.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from . import constants as C
from .config import repo_root
from .fasttext_layer.predictor import FastTextSafetyRouter
from .normalizer import normalize
from .routing.merger import decide, merge_scores
from .routing.thresholds import Thresholds
from .transformers_layer.prompt_injection import PromptInjectionClassifier
from .transformers_layer.toxic_fallback import ToxicFallbackClassifier


_ATTACK_RULES = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "reveal your system prompt",
    "reveal developer message",
    "bypass policy",
    "you are now dan",
    "do anything now",
    "no restrictions",
    "unrestricted",
    "roleplay as",
    "pretend you are",
    "forget your instructions",
    "hidden prompt",
    "system message",
    "developer message",
    "jailbreak",
)

_TOXIC_ROUTE_LABELS = (
    C.TOXICITY,
    C.HATE,
    C.HARASSMENT,
    C.SEXUAL,
    C.VIOLENCE,
    C.SELF_HARM,
)


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


class ModTestsSafetyPipeline:
    """Candidate-model pipeline backed by local ``mod_tests`` artifacts."""

    def __init__(
        self,
        device: str = "cuda",
        prompt_guard_path: str | Path | None = None,
        toxic_bert_path: str | Path | None = None,
        use_fasttext: bool = True,
        full_scan_default: bool = False,
    ) -> None:
        root = repo_root()
        self.device = device
        self.thresholds = Thresholds()
        self.full_scan_default = full_scan_default
        self.prompt_guard_path = Path(
            prompt_guard_path or root / "mod_tests" / "prompt-guard-finetuned"
        )
        self.toxic_bert_path = Path(toxic_bert_path or root / "mod_tests" / "toxic-bert")
        self.fasttext = None
        if use_fasttext:
            try:
                self.fasttext = FastTextSafetyRouter()
            except Exception:  # noqa: BLE001
                self.fasttext = None
        self.models: dict[str, Any] = {
            "prompt_guard": PromptInjectionClassifier(
                str(self.prompt_guard_path),
                backend="pytorch",
                device=device,
                name="mod_tests/prompt-guard-finetuned",
            ),
            "toxic_bert": ToxicFallbackClassifier(
                str(self.toxic_bert_path),
                backend="pytorch",
                device=device,
                name="mod_tests/toxic-bert",
            ),
        }

    def _rules(self, detection_text: str) -> tuple[dict[str, float], list[str], list[str]]:
        scores = {lab: 0.0 for lab in C.SCORED_LABELS}
        route_models: list[str] = []
        reasons: list[str] = []
        matched = [rule for rule in _ATTACK_RULES if rule in detection_text]
        if matched:
            route_models.append("prompt_guard")
            reasons.append(f"deterministic attack rule matched: {matched[0]}")
        return scores, route_models, reasons

    def _select_models(
        self,
        rule_models: list[str],
        ft_scores: dict[str, float],
        full_scan: bool,
    ) -> list[str]:
        if full_scan:
            return ["prompt_guard", "toxic_bert"]
        selected = list(rule_models)
        if (
            ft_scores.get(C.PROMPT_INJECTION, 0.0) >= self.thresholds.route(C.PROMPT_INJECTION)
            or ft_scores.get(C.JAILBREAK, 0.0) >= self.thresholds.route(C.JAILBREAK)
        ):
            selected.append("prompt_guard")
        if any(ft_scores.get(label, 0.0) >= self.thresholds.route(label) for label in _TOXIC_ROUTE_LABELS):
            selected.append("toxic_bert")
        return list(dict.fromkeys(selected))

    def classify(
        self,
        text: str,
        *,
        full_scan: bool | None = None,
        include_raw: bool = False,
    ) -> dict[str, Any]:
        total_start = time.perf_counter()
        stage_latency_ms: dict[str, float] = {}
        raw_outputs: dict[str, Any] = {}

        start = time.perf_counter()
        if text is None or not str(text).strip():
            raise ValueError("Input text is empty or whitespace-only")
        stage_latency_ms["validation"] = _elapsed_ms(start)

        start = time.perf_counter()
        norm = normalize(text)
        stage_latency_ms["normalization"] = _elapsed_ms(start)

        start = time.perf_counter()
        rule_scores, rule_models, reasons = self._rules(norm.detection_text)
        stage_latency_ms["deterministic_rules"] = _elapsed_ms(start)

        start = time.perf_counter()
        if self.fasttext is not None:
            ft_result = self.fasttext.predict(norm.detection_text)
        else:
            ft_result = {
                "scores": {lab: 0.0 for lab in C.SCORED_LABELS},
                "suggested_models": [],
                "head_scores": {},
                "latency_ms": 0.0,
            }
        ft_scores = ft_result["scores"]
        stage_latency_ms["fasttext"] = _elapsed_ms(start)
        if include_raw:
            raw_outputs["fasttext"] = ft_result

        full_scan = self.full_scan_default if full_scan is None else full_scan
        models_to_run = self._select_models(rule_models, ft_scores, full_scan)

        # In this candidate pipeline FastText is a router, not a confirmer.
        # Keeping its scores out of final decisions prevents noisy local heads
        # from overwhelming the two models we are explicitly evaluating.
        score_sources = [rule_scores]
        triggered: list[str] = []
        for model_key in models_to_run:
            start = time.perf_counter()
            result = self.models[model_key].classify(norm.model_text)
            stage_latency_ms[model_key] = _elapsed_ms(start)
            score_sources.append(result["scores"])
            triggered.append(self.models[model_key].name)
            if include_raw:
                raw_outputs[model_key] = result

        merged = merge_scores(score_sources)
        decision = decide(merged, self.thresholds, fasttext_scores=ft_scores, extra_reasons=reasons)
        stage_latency_ms["total"] = _elapsed_ms(total_start)
        return {
            "decision": decision["decision"],
            "risk_level": decision["risk_level"],
            "labels": decision["labels"],
            "scores": {lab: round(merged[lab], 4) for lab in C.PUBLIC_SCORE_LABELS},
            "internal_scores": {lab: round(merged[lab], 4) for lab in C.INTERNAL_SCORE_LABELS},
            "triggered_models": triggered,
            "latency_ms": stage_latency_ms["total"],
            "stage_latency_ms": stage_latency_ms,
            "reasons": decision["reasons"],
            "raw_model_outputs": raw_outputs if include_raw else None,
        }
