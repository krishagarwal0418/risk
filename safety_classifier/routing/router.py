"""Routing orchestration: FastText first, then transformer confirmers.

The router is model-agnostic: it receives a FastText router and a dict of
transformer "confirmer" models keyed by ``prompt_injection`` / ``jailbreak`` /
``moderation`` / ``toxic``. Each confirmer exposes ``classify(text) -> {scores: ...}``.
This makes the router straightforward to unit-test with mock models.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from .. import constants as C
from ..normalizer import NormalizedText, normalize
from .merger import decide, merge_scores
from .thresholds import Thresholds
from .windowing import build_windows, merge_window_scores

# Heuristic phrases that route to the attack/jailbreak confirmers. These NEVER
# block on their own — they only cause a transformer model to run.
_ATTACK_HEURISTICS = (
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

# Labels handled by each confirmer model.
_MODEL_LABELS: dict[str, tuple[str, ...]] = {
    "prompt_injection": (C.PROMPT_INJECTION,),
    "jailbreak": (C.JAILBREAK,),
    "moderation": (
        C.TOXICITY, C.HATE, C.HARASSMENT, C.SEXUAL, C.VIOLENCE,
        C.SELF_HARM,
    ),
    "toxic": (C.TOXICITY, C.HATE),
}


class Router:
    def __init__(
        self,
        fasttext_router: Optional[Any] = None,
        models: Optional[dict[str, Any]] = None,
        thresholds: Optional[Thresholds] = None,
        enable_toxic_fallback: bool = False,
        full_scan_default: bool = False,
        window_token_budget: int = 128,
        long_text_threshold: int = 200,
        toxic_medium_band: tuple[float, float] = (0.30, 0.70),
        fast_block_threshold: float = 0.90,
    ) -> None:
        self.fasttext = fasttext_router
        self.models = models or {}
        self.thresholds = thresholds or Thresholds()
        self.enable_toxic_fallback = enable_toxic_fallback
        self.full_scan_default = full_scan_default
        self.window_token_budget = window_token_budget
        self.long_text_threshold = long_text_threshold
        self.toxic_medium_band = toxic_medium_band
        self.fast_block_threshold = fast_block_threshold

    # ------------------------------------------------------------------ #
    def _heuristic_attack(self, detection_text: str) -> bool:
        return any(h in detection_text for h in _ATTACK_HEURISTICS)

    def _select_models(
        self,
        ft_scores: dict[str, float],
        suggested: list[str],
        detection_text: str,
        full_scan: bool,
    ) -> tuple[list[str], list[str]]:
        """Return (models_to_run, reasons)."""
        reasons: list[str] = []
        to_run: list[str] = []

        def add(model: str, reason: str) -> None:
            if model in self.models and model not in to_run:
                to_run.append(model)
                reasons.append(reason)

        if full_scan:
            for model in ("prompt_injection", "jailbreak", "moderation"):
                add(model, f"full_scan enabled -> running {model}")
            if self.enable_toxic_fallback:
                add("toxic", "full_scan enabled -> running toxic fallback")
            return to_run, reasons

        # Fast-block path: attack head is reliable (precision@1=0.907). When FastText
        # is very confident on jailbreak or prompt_injection, the transformer would
        # confirm it anyway — skip it and let the merged FastText score trigger the
        # block decision. Abuse/high_risk heads are NOT fast-blocked (precision@1=0.42).
        ft_jb = ft_scores.get(C.JAILBREAK, 0.0)
        ft_pi = ft_scores.get(C.PROMPT_INJECTION, 0.0)
        if ft_jb >= self.fast_block_threshold or ft_pi >= self.fast_block_threshold:
            reasons.append(
                f"FastText attack score {max(ft_jb, ft_pi):.3f} >= "
                f"fast_block_threshold {self.fast_block_threshold} -> skipping transformer"
            )
            return [], reasons  # no transformer needed; FastText score drives the decision

        # Prompt injection routing.
        if ft_pi >= self.thresholds.route(C.PROMPT_INJECTION):
            add("prompt_injection", "FastText attack head routed to prompt injection detector")
        if self._heuristic_attack(detection_text):
            add("prompt_injection", "Attack heuristic routed to prompt injection detector")
            add("jailbreak", "Attack heuristic routed to jailbreak detector")

        # Jailbreak routing.
        if ft_jb >= self.thresholds.route(C.JAILBREAK):
            add("jailbreak", "FastText attack head routed to jailbreak detector")

        # Moderation routing: any abuse/high-risk label over its route threshold.
        moderation_labels = (
            C.TOXICITY, C.HATE, C.HARASSMENT, C.SEXUAL, C.VIOLENCE,
            C.SELF_HARM,
        )
        for label in moderation_labels:
            if ft_scores.get(label, 0.0) >= self.thresholds.route(label):
                add("moderation", f"FastText routed {label} to moderation model")
                break

        return to_run, reasons

    def _run_model(self, model_key: str, norm: NormalizedText, full_scan: bool) -> dict[str, Any]:
        """Run a confirmer model with windowing; return ModelResult-like dict."""
        model = self.models[model_key]
        windows = build_windows(
            norm.model_text,
            token_budget=self.window_token_budget,
            long_text_threshold=self.long_text_threshold,
            full_scan=full_scan,
        )

        def scorer(text: str) -> dict[str, float]:
            return model.classify(text)["scores"]

        merged, winning = merge_window_scores(scorer, windows)
        window_reasons = []
        for label, win in winning.items():
            if win != "full" and merged.get(label, 0.0) >= self.thresholds.route(label):
                window_reasons.append(f"{win}_window triggered {label} score")
        return {"model_name": model_key, "scores": merged, "window_reasons": window_reasons}

    def route(
        self,
        text: str,
        full_scan: Optional[bool] = None,
        include_raw: bool = False,
    ) -> dict[str, Any]:
        started = time.time()
        full_scan = self.full_scan_default if full_scan is None else full_scan

        norm = normalize(text)
        reasons: list[str] = []
        raw_outputs: dict[str, Any] = {}

        # 1-4. FastText heads.
        if self.fasttext is not None:
            ft_result = self.fasttext.predict(norm.detection_text)
        else:
            ft_result = {"scores": {l: 0.0 for l in C.SCORED_LABELS}, "suggested_models": []}
        ft_scores = ft_result["scores"]
        suggested = ft_result.get("suggested_models", [])
        if include_raw:
            raw_outputs["fasttext"] = ft_result

        # 5. Decide which transformers to run.
        models_to_run, route_reasons = self._select_models(
            ft_scores, suggested, norm.detection_text, full_scan
        )
        reasons.extend(route_reasons)

        # 6. Run transformer confirmers.
        score_sources: list[dict[str, float]] = [ft_scores]
        triggered: list[str] = []
        for model_key in models_to_run:
            result = self._run_model(model_key, norm, full_scan)
            score_sources.append(result["scores"])
            triggered.append(self._display_name(model_key))
            reasons.extend(result.get("window_reasons", []))
            if include_raw:
                raw_outputs[model_key] = result["scores"]

        # Toxic fallback: medium-confidence moderation toxicity/hate, or abuse-head
        # high but moderation low.
        if self.enable_toxic_fallback and "toxic" in self.models and not full_scan:
            merged_so_far = merge_scores(score_sources)
            lo, hi = self.toxic_medium_band
            tox = merged_so_far.get(C.TOXICITY, 0.0)
            hate = merged_so_far.get(C.HATE, 0.0)
            ft_abuse_high = max(ft_scores.get(C.TOXICITY, 0.0), ft_scores.get(C.HATE, 0.0)) >= 0.60
            medium = (lo <= tox <= hi) or (lo <= hate <= hi)
            if (medium or (ft_abuse_high and max(tox, hate) < lo)) and "moderation" in models_to_run:
                result = self._run_model("toxic", norm, full_scan)
                score_sources.append(result["scores"])
                triggered.append(self._display_name("toxic"))
                reasons.append("Toxic fallback invoked on medium-confidence abuse result")

        # 7. Merge.
        merged = merge_scores(score_sources)

        # 8. Decide.
        decision = decide(merged, self.thresholds, fasttext_scores=ft_scores, extra_reasons=reasons)

        skipped = [
            self._display_name(k)
            for k in self.models
            if self._display_name(k) not in triggered
        ]

        latency_ms = round((time.time() - started) * 1000, 2)
        public = {lab: round(merged[lab], 4) for lab in C.PUBLIC_SCORE_LABELS}
        internal = {lab: round(merged[lab], 4) for lab in C.INTERNAL_SCORE_LABELS}

        return {
            "decision": decision["decision"],
            "risk_level": decision["risk_level"],
            "labels": decision["labels"],
            "scores": public,
            "internal_scores": internal,
            "triggered_models": self._triggered_with_fasttext(triggered),
            "skipped_models": skipped,
            "latency_ms": latency_ms,
            "reasons": decision["reasons"],
            "raw_model_outputs": raw_outputs if include_raw else None,
        }

    def _display_name(self, model_key: str) -> str:
        """Return a human-friendly model name (HF id if known, else key)."""
        model = self.models.get(model_key)
        return getattr(model, "name", model_key) if model is not None else model_key

    def _triggered_with_fasttext(self, triggered: list[str]) -> list[str]:
        if self.fasttext is not None and getattr(self.fasttext, "loaded", False):
            return ["fasttext_heads", *triggered]
        return triggered
