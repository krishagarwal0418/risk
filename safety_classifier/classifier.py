"""The main :class:`SafetyClassifier` facade.

Loads the FastText router and the transformer confirmer models once, then routes
every request through :class:`routing.router.Router`. Models are never reloaded
per request. Raw user text is never logged.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Optional

from . import constants as C
from .config import RuntimeConfig, load_models_config, resolve_path
from .fasttext_layer.predictor import FastTextSafetyRouter
from .routing.router import Router
from .routing.thresholds import Thresholds
from .transformers_layer.base import resolve_device
from .transformers_layer.jailbreak import JailbreakClassifier
from .transformers_layer.moderation import ModerationClassifier
from .transformers_layer.prompt_injection import PromptInjectionClassifier
from .transformers_layer.toxic_fallback import ToxicFallbackClassifier


def _model_source(entry: dict[str, Any], backend: str) -> str:
    """Resolve the path/id to load for a model given the backend.

    Priority:
      1. Fine-tuned model weights (if present)
      2. ONNX (FP32 preferred; INT8 discouraged due to quantization artifacts)
      3. Local PyTorch weights
      4. HuggingFace model ID (auto-download)
    """
    # Check for fine-tuned weights first (highest priority).
    finetuned_key = entry.get("finetuned_path")
    if finetuned_key:
        p = resolve_path(finetuned_key)
        if p.exists():
            return str(p)

    if backend == "onnx_int8":
        p = resolve_path(entry.get("int8_path", ""))
        if p.exists():
            return str(p)
    if backend in ("onnx", "onnx_int8"):
        p = resolve_path(entry.get("onnx_path", ""))
        if p.exists():
            return str(p)
    # Prefer a local PyTorch download if present, else the HF id.
    local = resolve_path(entry.get("local_path", ""))
    if local.exists():
        return str(local)
    return entry["hf_name"]


class SafetyClassifier:
    """High-level safety classifier facade."""

    def __init__(
        self,
        device: str = "cuda",
        backend: str = "pytorch",
        use_fasttext: bool = True,
        use_onnx: bool = False,
        use_int8: bool = False,
        full_scan_default: bool = False,
        enable_toxic_fallback: bool = False,
        moderation_backend: str = "moderationbert",
        lazy: bool = False,
    ) -> None:
        cfg = RuntimeConfig.from_yaml()
        self.device = resolve_device(device)
        # Backend precedence: explicit flags override the backend string.
        if use_int8:
            backend = "onnx_int8"
        elif use_onnx:
            backend = "onnx"
        self.backend = backend
        self.use_fasttext = use_fasttext
        self.full_scan_default = full_scan_default
        self.enable_toxic_fallback = enable_toxic_fallback
        self.moderation_backend = moderation_backend
        self._cfg = cfg

        self.fasttext: Optional[FastTextSafetyRouter] = None
        self.models: dict[str, Any] = {}
        self.router: Optional[Router] = None
        if not lazy:
            self._build()

    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        models_cfg = load_models_config().get("transformers", {})
        thresholds = Thresholds()

        if self.use_fasttext:
            try:
                self.fasttext = FastTextSafetyRouter(
                    route_threshold=min(
                        thresholds.route(C.PROMPT_INJECTION),
                        thresholds.route(C.SELF_HARM),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"FastText router unavailable: {exc}", RuntimeWarning)
                self.fasttext = None

        common = dict(
            backend=self.backend,
            device=self.device,
            max_length=self._cfg.max_length,
            fp16_on_cuda=self._cfg.fp16_on_cuda,
        )

        self.models = {}
        self._load_model(
            "prompt_injection", PromptInjectionClassifier, models_cfg, common
        )
        self._load_model("jailbreak", JailbreakClassifier, models_cfg, common)

        mod_key = (
            "moderation_fallback"
            if self.moderation_backend == "oxyapi"
            else "moderation_primary"
        )
        self._load_model("moderation", ModerationClassifier, models_cfg, common, cfg_key=mod_key)

        if self.enable_toxic_fallback:
            self._load_model(
                "toxic", ToxicFallbackClassifier, models_cfg, common,
                cfg_key="toxic_fallback", backend_override="pytorch",
            )

        self.router = Router(
            fasttext_router=self.fasttext,
            models=self.models,
            thresholds=thresholds,
            enable_toxic_fallback=self.enable_toxic_fallback,
            full_scan_default=self.full_scan_default,
        )
        self._print_summary()

    def _load_model(
        self,
        model_key: str,
        cls,
        models_cfg: dict,
        common: dict,
        cfg_key: Optional[str] = None,
        backend_override: Optional[str] = None,
    ) -> None:
        entry = models_cfg.get(cfg_key or model_key)
        if not entry:
            warnings.warn(f"No config for model '{model_key}'", RuntimeWarning)
            return
        backend = backend_override or common["backend"]
        source = _model_source(entry, backend)
        try:
            kwargs = {**common, "backend": backend, "name": entry["hf_name"]}
            self.models[model_key] = cls(source, **kwargs)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Could not load model '{model_key}' ({entry['hf_name']}): {exc}. "
                "It will be skipped at routing time.",
                RuntimeWarning,
            )

    def _print_summary(self) -> None:
        loaded = [m.name for m in self.models.values()]
        ft = "loaded" if (self.fasttext and self.fasttext.loaded) else "missing"
        calib = (
            "calibrated"
            if (self.router and getattr(self.router.thresholds, "calibration_loaded", False))
            else "config defaults"
        )
        print("=" * 60)
        print("SafetyClassifier ready")
        print(f"  device:     {self.device}")
        print(f"  backend:    {self.backend}")
        print(f"  fasttext:   {ft}")
        print(f"  thresholds: {calib}")
        print(f"  models:     {loaded or '(none loaded)'}")
        print("=" * 60)

    # ------------------------------------------------------------------ #
    @property
    def loaded_models(self) -> list[str]:
        names = [m.name for m in self.models.values()]
        if self.fasttext and self.fasttext.loaded:
            names = ["fasttext_heads", *names]
        return names

    def classify(
        self,
        text: str,
        full_scan: Optional[bool] = None,
        include_raw: bool = False,
    ) -> dict[str, Any]:
        if self.router is None:
            self._build()
        assert self.router is not None
        return self.router.route(text, full_scan=full_scan, include_raw=include_raw)

    def classify_batch(
        self,
        texts: list[str],
        full_scan: Optional[bool] = None,
        include_raw: bool = False,
    ) -> list[dict[str, Any]]:
        return [self.classify(t, full_scan=full_scan, include_raw=include_raw) for t in texts]
