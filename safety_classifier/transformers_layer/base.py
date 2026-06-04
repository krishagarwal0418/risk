"""Base transformer classifier with PyTorch / ONNX Runtime backends.

Subclasses implement :meth:`map_scores` to translate the model's raw
``id2label`` outputs into canonical-label scores. Models are loaded once at
construction and reused for every request.

Label mappings are NEVER hard-coded blindly: the raw ``id2label`` is inspected at
load time and exposed via :attr:`id2label`. Subclasses map known labels and keep
unknown ones in the raw payload rather than guessing.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from .. import constants as C


def resolve_device(requested: str) -> str:
    """Return an available device string, falling back to CPU if needed."""
    if requested == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:  # noqa: BLE001
            pass
        return "cpu"
    return requested


class BaseHFClassifier:
    """Shared loading + inference logic for HF sequence-classification models."""

    #: Default canonical-score template (all zero).
    def _zero_scores(self) -> dict[str, float]:
        return {lab: 0.0 for lab in C.SCORED_LABELS}

    def __init__(
        self,
        model_id_or_path: str,
        backend: str = "pytorch",
        device: str = "cuda",
        max_length: int = 128,
        fp16_on_cuda: bool = True,
        name: Optional[str] = None,
    ) -> None:
        self.model_id = model_id_or_path
        self.name = name or model_id_or_path
        self.backend = backend
        self.device = resolve_device(device)
        self.max_length = max_length
        self.fp16_on_cuda = fp16_on_cuda
        self.id2label: dict[int, str] = {}
        self._tokenizer = None
        self._model = None
        self._load()

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.backend in ("onnx", "onnx_int8"):
            self._load_onnx()
        else:
            self._load_pytorch()
        # Record id2label from the model config for dynamic mapping.
        config = getattr(self._model, "config", None)
        if config is not None and getattr(config, "id2label", None):
            self.id2label = {int(k): v for k, v in config.id2label.items()}

    def _load_pytorch(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        model.eval()
        if self.device == "cuda":
            model = model.to("cuda")
            if self.fp16_on_cuda:
                model = model.half()
        self._model = model

    def _load_onnx(self) -> None:
        from optimum.onnxruntime import ORTModelForSequenceClassification

        provider = (
            "CUDAExecutionProvider" if self.device == "cuda" else "CPUExecutionProvider"
        )
        self._model = ORTModelForSequenceClassification.from_pretrained(
            self.model_id, provider=provider
        )

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def _logits(self, texts: list[str]):
        import torch

        enc = self._tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if self.backend in ("onnx", "onnx_int8"):
            outputs = self._model(**{k: v for k, v in enc.items()})
            return outputs.logits
        if self.device == "cuda":
            enc = {k: v.to("cuda") for k, v in enc.items()}
        with torch.inference_mode():
            outputs = self._model(**enc)
        return outputs.logits

    def _probabilities(self, texts: list[str]) -> list[dict[str, float]]:
        """Return a list of ``raw_label -> probability`` dicts."""
        import torch

        logits = self._logits(texts)
        num_labels = logits.shape[-1]
        # Multi-label heads (single logit or sigmoid-style) vs softmax classifiers:
        # use softmax when >1 class and labels look mutually exclusive; otherwise
        # sigmoid. We default to softmax for >=2 labels (typical for these models).
        if num_labels == 1:
            probs = torch.sigmoid(logits).squeeze(-1)
            return [{self.id2label.get(0, "positive"): float(p)} for p in probs]
        probs = torch.softmax(logits.float(), dim=-1)
        results: list[dict[str, float]] = []
        for row in probs:
            results.append(
                {self.id2label.get(i, str(i)): float(row[i]) for i in range(num_labels)}
            )
        return results

    # ------------------------------------------------------------------ #
    # Mapping (subclass responsibility)
    # ------------------------------------------------------------------ #
    def map_scores(self, raw: dict[str, float]) -> dict[str, float]:  # noqa: D401
        """Map raw ``label->prob`` to canonical scores. Override in subclasses."""
        raise NotImplementedError

    def classify(self, text: str) -> dict[str, Any]:
        return self.classify_batch([text])[0]

    def classify_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        started = time.time()
        raw_list = self._probabilities(texts)
        latency = (time.time() - started) * 1000 / max(len(texts), 1)
        results = []
        for raw in raw_list:
            scores = self._zero_scores()
            mapped = self.map_scores(raw)
            for lab, val in mapped.items():
                if lab in scores:
                    scores[lab] = max(scores[lab], val)
            results.append(
                {
                    "model_name": self.name,
                    "scores": scores,
                    "raw": raw,
                    "latency_ms": round(latency, 3),
                }
            )
        return results
