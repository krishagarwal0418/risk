"""FastAPI service exposing the SafetyClassifier.

Run with::

    uvicorn safety_classifier.api:app --host 0.0.0.0 --port 8000

Runtime options are read from environment variables (so they survive the uvicorn
reload boundary):

    SC_DEVICE    cuda|cpu             (default cuda)
    SC_BACKEND   pytorch|onnx|onnx_int8 (default pytorch)
    SC_TOXIC_FALLBACK  0|1            (default 0)
    SC_MODERATION  moderationbert|tinysafe|oxyapi (default moderationbert)
"""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI

from .classifier import SafetyClassifier
from .schemas import (
    BatchSafetyRequest,
    HealthResponse,
    SafetyRequest,
    SafetyResult,
)

app = FastAPI(title="Safety Classifier", version="0.1.0")


@lru_cache(maxsize=1)
def get_classifier() -> SafetyClassifier:
    return SafetyClassifier(
        device=os.environ.get("SC_DEVICE", "cuda"),
        backend=os.environ.get("SC_BACKEND", "pytorch"),
        enable_toxic_fallback=os.environ.get("SC_TOXIC_FALLBACK", "0") == "1",
        moderation_backend=os.environ.get("SC_MODERATION", "moderationbert"),
    )


@app.on_event("startup")
def _warmup() -> None:
    # Load models once at startup so the first request is not slow.
    get_classifier()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    clf = get_classifier()
    return HealthResponse(
        status="ok",
        device=clf.device,
        backend=clf.backend,
        fasttext_loaded=bool(clf.fasttext and clf.fasttext.loaded),
        models_loaded=clf.loaded_models,
    )


@app.post("/classify", response_model=SafetyResult)
def classify(req: SafetyRequest) -> SafetyResult:
    clf = get_classifier()
    result = clf.classify(req.text, full_scan=req.full_scan, include_raw=req.include_raw)
    return SafetyResult(**result)


@app.post("/classify/batch", response_model=list[SafetyResult])
def classify_batch(req: BatchSafetyRequest) -> list[SafetyResult]:
    clf = get_classifier()
    results = clf.classify_batch(
        req.texts, full_scan=req.full_scan, include_raw=req.include_raw
    )
    return [SafetyResult(**r) for r in results]
