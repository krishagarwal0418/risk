"""Pydantic request/response schemas for the safety classifier."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class SafetyRequest(BaseModel):
    """Single classification request."""

    text: str
    full_scan: bool = False
    include_raw: bool = False
    request_id: Optional[str] = None


class BatchSafetyRequest(BaseModel):
    """Batch classification request."""

    texts: list[str]
    full_scan: bool = False
    include_raw: bool = False


class ModelResult(BaseModel):
    """Result from a single transformer model."""

    model_name: str
    scores: dict[str, float]
    raw: Optional[dict[str, Any]] = None
    latency_ms: float


class FastTextResult(BaseModel):
    """Result from a single FastText head."""

    head_name: str
    scores: dict[str, float]
    suggested_models: list[str] = Field(default_factory=list)
    latency_ms: float


class SafetyResult(BaseModel):
    """Final public classification result."""

    decision: str
    risk_level: str
    labels: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    internal_scores: dict[str, float] = Field(default_factory=dict)
    triggered_models: list[str] = Field(default_factory=list)
    skipped_models: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    reasons: list[str] = Field(default_factory=list)
    raw_model_outputs: Optional[dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Response for the ``/health`` endpoint."""

    status: str
    device: str
    backend: str
    fasttext_loaded: bool
    models_loaded: list[str] = Field(default_factory=list)
