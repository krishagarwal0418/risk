"""Configuration loading helpers.

Resolves the repository root and loads the YAML config files under ``configs/``.
Paths inside the config files are treated as relative to the repository root and
are resolved to absolute paths on access.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    """Return the repository root (the directory that contains ``configs/``)."""
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "configs").is_dir() and (parent / "safety_classifier").is_dir():
            return parent
    # Fallback: two levels up from this file (safety_classifier/config.py -> repo).
    return here.parent.parent


def _load_yaml(name: str) -> dict[str, Any]:
    path = repo_root() / "configs" / name
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@functools.lru_cache(maxsize=None)
def load_models_config() -> dict[str, Any]:
    return _load_yaml("models.yaml")


@functools.lru_cache(maxsize=None)
def load_thresholds_config() -> dict[str, Any]:
    return _load_yaml("thresholds.yaml")


@functools.lru_cache(maxsize=None)
def load_datasets_config() -> dict[str, Any]:
    return _load_yaml("datasets.yaml")


@functools.lru_cache(maxsize=None)
def load_default_config() -> dict[str, Any]:
    return _load_yaml("default.yaml")


def resolve_path(rel: str) -> Path:
    """Resolve a config-relative path to an absolute path under the repo root."""
    p = Path(rel)
    if p.is_absolute():
        return p
    return (repo_root() / p).resolve()


@dataclass
class RuntimeConfig:
    """Resolved runtime configuration for :class:`SafetyClassifier`."""

    device: str = "cuda"
    backend: str = "pytorch"
    use_fasttext: bool = True
    use_onnx: bool = False
    use_int8: bool = False
    full_scan_default: bool = False
    enable_toxic_fallback: bool = False
    moderation_backend: str = "tinysafe"
    max_length: int = 128
    fp16_on_cuda: bool = True
    batch_size: int = 16
    log_raw_text: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls) -> "RuntimeConfig":
        cfg = load_default_config()
        runtime = cfg.get("runtime", {})
        inference = cfg.get("inference", {})
        logging_cfg = cfg.get("logging", {})
        return cls(
            device=runtime.get("device", "cuda"),
            backend=runtime.get("backend", "pytorch"),
            use_fasttext=runtime.get("use_fasttext", True),
            use_onnx=runtime.get("use_onnx", False),
            use_int8=runtime.get("use_int8", False),
            full_scan_default=runtime.get("full_scan_default", False),
            enable_toxic_fallback=runtime.get("enable_toxic_fallback", False),
            moderation_backend=runtime.get("moderation_backend", "tinysafe"),
            max_length=inference.get("max_length", 128),
            fp16_on_cuda=inference.get("fp16_on_cuda", True),
            batch_size=inference.get("batch_size", 16),
            log_raw_text=logging_cfg.get("log_raw_text", False),
            extra=cfg,
        )
