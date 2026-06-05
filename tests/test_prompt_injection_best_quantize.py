"""Unit tests for prompt-injection quantization scoring helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_script():
    root = Path(__file__).resolve().parents[1]
    scripts_dir = root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    path = scripts_dir / "18_quantize_prompt_injection_best.py"
    spec = importlib.util.spec_from_file_location("pi_quant", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_metrics_at_counts_confusion_matrix():
    mod = _load_script()
    metrics = mod.metrics_at([1, 1, 0, 0], [0.9, 0.2, 0.7, 0.1], 0.5)

    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tn"] == 1


def test_recommendation_picks_smallest_candidate_within_tolerance():
    mod = _load_script()
    results = {
        "candidates": {
            "large_good": {
                "size_bytes": 200,
                "comparison_to_pytorch": {"best_f1_drop": 0.001, "pr_auc_drop": 0.001},
            },
            "small_good": {
                "size_bytes": 100,
                "comparison_to_pytorch": {"best_f1_drop": 0.002, "pr_auc_drop": 0.0},
            },
            "small_bad": {
                "size_bytes": 50,
                "comparison_to_pytorch": {"best_f1_drop": 0.01, "pr_auc_drop": 0.0},
            },
        }
    }

    assert mod.choose_recommendation(results, max_f1_drop=0.002, max_pr_auc_drop=0.002) == "small_good"


def test_recommendation_falls_back_to_fp32_when_all_quantized_drop_too_much():
    mod = _load_script()
    results = {
        "candidates": {
            "tiny": {
                "size_bytes": 1,
                "comparison_to_pytorch": {"best_f1_drop": 0.01, "pr_auc_drop": 0.01},
            }
        }
    }

    assert mod.choose_recommendation(results, max_f1_drop=0.002, max_pr_auc_drop=0.002) == "fp32_onnx"


def test_auto_weight_resolution_prefers_best_file(tmp_path, monkeypatch):
    mod = _load_script()
    model_dir = tmp_path / "models" / "fine" / "prompt"
    model_dir.mkdir(parents=True)
    stale = model_dir / "model.safetensors"
    best = model_dir / "model_best.safetensors"
    stale.write_bytes(b"small")
    best.write_bytes(b"larger-best")

    monkeypatch.setattr(mod, "_is_deberta_prompt_weights", lambda p: p.name.endswith(".safetensors"))

    assert mod.resolve_weights_path(tmp_path, "auto") == best


def test_missing_weight_error_lists_candidates(tmp_path, monkeypatch):
    mod = _load_script()
    model_dir = tmp_path / "models" / "fine" / "prompt"
    model_dir.mkdir(parents=True)
    best = model_dir / "model_best.safetensors"
    best.write_bytes(b"larger-best")
    monkeypatch.setattr(mod, "_is_deberta_prompt_weights", lambda p: p == best)

    try:
        mod.resolve_weights_path(tmp_path, "missing/model_best.safetensors")
    except FileNotFoundError as exc:
        msg = str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")

    assert "Weights file not found" in msg
    assert "model_best.safetensors" in msg
    assert "--weights auto" in msg
