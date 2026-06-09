"""Tests for merged moderation FastText gate evaluation math."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_script():
    root = Path(__file__).resolve().parents[1]
    scripts_dir = root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    path = scripts_dir / "24_eval_fasttext_merged_moderation_gate.py"
    spec = importlib.util.spec_from_file_location("fasttext_merged_gate", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_gate_metrics_counts_false_passes_and_skip_coverage():
    mod = _load_script()
    scored = [
        {"gold": {"unsafe": False}, "scores": {"harmful_content": 0.01, "sexual": 0.02}},
        {"gold": {"unsafe": True}, "scores": {"harmful_content": 0.9, "sexual": 0.01}},
        {"gold": {"unsafe": True}, "scores": {"harmful_content": 0.03, "sexual": 0.02}},
        {"gold": {"unsafe": False}, "scores": {"harmful_content": 0.8, "sexual": 0.01}},
    ]

    metrics = mod._gate_metrics(scored, threshold=0.05)

    assert metrics["passed"] == 2
    assert metrics["routed"] == 2
    assert metrics["false_pass"] == 1
    assert metrics["pass_pct"] == 0.5
    assert metrics["unsafe_miss_rate"] == 0.5
    assert metrics["route_precision"] == 0.5
    assert metrics["route_recall"] == 0.5


def test_per_label_metrics_at_threshold():
    mod = _load_script()
    scored = [
        {
            "gold": {"harmful_content": True, "sexual": False, "unsafe": True},
            "scores": {"harmful_content": 0.7, "sexual": 0.1},
        },
        {
            "gold": {"harmful_content": False, "sexual": True, "unsafe": True},
            "scores": {"harmful_content": 0.1, "sexual": 0.8},
        },
        {
            "gold": {"harmful_content": False, "sexual": False, "unsafe": False},
            "scores": {"harmful_content": 0.6, "sexual": 0.1},
        },
    ]

    metrics = mod._per_label_metrics(scored, threshold=0.5)

    assert metrics["harmful_content"]["confusion"] == {"tp": 1, "fp": 1, "fn": 0, "tn": 1}
    assert metrics["sexual"]["confusion"] == {"tp": 1, "fp": 0, "fn": 0, "tn": 2}
