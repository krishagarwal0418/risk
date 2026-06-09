"""Tests for the coarse FastText attack/moderation router script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from safety_classifier import constants as C


def _load_script():
    root = Path(__file__).resolve().parents[1]
    scripts_dir = root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    path = scripts_dir / "21_train_fasttext_router.py"
    spec = importlib.util.spec_from_file_location("fasttext_router_train", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_route_labels_cover_attack_moderation_and_safe():
    mod = _load_script()

    assert mod._route_labels([C.PROMPT_INJECTION]) == ["attack"]
    assert mod._route_labels([C.JAILBREAK, C.HATE]) == ["attack", "moderation"]
    assert mod._route_labels([C.SEXUAL]) == ["moderation"]
    assert mod._route_labels([C.SAFE]) == ["safe"]
    assert mod._route_labels([C.UNKNOWN]) == []


def test_fast_allow_metrics_counts_passed_unsafe_as_false_pass():
    mod = _load_script()
    scored = [
        ({"attack": 0.01, "moderation": 0.01}, {"safe"}),
        ({"attack": 0.80, "moderation": 0.01}, {"attack"}),
        ({"attack": 0.01, "moderation": 0.03}, {"moderation"}),
        ({"attack": 0.01, "moderation": 0.90}, {"safe"}),
    ]

    metrics = mod._fast_allow_metrics(scored, threshold=0.05)

    assert metrics["skipped"] == 2
    assert metrics["false_pass"] == 1
    assert metrics["missed_attack"] == 0
    assert metrics["missed_moderation"] == 1
    assert metrics["skip_pct"] == 0.5
    assert metrics["unsafe_miss_rate"] == 0.5
