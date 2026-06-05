"""Tests for the Koala moderation fine-tune data helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from safety_classifier import constants as C
from safety_classifier.transformers_layer.finetune import TASK_LABELS


def _load_script():
    root = Path(__file__).resolve().parents[1]
    scripts_dir = root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    path = scripts_dir / "17_finetune_koala_moderation_best.py"
    spec = importlib.util.spec_from_file_location("koala_best", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_koala_task_labels_match_supported_moderation_surface():
    labels = TASK_LABELS["koala_moderation"]

    assert labels == [
        C.TOXICITY,
        C.HATE,
        C.HARASSMENT,
        C.SEXUAL,
        C.VIOLENCE,
        C.SELF_HARM,
    ]
    assert C.DANGEROUS_INFORMATION not in labels
    assert C.ILLEGAL_ACTIVITY not in labels


def test_clean_record_keeps_supported_labels_and_drops_safe_conflict():
    mod = _load_script()
    row, reason = mod._clean_record(
        {
            "text": "This is targeted harassment and violent abuse.",
            "labels": [C.SAFE, C.HARASSMENT, C.VIOLENCE],
            "source": "text_moderation_410k",
            "hash": "h1",
        },
        split="train",
        min_length=4,
        max_length=1000,
    )

    assert reason == "kept"
    assert row is not None
    assert row["labels"] == [C.HARASSMENT, C.VIOLENCE]
    assert row["original_labels"] == [C.HARASSMENT, C.SAFE, C.VIOLENCE]


def test_clean_record_rejects_unsupported_only_risk():
    mod = _load_script()
    row, reason = mod._clean_record(
        {
            "text": "Describe a risky illegal process in detail.",
            "labels": [C.ILLEGAL_ACTIVITY, C.DANGEROUS_INFORMATION],
            "source": "text_moderation_410k",
        },
        split="train",
        min_length=4,
        max_length=1000,
    )

    assert row is None
    assert reason == "unsupported_risk_only"


def test_build_koala_data_filters_and_balances(tmp_path):
    mod = _load_script()
    processed = tmp_path / "processed"
    processed.mkdir()
    records = {
        "train": [
            {"text": "safe moderation sample one", "labels": [C.SAFE], "source": "xstest", "hash": "s1"},
            {"text": "hate moderation sample", "labels": [C.HATE], "source": "toxigen", "hash": "h1"},
            {"text": "illegal only sample", "labels": [C.ILLEGAL_ACTIVITY], "source": "text_moderation_410k", "hash": "i1"},
            {"text": "attack only sample", "labels": [C.PROMPT_INJECTION], "source": "wildguardmix", "hash": "p1"},
        ],
        "val": [
            {"text": "sexual moderation sample", "labels": [C.SEXUAL], "source": "text_moderation_410k", "hash": "v1"},
            {"text": "safe validation sample", "labels": [C.SAFE], "source": "xstest", "hash": "v2"},
        ],
        "test": [
            {"text": "self harm moderation sample", "labels": [C.SELF_HARM], "source": "wildguardmix", "hash": "t1"},
            {"text": "safe test sample", "labels": [C.SAFE], "source": "xstest", "hash": "t2"},
        ],
    }
    for split, rows in records.items():
        (processed / f"all_{split}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )

    summary = mod.build_koala_data(
        processed_dir=processed,
        output_dir=tmp_path / "koala",
        min_length=4,
        max_length=1000,
        max_per_label=10,
        safe_ratio=1.0,
        safe_cap=10,
        max_train=100,
        max_eval=100,
        seed=13,
    )

    train = [
        json.loads(line)
        for line in (tmp_path / "koala" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    train_labels = {lab for row in train for lab in row["labels"]}

    assert C.HATE in train_labels
    assert C.ILLEGAL_ACTIVITY not in train_labels
    assert C.PROMPT_INJECTION not in train_labels
    assert summary["splits"]["train"]["rows"] == 2
