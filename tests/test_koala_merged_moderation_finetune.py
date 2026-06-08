"""Tests for the merged 2-label Koala moderation fine-tune script."""

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
    path = scripts_dir / "19_finetune_koala_merged_moderation.py"
    spec = importlib.util.spec_from_file_location("koala_merged", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_merged_task_has_two_scored_outputs_and_no_benign_class():
    assert TASK_LABELS["koala_merged_moderation"] == ["harmful_content", C.SEXUAL]


def test_merge_labels_groups_abuse_violence_self_harm_and_keeps_sexual_separate():
    mod = _load_script()

    assert mod._merge_labels({C.HATE}) == ["harmful_content"]
    assert mod._merge_labels({C.VIOLENCE, C.SELF_HARM}) == ["harmful_content"]
    assert mod._merge_labels({C.SEXUAL}) == [C.SEXUAL]
    assert mod._merge_labels({C.SEXUAL, C.VIOLENCE}) == ["harmful_content", C.SEXUAL]
    assert mod._merge_labels({C.PROMPT_INJECTION}) == []


def test_clean_record_safe_is_empty_label_vector():
    mod = _load_script()
    row, reason = mod._clean_record(
        {
            "text": "a harmless moderation message",
            "labels": [C.SAFE],
            "source": "text_moderation_410k",
            "hash": "s1",
        },
        split="train",
        min_length=4,
        max_length=1000,
    )

    assert reason == "kept"
    assert row is not None
    assert row["labels"] == []


def test_build_merged_data_filters_unsupported_and_balances(tmp_path):
    mod = _load_script()
    processed = tmp_path / "processed"
    processed.mkdir()
    records = {
        "train": [
            {"text": "safe sample", "labels": [C.SAFE], "source": "xstest", "hash": "s1"},
            {"text": "hate sample", "labels": [C.HATE], "source": "toxigen", "hash": "h1"},
            {"text": "sexual sample", "labels": [C.SEXUAL], "source": "text_moderation_410k", "hash": "x1"},
            {"text": "mixed sample", "labels": [C.SEXUAL, C.VIOLENCE], "source": "wildguardmix", "hash": "m1"},
            {"text": "illegal sample", "labels": [C.ILLEGAL_ACTIVITY], "source": "text_moderation_410k", "hash": "i1"},
        ],
        "val": [
            {"text": "harassment val", "labels": [C.HARASSMENT], "source": "text_moderation_410k", "hash": "v1"},
            {"text": "safe val", "labels": [C.SAFE], "source": "xstest", "hash": "v2"},
        ],
        "test": [
            {"text": "self harm test", "labels": [C.SELF_HARM], "source": "wildguardmix", "hash": "t1"},
            {"text": "safe test", "labels": [C.SAFE], "source": "xstest", "hash": "t2"},
        ],
    }
    for split, rows in records.items():
        (processed / f"all_{split}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )

    summary = mod.build_merged_data(
        processed_dir=processed,
        output_dir=tmp_path / "merged",
        min_length=4,
        max_length=1000,
        max_harmful=10,
        max_sexual=10,
        safe_ratio=1.0,
        safe_cap=10,
        min_sexual_target=0,
        max_train=100,
        max_eval=100,
        seed=13,
    )
    train = [
        json.loads(line)
        for line in (tmp_path / "merged" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    labels = [tuple(row["labels"]) for row in train]

    assert ("harmful_content",) in labels
    assert (C.SEXUAL,) in labels
    assert ("harmful_content", C.SEXUAL) in labels
    assert all(C.ILLEGAL_ACTIVITY not in row["labels"] for row in train)
    assert summary["splits"]["train"]["labels"]["harmful_content"] == 2
    assert summary["splits"]["train"]["labels"][C.SEXUAL] == 2
