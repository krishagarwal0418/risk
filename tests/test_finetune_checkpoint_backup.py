"""Tests for fine-tuning checkpoint backup helpers."""

from __future__ import annotations

from safety_classifier.transformers_layer.finetune import (
    _copy_checkpoint_to_backup,
    _copy_final_model_to_backup,
)


def test_copy_checkpoint_to_backup_and_prune_oldest(tmp_path):
    output = tmp_path / "output"
    backup = tmp_path / "backup"
    for step in (250, 500, 750):
        ckpt = output / f"checkpoint-{step}"
        ckpt.mkdir(parents=True)
        (ckpt / "model.safetensors").write_text(f"step {step}", encoding="utf-8")
        _copy_checkpoint_to_backup(ckpt, backup, backup_limit=2)

    assert not (backup / "checkpoint-250").exists()
    assert (backup / "checkpoint-500" / "model.safetensors").read_text(encoding="utf-8") == "step 500"
    assert (backup / "checkpoint-750" / "model.safetensors").read_text(encoding="utf-8") == "step 750"


def test_copy_final_model_to_backup_skips_checkpoint_dirs(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "config.json").write_text("{}", encoding="utf-8")
    (output / "checkpoint-250").mkdir()
    (output / "checkpoint-250" / "model.safetensors").write_text("checkpoint", encoding="utf-8")

    _copy_final_model_to_backup(output, tmp_path / "backup")

    assert (tmp_path / "backup" / "final_model" / "config.json").exists()
    assert not (tmp_path / "backup" / "final_model" / "checkpoint-250").exists()
