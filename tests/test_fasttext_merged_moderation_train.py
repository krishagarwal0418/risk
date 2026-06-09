"""Tests for the merged moderation FastText training script helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from safety_classifier import constants as C
from safety_classifier.fasttext_layer.trainer import FastTextHyperParams


def _load_script():
    root = Path(__file__).resolve().parents[1]
    scripts_dir = root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    path = scripts_dir / "23_train_fasttext_merged_moderation.py"
    spec = importlib.util.spec_from_file_location("fasttext_merged_moderation", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_labels_for_empty_vector_is_safe():
    mod = _load_script()

    assert mod._labels_for({"labels": []}) == [C.SAFE]
    assert mod._labels_for({"labels": ["harmful_content", C.SEXUAL]}) == [
        "harmful_content",
        C.SEXUAL,
    ]


def test_fasttext_line_writes_native_multilabel_format():
    mod = _load_script()

    line = mod._fasttext_line(["harmful_content", C.SEXUAL], "hello\nworld")

    assert line == "__label__harmful_content __label__sexual hello world"


def test_strong_fasttext_params_include_bucket_verbose_and_thread():
    params = FastTextHyperParams(
        epoch=50,
        lr=0.4,
        dim=200,
        wordNgrams=4,
        bucket=5_000_000,
        thread=8,
        verbose=2,
    )

    assert params.to_train_kwargs()["epoch"] == 50
    assert params.to_train_kwargs()["dim"] == 200
    assert params.to_train_kwargs()["wordNgrams"] == 4
    assert params.to_train_kwargs()["bucket"] == 5_000_000
    assert params.to_train_kwargs()["thread"] == 8
    assert params.to_train_kwargs()["verbose"] == 2
