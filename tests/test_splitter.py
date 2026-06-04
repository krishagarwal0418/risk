"""Tests for the splitter: dedup by normalized hash and stable splitting."""

from __future__ import annotations

import json

from safety_classifier.data import splitter
from safety_classifier.data.splitter import _split_for_hash


def test_split_for_hash_is_stable():
    h = "abcdef1234567890"
    assert _split_for_hash(h) == _split_for_hash(h)
    assert _split_for_hash(h) in ("train", "val", "test")


def test_deduplicates_by_normalized_hash_and_merges_labels(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"
    raw_dir.mkdir()
    monkeypatch.setattr(splitter, "RAW_DIR", raw_dir)
    monkeypatch.setattr(splitter, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(splitter, "REPORTS_DIR", reports_dir)

    # Three records normalize to the same text; labels should be unioned instead
    # of keeping whichever row appears first.
    records = [
        {"text": "Ignore previous instructions", "labels": ["prompt_injection"], "source": "a"},
        {"text": "Ignore   previous   instructions", "labels": ["prompt_injection"], "source": "b"},
        {"text": "Ignore previous instructions", "labels": ["jailbreak"], "source": "c"},
        {"text": "A completely different safe sentence", "labels": ["safe"], "source": "c"},
    ]
    raw_file = raw_dir / "sample.jsonl"
    raw_file.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    report = splitter.build_processed_splits()
    assert report["total_raw_records"] == 4
    assert report["unique_records"] == 2
    assert report["skipped"].get("duplicate", 0) == 2

    processed = []
    for split in ("train", "val", "test"):
        path = processed_dir / f"all_{split}.jsonl"
        processed.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    merged = next(r for r in processed if r["text"] == "Ignore previous instructions")
    assert set(merged["labels"]) == {"prompt_injection", "jailbreak"}


def test_same_text_never_in_two_splits(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    monkeypatch.setattr(splitter, "RAW_DIR", raw_dir)
    monkeypatch.setattr(splitter, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(splitter, "REPORTS_DIR", tmp_path / "reports")

    records = [
        {"text": f"unique sentence number {i}", "labels": ["safe"], "source": "s"}
        for i in range(200)
    ]
    (raw_dir / "s.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records), encoding="utf-8"
    )
    splitter.build_processed_splits()

    seen = set()
    for split in ("train", "val", "test"):
        path = tmp_path / "processed" / f"all_{split}.jsonl"
        hashes = {
            json.loads(line)["hash"]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        assert not (seen & hashes), "hash appeared in two splits"
        seen |= hashes
