"""Build processed splits and FastText training files from raw datasets.

Pipeline:
  1. Read every ``data/raw/*.jsonl`` record.
  2. Normalize text + map external labels to canonical labels.
  3. Merge duplicate normalized text hashes, preserving the union of labels.
  4. Stable hash-based split into train/val/test (80/10/10) so the *same*
     normalized text can never appear in two splits.
  5. Write processed JSONL + per-head FastText files + distribution reports.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

from .. import constants as C
from ..config import repo_root
from ..normalizer import normalize
from .label_mapper import map_labels
from .quality_checks import check_record

RAW_DIR = repo_root() / "data" / "raw"
PROCESSED_DIR = repo_root() / "data" / "processed"
FASTTEXT_DIR = repo_root() / "data" / "fasttext"
REPORTS_DIR = repo_root() / "reports"

# Fraction of the hash space allocated to each split (stable, deterministic).
_VAL_CUTOFF = 80
_TEST_CUTOFF = 90  # [0,80) train, [80,90) val, [90,100) test


def _iter_raw_records() -> Iterator[dict[str, Any]]:
    for path in sorted(RAW_DIR.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _split_for_hash(text_hash: str) -> str:
    bucket = int(text_hash[:8], 16) % 100
    if bucket < _VAL_CUTOFF:
        return "train"
    if bucket < _TEST_CUTOFF:
        return "val"
    return "test"


def _merge_canonical_labels(existing: list[str], incoming: list[str]) -> list[str]:
    """Merge canonical labels while keeping ``safe`` out of mixed-risk records."""
    merged = list(existing)
    for lab in incoming:
        if lab not in merged:
            merged.append(lab)
    if any(lab in C.RISK_LABELS or lab == C.UNKNOWN for lab in merged):
        merged = [lab for lab in merged if lab != C.SAFE]
    return merged or [C.UNKNOWN]


def build_processed_splits() -> dict[str, Any]:
    """Normalize, map, merge duplicates and split all raw records."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    records_by_hash: dict[str, dict] = {}
    source_counts: Counter = Counter()
    skipped: Counter = Counter()
    total = 0

    for raw in _iter_raw_records():
        total += 1
        ok, reason = check_record(raw)
        if not ok:
            skipped[reason or "unknown"] += 1
            continue
        try:
            norm = normalize(raw["text"])
        except ValueError:
            skipped["empty_after_normalize"] += 1
            continue

        canonical = map_labels(raw.get("labels", []), context=norm.detection_text)
        source = raw.get("source", "unknown")
        if norm.text_hash in records_by_hash:
            skipped["duplicate"] += 1
            existing = records_by_hash[norm.text_hash]
            existing["labels"] = _merge_canonical_labels(existing["labels"], canonical)
            for lab in raw.get("labels", []):
                if lab not in existing["raw_labels"]:
                    existing["raw_labels"].append(lab)
            if source not in existing["sources"]:
                existing["sources"].append(source)
            continue

        records_by_hash[norm.text_hash] = {
            "text": norm.model_text,
            "detection_text": norm.detection_text,
            "labels": canonical,
            "source": source,
            "sources": [source],
            "hash": norm.text_hash,
            "raw_labels": list(raw.get("labels", [])),
        }

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    class_counts: dict[str, Counter] = {k: Counter() for k in splits}

    for record in records_by_hash.values():
        split = _split_for_hash(record["hash"])
        splits[split].append(record)
        source_counts[record["source"]] += 1
        for lab in record["labels"]:
            class_counts[split][lab] += 1

    for split, records in splits.items():
        out_path = PROCESSED_DIR / f"all_{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    report = {
        "total_raw_records": total,
        "unique_records": len(records_by_hash),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "class_distribution": {k: dict(v) for k, v in class_counts.items()},
        "source_distribution": dict(source_counts),
        "skipped": dict(skipped),
    }
    (REPORTS_DIR / "split_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


# --------------------------------------------------------------------------- #
# FastText file generation
# --------------------------------------------------------------------------- #
def _fasttext_line(labels: list[str], text: str) -> list[str]:
    """One line per label (multi-label -> duplicated lines)."""
    flat = text.replace("\n", " ").strip()
    return [f"{C.FASTTEXT_LABEL_PREFIX}{lab} {flat}" for lab in labels]


def _head_labels_for(record_labels: list[str], head_labels: tuple[str, ...]) -> list[str]:
    """Project a record's canonical labels onto a head's label set.

    A record contributes to a head if it has a risk label the head covers, OR it
    is safe (safe examples are used in all three heads). Records whose only labels
    are out-of-scope for this head are skipped for that head.
    """
    relevant = [lab for lab in record_labels if lab in head_labels and lab != C.SAFE]
    if relevant:
        return relevant
    if C.SAFE in record_labels and not any(
        lab in C.RISK_LABELS for lab in record_labels
    ):
        return [C.SAFE]
    return []


def _label_cap_for(split: str) -> int:
    """Per-label cap on examples per head (0 = no cap).

    ``SC_MAX_PER_LABEL`` caps every label uniformly so no single class drowns
    out smaller ones. Val/test get 1/8 to preserve the 80/10/10 ratio.

    Recommended values:
      - attack head:    ~20000  (safe 40k, PI 4k, jailbreak 5k → set ~6000)
      - abuse head:     ~25000  (harassment 22k is the bottleneck → set ~22000)
      - high_risk head: ~5000   (self_harm/dangerous_info are tiny)
    A single global value is a compromise; the default 0 applies no cap.

    ``SC_MAX_SAFE_PER_HEAD`` is kept for backwards-compat but SC_MAX_PER_LABEL
    takes precedence when set.
    """
    import os

    n = int(os.environ.get("SC_MAX_PER_LABEL", "0") or "0")
    if n <= 0:
        # Fall back to legacy safe-only cap if set.
        n = int(os.environ.get("SC_MAX_SAFE_PER_HEAD", "0") or "0")
    if n <= 0:
        return 0
    return n if split == "train" else max(n // 8, 1)


def _labels_under_cap(labels: list[str], label_written: Counter, cap: int) -> list[str]:
    """Filter projected labels to those still below the per-label cap."""
    if cap <= 0:
        return labels
    return [lab for lab in labels if label_written[lab] < cap]


def build_fasttext_files() -> dict[str, Any]:
    """Generate per-head FastText train/val/test files from processed splits."""
    FASTTEXT_DIR.mkdir(parents=True, exist_ok=True)
    head_defs = {
        "attack": C.ATTACK_HEAD_LABELS,
        "abuse": C.ABUSE_HEAD_LABELS,
        "high_risk": C.HIGH_RISK_HEAD_LABELS,
    }
    counts: dict[str, dict[str, Counter]] = {
        head: {"train": Counter(), "val": Counter(), "test": Counter()}
        for head in head_defs
    }

    for split in ("train", "val", "test"):
        in_path = PROCESSED_DIR / f"all_{split}.jsonl"
        if not in_path.exists():
            continue
        records = [
            json.loads(line)
            for line in in_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        cap = _label_cap_for(split)
        for head, head_labels in head_defs.items():
            out_path = FASTTEXT_DIR / f"{head}_{split}.txt"
            label_written: Counter = Counter()
            with out_path.open("w", encoding="utf-8") as fh:
                for rec in records:
                    projected = _head_labels_for(rec["labels"], head_labels)
                    if not projected:
                        continue
                    writable = _labels_under_cap(projected, label_written, cap)
                    if not writable:
                        continue
                    for line in _fasttext_line(writable, rec["text"]):
                        fh.write(line + "\n")
                    for lab in writable:
                        counts[head][split][lab] += 1
                        label_written[lab] += 1

    report = {head: {s: dict(c) for s, c in splits.items()} for head, splits in counts.items()}
    (REPORTS_DIR / "fasttext_data_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
