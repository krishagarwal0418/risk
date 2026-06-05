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
    """Single line with all labels (FastText native multi-label format).

    Writing one line per label causes contradictory OVA gradients: the second
    line for a hate+toxicity text contains "not hate" signal, which collapses
    the model toward always predicting toxicity. Multi-label format avoids this.
    """
    flat = text.replace("\n", " ").strip()
    prefix = " ".join(f"{C.FASTTEXT_LABEL_PREFIX}{lab}" for lab in labels)
    return [f"{prefix} {flat}"]


def _head_labels_for(
    record_labels: list[str],
    head_labels: tuple[str, ...],
    sources: list[str] | None = None,
) -> list[str]:
    """Project a record's canonical labels onto a head's label set.

    A record contributes to a head if it has a risk label the head covers, OR it
    is safe (safe examples are used in all three heads). Records whose only labels
    are out-of-scope for this head are skipped for that head.

    BeaverTails violence examples (violence,aiding_and_abetting,incitement) are
    policy/legal discussion text — the model can't find violent-language features.
    Only text_moderation_410K violence labels (real violent content flagged by
    OpenAI's API) are reliable for the violence classifier.
    """
    relevant = [lab for lab in record_labels if lab in head_labels and lab != C.SAFE]
    # Filter out unreliable violence labels from BeaverTails.
    if C.VIOLENCE in relevant and sources and all(s == "beavertails" for s in sources):
        relevant = [l for l in relevant if l != C.VIOLENCE]
    if relevant:
        return relevant
    if C.SAFE in record_labels and not any(
        lab in C.RISK_LABELS for lab in record_labels
    ):
        return [C.SAFE]
    return []


# Head-specific default caps applied when no env override is set. These make the
# pipeline produce balanced FastText files out of the box — no env vars required.
# Rationale:
#   - attack / abuse: 25k per label keeps safe from drowning out attack/abuse signal
#     while retaining enough data for a strong model.
#   - high_risk: 3k, because its risk classes are tiny (self_harm=965,
#     dangerous_information=2638); a large safe pool would swamp them.
_DEFAULT_HEAD_CAPS: dict[str, int] = {
    "attack": 25000,
    "abuse": 25000,
    "high_risk": 3000,
}


def _label_cap_for(split: str, head: str | None = None) -> int:
    """Per-label cap on examples per head.

    Resolution order (first match wins):
      1. ``SC_MAX_PER_LABEL_<HEAD>``  (e.g. SC_MAX_PER_LABEL_HIGH_RISK)
      2. ``SC_MAX_PER_LABEL``         (global override across all heads)
      3. ``SC_MAX_SAFE_PER_HEAD``     (legacy, backwards-compat)
      4. ``_DEFAULT_HEAD_CAPS[head]`` (sensible per-head default — the common path)

    Set any env var to ``-1`` to explicitly disable capping for that scope.
    Val/test get 1/8 of the train cap to preserve the ~80/10/10 ratio.
    """
    import os

    def _env(key: str) -> int | None:
        raw = os.environ.get(key)
        if raw is None or raw.strip() == "":
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    head_key = f"SC_MAX_PER_LABEL_{head.upper()}" if head else None
    candidates = [
        _env(head_key) if head_key else None,
        _env("SC_MAX_PER_LABEL"),
        _env("SC_MAX_SAFE_PER_HEAD"),
    ]
    n: int | None = next((c for c in candidates if c is not None), None)
    if n is None:
        # No override anywhere -> use the baked-in per-head default.
        n = _DEFAULT_HEAD_CAPS.get(head or "", 25000)
    if n < 0:
        return 0  # explicit opt-out: no cap
    if n == 0:
        return 0
    return n if split == "train" else max(n // 8, 1)


def _labels_under_cap(labels: list[str], label_written: Counter, cap: int) -> list[str]:
    """Filter projected labels to those still below the per-label cap.

    For the abuse head, hate and toxicity often point to the SAME underlying text
    (ToxiGen labels both). We count them against a SHARED budget so a text with
    both labels only consumes one slot from the hate+toxicity pool, keeping the
    effective class balance honest (22k unique abuse texts vs 22k safe).
    """
    if cap <= 0:
        return labels
    # Shared budget for semantically-overlapping abuse labels.
    _SHARED_BUDGET_GROUPS = ((C.HATE, C.TOXICITY),)
    def _group_key(lab: str) -> str:
        for group in _SHARED_BUDGET_GROUPS:
            if lab in group:
                return group[0]  # canonical key for this group
        return lab
    return [lab for lab in labels if label_written[_group_key(lab)] < cap]


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
        for head, head_labels in head_defs.items():
            cap = _label_cap_for(split, head)
            out_path = FASTTEXT_DIR / f"{head}_{split}.txt"
            label_written: Counter = Counter()
            with out_path.open("w", encoding="utf-8") as fh:
                for rec in records:
                    projected = _head_labels_for(rec["labels"], head_labels, rec.get("sources"))
                    if not projected:
                        continue
                    writable = _labels_under_cap(projected, label_written, cap)
                    if not writable:
                        continue
                    for line in _fasttext_line(writable, rec["text"]):
                        fh.write(line + "\n")
                    _SHARED = ((C.HATE, C.TOXICITY),)
                    def _gkey(l: str) -> str:
                        for g in _SHARED:
                            if l in g: return g[0]
                        return l
                    for lab in writable:
                        counts[head][split][lab] += 1
                        label_written[_gkey(lab)] += 1  # shared budget for hate/toxicity

    report = {head: {s: dict(c) for s, c in splits.items()} for head, splits in counts.items()}
    (REPORTS_DIR / "fasttext_data_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
