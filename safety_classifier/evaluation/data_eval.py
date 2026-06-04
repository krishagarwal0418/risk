"""Data distribution + quality report (after scripts/02_prepare_fasttext_data.py).

Reads the processed splits and per-head FastText files and produces
``reports/data_distribution_report.{json,md}`` with global stats, label
distribution, per-head class balance, and a battery of quality checks.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .. import constants as C
from ..config import repo_root
from ..reporting import ReportSection, Status, print_summary, write_report

PROCESSED_DIR = repo_root() / "data" / "processed"
FASTTEXT_DIR = repo_root() / "data" / "fasttext"
REPORTS_DIR = repo_root() / "reports"

WARN_MIN_COUNT = 100  # warn if a label has fewer than this many examples
SPLITS = ("train", "val", "test")
HEAD_DEFS = {
    "attack": C.ATTACK_HEAD_LABELS,
    "abuse": C.ABUSE_HEAD_LABELS,
    "high_risk": C.HIGH_RISK_HEAD_LABELS,
}


def _read_processed(split: str) -> list[dict]:
    path = PROCESSED_DIR / f"all_{split}.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _text_length_stats(records: list[dict]) -> dict[str, float]:
    lengths = sorted(len(r.get("text", "")) for r in records)
    if not lengths:
        return {"avg": 0, "p50": 0, "p95": 0, "max": 0}
    n = len(lengths)
    return {
        "avg": round(sum(lengths) / n, 1),
        "p50": lengths[int(0.50 * (n - 1))],
        "p95": lengths[int(0.95 * (n - 1))],
        "max": lengths[-1],
    }


def _imbalance_ratio(counts: dict[str, int]) -> float:
    nonzero = [v for v in counts.values() if v > 0]
    if not nonzero:
        return 0.0
    return round(max(nonzero) / max(min(nonzero), 1), 2)


def _check_fasttext_files() -> tuple[list[str], list[str]]:
    """Validate FastText files exist and lines are well-formed. Returns (warns, fails)."""
    warns: list[str] = []
    fails: list[str] = []
    for head in HEAD_DEFS:
        for split in SPLITS:
            path = FASTTEXT_DIR / f"{head}_{split}.txt"
            if not path.exists():
                fails.append(f"missing FastText file {head}_{split}.txt")
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
            bad = [i for i, ln in enumerate(lines) if ln.strip() and not ln.startswith("__label__")]
            if bad:
                fails.append(f"{head}_{split}.txt has {len(bad)} malformed lines")
    return warns, fails


def _check_cross_split_leakage() -> list[str]:
    fails: list[str] = []
    seen: dict[str, str] = {}
    for split in SPLITS:
        for rec in _read_processed(split):
            h = rec.get("hash")
            if h in seen and seen[h] != split:
                fails.append(f"hash leakage: {h[:12]} in {seen[h]} and {split}")
                break
            seen[h] = split
    return fails


def build_distribution_report() -> dict[str, Any]:
    by_split = {s: _read_processed(s) for s in SPLITS}
    all_records = [r for recs in by_split.values() for r in recs]
    total = len(all_records)

    # Global label distribution.
    label_counts: Counter = Counter()
    label_by_split: dict[str, Counter] = {s: Counter() for s in SPLITS}
    source_by_label: dict[str, Counter] = defaultdict(Counter)
    unknown_count = 0
    empty_text = 0
    missing_labels = 0
    for split, recs in by_split.items():
        for r in recs:
            labels = r.get("labels", [])
            if not r.get("text", "").strip():
                empty_text += 1
            if not labels:
                missing_labels += 1
            for lab in labels:
                label_counts[lab] += 1
                label_by_split[split][lab] += 1
                source_by_label[lab][r.get("source", "unknown")] += 1
                if lab == C.UNKNOWN:
                    unknown_count += 1

    # Per-head stats.
    heads: dict[str, Any] = {}
    head_warnings: list[str] = []
    head_failures: list[str] = []
    for head, head_labels in HEAD_DEFS.items():
        per_split = {s: Counter() for s in SPLITS}
        for split in SPLITS:
            path = FASTTEXT_DIR / f"{head}_{split}.txt"
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("__label__"):
                    lab = line.split(" ", 1)[0][len("__label__"):]
                    per_split[split][lab] += 1
        totals = Counter()
        for s in SPLITS:
            totals.update(per_split[s])
        for lab in head_labels:
            cnt = totals.get(lab, 0)
            if cnt == 0:
                head_failures.append(
                    f"Cannot train reliable classifier for label {lab} in {head}_head "
                    "because no examples were found."
                )
            elif cnt < WARN_MIN_COUNT:
                head_warnings.append(f"{head}_head label '{lab}' has only {cnt} examples (<{WARN_MIN_COUNT})")
        heads[head] = {
            "labels": list(head_labels),
            "totals": dict(totals),
            "per_split": {s: dict(per_split[s]) for s in SPLITS},
            "imbalance_ratio": _imbalance_ratio(dict(totals)),
        }

    # Quality checks.
    ft_warns, ft_fails = _check_fasttext_files()
    leakage_fails = _check_cross_split_leakage()

    warnings = head_warnings + ft_warns
    failures = head_failures + ft_fails + leakage_fails
    if empty_text:
        failures.append(f"{empty_text} records have empty text")
    if missing_labels:
        failures.append(f"{missing_labels} records have missing labels")
    if unknown_count:
        warnings.append(f"{unknown_count} label occurrences mapped to 'unknown' (reported, not dropped)")
    if total == 0:
        failures.append("No usable processed records found")

    status = Status.FAIL if failures else (Status.WARN if warnings else Status.PASS)

    length_stats = _text_length_stats(all_records)
    summary = {
        "global": {
            "total_usable_rows": total,
            "train_rows": len(by_split["train"]),
            "val_rows": len(by_split["val"]),
            "test_rows": len(by_split["test"]),
            "text_length": length_stats,
            "unknown_label_occurrences": unknown_count,
        },
        "label_distribution": {
            lab: {
                "count": label_counts[lab],
                "pct": round(100 * label_counts[lab] / total, 2) if total else 0,
                "train": label_by_split["train"].get(lab, 0),
                "val": label_by_split["val"].get(lab, 0),
                "test": label_by_split["test"].get(lab, 0),
                "top_sources": dict(source_by_label[lab].most_common(5)),
            }
            for lab in sorted(label_counts)
        },
        "heads": heads,
    }

    # Tables for markdown.
    label_rows = [
        [lab, d["count"], f"{d['pct']}%", d["train"], d["val"], d["test"]]
        for lab, d in summary["label_distribution"].items()
    ]
    head_rows = []
    for head, h in heads.items():
        for lab in h["labels"]:
            head_rows.append([head, lab, h["totals"].get(lab, 0), h["imbalance_ratio"]])

    global_section = ReportSection("Global")
    for k, v in summary["global"].items():
        global_section.add(k, v)

    write_report(
        "data_distribution_report",
        status=status,
        title="Data Distribution Report",
        summary=summary,
        sections=[global_section],
        tables=[
            ("Label Distribution", ["Label", "Count", "Pct", "Train", "Val", "Test"], label_rows),
            ("Per-Head Class Balance", ["Head", "Label", "Count", "Imbalance"], head_rows),
        ],
        warnings=warnings,
        failures=failures,
    )
    print_summary(
        "Data Distribution", status,
        ["Label", "Count", "Pct", "Train", "Val", "Test"], label_rows,
    )
    return summary
