#!/usr/bin/env python3
"""Validate and prepare fine-tuning data: dedup, balance, filter low-quality.

Usage:
    python scripts/09_validate_finetuning_data.py
        --input data/processed/all_train.jsonl
        --output data/finetuning_train.jsonl
        --min-length 10
        --max-length 512
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from safety_classifier import constants as C


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate & prepare fine-tuning data")
    parser.add_argument("--input", default="data/processed/all_train.jsonl")
    parser.add_argument("--output", default="data/finetuning_train.jsonl")
    parser.add_argument("--min-length", type=int, default=10, help="Min text length")
    parser.add_argument("--max-length", type=int, default=512, help="Max text length")
    parser.add_argument("--min-label-count", type=int, default=50,
                        help="Minimum examples per label for fine-tuning")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        print(f"✗ Input file not found: {in_path}")
        return

    print("=" * 70)
    print("Fine-tuning Data Validation & Preparation")
    print("=" * 70)
    print(f"Input: {in_path}")
    print(f"Output: {out_path}")
    print()

    # Load all records
    records = []
    with in_path.open() as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  ⚠️  Skipped malformed JSON at line {i+1}")
                continue

    print(f"Loaded {len(records)} records")
    print()

    # Quality checks & filtering
    filtered = []
    issues = defaultdict(int)

    for i, rec in enumerate(records):
        text = rec.get("text", "").strip()
        labels = rec.get("labels", [])

        # Check 1: Empty text
        if not text:
            issues["empty_text"] += 1
            continue

        # Check 2: Text length
        if len(text) < args.min_length:
            issues["too_short"] += 1
            continue
        if len(text) > args.max_length:
            issues["too_long"] += 1
            continue

        # Check 3: Empty labels
        if not labels:
            issues["no_labels"] += 1
            continue

        # Check 4: Only "unknown" label (low signal)
        if labels == [C.UNKNOWN]:
            issues["only_unknown"] += 1
            continue

        # Check 5: Conflicting labels (safe + risk)
        has_safe = C.SAFE in labels
        has_risk = any(lab in C.RISK_LABELS for lab in labels)
        if has_safe and has_risk:
            # Mixed label = low confidence, skip
            issues["mixed_safe_risk"] += 1
            continue

        # Check 6: Duplicate text hash
        if "hash" in rec:
            # Dedup by hash
            if any(r.get("hash") == rec.get("hash") for r in filtered):
                issues["duplicate"] += 1
                continue

        filtered.append(rec)

    print("Quality Checks:")
    for issue, count in sorted(issues.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(records)
        print(f"  {issue:<25} {count:>6} ({pct:>5.1f}%)")
    print()
    print(f"Retained: {len(filtered)} records ({100*len(filtered)/len(records):.1f}%)")
    print()

    # Label distribution
    label_counts = Counter()
    for rec in filtered:
        for lab in rec.get("labels", []):
            label_counts[lab] += 1

    print("Label Distribution (fine-tuning set):")
    warnings = []
    for lab, count in label_counts.most_common():
        status = "✓" if count >= args.min_label_count else "✗"
        print(f"  {status} {lab:<30} {count:>6}")
        if count < args.min_label_count:
            warnings.append(f"{lab} has only {count} examples (need {args.min_label_count})")
    print()

    if warnings:
        print("⚠️  Warnings:")
        for w in warnings:
            print(f"  - {w}")
        print()

    # Task-specific validation
    print("Task Coverage:")

    attack_labels = {C.PROMPT_INJECTION, C.JAILBREAK}
    attack_count = sum(1 for r in filtered
                       if any(lab in attack_labels for lab in r.get("labels", [])))
    print(f"  Attack (PI+JB): {attack_count} examples")

    moderation_labels = {C.TOXICITY, C.HATE, C.HARASSMENT, C.SEXUAL,
                         C.VIOLENCE, C.SELF_HARM, C.DANGEROUS_INFORMATION,
                         C.ILLEGAL_ACTIVITY}
    mod_count = sum(1 for r in filtered
                    if any(lab in moderation_labels for lab in r.get("labels", [])))
    print(f"  Moderation:    {mod_count} examples")
    print()

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for rec in filtered:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✓ Wrote {len(filtered)} records to {out_path}")
    print()

    # Summary
    print("=" * 70)
    if warnings:
        print("Status: ⚠️  WARN (some labels sparse, may impact fine-tuning)")
    else:
        print("Status: ✓ PASS (data ready for fine-tuning)")
    print("=" * 70)


if __name__ == "__main__":
    main()
