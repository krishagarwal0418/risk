#!/usr/bin/env python3
"""Build a balanced, quality-filtered fine-tuning dataset from processed splits.

The processed training set is ~73% safe (409k safe vs e.g. 965 self_harm). Fine-
tuning a transformer on that distribution teaches it to predict "safe" for almost
everything. This script fixes that by:

  1. Quality-filtering (length bounds, malformed/empty labels, conflicting labels).
  2. Deduplicating by text hash (set-based, O(n)).
  3. Keeping ALL risk examples (they are rare and valuable).
  4. Down-sampling the dominant "safe" class to a sane cap so it does not swamp
     the risk signal — class weights in the trainer handle the residual imbalance.

Output: data/finetuning_train.jsonl (used by every 09x fine-tune script).

Usage:
    python scripts/09_validate_finetuning_data.py \
        --input data/processed/all_train.jsonl \
        --output data/finetuning_train.jsonl \
        --safe-cap 60000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from safety_classifier import constants as C

_RISK = set(C.RISK_LABELS)


def _quality_ok(text: str, labels: list[str], min_len: int, max_len: int) -> str | None:
    """Return a rejection reason, or None if the record passes all checks."""
    if not text:
        return "empty_text"
    if len(text) < min_len:
        return "too_short"
    if len(text) > max_len:
        return "too_long"
    if not labels:
        return "no_labels"
    if labels == [C.UNKNOWN]:
        return "only_unknown"
    has_safe = C.SAFE in labels
    has_risk = any(lab in _RISK for lab in labels)
    if has_safe and has_risk:
        return "mixed_safe_risk"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build balanced fine-tuning data")
    parser.add_argument("--input", default="data/processed/all_train.jsonl")
    parser.add_argument("--output", default="data/finetuning_train.jsonl")
    parser.add_argument("--min-length", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=2000)
    parser.add_argument("--min-label-count", type=int, default=50,
                        help="Warn if a label has fewer than this many examples")
    parser.add_argument("--safe-cap", type=int, default=60000,
                        help="Max number of safe-only examples to keep (0 = keep all)")
    parser.add_argument("--attack-output", default="data/finetuning_attack.jsonl",
                        help="Attack-focused subset for the PI/JB fine-tunes")
    parser.add_argument("--attack-neg-ratio", type=int, default=6,
                        help="Hard negatives per attack positive in the attack file")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        print(f"✗ Input file not found: {in_path}")
        sys.exit(1)

    print("=" * 70)
    print("Fine-tuning Data: Validate + Balance")
    print("=" * 70)
    print(f"Input:    {in_path}")
    print(f"Output:   {out_path}")
    print(f"Safe cap: {args.safe_cap or 'none'}")
    print()

    rng = random.Random(args.seed)
    issues: Counter = Counter()
    seen_hashes: set[str] = set()
    risk_records: list[dict] = []
    safe_records: list[dict] = []
    total = 0

    with in_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                issues["malformed_json"] += 1
                continue

            text = (rec.get("text") or "").strip()
            labels = rec.get("labels", [])

            reason = _quality_ok(text, labels, args.min_length, args.max_length)
            if reason:
                issues[reason] += 1
                continue

            # O(1) dedup by hash (fall back to text if hash missing).
            key = rec.get("hash") or text
            if key in seen_hashes:
                issues["duplicate"] += 1
                continue
            seen_hashes.add(key)

            if any(lab in _RISK for lab in labels):
                risk_records.append(rec)
            else:
                safe_records.append(rec)

    print(f"Loaded {total} records")
    print()
    print("Filtered out:")
    for reason, count in issues.most_common():
        print(f"  {reason:<22} {count:>7} ({100 * count / max(total, 1):>5.1f}%)")
    print()

    # Balance: keep all risk, down-sample safe.
    kept_safe = safe_records
    if args.safe_cap and len(safe_records) > args.safe_cap:
        rng.shuffle(safe_records)
        kept_safe = safe_records[: args.safe_cap]
        print(f"Down-sampled safe: {len(safe_records)} -> {len(kept_safe)}")

    final = risk_records + kept_safe
    rng.shuffle(final)

    print(f"Risk examples kept: {len(risk_records)}")
    print(f"Safe examples kept: {len(kept_safe)}")
    print(f"Total fine-tuning examples: {len(final)}")
    print()

    # Per-label distribution of the final set.
    label_counts: Counter = Counter()
    for rec in final:
        for lab in rec.get("labels", []):
            label_counts[lab] += 1

    print("Final label distribution:")
    warnings: list[str] = []
    for lab, count in label_counts.most_common():
        flag = "✓" if count >= args.min_label_count else "✗"
        print(f"  {flag} {lab:<24} {count:>7}")
        if lab in _RISK and count < args.min_label_count:
            warnings.append(f"{lab}={count} (<{args.min_label_count})")
    print()

    # Task coverage sanity.
    attack_pos = sum(
        1 for r in final
        if any(l in (C.PROMPT_INJECTION, C.JAILBREAK) for l in r.get("labels", []))
    )
    mod_pos = sum(
        1 for r in final
        if any(l in (C.TOXICITY, C.HATE, C.HARASSMENT, C.SEXUAL, C.VIOLENCE,
                     C.SELF_HARM, C.DANGEROUS_INFORMATION, C.ILLEGAL_ACTIVITY)
               for l in r.get("labels", []))
    )
    print(f"Attack positives (PI+JB): {attack_pos}")
    print(f"Moderation positives:     {mod_pos}")
    print()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in final:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"✓ Wrote {len(final)} records to {out_path}  (moderation fine-tunes)")

    # ---- Attack-focused subset (for the PI/JB fine-tunes) -------------------
    # The attack task has only ~{attack_pos} positives in {len(final)} rows, so
    # training the PI/JB models on the full file is ~6x slower for no gain. Keep
    # ALL attack positives + a bounded sample of hard negatives (a mix of safe
    # and other-risk text, which teaches "toxic-but-not-injection != injection").
    attack_labels = (C.PROMPT_INJECTION, C.JAILBREAK)
    pos_rows = [r for r in final if any(l in attack_labels for l in r.get("labels", []))]
    neg_rows = [r for r in final if not any(l in attack_labels for l in r.get("labels", []))]
    rng.shuffle(neg_rows)
    neg_keep = min(len(neg_rows), max(len(pos_rows) * args.attack_neg_ratio, 1))
    attack_set = pos_rows + neg_rows[:neg_keep]
    rng.shuffle(attack_set)

    attack_path = Path(args.attack_output)
    with attack_path.open("w", encoding="utf-8") as f:
        for rec in attack_set:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"✓ Wrote {len(attack_set)} records to {attack_path}  "
          f"({len(pos_rows)} attack-pos + {neg_keep} hard-neg, attack fine-tunes)")

    print("=" * 70)
    if warnings:
        print("Status: ⚠️  WARN — sparse risk labels: " + ", ".join(warnings))
    else:
        print("Status: ✓ PASS — balanced and ready for fine-tuning")
    print("=" * 70)


if __name__ == "__main__":
    main()
