#!/usr/bin/env python3
"""Evaluate a fine-tuned attack transformer on the rogue-security benchmark.

rogue-security is part of the TRAINING data, so we evaluate ONLY on the held-out
test split (the rows whose normalize+hash bucket lands in [90,100), exactly the
rows the splitter keeps out of training). This gives a leakage-free number.

Usage:
    HF_TOKEN=... python scripts/13_eval_finetuned_on_rogue.py \
        --model models/fine/prompt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from safety_classifier.data.splitter import _split_for_hash
from safety_classifier.normalizer import normalize


def _metrics_at(gold, score, thr):
    pred = [1 if s >= thr else 0 for s in score]
    tp = sum(1 for g, p in zip(gold, pred) if g == 1 and p == 1)
    fp = sum(1 for g, p in zip(gold, pred) if g == 0 and p == 1)
    fn = sum(1 for g, p in zip(gold, pred) if g == 1 and p == 0)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/fine/prompt")
    ap.add_argument("--dataset", default="rogue-security/prompt-injections-benchmark")
    ap.add_argument("--held-out", default="test", choices=["test", "val", "train", "all"])
    ap.add_argument("--score", default="prompt_injection",
                    choices=["prompt_injection", "jailbreak", "attack"])
    args = ap.parse_args()

    print("=" * 70)
    print(f"Fine-tuned model: {args.model}")
    print(f"Benchmark: {args.dataset}  | held-out split: {args.held_out} | score: {args.score}")
    print("=" * 70)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label2idx = {v: k for k, v in id2label.items()}
    print(f"Model labels: {id2label}")

    ds = load_dataset(args.dataset)
    split = "test" if "test" in ds else list(ds.keys())[0]
    rows = ds[split]

    def keep(text: str) -> bool:
        if args.held_out == "all":
            return True
        return _split_for_hash(normalize(text).text_hash) == args.held_out

    gold, texts = [], []
    for i in range(len(rows)):
        t = rows[i]["text"]
        lab = str(rows[i]["label"]).strip().lower()
        y = 1 if lab in ("jailbreak", "injection", "prompt_injection", "1", "true") else 0
        if not isinstance(t, str) or not t.strip():
            continue
        if not keep(t):
            continue
        gold.append(y)
        texts.append(t)

    print(f"Held-out '{args.held_out}' rows: {len(gold)} "
          f"({sum(gold)} injection / {len(gold) - sum(gold)} benign)")
    if not gold or sum(gold) == 0 or sum(gold) == len(gold):
        print("✗ Need both classes in the held-out split.")
        sys.exit(1)

    # Batched inference.
    scores = []
    pi_idx = label2idx.get("prompt_injection", 0)
    jb_idx = label2idx.get("jailbreak", 1)
    with torch.inference_mode():
        for i in range(0, len(texts), 64):
            chunk = texts[i:i + 64]
            enc = tok(chunk, return_tensors="pt", truncation=True,
                      padding=True, max_length=128)
            logits = model(**enc).logits
            probs = torch.sigmoid(logits).numpy()
            for p in probs:
                if args.score == "prompt_injection":
                    scores.append(float(p[pi_idx]))
                elif args.score == "jailbreak":
                    scores.append(float(p[jb_idx]))
                else:
                    scores.append(float(max(p[pi_idx], p[jb_idx])))

    pr_auc = average_precision_score(gold, scores)
    roc_auc = roc_auc_score(gold, scores)
    grid = [round(t, 2) for t in np.linspace(0.05, 0.95, 19)]
    best = max(grid, key=lambda t: _metrics_at(gold, scores, t)[2])
    bp, br, bf = _metrics_at(gold, scores, best)
    dp, dr, df = _metrics_at(gold, scores, 0.5)
    # 90%-recall operating point.
    rec90 = next((t for t in grid if _metrics_at(gold, scores, t)[1] >= 0.90), None)
    r90 = _metrics_at(gold, scores, rec90) if rec90 is not None else None

    print()
    print("Ranking quality (threshold-independent):")
    print(f"  PR-AUC  : {pr_auc:.4f}")
    print(f"  ROC-AUC : {roc_auc:.4f}")
    print()
    print("Operating points:")
    print(f"  @0.50           P={dp:.3f} R={dr:.3f} F1={df:.3f}")
    print(f"  @{best:.2f} (best-F1) P={bp:.3f} R={br:.3f} F1={bf:.3f}")
    if r90:
        print(f"  @{rec90:.2f} (~90% recall) P={r90[0]:.3f} R={r90[1]:.3f} F1={r90[2]:.3f}")
    else:
        print("  (90% recall not reachable on this grid)")

    verdict = "STRONG" if pr_auc >= 0.85 else "OK" if pr_auc >= 0.70 else "WEAK"
    print()
    print("=" * 70)
    print(f"Verdict: {verdict}  (PR-AUC={pr_auc:.3f}) on held-out rogue '{args.held_out}'")
    print("=" * 70)


if __name__ == "__main__":
    main()
