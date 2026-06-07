#!/usr/bin/env python3
"""Find per-label decision thresholds for the fine-tuned KoalaAI moderation model.

Runs the model on the val split, sweeps thresholds per label, and writes the
F1-optimal threshold for each label to a YAML file that inference can load.
This recovers the gap between F1@0.5 and best-F1 at zero training cost.

Usage:
    python scripts/20_calibrate_moderation_thresholds.py \
        --model-dir models/finetuned/moderation \
        --data data/koala_moderation/val.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from safety_classifier.transformers_layer.finetune import TASK_LABELS

LABELS = list(TASK_LABELS["koala_moderation"])
GRID = [round(0.02 * i, 2) for i in range(1, 50)]  # 0.02 .. 0.98


def _read_jsonl(path: Path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def _f1(gold, scores, thr):
    tp = fp = fn = 0
    for y, s in zip(gold, scores):
        p = s >= thr
        if p and y:
            tp += 1
        elif p and not y:
            fp += 1
        elif not p and y:
            fn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/finetuned/moderation")
    ap.add_argument("--data", default="data/koala_moderation/val.jsonl")
    ap.add_argument("--out", default="reports/koala_moderation_thresholds.yaml")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=128)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device).eval()

    rows = _read_jsonl(Path(args.data))
    scores = {lab: [] for lab in LABELS}
    gold = {lab: [] for lab in LABELS}
    with torch.inference_mode():
        for i in range(0, len(rows), args.batch_size):
            chunk = rows[i:i + args.batch_size]
            enc = tok([r["text"] for r in chunk], return_tensors="pt",
                      truncation=True, padding=True, max_length=args.max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            probs = torch.sigmoid(model(**enc).logits.float().cpu()).numpy()
            for row, pr in zip(chunk, probs):
                labs = set(row.get("labels", []))
                for j, lab in enumerate(LABELS):
                    scores[lab].append(float(pr[j]))
                    gold[lab].append(1 if lab in labs else 0)

    print(f"calibrating on {len(rows)} val rows\n")
    print(f"{'label':<22} {'thr':>5} {'P':>6} {'R':>6} {'F1':>6}  (vs F1@0.5)")
    out_thresholds = {}
    macro_best = macro_half = 0.0
    for lab in LABELS:
        g, s = gold[lab], scores[lab]
        best_thr = max(GRID, key=lambda t: _f1(g, s, t)[2])
        bp, br, bf = _f1(g, s, best_thr)
        _, _, half = _f1(g, s, 0.5)
        out_thresholds[lab] = {"threshold": best_thr, "precision": round(bp, 4),
                               "recall": round(br, 4), "f1": round(bf, 4)}
        macro_best += bf
        macro_half += half
        print(f"{lab:<22} {best_thr:>5} {bp:>6.3f} {br:>6.3f} {bf:>6.3f}   ({half:.3f})")
    n = len(LABELS)
    print(f"\nmacro best-F1: {macro_best/n:.4f}   |   macro F1@0.5: {macro_half/n:.4f}"
          f"   (+{(macro_best-macro_half)/n:.4f})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
        out_path.write_text(yaml.safe_dump(out_thresholds, sort_keys=False), encoding="utf-8")
    except Exception:
        out_path = out_path.with_suffix(".json")
        out_path.write_text(json.dumps(out_thresholds, indent=2), encoding="utf-8")
    print(f"\n✓ wrote per-label thresholds to {out_path}")


if __name__ == "__main__":
    main()
