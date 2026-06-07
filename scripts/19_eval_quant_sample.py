#!/usr/bin/env python3
"""Evaluate a prompt-injection ONNX model on a random sample: accuracy + batch latency.

Production serves in batches, so latency is measured as per-BATCH wall-time and its
percentiles (p95/p99/max) across batches, for several batch sizes. Accuracy is
computed once (it does not depend on batch size).

Usage:
    python scripts/19_eval_quant_sample.py \
        --model models/quant/prompt_injection_best/int8_quint8_per_channel/model.onnx \
        --n 2000 --batch-sizes 1,8,16,32,64,128
"""

from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import onnxruntime as ort
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoTokenizer


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to the ONNX model file")
    ap.add_argument("--data", default="data/prompt_injection_best/test.jsonl")
    ap.add_argument("--base", default="protectai/deberta-v3-base-prompt-injection-v2")
    ap.add_argument("--n", type=int, default=2000, help="Random sample size (0 = all)")
    ap.add_argument("--batch-sizes", default="1,8,16,32,64,128")
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--provider", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--pad", default="dynamic", choices=["dynamic", "max"],
                    help="dynamic = pad to longest in batch (realistic); "
                         "max = always pad to max_length (worst-case)")
    args = ap.parse_args()

    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if args.provider == "cuda" else ["CPUExecutionProvider"])
    sess = ort.InferenceSession(args.model, providers=providers)
    in_names = {i.name for i in sess.get_inputs()}
    tok = AutoTokenizer.from_pretrained(args.base)
    padding = "max_length" if args.pad == "max" else True

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    rng = random.Random(args.seed)
    if args.n and len(rows) > args.n:
        rows = rng.sample(rows, args.n)
    texts = [r["text"] for r in rows]
    gold = [int(r["label"]) for r in rows]
    pos = sum(gold)

    print(f"model:    {args.model}")
    print(f"provider: {sess.get_providers()[0]}  | pad: {args.pad} | max_len: {args.max_length}")
    print(f"sample:   {len(rows)} random rows ({pos} injection / {len(gold)-pos} safe)\n")

    def run_batch(batch_texts):
        enc = tok(batch_texts, truncation=True, padding=padding,
                  max_length=args.max_length, return_tensors="np")
        feed = {k: v for k, v in enc.items() if k in in_names}
        return _softmax(sess.run(None, feed)[0])[:, 1]

    # ---- Accuracy (computed once; batch size doesn't change it) ----
    scores: list[float] = []
    for i in range(0, len(texts), 64):
        scores += run_batch(texts[i:i + 64]).tolist()

    def m(thr: float):
        pred = [1 if s >= thr else 0 for s in scores]
        tp = sum(1 for g, p in zip(gold, pred) if g == 1 and p == 1)
        fp = sum(1 for g, p in zip(gold, pred) if g == 0 and p == 1)
        fn = sum(1 for g, p in zip(gold, pred) if g == 1 and p == 0)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0)

    grid = [round(0.05 * i, 2) for i in range(1, 20)]
    best = max(grid, key=lambda t: m(t)[2])

    print("=== ACCURACY (same for all batch sizes) ===")
    if 0 < pos < len(gold):
        print(f"  PR-AUC : {average_precision_score(gold, scores):.4f}")
        print(f"  ROC-AUC: {roc_auc_score(gold, scores):.4f}")
    for nm, t in [("0.50", 0.5), (f"{best:.2f} (best-F1)", best)]:
        pr, rc, f1 = m(t)
        print(f"  @{nm:<14} P={pr:.3f}  R={rc:.3f}  F1={f1:.3f}")
    r90 = next((t for t in grid if m(t)[1] >= 0.90), None)
    if r90 is not None:
        pr, rc, f1 = m(r90)
        print(f"  @{r90} (~90% rec)  P={pr:.3f}  R={rc:.3f}  F1={f1:.3f}")

    # ---- Latency sweep across batch sizes (per-batch wall-time) ----
    batch_sizes = [int(b) for b in args.batch_sizes.split(",") if b.strip()]
    print(f"\n=== LATENCY by batch size ({args.provider}, per-batch wall-time) ===")
    print(f"{'batch':>6} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8} {'mean':>8} "
          f"{'rec/s':>9}")
    for bs in batch_sizes:
        # warmup
        for _ in range(3):
            run_batch(texts[:bs])
        per_batch_ms, n_done = [], 0
        t_all = time.perf_counter()
        for i in range(0, len(texts), bs):
            chunk = texts[i:i + bs]
            s = time.perf_counter()
            run_batch(chunk)
            per_batch_ms.append((time.perf_counter() - s) * 1000)
            n_done += len(chunk)
        total_s = time.perf_counter() - t_all
        a = np.array(per_batch_ms)
        thru = n_done / total_s
        print(f"{bs:>6} {np.percentile(a,50):>7.1f} {np.percentile(a,95):>7.1f} "
              f"{np.percentile(a,99):>7.1f} {a.max():>7.1f} {a.mean():>7.1f} "
              f"{thru:>9.0f}")
    print("\nNote: p95/p99/max are PER-BATCH latency; rec/s is overall throughput.")


if __name__ == "__main__":
    main()
