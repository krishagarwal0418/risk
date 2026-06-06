#!/usr/bin/env python3
"""Evaluate a quantized prompt-injection ONNX model on a random sample.

Reports accuracy (PR-AUC, ROC-AUC, F1 at thresholds) from BATCHED inference, and
true per-request latency percentiles (p50/p95/p99) from batch=1 timing.

Usage:
    python scripts/19_eval_quant_sample.py \
        --model models/quant/prompt_injection_best/int8_quint8_per_channel/model.onnx \
        --n 1000 --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

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
    ap.add_argument("--base", default="protectai/deberta-v3-base-prompt-injection-v2",
                    help="Tokenizer source")
    ap.add_argument("--n", type=int, default=1000, help="Random sample size")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--provider", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--lat-requests", type=int, default=1000,
                    help="How many batch=1 requests to time for percentiles")
    args = ap.parse_args()

    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if args.provider == "cuda" else ["CPUExecutionProvider"])
    sess = ort.InferenceSession(args.model, providers=providers)
    in_names = {i.name for i in sess.get_inputs()}
    tok = AutoTokenizer.from_pretrained(args.base)
    print(f"model:    {args.model}")
    print(f"provider: {sess.get_providers()[0]}  | inputs: {sorted(in_names)}")

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    rng = random.Random(args.seed)
    if args.n and len(rows) > args.n:
        rows = rng.sample(rows, args.n)
    texts = [r["text"] for r in rows]
    gold = [int(r["label"]) for r in rows]
    pos = sum(gold)
    print(f"sample:   {len(rows)} random rows ({pos} injection / {len(gold)-pos} safe)\n")

    # ---- Accuracy: batched ----
    scores: list[float] = []
    bs = args.batch_size
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], truncation=True, padding=True,
                  max_length=args.max_length, return_tensors="np")
        feed = {k: v for k, v in enc.items() if k in in_names}
        scores += _softmax(sess.run(None, feed)[0])[:, 1].tolist()

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

    print("=== ACCURACY (batched) ===")
    if 0 < pos < len(gold):
        print(f"  PR-AUC : {average_precision_score(gold, scores):.4f}")
        print(f"  ROC-AUC: {roc_auc_score(gold, scores):.4f}")
    for nm, t in [("0.50", 0.5), (f"{best:.2f} bestF1", best)]:
        pr, rc, f1 = m(t)
        print(f"  @{nm:<11} P={pr:.3f} R={rc:.3f} F1={f1:.3f}")
    r90 = next((t for t in grid if m(t)[1] >= 0.90), None)
    if r90 is not None:
        pr, rc, f1 = m(r90)
        print(f"  @{r90} ~90%rec  P={pr:.3f} R={rc:.3f} F1={f1:.3f}")

    # ---- Latency: batch=1 (true per-request percentiles) ----
    n_lat = min(args.lat_requests, len(texts))
    for t in texts[:20]:  # warmup
        enc = tok(t, truncation=True, padding="max_length",
                  max_length=args.max_length, return_tensors="np")
        sess.run(None, {k: v for k, v in enc.items() if k in in_names})
    lat = []
    for t in texts[:n_lat]:
        enc = tok(t, truncation=True, padding="max_length",
                  max_length=args.max_length, return_tensors="np")
        feed = {k: v for k, v in enc.items() if k in in_names}
        s = time.perf_counter()
        sess.run(None, feed)
        lat.append((time.perf_counter() - s) * 1000)
    lat = np.array(lat)

    print(f"\n=== LATENCY (batch=1, {n_lat} requests, {args.provider}) ===")
    print(f"  mean {lat.mean():.1f}ms | p50 {np.percentile(lat,50):.1f} | "
          f"p90 {np.percentile(lat,90):.1f} | p95 {np.percentile(lat,95):.1f} | "
          f"p99 {np.percentile(lat,99):.1f} | max {lat.max():.1f}ms")
    print(f"  throughput (batch=1): {1000/lat.mean():.0f} req/s/core")


if __name__ == "__main__":
    main()
