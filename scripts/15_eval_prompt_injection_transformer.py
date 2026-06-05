#!/usr/bin/env python3
"""Evaluate a prompt-injection transformer on external benchmarks.

Works with binary models that expose labels like SAFE/INJECTION, BENIGN/ATTACK,
or generic LABEL_0/LABEL_1. Reports ranking metrics, thresholded metrics, and
batched inference latency.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from safety_classifier.data.splitter import _split_for_hash
from safety_classifier.normalizer import normalize
from safety_classifier.reporting import Status, percentiles, print_summary, write_report


_TEXT_KEYS = (
    "text", "prompt", "input", "content", "sentence", "message",
    "instruction", "query", "user_input", "jailbreak_query", "goal",
)
_LABEL_KEYS = (
    "label", "labels", "is_injection", "injection", "class", "target",
    "is_malicious", "malicious", "type", "category", "ground_truth",
)
_POS_TOKENS = (
    "inject", "injection", "prompt_injection", "attack", "jailbreak",
    "malicious", "unsafe", "harmful", "true", "1", "yes", "positive",
)
_NEG_TOKENS = (
    "safe", "benign", "legit", "clean", "normal", "harmless",
    "false", "0", "no", "negative",
)


def _detect_key(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _to_binary(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if value >= 0.5 else 0
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return None
        joined = " ".join(str(v) for v in value)
        return _to_binary(joined)
    s = str(value).strip().lower()
    if not s or s in ("none", "nan"):
        return None
    if any(tok == s for tok in ("1", "true", "yes")):
        return 1
    if any(tok == s for tok in ("0", "false", "no")):
        return 0
    if any(tok in s for tok in _POS_TOKENS):
        return 1
    if any(tok in s for tok in _NEG_TOKENS):
        return 0
    return None


def _positive_index(id2label: dict[int, str], num_labels: int) -> int:
    for idx, label in id2label.items():
        low = label.lower()
        if any(tok in low for tok in ("inject", "attack", "jailbreak", "malicious", "unsafe")):
            return idx
    if num_labels == 2:
        return 1
    return 0


def _scores_from_logits(logits, id2label: dict[int, str]) -> list[float]:
    num_labels = int(logits.shape[-1])
    pos_idx = _positive_index(id2label, num_labels)
    if num_labels == 1:
        probs = torch.sigmoid(logits).reshape(-1)
        return [float(p) for p in probs]
    probs = torch.softmax(logits.float(), dim=-1)
    return [float(row[pos_idx]) for row in probs]


def _metrics_at(gold: list[int], scores: list[float], thr: float) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for g, score in zip(gold, scores):
        pred = score >= thr
        if pred and g:
            tp += 1
        elif pred and not g:
            fp += 1
        elif not pred and g:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": thr,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "predicted_positive": tp + fp,
    }


def _load_rows(dataset: str, split: str | None) -> tuple[str, Any]:
    ds = load_dataset(dataset)
    chosen = split or ("test" if "test" in ds else list(ds.keys())[0])
    return chosen, ds[chosen]


def _collect_examples(
    dataset: str,
    split: str | None,
    limit: int | None,
    held_out: str,
) -> tuple[list[str], list[int], dict[str, Any]]:
    chosen_split, rows = _load_rows(dataset, split)
    text_key = _detect_key(rows.column_names, _TEXT_KEYS)
    label_key = _detect_key(rows.column_names, _LABEL_KEYS)
    if not text_key or not label_key:
        raise RuntimeError(
            f"Could not detect text/label columns for {dataset}: "
            f"text={text_key}, label={label_key}, columns={rows.column_names}"
        )

    def keep(text: str) -> bool:
        if held_out == "none":
            return True
        return _split_for_hash(normalize(text).text_hash) == held_out

    texts: list[str] = []
    gold: list[int] = []
    skipped = 0
    held_out_drop = 0
    n = len(rows) if limit is None else min(limit, len(rows))
    for i in range(n):
        row = rows[i]
        text = row.get(text_key)
        y = _to_binary(row.get(label_key))
        if not isinstance(text, str) or not text.strip() or y is None:
            skipped += 1
            continue
        if not keep(text):
            held_out_drop += 1
            continue
        texts.append(text)
        gold.append(y)
    meta = {
        "dataset": dataset,
        "split": chosen_split,
        "text_key": text_key,
        "label_key": label_key,
        "raw_rows_considered": n,
        "skipped": skipped,
        "held_out": held_out,
        "held_out_dropped": held_out_drop,
    }
    return texts, gold, meta


def _evaluate_dataset(
    model,
    tokenizer,
    id2label: dict[int, str],
    dataset: str,
    split: str | None,
    limit: int | None,
    batch_size: int,
    device: str,
    held_out: str,
) -> dict[str, Any]:
    texts, gold, meta = _collect_examples(dataset, split, limit, held_out)
    if sum(gold) == 0 or sum(gold) == len(gold):
        raise RuntimeError(
            f"{dataset} needs both classes after filtering; got "
            f"{sum(gold)} positive / {len(gold) - sum(gold)} negative"
        )

    scores: list[float] = []
    latencies: list[float] = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            enc = tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128,
            )
            if device == "cuda":
                enc = {k: v.to("cuda") for k, v in enc.items()}
            start = time.perf_counter()
            logits = model(**enc).logits
            if device == "cuda":
                torch.cuda.synchronize()
            per_example_ms = (time.perf_counter() - start) * 1000 / max(len(chunk), 1)
            scores.extend(_scores_from_logits(logits.cpu(), id2label))
            latencies.extend([per_example_ms] * len(chunk))

    pr_auc = round(float(average_precision_score(gold, scores)), 4)
    roc_auc = round(float(roc_auc_score(gold, scores)), 4)
    grid = [round(float(t), 2) for t in np.linspace(0.05, 0.95, 19)]
    by_thr = {thr: _metrics_at(gold, scores, thr) for thr in grid}
    best_thr = max(grid, key=lambda thr: by_thr[thr]["f1"])
    high_recall = [thr for thr in grid if by_thr[thr]["recall"] >= 0.90]
    high_recall_thr = max(high_recall) if high_recall else None
    return {
        **meta,
        "examples": len(gold),
        "positives": int(sum(gold)),
        "negatives": int(len(gold) - sum(gold)),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "at_0_5": by_thr[0.5],
        "best_f1": by_thr[best_thr],
        "high_recall": by_thr[high_recall_thr] if high_recall_thr is not None else None,
        "latency_ms": percentiles(latencies),
    }


def run_eval(
    model_path: str,
    datasets: list[str],
    split: str | None,
    limit: int | None,
    batch_size: int,
    device: str,
    held_out: str,
) -> dict[str, Any]:
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    if device == "cuda":
        model = model.to("cuda")
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    results: dict[str, Any] = {}
    warnings: list[str] = []
    failures: list[str] = []
    for dataset in datasets:
        try:
            results[dataset] = _evaluate_dataset(
                model, tokenizer, id2label, dataset, split, limit, batch_size, device, held_out
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{dataset}: {type(exc).__name__}: {exc}")

    status = Status.FAIL if not results else (Status.WARN if failures or warnings else Status.PASS)
    table_rows = [
        [
            name,
            r["examples"],
            r["positives"],
            r["negatives"],
            r["pr_auc"],
            r["roc_auc"],
            r["at_0_5"]["f1"],
            r["best_f1"]["f1"],
            r["best_f1"]["threshold"],
            r["high_recall"]["precision"] if r["high_recall"] else "n/a",
            r["high_recall"]["recall"] if r["high_recall"] else "n/a",
            r["latency_ms"]["p95"],
        ]
        for name, r in results.items()
    ]
    write_report(
        "prompt_injection_transformer_benchmark",
        status=status,
        title="Prompt Injection Transformer Benchmark",
        summary={
            "model": model_path,
            "device": device,
            "id2label": id2label,
            "results": results,
        },
        tables=[(
            "Benchmark Results",
            [
                "Dataset", "Examples", "Pos", "Neg", "PR-AUC", "ROC-AUC",
                "F1@0.5", "Best F1", "Best Thr", "HR Precision", "HR Recall", "p95 ms",
            ],
            table_rows,
        )],
        warnings=warnings,
        failures=failures,
    )
    print_summary(
        "Prompt Injection Transformer Benchmark",
        status,
        [
            "Dataset", "Examples", "Pos", "Neg", "PR-AUC", "ROC-AUC",
            "F1@0.5", "Best F1", "Best Thr", "HR Precision", "HR Recall", "p95 ms",
        ],
        table_rows,
    )
    return {"status": status.value, "model": model_path, "results": results, "failures": failures}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mod_tests/protectai-deberta-v3-base-prompt-injection")
    p.add_argument(
        "--datasets",
        default="deepset/prompt-injections,rogue-security/prompt-injections-benchmark",
        help="Comma-separated Hugging Face dataset ids",
    )
    p.add_argument("--split", default=None)
    p.add_argument("--limit", type=int, default=0, help="per-dataset cap; 0 = all")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument(
        "--held-out",
        default="none",
        choices=["none", "train", "val", "test"],
        help="Hash-split filter for datasets that may overlap training data",
    )
    args = p.parse_args()
    run_eval(
        model_path=args.model,
        datasets=[d.strip() for d in args.datasets.split(",") if d.strip()],
        split=args.split,
        limit=args.limit or None,
        batch_size=args.batch_size,
        device=args.device,
        held_out=args.held_out,
    )
    print("[benchmark] reports/prompt_injection_transformer_benchmark.{json,md} written")


if __name__ == "__main__":
    main()
