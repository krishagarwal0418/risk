#!/usr/bin/env python3
"""Build + fine-tune a KoalaAI/Text-Moderation model on supported labels.

KoalaAI/Text-Moderation has native moderation codes for hate, harassment,
sexual, self-harm, and violence. This pipeline trains only those supported
canonical labels plus toxicity from the harassment/toxicity signal. It filters
out unsupported labels instead of asking Koala to learn labels its head cannot
represent well.

Input must already exist from ``scripts/02_prepare_fasttext_data.py``:
    data/processed/all_{train,val,test}.jsonl

Output:
    data/koala_moderation/{train,val,test}.jsonl
    models/finetuned/moderation
    reports/koala_moderation_finetune.{json,md}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from safety_classifier import constants as C
from safety_classifier.config import load_models_config, repo_root
from safety_classifier.reporting import Status, percentiles, print_summary, write_report
from safety_classifier.transformers_layer.finetune import TASK_LABELS, finetune

TASK = "koala_moderation"
KOALA_LABELS = tuple(TASK_LABELS[TASK])
KOALA_LABEL_SET = set(KOALA_LABELS)

# Prefer datasets whose labels are direct moderation signals. Weak "harmful"
# corpora are useful for broad routing, but they add noise for Koala's supported
# categories.
TRUSTED_POSITIVE_SOURCE_HINTS = (
    "text_moderation",
    "wildguard",
    "toxic_chat",
    "toxigen",
    "jigsaw",
)
TRUSTED_SAFE_SOURCE_HINTS = TRUSTED_POSITIVE_SOURCE_HINTS + (
    "xstest",
    "aegis",
)

UNSUPPORTED_RISK = {
    C.PROMPT_INJECTION,
    C.JAILBREAK,
    C.DANGEROUS_INFORMATION,
    C.ILLEGAL_ACTIVITY,
    C.UNKNOWN,
}

_THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]


def _source_ok(source: str, hints: tuple[str, ...]) -> bool:
    s = (source or "").lower()
    return any(hint in s for hint in hints)


def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _clean_record(
    rec: dict[str, Any],
    *,
    split: str,
    min_length: int,
    max_length: int,
) -> tuple[dict[str, Any] | None, str]:
    text = " ".join(str(rec.get("text") or "").split())
    if not text:
        return None, "empty_text"
    if len(text) < min_length:
        return None, "too_short"
    if len(text) > max_length:
        return None, "too_long"

    labels = set(rec.get("labels") or [])
    source = str(rec.get("source") or "")
    supported = sorted(labels & KOALA_LABEL_SET)

    if supported:
        if not _source_ok(source, TRUSTED_POSITIVE_SOURCE_HINTS):
            return None, "untrusted_positive_source"
        clean_labels = supported
    elif C.SAFE in labels and not (labels & (set(C.RISK_LABELS) | {C.UNKNOWN})):
        if not _source_ok(source, TRUSTED_SAFE_SOURCE_HINTS):
            return None, "untrusted_safe_source"
        clean_labels = [C.SAFE]
    else:
        reason = "unsupported_risk_only" if labels & UNSUPPORTED_RISK else "no_supported_label"
        return None, reason

    return {
        "text": text,
        "labels": clean_labels,
        "source": source,
        "hash": rec.get("hash") or _stable_hash(text),
        "original_labels": sorted(labels),
        "split": split,
    }, "kept"


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = row.get("hash") or _stable_hash(row["text"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _has_positive(row: dict[str, Any]) -> bool:
    return any(lab in KOALA_LABEL_SET for lab in row.get("labels", []))


def _cap_train_rows(
    rows: list[dict[str, Any]],
    *,
    max_per_label: int,
    safe_ratio: float,
    safe_cap: int,
    max_train: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    positives = [r for r in rows if _has_positive(r)]
    safe = [r for r in rows if not _has_positive(r)]
    rng.shuffle(positives)
    rng.shuffle(safe)

    label_written: Counter = Counter()
    kept_pos: list[dict[str, Any]] = []
    for row in positives:
        row_labels = [lab for lab in row["labels"] if lab in KOALA_LABEL_SET]
        if not row_labels:
            continue
        if any(label_written[lab] < max_per_label for lab in row_labels):
            kept_pos.append(row)
            for lab in row_labels:
                label_written[lab] += 1

    safe_keep = min(len(safe), safe_cap, int(max(len(kept_pos), 1) * safe_ratio))
    final = kept_pos + safe[:safe_keep]
    rng.shuffle(final)
    if max_train and len(final) > max_train:
        final = final[:max_train]
    return final


def _cap_eval_rows(rows: list[dict[str, Any]], *, max_eval: int, seed: int) -> list[dict[str, Any]]:
    if not max_eval or len(rows) <= max_eval:
        return rows
    rng = random.Random(seed)
    selected: dict[str, dict[str, Any]] = {}
    per_label = max(1, max_eval // (len(KOALA_LABELS) + 2))
    for lab in KOALA_LABELS:
        positives = [r for r in rows if lab in r.get("labels", [])]
        positives.sort(key=lambda r: r.get("hash") or _stable_hash(r["text"]))
        for row in positives[:per_label]:
            selected[row["hash"]] = row
    remaining = [r for r in rows if r["hash"] not in selected]
    rng.shuffle(remaining)
    for row in remaining:
        if len(selected) >= max_eval:
            break
        selected[row["hash"]] = row
    out = list(selected.values())
    rng.shuffle(out)
    return out


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels: Counter = Counter()
    sources: Counter = Counter()
    for row in rows:
        sources[row.get("source", "")] += 1
        labs = row.get("labels", [])
        if labs == [C.SAFE]:
            labels[C.SAFE] += 1
        else:
            labels.update(labs)
    return {
        "rows": len(rows),
        "positive_rows": sum(1 for r in rows if _has_positive(r)),
        "safe_rows": sum(1 for r in rows if r.get("labels") == [C.SAFE]),
        "labels": dict(labels),
        "sources": dict(sources.most_common(20)),
    }


def build_koala_data(
    *,
    processed_dir: Path,
    output_dir: Path,
    min_length: int,
    max_length: int,
    max_per_label: int,
    safe_ratio: float,
    safe_cap: int,
    max_train: int,
    max_eval: int,
    seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    splits: dict[str, list[dict[str, Any]]] = {}
    issues: dict[str, dict[str, int]] = {}

    assigned: set[str] = set()
    for split in ("test", "val", "train"):
        path = processed_dir / f"all_{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing processed split: {path}")
        raw = _read_jsonl(path)
        split_issues: Counter = Counter()
        cleaned: list[dict[str, Any]] = []
        for rec in raw:
            row, reason = _clean_record(
                rec, split=split, min_length=min_length, max_length=max_length
            )
            split_issues[reason] += 1
            if row is None:
                continue
            key = row.get("hash") or _stable_hash(row["text"])
            if key in assigned:
                split_issues["cross_split_duplicate"] += 1
                continue
            assigned.add(key)
            row["hash"] = key
            cleaned.append(row)
        cleaned = _dedupe_rows(cleaned)
        if split == "train":
            cleaned = _cap_train_rows(
                cleaned,
                max_per_label=max_per_label,
                safe_ratio=safe_ratio,
                safe_cap=safe_cap,
                max_train=max_train,
                seed=seed,
            )
        else:
            cleaned = _cap_eval_rows(cleaned, max_eval=max_eval, seed=seed)
        splits[split] = cleaned
        issues[split] = dict(split_issues)

    for split, rows in splits.items():
        _write_jsonl(output_dir / f"{split}.jsonl", rows)

    summary = {
        "task": TASK,
        "labels": list(KOALA_LABELS),
        "processed_dir": str(processed_dir),
        "output_dir": str(output_dir),
        "filters": {
            "trusted_positive_source_hints": TRUSTED_POSITIVE_SOURCE_HINTS,
            "trusted_safe_source_hints": TRUSTED_SAFE_SOURCE_HINTS,
            "min_length": min_length,
            "max_length": max_length,
            "max_per_label": max_per_label,
            "safe_ratio": safe_ratio,
            "safe_cap": safe_cap,
            "max_train": max_train,
            "max_eval": max_eval,
        },
        "issues": issues,
        "splits": {split: _summarize(rows) for split, rows in splits.items()},
    }
    (output_dir / "build_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _metrics_at(gold: list[int], scores: list[float], threshold: float) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for y, score in zip(gold, scores):
        pred = score >= threshold
        if pred and y:
            tp += 1
        elif pred and not y:
            fp += 1
        elif not pred and y:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "support_positive": tp + fn,
        "predicted_positive": tp + fp,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def _label_threshold_metrics(gold: list[int], scores: list[float]) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score

    by_thr = {thr: _metrics_at(gold, scores, thr) for thr in _THRESHOLDS}
    best_thr = max(_THRESHOLDS, key=lambda thr: by_thr[thr]["f1"])
    out = dict(by_thr[0.5])
    out.update({
        "best_f1_threshold": best_thr,
        "best_f1": by_thr[best_thr]["f1"],
        "best_f1_precision": by_thr[best_thr]["precision"],
        "best_f1_recall": by_thr[best_thr]["recall"],
        "pr_auc": round(float(average_precision_score(gold, scores)), 4)
        if 0 < sum(gold) < len(gold) else 0.0,
    })
    return out


def evaluate(output_dir: Path, data_dir: Path, batch_size: int) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(output_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(output_dir))
    model.eval()
    if device == "cuda":
        model = model.to("cuda")

    results: dict[str, Any] = {}
    with torch.inference_mode():
        for split in ("val", "test"):
            rows = _read_jsonl(data_dir / f"{split}.jsonl")
            scores_by_label = {lab: [] for lab in KOALA_LABELS}
            gold_by_label = {lab: [] for lab in KOALA_LABELS}
            latencies: list[float] = []
            for i in range(0, len(rows), batch_size):
                chunk = rows[i : i + batch_size]
                enc = tokenizer(
                    [r["text"] for r in chunk],
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
                latency = (time.perf_counter() - start) * 1000 / max(len(chunk), 1)
                probs = torch.sigmoid(logits.float().cpu()).numpy()
                latencies.extend([latency] * len(chunk))
                for row, score_row in zip(chunk, probs):
                    labels = set(row.get("labels", []))
                    for idx, lab in enumerate(KOALA_LABELS):
                        scores_by_label[lab].append(float(score_row[idx]))
                        gold_by_label[lab].append(1 if lab in labels else 0)

            per_label = {
                lab: _label_threshold_metrics(gold_by_label[lab], scores_by_label[lab])
                for lab in KOALA_LABELS
            }
            results[split] = {
                "examples": len(rows),
                "latency_ms": percentiles(latencies),
                "macro": {
                    "f1@0.5": round(
                        sum(m["f1"] for m in per_label.values()) / len(per_label), 4
                    ),
                    "best_f1": round(
                        sum(m["best_f1"] for m in per_label.values()) / len(per_label), 4
                    ),
                    "pr_auc": round(
                        sum(m["pr_auc"] for m in per_label.values()) / len(per_label), 4
                    ),
                },
                "per_label": per_label,
            }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Koala moderation cleanly")
    parser.add_argument("--data-dir", default="data/koala_moderation")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output", default="models/finetuned/moderation")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--min-length", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2000)
    parser.add_argument("--max-per-label", type=int, default=35000)
    parser.add_argument("--safe-ratio", type=float, default=1.25)
    parser.add_argument("--safe-cap", type=int, default=50000)
    parser.add_argument("--max-train", type=int, default=180000)
    parser.add_argument("--max-eval", type=int, default=12000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--val-limit", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--init-weights", default=None)
    args = parser.parse_args()

    root = repo_root()
    data_dir = root / args.data_dir
    processed_dir = root / args.processed_dir
    output_dir = root / args.output
    model_name = load_models_config()["transformers"]["moderation_primary"]["hf_name"]

    build_summary = None
    if not args.train_only:
        build_summary = build_koala_data(
            processed_dir=processed_dir,
            output_dir=data_dir,
            min_length=args.min_length,
            max_length=args.max_length,
            max_per_label=args.max_per_label,
            safe_ratio=args.safe_ratio,
            safe_cap=args.safe_cap,
            max_train=args.max_train,
            max_eval=args.max_eval,
            seed=args.seed,
        )
        print(json.dumps(build_summary["splits"], indent=2))
    if args.build_only:
        return

    metrics = finetune(
        model_name=model_name,
        task=TASK,
        train_path=str(data_dir / "train.jsonl"),
        val_path=str(data_dir / "val.jsonl"),
        output_dir=str(output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_limit=args.val_limit,
        init_weights_path=args.init_weights,
        metric_for_best_model="macro_pr_auc",
    )
    eval_results = evaluate(output_dir, data_dir, args.batch_size)
    table_rows = [
        [
            split,
            result["examples"],
            result["macro"]["f1@0.5"],
            result["macro"]["best_f1"],
            result["macro"]["pr_auc"],
            result["latency_ms"]["p95"],
        ]
        for split, result in eval_results.items()
    ]
    write_report(
        "koala_moderation_finetune",
        status=Status.PASS,
        title="Koala Moderation Fine-Tune",
        summary={
            "model": model_name,
            "output": str(output_dir),
            "data_dir": str(data_dir),
            "build": build_summary,
            "train_metrics": metrics,
            "eval": eval_results,
        },
        tables=[(
            "Evaluation",
            ["Split", "Examples", "F1@0.5", "Best F1", "PR-AUC", "p95 ms"],
            table_rows,
        )],
    )
    print_summary(
        "Koala Moderation Fine-Tune",
        Status.PASS,
        ["Split", "Examples", "F1@0.5", "Best F1", "PR-AUC", "p95 ms"],
        table_rows,
    )
    print("[finetune] reports/koala_moderation_finetune.{json,md} written")


if __name__ == "__main__":
    main()
