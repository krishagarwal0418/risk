#!/usr/bin/env python3
"""Fine-tune Koala into a compact 2-label moderation model.

Outputs:
  * harmful_content = toxicity + hate + harassment + violence + self_harm
  * sexual          = sexual

Benign/safe is implicit: both scores below threshold.

The script uses processed train/val/test splits, builds a high-quality merged
dataset, then trains a new 2-output multi-label head. If
``models/fine/moderation/rw500`` exists, it is used as the starting checkpoint;
otherwise the base KoalaAI/Text-Moderation model is used.
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

TASK = "koala_merged_moderation"
MERGED_LABELS = tuple(TASK_LABELS[TASK])
HARMFUL = "harmful_content"

HARMFUL_SOURCE_LABELS = {
    C.TOXICITY,
    C.HATE,
    C.HARASSMENT,
    C.VIOLENCE,
    C.SELF_HARM,
}
SUPPORTED_SOURCE_LABELS = HARMFUL_SOURCE_LABELS | {C.SEXUAL}
UNSUPPORTED_RISK = {
    C.PROMPT_INJECTION,
    C.JAILBREAK,
    C.DANGEROUS_INFORMATION,
    C.ILLEGAL_ACTIVITY,
    C.UNKNOWN,
}

TRUSTED_POSITIVE_SOURCE_HINTS = (
    "text_moderation",
    "wildguard",
    "toxic_chat",
    "toxigen",
    "jigsaw",
    "aegis",
)
TRUSTED_SAFE_SOURCE_HINTS = TRUSTED_POSITIVE_SOURCE_HINTS + (
    "xstest",
)

DEFAULT_WARM_START = "models/fine/moderation/rw500"
_THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]


def _source_ok(source: str, hints: tuple[str, ...]) -> bool:
    s = (source or "").lower()
    return any(hint in s for hint in hints)


def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _merge_labels(labels: set[str]) -> list[str]:
    out: list[str] = []
    if labels & HARMFUL_SOURCE_LABELS:
        out.append(HARMFUL)
    if C.SEXUAL in labels:
        out.append(C.SEXUAL)
    return out


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
    merged = _merge_labels(labels)
    if merged:
        if not _source_ok(source, TRUSTED_POSITIVE_SOURCE_HINTS):
            return None, "untrusted_positive_source"
        clean_labels = merged
    elif C.SAFE in labels and not (labels & (set(C.RISK_LABELS) | {C.UNKNOWN})):
        if not _source_ok(source, TRUSTED_SAFE_SOURCE_HINTS):
            return None, "untrusted_safe_source"
        clean_labels = []
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


def _label_counts(rows: list[dict[str, Any]]) -> Counter:
    counts: Counter = Counter()
    for row in rows:
        labels = row.get("labels", [])
        if not labels:
            counts[C.SAFE] += 1
        else:
            counts.update(labels)
    return counts


def _dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = row.get("hash") or _stable_hash(row["text"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _balance_train(
    rows: list[dict[str, Any]],
    *,
    max_harmful: int,
    max_sexual: int,
    safe_ratio: float,
    safe_cap: int,
    min_sexual_target: int,
    max_train: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    positives = [r for r in rows if r.get("labels")]
    safe = [r for r in rows if not r.get("labels")]
    rng.shuffle(positives)
    rng.shuffle(safe)

    written: Counter = Counter()
    kept_pos: list[dict[str, Any]] = []
    for row in positives:
        labels = row.get("labels", [])
        ok = False
        if HARMFUL in labels and written[HARMFUL] < max_harmful:
            ok = True
        if C.SEXUAL in labels and written[C.SEXUAL] < max_sexual:
            ok = True
        if not ok:
            continue
        kept_pos.append(row)
        written.update(labels)

    sexual_rows = [r for r in kept_pos if C.SEXUAL in r.get("labels", [])]
    additions: list[dict[str, Any]] = []
    room = max_train - len(kept_pos) if max_train else 10**9
    if sexual_rows and written[C.SEXUAL] < min_sexual_target:
        shuffled = sexual_rows[:]
        rng.shuffle(shuffled)
        i = 0
        while written[C.SEXUAL] < min_sexual_target and room > 0:
            src = shuffled[i % len(shuffled)]
            i += 1
            dup = dict(src)
            dup["source"] = f"{src.get('source', '')}::sexual_oversample"
            dup["hash"] = f"{src.get('hash') or _stable_hash(src['text'])}::sexual_oversample::{i}"
            additions.append(dup)
            written.update(dup.get("labels", []))
            room -= 1
    kept_pos.extend(additions)

    safe_keep = min(len(safe), safe_cap, int(max(len(kept_pos), 1) * safe_ratio))
    final = kept_pos + safe[:safe_keep]
    rng.shuffle(final)
    if max_train and len(final) > max_train:
        final = final[:max_train]
    return final


def _cap_eval(rows: list[dict[str, Any]], *, max_eval: int, seed: int) -> list[dict[str, Any]]:
    if not max_eval or len(rows) <= max_eval:
        return rows
    rng = random.Random(seed)
    selected: dict[str, dict[str, Any]] = {}
    per_label = max(1, max_eval // 4)
    for label in MERGED_LABELS:
        positives = [r for r in rows if label in r.get("labels", [])]
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
    return {
        "rows": len(rows),
        "labels": dict(_label_counts(rows)),
        "sources": dict(Counter(r.get("source", "") for r in rows).most_common(20)),
    }


def build_merged_data(
    *,
    processed_dir: Path,
    output_dir: Path,
    min_length: int,
    max_length: int,
    max_harmful: int,
    max_sexual: int,
    safe_ratio: float,
    safe_cap: int,
    min_sexual_target: int,
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
        split_issues: Counter = Counter()
        cleaned: list[dict[str, Any]] = []
        for rec in _read_jsonl(path):
            row, reason = _clean_record(
                rec, split=split, min_length=min_length, max_length=max_length
            )
            split_issues[reason] += 1
            if row is None:
                continue
            if row["hash"] in assigned:
                split_issues["cross_split_duplicate"] += 1
                continue
            assigned.add(row["hash"])
            cleaned.append(row)
        cleaned = _dedupe(cleaned)
        splits[split] = cleaned
        issues[split] = dict(split_issues)

    splits["train"] = _balance_train(
        splits["train"],
        max_harmful=max_harmful,
        max_sexual=max_sexual,
        safe_ratio=safe_ratio,
        safe_cap=safe_cap,
        min_sexual_target=min_sexual_target,
        max_train=max_train,
        seed=seed,
    )
    for split in ("val", "test"):
        splits[split] = _cap_eval(splits[split], max_eval=max_eval, seed=seed)
    for split, rows in splits.items():
        _write_jsonl(output_dir / f"{split}.jsonl", rows)

    summary = {
        "task": TASK,
        "labels": list(MERGED_LABELS),
        "mapping": {
            HARMFUL: sorted(HARMFUL_SOURCE_LABELS),
            C.SEXUAL: [C.SEXUAL],
            "benign": "implicit when both labels are 0",
        },
        "filters": {
            "trusted_positive_source_hints": TRUSTED_POSITIVE_SOURCE_HINTS,
            "trusted_safe_source_hints": TRUSTED_SAFE_SOURCE_HINTS,
            "min_length": min_length,
            "max_length": max_length,
            "max_harmful": max_harmful,
            "max_sexual": max_sexual,
            "safe_ratio": safe_ratio,
            "safe_cap": safe_cap,
            "min_sexual_target": min_sexual_target,
            "max_train": max_train,
            "max_eval": max_eval,
        },
        "issues": issues,
        "splits": {split: _summarize(rows) for split, rows in splits.items()},
    }
    (output_dir / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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


def _label_metrics(gold: list[int], scores: list[float]) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score

    by_thr = {thr: _metrics_at(gold, scores, thr) for thr in _THRESHOLDS}
    best_thr = max(_THRESHOLDS, key=lambda thr: by_thr[thr]["f1"])
    return {
        **by_thr[0.5],
        "best_f1_threshold": best_thr,
        "best_f1": by_thr[best_thr]["f1"],
        "best_f1_precision": by_thr[best_thr]["precision"],
        "best_f1_recall": by_thr[best_thr]["recall"],
        "pr_auc": round(float(average_precision_score(gold, scores)), 4)
        if 0 < sum(gold) < len(gold) else 0.0,
    }


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
            scores = {label: [] for label in MERGED_LABELS}
            gold = {label: [] for label in MERGED_LABELS}
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
                latencies.extend([(time.perf_counter() - start) * 1000 / max(len(chunk), 1)] * len(chunk))
                probs = torch.sigmoid(logits.float().cpu()).numpy()
                for row, score_row in zip(chunk, probs):
                    labels = set(row.get("labels", []))
                    for idx, label in enumerate(MERGED_LABELS):
                        scores[label].append(float(score_row[idx]))
                        gold[label].append(1 if label in labels else 0)
            per_label = {label: _label_metrics(gold[label], scores[label]) for label in MERGED_LABELS}
            results[split] = {
                "examples": len(rows),
                "latency_ms": percentiles(latencies),
                "macro": {
                    "f1@0.5": round(sum(m["f1"] for m in per_label.values()) / len(per_label), 4),
                    "best_f1": round(sum(m["best_f1"] for m in per_label.values()) / len(per_label), 4),
                    "pr_auc": round(sum(m["pr_auc"] for m in per_label.values()) / len(per_label), 4),
                },
                "per_label": per_label,
            }
    return results


def _model_source(root: Path, explicit: str | None) -> str:
    if explicit:
        return str(root / explicit) if not Path(explicit).is_absolute() else explicit
    warm = root / DEFAULT_WARM_START
    if warm.exists():
        return str(warm)
    return load_models_config()["transformers"]["moderation_primary"]["hf_name"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune merged 2-label Koala moderation")
    parser.add_argument("--model", default=None,
                        help="Starting model dir/id. Defaults to models/fine/moderation/rw500 if present, else KoalaAI/Text-Moderation")
    parser.add_argument("--tokenizer", default=None,
                        help="Tokenizer source. Defaults to KoalaAI/Text-Moderation for local warm starts.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--data-dir", default="data/koala_merged_moderation")
    parser.add_argument("--output", default="models/finetuned/koala_merged_moderation")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--min-length", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2000)
    parser.add_argument("--max-harmful", type=int, default=70000)
    parser.add_argument("--max-sexual", type=int, default=25000)
    parser.add_argument("--min-sexual-target", type=int, default=16000)
    parser.add_argument("--safe-ratio", type=float, default=1.0)
    parser.add_argument("--safe-cap", type=int, default=70000)
    parser.add_argument("--max-train", type=int, default=180000)
    parser.add_argument("--max-eval", type=int, default=12000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--val-limit", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    root = repo_root()
    data_dir = root / args.data_dir
    output_dir = root / args.output
    model_name = _model_source(root, args.model)
    tokenizer_name = args.tokenizer or load_models_config()["transformers"]["moderation_primary"]["hf_name"]

    build_summary = None
    if not args.train_only:
        build_summary = build_merged_data(
            processed_dir=root / args.processed_dir,
            output_dir=data_dir,
            min_length=args.min_length,
            max_length=args.max_length,
            max_harmful=args.max_harmful,
            max_sexual=args.max_sexual,
            safe_ratio=args.safe_ratio,
            safe_cap=args.safe_cap,
            min_sexual_target=args.min_sexual_target,
            max_train=args.max_train,
            max_eval=args.max_eval,
            seed=args.seed,
        )
        print(json.dumps(build_summary["splits"], indent=2))
    if args.build_only:
        return

    metrics = finetune(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        task=TASK,
        train_path=str(data_dir / "train.jsonl"),
        val_path=str(data_dir / "val.jsonl"),
        output_dir=str(output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_limit=args.val_limit,
        metric_for_best_model="macro_pr_auc",
    )
    eval_results = evaluate(output_dir, data_dir, args.batch_size)
    rows = [
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
        "koala_merged_moderation_finetune",
        status=Status.PASS,
        title="Koala Merged Moderation Fine-Tune",
        summary={
            "model": model_name,
            "tokenizer": tokenizer_name,
            "output": str(output_dir),
            "data_dir": str(data_dir),
            "build": build_summary,
            "train_metrics": metrics,
            "eval": eval_results,
        },
        tables=[(
            "Evaluation",
            ["Split", "Examples", "F1@0.5", "Best F1", "PR-AUC", "p95 ms"],
            rows,
        )],
    )
    print_summary(
        "Koala Merged Moderation Fine-Tune",
        Status.PASS,
        ["Split", "Examples", "F1@0.5", "Best F1", "PR-AUC", "p95 ms"],
        rows,
    )
    print("[finetune] reports/koala_merged_moderation_finetune.{json,md} written")


if __name__ == "__main__":
    main()
