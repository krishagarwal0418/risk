#!/usr/bin/env python3
"""Build + fine-tune a strong binary prompt-injection detector.

Data strategy:
  * Use multiple public prompt-injection datasets.
  * Use official train/validation/test splits when available.
  * For test-only Rogue, use the repo's deterministic hash split so train/eval
    remain leakage-free.
  * Add deterministic masked/obfuscated variants for positive examples.
  * Cap and balance negatives so the model does not learn "safe always".

The script reads HF_TOKEN from the environment. Do not hardcode tokens here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import _bootstrap  # noqa: F401

from safety_classifier.config import repo_root
from safety_classifier.data.splitter import _split_for_hash
from safety_classifier.normalizer import normalize
from safety_classifier.reporting import Status, percentiles, print_summary, write_report


DEFAULT_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"
DEFAULT_DATASETS = [
    "dmilush/shieldlm-prompt-injection",
    "neuralchemy/prompt-injection-Threat-Matrix",
    "wambosec/prompt-injections",
    "deepset/prompt-injections",
    "rogue-security/prompt-injections-benchmark",
]

_TEXT_KEYS = (
    "text", "prompt", "input", "content", "sentence", "message",
    "instruction", "query", "user_input", "jailbreak_query", "goal",
)
_LABEL_KEYS = (
    "label", "binary_label", "label_binary", "labels", "is_injection",
    "injection", "is_malicious", "malicious", "class", "target", "type",
    "category", "ground_truth",
)
_POS_TOKENS = (
    "inject", "injection", "prompt_injection", "attack", "jailbreak",
    "malicious", "unsafe", "harmful", "true", "1", "yes", "positive",
)
_NEG_TOKENS = (
    "safe", "benign", "legit", "clean", "normal", "harmless", "none",
    "false", "0", "no", "negative",
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _detect_key(columns: Iterable[str], candidates: tuple[str, ...]) -> str | None:
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
        return _to_binary(" ".join(str(v) for v in value))
    s = str(value).strip().lower()
    if not s or s in ("nan", "null"):
        return None
    if s in ("1", "true", "yes", "positive"):
        return 1
    if s in ("0", "false", "no", "negative"):
        return 0
    if any(tok in s for tok in _POS_TOKENS):
        return 1
    if any(tok in s for tok in _NEG_TOKENS):
        return 0
    return None


def _split_name(dataset_id: str, split: str, text: str) -> str:
    # Rogue is test-only; use deterministic hash buckets to create train/val/test
    # while preserving a leakage-free held-out test set.
    if dataset_id == "rogue-security/prompt-injections-benchmark":
        return _split_for_hash(normalize(text).text_hash)
    if split in ("validation", "val"):
        return "val"
    if split == "test":
        return "test"
    return "train"


def _masked_variant(text: str) -> str:
    words = text.split()
    if len(words) < 6:
        return text
    out = words[:]
    # Deterministic sparse masking, preserving most context.
    for i in range(2, len(out), 9):
        if out[i].isalpha() and len(out[i]) > 3:
            out[i] = "[MASK]"
    return " ".join(out)


def _zero_width_variant(text: str) -> str:
    replacements = {
        "ignore": "ign\u200bore",
        "previous": "prev\u200bious",
        "instructions": "instr\u200buctions",
        "system": "sys\u200btem",
        "prompt": "pro\u200bmpt",
        "jailbreak": "jail\u200bbreak",
    }
    out = text
    for src, dst in replacements.items():
        out = out.replace(src, dst).replace(src.title(), dst)
    return out


def _spaced_variant(text: str) -> str:
    replacements = {
        "ignore": "i g n o r e",
        "bypass": "b y p a s s",
        "jailbreak": "j a i l b r e a k",
    }
    out = text
    for src, dst in replacements.items():
        out = out.replace(src, dst).replace(src.title(), dst)
    return out


def _augment_positive(text: str, max_variants: int) -> list[str]:
    if max_variants <= 0:
        return []
    variants = [_masked_variant(text), _zero_width_variant(text), _spaced_variant(text)]
    uniq = []
    seen = {text}
    for variant in variants:
        if variant and variant not in seen:
            seen.add(variant)
            uniq.append(variant)
        if len(uniq) >= max_variants:
            break
    return uniq


def _load_source(dataset_id: str) -> dict[str, list[dict[str, Any]]]:
    from datasets import load_dataset

    ds = load_dataset(dataset_id)
    out: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for split, rows in ds.items():
        text_key = _detect_key(rows.column_names, _TEXT_KEYS)
        label_key = _detect_key(rows.column_names, _LABEL_KEYS)
        if not text_key or not label_key:
            raise RuntimeError(
                f"could not detect text/label columns; columns={rows.column_names}"
            )
        for row in rows:
            text = row.get(text_key)
            label = _to_binary(row.get(label_key))
            if not isinstance(text, str) or not text.strip() or label is None:
                continue
            text = " ".join(text.split())
            if len(text) < 8 or len(text) > 4000:
                continue
            dest = _split_name(dataset_id, split, text)
            out[dest].append({
                "text": text,
                "label": int(label),
                "source": dataset_id,
                "hash": _sha(text),
            })
    return out


def build_training_data(
    datasets: list[str],
    output_dir: Path,
    augment_per_positive: int,
    neg_pos_ratio: float,
    max_train: int,
    max_per_source: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    failures: list[str] = []
    source_counts: dict[str, dict[str, int]] = {}

    for dataset_id in datasets:
        try:
            splits = _load_source(dataset_id)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{dataset_id}: {type(exc).__name__}: {exc}")
            continue
        source_counts[dataset_id] = {k: len(v) for k, v in splits.items()}
        for split, rows in splits.items():
            combined[split].extend(rows)

    # Add obfuscated/masked positive variants only to training data.
    augmented: list[dict[str, Any]] = []
    for row in combined["train"]:
        if row["label"] != 1:
            continue
        for variant in _augment_positive(row["text"], augment_per_positive):
            augmented.append({
                "text": variant,
                "label": 1,
                "source": row["source"] + "::aug",
                "hash": _sha(variant),
            })
    combined["train"].extend(augmented)

    # Deduplicate across all splits by hash, preferring test > val > train.
    assigned: set[str] = set()
    deduped: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for split in ("test", "val", "train"):
        for row in combined[split]:
            if row["hash"] in assigned:
                continue
            assigned.add(row["hash"])
            deduped[split].append(row)

    # Source caps and negative balancing for train.
    pos = [r for r in deduped["train"] if r["label"] == 1]
    neg = [r for r in deduped["train"] if r["label"] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    def cap_source(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        counts: Counter = Counter()
        kept = []
        for row in rows:
            src = row["source"]
            if max_per_source and counts[src] >= max_per_source:
                continue
            counts[src] += 1
            kept.append(row)
        return kept

    pos = cap_source(pos)
    neg_cap = min(len(neg), int(max(len(pos), 1) * neg_pos_ratio))
    neg = cap_source(neg[:neg_cap])
    train = pos + neg
    rng.shuffle(train)
    if max_train and len(train) > max_train:
        train = train[:max_train]

    deduped["train"] = train

    for split, rows in deduped.items():
        path = output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "datasets": datasets,
        "source_counts": source_counts,
        "failures": failures,
        "augment_per_positive": augment_per_positive,
        "neg_pos_ratio": neg_pos_ratio,
        "max_train": max_train,
        "max_per_source": max_per_source,
        "splits": {
            split: {
                "rows": len(rows),
                "positive": sum(r["label"] for r in rows),
                "negative": len(rows) - sum(r["label"] for r in rows),
                "sources": dict(Counter(r["source"] for r in rows).most_common()),
            }
            for split, rows in deduped.items()
        },
    }
    (output_dir / "build_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _dataset_from_rows(rows: list[dict[str, Any]], tokenizer, max_length: int):
    from datasets import Dataset

    ds = Dataset.from_dict({
        "text": [r["text"] for r in rows],
        "labels": [int(r["label"]) for r in rows],
    })

    def tok(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        enc["labels"] = batch["labels"]
        return enc

    return ds.map(tok, batched=True)


def _metrics_at(gold: list[int], scores: list[float], threshold: float) -> dict[str, float]:
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
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _evaluate_rows(model, tokenizer, rows: list[dict[str, Any]], device: str, batch_size: int):
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score

    scores: list[float] = []
    gold = [int(r["label"]) for r in rows]
    latencies: list[float] = []
    with torch.inference_mode():
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
            probs = torch.softmax(logits.float().cpu(), dim=-1)
            scores.extend(float(row[1]) for row in probs)
            latencies.extend([latency] * len(chunk))

    grid = [round(x / 100, 2) for x in range(5, 96, 5)]
    best_thr = max(grid, key=lambda thr: _metrics_at(gold, scores, thr)["f1"])
    high_recall = [thr for thr in grid if _metrics_at(gold, scores, thr)["recall"] >= 0.90]
    high_recall_thr = max(high_recall) if high_recall else None
    return {
        "examples": len(rows),
        "positive": sum(gold),
        "negative": len(gold) - sum(gold),
        "pr_auc": round(float(average_precision_score(gold, scores)), 4),
        "roc_auc": round(float(roc_auc_score(gold, scores)), 4),
        "at_0_5": _metrics_at(gold, scores, 0.5),
        "best_f1_threshold": best_thr,
        "best_f1": _metrics_at(gold, scores, best_thr),
        "high_recall_threshold": high_recall_thr,
        "high_recall": _metrics_at(gold, scores, high_recall_thr) if high_recall_thr else None,
        "latency_ms": percentiles(latencies),
    }


def train(
    data_dir: Path,
    model_name: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
    val_limit: int,
    seed: int,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from sklearn.metrics import average_precision_score, f1_score
    from torch import nn
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    train_rows = _read_jsonl(data_dir / "train.jsonl")
    val_rows = _read_jsonl(data_dir / "val.jsonl")
    if val_limit and len(val_rows) > val_limit:
        random.Random(seed).shuffle(val_rows)
        val_rows = val_rows[:val_limit]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "SAFE", 1: "INJECTION"},
        label2id={"SAFE": 0, "INJECTION": 1},
        ignore_mismatched_sizes=True,
    )

    pos = max(sum(r["label"] for r in train_rows), 1)
    neg = max(len(train_rows) - pos, 1)
    class_weight = torch.tensor([1.0, min((neg / pos) ** 0.5, 10.0)], dtype=torch.float)

    train_ds = _dataset_from_rows(train_rows, tokenizer, max_length)
    val_ds = _dataset_from_rows(val_rows, tokenizer, max_length)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits).float(), dim=-1).numpy()[:, 1]
        preds = probs >= 0.5
        return {
            "f1": f1_score(labels, preds, zero_division=0),
            "pr_auc": average_precision_score(labels, probs),
        }

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = nn.CrossEntropyLoss(weight=class_weight.to(outputs.logits.device))(
                outputs.logits, labels.long()
            )
            return (loss, outputs) if return_outputs else loss

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.to("cuda")
    per_device = min(batch_size, 24)
    grad_accum = max(1, round(batch_size / per_device))
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device,
        per_device_eval_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        fp16=use_cuda,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=300,
        save_steps=300,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="pr_auc",
        greater_is_better=True,
        logging_steps=50,
        report_to=[],
        seed=seed,
        dataloader_num_workers=2,
    )
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    from transformers.trainer_utils import get_last_checkpoint

    last_ckpt = get_last_checkpoint(str(output_dir)) if output_dir.is_dir() else None
    if last_ckpt:
        print(f"[train] resuming from checkpoint: {last_ckpt}")
    trainer.train(resume_from_checkpoint=last_ckpt)
    metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "finetune_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def evaluate(output_dir: Path, data_dir: Path, batch_size: int) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(output_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(output_dir))
    model.eval()
    if device == "cuda":
        model = model.to("cuda")
    results = {}
    for split in ("val", "test"):
        rows = _read_jsonl(data_dir / f"{split}.jsonl")
        if rows and sum(r["label"] for r in rows) not in (0, len(rows)):
            results[split] = _evaluate_rows(model, tokenizer, rows, device, batch_size)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--data-dir", default="data/prompt_injection_best")
    parser.add_argument("--output", default="models/finetuned/prompt_injection_best")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--augment-per-positive", type=int, default=2)
    parser.add_argument("--neg-pos-ratio", type=float, default=2.5)
    parser.add_argument("--max-train", type=int, default=90000)
    parser.add_argument("--max-per-source", type=int, default=30000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--val-limit", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    data_dir = repo_root() / args.data_dir
    output_dir = repo_root() / args.output
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    build_summary = None
    if not args.train_only:
        build_summary = build_training_data(
            datasets=datasets,
            output_dir=data_dir,
            augment_per_positive=args.augment_per_positive,
            neg_pos_ratio=args.neg_pos_ratio,
            max_train=args.max_train,
            max_per_source=args.max_per_source,
            seed=args.seed,
        )
        print(json.dumps(build_summary["splits"], indent=2))
        if build_summary["failures"]:
            print("[data] skipped sources:")
            for failure in build_summary["failures"]:
                print("  -", failure)
    if args.build_only:
        return

    train_metrics = train(
        data_dir=data_dir,
        model_name=args.model,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        val_limit=args.val_limit,
        seed=args.seed,
    )
    eval_results = evaluate(output_dir, data_dir, args.batch_size)
    rows = [
        [
            split,
            result["examples"],
            result["positive"],
            result["negative"],
            result["pr_auc"],
            result["roc_auc"],
            result["at_0_5"]["f1"],
            result["best_f1"]["f1"],
            result["best_f1_threshold"],
            result["latency_ms"]["p95"],
        ]
        for split, result in eval_results.items()
    ]
    write_report(
        "prompt_injection_best_finetune",
        status=Status.PASS,
        title="Prompt Injection Best Fine-Tune",
        summary={
            "model": args.model,
            "output": str(output_dir),
            "data_dir": str(data_dir),
            "build": build_summary,
            "train_metrics": train_metrics,
            "eval": eval_results,
        },
        tables=[(
            "Evaluation",
            ["Split", "Examples", "Pos", "Neg", "PR-AUC", "ROC-AUC", "F1@0.5", "Best F1", "Best Thr", "p95 ms"],
            rows,
        )],
    )
    print_summary(
        "Prompt Injection Best Fine-Tune",
        Status.PASS,
        ["Split", "Examples", "Pos", "Neg", "PR-AUC", "ROC-AUC", "F1@0.5", "Best F1", "Best Thr", "p95 ms"],
        rows,
    )
    print("[finetune] reports/prompt_injection_best_finetune.{json,md} written")


if __name__ == "__main__":
    main()
