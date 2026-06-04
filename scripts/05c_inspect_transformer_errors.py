"""Inspect transformer baseline errors for one model/label.

This is intended for Colab debugging after ``scripts/05b_eval_transformers_baseline.py``.
It reruns the selected model on the same deterministic sample and writes compact
error examples to ``reports/transformer_error_inspection.{json,md}``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

import _bootstrap  # noqa: F401

from safety_classifier.config import load_models_config, repo_root
from safety_classifier.evaluation.transformer_eval import MODEL_TARGETS, _load_test
from safety_classifier.reporting import Status, print_summary, write_report


REPORTS_DIR = repo_root() / "reports"


def _truncate(text: str, n: int = 260) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3] + "..."


def _threshold_from_baseline(model_key: str, label: str, fallback: float) -> float:
    path = REPORTS_DIR / "transformer_baseline_eval.json"
    if not path.exists():
        return fallback
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        metric = report["models"][model_key]["per_label"][label]
        return float(metric.get("best_f1_threshold", fallback))
    except Exception:  # noqa: BLE001
        return fallback


def _bucket(score: float) -> str:
    if score >= 0.9:
        return "0.90-1.00"
    if score >= 0.75:
        return "0.75-0.90"
    if score >= 0.5:
        return "0.50-0.75"
    if score >= 0.25:
        return "0.25-0.50"
    if score >= 0.1:
        return "0.10-0.25"
    return "0.00-0.10"


def _load_model(model_key: str, device: str):
    from safety_classifier import transformers_layer as TL

    cls_name, cfg_key, _targets = MODEL_TARGETS[model_key]
    cfg = load_models_config().get("transformers", {}).get(cfg_key)
    if not cfg:
        raise RuntimeError(f"no transformer config for {model_key}")
    cls = getattr(TL, cls_name)
    return cls(cfg["hf_name"], backend="pytorch", device=device, name=cfg["hf_name"]), cfg


def inspect_errors(
    model_key: str,
    label: str,
    device: str = "cuda",
    limit: int | None = 10000,
    batch_size: int = 64,
    threshold: float | None = None,
    examples: int = 20,
) -> dict[str, Any]:
    if model_key not in MODEL_TARGETS:
        raise ValueError(f"unknown model_key: {model_key}")
    targets = MODEL_TARGETS[model_key][2]
    if label not in targets:
        raise ValueError(f"{model_key} does not evaluate label {label}; targets={targets}")

    threshold = threshold if threshold is not None else _threshold_from_baseline(
        model_key, label, fallback=0.5
    )
    rows = _load_test(limit, targets=targets)
    model, cfg = _load_model(model_key, device=device)

    scored: list[dict[str, Any]] = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        results = model.classify_batch([r["text"] for r in chunk])
        for row, result in zip(chunk, results):
            labels = set(row.get("labels", []))
            score = float(result["scores"].get(label, 0.0))
            gold = label in labels
            pred = score >= threshold
            scored.append({
                "text": row.get("text", ""),
                "snippet": _truncate(row.get("text", "")),
                "source": row.get("source", "unknown"),
                "labels": sorted(labels),
                "score": round(score, 6),
                "gold": gold,
                "pred": pred,
            })

    tp = [r for r in scored if r["gold"] and r["pred"]]
    fp = [r for r in scored if not r["gold"] and r["pred"]]
    fn = [r for r in scored if r["gold"] and not r["pred"]]
    tn = [r for r in scored if not r["gold"] and not r["pred"]]

    score_buckets = Counter(_bucket(r["score"]) for r in scored)
    positive_sources = Counter(r["source"] for r in scored if r["gold"])
    fp_sources = Counter(r["source"] for r in fp)
    fn_sources = Counter(r["source"] for r in fn)

    top_false_positives = sorted(fp, key=lambda r: r["score"], reverse=True)[:examples]
    low_score_positives = sorted([r for r in scored if r["gold"]], key=lambda r: r["score"])[:examples]

    precision = len(tp) / (len(tp) + len(fp)) if (tp or fp) else 0.0
    recall = len(tp) / (len(tp) + len(fn)) if (tp or fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    summary = {
        "model_key": model_key,
        "hf_name": cfg["hf_name"],
        "label": label,
        "threshold": threshold,
        "test_examples": len(scored),
        "metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": len(tp),
            "fp": len(fp),
            "fn": len(fn),
            "tn": len(tn),
        },
        "score_buckets": dict(score_buckets),
        "positive_sources": dict(positive_sources.most_common(20)),
        "false_positive_sources": dict(fp_sources.most_common(20)),
        "false_negative_sources": dict(fn_sources.most_common(20)),
        "top_false_positives": top_false_positives,
        "low_score_positives": low_score_positives,
    }

    example_rows = [
        [r["score"], r["source"], ",".join(r["labels"]), r["snippet"]]
        for r in top_false_positives[:10]
    ]
    low_positive_rows = [
        [r["score"], r["source"], ",".join(r["labels"]), r["snippet"]]
        for r in low_score_positives[:10]
    ]
    write_report(
        "transformer_error_inspection",
        status=Status.PASS,
        title="Transformer Error Inspection",
        summary=summary,
        tables=[
            ("Top False Positives", ["Score", "Source", "Gold Labels", "Text"], example_rows),
            ("Lowest-Score Positives", ["Score", "Source", "Gold Labels", "Text"], low_positive_rows),
        ],
    )
    print_summary(
        "Transformer Error Inspection",
        Status.PASS,
        ["Metric", "Value"],
        [
            ["model", cfg["hf_name"]],
            ["label", label],
            ["threshold", threshold],
            ["precision", round(precision, 4)],
            ["recall", round(recall, 4)],
            ["f1", round(f1, 4)],
            ["tp/fp/fn/tn", f"{len(tp)}/{len(fp)}/{len(fn)}/{len(tn)}"],
        ],
    )
    print("[inspect] reports/transformer_error_inspection.{json,md} written")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-key", default="prompt_injection", choices=sorted(MODEL_TARGETS))
    p.add_argument("--label", default="prompt_injection")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--limit", type=int, default=10000, help="max test examples (0 = all)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--examples", type=int, default=20)
    args = p.parse_args()
    inspect_errors(
        model_key=args.model_key,
        label=args.label,
        device=args.device,
        limit=args.limit or None,
        batch_size=args.batch_size,
        threshold=args.threshold,
        examples=args.examples,
    )


if __name__ == "__main__":
    main()
