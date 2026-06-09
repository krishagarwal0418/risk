#!/usr/bin/env python3
"""Evaluate FastText merged moderation as a BERT traffic gate.

The key question is: how much traffic can skip the moderation BERT while keeping
false passes (unsafe text allowed through the FastText gate) extremely low?
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from safety_classifier import constants as C
from safety_classifier.config import repo_root
from safety_classifier.fasttext_layer.compat import predict_fasttext
from safety_classifier.reporting import ReportSection, Status, percentiles, print_summary, write_report

HARMFUL = "harmful_content"
LABELS = (HARMFUL, C.SEXUAL)
THRESHOLDS = tuple(round(0.01 * i, 2) for i in range(1, 100))


def _load_rows(path: Path, limit: int = 0, seed: int = 13) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit and len(rows) > limit:
        random.Random(seed).shuffle(rows)
        rows = rows[:limit]
    return rows


def _gold(row: dict[str, Any]) -> dict[str, bool]:
    labs = set(row.get("labels") or [])
    return {
        HARMFUL: HARMFUL in labs,
        C.SEXUAL: C.SEXUAL in labs,
        "unsafe": bool(labs),
    }


def _predict_scores(model: Any, text: str) -> dict[str, float]:
    labels, probs = predict_fasttext(model, " ".join(text.split()), k=-1)
    out = {HARMFUL: 0.0, C.SEXUAL: 0.0, C.SAFE: 0.0}
    for lab, prob in zip(labels, probs):
        key = lab[len("__label__"):] if lab.startswith("__label__") else lab
        if key in out:
            out[key] = float(prob)
    return out


def _binary_metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _score_rows(rows: list[dict[str, Any]], model: Any) -> tuple[list[dict[str, Any]], dict[str, float]]:
    scored: list[dict[str, Any]] = []
    latencies: list[float] = []
    for row in rows:
        start = time.perf_counter()
        scores = _predict_scores(model, row["text"])
        latencies.append((time.perf_counter() - start) * 1000)
        scored.append({"gold": _gold(row), "scores": scores, "text": row["text"]})
    return scored, percentiles(latencies)


def _gate_metrics(scored: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    total = len(scored)
    unsafe_total = sum(1 for r in scored if r["gold"]["unsafe"])
    safe_total = total - unsafe_total
    passed = []
    routed = []
    for row in scored:
        scores = row["scores"]
        should_route = max(scores.get(HARMFUL, 0.0), scores.get(C.SEXUAL, 0.0)) >= threshold
        (routed if should_route else passed).append(row)

    false_pass = sum(1 for r in passed if r["gold"]["unsafe"])
    true_pass = sum(1 for r in passed if not r["gold"]["unsafe"])
    routed_unsafe = sum(1 for r in routed if r["gold"]["unsafe"])
    routed_safe = sum(1 for r in routed if not r["gold"]["unsafe"])
    route = _binary_metrics(routed_unsafe, routed_safe, false_pass)
    return {
        "threshold": threshold,
        "total": total,
        "safe_total": safe_total,
        "unsafe_total": unsafe_total,
        "passed": len(passed),
        "routed": len(routed),
        "pass_pct": round(len(passed) / total, 4) if total else 0.0,
        "bert_call_pct": round(len(routed) / total, 4) if total else 0.0,
        "false_pass": false_pass,
        "true_pass": true_pass,
        "unsafe_miss_rate": round(false_pass / unsafe_total, 4) if unsafe_total else 0.0,
        "false_pass_rate_all": round(false_pass / total, 4) if total else 0.0,
        "safe_pass_rate": round(true_pass / safe_total, 4) if safe_total else 0.0,
        "safe_routed_to_bert": routed_safe,
        "route_precision": route["precision"],
        "route_recall": route["recall"],
        "route_f1": route["f1"],
    }


def _per_label_metrics(scored: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label in LABELS:
        tp = fp = fn = tn = 0
        for row in scored:
            pred = row["scores"].get(label, 0.0) >= threshold
            gold = row["gold"][label]
            if pred and gold:
                tp += 1
            elif pred and not gold:
                fp += 1
            elif not pred and gold:
                fn += 1
            else:
                tn += 1
        metrics = _binary_metrics(tp, fp, fn)
        out[label] = {
            **metrics,
            "support": tp + fn,
            "predicted": tp + fp,
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        }
    return out


def evaluate(
    model_path: Path,
    data_path: Path,
    *,
    limit: int,
    seed: int,
    target_miss_rate: float,
) -> dict[str, Any]:
    import fasttext

    fasttext.FastText.eprint = lambda *a, **k: None  # type: ignore[attr-defined]
    model = fasttext.load_model(str(model_path))
    rows = _load_rows(data_path, limit=limit, seed=seed)
    scored, latency = _score_rows(rows, model)
    sweep = [_gate_metrics(scored, thr) for thr in THRESHOLDS]
    viable = [m for m in sweep if m["unsafe_miss_rate"] <= target_miss_rate]
    recommended = max(viable, key=lambda m: (m["pass_pct"], m["threshold"])) if viable else sweep[0]
    return {
        "model_path": str(model_path),
        "data_path": str(data_path),
        "rows": len(rows),
        "target_miss_rate": target_miss_rate,
        "latency_ms": latency,
        "recommended": recommended,
        "per_label_at_recommended": _per_label_metrics(scored, recommended["threshold"]),
        "threshold_sweep": sweep,
        "unsafe_false_pass_samples": [
            {
                "text": row["text"][:240],
                "scores": {
                    HARMFUL: round(row["scores"].get(HARMFUL, 0.0), 4),
                    C.SEXUAL: round(row["scores"].get(C.SEXUAL, 0.0), 4),
                    C.SAFE: round(row["scores"].get(C.SAFE, 0.0), 4),
                },
            }
            for row in scored
            if row["gold"]["unsafe"]
            and max(row["scores"].get(HARMFUL, 0.0), row["scores"].get(C.SEXUAL, 0.0))
            < recommended["threshold"]
        ][:25],
    }


def _pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate merged FastText moderation as traffic gate")
    p.add_argument("--model", default="models/fasttext/merged_moderation_head.ftz")
    p.add_argument("--data", default="data/koala_merged_moderation/test.jsonl")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--target-miss-rate", type=float, default=0.005,
                   help="Max unsafe false-pass rate among unsafe rows. 0.005 = 0.5%%.")
    args = p.parse_args()

    root = repo_root()
    model_path = Path(args.model)
    data_path = Path(args.data)
    if not model_path.is_absolute():
        model_path = root / model_path
    if not data_path.is_absolute():
        data_path = root / data_path
    result = evaluate(
        model_path,
        data_path,
        limit=args.limit,
        seed=args.seed,
        target_miss_rate=args.target_miss_rate,
    )
    rec = result["recommended"]
    status = Status.PASS if rec["unsafe_miss_rate"] <= args.target_miss_rate else Status.WARN
    warnings = [] if status == Status.PASS else [
        f"No threshold met target miss rate {args.target_miss_rate}; using lowest threshold."
    ]
    sweep_rows = [
        [
            m["threshold"],
            _pct(m["pass_pct"]),
            _pct(m["unsafe_miss_rate"]),
            m["false_pass"],
            _pct(m["route_precision"]),
            _pct(m["route_recall"]),
            _pct(m["route_f1"]),
            _pct(m["bert_call_pct"]),
        ]
        for m in result["threshold_sweep"]
        if m["threshold"] in (0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
    ]
    label_rows = [
        [
            label,
            m["support"],
            m["predicted"],
            _pct(m["precision"]),
            _pct(m["recall"]),
            _pct(m["f1"]),
        ]
        for label, m in result["per_label_at_recommended"].items()
    ]
    section = ReportSection("Recommended Gate")
    section.add("threshold", rec["threshold"])
    section.add("traffic_pass_without_bert", _pct(rec["pass_pct"]))
    section.add("bert_call_rate", _pct(rec["bert_call_pct"]))
    section.add("unsafe_false_pass_rate", _pct(rec["unsafe_miss_rate"]))
    section.add("false_pass_count", rec["false_pass"])
    section.add("routing_precision", _pct(rec["route_precision"]))
    section.add("routing_recall", _pct(rec["route_recall"]))
    section.add("routing_f1", _pct(rec["route_f1"]))
    section.add("p95_latency_ms", result["latency_ms"]["p95"])
    write_report(
        "fasttext_merged_moderation_gate_eval",
        status=status,
        title="FastText Merged Moderation Gate Evaluation",
        summary=result,
        sections=[section],
        tables=[
            (
                "Threshold Sweep",
                [
                    "Threshold",
                    "Pass w/o BERT",
                    "Unsafe Miss",
                    "False Pass",
                    "Route Precision",
                    "Route Recall",
                    "Route F1",
                    "BERT Calls",
                ],
                sweep_rows,
            ),
            (
                "Per-Label At Recommended Threshold",
                ["Label", "Support", "Predicted", "Precision", "Recall", "F1"],
                label_rows,
            ),
        ],
        warnings=warnings,
    )
    print_summary(
        "FastText Merged Moderation Gate Evaluation",
        status,
        ["Threshold", "Pass w/o BERT", "Unsafe Miss", "Route Precision", "Route Recall", "p95 ms"],
        [[
            rec["threshold"],
            _pct(rec["pass_pct"]),
            _pct(rec["unsafe_miss_rate"]),
            _pct(rec["route_precision"]),
            _pct(rec["route_recall"]),
            result["latency_ms"]["p95"],
        ]],
    )
    print("[eval] reports/fasttext_merged_moderation_gate_eval.{json,md} written")


if __name__ == "__main__":
    main()
