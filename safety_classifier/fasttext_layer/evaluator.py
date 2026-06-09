"""Evaluate FastText heads on their test files.

Exposes a reusable :func:`evaluate_model` (no file IO, used by both the simple
evaluator and the rich bin-vs-ftz comparison reporter) plus :func:`evaluate_all`
which writes the simple per-head JSON reports.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..config import repo_root
from .compat import predict_fasttext

FASTTEXT_DATA_DIR = repo_root() / "data" / "fasttext"
MODELS_DIR = repo_root() / "models" / "fasttext"
REPORTS_DIR = repo_root() / "reports"

_MAX_ERROR_SAMPLES = 25
_THRESHOLDS = tuple(round(0.05 * i, 2) for i in range(1, 19))
_ROUTE_FLOOR = 0.05
_HIGH_RECALL_TARGET = 0.90


def _load_model(path: str):
    import fasttext

    fasttext.FastText.eprint = lambda *a, **k: None  # type: ignore[attr-defined]
    return fasttext.load_model(path)


def _parse_line(line: str) -> tuple[set[str], str]:
    """Return (labels, text) for a native FastText multi-label line."""
    labels: set[str] = set()
    parts = line.split()
    text_start = 0
    for idx, part in enumerate(parts):
        if not part.startswith("__label__"):
            text_start = idx
            break
        labels.add(part[len("__label__"):])
    else:
        text_start = len(parts)
    return labels or {"unknown"}, " ".join(parts[text_start:])


def _read_test(test_file: Path) -> list[tuple[set[str], str]]:
    """Return grouped multi-label examples from a FastText file.

    FastText files duplicate multi-label examples as one line per label. A
    line-by-line top-1 evaluation incorrectly marks the same text wrong when the
    model predicts another valid label for that text. Grouping by text recovers
    the original multi-label gold set.
    """
    grouped: dict[str, set[str]] = {}
    order: list[str] = []
    for line in test_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        labels, text = _parse_line(line)
        if text not in grouped:
            grouped[text] = set()
            order.append(text)
        grouped[text].update(labels)
    return [(grouped[text], text) for text in order]


def _score_label_at_threshold(
    gold_sets: list[set[str]],
    probs: list[dict[str, float]],
    label: str,
    threshold: float,
) -> dict[str, float]:
    tp = fp = fn = 0
    for gold, pred in zip(gold_sets, probs):
        predicted = pred.get(label, 0.0) >= threshold
        actual = label in gold
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": tp + fn,
        "predicted": tp + fp,
    }


def _threshold_sweep(
    labels: list[str],
    gold_sets: list[set[str]],
    probs: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for label in labels:
        sweep = {
            threshold: _score_label_at_threshold(gold_sets, probs, label, threshold)
            for threshold in _THRESHOLDS
        }
        best_f1_threshold = max(sweep, key=lambda t: sweep[t]["f1"])
        high_recall_candidates = [
            threshold
            for threshold in _THRESHOLDS
            if sweep[threshold]["recall"] >= _HIGH_RECALL_TARGET
        ]
        high_recall_threshold = (
            max(high_recall_candidates) if high_recall_candidates else best_f1_threshold
        )
        floor = sweep[_ROUTE_FLOOR]
        high_recall = sweep[high_recall_threshold]
        best_f1 = sweep[best_f1_threshold]
        out[label] = {
            "support": int(best_f1["support"]),
            "best_f1_threshold": best_f1_threshold,
            "best_f1": round(best_f1["f1"], 4),
            "best_f1_precision": round(best_f1["precision"], 4),
            "best_f1_recall": round(best_f1["recall"], 4),
            "high_recall_threshold": high_recall_threshold,
            "high_recall_precision": round(high_recall["precision"], 4),
            "high_recall_recall": round(high_recall["recall"], 4),
            "route_floor_precision": round(floor["precision"], 4),
            "route_floor_recall": round(floor["recall"], 4),
            "route_floor_predicted": int(floor["predicted"]),
        }
    return out


def evaluate_model(model: Any, test_file: Path, threshold: float = 0.50) -> dict[str, Any]:
    """Compute metrics + latency for a loaded model against a test file."""
    from ..reporting import percentiles

    examples = _read_test(test_file)
    labels = sorted({lab[len("__label__"):] for lab in model.get_labels()})
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    fp_samples: list[dict] = []
    fn_samples: list[dict] = []
    latencies: list[float] = []
    gold_sets: list[set[str]] = []
    prob_rows: list[dict[str, float]] = []
    top1_correct = 0
    total = 0

    for gold, text in examples:
        start = time.perf_counter()
        pred_labels, pred_probs = predict_fasttext(model, text, k=-1)
        latencies.append((time.perf_counter() - start) * 1000)
        probs = {
            lab[len("__label__"):]: float(prob)
            for lab, prob in zip(pred_labels, pred_probs)
        }
        gold_sets.append(gold)
        prob_rows.append(probs)
        pred = pred_labels[0][len("__label__"):] if pred_labels else "unknown"
        predicted = {lab for lab, prob in probs.items() if prob >= threshold}
        if not predicted and pred != "unknown":
            predicted = {pred}
        total += 1
        confusion["|".join(sorted(gold))][pred] += 1
        if pred in gold:
            top1_correct += 1
        for lab in labels:
            actual = lab in gold
            is_predicted = lab in predicted
            if is_predicted and actual:
                tp[lab] += 1
            elif is_predicted and not actual:
                fp[lab] += 1
            elif not is_predicted and actual:
                fn[lab] += 1
        if not predicted <= gold or not gold <= predicted:
            if len(fp_samples) < _MAX_ERROR_SAMPLES:
                fp_samples.append({
                    "text": text[:200],
                    "gold": sorted(gold),
                    "predicted": sorted(predicted),
                    "top1": pred,
                })
            if len(fn_samples) < _MAX_ERROR_SAMPLES:
                fn_samples.append({
                    "text": text[:200],
                    "gold": sorted(gold),
                    "predicted": sorted(predicted),
                    "top1": pred,
                })

    per_label: dict[str, dict[str, float]] = {}
    f1s, precs, recs, weights = [], [], [], []
    for lab in labels:
        prec = tp[lab] / (tp[lab] + fp[lab]) if (tp[lab] + fp[lab]) else 0.0
        rec = tp[lab] / (tp[lab] + fn[lab]) if (tp[lab] + fn[lab]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        support = tp[lab] + fn[lab]
        per_label[lab] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        f1s.append(f1); precs.append(prec); recs.append(rec); weights.append(support)

    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    weighted_f1 = sum(f * w for f, w in zip(f1s, weights)) / sum(weights) if sum(weights) else 0.0
    action = _threshold_sweep(labels, gold_sets, prob_rows)
    non_safe_action = [m for lab, m in action.items() if lab != "safe"]
    macro_route_floor_recall = (
        sum(m["route_floor_recall"] for m in non_safe_action) / len(non_safe_action)
        if non_safe_action else 0.0
    )
    macro_best_f1 = (
        sum(m["best_f1"] for m in non_safe_action) / len(non_safe_action)
        if non_safe_action else 0.0
    )

    return {
        "accuracy": round(top1_correct / total, 4) if total else 0.0,
        "top1_accuracy": round(top1_correct / total, 4) if total else 0.0,
        "score_threshold": threshold,
        "macro_precision": round(sum(precs) / len(precs), 4) if precs else 0.0,
        "macro_recall": round(sum(recs) / len(recs), 4) if recs else 0.0,
        "macro_f1": round(macro_f1, 4),
        "macro_best_f1": round(macro_best_f1, 4),
        "macro_route_floor_recall": round(macro_route_floor_recall, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_label": per_label,
        "action_metrics": action,
        "confusion_matrix": {g: dict(p) for g, p in confusion.items()},
        "false_positive_samples": fp_samples,
        "false_negative_samples": fn_samples,
        "total_samples": total,
        "latency_ms": percentiles(latencies),
    }


def evaluate_head(head: str) -> dict[str, Any]:
    """Evaluate the .ftz of a single head; writes reports/fasttext_<head>_eval.json."""
    ftz = MODELS_DIR / f"{head}_head.ftz"
    test_file = FASTTEXT_DATA_DIR / f"{head}_test.txt"
    if not ftz.exists():
        raise FileNotFoundError(f"Model not found: {ftz}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    model = _load_model(str(ftz))
    report = {"head": head, **evaluate_model(model, test_file)}
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / f"fasttext_{head}_eval.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def evaluate_all() -> dict[str, Any]:
    results = {}
    for head in ("attack", "abuse", "high_risk"):
        try:
            results[head] = evaluate_head(head)
            print(
                f"[eval] {head}: acc={results[head]['accuracy']} "
                f"macro_f1={results[head]['macro_f1']}"
            )
        except FileNotFoundError as exc:
            print(f"[eval] skipping {head}: {exc}")
    return results
