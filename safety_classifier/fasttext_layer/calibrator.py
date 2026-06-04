"""Threshold calibration for FastText heads.

Sweeps thresholds 0.05..0.95 per label on the validation file and records the
best-F1, high-recall and high-precision operating points. For routing we prefer
high recall (we'd rather over-route to a transformer than miss an attack).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from ..config import repo_root
from .compat import predict_fasttext
from .evaluator import _parse_line

FASTTEXT_DATA_DIR = repo_root() / "data" / "fasttext"
MODELS_DIR = repo_root() / "models" / "fasttext"
REPORTS_DIR = repo_root() / "reports"

_THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 19)]  # 0.05 .. 0.90
_HIGH_RECALL_TARGET = 0.90
_HIGH_PRECISION_TARGET = 0.90


def _score_at_threshold(
    gold: list[str], probs: list[dict[str, float]], label: str, thr: float
) -> dict[str, float]:
    tp = fp = fn = 0
    for g, p in zip(gold, probs):
        predicted = p.get(label, 0.0) >= thr
        actual = g == label
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def calibrate_head(head: str) -> dict[str, Any]:
    import fasttext

    fasttext.FastText.eprint = lambda *a, **k: None  # type: ignore[attr-defined]
    ftz = MODELS_DIR / f"{head}_head.ftz"
    val_file = FASTTEXT_DATA_DIR / f"{head}_val.txt"
    if not ftz.exists() or not val_file.exists():
        raise FileNotFoundError(f"Missing model or val file for head '{head}'")

    model = fasttext.load_model(str(ftz))
    labels = sorted({lab[len("__label__"):] for lab in model.get_labels()})

    gold: list[str] = []
    probs: list[dict[str, float]] = []
    for line in val_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        g, text = _parse_line(line)
        gold.append(g)
        pred_labels, pred_probs = predict_fasttext(model, text, k=-1)
        probs.append(
            {l[len("__label__"):]: float(p) for l, p in zip(pred_labels, pred_probs)}
        )

    recommendations: dict[str, dict[str, Any]] = {}
    for label in labels:
        sweep = {
            thr: _score_at_threshold(gold, probs, label, thr) for thr in _THRESHOLDS
        }
        best_f1_thr = max(sweep, key=lambda t: sweep[t]["f1"])
        # High recall: lowest threshold that still hits the recall target.
        high_recall_thr = next(
            (t for t in _THRESHOLDS if sweep[t]["recall"] >= _HIGH_RECALL_TARGET),
            best_f1_thr,
        )
        # High precision: highest threshold meeting precision target.
        high_prec_candidates = [
            t for t in _THRESHOLDS if sweep[t]["precision"] >= _HIGH_PRECISION_TARGET
        ]
        high_precision_thr = max(high_prec_candidates) if high_prec_candidates else best_f1_thr
        recommendations[label] = {
            "best_f1_threshold": best_f1_thr,
            "best_f1": round(sweep[best_f1_thr]["f1"], 4),
            "high_recall_threshold": high_recall_thr,
            "high_precision_threshold": high_precision_thr,
            "sweep": {str(t): {k: round(v, 4) for k, v in s.items()} for t, s in sweep.items()},
        }
    return {"head": head, "recommendations": recommendations}


def calibrate_all() -> dict[str, Any]:
    """Calibrate every head and write recommended route thresholds (high recall)."""
    from ..reporting import ReportSection, Status, print_summary, write_report

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    route_thresholds: dict[str, float] = {}
    full: dict[str, Any] = {}
    warnings: list[str] = []
    for head in ("attack", "abuse", "high_risk"):
        try:
            result = calibrate_head(head)
        except FileNotFoundError as exc:
            warnings.append(f"skipped {head}: {exc}")
            print(f"[calibrate] skipping {head}: {exc}")
            continue
        full[head] = result
        for label, rec in result["recommendations"].items():
            if label == "safe":
                continue
            route_thresholds[label] = rec["high_recall_threshold"]

    # Recommended route thresholds the router can load directly.
    (REPORTS_DIR / "fasttext_thresholds_recommended.yaml").write_text(
        yaml.safe_dump({k: {"route": v} for k, v in route_thresholds.items()}, sort_keys=True),
        encoding="utf-8",
    )

    status = Status.PASS if route_thresholds else Status.WARN
    table_rows = []
    for head, result in full.items():
        for label, rec in result["recommendations"].items():
            if label == "safe":
                continue
            table_rows.append([
                head, label, rec["best_f1_threshold"], rec["best_f1"],
                rec["high_recall_threshold"], rec["high_precision_threshold"],
                rec["high_recall_threshold"],
            ])

    write_report(
        "fasttext_threshold_calibration",
        status=status,
        title="FastText Threshold Calibration",
        summary={"route_thresholds": route_thresholds, "detail": full},
        sections=[_route_section(route_thresholds)],
        tables=[(
            "Per-Label Calibration",
            ["Head", "Label", "Best-F1 thr", "Best F1", "High-Recall thr", "High-Prec thr", "Recommended"],
            table_rows,
        )],
        warnings=warnings,
    )
    print_summary(
        "FastText Threshold Calibration", status,
        ["Label", "Recommended route threshold"],
        [[k, v] for k, v in route_thresholds.items()],
    )
    return {"route_thresholds": route_thresholds, "detail": full}


def _route_section(route_thresholds: dict[str, float]):
    from ..reporting import ReportSection

    section = ReportSection("Recommended Route Thresholds (high recall)")
    for label, thr in route_thresholds.items():
        section.add(label, thr)
    return section
