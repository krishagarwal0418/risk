"""Rich FastText evaluation: evaluates .bin and .ftz, compares them, and writes
per-head ``reports/fasttext_<head>_eval.{json,md}`` with PASS/WARN/FAIL status.

Warn if quantization drops macro F1 by more than 3 percentage points.
Fail if the quantized model cannot load.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import repo_root
from ..fasttext_layer.evaluator import _load_model, evaluate_model
from ..reporting import ReportSection, Status, file_size, human_size, print_summary, write_report

FASTTEXT_DATA_DIR = repo_root() / "data" / "fasttext"
MODELS_DIR = repo_root() / "models" / "fasttext"

MACRO_F1_DROP_WARN = 0.03  # 3 percentage points

HEADS = ("attack", "abuse", "high_risk")


def evaluate_head_report(head: str) -> dict[str, Any]:
    bin_path = MODELS_DIR / f"{head}_head.bin"
    ftz_path = MODELS_DIR / f"{head}_head.ftz"
    test_file = FASTTEXT_DATA_DIR / f"{head}_test.txt"

    warnings: list[str] = []
    failures: list[str] = []

    if not test_file.exists():
        failures.append(f"test file missing: {test_file.name}")
    if not ftz_path.exists():
        failures.append(f"quantized model missing: {ftz_path.name}")

    bin_metrics: dict[str, Any] | None = None
    ftz_metrics: dict[str, Any] | None = None

    if test_file.exists():
        if bin_path.exists():
            try:
                bin_metrics = evaluate_model(_load_model(str(bin_path)), test_file)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f".bin evaluation failed: {exc}")
        if ftz_path.exists():
            try:
                ftz_metrics = evaluate_model(_load_model(str(ftz_path)), test_file)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"quantized .ftz model cannot load/evaluate: {exc}")

    # Comparison.
    comparison: dict[str, Any] = {}
    if bin_metrics and ftz_metrics:
        macro_drop = bin_metrics["macro_f1"] - ftz_metrics["macro_f1"]
        comparison = {
            "accuracy_diff": round(bin_metrics["accuracy"] - ftz_metrics["accuracy"], 4),
            "macro_f1_diff": round(macro_drop, 4),
            "per_label_f1_diff": {
                lab: round(
                    bin_metrics["per_label"].get(lab, {}).get("f1", 0)
                    - ftz_metrics["per_label"].get(lab, {}).get("f1", 0),
                    4,
                )
                for lab in ftz_metrics["per_label"]
            },
            "latency_diff_ms": round(
                ftz_metrics["latency_ms"]["avg"] - bin_metrics["latency_ms"]["avg"], 4
            ),
        }
        if macro_drop > MACRO_F1_DROP_WARN:
            warnings.append(
                f"quantization dropped macro F1 by {macro_drop*100:.1f}pp (>3pp)"
            )

    bin_size = file_size(bin_path)
    ftz_size = file_size(ftz_path)
    size_reduction = (
        round(100 * (1 - ftz_size / bin_size), 1) if (bin_size and ftz_size) else None
    )

    status = Status.FAIL if failures else (Status.WARN if warnings else Status.PASS)

    summary = {
        "head": head,
        "bin_metrics": bin_metrics,
        "ftz_metrics": ftz_metrics,
        "comparison": comparison,
        "sizes": {
            "bin_bytes": bin_size,
            "ftz_bytes": ftz_size,
            "size_reduction_pct": size_reduction,
        },
    }

    # Markdown content.
    sections = []
    if ftz_metrics:
        overview = ReportSection("Quantized (.ftz) Overview")
        overview.add("accuracy", ftz_metrics["accuracy"])
        overview.add("macro_precision", ftz_metrics["macro_precision"])
        overview.add("macro_recall", ftz_metrics["macro_recall"])
        overview.add("macro_f1_at_0.50", ftz_metrics["macro_f1"])
        overview.add("macro_best_f1_non_safe", ftz_metrics.get("macro_best_f1", "n/a"))
        overview.add(
            "macro_action_recall_at_0.05_non_safe",
            ftz_metrics.get("macro_route_floor_recall", "n/a"),
        )
        overview.add("top1_any_true_label_accuracy", ftz_metrics.get("top1_accuracy", "n/a"))
        overview.add("weighted_f1", ftz_metrics["weighted_f1"])
        overview.add("avg_latency_ms", ftz_metrics["latency_ms"]["avg"])
        overview.add("p95_latency_ms", ftz_metrics["latency_ms"]["p95"])
        overview.add("p99_latency_ms", ftz_metrics["latency_ms"]["p99"])
        overview.add("bin_size", human_size(bin_size))
        overview.add("ftz_size", human_size(ftz_size))
        overview.add("size_reduction", f"{size_reduction}%" if size_reduction else "n/a")
        sections.append(overview)

    tables = []
    if ftz_metrics:
        per_label_rows = [
            [lab, m["precision"], m["recall"], m["f1"], m["support"]]
            for lab, m in ftz_metrics["per_label"].items()
        ]
        tables.append(
            (
                "Per-Label at score >= 0.50 (.ftz)",
                ["Label", "Precision", "Recall", "F1", "Support"],
                per_label_rows,
            )
        )
        action_rows = [
            [
                lab,
                m["support"],
                m["route_floor_recall"],
                m["route_floor_precision"],
                m["route_floor_predicted"],
                m["high_recall_threshold"],
                m["high_recall_recall"],
                m["high_recall_precision"],
                m["best_f1_threshold"],
                m["best_f1"],
            ]
            for lab, m in ftz_metrics.get("action_metrics", {}).items()
            if lab != "safe"
        ]
        tables.append((
            "Action / Routing Metrics (.ftz)",
            [
                "Label", "Support", "Recall@0.05", "Precision@0.05", "Pred@0.05",
                "High-recall thr", "Recall@HR", "Precision@HR", "Best-F1 thr", "Best F1",
            ],
            action_rows,
        ))
    if comparison:
        cmp_rows = [
            ["accuracy_diff", comparison["accuracy_diff"]],
            ["macro_f1_diff (bin - ftz)", comparison["macro_f1_diff"]],
            ["latency_diff_ms (ftz - bin)", comparison["latency_diff_ms"]],
        ]
        tables.append(("bin vs ftz", ["Metric", "Value"], cmp_rows))

    write_report(
        f"fasttext_{head}_eval",
        status=status,
        title=f"FastText {head} Head Evaluation",
        summary=summary,
        sections=sections,
        tables=tables,
        warnings=warnings,
        failures=failures,
    )
    macro = ftz_metrics.get("macro_best_f1", ftz_metrics["macro_f1"]) if ftz_metrics else "n/a"
    action_recall = ftz_metrics.get("macro_route_floor_recall", "n/a") if ftz_metrics else "n/a"
    print_summary(
        f"FastText {head} head", status,
        ["Metric", "Value"],
        [
            ["macro_best_f1_non_safe", macro],
            ["macro_action_recall@0.05_non_safe", action_recall],
            ["size_reduction", f"{size_reduction}%" if size_reduction else "n/a"],
        ],
    )
    return summary


def evaluate_all_heads_report() -> dict[str, Any]:
    return {head: evaluate_head_report(head) for head in HEADS}
