"""Evaluate the local mod_tests pipeline on processed data.

Pipeline under test:
  validation -> normalization -> deterministic rules -> FastText heads ->
  mod_tests/protectai-deberta-v3-base-prompt-injection + mod_tests/toxic-bert
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Any

import _bootstrap  # noqa: F401

from safety_classifier import constants as C
from safety_classifier.config import repo_root
from safety_classifier.evaluation.transformer_eval import _load_test, _threshold_metrics
from safety_classifier.mod_tests_pipeline import ModTestsSafetyPipeline
from safety_classifier.reporting import Status, percentiles, print_summary, write_report


TARGET_LABELS = [C.PROMPT_INJECTION, C.TOXICITY, C.HATE, C.VIOLENCE]


def _score_for(result: dict[str, Any], label: str) -> float:
    if label in C.PUBLIC_SCORE_LABELS:
        return float(result["scores"].get(label, 0.0))
    return float(result["internal_scores"].get(label, 0.0))


def _source_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(row.get("source", "unknown") for row in rows).most_common(20))


def evaluate_mod_tests_pipeline(
    *,
    device: str = "cuda",
    limit: int | None = 2000,
    full_scan: bool = True,
    use_fasttext: bool = True,
) -> dict[str, Any]:
    rows = _load_test(limit, targets=TARGET_LABELS)
    if not rows:
        raise RuntimeError(
            "No processed test rows found. Run scripts/02_prepare_fasttext_data.py first."
        )
    clf = ModTestsSafetyPipeline(
        device=device,
        use_fasttext=use_fasttext,
        full_scan_default=full_scan,
    )

    gold: dict[str, list[int]] = {label: [] for label in TARGET_LABELS}
    scores: dict[str, list[float]] = {label: [] for label in TARGET_LABELS}
    total_latencies: list[float] = []
    stage_latencies: dict[str, list[float]] = {
        "validation": [],
        "normalization": [],
        "deterministic_rules": [],
        "fasttext": [],
        "prompt_guard": [],
        "toxic_bert": [],
        "total": [],
    }
    triggered_counts: Counter = Counter()
    decision_counts: Counter = Counter()

    for row in rows:
        result = clf.classify(row["text"], full_scan=full_scan, include_raw=False)
        labels = set(row.get("labels", []))
        for label in TARGET_LABELS:
            gold[label].append(1 if label in labels else 0)
            scores[label].append(_score_for(result, label))
        total_latencies.append(float(result["latency_ms"]))
        for stage, latency in result["stage_latency_ms"].items():
            stage_latencies.setdefault(stage, []).append(float(latency))
        for model in result["triggered_models"]:
            triggered_counts[model] += 1
        decision_counts[result["decision"]] += 1

    per_label = {
        label: _threshold_metrics(gold[label], scores[label])
        for label in TARGET_LABELS
        if sum(gold[label]) > 0
    }
    macro_best_f1 = (
        round(sum(m["best_f1"] for m in per_label.values()) / len(per_label), 4)
        if per_label else None
    )
    pr_auc_values = [m["pr_auc"] for m in per_label.values() if "pr_auc" in m]
    macro_pr_auc = (
        round(sum(pr_auc_values) / len(pr_auc_values), 4)
        if pr_auc_values else None
    )
    latency_summary = {
        "total": percentiles(total_latencies),
        "by_stage": {
            stage: percentiles(values)
            for stage, values in stage_latencies.items()
            if values
        },
    }
    summary = {
        "mode": "full_scan" if full_scan else "routed",
        "device": device,
        "use_fasttext": use_fasttext,
        "test_examples": len(rows),
        "source_distribution": _source_counts(rows),
        "models": {
            "prompt_guard": str(clf.prompt_guard_path),
            "toxic_bert": str(clf.toxic_bert_path),
        },
        "macro_best_f1": macro_best_f1,
        "macro_pr_auc": macro_pr_auc,
        "per_label": per_label,
        "latency_ms": latency_summary,
        "triggered_models": dict(triggered_counts),
        "decision_distribution": dict(decision_counts),
    }

    warnings = []
    if macro_best_f1 is not None and macro_best_f1 < 0.5:
        warnings.append(f"low macro best-F1 ({macro_best_f1})")
    status = Status.WARN if warnings else Status.PASS

    metric_rows = [
        [
            label,
            m["support_positive"],
            m["f1"],
            m["best_f1"],
            m["best_f1_threshold"],
            m.get("pr_auc", "n/a"),
        ]
        for label, m in per_label.items()
    ]
    latency_rows = [
        [stage, vals["avg"], vals["p50"], vals["p95"], vals["p99"]]
        for stage, vals in latency_summary["by_stage"].items()
    ]
    write_report(
        "mod_tests_pipeline_eval",
        status=status,
        title="mod_tests Pipeline Evaluation",
        summary=summary,
        tables=[
            (
                "Per-Label Metrics",
                ["Label", "Support", "F1@0.5", "Best F1", "Best Thr", "PR-AUC"],
                metric_rows,
            ),
            ("Latency", ["Stage", "Avg ms", "P50 ms", "P95 ms", "P99 ms"], latency_rows),
        ],
        warnings=warnings,
    )
    print_summary(
        "mod_tests Pipeline",
        status,
        ["Label", "Support", "F1@0.5", "Best F1", "Best Thr", "PR-AUC"],
        metric_rows,
    )
    print_summary(
        "mod_tests Latency",
        status,
        ["Stage", "Avg ms", "P50 ms", "P95 ms", "P99 ms"],
        latency_rows,
    )
    print("[mod_tests] report written to reports/mod_tests_pipeline_eval.{json,md}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--limit", type=int, default=2000, help="max test examples (0 = all)")
    p.add_argument("--routed", action="store_true", help="route models instead of full scan")
    p.add_argument("--no-fasttext", action="store_true")
    args = p.parse_args()
    evaluate_mod_tests_pipeline(
        device=args.device,
        limit=args.limit or None,
        full_scan=not args.routed,
        use_fasttext=not args.no_fasttext,
    )


if __name__ == "__main__":
    main()
