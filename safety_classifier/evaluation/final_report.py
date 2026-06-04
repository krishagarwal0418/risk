"""Combine all stage reports into reports/final_pipeline_report.{json,md}.

Reads whatever stage reports exist (missing ones are marked NOT-RUN) and produces
a single end-to-end summary: component status table, answers to the 17 questions,
and a recommended deployment mode.
"""

from __future__ import annotations

from typing import Any

from ..reporting import (
    ReportSection,
    Status,
    load_report,
    markdown_table,
    print_summary,
    write_report,
)

REPORTS_DIR = "reports"


def _status_of(report: dict[str, Any] | None) -> str:
    return report.get("status", "NOT-RUN") if report else "NOT-RUN"


def _fastest_mode(benchmark: dict[str, Any] | None) -> str | None:
    if not benchmark:
        return None
    scenarios = benchmark.get("scenarios", {})
    routed = {k: v for k, v in scenarios.items() if k.startswith("routed")}
    if not routed:
        return None
    return min(routed, key=lambda k: routed[k]["avg"])


def _recommend_backend(quant: dict[str, Any] | None) -> str:
    if quant and quant.get("status") in ("PASS", "WARN"):
        return (
            "Production CPU: ONNX INT8 (smallest + fastest on CPU). "
            "Colab/GPU: PyTorch CUDA. Local CPU dev: FP32 ONNX."
        )
    return "PyTorch CUDA on Colab/GPU; PyTorch CPU locally until ONNX export succeeds."


def _weak_labels(*fasttext_reports: dict[str, Any] | None) -> list[str]:
    weak: list[str] = []
    for rep in fasttext_reports:
        if not rep:
            continue
        ftz = rep.get("ftz_metrics") or {}
        for lab, m in (ftz.get("per_label") or {}).items():
            if lab != "safe" and m.get("f1", 1.0) < 0.6:
                weak.append(f"{rep.get('head')}::{lab} (F1={m['f1']})")
    return weak


def generate_final_report() -> dict[str, Any]:
    download = load_report("data_download_report")
    distribution = load_report("data_distribution_report")
    ft_attack = load_report("fasttext_attack_eval")
    ft_abuse = load_report("fasttext_abuse_eval")
    ft_high = load_report("fasttext_high_risk_eval")
    calibration = load_report("fasttext_threshold_calibration")
    baseline = load_report("transformer_baseline_eval")
    onnx_export = load_report("onnx_export_report")
    quantization = load_report("quantization_report")
    benchmark = load_report("benchmark_report")

    # Component status table.
    def macro(rep):
        return (rep.get("ftz_metrics") or {}).get("macro_f1", "n/a") if rep else "n/a"

    component_rows = [
        ["Data download", _status_of(download),
         f"{(download or {}).get('available_count', '?')}/{(download or {}).get('total_count', '?')} datasets" if download else "not run"],
        ["Data distribution", _status_of(distribution),
         f"{(distribution or {}).get('global', {}).get('total_usable_rows', '?')} usable rows" if distribution else "not run"],
        ["FastText attack head", _status_of(ft_attack), f"macro F1 {macro(ft_attack)}"],
        ["FastText abuse head", _status_of(ft_abuse), f"macro F1 {macro(ft_abuse)}"],
        ["FastText high-risk head", _status_of(ft_high), f"macro F1 {macro(ft_high)}"],
        ["Threshold calibration", _status_of(calibration), "route thresholds tuned"],
        ["Transformer baselines", _status_of(baseline), "see report"],
        ["ONNX export", _status_of(onnx_export),
         f"{(onnx_export or {}).get('exported_ok', '?')}/{(onnx_export or {}).get('total', '?')} exported" if onnx_export else "not run"],
        ["INT8 quantization", _status_of(quantization), "see report"],
        ["Routed pipeline", _status_of(benchmark),
         f"fastest routed: {_fastest_mode(benchmark)}" if benchmark else "not run"],
    ]

    statuses = [r[1] for r in component_rows if r[1] != "NOT-RUN"]
    overall = Status.worst([s for s in statuses]) if statuses else Status.WARN

    weak = _weak_labels(ft_attack, ft_abuse, ft_high)

    # 17 questions.
    qa = {
        "1_datasets_used": [d["name"] for d in (download or {}).get("datasets", []) if d.get("status") == "ok"],
        "2_total_rows_used": (distribution or {}).get("global", {}).get("total_usable_rows"),
        "3_rows_per_label": {
            lab: d["count"] for lab, d in (distribution or {}).get("label_distribution", {}).items()
        },
        "4_balance": (distribution or {}).get("heads", {}),
        "5_fasttext_heads": ["attack", "abuse", "high_risk"],
        "6_fasttext_performance": {
            "attack": macro(ft_attack), "abuse": macro(ft_abuse), "high_risk": macro(ft_high),
        },
        "7_quantization_f1_loss_fasttext": {
            "attack": (ft_attack or {}).get("comparison", {}).get("macro_f1_diff"),
            "abuse": (ft_abuse or {}).get("comparison", {}).get("macro_f1_diff"),
            "high_risk": (ft_high or {}).get("comparison", {}).get("macro_f1_diff"),
        },
        "8_transformers_downloaded": list((baseline or {}).get("models", {}).keys()),
        "9_transformer_baselines": {
            k: v.get("macro_f1") for k, v in (baseline or {}).get("models", {}).items()
        },
        "10_onnx_exported": [
            k for k, m in (onnx_export or {}).get("models", {}).items()
            if m.get("sample_inference_ok")
        ],
        "11_int8_quantized": [
            k for k, m in (quantization or {}).get("models", {}).items()
            if m.get("status") == "ok"
        ],
        "12_int8_performance_change": {
            k: m.get("max_abs_diff_onnx_vs_int8")
            for k, m in (quantization or {}).get("models", {}).items()
            if m.get("status") == "ok"
        },
        "13_fastest_pipeline_mode": _fastest_mode(benchmark),
        "14_most_accurate_mode": "PyTorch CUDA baseline (pre-quantization) is typically most accurate",
        "15_recommended_deployment": _recommend_backend(quantization),
        "16_known_weak_labels": weak,
        "17_recommended_finetuning_targets": weak[:5],
    }

    summary = {
        "overall_status": overall.value,
        "components": component_rows,
        "questions": qa,
    }

    sections = [
        ReportSection("Recommended Deployment").add("recommendation", qa["15_recommended_deployment"]),
    ]
    if weak:
        ws = ReportSection("Known Weak Labels")
        for w in weak:
            ws.add(w, "review / consider fine-tuning")
        sections.append(ws)

    # Build a Q&A markdown block embedded as a table-ish section.
    qa_rows = [[k, str(v)[:120]] for k, v in qa.items()]

    write_report(
        "final_pipeline_report",
        status=overall,
        title="Final Pipeline Report",
        summary=summary,
        sections=sections,
        tables=[
            ("Component Status", ["Component", "Status", "Key Metric"], component_rows),
            ("Pipeline Q&A", ["Question", "Answer"], qa_rows),
        ],
    )
    print_summary(
        "FINAL PIPELINE REPORT", overall,
        ["Component", "Status", "Key Metric"], component_rows,
    )
    return summary
