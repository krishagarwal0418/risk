"""ONNX export validation + INT8 quantization evaluation reports.

* :func:`write_export_report` validates each export dir (file present, tokenizer/
  config copied, loads under ONNX Runtime, sample inference works) and writes
  ``reports/onnx_export_report.{json,md}``.

* :func:`run_quantization_eval` compares PyTorch vs FP32 ONNX vs INT8 ONNX over a
  set of mild test strings (sizes, latency percentiles, score deltas, agreement)
  and writes ``reports/quantization_report.{json,md}``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..config import load_models_config, resolve_path
from ..reporting import (
    ReportSection,
    Status,
    file_size,
    human_size,
    percentiles,
    print_summary,
    write_report,
)
from ..transformers_layer.quantize_onnx import _VALIDATION_TEXTS

_EXPORT_KEYS = ("prompt_injection", "jailbreak", "moderation_fallback")


# --------------------------------------------------------------------------- #
# Export validation
# --------------------------------------------------------------------------- #
def _validate_export(onnx_dir: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "path": str(onnx_dir),
        "onnx_present": False,
        "onnx_size": None,
        "tokenizer_copied": False,
        "config_copied": False,
        "loads_with_ort": False,
        "sample_inference_ok": False,
    }
    if not onnx_dir.exists():
        return info
    onnx_files = list(onnx_dir.glob("*.onnx"))
    info["onnx_present"] = bool(onnx_files)
    info["onnx_size"] = file_size(onnx_files[0]) if onnx_files else None
    info["tokenizer_copied"] = any(
        (onnx_dir / f).exists() for f in ("tokenizer.json", "tokenizer_config.json", "vocab.txt", "spm.model")
    )
    info["config_copied"] = (onnx_dir / "config.json").exists()
    if not onnx_files:
        return info
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer

        model = ORTModelForSequenceClassification.from_pretrained(str(onnx_dir))
        info["loads_with_ort"] = True
        tok = AutoTokenizer.from_pretrained(str(onnx_dir))
        enc = tok("hello world", return_tensors="pt", truncation=True, max_length=32)
        _ = model(**enc)
        info["sample_inference_ok"] = True
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def write_export_report(export_results: dict[str, Any]) -> dict[str, Any]:
    cfg = load_models_config().get("transformers", {})
    models: dict[str, Any] = {}
    warnings: list[str] = []
    for key in _EXPORT_KEYS:
        entry = cfg.get(key)
        if not entry:
            continue
        exported = export_results.get(key, {})
        validation = _validate_export(resolve_path(entry["onnx_path"]))
        models[key] = {
            "hf_name": entry["hf_name"],
            "export_attempted": key in export_results,
            "export_status": exported.get("status", "unknown"),
            "export_error": exported.get("cli_error") or exported.get("python_error"),
            **validation,
        }
        if not validation["sample_inference_ok"]:
            warnings.append(f"{key}: ONNX export/validation failed")

    ok = sum(1 for m in models.values() if m["sample_inference_ok"])
    total = len(models)
    status = Status.FAIL if ok == 0 else (Status.WARN if ok < total else Status.PASS)

    table_rows = [
        [
            m["hf_name"], m["export_status"].upper(),
            "yes" if m["onnx_present"] else "no",
            human_size(m["onnx_size"]),
            "yes" if m["loads_with_ort"] else "no",
            "yes" if m["sample_inference_ok"] else "no",
        ]
        for m in models.values()
    ]
    write_report(
        "onnx_export_report",
        status=status,
        title="ONNX Export Report",
        summary={"exported_ok": ok, "total": total, "models": models},
        tables=[(
            "Exports",
            ["Model", "Status", "ONNX", "Size", "Loads", "Inference"],
            table_rows,
        )],
        warnings=warnings,
    )
    print_summary(
        "ONNX Export", status,
        ["Model", "Status", "ONNX", "Size", "Loads", "Inference"], table_rows,
    )
    return {"exported_ok": ok, "total": total, "models": models, "status": status.value}


# --------------------------------------------------------------------------- #
# Quantization evaluation (PyTorch vs FP32 ONNX vs INT8 ONNX)
# --------------------------------------------------------------------------- #
def _logits_and_latency(predict, texts: list[str]) -> tuple[list, list[float]]:
    import numpy as np  # noqa: F401

    all_logits = []
    latencies = []
    for text in texts:
        start = time.perf_counter()
        logits = predict(text)
        latencies.append((time.perf_counter() - start) * 1000)
        all_logits.append(logits)
    return all_logits, latencies


def _eval_one_quantization(key: str, entry: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    fp32_dir = resolve_path(entry["onnx_path"])
    int8_dir = resolve_path(entry["int8_path"])
    out: dict[str, Any] = {"hf_name": entry["hf_name"]}

    if not fp32_dir.exists() or not list(fp32_dir.glob("*.onnx")):
        return {**out, "status": "skipped", "error": "no FP32 ONNX export"}
    if not int8_dir.exists() or not list(int8_dir.glob("*.onnx")):
        return {**out, "status": "skipped", "error": "no INT8 model"}

    try:
        import torch
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        tok = AutoTokenizer.from_pretrained(str(fp32_dir))

        def make_torch():
            m = AutoModelForSequenceClassification.from_pretrained(entry["hf_name"])
            m.eval()

            def predict(text):
                enc = tok(text, return_tensors="pt", truncation=True, max_length=128)
                with torch.inference_mode():
                    return m(**enc).logits.detach().numpy()[0]

            return predict, m

        def make_ort(path):
            m = ORTModelForSequenceClassification.from_pretrained(str(path))

            def predict(text):
                enc = tok(text, return_tensors="pt", truncation=True, max_length=128)
                return m(**enc).logits.detach().numpy()[0]

            return predict

        torch_predict, _ = make_torch()
        fp32_predict = make_ort(fp32_dir)
        int8_predict = make_ort(int8_dir)

        t_logits, t_lat = _logits_and_latency(torch_predict, _VALIDATION_TEXTS)
        f_logits, f_lat = _logits_and_latency(fp32_predict, _VALIDATION_TEXTS)
        i_logits, i_lat = _logits_and_latency(int8_predict, _VALIDATION_TEXTS)

        if np.array(i_logits).shape != np.array(f_logits).shape:
            return {**out, "status": "fail", "error": "INT8 output shape differs from FP32 ONNX"}

        def _softmax(x):
            e = np.exp(x - np.max(x))
            return e / e.sum()

        t_probs = [_softmax(l) for l in t_logits]
        f_probs = [_softmax(l) for l in f_logits]
        i_probs = [_softmax(l) for l in i_logits]

        max_diff_torch_onnx = float(np.max(np.abs(np.array(t_probs) - np.array(f_probs))))
        max_diff_onnx_int8 = float(np.max(np.abs(np.array(f_probs) - np.array(i_probs))))
        avg_diff_onnx_int8 = float(np.mean(np.abs(np.array(f_probs) - np.array(i_probs))))
        agree = float(
            np.mean([np.argmax(f) == np.argmax(i) for f, i in zip(f_probs, i_probs)]) * 100
        )

        return {
            **out,
            "status": "ok",
            "sizes": {
                "fp32_onnx": file_size(fp32_dir),
                "int8_onnx": file_size(int8_dir),
                "compression_ratio": round(file_size(fp32_dir) / max(file_size(int8_dir), 1), 2),
            },
            "latency_ms": {
                "pytorch": percentiles(t_lat),
                "fp32_onnx": percentiles(f_lat),
                "int8_onnx": percentiles(i_lat),
            },
            "max_abs_diff_pytorch_vs_onnx": round(max_diff_torch_onnx, 5),
            "max_abs_diff_onnx_vs_int8": round(max_diff_onnx_int8, 5),
            "avg_abs_diff_onnx_vs_int8": round(avg_diff_onnx_int8, 5),
            "prediction_agreement_pct": round(agree, 2),
        }
    except Exception as exc:  # noqa: BLE001
        return {**out, "status": "error", "error": f"{type(exc).__name__}: {exc}"}


def run_quantization_eval() -> dict[str, Any]:
    cfg = load_models_config().get("transformers", {})
    results: dict[str, Any] = {}
    warnings: list[str] = []
    failures: list[str] = []
    for key in _EXPORT_KEYS:
        entry = cfg.get(key)
        if not entry:
            continue
        r = _eval_one_quantization(key, entry)
        results[key] = r
        if r["status"] == "fail":
            failures.append(f"{key}: {r.get('error')}")
        elif r["status"] == "error":
            warnings.append(f"{key}: {r.get('error')}")
        elif r["status"] == "ok":
            if r["max_abs_diff_onnx_vs_int8"] > 0.05:
                warnings.append(
                    f"{key}: INT8 max abs prob diff {r['max_abs_diff_onnx_vs_int8']} > 0.05"
                )

    ok = [r for r in results.values() if r["status"] == "ok"]
    status = Status.FAIL if failures else (Status.WARN if (warnings or not ok) else Status.PASS)

    table_rows = []
    for key, r in results.items():
        if r["status"] == "ok":
            table_rows.append([
                r["hf_name"],
                f"{r['sizes']['compression_ratio']}x",
                r["max_abs_diff_onnx_vs_int8"],
                f"{r['prediction_agreement_pct']}%",
                r["latency_ms"]["int8_onnx"]["p95"],
            ])
        else:
            table_rows.append([r["hf_name"], r["status"].upper(), r.get("error", "")[:30], "-", "-"])

    write_report(
        "quantization_report",
        status=status,
        title="ONNX INT8 Quantization Report",
        summary={"models": results},
        tables=[(
            "Quantization",
            ["Model", "Compression", "Max prob diff (FP32→INT8)", "Agreement", "INT8 p95 ms"],
            table_rows,
        )],
        warnings=warnings,
        failures=failures,
    )
    print_summary(
        "ONNX INT8 Quantization", status,
        ["Model", "Compression", "MaxDiff", "Agreement", "INT8 p95 ms"], table_rows,
    )
    return {"models": results, "status": status.value}
