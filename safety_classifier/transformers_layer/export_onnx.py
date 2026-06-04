"""Export transformer models to ONNX using Hugging Face Optimum.

Tries the Python API (``ORTModelForSequenceClassification.from_pretrained(...,
export=True)``) and falls back to the ``optimum-cli`` command. Each model is
attempted independently so a single failure (e.g. a custom TinySafe architecture)
does not block the rest of the pipeline.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from ..config import load_models_config, resolve_path


def export_model_python(hf_name: str, out_dir: Path) -> dict[str, Any]:
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    model = ORTModelForSequenceClassification.from_pretrained(hf_name, export=True)
    model.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(hf_name).save_pretrained(out_dir)
    return {"status": "ok", "method": "python", "path": str(out_dir)}


def export_model_cli(hf_name: str, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "optimum-cli", "export", "onnx",
        "--model", hf_name,
        "--task", "text-classification",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return {"status": "error", "method": "cli", "stderr": proc.stderr[-1000:]}
    return {"status": "ok", "method": "cli", "path": str(out_dir)}


def export_model(hf_name: str, out_dir: Path) -> dict[str, Any]:
    """Export one model, trying the Python API then the CLI."""
    try:
        return export_model_python(hf_name, out_dir)
    except Exception as exc:  # noqa: BLE001
        py_err = f"{type(exc).__name__}: {exc}"
    try:
        result = export_model_cli(hf_name, out_dir)
        if result["status"] == "ok":
            return result
        result["python_error"] = py_err
        return result
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "python_error": py_err,
            "cli_error": f"{type(exc).__name__}: {exc}",
        }


# Models that should be exported by default. TinySafe is kept in the config as an
# optional PyTorch-only experiment, but the standard pipeline uses oxyapi because
# it loads through the normal Hugging Face/ONNX path.
_EXPORT_KEYS = (
    "prompt_injection",
    "jailbreak",
    "moderation_fallback",
)


def export_all(include_toxic_fallback: bool = False) -> dict[str, Any]:
    cfg = load_models_config().get("transformers", {})
    keys = list(_EXPORT_KEYS)
    if include_toxic_fallback:
        keys.append("toxic_fallback")
    results: dict[str, Any] = {}
    for key in keys:
        entry = cfg.get(key)
        if not entry:
            continue
        out_dir = resolve_path(entry["onnx_path"])
        print(f"[onnx-export] {key}: {entry['hf_name']} -> {out_dir}")
        results[key] = export_model(entry["hf_name"], out_dir)
        status = results[key]["status"]
        print(f"[onnx-export] {key}: {status}")
    return results
