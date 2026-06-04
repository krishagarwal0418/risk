"""Quantize exported ONNX models to INT8 and validate."""

from __future__ import annotations

import json

import _bootstrap  # noqa: F401

from safety_classifier.evaluation.onnx_eval import run_quantization_eval
from safety_classifier.transformers_layer.quantize_onnx import quantize_all


def main() -> None:
    results = quantize_all()
    print(json.dumps(results, indent=2))
    # Evaluate quantization quality (PyTorch vs FP32 ONNX vs INT8 ONNX).
    run_quantization_eval()
    print("[quantize] report written to reports/quantization_report.{json,md}")


if __name__ == "__main__":
    main()
