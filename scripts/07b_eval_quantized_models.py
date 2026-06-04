"""Evaluate INT8-quantized ONNX models vs FP32 ONNX vs PyTorch.

Writes reports/quantization_report.{json,md} comparing sizes, latency and score
deltas across the three backends.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

from safety_classifier.evaluation.onnx_eval import run_quantization_eval


def main() -> None:
    run_quantization_eval()
    print("[quant-eval] report written to reports/quantization_report.{json,md}")


if __name__ == "__main__":
    main()
