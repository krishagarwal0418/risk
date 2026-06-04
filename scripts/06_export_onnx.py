"""Export transformer models to ONNX (Optimum)."""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from safety_classifier.evaluation.onnx_eval import write_export_report
from safety_classifier.transformers_layer.export_onnx import export_all


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--toxic-fallback", action="store_true")
    args = p.parse_args()
    results = export_all(include_toxic_fallback=args.toxic_fallback)
    print(json.dumps(results, indent=2))
    write_export_report(results)
    print("[export] report written to reports/onnx_export_report.{json,md}")


if __name__ == "__main__":
    main()
