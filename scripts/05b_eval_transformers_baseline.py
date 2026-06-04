"""Evaluate pretrained transformer baselines before ONNX export / quantization."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from safety_classifier.evaluation.transformer_eval import run_baseline_eval


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--toxic-fallback", action="store_true")
    p.add_argument("--limit", type=int, default=2000, help="max test examples (0 = all)")
    p.add_argument("--batch-size", type=int, default=64, help="GPU inference batch size")
    args = p.parse_args()
    run_baseline_eval(
        device=args.device,
        include_toxic_fallback=args.toxic_fallback,
        limit=args.limit or None,
        batch_size=args.batch_size,
    )
    print("[baseline] report written to reports/transformer_baseline_eval.{json,md}")


if __name__ == "__main__":
    main()
