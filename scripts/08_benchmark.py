"""Benchmark the safety classifier pipeline."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from safety_classifier.benchmark import run_benchmark
from safety_classifier.classifier import SafetyClassifier


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--backend", default="pytorch", choices=["pytorch", "onnx", "onnx_int8"])
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=200)
    args = p.parse_args()

    clf = SafetyClassifier(device=args.device, backend=args.backend)
    report = run_benchmark(clf, warmup=args.warmup, iters=args.iters)
    print("\n=== Benchmark scenarios ===")
    for name, s in report["scenarios"].items():
        print(f"  {name:<28} avg={s['avg']}ms p95={s['p95']}ms rps={s['throughput_rps']}")
    print("\nReports written to reports/benchmark_report.json and reports/benchmark_report.md")


if __name__ == "__main__":
    main()
