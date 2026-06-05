#!/usr/bin/env python3
"""Full end-to-end pipeline: data → FastText → fine-tune transformers → benchmark.

Skips INT8 quantization (FP32 ONNX is sufficient). With 256 batch size, completes
in ~2-3 hours on a T4 GPU (most time is data download + FastText training).

Usage:
    python scripts/10_full_pipeline.py [--skip-download] [--skip-training] [--skip-finetuning]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from safety_classifier.config import repo_root


def run_stage(name: str, cmd: list[str], skip: bool = False) -> bool:
    """Run a pipeline stage and report results."""
    if skip:
        print(f"⊘ SKIPPED: {name}")
        return True

    print()
    print("=" * 70)
    print(f"STAGE: {name}")
    print("=" * 70)
    try:
        result = subprocess.run(cmd, check=True, text=True)
        print(f"✓ {name} done")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {name} FAILED")
        print(e.stderr)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full pipeline with fine-tuning, no quantization"
    )
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip dataset download if already cached")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip FastText training if already done")
    parser.add_argument("--skip-finetuning", action="store_true",
                        help="Skip transformer fine-tuning")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Fine-tuning batch size")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Fine-tuning epochs")
    args = parser.parse_args()

    root = repo_root()
    stages = [
        ("01. Download datasets", [
            sys.executable, str(root / "scripts" / "01_download_datasets.py")
        ], args.skip_download),

        ("02. Prepare FastText data", [
            sys.executable, str(root / "scripts" / "02_prepare_fasttext_data.py")
        ], args.skip_download or args.skip_training),

        ("03. Train FastText heads", [
            sys.executable, str(root / "scripts" / "03_train_fasttext_heads.py")
        ], args.skip_training),

        ("04. Evaluate FastText heads", [
            sys.executable, str(root / "scripts" / "04_eval_fasttext_heads.py")
        ], args.skip_training),

        ("04b. Calibrate FastText thresholds", [
            sys.executable, str(root / "scripts" / "04b_calibrate_fasttext_thresholds.py")
        ], args.skip_training),

        ("05. Download transformers", [
            sys.executable, str(root / "scripts" / "05_download_transformers.py")
        ], False),

        ("05b. Baseline transformer eval", [
            sys.executable, str(root / "scripts" / "05b_eval_transformers_baseline.py"),
            "--device", "cuda", "--batch-size", "64"
        ], False),

        ("09a. Fine-tune prompt injection", [
            sys.executable, str(root / "scripts" / "09a_finetune_prompt_injection.py"),
            "--device", "cuda",
            "--batch-size", str(args.batch_size),
            "--epochs", str(args.epochs),
            "--lr", "2e-5"
        ], args.skip_finetuning),

        ("09b. Fine-tune jailbreak detector", [
            sys.executable, str(root / "scripts" / "09b_finetune_jailbreak.py"),
            "--device", "cuda",
            "--batch-size", str(args.batch_size),
            "--epochs", str(args.epochs),
            "--lr", "2e-5"
        ], args.skip_finetuning),

        ("09c. Fine-tune moderation (primary)", [
            sys.executable, str(root / "scripts" / "09_finetune_moderation.py"),
            "--device", "cuda",
            "--batch-size", str(args.batch_size),
            "--epochs", str(args.epochs),
            "--lr", "3e-5"
        ], args.skip_finetuning),

        ("09d. Fine-tune moderation (fallback)", [
            sys.executable, str(root / "scripts" / "09c_finetune_moderation_fallback.py"),
            "--device", "cuda",
            "--batch-size", str(args.batch_size),
            "--epochs", str(args.epochs),
            "--lr", "3e-5"
        ], args.skip_finetuning),

        ("06. Export to ONNX (FP32)", [
            sys.executable, str(root / "scripts" / "06_export_onnx.py")
        ], False),

        ("07b. Validate ONNX", [
            sys.executable, str(root / "scripts" / "07b_eval_quantized_models.py")
        ], False),

        ("08. Benchmark end-to-end", [
            sys.executable, str(root / "scripts" / "08_benchmark.py"),
            "--device", "cuda"
        ], False),

        ("11. Generate final report", [
            sys.executable, str(root / "scripts" / "11_generate_final_report.py")
        ], False),
    ]

    failed = []
    for name, cmd, skip in stages:
        success = run_stage(name, cmd, skip)
        if not success:
            failed.append(name)
            print(f"\n⚠️  Pipeline halted at {name}")
            break

    print()
    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    if failed:
        print(f"✗ Failed stages: {', '.join(failed)}")
        print(f"\nTo resume, fix the error and re-run with appropriate --skip-* flags")
    else:
        print("✓ All stages passed")
        print(f"\nReports: {root / 'reports'}")
        print(f"Models: {root / 'models'}")


if __name__ == "__main__":
    main()
