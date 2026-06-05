#!/usr/bin/env python3
"""Full end-to-end pipeline: data -> FastText -> fine-tune transformers -> benchmark.

Self-contained: sensible FastText caps and a balanced fine-tuning set are produced
automatically — no environment variables required. INT8 quantization is skipped
(FP32 ONNX is used; INT8 had unacceptable logit drift on the moderation models).

Usage:
    python scripts/10_full_pipeline.py
    python scripts/10_full_pipeline.py --batch-size 64 --epochs 2
    python scripts/10_full_pipeline.py --skip-download --skip-training
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from safety_classifier.config import repo_root


def run_stage(name: str, cmd: list[str], skip: bool = False,
              env: dict | None = None) -> bool:
    """Run a pipeline stage, streaming its output live. Returns success."""
    if skip:
        print(f"\n⊘ SKIPPED: {name}")
        return True

    print()
    print("=" * 70)
    print(f"STAGE: {name}")
    print("=" * 70)
    try:
        # No capture_output: child stdout/stderr stream straight to the console,
        # so errors are visible in the log as they happen.
        subprocess.run(cmd, check=True, text=True, env=env)
        print(f"✓ {name} done")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {name} FAILED (exit code {e.returncode}) — see output above")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full pipeline with fine-tuning, no quantization"
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip data prep + FastText training")
    parser.add_argument("--skip-fasttext", action="store_true",
                        help="Skip ONLY the FastText train/eval/calibrate stages "
                             "(still prepares data for fine-tuning)")
    parser.add_argument("--skip-finetuning", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Fine-tuning per-device batch size")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--moderation-epochs", type=int, default=None,
                        help="Epochs for the PRIMARY moderation model (KoalaAI). "
                             "Defaults to --epochs. Use e.g. 2 when --epochs 1.")
    parser.add_argument("--max-per-label", type=int, default=25000,
                        help="FastText per-label cap for attack/abuse heads")
    parser.add_argument("--max-per-label-high-risk", type=int, default=3000,
                        help="FastText per-label cap for the high_risk head")
    parser.add_argument("--safe-cap", type=int, default=60000,
                        help="Max safe-only examples in the fine-tuning set")
    parser.add_argument("--eval-batch-size", type=int, default=64)
    args = parser.parse_args()

    root = repo_root()

    # Environment for the data-prep stage: bakes the FastText caps in so the user
    # never has to remember to export SC_MAX_PER_LABEL.
    prep_env = {
        **os.environ,
        "SC_MAX_PER_LABEL": str(args.max_per_label),
        "SC_MAX_PER_LABEL_HIGH_RISK": str(args.max_per_label_high_risk),
    }

    def script(name: str) -> str:
        return str(root / "scripts" / name)

    bs = str(args.batch_size)
    ep = str(args.epochs)
    mod_ep = str(args.moderation_epochs if args.moderation_epochs is not None else args.epochs)
    dev = args.device

    stages = [
        ("01. Download datasets",
         [sys.executable, script("01_download_datasets.py")],
         args.skip_download, None),

        ("02. Prepare + balance FastText data",
         [sys.executable, script("02_prepare_fasttext_data.py")],
         args.skip_training, prep_env),

        ("03. Train FastText heads",
         [sys.executable, script("03_train_fasttext_heads.py")],
         args.skip_training or args.skip_fasttext, prep_env),

        ("04. Evaluate FastText heads",
         [sys.executable, script("04_eval_fasttext_heads.py")],
         args.skip_training or args.skip_fasttext, None),

        ("04b. Calibrate FastText thresholds",
         [sys.executable, script("04b_calibrate_fasttext_thresholds.py")],
         args.skip_training or args.skip_fasttext, None),

        ("05. Download transformers",
         [sys.executable, script("05_download_transformers.py")],
         False, None),

        ("05b. Baseline transformer eval",
         [sys.executable, script("05b_eval_transformers_baseline.py"),
          "--device", dev, "--batch-size", str(args.eval_batch_size)],
         False, None),

        ("09. Validate + balance fine-tuning data",
         [sys.executable, script("09_validate_finetuning_data.py"),
          "--input", "data/processed/all_train.jsonl",
          "--output", "data/finetuning_train.jsonl",
          "--safe-cap", str(args.safe_cap)],
         args.skip_finetuning, None),

        ("09a. Fine-tune prompt injection",
         [sys.executable, script("09a_finetune_prompt_injection.py"),
          "--device", dev, "--batch-size", bs, "--epochs", ep, "--lr", "2e-5"],
         args.skip_finetuning, None),

        ("09b. Fine-tune jailbreak detector",
         [sys.executable, script("09b_finetune_jailbreak.py"),
          "--device", dev, "--batch-size", bs, "--epochs", ep, "--lr", "2e-5"],
         args.skip_finetuning, None),

        ("09c. Fine-tune moderation (primary)",
         [sys.executable, script("09_finetune_moderation.py"),
          "--device", dev, "--batch-size", bs, "--epochs", mod_ep, "--lr", "3e-5"],
         args.skip_finetuning, None),

        ("09d. Fine-tune moderation (fallback)",
         [sys.executable, script("09c_finetune_moderation_fallback.py"),
          "--device", dev, "--batch-size", bs, "--epochs", ep, "--lr", "3e-5"],
         args.skip_finetuning, None),

        ("05c. Re-evaluate fine-tuned transformers",
         [sys.executable, script("05b_eval_transformers_baseline.py"),
          "--device", dev, "--batch-size", str(args.eval_batch_size)],
         args.skip_finetuning, None),

        ("06. Export to ONNX (FP32)",
         [sys.executable, script("06_export_onnx.py")],
         False, None),

        ("07b. Validate ONNX",
         [sys.executable, script("07b_eval_quantized_models.py")],
         False, None),

        ("08. Benchmark end-to-end",
         [sys.executable, script("08_benchmark.py"), "--device", dev],
         False, None),

        ("11. Generate final report",
         [sys.executable, script("11_generate_final_report.py")],
         False, None),
    ]

    failed = None
    for name, cmd, skip, env in stages:
        if not run_stage(name, cmd, skip, env):
            failed = name
            print(f"\n⚠️  Pipeline halted at {name}")
            break

    print()
    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    if failed:
        print(f"✗ Failed at: {failed}")
        print("Fix the error above, then re-run with --skip-* to resume.")
        sys.exit(1)
    print("✓ All stages passed")
    print(f"\nReports: {root / 'reports'}")
    print(f"Models:  {root / 'models'}")


if __name__ == "__main__":
    main()
