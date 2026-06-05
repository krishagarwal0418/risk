#!/usr/bin/env python3
"""Fine-tune the moderation model on processed data with aggressive batching.

Usage:
    python scripts/09_finetune_moderation.py \
        --device cuda \
        --batch-size 64 \
        --epochs 2 \
        --lr 3e-5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from safety_classifier.transformers_layer.finetune import finetune
from safety_classifier.config import repo_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune oxyapi moderation model on processed safety data"
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Per-device batch size (scales to 256+ with multi-GPU)")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Fine-tune epochs (2-3 recommended)")
    parser.add_argument("--lr", type=float, default=3e-5,
                        help="Learning rate (3e-5 for moderation task)")
    args = parser.parse_args()

    root = repo_root()
    train_path = str(root / "data" / "processed" / "all_train.jsonl")
    val_path = str(root / "data" / "processed" / "all_val.jsonl")
    output_dir = str(root / "models" / "finetuned" / "moderation")

    print("=" * 60)
    print("Fine-tuning oxyapi/albert-moderation-001")
    print("=" * 60)
    print(f"Training: {train_path}")
    print(f"Validation: {val_path}")
    print(f"Output: {output_dir}")
    print(f"Batch size: {args.batch_size} (effective 256+ with multi-GPU)")
    print(f"Epochs: {args.epochs}, LR: {args.lr}")
    print()

    metrics = finetune(
        model_name="oxyapi/albert-moderation-001",
        task="moderation",
        train_path=train_path,
        val_path=val_path,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    print()
    print("=" * 60)
    print("Fine-tuning complete")
    print("=" * 60)
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")


if __name__ == "__main__":
    main()
