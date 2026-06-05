#!/usr/bin/env python3
"""Fine-tune ModerationBERT fallback model on processed data.

Usage:
    python scripts/09c_finetune_moderation_fallback.py \
        --device cuda \
        --batch-size 64 \
        --epochs 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from safety_classifier.transformers_layer.finetune import finetune
from safety_classifier.config import repo_root, load_models_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune ModerationBERT fallback on processed safety data"
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Per-device batch size")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Fine-tune epochs")
    parser.add_argument("--lr", type=float, default=3e-5,
                        help="Learning rate")
    args = parser.parse_args()

    root = repo_root()
    # Use validated fine-tuning data (deduplicated, filtered for quality)
    train_path = str(root / "data" / "finetuning_train.jsonl")
    # Use validation set (not filtered to keep real distribution)
    val_path = str(root / "data" / "processed" / "all_val.jsonl")
    output_dir = str(root / "models" / "finetuned" / "moderation_bert")

    model_name = load_models_config()["transformers"]["moderation_fallback"]["hf_name"]
    print("=" * 60)
    print(f"Fine-tuning {model_name} (fallback)")
    print("=" * 60)
    print(f"Training: {train_path}")
    print(f"Output: {output_dir}")
    print(f"Batch size: {args.batch_size}, Epochs: {args.epochs}, LR: {args.lr}")
    print("Note: Used as fallback; slightly lower baseline (F1=0.556)")
    print()

    metrics = finetune(
        model_name=model_name,
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
