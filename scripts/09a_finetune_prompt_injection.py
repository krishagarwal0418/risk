#!/usr/bin/env python3
"""Fine-tune the prompt injection detector on processed data.

Usage:
    python scripts/09a_finetune_prompt_injection.py \
        --device cuda \
        --batch-size 64 \
        --epochs 3 \
        --lr 2e-5
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
        description="Fine-tune prompt injection detector on processed safety data"
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Per-device batch size")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Fine-tune epochs (3 recommended for 4k examples)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate (2e-5 for attack task)")
    args = parser.parse_args()

    root = repo_root()
    # Attack-focused subset (all PI/JB positives + bounded hard negatives) — ~6x
    # smaller than the full file, so this fine-tune is ~6x faster with no quality
    # loss. Falls back to the full file if the attack subset wasn't built.
    train_path = str(root / "data" / "finetuning_attack.jsonl")
    if not (root / "data" / "finetuning_attack.jsonl").exists():
        train_path = str(root / "data" / "finetuning_train.jsonl")
    # Use validation set (not filtered to keep real distribution)
    val_path = str(root / "data" / "processed" / "all_val.jsonl")
    output_dir = str(root / "models" / "finetuned" / "prompt_injection")

    # Read the base model from models.yaml (source of truth) so a config swap
    # actually changes which model gets fine-tuned.
    model_name = load_models_config()["transformers"]["prompt_injection"]["hf_name"]

    print("=" * 60)
    print(f"Fine-tuning {model_name}")
    print("=" * 60)
    print(f"Training: {train_path}")
    print(f"Validation: {val_path}")
    print(f"Output: {output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}")
    print()

    metrics = finetune(
        model_name=model_name,
        task="attack",
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
