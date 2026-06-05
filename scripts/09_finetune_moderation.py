#!/usr/bin/env python3
"""Fine-tune Koala moderation on the Koala-specific filtered data.

Usage:
    python scripts/17_finetune_koala_moderation_best.py --build-only
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
from safety_classifier.config import repo_root, load_models_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune KoalaAI moderation on supported-label safety data"
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Per-device batch size (scales to 256+ with multi-GPU)")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Fine-tune epochs (2-3 recommended)")
    parser.add_argument("--lr", type=float, default=3e-5,
                        help="Learning rate (3e-5 for moderation task)")
    parser.add_argument("--init-weights", default=None,
                        help="Warm-start from a saved .safetensors (e.g. weights "
                             "from an interrupted run) instead of the base model.")
    args = parser.parse_args()

    root = repo_root()
    train_path = root / "data" / "koala_moderation" / "train.jsonl"
    val_path = root / "data" / "koala_moderation" / "val.jsonl"
    if not train_path.exists() or not val_path.exists():
        raise SystemExit(
            "Koala-specific fine-tuning data is missing. Run:\n"
            "  python scripts/17_finetune_koala_moderation_best.py --build-only"
        )
    output_dir = str(root / "models" / "finetuned" / "moderation")

    model_name = load_models_config()["transformers"]["moderation_primary"]["hf_name"]
    print("=" * 60)
    print(f"Fine-tuning {model_name}")
    print("=" * 60)
    print(f"Training: {train_path}")
    print(f"Validation: {val_path}")
    print(f"Output: {output_dir}")
    print(f"Batch size: {args.batch_size} (effective 256+ with multi-GPU)")
    print(f"Epochs: {args.epochs}, LR: {args.lr}")
    print()

    if args.init_weights:
        print(f"Warm-starting from: {args.init_weights}")
    metrics = finetune(
        model_name=model_name,
        task="koala_moderation",
        train_path=str(train_path),
        val_path=str(val_path),
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        init_weights_path=args.init_weights,
        metric_for_best_model="macro_pr_auc",
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
