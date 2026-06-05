"""Optional fine-tuning harness (Hugging Face Trainer).

Supports multi-label classification with class weights and early stopping for:
  * task=attack       -> labels: prompt_injection, jailbreak
  * task=moderation   -> labels: toxicity, hate, harassment, sexual, violence,
                                 self_harm, dangerous_information, illegal_activity

The default system uses the pretrained models; this harness exists for teams that
want to adapt a model to their own validation data.

Usage:
    python -m safety_classifier.transformers_layer.finetune \
        --model devndeploy/bert-prompt-injection-detector \
        --task attack \
        --train data/processed/all_train.jsonl \
        --val data/processed/all_val.jsonl \
        --output models/finetuned/prompt_injection
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .. import constants as C

TASK_LABELS: dict[str, list[str]] = {
    "attack": [C.PROMPT_INJECTION, C.JAILBREAK],
    "moderation": [
        C.TOXICITY,
        C.HATE,
        C.HARASSMENT,
        C.SEXUAL,
        C.VIOLENCE,
        C.SELF_HARM,
        C.DANGEROUS_INFORMATION,
        C.ILLEGAL_ACTIVITY,
    ],
}


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _multi_hot(labels: list[str], label_list: list[str]) -> list[float]:
    idx = {lab: i for i, lab in enumerate(label_list)}
    vec = [0.0] * len(label_list)
    for lab in labels:
        if lab in idx:
            vec[idx[lab]] = 1.0
    return vec


def build_dataset(rows: list[dict], label_list: list[str]):
    from datasets import Dataset

    data = {
        "text": [r["text"] for r in rows],
        "labels": [_multi_hot(r.get("labels", []), label_list) for r in rows],
    }
    return Dataset.from_dict(data)


def compute_class_weights(rows: list[dict], label_list: list[str]) -> list[float]:
    """Per-label ``pos_weight`` for ``BCEWithLogitsLoss`` (one entry per label).

    For BCE, ``pos_weight`` is the factor by which a positive is weighted relative
    to a negative for THAT label, and the correct value is ``num_neg / num_pos``.
    A previous version normalised sqrt-inverse-frequency *across labels* (÷ mean),
    which collapsed to ~1.0 whenever the labels had similar frequency — e.g. the
    attack task (PI ~1.5%, JB ~1.2%) got pos_weight ≈ 1.0 and the model learned to
    predict "negative" almost always (injection scores crushed near 0, ~66% recall
    ceiling). That is wrong: each label's pos_weight must reflect its OWN pos/neg
    balance, independent of the other labels.

    We use ``sqrt(neg / pos)`` — the full ``neg/pos`` ratio is correct but can be
    large (≈137 for self_harm) and destabilise training, so the sqrt softens it.
    Clamped to [1, 30]: never down-weight positives, cap extreme upweighting.
    """
    import numpy as np

    counts = np.zeros(len(label_list))
    for r in rows:
        for i, lab in enumerate(label_list):
            if lab in r.get("labels", []):
                counts[i] += 1
    total = max(len(rows), 1)
    pos = np.maximum(counts, 1.0)
    neg = np.maximum(total - counts, 1.0)
    pos_weight = np.sqrt(neg / pos)
    pos_weight = np.clip(pos_weight, 1.0, 30.0)
    return pos_weight.tolist()


def finetune(
    model_name: str,
    task: str,
    train_path: str,
    val_path: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    val_limit: int = 4000,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from torch import nn
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    label_list = TASK_LABELS[task]
    train_rows = _load_jsonl(train_path)
    val_rows = _load_jsonl(val_path)
    # The full val set (~76k) is needlessly slow to evaluate every checkpoint.
    # A deterministic random sample is plenty for model selection.
    if val_limit and len(val_rows) > val_limit:
        import random

        random.Random(13).shuffle(val_rows)
        val_rows = val_rows[:val_limit]
    class_weights = torch.tensor(
        compute_class_weights(train_rows, label_list), dtype=torch.float
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        problem_type="multi_label_classification",
        id2label={i: l for i, l in enumerate(label_list)},
        label2id={l: i for i, l in enumerate(label_list)},
        ignore_mismatched_sizes=True,
    )

    def tokenize(batch):
        # padding="max_length" pads EVERY example to exactly max_length. With
        # padding=True each .map chunk padded to its own batch-max (e.g. 103 vs
        # 128), and the default collator can't stack mismatched lengths
        # ("expected sequence of length 103 ... got 128"). Fixed length avoids it.
        enc = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
        enc["labels"] = batch["labels"]
        return enc

    train_ds = build_dataset(train_rows, label_list).map(tokenize, batched=True)
    val_ds = build_dataset(val_rows, label_list).map(tokenize, batched=True)

    def compute_metrics(eval_pred):
        from sklearn.metrics import f1_score

        logits, labels = eval_pred
        preds = (1 / (1 + np.exp(-logits))) >= 0.5
        return {
            "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        }

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss_fct = nn.BCEWithLogitsLoss(
                pos_weight=class_weights.to(outputs.logits.device)
            )
            loss = loss_fct(outputs.logits, labels.float())
            return (loss, outputs) if return_outputs else loss

    use_cuda = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        # Mixed precision on GPU: ~2x faster on a T4 with negligible quality impact.
        fp16=use_cuda,
        dataloader_num_workers=2,
        # Step-based save/eval so a crash or disconnect only loses the steps since
        # the last checkpoint (not the whole run). save_total_limit caps disk use.
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=250,
        save_steps=250,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to=[],
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Auto-resume: if a checkpoint already exists in output_dir (from an
    # interrupted run), continue from it; otherwise train from scratch.
    from transformers.trainer_utils import get_last_checkpoint

    last_ckpt = get_last_checkpoint(output_dir) if Path(output_dir).is_dir() else None
    if last_ckpt:
        print(f"[finetune] resuming from checkpoint: {last_ckpt}")
    trainer.train(resume_from_checkpoint=last_ckpt)
    metrics = trainer.evaluate()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    Path(output_dir, "finetune_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a safety classifier head")
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True, choices=sorted(TASK_LABELS))
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()
    metrics = finetune(
        args.model, args.task, args.train, args.val, args.output,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
