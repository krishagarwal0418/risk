"""Optional fine-tuning harness (Hugging Face Trainer).

Supports multi-label classification with class weights and early stopping for:
  * task=attack       -> labels: prompt_injection, jailbreak
  * task=moderation   -> labels: toxicity, hate, harassment, sexual, violence,
                                 self_harm, dangerous_information, illegal_activity
  * task=koala_moderation -> Koala-supported labels only: toxicity, hate,
                             harassment, sexual, violence, self_harm

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
    "koala_moderation": [
        C.TOXICITY,
        C.HATE,
        C.HARASSMENT,
        C.SEXUAL,
        C.VIOLENCE,
        C.SELF_HARM,
    ],
}

_METRIC_THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]


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


def _checkpoint_is_compatible(checkpoint: str | Path, label_list: list[str]) -> bool:
    """Return True when an auto-resume checkpoint matches the current label head."""
    checkpoint = Path(checkpoint)
    cfg_path = checkpoint / "config.json"
    if not cfg_path.exists():
        return True
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if int(cfg.get("num_labels", len(label_list))) != len(label_list):
        return False
    id2label = cfg.get("id2label") or {}
    if id2label:
        labels = [id2label[str(i)] for i in range(len(label_list)) if str(i) in id2label]
        if labels and labels != label_list:
            return False
    safetensors_path = checkpoint / "model.safetensors"
    if safetensors_path.exists():
        try:
            from safetensors import safe_open

            with safe_open(str(safetensors_path), framework="pt", device="cpu") as fh:
                if "classifier.weight" in fh.keys():
                    shape = fh.get_tensor("classifier.weight").shape
                    return int(shape[0]) == len(label_list)
        except Exception:  # noqa: BLE001 - corrupt/incomplete checkpoint
            return False
    bin_path = checkpoint / "pytorch_model.bin"
    if bin_path.exists():
        try:
            import torch

            state = torch.load(bin_path, map_location="cpu")
            weight = state.get("classifier.weight")
            if weight is not None:
                return int(weight.shape[0]) == len(label_list)
        except Exception:  # noqa: BLE001 - corrupt/incomplete checkpoint
            return False
    return True


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
    max_device_batch: int = 32,
    init_weights_path: str | None = None,
    metric_for_best_model: str = "macro_pr_auc",
) -> dict[str, Any]:
    import os as _os
    # Reduce CUDA fragmentation OOMs (the allocator can reuse freed blocks).
    _os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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

    # Warm-start: overlay previously-trained weights (e.g. a saved model.safetensors
    # from an interrupted run) onto the architecture. This is NOT a true resume
    # (no optimizer/scheduler state), but it continues from the partially-trained
    # weights instead of the base model, preserving prior progress.
    if init_weights_path:
        from safetensors.torch import load_file as _load_safetensors

        state = _load_safetensors(init_weights_path)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[finetune] warm-started from {init_weights_path} "
              f"(missing={len(missing)} unexpected={len(unexpected)} keys)")

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
        from sklearn.metrics import average_precision_score, f1_score

        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))
        preds = probs >= 0.5
        best_f1s: list[float] = []
        pr_aucs: list[float] = []
        out: dict[str, float] = {
            "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        }
        for idx, lab in enumerate(label_list):
            gold = labels[:, idx]
            scores = probs[:, idx]
            if np.sum(gold) > 0 and np.sum(gold) < len(gold):
                pr_aucs.append(float(average_precision_score(gold, scores)))
            best = 0.0
            for thr in _METRIC_THRESHOLDS:
                pred = scores >= thr
                best = max(best, float(f1_score(gold, pred, zero_division=0)))
            best_f1s.append(best)
            out[f"best_f1_{lab}"] = best
        out["macro_best_f1"] = float(np.mean(best_f1s)) if best_f1s else 0.0
        out["macro_pr_auc"] = float(np.mean(pr_aucs)) if pr_aucs else 0.0
        return {
            key: round(val, 6) for key, val in out.items()
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
    # Cap the per-device batch so memory-heavy models (DeBERTa-v3) fit on a T4,
    # and use gradient accumulation to preserve the requested *effective* batch.
    # e.g. requested 320 -> per_device 32 x accum 10. Memory is bounded by the
    # per-device batch; the optimizer still sees the full effective batch.
    per_device = min(batch_size, max_device_batch)
    grad_accum = max(1, round(batch_size / per_device))
    # DeBERTa-v1 ("deberta") masks attention with finfo.min, which overflows in
    # fp16 ("value cannot be converted to type c10::Half without overflow").
    # Disable fp16 for that architecture; keep it for deberta-v2, albert, bert, etc.
    model_type = getattr(model.config, "model_type", "")
    use_fp16 = use_cuda and model_type != "deberta"
    print(f"[finetune] per_device_batch={per_device} x grad_accum={grad_accum} "
          f"= effective {per_device * grad_accum} | fp16={use_fp16} "
          f"(model_type={model_type})")
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device,
        per_device_eval_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        # Mixed precision on GPU: ~2x faster. Skipped for deberta-v1 (fp16 overflow).
        fp16=use_fp16,
        dataloader_num_workers=2,
        # Step-based save/eval so a crash or disconnect only loses the steps since
        # the last checkpoint (not the whole run). save_total_limit caps disk use.
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=250,
        save_steps=250,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
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
        if _checkpoint_is_compatible(last_ckpt, label_list):
            print(f"[finetune] resuming from checkpoint: {last_ckpt}")
        else:
            print(
                f"[finetune] ignoring incompatible checkpoint: {last_ckpt} "
                f"(expected labels={label_list})"
            )
            last_ckpt = None
    try:
        trainer.train(resume_from_checkpoint=last_ckpt)
    except RuntimeError as exc:
        msg = str(exc)
        if last_ckpt and "size mismatch for classifier" in msg:
            print(
                f"[finetune] checkpoint had incompatible classifier weights; "
                f"restarting from base model instead: {last_ckpt}"
            )
            trainer.train(resume_from_checkpoint=None)
        else:
            raise
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
