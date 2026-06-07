#!/usr/bin/env python3
"""Adapt the moderation model to REAL-WORLD toxicity (google/civil_comments).

Your hate/toxicity training leans on ToxiGen (AI-generated), which inflates eval
and may not generalize. This trains one warm-started epoch on civil_comments
(real Wikipedia/news comments), blended with the existing koala data so the model
doesn't forget self_harm/sexual (which civil_comments barely covers). The val set
is real-world (held-out civil_comments), so the reported numbers are honest.

Maps civil_comments float scores (>=0.5 = positive) to the 6 koala labels:
  toxicity<-toxicity, hate<-identity_attack, harassment<-insult,
  violence<-threat, sexual<-sexual_explicit   (self_harm: none in civil)

Usage:
    python scripts/22_finetune_realworld_toxicity.py \
        --init-weights models/finetuned/moderation/model.safetensors \
        --epochs 1 --batch-size 320
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import _bootstrap  # noqa: F401

from safety_classifier import constants as C
from safety_classifier.config import repo_root, load_models_config
from safety_classifier.transformers_layer.finetune import finetune

CIVIL_MAP = {
    "toxicity": C.TOXICITY,
    "identity_attack": C.HATE,
    "insult": C.HARASSMENT,
    "threat": C.VIOLENCE,
    "sexual_explicit": C.SEXUAL,
}
OUT = repo_root() / "data" / "realworld_tox"


def _civil_labels(row, thr=0.5) -> list[str]:
    out = []
    for col, canon in CIVIL_MAP.items():
        v = row.get(col)
        if v is not None and float(v) >= thr:
            out.append(canon)
    return list(dict.fromkeys(out))


def build_civil(max_toxic: int, neg_ratio: float, seed: int) -> dict:
    from datasets import load_dataset
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    out = {}
    for split, hf_split in [("train", "train"), ("test", "test")]:
        ds = load_dataset("google/civil_comments", split=hf_split, streaming=True)
        cap_tox = max_toxic if split == "train" else max_toxic // 8
        tox, clean = [], []
        for row in ds:
            text = (row.get("text") or "").strip()
            if not text or len(text) < 8 or len(text) > 2000:
                continue
            labs = _civil_labels(row)
            if labs:
                if len(tox) < cap_tox:
                    tox.append({"text": text, "labels": labs})
            else:
                clean.append({"text": text, "labels": [C.SAFE]})
            if len(tox) >= cap_tox and len(clean) >= int(cap_tox * neg_ratio):
                break
        rng.shuffle(clean)
        rows = tox + clean[: int(len(tox) * neg_ratio)]
        rng.shuffle(rows)
        (OUT / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        from collections import Counter
        c = Counter(l for r in rows for l in r["labels"])
        print(f"  civil {split}: {len(rows)} rows | {dict(c)}")
        out[split] = len(rows)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-weights", default="models/finetuned/moderation/model.safetensors")
    ap.add_argument("--output", default="models/finetuned/moderation_rw")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=320)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-toxic", type=int, default=40000,
                    help="Max toxic civil_comments rows for train")
    ap.add_argument("--neg-ratio", type=float, default=1.5)
    ap.add_argument("--mix-koala", action="store_true", default=True,
                    help="Blend existing koala train to avoid forgetting self_harm/sexual")
    ap.add_argument("--no-mix-koala", dest="mix_koala", action="store_false")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()

    root = repo_root()
    if not args.skip_build:
        print("[realworld] building civil_comments data ...")
        build_civil(args.max_toxic, args.neg_ratio, args.seed)

    # Train file: civil train (+ optionally blended koala train).
    civil_train = OUT / "train.jsonl"
    train_rows = [json.loads(l) for l in civil_train.read_text().splitlines() if l.strip()]
    if args.mix_koala:
        koala_train = root / "data" / "koala_moderation" / "train.jsonl"
        if koala_train.exists():
            extra = [json.loads(l) for l in koala_train.read_text().splitlines() if l.strip()]
            train_rows += extra
            print(f"[realworld] blended {len(extra)} koala rows "
                  f"(total {len(train_rows)}) to retain self_harm/sexual")
        else:
            print("[realworld] WARN: no koala train to blend; self_harm/sexual may degrade")
    random.Random(args.seed).shuffle(train_rows)
    blended = OUT / "train_blended.jsonl"
    blended.write_text("\n".join(json.dumps(r) for r in train_rows), encoding="utf-8")

    model_name = load_models_config()["transformers"]["moderation_primary"]["hf_name"]
    print(f"[realworld] warm-start {model_name} from {args.init_weights}")
    metrics = finetune(
        model_name=model_name,
        task="koala_moderation",
        train_path=str(blended),
        val_path=str(OUT / "test.jsonl"),   # REAL-WORLD val = honest metric
        output_dir=str(root / args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        init_weights_path=args.init_weights,
        metric_for_best_model="macro_pr_auc",
    )
    print("\n=== REAL-WORLD (civil_comments) eval ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
