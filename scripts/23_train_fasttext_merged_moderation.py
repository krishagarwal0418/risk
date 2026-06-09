#!/usr/bin/env python3
"""Train a stronger FastText head for the merged Koala moderation task.

Labels:
  * safe
  * harmful_content
  * sexual

Input data comes from ``data/koala_merged_moderation/{train,val,test}.jsonl``.
Run ``scripts/19_finetune_koala_merged_moderation.py --build-only`` first.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from safety_classifier import constants as C
from safety_classifier.config import repo_root
from safety_classifier.fasttext_layer.evaluator import evaluate_model
from safety_classifier.fasttext_layer.trainer import FastTextHyperParams
from safety_classifier.reporting import Status, print_summary, write_report

DATA_DIR = repo_root() / "data" / "koala_merged_moderation"
FASTTEXT_DIR = repo_root() / "data" / "fasttext"
MODELS_DIR = repo_root() / "models" / "fasttext"
REPORTS_DIR = repo_root() / "reports"
HEAD = "merged_moderation"
HARMFUL = "harmful_content"
LABELS = (C.SAFE, HARMFUL, C.SEXUAL)


def _labels_for(row: dict[str, Any]) -> list[str]:
    labels = list(row.get("labels") or [])
    return labels if labels else [C.SAFE]


def _fasttext_line(labels: list[str], text: str) -> str:
    flat = " ".join(str(text).split())
    prefix = " ".join(f"__label__{lab}" for lab in labels)
    return f"{prefix} {flat}"


def build_fasttext_files(max_train: int = 0) -> dict[str, Any]:
    FASTTEXT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        src = DATA_DIR / f"{split}.jsonl"
        if not src.exists():
            raise FileNotFoundError(
                f"missing {src}. Run scripts/19_finetune_koala_merged_moderation.py --build-only first."
            )
        rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
        if split == "train" and max_train and len(rows) > max_train:
            rows = rows[:max_train]
        counts: Counter = Counter()
        out_path = FASTTEXT_DIR / f"{HEAD}_{split}.txt"
        with out_path.open("w", encoding="utf-8") as fh:
            for row in rows:
                labels = _labels_for(row)
                counts.update(labels)
                fh.write(_fasttext_line(labels, row["text"]) + "\n")
        summary[split] = {
            "rows": len(rows),
            "labels": dict(counts),
            "path": str(out_path.relative_to(repo_root())),
        }
        print(f"[fasttext:{HEAD}] {split}: rows={len(rows):,} labels={dict(counts)}", flush=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / f"fasttext_{HEAD}_data.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _load_model(path: Path):
    import fasttext

    fasttext.FastText.eprint = lambda *a, **k: None  # type: ignore[attr-defined]
    return fasttext.load_model(str(path))


def train(params: FastTextHyperParams, cutoff: int) -> dict[str, Any]:
    import fasttext

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train_file = FASTTEXT_DIR / f"{HEAD}_train.txt"
    val_file = FASTTEXT_DIR / f"{HEAD}_val.txt"
    test_file = FASTTEXT_DIR / f"{HEAD}_test.txt"
    if not train_file.exists():
        build_fasttext_files()

    started = time.time()
    train_kwargs = params.to_train_kwargs()
    print(f"[fasttext:{HEAD}] training .bin with params={train_kwargs}", flush=True)
    model = fasttext.train_supervised(input=str(train_file), **train_kwargs)

    bin_path = MODELS_DIR / f"{HEAD}_head.bin"
    ftz_path = MODELS_DIR / f"{HEAD}_head.ftz"
    model.save_model(str(bin_path))
    bin_size = bin_path.stat().st_size

    print(f"[fasttext:{HEAD}] quantizing .ftz cutoff={cutoff:,}", flush=True)
    model.quantize(input=str(train_file), qnorm=True, retrain=True, cutoff=cutoff)
    model.save_model(str(ftz_path))
    ftz_size = ftz_path.stat().st_size

    print(f"[fasttext:{HEAD}] evaluating .ftz", flush=True)
    ftz_model = _load_model(ftz_path)
    val_metrics = evaluate_model(ftz_model, val_file) if val_file.exists() else None
    test_metrics = evaluate_model(ftz_model, test_file) if test_file.exists() else None

    metadata = {
        "head": HEAD,
        "labels": [lab[len("__label__"):] for lab in ftz_model.get_labels()],
        "hyperparameters": asdict(params),
        "cutoff": cutoff,
        "bin_path": str(bin_path.relative_to(repo_root())),
        "ftz_path": str(ftz_path.relative_to(repo_root())),
        "model_size_bytes_before_quantization": bin_size,
        "model_size_bytes_after_quantization": ftz_size,
        "compression_ratio": round(bin_size / ftz_size, 2) if ftz_size else None,
        "train_seconds": round(time.time() - started, 2),
        "metrics": {"val": val_metrics, "test": test_metrics},
    }
    (MODELS_DIR / f"{HEAD}_head_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    write_report(
        f"fasttext_{HEAD}_eval",
        status=Status.PASS,
        title="FastText Merged Moderation Head",
        summary=metadata,
        tables=[
            (
                "Evaluation",
                ["Split", "Macro Best F1", "Route Floor Recall", "p95 ms"],
                [
                    [
                        split,
                        metrics["macro_best_f1"],
                        metrics["macro_route_floor_recall"],
                        metrics["latency_ms"]["p95"],
                    ]
                    for split, metrics in (("val", val_metrics), ("test", test_metrics))
                    if metrics
                ],
            )
        ],
    )
    rows = [
        [split, metrics["macro_best_f1"], metrics["macro_route_floor_recall"], metrics["latency_ms"]["p95"]]
        for split, metrics in (("val", val_metrics), ("test", test_metrics))
        if metrics
    ]
    print_summary(
        "FastText Merged Moderation Head",
        Status.PASS,
        ["Split", "Macro Best F1", "Route Floor Recall", "p95 ms"],
        rows,
    )
    print(f"[fasttext:{HEAD}] wrote {bin_path} and {ftz_path}", flush=True)
    return metadata


def main() -> None:
    p = argparse.ArgumentParser(description="Train the merged moderation FastText head")
    p.add_argument("--build-only", action="store_true")
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--profile", choices=["strong", "fast"], default="strong")
    p.add_argument("--epoch", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--dim", type=int, default=None)
    p.add_argument("--wordNgrams", type=int, default=None)
    p.add_argument("--minn", type=int, default=2)
    p.add_argument("--maxn", type=int, default=5)
    p.add_argument("--bucket", type=int, default=None)
    p.add_argument("--thread", type=int, default=None)
    p.add_argument("--verbose", type=int, default=2)
    p.add_argument("--cutoff", type=int, default=200_000)
    args = p.parse_args()

    build_fasttext_files(max_train=args.max_train)
    if args.build_only:
        return

    defaults = {
        "fast": {"epoch": 25, "lr": 0.5, "dim": 100, "wordNgrams": 3, "bucket": 2_000_000},
        "strong": {"epoch": 50, "lr": 0.4, "dim": 200, "wordNgrams": 4, "bucket": 5_000_000},
    }[args.profile]
    params = FastTextHyperParams(
        epoch=args.epoch or defaults["epoch"],
        lr=args.lr or defaults["lr"],
        dim=args.dim or defaults["dim"],
        wordNgrams=args.wordNgrams or defaults["wordNgrams"],
        minn=args.minn,
        maxn=args.maxn,
        bucket=args.bucket or defaults["bucket"],
        thread=args.thread,
        verbose=args.verbose,
    )
    train(params, cutoff=args.cutoff)


if __name__ == "__main__":
    main()
