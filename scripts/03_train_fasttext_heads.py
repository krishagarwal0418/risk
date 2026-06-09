"""Train + quantize the FastText safety heads."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from safety_classifier.fasttext_layer.trainer import FastTextHyperParams, train_all_heads


def main() -> None:
    p = argparse.ArgumentParser(description="Train FastText safety heads")
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
    args = p.parse_args()

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
    results = train_all_heads(params=params)
    for head, meta in results.items():
        print(f"[train] {head}: labels={meta['labels']} metrics={meta.get('metrics')}")


if __name__ == "__main__":
    main()
