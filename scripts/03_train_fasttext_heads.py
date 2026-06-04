"""Train + quantize the three FastText safety heads."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from safety_classifier.fasttext_layer.trainer import FastTextHyperParams, train_all_heads


def main() -> None:
    p = argparse.ArgumentParser(description="Train FastText safety heads")
    p.add_argument("--epoch", type=int, default=25)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--dim", type=int, default=100)
    p.add_argument("--wordNgrams", type=int, default=2)
    p.add_argument("--minn", type=int, default=2)
    p.add_argument("--maxn", type=int, default=5)
    args = p.parse_args()

    params = FastTextHyperParams(
        epoch=args.epoch,
        lr=args.lr,
        dim=args.dim,
        wordNgrams=args.wordNgrams,
        minn=args.minn,
        maxn=args.maxn,
    )
    results = train_all_heads(params=params)
    for head, meta in results.items():
        print(f"[train] {head}: labels={meta['labels']} metrics={meta.get('metrics')}")


if __name__ == "__main__":
    main()
