"""Build processed splits and per-head FastText training files."""

from __future__ import annotations

import json

import _bootstrap  # noqa: F401

from safety_classifier.data.splitter import build_fasttext_files, build_processed_splits
from safety_classifier.evaluation.data_eval import build_distribution_report


def main() -> None:
    print("[prepare] building processed train/val/test splits ...")
    split_report = build_processed_splits()
    print(json.dumps(split_report["split_sizes"], indent=2))
    print("[prepare] building FastText head files ...")
    ft_report = build_fasttext_files()
    for head, splits in ft_report.items():
        print(f"  {head}: {splits.get('train', {})}")
    print("[prepare] generating data distribution report ...")
    build_distribution_report()
    print("[prepare] done. See reports/data_distribution_report.md")


if __name__ == "__main__":
    main()
