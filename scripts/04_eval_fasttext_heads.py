"""Evaluate + calibrate the FastText heads, writing reports."""

from __future__ import annotations

import _bootstrap  # noqa: F401

from safety_classifier.evaluation.fasttext_eval import evaluate_all_heads_report


def main() -> None:
    print("[eval] evaluating heads (.bin vs .ftz) ...")
    evaluate_all_heads_report()
    print(
        "[eval] reports written to reports/fasttext_{attack,abuse,high_risk}_eval.{json,md}\n"
        "[eval] run scripts/04b_calibrate_fasttext_thresholds.py to calibrate thresholds"
    )


if __name__ == "__main__":
    main()
