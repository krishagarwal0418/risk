"""Combine all stage reports into a single final pipeline report."""

from __future__ import annotations

import _bootstrap  # noqa: F401

from safety_classifier.evaluation.final_report import generate_final_report


def main() -> None:
    generate_final_report()
    print("\n[final] reports/final_pipeline_report.{json,md} written")


if __name__ == "__main__":
    main()
