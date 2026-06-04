"""Calibrate FastText routing thresholds on the validation set.

Writes:
  reports/fasttext_thresholds_recommended.yaml   (loaded by the router)
  reports/fasttext_threshold_calibration.{json,md}
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

from safety_classifier.fasttext_layer.calibrator import calibrate_all


def main() -> None:
    calib = calibrate_all()
    print("[calibrate] recommended route thresholds (high recall):")
    for label, thr in calib.get("route_thresholds", {}).items():
        print(f"  {label}: {thr}")
    print("[calibrate] reports/fasttext_thresholds_recommended.yaml written")


if __name__ == "__main__":
    main()
