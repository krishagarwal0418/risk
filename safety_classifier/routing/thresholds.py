"""Threshold access + calibration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .. import constants as C
from ..config import load_thresholds_config, repo_root

_DEFAULT = {"route": 0.20, "review": 0.50, "block": 0.80}


@dataclass
class LabelThresholds:
    route: float
    review: float
    block: float


_RECOMMENDED_THRESHOLDS = repo_root() / "reports" / "fasttext_thresholds_recommended.yaml"


class Thresholds:
    """Per-label route/review/block thresholds, loaded from configs/thresholds.yaml.

    If ``use_calibrated`` is set and ``reports/fasttext_thresholds_recommended.yaml``
    exists, the calibrated (high-recall) route thresholds overlay the config
    defaults — review/block thresholds are kept from config.
    """

    def __init__(
        self,
        raw: dict[str, Any] | None = None,
        use_calibrated: bool = True,
    ) -> None:
        raw = raw if raw is not None else load_thresholds_config()
        calibrated = self._load_calibrated() if use_calibrated else {}
        self.calibration_loaded = bool(calibrated)
        self._map: dict[str, LabelThresholds] = {}
        for label in C.SCORED_LABELS:
            entry = raw.get(label, _DEFAULT)
            route = float(entry.get("route", _DEFAULT["route"]))
            if label in calibrated and "route" in calibrated[label]:
                route = float(calibrated[label]["route"])
            self._map[label] = LabelThresholds(
                route=route,
                review=float(entry.get("review", _DEFAULT["review"])),
                block=float(entry.get("block", _DEFAULT["block"])),
            )

    @staticmethod
    def _load_calibrated() -> dict[str, Any]:
        if _RECOMMENDED_THRESHOLDS.exists():
            try:
                return yaml.safe_load(_RECOMMENDED_THRESHOLDS.read_text(encoding="utf-8")) or {}
            except Exception:  # noqa: BLE001
                return {}
        return {}

    def for_label(self, label: str) -> LabelThresholds:
        return self._map.get(label, LabelThresholds(**_DEFAULT))

    def route(self, label: str) -> float:
        return self.for_label(label).route

    def review(self, label: str) -> float:
        return self.for_label(label).review

    def block(self, label: str) -> float:
        return self.for_label(label).block

    def as_dict(self) -> dict[str, dict[str, float]]:
        return {
            lab: {"route": t.route, "review": t.review, "block": t.block}
            for lab, t in self._map.items()
        }


def save_thresholds(thresholds: dict[str, dict[str, float]], path: Path | None = None) -> Path:
    """Persist calibrated thresholds to a YAML file."""
    path = path or (repo_root() / "configs" / "thresholds.yaml")
    path.write_text(yaml.safe_dump(thresholds, sort_keys=True), encoding="utf-8")
    return path


def calibrate_review_block_from_val(
    val_jsonl: Path, predict_fn, labels: list[str] | None = None
) -> dict[str, dict[str, float]]:
    """Sweep review/block thresholds against a validation JSONL.

    ``predict_fn(text) -> dict[label, score]`` is the merged scorer. For each label
    we choose review at the best-F1 point and block at a high-precision point.
    """
    import json

    labels = labels or list(C.SCORED_LABELS)
    rows = [
        json.loads(line)
        for line in Path(val_jsonl).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gold = [set(r.get("labels", [])) for r in rows]
    preds = [predict_fn(r["text"]) for r in rows]

    grid = [round(0.05 * i, 2) for i in range(1, 20)]
    out: dict[str, dict[str, float]] = {}
    for label in labels:
        best_f1, best_thr = 0.0, 0.5
        best_prec_thr = 0.8
        for thr in grid:
            tp = fp = fn = 0
            for g, p in zip(gold, preds):
                predicted = p.get(label, 0.0) >= thr
                actual = label in g
                if predicted and actual:
                    tp += 1
                elif predicted and not actual:
                    fp += 1
                elif not predicted and actual:
                    fn += 1
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
            if prec >= 0.95:
                best_prec_thr = min(best_prec_thr, thr) if best_prec_thr != 0.8 else thr
        out[label] = {
            "route": 0.20,
            "review": best_thr,
            "block": max(best_thr, best_prec_thr),
        }
    return out
