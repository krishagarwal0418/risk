"""Train and quantize the three FastText safety heads.

Heads:
  * attack     -> safe / prompt_injection / jailbreak
  * abuse      -> safe / toxicity / hate / harassment
  * high_risk  -> safe / sexual / violence / self_harm /
                  dangerous_information / illegal_activity

Each head is trained with ``fasttext.train_supervised`` using a one-vs-all loss
(so it behaves as a multi-label scorer), then quantized to a compact ``.ftz``.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..config import repo_root

FASTTEXT_DATA_DIR = repo_root() / "data" / "fasttext"
MODELS_DIR = repo_root() / "models" / "fasttext"

HEADS = ("attack", "abuse", "high_risk")


@dataclass
class FastTextHyperParams:
    loss: str = "ova"
    wordNgrams: int = 3   # capture multi-word attack phrases
    dim: int = 200        # richer embeddings (data is pre-balanced upstream)
    epoch: int = 35
    lr: float = 0.5
    minn: int = 2
    maxn: int = 5

    def to_train_kwargs(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "wordNgrams": self.wordNgrams,
            "dim": self.dim,
            "epoch": self.epoch,
            "lr": self.lr,
            "minn": self.minn,
            "maxn": self.maxn,
        }


def _label_counts(train_file: Path) -> dict[str, int]:
    counts: Counter = Counter()
    if not train_file.exists():
        return {}
    for line in train_file.read_text(encoding="utf-8").splitlines():
        for tok in line.split():
            if tok.startswith("__label__"):
                counts[tok[len("__label__"):]] += 1
            else:
                break
    return dict(counts)


def _collect_sources(head: str) -> list[str]:
    """Best-effort list of dataset sources contributing to this head."""
    report = repo_root() / "reports" / "split_report.json"
    if report.exists():
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
            return sorted(data.get("source_distribution", {}).keys())
        except Exception:  # noqa: BLE001
            return []
    return []


def train_head(
    head: str,
    params: FastTextHyperParams | None = None,
    cutoff: int = 100_000,
) -> dict[str, Any]:
    """Train + quantize a single head, writing ``.bin``, ``.ftz`` and metadata."""
    import fasttext  # imported lazily so unit tests don't require the wheel

    params = params or FastTextHyperParams()
    train_file = FASTTEXT_DATA_DIR / f"{head}_train.txt"
    val_file = FASTTEXT_DATA_DIR / f"{head}_val.txt"
    test_file = FASTTEXT_DATA_DIR / f"{head}_test.txt"
    if not train_file.exists():
        raise FileNotFoundError(
            f"Training file missing for head '{head}': {train_file}. "
            "Run scripts/02_prepare_fasttext_data.py first."
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model = fasttext.train_supervised(input=str(train_file), **params.to_train_kwargs())

    bin_path = MODELS_DIR / f"{head}_head.bin"
    ftz_path = MODELS_DIR / f"{head}_head.ftz"
    model.save_model(str(bin_path))
    bin_size = bin_path.stat().st_size if bin_path.exists() else 0

    # Quantize to .ftz.
    model.quantize(input=str(train_file), qnorm=True, retrain=True, cutoff=cutoff)
    model.save_model(str(ftz_path))
    ftz_size = ftz_path.stat().st_size if ftz_path.exists() else 0

    # Validation metric (fastText built-in test).
    metrics: dict[str, Any] = {}
    if val_file.exists():
        n, p, r = model.test(str(val_file))
        metrics["val"] = {"samples": n, "precision_at_1": p, "recall_at_1": r}
    if test_file.exists():
        n, p, r = model.test(str(test_file))
        metrics["test"] = {"samples": n, "precision_at_1": p, "recall_at_1": r}

    metadata = {
        "head": head,
        "labels": [lab[len("__label__"):] for lab in model.get_labels()],
        "train_file": str(train_file.relative_to(repo_root())),
        "val_file": str(val_file.relative_to(repo_root())) if val_file.exists() else None,
        "test_file": str(test_file.relative_to(repo_root())) if test_file.exists() else None,
        "dataset_sources": _collect_sources(head),
        "class_counts": _label_counts(train_file),
        "hyperparameters": asdict(params),
        "train_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "metrics": metrics,
        "bin_path": str(bin_path.relative_to(repo_root())),
        "ftz_path": str(ftz_path.relative_to(repo_root())),
        "model_size_bytes_before_quantization": bin_size,
        "model_size_bytes_after_quantization": ftz_size,
        "compression_ratio": round(bin_size / ftz_size, 2) if ftz_size else None,
        "train_seconds": round(time.time() - started, 2),
        # Threshold recommendations are refined by the calibrator; defaults here.
        "threshold_recommendations": {"route": 0.20},
    }
    meta_path = MODELS_DIR / f"{head}_head_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def train_all_heads(params: FastTextHyperParams | None = None) -> dict[str, Any]:
    results = {}
    for head in HEADS:
        print(f"[fasttext] training head: {head}")
        results[head] = train_head(head, params=params)
        m = results[head].get("metrics", {})
        print(f"[fasttext] {head} done -> metrics: {m}")
    return results
