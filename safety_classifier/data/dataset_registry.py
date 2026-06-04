"""Dataset registry helpers — reads ``configs/datasets.yaml``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..config import load_datasets_config


@dataclass
class DatasetSpec:
    name: str
    loader: str          # "hf" | "local"
    kind: str            # selects the adapter
    enabled: bool = True
    hf_name: Optional[str] = None
    config: Optional[str] = None
    split: Optional[str] = None
    path: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DatasetSpec":
        return cls(
            name=d["name"],
            loader=d.get("loader", "hf"),
            kind=d.get("kind", "generic"),
            enabled=d.get("enabled", True),
            hf_name=d.get("hf_name"),
            config=d.get("config"),
            split=d.get("split"),
            path=d.get("path"),
        )


def load_dataset_specs(include_disabled: bool = False) -> list[DatasetSpec]:
    cfg = load_datasets_config()
    specs = [DatasetSpec.from_dict(d) for d in cfg.get("datasets", [])]
    if include_disabled:
        return specs
    return [s for s in specs if s.enabled]
