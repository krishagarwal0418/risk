"""Download datasets from Hugging Face (and local files) into ``data/raw/``.

Robust to gated / unavailable datasets: a single failure is recorded and does not
crash the pipeline. Writes ``data/raw/<name>.jsonl`` plus ``data/raw/metadata.json``
with a final availability report.
"""

from __future__ import annotations

import glob
import json
import time
from pathlib import Path
from typing import Any, Iterable

from ..config import repo_root
from .dataset_adapters import get_adapter
from .dataset_registry import DatasetSpec, load_dataset_specs

RAW_DIR = repo_root() / "data" / "raw"
CUSTOM_DIR = repo_root() / "data" / "custom"


def _load_hf_dataset(spec: DatasetSpec):
    """Load a HF dataset and return ``(rows_iterable, columns, license, n_raw)``."""
    from datasets import load_dataset  # imported lazily

    kwargs: dict[str, Any] = {}
    if spec.config:
        kwargs["name"] = spec.config
    ds = load_dataset(spec.hf_name, **kwargs)

    # Collect column names + license from the first available split's features.
    if hasattr(ds, "keys"):
        first = ds[next(iter(ds.keys()))]
    else:
        first = ds
    columns = list(getattr(first, "column_names", []) or [])
    license_str = None
    info = getattr(first, "info", None)
    if info is not None:
        license_str = getattr(info, "license", None) or None

    def _rows():
        if hasattr(ds, "keys"):
            for split in ds.keys():
                for row in ds[split]:
                    yield dict(row)
        else:
            for row in ds:
                yield dict(row)

    n_raw = 0
    if hasattr(ds, "keys"):
        n_raw = sum(len(ds[s]) for s in ds.keys())
    else:
        n_raw = len(ds)
    return _rows(), columns, license_str, n_raw


def _iter_hf_rows(spec: DatasetSpec) -> Iterable[dict]:
    """Yield rows from a HF dataset, concatenating all available splits."""
    rows, _cols, _lic, _n = _load_hf_dataset(spec)
    yield from rows


def _detect_columns(columns: list[str]) -> tuple[str | None, str | None]:
    """Best-effort detection of the text + label columns for reporting."""
    from .dataset_adapters import _TEXT_KEYS

    text_col = next((c for c in _TEXT_KEYS if c in columns), None)
    label_candidates = (
        "label", "labels", "type", "toxicity", "category",
        "prompt_label", "prompt_harm_label", "is_safe", "moderation",
    )
    label_col = next((c for c in label_candidates if c in columns), None)
    return text_col, label_col


def _iter_local_rows(path: Path) -> Iterable[dict]:
    import csv

    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
    elif path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                yield dict(row)


def _write_jsonl(records: Iterable[dict], out_path: Path) -> tuple[int, int]:
    """Write records; return (written, skipped_empty_text)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    skipped = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            if not rec.get("text") or not str(rec["text"]).strip():
                skipped += 1
                continue
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count, skipped


def _adapter_exists(kind: str) -> bool:
    from .dataset_adapters import ADAPTERS

    return kind in ADAPTERS


def download_one(spec: DatasetSpec) -> dict[str, Any]:
    """Download and adapt a single dataset spec. Never raises."""
    out_path = RAW_DIR / f"{spec.name}.jsonl"
    started = time.time()
    base = {
        "name": spec.name,
        "hf_name": spec.hf_name,
        "kind": spec.kind,
        "adapter_exists": _adapter_exists(spec.kind),
    }
    try:
        rows, columns, license_str, n_raw = _load_hf_dataset(spec)
        text_col, label_col = _detect_columns(columns)
        adapter = get_adapter(spec.kind)
        written, skipped = _write_jsonl(adapter(rows, spec.name), out_path)
        return {
            **base,
            "status": "ok" if written > 0 else "empty",
            "rows_raw": n_raw,
            "records": written,
            "samples_used": written,
            "samples_skipped": skipped,
            "columns": columns,
            "detected_text_column": text_col,
            "detected_label_column": label_col,
            "license": license_str,
            "gated": False,
            "path": str(out_path.relative_to(repo_root())),
            "seconds": round(time.time() - started, 2),
        }
    except Exception as exc:  # noqa: BLE001 - intentionally broad; report and continue
        msg = f"{type(exc).__name__}: {exc}"
        gated = any(
            tok in msg.lower() for tok in ("gated", "401", "403", "authentication", "access")
        )
        return {
            **base,
            "status": "unavailable",
            "error": msg,
            "gated": gated,
            "rows_raw": 0,
            "records": 0,
            "samples_used": 0,
            "samples_skipped": 0,
            "columns": [],
            "detected_text_column": None,
            "detected_label_column": None,
            "license": None,
            "seconds": round(time.time() - started, 2),
        }


def download_custom() -> list[dict[str, Any]]:
    """Adapt local CSV/JSONL files under ``data/custom/``."""
    results: list[dict[str, Any]] = []
    patterns = [str(CUSTOM_DIR / "*.csv"), str(CUSTOM_DIR / "*.jsonl")]
    for pattern in patterns:
        for fpath in sorted(glob.glob(pattern)):
            path = Path(fpath)
            name = f"custom_{path.stem}"
            out_path = RAW_DIR / f"{name}.jsonl"
            try:
                adapter = get_adapter("generic")
                written, skipped = _write_jsonl(adapter(_iter_local_rows(path), name), out_path)
                results.append(
                    {
                        "name": name, "status": "ok", "records": written,
                        "samples_used": written, "samples_skipped": skipped,
                        "kind": "generic", "adapter_exists": True, "gated": False,
                        "source_file": str(path),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                results.append(
                    {
                        "name": name, "status": "error", "error": str(exc),
                        "records": 0, "kind": "generic", "gated": False,
                    }
                )
    return results


def download_all() -> dict[str, Any]:
    """Download every enabled dataset + custom files; write metadata + report."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    specs = load_dataset_specs()
    results = [download_one(spec) for spec in specs]
    results.extend(download_custom())

    available = [r for r in results if r["status"] == "ok"]
    metadata = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "datasets": results,
        "total_records": sum(r["records"] for r in results),
        "available_count": len(available),
        "total_count": len(results),
    }
    (RAW_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    _write_download_report(metadata)
    return metadata


def _write_download_report(metadata: dict[str, Any]) -> dict[str, Any]:
    """Write reports/data_download_report.{json,md} with PASS/WARN/FAIL status."""
    from ..reporting import ReportSection, Status, print_summary, write_report

    results = metadata["datasets"]
    available = metadata["available_count"]
    total = metadata["total_count"]
    gated = [r for r in results if r.get("gated")]

    if available == 0:
        status = Status.FAIL
    elif available < total:
        status = Status.WARN
    else:
        status = Status.PASS

    warnings = [
        f"{r['name']} unavailable: {r.get('error', 'unknown')}"
        for r in results
        if r["status"] not in ("ok",)
    ]
    failures = ["No datasets downloaded successfully"] if available == 0 else []

    headers = ["Dataset", "Status", "Rows", "Notes"]
    table_rows = []
    for r in results:
        note = "usable"
        if r["status"] != "ok":
            note = "gated/unavailable" if r.get("gated") else r.get("error", "failed")[:40]
        elif not r.get("adapter_exists", True):
            note = "no adapter"
        table_rows.append([
            r.get("hf_name") or r["name"],
            r["status"].upper(),
            r.get("records", 0),
            note,
        ])

    section = ReportSection("Summary")
    section.add("available_datasets", f"{available}/{total}")
    section.add("total_records", metadata["total_records"])
    section.add("gated_or_unavailable", len(results) - available)

    write_report(
        "data_download_report",
        status=status,
        title="Data Download Report",
        summary={
            "available_count": available,
            "total_count": total,
            "total_records": metadata["total_records"],
            "datasets": results,
        },
        sections=[section],
        tables=[("Datasets", headers, table_rows)],
        warnings=warnings,
        failures=failures,
    )
    print_summary("Data Download", status, headers, table_rows)
    return metadata


def print_report(metadata: dict[str, Any]) -> None:
    """Backwards-compatible terminal report (the structured report is written
    automatically by :func:`download_all`)."""
    headers = ["Dataset", "Status", "Rows", "Notes"]
    rows = []
    for r in metadata["datasets"]:
        note = "usable" if r["status"] == "ok" else (r.get("error", "") or "")[:40]
        rows.append([r.get("hf_name") or r["name"], r["status"].upper(), r.get("records", 0), note])
    from ..reporting import format_table

    print("\n================ DATASET AVAILABILITY REPORT ================")
    print(format_table(headers, rows))
    print("------------------------------------------------------------")
    print(
        f"  Available: {metadata['available_count']}/{metadata['total_count']} "
        f"datasets, {metadata['total_records']} total records"
    )
    print("============================================================\n")
