"""Shared reporting utilities: PASS/WARN/FAIL status, JSON+MD report writers,
summary tables, latency percentiles and human-readable sizes.

Every evaluation step produces three things via :func:`write_report`:
  1. a machine-readable ``reports/<name>.json``
  2. a human-readable ``reports/<name>.md``
  3. a clear ``status`` field (PASS / WARN / FAIL)

and prints a terminal summary table.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

from .config import repo_root

REPORTS_DIR = repo_root() / "reports"


class Status(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"

    @staticmethod
    def worst(statuses: Sequence["Status | str"]) -> "Status":
        order = {Status.PASS: 0, Status.WARN: 1, Status.FAIL: 2}
        worst = Status.PASS
        for s in statuses:
            s = Status(s) if not isinstance(s, Status) else s
            if order[s] > order[worst]:
                worst = s
        return worst


@dataclass
class ReportSection:
    """A labelled block of key/value metrics rendered into the markdown report."""

    title: str
    rows: list[tuple[str, Any]] = field(default_factory=list)

    def add(self, key: str, value: Any) -> "ReportSection":
        self.rows.append((key, value))
        return self


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def percentiles(samples: list[float]) -> dict[str, float]:
    """avg/p50/p95/p99/min/max in the same unit as the samples."""
    if not samples:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
    s = sorted(samples)
    n = len(s)
    return {
        "avg": round(statistics.mean(s), 3),
        "p50": round(s[int(0.50 * (n - 1))], 3),
        "p95": round(s[int(0.95 * (n - 1))], 3),
        "p99": round(s[int(0.99 * (n - 1))], 3),
        "min": round(s[0], 3),
        "max": round(s[-1], 3),
    }


def human_size(num_bytes: float | None) -> str:
    if num_bytes is None:
        return "n/a"
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


def file_size(path: str | Path | None) -> int | None:
    if path is None:
        return None
    p = Path(path)
    if p.is_file():
        return p.stat().st_size
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return None


def format_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Render a fixed-width plaintext table (for terminal output)."""
    cols = [str(h) for h in headers]
    str_rows = [[str(c) for c in row] for row in rows]
    widths = [len(c) for c in cols]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    line = "  ".join(c.ljust(widths[i]) for i, c in enumerate(cols))
    out = [line]
    for row in str_rows:
        out.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(out)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    head = "| " + " | ".join(str(h) for h in headers) + " |"
    sep = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


# --------------------------------------------------------------------------- #
# Report writer
# --------------------------------------------------------------------------- #
def write_report(
    name: str,
    *,
    status: Status | str,
    summary: dict[str, Any],
    title: str | None = None,
    sections: list[ReportSection] | None = None,
    tables: list[tuple[str, Sequence[str], Sequence[Sequence[Any]]]] | None = None,
    warnings: list[str] | None = None,
    failures: list[str] | None = None,
) -> dict[str, Any]:
    """Write ``reports/<name>.json`` + ``reports/<name>.md`` and return the JSON dict.

    ``summary`` is the full machine-readable payload. ``sections`` and ``tables``
    drive the human-readable markdown.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    status = Status(status) if not isinstance(status, Status) else status
    warnings = warnings or []
    failures = failures or []

    payload = {
        "report": name,
        "status": status.value,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "warnings": warnings,
        "failures": failures,
        **summary,
    }
    (REPORTS_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Markdown.
    md_title = title or name.replace("_", " ").title()
    lines = [f"# {md_title}", "", f"**Status: {status.value}**", ""]
    if failures:
        lines += ["## Failures", *[f"- {f}" for f in failures], ""]
    if warnings:
        lines += ["## Warnings", *[f"- {w}" for w in warnings], ""]
    for section in sections or []:
        lines += [f"## {section.title}", ""]
        if section.rows:
            lines.append(markdown_table(["Metric", "Value"], section.rows))
            lines.append("")
    for table_title, headers, rows in tables or []:
        lines += [f"## {table_title}", "", markdown_table(headers, rows), ""]
    (REPORTS_DIR / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def print_summary(title: str, status: Status | str, headers, rows) -> None:
    """Print a step summary table + status banner to the terminal."""
    status = Status(status) if not isinstance(status, Status) else status
    print(f"\n{'=' * 64}\n{title}  [{status.value}]\n{'=' * 64}")
    if rows:
        print(format_table(headers, rows))
    print("=" * 64)


def load_report(name: str) -> dict[str, Any] | None:
    path = REPORTS_DIR / f"{name}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None
    return None
