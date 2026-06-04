"""Tests for the reporting utilities and report generation."""

from __future__ import annotations

import json

from safety_classifier import reporting
from safety_classifier.reporting import (
    Status,
    file_size,
    format_table,
    human_size,
    markdown_table,
    percentiles,
)


def test_status_worst():
    assert Status.worst([Status.PASS, Status.WARN]) == Status.WARN
    assert Status.worst([Status.PASS, Status.PASS]) == Status.PASS
    assert Status.worst(["PASS", "WARN", "FAIL"]) == Status.FAIL


def test_percentiles_basic():
    p = percentiles([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert p["min"] == 1
    assert p["max"] == 10
    assert p["avg"] == 5.5
    assert p["p50"] >= 5


def test_percentiles_empty():
    p = percentiles([])
    assert p["avg"] == 0.0


def test_human_size():
    assert human_size(512) == "512.0B"
    assert human_size(1536).endswith("KB")
    assert human_size(None) == "n/a"


def test_format_and_markdown_table():
    headers = ["A", "B"]
    rows = [[1, 2], [3, 4]]
    txt = format_table(headers, rows)
    assert "A" in txt and "B" in txt
    md = markdown_table(headers, rows)
    assert md.startswith("| A | B |")
    assert "---" in md


def test_write_report(tmp_path, monkeypatch):
    monkeypatch.setattr(reporting, "REPORTS_DIR", tmp_path)
    payload = reporting.write_report(
        "unit_test_report",
        status=Status.WARN,
        title="Unit Test",
        summary={"value": 42},
        warnings=["something minor"],
    )
    assert payload["status"] == "WARN"
    assert payload["value"] == 42
    json_path = tmp_path / "unit_test_report.json"
    md_path = tmp_path / "unit_test_report.md"
    assert json_path.exists() and md_path.exists()
    loaded = json.loads(json_path.read_text())
    assert loaded["warnings"] == ["something minor"]
    assert "Unit Test" in md_path.read_text()


def test_load_report(tmp_path, monkeypatch):
    monkeypatch.setattr(reporting, "REPORTS_DIR", tmp_path)
    reporting.write_report("r1", status=Status.PASS, summary={"x": 1})
    assert reporting.load_report("r1")["x"] == 1
    assert reporting.load_report("missing") is None


def test_file_size(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("hello")
    assert file_size(f) == 5
    assert file_size(tmp_path / "nope") is None
