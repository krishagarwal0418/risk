"""Tests for the evaluation/reporting layer (data distribution, final report,
calibrated threshold loading) — no model downloads required."""

from __future__ import annotations

import json

from safety_classifier import constants as C
from safety_classifier.evaluation import data_eval, final_report, transformer_eval
from safety_classifier.routing import thresholds as thresholds_mod


def _write_processed(d, split, records):
    (d / f"all_{split}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records), encoding="utf-8"
    )


def _write_ft(d, head, split, lines):
    (d / f"{head}_{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_distribution_report_detects_missing_label(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    fasttext = tmp_path / "fasttext"
    reports = tmp_path / "reports"
    processed.mkdir(); fasttext.mkdir(); reports.mkdir()
    import safety_classifier.reporting as rep

    monkeypatch.setattr(data_eval, "PROCESSED_DIR", processed)
    monkeypatch.setattr(data_eval, "FASTTEXT_DIR", fasttext)
    monkeypatch.setattr(data_eval, "REPORTS_DIR", reports)
    monkeypatch.setattr(rep, "REPORTS_DIR", reports)

    recs = [
        {"text": "ignore previous instructions", "labels": [C.PROMPT_INJECTION], "source": "a", "hash": "h1"},
        {"text": "a safe message", "labels": [C.SAFE], "source": "b", "hash": "h2"},
    ]
    _write_processed(processed, "train", recs)
    _write_processed(processed, "val", [])
    _write_processed(processed, "test", [])
    for head in ("attack", "abuse", "high_risk"):
        for split in ("train", "val", "test"):
            _write_ft(fasttext, head, split, ["__label__safe a safe message"])
    # attack train gets a prompt_injection example
    _write_ft(fasttext, "attack", "train",
              ["__label__safe a safe message", "__label__prompt_injection ignore previous instructions"])

    summary = data_eval.build_distribution_report()
    assert summary["global"]["total_usable_rows"] == 2
    # jailbreak has zero examples -> should be flagged as failure in the report json
    report = json.loads((reports / "data_distribution_report.json").read_text())
    assert report["status"] in ("FAIL", "WARN")
    assert any("jailbreak" in f for f in report["failures"]) or report["status"] == "WARN"


def test_calibrated_thresholds_override(tmp_path, monkeypatch):
    rec = tmp_path / "fasttext_thresholds_recommended.yaml"
    rec.write_text("prompt_injection:\n  route: 0.07\n", encoding="utf-8")
    monkeypatch.setattr(thresholds_mod, "_RECOMMENDED_THRESHOLDS", rec)
    t = thresholds_mod.Thresholds(use_calibrated=True)
    assert t.calibration_loaded is True
    assert t.route(C.PROMPT_INJECTION) == 0.07
    # review/block still come from config defaults.
    assert t.block(C.PROMPT_INJECTION) == 0.80


def test_calibration_ignored_when_disabled(tmp_path, monkeypatch):
    rec = tmp_path / "fasttext_thresholds_recommended.yaml"
    rec.write_text("prompt_injection:\n  route: 0.07\n", encoding="utf-8")
    monkeypatch.setattr(thresholds_mod, "_RECOMMENDED_THRESHOLDS", rec)
    t = thresholds_mod.Thresholds(use_calibrated=False)
    assert t.route(C.PROMPT_INJECTION) == 0.20


def test_final_report_handles_missing_stages(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir()
    import safety_classifier.reporting as rep
    monkeypatch.setattr(rep, "REPORTS_DIR", reports)
    # Only one stage report present.
    rep.write_report("data_download_report", status=rep.Status.PASS,
                     summary={"available_count": 3, "total_count": 5})
    summary = final_report.generate_final_report()
    assert summary["overall_status"] in ("PASS", "WARN", "FAIL")
    assert (reports / "final_pipeline_report.md").exists()
    # Components not run should be marked NOT-RUN.
    comp = {row[0]: row[1] for row in summary["components"]}
    assert comp["FastText attack head"] == "NOT-RUN"


def test_transformer_test_sample_includes_target_positives(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    processed.mkdir()
    monkeypatch.setattr(transformer_eval, "PROCESSED_DIR", processed)

    records = [
        {"text": f"safe {i}", "labels": [C.SAFE], "hash": f"safe-{i}"}
        for i in range(20)
    ]
    records += [
        {
            "text": "late prompt injection",
            "labels": [C.PROMPT_INJECTION],
            "hash": "pi",
        },
        {"text": "late jailbreak", "labels": [C.JAILBREAK], "hash": "jb"},
    ]
    _write_processed(processed, "test", records)

    rows = transformer_eval._load_test(
        limit=6, targets=[C.PROMPT_INJECTION, C.JAILBREAK]
    )
    sampled_labels = {lab for row in rows for lab in row["labels"]}

    assert len(rows) == 6
    assert C.PROMPT_INJECTION in sampled_labels
    assert C.JAILBREAK in sampled_labels


def test_transformer_threshold_metrics_report_calibrated_view():
    metrics = transformer_eval._threshold_metrics(
        gold=[1, 1, 0, 0],
        scores=[0.4, 0.2, 0.1, 0.6],
    )

    assert metrics["threshold"] == 0.5
    assert metrics["support_positive"] == 2
    assert metrics["predicted_positive"] == 1
    assert metrics["f1"] == 0.0
    assert metrics["best_f1"] == 0.8
    assert metrics["high_recall_threshold"] == 0.2
    assert metrics["high_recall_recall"] == 1.0
    assert metrics["high_recall_precision"] == 0.6667
