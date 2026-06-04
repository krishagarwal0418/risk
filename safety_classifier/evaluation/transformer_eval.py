"""Baseline evaluation of pretrained transformer models (before ONNX/INT8).

Evaluates each model on the processed test set, restricted to the labels the model
is responsible for, and writes ``reports/transformer_baseline_eval.{json,md}``.

This is GPU/compute-heavy and intended to run in Colab. Each model is wrapped in
try/except so one failure does not abort the whole report.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from .. import constants as C
from ..config import load_models_config, repo_root
from ..reporting import ReportSection, Status, percentiles, print_summary, write_report

PROCESSED_DIR = repo_root() / "data" / "processed"

_MODERATION_SUPPORTED_LABELS = [
    C.TOXICITY,
    C.HATE,
    C.HARASSMENT,
    C.SEXUAL,
    C.VIOLENCE,
    C.SELF_HARM,
]

# model_key -> (wrapper import path, cfg key, target canonical labels)
MODEL_TARGETS: dict[str, tuple[str, str, list[str]]] = {
    "prompt_injection": ("PromptInjectionClassifier", "prompt_injection", [C.PROMPT_INJECTION]),
    "jailbreak": ("JailbreakClassifier", "jailbreak", [C.JAILBREAK]),
    "moderation_primary": (
        "ModerationClassifier", "moderation_primary", _MODERATION_SUPPORTED_LABELS
    ),
    "moderation_fallback": (
        "ModerationClassifier", "moderation_fallback", _MODERATION_SUPPORTED_LABELS
    ),
    "toxic_fallback": ("ToxicFallbackClassifier", "toxic_fallback", [C.TOXICITY, C.HATE]),
}


_THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 19)]


def _load_test(limit: int | None = None, targets: list[str] | None = None) -> list[dict]:
    path = PROCESSED_DIR / "all_test.jsonl"
    if not path.exists():
        return []
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit is None or len(rows) <= limit:
        return rows
    if not targets:
        return rows[:limit]

    # Deterministic stratified sample: include positives for each evaluated
    # target label first, then fill with hash-sorted background rows.
    selected: dict[str, dict] = {}
    per_label_budget = max(1, limit // (len(targets) + 1))
    for label in targets:
        positives = [r for r in rows if label in set(r.get("labels", []))]
        positives.sort(key=lambda r: r.get("hash") or _stable_key(r["text"]))
        for row in positives[:per_label_budget]:
            selected[row.get("hash") or _stable_key(row["text"])] = row
    remaining = [r for r in rows if (r.get("hash") or _stable_key(r["text"])) not in selected]
    remaining.sort(key=lambda r: r.get("hash") or _stable_key(r["text"]))
    for row in remaining:
        if len(selected) >= limit:
            break
        selected[row.get("hash") or _stable_key(row["text"])] = row
    return list(selected.values())[:limit]


def _stable_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _binary_metrics(gold: list[int], scores: list[float], thr: float = 0.5) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for g, s in zip(gold, scores):
        pred = 1 if s >= thr else 0
        if pred and g:
            tp += 1
        elif pred and not g:
            fp += 1
        elif not pred and g:
            fn += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / max(len(gold), 1)
    out = {
        "threshold": thr,
        "support_positive": tp + fn,
        "predicted_positive": tp + fp,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "accuracy": round(acc, 4),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }
    return out


def _threshold_metrics(gold: list[int], scores: list[float]) -> dict[str, Any]:
    by_threshold = {thr: _binary_metrics(gold, scores, thr=thr) for thr in _THRESHOLDS}
    best_thr = max(_THRESHOLDS, key=lambda thr: by_threshold[thr]["f1"])
    high_recall_candidates = [
        thr for thr in _THRESHOLDS if by_threshold[thr]["recall"] >= 0.90
    ]
    high_recall_thr = max(high_recall_candidates) if high_recall_candidates else best_thr
    default = _binary_metrics(gold, scores, thr=0.5)
    best = by_threshold[best_thr]
    high_recall = by_threshold[high_recall_thr]
    out = {
        **default,
        "best_f1_threshold": best_thr,
        "best_f1": best["f1"],
        "best_f1_precision": best["precision"],
        "best_f1_recall": best["recall"],
        "high_recall_threshold": high_recall_thr,
        "high_recall_precision": high_recall["precision"],
        "high_recall_recall": high_recall["recall"],
        "high_recall_predicted_positive": high_recall["predicted_positive"],
    }
    # ROC-AUC / PR-AUC if both classes present and sklearn available.
    if 0 < sum(gold) < len(gold):
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score

            out["roc_auc"] = round(float(roc_auc_score(gold, scores)), 4)
            out["pr_auc"] = round(float(average_precision_score(gold, scores)), 4)
        except Exception:  # noqa: BLE001
            pass
    return out


def _gpu_memory_mb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
    except Exception:  # noqa: BLE001
        pass
    return None


def _evaluate_one(
    model_key: str, device: str, limit: int | None, batch_size: int = 64
) -> dict[str, Any]:
    from .. import transformers_layer as TL

    cls_name, cfg_key, targets = MODEL_TARGETS[model_key]
    cls = getattr(TL, cls_name)
    cfg = load_models_config().get("transformers", {}).get(cfg_key)
    if not cfg:
        return {"model_key": model_key, "status": "error", "error": "no config"}

    rows = _load_test(limit, targets=targets)
    if not rows:
        return {"model_key": model_key, "status": "skipped", "error": "no test data"}

    model = cls(cfg["hf_name"], backend="pytorch", device=device, name=cfg["hf_name"])

    # Per-target gold + score collection.
    gold_by_label: dict[str, list[int]] = {t: [] for t in targets}
    score_by_label: dict[str, list[float]] = {t: [] for t in targets}
    latencies: list[float] = []

    # Batched inference so the GPU is actually utilised (one padded forward pass
    # per batch instead of one per example).
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        texts = [r["text"] for r in chunk]
        start = time.perf_counter()
        results = model.classify_batch(texts)
        per_example_ms = (time.perf_counter() - start) * 1000 / max(len(texts), 1)
        for row, result in zip(chunk, results):
            labels = set(row.get("labels", []))
            scores = result["scores"]
            latencies.append(per_example_ms)
            for t in targets:
                gold_by_label[t].append(1 if t in labels else 0)
                score_by_label[t].append(scores.get(t, 0.0))

    per_label = {
        t: _threshold_metrics(gold_by_label[t], score_by_label[t])
        for t in targets
        if sum(gold_by_label[t]) > 0  # only labels actually present in the test set
    }
    macro_f1 = (
        round(sum(m["f1"] for m in per_label.values()) / len(per_label), 4)
        if per_label else None
    )
    macro_best_f1 = (
        round(sum(m["best_f1"] for m in per_label.values()) / len(per_label), 4)
        if per_label else None
    )
    pr_auc_values = [m["pr_auc"] for m in per_label.values() if "pr_auc" in m]
    macro_pr_auc = (
        round(sum(pr_auc_values) / len(pr_auc_values), 4)
        if pr_auc_values else None
    )
    return {
        "model_key": model_key,
        "hf_name": cfg["hf_name"],
        "status": "ok",
        "test_examples": len(rows),
        "labels_evaluated": list(per_label.keys()),
        "macro_f1": macro_f1,
        "macro_best_f1": macro_best_f1,
        "macro_pr_auc": macro_pr_auc,
        "per_label": per_label,
        "latency_ms": percentiles(latencies),
        "gpu_memory_mb": _gpu_memory_mb(),
    }


def run_baseline_eval(
    device: str = "cuda",
    include_toxic_fallback: bool = False,
    limit: int | None = 2000,
    batch_size: int = 64,
) -> dict[str, Any]:
    keys = ["prompt_injection", "jailbreak", "moderation_primary", "moderation_fallback"]
    if include_toxic_fallback:
        keys.append("toxic_fallback")

    results: dict[str, Any] = {}
    warnings: list[str] = []
    for key in keys:
        try:
            results[key] = _evaluate_one(key, device, limit, batch_size=batch_size)
        except Exception as exc:  # noqa: BLE001
            results[key] = {"model_key": key, "status": "error", "error": str(exc)}
        if results[key].get("status") != "ok":
            warnings.append(f"{key}: {results[key].get('error', results[key].get('status'))}")
        else:
            macro = results[key].get("macro_best_f1")
            if macro is not None and macro < 0.5:
                warnings.append(f"{key}: low best-threshold macro F1 ({macro})")

    ok = [r for r in results.values() if r.get("status") == "ok"]
    status = Status.FAIL if not ok else (Status.WARN if warnings else Status.PASS)

    table_rows = []
    for key, r in results.items():
        table_rows.append([
            r.get("hf_name", key),
            r.get("status", "?").upper(),
            r.get("test_examples", 0),
            r.get("macro_f1", "n/a"),
            r.get("macro_best_f1", "n/a"),
            r.get("macro_pr_auc", "n/a"),
            r.get("latency_ms", {}).get("p95", "n/a") if r.get("status") == "ok" else "n/a",
        ])

    write_report(
        "transformer_baseline_eval",
        status=status,
        title="Transformer Baseline Evaluation",
        summary={"models": results},
        tables=[(
            "Baseline Metrics",
            ["Model", "Status", "Test ex.", "F1@0.5", "Best F1", "PR-AUC", "p95 ms"],
            table_rows,
        )],
        warnings=warnings,
    )
    print_summary(
        "Transformer Baseline", status,
        ["Model", "Status", "Test ex.", "F1@0.5", "Best F1", "PR-AUC", "p95 ms"], table_rows,
    )
    return {"models": results, "status": status.value}
