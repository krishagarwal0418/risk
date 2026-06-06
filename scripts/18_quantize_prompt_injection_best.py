#!/usr/bin/env python3
"""Export and quantize the best prompt-injection fine-tune with score checks.

This script is specifically for the DeBERTa weights saved as
``models/fine/prompt/model_best.safetensors``. It does not use the stale
``model.safetensors`` in that folder.

Pipeline:
  1. Build a clean Hugging Face folder from the base ProtectAI config/tokenizer
     plus ``model_best.safetensors``.
  2. Export FP32 ONNX.
  3. Create multiple dynamic INT8 ONNX candidates.
  4. Evaluate PyTorch, FP32 ONNX, and INT8 candidates on held-out data.
  5. Recommend the smallest candidate whose PR-AUC/F1 drop is within tolerance.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from safety_classifier.config import repo_root
from safety_classifier.reporting import Status, file_size, human_size, percentiles, print_summary, write_report

DEFAULT_BASE = "protectai/deberta-v3-base-prompt-injection-v2"
DEFAULT_WEIGHTS = "models/fine/prompt/model_best.safetensors"
DEFAULT_DATA = "data/prompt_injection_best/test.jsonl"

THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]

WEIGHT_SEARCH_ROOTS = (
    "models/fine/prompt",
    "models/finetuned/prompt_injection_best",
    "models/finetuned/prompt_injection",
    "models",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_tokenizer(source: str | Path):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(str(source), fix_mistral_regex=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(str(source))


def _ort_provider(requested: str) -> str | None:
    if requested == "cpu":
        return "CPUExecutionProvider"
    if requested == "cuda":
        return "CUDAExecutionProvider"
    if requested != "auto":
        return requested
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            return "CUDAExecutionProvider"
        if "CPUExecutionProvider" in providers:
            return "CPUExecutionProvider"
    except Exception:  # noqa: BLE001
        return None
    return None


def _sample_rows(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if not limit or len(rows) <= limit:
        return rows
    pos = [r for r in rows if int(r["label"]) == 1]
    neg = [r for r in rows if int(r["label"]) == 0]
    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    half = limit // 2
    out = pos[:half] + neg[: limit - half]
    rng.shuffle(out)
    return out


def _is_deberta_prompt_weights(path: Path) -> bool:
    if not path.exists() or path.suffix != ".safetensors":
        return False
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as fh:
            keys = set(fh.keys())
            if "deberta.embeddings.word_embeddings.weight" not in keys:
                return False
            if "classifier.weight" not in keys:
                return False
            return tuple(fh.get_tensor("classifier.weight").shape) == (2, 768)
    except Exception:  # noqa: BLE001
        return False


def discover_weight_candidates(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for rel in WEIGHT_SEARCH_ROOTS:
        base = root / rel
        if not base.exists():
            continue
        for path in base.rglob("*.safetensors"):
            if _is_deberta_prompt_weights(path):
                candidates.append(path)
    # Prefer explicit "best" files, then larger/newer paths.
    candidates = sorted(
        set(candidates),
        key=lambda p: (
            "best" not in p.name.lower(),
            -(p.stat().st_size if p.exists() else 0),
            str(p),
        ),
    )
    return candidates


def resolve_weights_path(root: Path, requested: str) -> Path:
    if requested == "auto":
        candidates = discover_weight_candidates(root)
        if candidates:
            print("[quant] auto-selected weights:", candidates[0])
            if len(candidates) > 1:
                print("[quant] other DeBERTa weight candidates:")
                for cand in candidates[1:8]:
                    print(f"  - {cand} ({human_size(cand.stat().st_size)})")
            return candidates[0]
        raise FileNotFoundError(
            "Could not auto-discover DeBERTa prompt weights. Expected a "
            "model_best.safetensors-like file under models/. Pass --weights "
            "/path/to/model_best.safetensors."
        )

    path = Path(requested)
    if not path.is_absolute():
        path = root / path
    if path.exists():
        return path

    candidates = discover_weight_candidates(root)
    msg = [f"Weights file not found: {path}"]
    if candidates:
        msg.append("Detected DeBERTa prompt weight candidates:")
        msg.extend(f"  - {cand} ({human_size(cand.stat().st_size)})" for cand in candidates[:10])
        msg.append("Re-run with --weights auto or one of the paths above.")
    else:
        msg.append("No DeBERTa-shaped *.safetensors candidates found under models/.")
    raise FileNotFoundError("\n".join(msg))


def metrics_at(y: list[int], scores: list[float], threshold: float) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for gold, score in zip(y, scores):
        pred = score >= threshold
        if pred and gold:
            tp += 1
        elif pred and not gold:
            fp += 1
        elif not pred and gold:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "predicted_positive": tp + fp,
    }


def summarize_scores(y: list[int], scores: list[float]) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    best_thr = max(THRESHOLDS, key=lambda thr: metrics_at(y, scores, thr)["f1"])
    high_recall = [thr for thr in THRESHOLDS if metrics_at(y, scores, thr)["recall"] >= 0.95]
    high_recall_thr = max(high_recall) if high_recall else None
    return {
        "pr_auc": round(float(average_precision_score(y, scores)), 6),
        "roc_auc": round(float(roc_auc_score(y, scores)), 6),
        "at_0_5": metrics_at(y, scores, 0.5),
        "best_f1": metrics_at(y, scores, best_thr),
        "high_recall_0_95": metrics_at(y, scores, high_recall_thr) if high_recall_thr else None,
    }


def build_clean_hf_dir(base_model: str, weights: Path, out_dir: Path) -> dict[str, Any]:
    from safetensors import safe_open
    from transformers import AutoConfig

    out_dir.mkdir(parents=True, exist_ok=True)
    with safe_open(str(weights), framework="pt", device="cpu") as fh:
        keys = set(fh.keys())
        if "deberta.embeddings.word_embeddings.weight" not in keys:
            raise RuntimeError(f"{weights} is not the expected DeBERTa model_best file")
        if tuple(fh.get_tensor("classifier.weight").shape) != (2, 768):
            raise RuntimeError("expected 2-label classifier.weight with shape (2, 768)")

    cfg = AutoConfig.from_pretrained(base_model)
    cfg.num_labels = 2
    cfg.problem_type = "single_label_classification"
    cfg.id2label = {0: "SAFE", 1: "INJECTION"}
    cfg.label2id = {"SAFE": 0, "INJECTION": 1}
    cfg.save_pretrained(out_dir)
    _load_tokenizer(base_model).save_pretrained(out_dir)
    shutil.copy2(weights, out_dir / "model.safetensors")
    return {"path": str(out_dir), "base_model": base_model, "weights": str(weights)}


def export_fp32_onnx(hf_dir: Path, out_dir: Path) -> dict[str, Any]:
    from optimum.onnxruntime import ORTModelForSequenceClassification

    out_dir.mkdir(parents=True, exist_ok=True)
    model = ORTModelForSequenceClassification.from_pretrained(str(hf_dir), export=True)
    model.save_pretrained(str(out_dir))
    _load_tokenizer(hf_dir).save_pretrained(str(out_dir))
    return {"path": str(out_dir), "size_bytes": file_size(out_dir)}


def _copy_aux_files(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in src_dir.iterdir():
        if src.suffix == ".onnx":
            continue
        try:
            shutil.copy2(src, dst_dir / src.name)
        except IsADirectoryError:
            if (dst_dir / src.name).exists():
                shutil.rmtree(dst_dir / src.name)
            shutil.copytree(src, dst_dir / src.name)


def quantize_candidate(fp32_dir: Path, out_dir: Path, *, weight_type: str, per_channel: bool) -> dict[str, Any]:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    onnx_files = list(fp32_dir.glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"no .onnx file in {fp32_dir}")
    _copy_aux_files(fp32_dir, out_dir)
    qtype = QuantType.QInt8 if weight_type == "qint8" else QuantType.QUInt8
    for src in onnx_files:
        quantize_dynamic(
            model_input=str(src),
            model_output=str(out_dir / src.name),
            weight_type=qtype,
            per_channel=per_channel,
        )
    return {
        "path": str(out_dir),
        "weight_type": weight_type,
        "per_channel": per_channel,
        "size_bytes": file_size(out_dir),
    }


def eval_pytorch(
    hf_dir: Path,
    rows: list[dict[str, Any]],
    batch_size: int,
    device: str,
    progress_label: str,
    progress_every: int,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification

    tokenizer = _load_tokenizer(hf_dir)
    model = AutoModelForSequenceClassification.from_pretrained(str(hf_dir))
    if device == "cuda":
        model = model.to("cuda")
    model.eval()
    scores: list[float] = []
    latencies: list[float] = []
    with torch.inference_mode():
        for i in range(0, len(rows), batch_size):
            chunk = rows[i : i + batch_size]
            enc = tokenizer(
                [r["text"] for r in chunk],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            if device == "cuda":
                enc = {k: v.to("cuda") for k, v in enc.items()}
            start = time.perf_counter()
            logits = model(**enc).logits.float()
            if device == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000 / max(len(chunk), 1))
            probs = torch.softmax(logits.detach().cpu(), dim=-1)[:, 1]
            scores.extend(float(x) for x in probs)
            done = min(i + len(chunk), len(rows))
            if progress_every and (done % progress_every == 0 or done == len(rows)):
                print(f"[quant] {progress_label}: scored {done}/{len(rows)}", flush=True)
    y = [int(r["label"]) for r in rows]
    return {
        "scores": scores,
        "metrics": summarize_scores(y, scores),
        "latency_ms": percentiles(latencies),
        "size_bytes": file_size(hf_dir),
        "device": device,
    }


def eval_ort(
    model_dir: Path,
    rows: list[dict[str, Any]],
    batch_size: int,
    provider: str | None,
    progress_label: str,
    progress_every: int,
) -> dict[str, Any]:
    import numpy as np
    from optimum.onnxruntime import ORTModelForSequenceClassification

    tokenizer = _load_tokenizer(model_dir)
    if provider:
        print(f"[quant] {progress_label}: ORT provider={provider}")
        model = ORTModelForSequenceClassification.from_pretrained(str(model_dir), provider=provider)
    else:
        model = ORTModelForSequenceClassification.from_pretrained(str(model_dir))
    scores: list[float] = []
    latencies: list[float] = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        enc = tokenizer(
            [r["text"] for r in chunk],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        start = time.perf_counter()
        logits = model(**enc).logits.detach().numpy()
        latencies.append((time.perf_counter() - start) * 1000 / max(len(chunk), 1))
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        scores.extend(float(x) for x in probs[:, 1])
        done = min(i + len(chunk), len(rows))
        if progress_every and (done % progress_every == 0 or done == len(rows)):
            print(f"[quant] {progress_label}: scored {done}/{len(rows)}", flush=True)
    y = [int(r["label"]) for r in rows]
    return {
        "scores": scores,
        "metrics": summarize_scores(y, scores),
        "latency_ms": percentiles(latencies),
        "size_bytes": file_size(model_dir),
    }


def compare_to_reference(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    cand_scores = candidate["scores"]
    ref_scores = reference["scores"]
    cand_metrics = candidate["metrics"]
    ref_metrics = reference["metrics"]
    return {
        "pr_auc_drop": round(ref_metrics["pr_auc"] - cand_metrics["pr_auc"], 6),
        "best_f1_drop": round(ref_metrics["best_f1"]["f1"] - cand_metrics["best_f1"]["f1"], 6),
        "f1_at_0_5_drop": round(ref_metrics["at_0_5"]["f1"] - cand_metrics["at_0_5"]["f1"], 6),
        "max_abs_prob_diff": round(float(np.max(np.abs(np.array(ref_scores) - np.array(cand_scores)))), 6),
        "avg_abs_prob_diff": round(float(np.mean(np.abs(np.array(ref_scores) - np.array(cand_scores)))), 6),
    }


def choose_recommendation(results: dict[str, Any], max_f1_drop: float, max_pr_auc_drop: float) -> str:
    candidates = []
    for name, result in results["candidates"].items():
        cmp = result["comparison_to_pytorch"]
        if cmp["best_f1_drop"] <= max_f1_drop and cmp["pr_auc_drop"] <= max_pr_auc_drop:
            candidates.append((result["size_bytes"] or 10**18, name))
    if candidates:
        return sorted(candidates)[0][1]
    return "fp32_onnx"


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize prompt-injection model_best with eval gates")
    parser.add_argument("--base-model", default=DEFAULT_BASE)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--hf-dir", default="models/quant/prompt_injection_best/hf")
    parser.add_argument("--fp32-dir", default="models/quant/prompt_injection_best/onnx_fp32")
    parser.add_argument("--int8-root", default="models/quant/prompt_injection_best")
    parser.add_argument("--limit", type=int, default=0,
                        help="Evaluation rows; 0 = full test set")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-f1-drop", type=float, default=0.002)
    parser.add_argument("--max-pr-auc-drop", type=float, default=0.002)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-pytorch", action="store_true",
                        help="Use FP32 ONNX as reference instead of PyTorch")
    parser.add_argument("--ort-provider", default="auto",
                        help="ONNXRuntime provider: auto, cuda, cpu, or explicit provider name")
    parser.add_argument("--progress-every", type=int, default=1000,
                        help="Print eval progress every N scored rows; 0 disables")
    parser.add_argument("--candidates", default="all",
                        help="Comma-separated candidate names, or all")
    args = parser.parse_args()

    root = repo_root()
    weights = resolve_weights_path(root, args.weights)
    data_path = root / args.data
    hf_dir = root / args.hf_dir
    fp32_dir = root / args.fp32_dir
    int8_root = root / args.int8_root
    ort_provider = _ort_provider(args.ort_provider)

    rows = _sample_rows(_read_jsonl(data_path), args.limit, args.seed)
    y_counts = Counter(int(r["label"]) for r in rows)
    print(f"[quant] eval rows={len(rows)} labels={dict(y_counts)}")

    hf_info = build_clean_hf_dir(args.base_model, weights, hf_dir)
    if not args.skip_export:
        print(f"[quant] exporting FP32 ONNX -> {fp32_dir}")
        fp32_info = export_fp32_onnx(hf_dir, fp32_dir)
    else:
        fp32_info = {"path": str(fp32_dir), "size_bytes": file_size(fp32_dir)}

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.skip_pytorch:
        print("[quant] evaluating FP32 ONNX reference")
        reference = eval_ort(
            fp32_dir, rows, args.batch_size, ort_provider,
            "fp32_onnx_reference", args.progress_every,
        )
        reference_name = "fp32_onnx"
    else:
        print(f"[quant] evaluating PyTorch reference on {device}")
        reference = eval_pytorch(
            hf_dir, rows, args.batch_size, device,
            "pytorch_reference", args.progress_every,
        )
        reference_name = "pytorch"

    print("[quant] evaluating FP32 ONNX")
    fp32_eval = eval_ort(
        fp32_dir, rows, args.batch_size, ort_provider,
        "fp32_onnx", args.progress_every,
    )
    fp32_cmp = compare_to_reference(fp32_eval, reference)

    candidate_specs = {
        "int8_qint8": {"weight_type": "qint8", "per_channel": False},
        "int8_qint8_per_channel": {"weight_type": "qint8", "per_channel": True},
        "int8_quint8": {"weight_type": "quint8", "per_channel": False},
        "int8_quint8_per_channel": {"weight_type": "quint8", "per_channel": True},
    }
    if args.candidates != "all":
        requested = {name.strip() for name in args.candidates.split(",") if name.strip()}
        unknown = sorted(requested - set(candidate_specs))
        if unknown:
            raise SystemExit(f"Unknown candidates: {unknown}; valid={sorted(candidate_specs)}")
        candidate_specs = {
            name: spec for name, spec in candidate_specs.items()
            if name in requested
        }
    candidates: dict[str, Any] = {}
    for name, spec in candidate_specs.items():
        out_dir = int8_root / name
        print(f"[quant] quantizing {name} -> {out_dir}")
        q_info = quantize_candidate(fp32_dir, out_dir, **spec)
        print(f"[quant] evaluating {name}")
        q_eval = eval_ort(
            out_dir, rows, args.batch_size, ort_provider,
            name, args.progress_every,
        )
        candidates[name] = {
            **q_info,
            "metrics": q_eval["metrics"],
            "latency_ms": q_eval["latency_ms"],
            "comparison_to_pytorch": compare_to_reference(q_eval, reference),
        }

    results = {
        "model": "prompt_injection_best",
        "hf_model_dir": hf_info,
        "data": str(data_path),
        "examples": len(rows),
        "label_counts": dict(y_counts),
        "reference": reference_name,
        "reference_metrics": reference["metrics"],
        "reference_latency_ms": reference["latency_ms"],
        "reference_size_bytes": reference["size_bytes"],
        "fp32_onnx": {
            **fp32_info,
            "metrics": fp32_eval["metrics"],
            "latency_ms": fp32_eval["latency_ms"],
            "comparison_to_pytorch": fp32_cmp,
        },
        "candidates": candidates,
    }
    recommendation = choose_recommendation(results, args.max_f1_drop, args.max_pr_auc_drop)
    results["recommendation"] = recommendation

    table_rows = [
        [
            "fp32_onnx",
            human_size(results["fp32_onnx"]["size_bytes"]),
            results["fp32_onnx"]["metrics"]["pr_auc"],
            results["fp32_onnx"]["metrics"]["best_f1"]["f1"],
            results["fp32_onnx"]["comparison_to_pytorch"]["pr_auc_drop"],
            results["fp32_onnx"]["comparison_to_pytorch"]["best_f1_drop"],
            results["fp32_onnx"]["latency_ms"]["p95"],
        ]
    ]
    for name, result in candidates.items():
        table_rows.append([
            name,
            human_size(result["size_bytes"]),
            result["metrics"]["pr_auc"],
            result["metrics"]["best_f1"]["f1"],
            result["comparison_to_pytorch"]["pr_auc_drop"],
            result["comparison_to_pytorch"]["best_f1_drop"],
            result["latency_ms"]["p95"],
        ])

    status = Status.PASS if recommendation != "fp32_onnx" else Status.WARN
    warnings = [] if status == Status.PASS else [
        "No INT8 candidate met score-drop tolerance; use FP32 ONNX for no-loss deployment."
    ]
    write_report(
        "prompt_injection_best_quantization",
        status=status,
        title="Prompt Injection Best Quantization",
        summary=results,
        warnings=warnings,
        tables=[(
            "Candidates",
            ["Candidate", "Size", "PR-AUC", "Best F1", "PR-AUC Drop", "Best F1 Drop", "p95 ms"],
            table_rows,
        )],
    )
    print_summary(
        "Prompt Injection Best Quantization",
        status,
        ["Candidate", "Size", "PR-AUC", "Best F1", "PR-AUC Drop", "Best F1 Drop", "p95 ms"],
        table_rows,
    )
    print(f"[quant] recommendation: {recommendation}")
    print("[quant] report written to reports/prompt_injection_best_quantization.{json,md}")


if __name__ == "__main__":
    main()
