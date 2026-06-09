#!/usr/bin/env python3
"""Build + train a single strong FastText ROUTER head.

Instead of 3 fine-grained heads, one coarse router decides which BERT group to
wake up — or skips them entirely on confidently-safe traffic:

    labels: attack | moderation | safe   (attack & moderation can co-occur)
      attack score      -> run protectai (PI) + madhurjindal (JB)
      moderation score  -> run KoalaAI
      both low + no heuristic -> FAST-ALLOW (skip all BERTs)

FastText is good at this coarse decision (unlike fine-grained 12-label), so it
can be both strong and cheap. This script builds the data, trains, quantizes,
and prints a skip/coverage analysis: how much traffic can skip BERTs and at what
miss cost, with recommended thresholds.

Usage:
    python scripts/21_train_fasttext_router.py
    python scripts/21_train_fasttext_router.py --epoch 50 --dim 256
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from safety_classifier import constants as C
from safety_classifier.config import repo_root
from safety_classifier.reporting import ReportSection, Status, percentiles, print_summary, write_report

ATTACK_LABELS = {C.PROMPT_INJECTION, C.JAILBREAK}
MODERATION_LABELS = {
    C.TOXICITY, C.HATE, C.HARASSMENT, C.SEXUAL, C.VIOLENCE,
    C.SELF_HARM, C.DANGEROUS_INFORMATION, C.ILLEGAL_ACTIVITY,
}
PROCESSED = repo_root() / "data" / "processed"
OUT_DIR = repo_root() / "data" / "fasttext_router"
MODEL_DIR = repo_root() / "models" / "fasttext"
REPORTS_DIR = repo_root() / "reports"


def _route_labels(rec_labels) -> list[str]:
    s = set(rec_labels)
    out = []
    if s & ATTACK_LABELS:
        out.append("attack")
    if s & MODERATION_LABELS:
        out.append("moderation")
    if not out and C.SAFE in s:
        out.append("safe")
    return out


# civil_comments (real-world) float columns -> route. Any toxic col => moderation.
_CIVIL_COLS = ("toxicity", "identity_attack", "insult", "threat", "sexual_explicit",
               "severe_toxicity", "obscene")


def _civil_route(row, thr=0.5) -> list[str]:
    if any(row.get(c) is not None and float(row.get(c) or 0) >= thr for c in _CIVIL_COLS):
        return ["moderation"]
    return ["safe"]


def _load_civil(split: str, max_rows: int, seed: int):
    """Stream real-world civil_comments, return (route_labels, text) records."""
    from datasets import load_dataset
    hf_split = "train" if split == "train" else "test"
    ds = load_dataset("google/civil_comments", split=hf_split, streaming=True)
    mod, safe = [], []
    cap = max_rows if split == "train" else max(max_rows // 8, 1)
    for row in ds:
        text = (row.get("text") or "").strip()
        if not text or len(text) < 8 or len(text) > 2000:
            continue
        labs = _civil_route(row)
        (mod if labs == ["moderation"] else safe).append((labs, text.replace("\n", " ")))
        if len(mod) >= cap and len(safe) >= cap:
            break
    random.Random(seed).shuffle(safe)
    return mod + safe[:cap]


def build(per_route_cap: int, include_civil: bool, civil_max: int, seed: int) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    counts = {}
    for split in ("train", "val", "test"):
        inp = PROCESSED / f"all_{split}.jsonl"
        if not inp.exists():
            raise FileNotFoundError(f"missing {inp} (run 01 + 02 first)")
        rows = [json.loads(l) for l in inp.read_text(encoding="utf-8").splitlines() if l.strip()]

        attack, mod, safe = [], [], []
        for rec in rows:
            labs = _route_labels(rec.get("labels", []))
            if not labs:
                continue
            text = rec["text"].replace("\n", " ").strip()
            if "attack" in labs:
                attack.append((labs, text))
            elif "moderation" in labs:
                mod.append((labs, text))
            elif labs == ["safe"]:
                safe.append((labs, text))

        # Add real-world civil_comments to moderation + safe pools.
        if include_civil:
            civ = _load_civil(split, civil_max, seed)
            mod += [r for r in civ if r[0] == ["moderation"]]
            safe += [r for r in civ if r[0] == ["safe"]]
            print(f"  [{split}] +civil: {sum(1 for r in civ if r[0]==['moderation'])} mod, "
                  f"{sum(1 for r in civ if r[0]==['safe'])} safe")

        # Balance the routes. attack is the minority (~12k) vs moderation/safe.
        is_train = split == "train"
        cap = per_route_cap if is_train else max(per_route_cap // 8, 1)
        rng.shuffle(attack); rng.shuffle(mod); rng.shuffle(safe)
        mod_k, safe_k = mod[:cap], safe[:cap]
        if is_train:
            # Oversample attack (duplicate lines) up to ~1/3 of the larger routes
            # so it isn't drowned. TRAIN only — val/test stay natural for honest eval.
            target = min(cap, max(len(attack), (len(mod_k) + len(safe_k)) // 3))
            reps = (target // max(len(attack), 1)) + 1
            attack_k = (attack * reps)[:target]
        else:
            attack_k = attack
        kept = attack_k + mod_k + safe_k
        rng.shuffle(kept)

        c = {"attack": 0, "moderation": 0, "safe": 0}
        with (OUT_DIR / f"{split}.txt").open("w", encoding="utf-8") as f:
            for labs, text in kept:
                prefix = " ".join(f"__label__{l}" for l in labs)
                f.write(f"{prefix} {text}\n")
                for l in labs:
                    c[l] += 1
        counts[split] = c
        print(f"  {split}: {c}")
    return counts


def _predict_scores(model, text):
    # Use the NumPy-2-safe predict helper (model.predict() crashes under NumPy 2
    # with "Unable to avoid copy while creating an array").
    from safety_classifier.fasttext_layer.compat import predict_fasttext
    labels, probs = predict_fasttext(model, text.replace("\n", " "), k=-1)
    d = {"attack": 0.0, "moderation": 0.0, "safe": 0.0}
    for lab, p in zip(labels, probs):
        d[lab[len("__label__"):]] = float(p)
    return d


def _binary_metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _route_metrics(scored: list[tuple[dict[str, float], set[str]]], route: str, threshold: float) -> dict[str, Any]:
    tp = sum(1 for d, labs in scored if route in labs and d[route] >= threshold)
    fp = sum(1 for d, labs in scored if route not in labs and d[route] >= threshold)
    fn = sum(1 for d, labs in scored if route in labs and d[route] < threshold)
    return {
        "threshold": threshold,
        "support": tp + fn,
        "predicted": tp + fp,
        **_binary_metrics(tp, fp, fn),
    }


def _fast_allow_metrics(scored: list[tuple[dict[str, float], set[str]]], threshold: float) -> dict[str, Any]:
    n = len(scored)
    unsafe = [(d, labs) for d, labs in scored if "attack" in labs or "moderation" in labs]
    skipped = [
        (d, labs)
        for d, labs in scored
        if d["attack"] < threshold and d["moderation"] < threshold
    ]
    missed_attack = sum(1 for _d, labs in skipped if "attack" in labs)
    missed_moderation = sum(1 for _d, labs in skipped if "moderation" in labs)
    false_pass = sum(1 for _d, labs in skipped if "attack" in labs or "moderation" in labs)
    total_attack = sum(1 for _d, labs in scored if "attack" in labs)
    total_moderation = sum(1 for _d, labs in scored if "moderation" in labs)
    return {
        "threshold": threshold,
        "skipped": len(skipped),
        "skip_pct": round(len(skipped) / n, 4) if n else 0.0,
        "false_pass": false_pass,
        "unsafe_miss_rate": round(false_pass / len(unsafe), 4) if unsafe else 0.0,
        "missed_attack": missed_attack,
        "missed_attack_rate": round(missed_attack / total_attack, 4) if total_attack else 0.0,
        "missed_moderation": missed_moderation,
        "missed_moderation_rate": round(missed_moderation / total_moderation, 4) if total_moderation else 0.0,
        "bert_call_pct": round(1 - (len(skipped) / n), 4) if n else 0.0,
    }


def evaluate(model, test_path: Path, target_miss_rate: float = 0.005) -> dict[str, Any]:
    rows = []
    for line in test_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        toks = line.split()
        labs = [t[len("__label__"):] for t in toks if t.startswith("__label__")]
        text = " ".join(t for t in toks if not t.startswith("__label__"))
        rows.append((set(labs), text))

    latencies: list[float] = []
    sc = []
    for labs, text in rows:
        start = time.perf_counter()
        scores = _predict_scores(model, text)
        latencies.append((time.perf_counter() - start) * 1000)
        sc.append((scores, labs))
    n = len(sc)

    # Per-route recall/precision at the routing threshold (high recall).
    print("\n=== routing quality (per route) ===")
    print(f"{'route':<12} {'thr':>5} {'recall':>7} {'precision':>10} {'F1':>6}")
    route_thr: dict[str, float] = {}
    route_metrics: dict[str, Any] = {}
    for route in ("attack", "moderation"):
        best = (0.0, 0.0, 0.0, 0.0)
        for thr in [round(0.02 * i, 2) for i in range(1, 50)]:
            tp = sum(1 for d, labs in sc if route in labs and d[route] >= thr)
            fp = sum(1 for d, labs in sc if route not in labs and d[route] >= thr)
            fn = sum(1 for d, labs in sc if route in labs and d[route] < thr)
            rec = tp / (tp + fn) if tp + fn else 0
            prec = tp / (tp + fp) if tp + fp else 0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
            if f1 > best[3]:
                best = (thr, rec, prec, f1)
        # pick a HIGH-RECALL routing threshold (recall >= 0.97 if possible)
        hr_thr = next((round(0.02 * i, 2) for i in range(1, 50)
                       if (sum(1 for d, labs in sc if route in labs and d[route] >= round(0.02*i,2)) /
                           max(sum(1 for _, labs in sc if route in labs), 1)) < 0.97), 0.02)
        # step back one to keep >=0.97 recall
        route_thr[route] = max(0.02, hr_thr - 0.02)
        thr = route_thr[route]
        route_metrics[route] = {
            "best_f1_threshold": best[0],
            "best_f1_precision": round(best[2], 4),
            "best_f1_recall": round(best[1], 4),
            "best_f1": round(best[3], 4),
            "high_recall": _route_metrics(sc, route, thr),
        }
        rec = route_metrics[route]["high_recall"]["recall"]
        prec = route_metrics[route]["high_recall"]["precision"]
        f1 = route_metrics[route]["high_recall"]["f1"]
        print(f"{route:<12} {thr:>5} {rec:>7.3f} {prec:>10.3f} {f1:>6.3f}")

    # Skip/coverage analysis: FAST-ALLOW = both routes below threshold.
    print("\n=== FAST-ALLOW coverage (skip ALL BERTs when both routes low) ===")
    print(f"{'skip_thr':>9} {'%skipped':>9} {'missed_attack':>14} {'missed_mod':>11}")
    sweep = [_fast_allow_metrics(sc, round(0.01 * i, 2)) for i in range(1, 100)]
    for st in [0.02, 0.05, 0.1, 0.15, 0.2, 0.3]:
        m = _fast_allow_metrics(sc, st)
        print(
            f"{st:>9} {100*m['skip_pct']:>8.1f}% "
            f"{m['missed_attack']:>6} ({100*m['missed_attack_rate']:4.1f}%) "
            f"{m['missed_moderation']:>6} ({100*m['missed_moderation_rate']:4.1f}%)"
        )
    print("\nRead: pick the largest skip_thr where missed_attack% and missed_mod% are")
    print("acceptably low. That % of traffic skips the BERTs entirely (fast-allow).")
    viable = [m for m in sweep if m["unsafe_miss_rate"] <= target_miss_rate]
    recommended_fast_allow = max(viable, key=lambda m: (m["skip_pct"], m["threshold"])) if viable else sweep[0]
    return {
        "rows": n,
        "route_thresholds": route_thr,
        "route_metrics": route_metrics,
        "fast_allow_recommended": recommended_fast_allow,
        "fast_allow_sweep": sweep,
        "target_miss_rate": target_miss_rate,
        "latency_ms": percentiles(latencies),
    }


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def write_eval_report(result: dict[str, Any], train_counts: dict[str, Any] | None = None) -> None:
    rec = result["fast_allow_recommended"]
    route_rows = [
        [
            route,
            m["high_recall"]["threshold"],
            m["high_recall"]["support"],
            m["high_recall"]["predicted"],
            _pct(m["high_recall"]["precision"]),
            _pct(m["high_recall"]["recall"]),
            _pct(m["high_recall"]["f1"]),
            m["best_f1_threshold"],
            _pct(m["best_f1"]),
        ]
        for route, m in result["route_metrics"].items()
    ]
    sweep_rows = [
        [
            m["threshold"],
            _pct(m["skip_pct"]),
            _pct(m["unsafe_miss_rate"]),
            m["false_pass"],
            _pct(m["missed_attack_rate"]),
            _pct(m["missed_moderation_rate"]),
            _pct(m["bert_call_pct"]),
        ]
        for m in result["fast_allow_sweep"]
        if m["threshold"] in (0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
    ]
    section = ReportSection("Recommended Fast-Allow Gate")
    section.add("threshold", rec["threshold"])
    section.add("traffic_skipping_all_berts", _pct(rec["skip_pct"]))
    section.add("bert_call_rate", _pct(rec["bert_call_pct"]))
    section.add("unsafe_false_pass_rate", _pct(rec["unsafe_miss_rate"]))
    section.add("false_pass_count", rec["false_pass"])
    section.add("missed_attack_rate", _pct(rec["missed_attack_rate"]))
    section.add("missed_moderation_rate", _pct(rec["missed_moderation_rate"]))
    section.add("p95_latency_ms", result["latency_ms"]["p95"])
    write_report(
        "fasttext_router_eval",
        status=Status.PASS,
        title="FastText Attack/Moderation Router Evaluation",
        summary={**result, "train_counts": train_counts},
        sections=[section],
        tables=[
            (
                "Routing Metrics",
                [
                    "Route",
                    "HR Thr",
                    "Support",
                    "Predicted",
                    "Precision",
                    "Recall",
                    "F1",
                    "Best Thr",
                    "Best F1",
                ],
                route_rows,
            ),
            (
                "Fast-Allow Sweep",
                [
                    "Threshold",
                    "Skip All BERTs",
                    "Unsafe Miss",
                    "False Pass",
                    "Miss Attack",
                    "Miss Moderation",
                    "BERT Calls",
                ],
                sweep_rows,
            ),
        ],
    )
    print_summary(
        "FastText Attack/Moderation Router Evaluation",
        Status.PASS,
        ["Fast-Allow Thr", "Skip All BERTs", "Unsafe Miss", "BERT Calls", "p95 ms"],
        [[
            rec["threshold"],
            _pct(rec["skip_pct"]),
            _pct(rec["unsafe_miss_rate"]),
            _pct(rec["bert_call_pct"]),
            result["latency_ms"]["p95"],
        ]],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-route-cap", type=int, default=60000,
                    help="Cap moderation/safe rows per split (attack kept fully)")
    ap.add_argument("--include-civil", action="store_true", default=True,
                    help="Mix real-world civil_comments into moderation/safe")
    ap.add_argument("--no-civil", dest="include_civil", action="store_false")
    ap.add_argument("--civil-max", type=int, default=40000,
                    help="Max civil_comments rows per class per split")
    ap.add_argument("--profile", choices=["strong", "fast"], default="strong")
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--wordNgrams", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--bucket", type=int, default=None,
                    help="N-gram hash buckets. Bigger = fewer collisions (uses more "
                         "RAM): ~2M default; try 8M-20M with wordNgrams=3.")
    ap.add_argument("--thread", type=int, default=0,
                    help="Training threads (0 = all cores).")
    ap.add_argument("--quantize-retrain", action="store_true", default=False,
                    help="Retrain during quantization (slower, slightly smaller .ftz)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-train", action="store_true",
                    help="Load the already-trained router_head.ftz and just evaluate "
                         "(no retraining).")
    ap.add_argument("--target-miss-rate", type=float, default=0.005,
                    help="Max unsafe false-pass rate for recommended fast-allow threshold. 0.005 = 0.5%%.")
    args = ap.parse_args()
    defaults = {
        "fast": {"epoch": 25, "dim": 100, "wordNgrams": 2, "lr": 0.5, "bucket": 2_000_000},
        "strong": {"epoch": 50, "dim": 200, "wordNgrams": 4, "lr": 0.4, "bucket": 5_000_000},
    }[args.profile]
    epoch = args.epoch or defaults["epoch"]
    dim = args.dim or defaults["dim"]
    word_ngrams = args.wordNgrams or defaults["wordNgrams"]
    lr = args.lr or defaults["lr"]
    bucket = args.bucket or defaults["bucket"]

    import fasttext
    # NOTE: do NOT silence eprint here — we want the live "Progress: X%" bar so
    # training is visible. (The load-time deprecation warning is harmless.)

    if not args.skip_build:
        print("[router] building data (processed + real-world civil_comments) ...")
        train_counts = build(args.per_route_cap, args.include_civil, args.civil_max, args.seed)
    else:
        train_counts = None

    import os as _os
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bin_path = MODEL_DIR / "router_head.bin"
    ftz_path = MODEL_DIR / "router_head.ftz"

    if args.skip_train:
        # Load the already-trained model and just (re-)evaluate.
        load_path = ftz_path if ftz_path.exists() else bin_path
        print(f"[router] --skip-train: loading {load_path}")
        model = fasttext.load_model(str(load_path))
    else:
        print("\n[router] training ...")
        train_kwargs = dict(
            input=str(OUT_DIR / "train.txt"),
            loss="ova", epoch=epoch, dim=dim,
            wordNgrams=word_ngrams, lr=lr, minn=2, maxn=5,
            bucket=bucket,
        )
        if args.thread > 0:
            train_kwargs["thread"] = args.thread
        print(f"[router] profile={args.profile} bucket={bucket:,} dim={dim} epoch={epoch} "
              f"wordNgrams={word_ngrams} lr={lr} threads={args.thread or _os.cpu_count()}", flush=True)
        print("[router] FastText's % bar uses \\r and won't render in Colab — "
              "heartbeat below confirms it's alive.", flush=True)

        # Heartbeat: FastText's progress bar is invisible in notebooks, so print an
        # elapsed-time line every 30s from a daemon thread so it never looks hung.
        import threading
        import time as _time
        _stop = threading.Event()

        def _heartbeat():
            t0 = _time.time()
            while not _stop.wait(30):
                print(f"[router] still training... {int(_time.time() - t0)}s elapsed", flush=True)

        hb = threading.Thread(target=_heartbeat, daemon=True)
        hb.start()
        try:
            model = fasttext.train_supervised(**train_kwargs)
        finally:
            _stop.set()
        print("[router] training done; quantizing ...", flush=True)
        model.save_model(str(bin_path))
        model.quantize(input=str(OUT_DIR / "train.txt"), qnorm=True,
                       retrain=args.quantize_retrain, cutoff=100000)
        model.save_model(str(ftz_path))
        print(f"[router] saved {ftz_path} ({ftz_path.stat().st_size//1024} KB)")

    result = evaluate(model, OUT_DIR / "test.txt", target_miss_rate=args.target_miss_rate)
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "fasttext_router_thresholds.json").write_text(
        json.dumps(
            {
                "route": result["route_thresholds"],
                "fast_allow": result["fast_allow_recommended"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_eval_report(result, train_counts=train_counts)
    print("\n[router] recommended route thresholds:", result["route_thresholds"])
    print("[router] recommended fast-allow:", result["fast_allow_recommended"])


if __name__ == "__main__":
    main()
