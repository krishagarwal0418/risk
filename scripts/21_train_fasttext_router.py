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
from pathlib import Path

import _bootstrap  # noqa: F401

from safety_classifier import constants as C
from safety_classifier.config import repo_root

ATTACK_LABELS = {C.PROMPT_INJECTION, C.JAILBREAK}
MODERATION_LABELS = {
    C.TOXICITY, C.HATE, C.HARASSMENT, C.SEXUAL, C.VIOLENCE,
    C.SELF_HARM, C.DANGEROUS_INFORMATION, C.ILLEGAL_ACTIVITY,
}
PROCESSED = repo_root() / "data" / "processed"
OUT_DIR = repo_root() / "data" / "fasttext_router"
MODEL_DIR = repo_root() / "models" / "fasttext"


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


def evaluate(model, test_path: Path):
    rows = []
    for line in test_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        toks = line.split()
        labs = [t[len("__label__"):] for t in toks if t.startswith("__label__")]
        text = " ".join(t for t in toks if not t.startswith("__label__"))
        rows.append((set(labs), text))

    sc = [(_predict_scores(model, t), labs) for labs, t in rows]
    n = len(sc)

    # Per-route recall/precision at the routing threshold (high recall).
    print("\n=== routing quality (per route) ===")
    print(f"{'route':<12} {'thr':>5} {'recall':>7} {'precision':>10} {'F1':>6}")
    route_thr = {}
    for route in ("attack", "moderation"):
        best = (0, 0, 0, 0)
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
        tp = sum(1 for d, labs in sc if route in labs and d[route] >= thr)
        fp = sum(1 for d, labs in sc if route not in labs and d[route] >= thr)
        fn = sum(1 for d, labs in sc if route in labs and d[route] < thr)
        rec = tp / (tp + fn) if tp + fn else 0
        prec = tp / (tp + fp) if tp + fp else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        print(f"{route:<12} {thr:>5} {rec:>7.3f} {prec:>10.3f} {f1:>6.3f}")

    # Skip/coverage analysis: FAST-ALLOW = both routes below threshold.
    print("\n=== FAST-ALLOW coverage (skip ALL BERTs when both routes low) ===")
    print(f"{'skip_thr':>9} {'%skipped':>9} {'missed_attack':>14} {'missed_mod':>11}")
    total_attack = sum(1 for _, labs in sc if "attack" in labs)
    total_mod = sum(1 for _, labs in sc if "moderation" in labs)
    for st in [0.05, 0.1, 0.15, 0.2, 0.3]:
        skipped = [(d, labs) for d, labs in sc if d["attack"] < st and d["moderation"] < st]
        pct = 100 * len(skipped) / n
        miss_a = sum(1 for d, labs in skipped if "attack" in labs)
        miss_m = sum(1 for d, labs in skipped if "moderation" in labs)
        ra = 100 * miss_a / max(total_attack, 1)
        rm = 100 * miss_m / max(total_mod, 1)
        print(f"{st:>9} {pct:>8.1f}% {miss_a:>6} ({ra:4.1f}%) {miss_m:>6} ({rm:4.1f}%)")
    print("\nRead: pick the largest skip_thr where missed_attack% and missed_mod% are")
    print("acceptably low. That % of traffic skips the BERTs entirely (fast-allow).")
    return route_thr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-route-cap", type=int, default=60000,
                    help="Cap moderation/safe rows per split (attack kept fully)")
    ap.add_argument("--include-civil", action="store_true", default=True,
                    help="Mix real-world civil_comments into moderation/safe")
    ap.add_argument("--no-civil", dest="include_civil", action="store_false")
    ap.add_argument("--civil-max", type=int, default=40000,
                    help="Max civil_comments rows per class per split")
    # Light defaults: a coarse 3-way router doesn't need transformer-grade settings.
    # dim=100/epoch=25/wordNgrams=2 trains in minutes with ~the same routing recall.
    ap.add_argument("--epoch", type=int, default=25)
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--wordNgrams", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--bucket", type=int, default=2_000_000,
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
    args = ap.parse_args()

    import fasttext
    # NOTE: do NOT silence eprint here — we want the live "Progress: X%" bar so
    # training is visible. (The load-time deprecation warning is harmless.)

    if not args.skip_build:
        print("[router] building data (processed + real-world civil_comments) ...")
        build(args.per_route_cap, args.include_civil, args.civil_max, args.seed)

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
            loss="ova", epoch=args.epoch, dim=args.dim,
            wordNgrams=args.wordNgrams, lr=args.lr, minn=2, maxn=5,
            bucket=args.bucket,
        )
        if args.thread > 0:
            train_kwargs["thread"] = args.thread
        print(f"[router] bucket={args.bucket:,} dim={args.dim} epoch={args.epoch} "
              f"wordNgrams={args.wordNgrams} threads={args.thread or _os.cpu_count()}", flush=True)
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

    route_thr = evaluate(model, OUT_DIR / "test.txt")
    (repo_root() / "reports").mkdir(exist_ok=True)
    (repo_root() / "reports" / "fasttext_router_thresholds.json").write_text(
        json.dumps({"route": route_thr}, indent=2), encoding="utf-8")
    print("\n[router] recommended route thresholds:", route_thr)


if __name__ == "__main__":
    main()
