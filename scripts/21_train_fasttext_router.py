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


def build(safe_cap: int) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "val", "test"):
        inp = PROCESSED / f"all_{split}.jsonl"
        if not inp.exists():
            raise FileNotFoundError(f"missing {inp} (run 01 + 02 first)")
        rows = [json.loads(l) for l in inp.read_text(encoding="utf-8").splitlines() if l.strip()]
        cap = safe_cap if split == "train" else max(safe_cap // 8, 1)
        c = {"attack": 0, "moderation": 0, "safe": 0}
        safe_written = 0
        out_path = OUT_DIR / f"{split}.txt"
        with out_path.open("w", encoding="utf-8") as f:
            for rec in rows:
                labs = _route_labels(rec.get("labels", []))
                if not labs:
                    continue
                if labs == ["safe"]:
                    if safe_cap and safe_written >= cap:
                        continue
                    safe_written += 1
                text = rec["text"].replace("\n", " ").strip()
                prefix = " ".join(f"__label__{l}" for l in labs)
                f.write(f"{prefix} {text}\n")
                for l in labs:
                    c[l] += 1
        counts[split] = c
        print(f"  {split}: {c}")
    return counts


def _predict_scores(model, text):
    labels, probs = model.predict(text.replace("\n", " "), k=-1)
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
    ap.add_argument("--safe-cap", type=int, default=40000)
    ap.add_argument("--epoch", type=int, default=50)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--wordNgrams", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()

    import fasttext
    fasttext.FastText.eprint = lambda *a, **k: None

    if not args.skip_build:
        print("[router] building data ...")
        build(args.safe_cap)

    print("\n[router] training ...")
    model = fasttext.train_supervised(
        input=str(OUT_DIR / "train.txt"),
        loss="ova", epoch=args.epoch, dim=args.dim,
        wordNgrams=args.wordNgrams, lr=args.lr, minn=2, maxn=5,
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bin_path = MODEL_DIR / "router_head.bin"
    ftz_path = MODEL_DIR / "router_head.ftz"
    model.save_model(str(bin_path))
    model.quantize(input=str(OUT_DIR / "train.txt"), qnorm=True, retrain=True, cutoff=100000)
    model.save_model(str(ftz_path))
    print(f"[router] saved {ftz_path} ({ftz_path.stat().st_size//1024} KB)")

    route_thr = evaluate(model, OUT_DIR / "test.txt")
    (repo_root() / "reports").mkdir(exist_ok=True)
    (repo_root() / "reports" / "fasttext_router_thresholds.json").write_text(
        json.dumps({"route": route_thr}, indent=2), encoding="utf-8")
    print("\n[router] recommended route thresholds:", route_thr)


if __name__ == "__main__":
    main()
