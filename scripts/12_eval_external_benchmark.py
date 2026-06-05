#!/usr/bin/env python3
"""Evaluate a FastText attack head against an external prompt-injection benchmark.

Loads a HuggingFace dataset, auto-detects its text + label columns, scores every
example with the attack head, and reports how well the head separates injections
from benign prompts (PR-AUC, ROC-AUC, best-F1 threshold, recall at a chosen
operating point), plus a few false positives / negatives to eyeball.

Usage:
    python scripts/12_eval_external_benchmark.py \
        --model models/fasttext/attack_head_2.ftz \
        --dataset rogue-security/prompt-injections-benchmark
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# ---- label detection -------------------------------------------------------- #
_TEXT_KEYS = ("text", "prompt", "input", "content", "sentence", "message",
              "instruction", "query", "user_input", "jailbreak_query", "goal")
_LABEL_KEYS = ("label", "labels", "is_injection", "injection", "class", "target",
               "is_malicious", "malicious", "type", "category", "ground_truth")

_POS_TOKENS = ("inject", "malicious", "attack", "jailbreak", "unsafe", "harmful",
               "prompt_injection", "true", "1", "yes", "positive")
_NEG_TOKENS = ("benign", "safe", "legit", "clean", "false", "0", "no", "negative",
               "normal", "harmless")


def _detect_key(columns, candidates):
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _to_binary(value) -> int | None:
    """Map a raw label value to 1 (injection) / 0 (benign) / None (unknown)."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if value >= 0.5 else 0
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in ("", "none", "nan"):
        return None
    if any(tok == s for tok in ("1", "true", "yes")):
        return 1
    if any(tok == s for tok in ("0", "false", "no")):
        return 0
    if any(tok in s for tok in _POS_TOKENS):
        return 1
    if any(tok in s for tok in _NEG_TOKENS):
        return 0
    return None


# ---- fasttext scoring ------------------------------------------------------- #
def _load_model(path: str):
    import fasttext
    fasttext.FastText.eprint = lambda *a, **k: None  # silence load warning
    return fasttext.load_model(path)


def _attack_score(model, text: str) -> dict[str, float]:
    """Return {prompt_injection, jailbreak, attack=max} for one example."""
    from safety_classifier.fasttext_layer.compat import predict_fasttext
    labels, probs = predict_fasttext(model, text.replace("\n", " "), k=-1)
    scores = {lab[len("__label__"):]: float(p) for lab, p in zip(labels, probs)}
    pi = scores.get("prompt_injection", 0.0)
    jb = scores.get("jailbreak", 0.0)
    return {"prompt_injection": pi, "jailbreak": jb, "attack": max(pi, jb)}


# ---- metrics ---------------------------------------------------------------- #
def _metrics_at(gold, score, thr):
    pred = [1 if s >= thr else 0 for s in score]
    tp = sum(1 for g, p in zip(gold, pred) if g == 1 and p == 1)
    fp = sum(1 for g, p in zip(gold, pred) if g == 0 and p == 1)
    fn = sum(1 for g, p in zip(gold, pred) if g == 1 and p == 0)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1, tp, fp, fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/fasttext/attack_head_2.ftz")
    ap.add_argument("--dataset", default="rogue-security/prompt-injections-benchmark")
    ap.add_argument("--split", default=None, help="Force a split (else try all)")
    ap.add_argument("--held-out", default="none",
                    choices=["none", "train", "val", "test"],
                    help="If this dataset is in training, evaluate ONLY on rows "
                         "that fall in the given pipeline split (no leakage). "
                         "Uses the same normalize+hash bucketing as the splitter.")
    ap.add_argument("--limit", type=int, default=0, help="Cap examples (0 = all)")
    ap.add_argument("--score", default="attack",
                    choices=["attack", "prompt_injection", "jailbreak"],
                    help="Which head score to evaluate against the gold label")
    args = ap.parse_args()

    from datasets import load_dataset
    from sklearn.metrics import average_precision_score, roc_auc_score

    print("=" * 70)
    print(f"External benchmark eval: {args.dataset}")
    print(f"Model: {args.model}   |   scoring on: {args.score}")
    print("=" * 70)

    ds = load_dataset(args.dataset)
    split = args.split or ("test" if "test" in ds else list(ds.keys())[0])
    rows = ds[split]
    print(f"Split: {split}   rows: {len(rows)}   columns: {rows.column_names}")

    text_key = _detect_key(rows.column_names, _TEXT_KEYS)
    label_key = _detect_key(rows.column_names, _LABEL_KEYS)
    if not text_key or not label_key:
        print(f"✗ Could not detect text/label columns "
              f"(text={text_key}, label={label_key}).")
        print(f"  Available: {rows.column_names}")
        sys.exit(1)
    print(f"Detected: text='{text_key}', label='{label_key}'")
    print()

    # Optional held-out filtering: keep only rows whose normalized hash lands in
    # the requested pipeline split. This mirrors the splitter exactly, so if this
    # dataset is in training you can still get a leakage-free in-distribution score.
    split_filter = None
    if args.held_out != "none":
        from safety_classifier.data.splitter import _split_for_hash
        from safety_classifier.normalizer import normalize
        def split_filter(text: str) -> bool:
            return _split_for_hash(normalize(text).text_hash) == args.held_out
        print(f"Held-out mode: evaluating only on the '{args.held_out}' split "
              f"(leakage-free if the dataset is in training).")
        print()

    model = _load_model(args.model)

    gold, score, texts = [], [], []
    skipped = 0
    held_out_drop = 0
    n = len(rows) if args.limit == 0 else min(args.limit, len(rows))
    for i in range(n):
        row = rows[i]
        text = row.get(text_key)
        y = _to_binary(row.get(label_key))
        if not isinstance(text, str) or not text.strip() or y is None:
            skipped += 1
            continue
        if split_filter is not None and not split_filter(text):
            held_out_drop += 1
            continue
        gold.append(y)
        score.append(_attack_score(model, text)[args.score])
        texts.append(text)
    if args.held_out != "none":
        print(f"Held-out filter kept {len(gold)} rows "
              f"(dropped {held_out_drop} from other splits).")

    pos = sum(gold)
    neg = len(gold) - pos
    print(f"Evaluated {len(gold)} examples ({pos} injection / {neg} benign), "
          f"skipped {skipped}")
    if pos == 0 or neg == 0:
        print("✗ Need both classes present to compute ranking metrics.")
        sys.exit(1)
    print()

    pr_auc = average_precision_score(gold, score)
    roc_auc = roc_auc_score(gold, score)

    # Sweep thresholds for best F1 + high-recall operating point.
    grid = [round(t, 2) for t in np.linspace(0.05, 0.95, 19)]
    best = max(grid, key=lambda t: _metrics_at(gold, score, t)[2])
    bp, br, bf, *_ = _metrics_at(gold, score, best)
    # Calibrated router default for prompt_injection routing is ~0.05.
    rp, rr, rf, rtp, rfp, rfn = _metrics_at(gold, score, 0.05)
    dp, dr, df, *_ = _metrics_at(gold, score, 0.50)

    print("Ranking quality (threshold-independent):")
    print(f"  PR-AUC  : {pr_auc:.4f}")
    print(f"  ROC-AUC : {roc_auc:.4f}")
    print()
    print("Operating points:")
    print(f"  @0.05 (route)  P={rp:.3f} R={rr:.3f} F1={rf:.3f}")
    print(f"  @0.50 (default) P={dp:.3f} R={dr:.3f} F1={df:.3f}")
    print(f"  @{best:.2f} (best-F1) P={bp:.3f} R={br:.3f} F1={bf:.3f}")
    print()

    # A few misses to eyeball at the best-F1 threshold.
    fns = [t for g, s, t in zip(gold, score, texts) if g == 1 and s < best][:5]
    fps = [t for g, s, t in zip(gold, score, texts) if g == 0 and s >= best][:5]
    if fns:
        print(f"Sample FALSE NEGATIVES (missed injections @ {best:.2f}):")
        for t in fns:
            print(f"  - {t[:90]}")
        print()
    if fps:
        print(f"Sample FALSE POSITIVES (benign flagged @ {best:.2f}):")
        for t in fps:
            print(f"  - {t[:90]}")
        print()

    verdict = "STRONG" if pr_auc >= 0.85 else "OK" if pr_auc >= 0.70 else "WEAK"
    print("=" * 70)
    print(f"Verdict: {verdict} (PR-AUC={pr_auc:.3f}) on {args.dataset}")
    print("=" * 70)


if __name__ == "__main__":
    main()
