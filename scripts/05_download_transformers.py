"""Download transformer models from the Hugging Face Hub into models/transformers/."""

from __future__ import annotations

import _bootstrap  # noqa: F401

from safety_classifier.config import load_models_config, resolve_path

_KEYS = (
    "prompt_injection",
    "jailbreak",
    "moderation_fallback",
    "toxic_fallback",
)


def download_one(hf_name: str, local_path: str) -> dict:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out = resolve_path(local_path)
    out.mkdir(parents=True, exist_ok=True)
    try:
        tok = AutoTokenizer.from_pretrained(hf_name)
        model = AutoModelForSequenceClassification.from_pretrained(hf_name)
        tok.save_pretrained(out)
        model.save_pretrained(out)
        # Surface the model's label scheme so mappings can be verified.
        id2label = getattr(model.config, "id2label", {})
        print(f"[download] {hf_name}: id2label={id2label}")
        return {"status": "ok", "id2label": id2label}
    except Exception as exc:  # noqa: BLE001
        print(f"[download] FAILED {hf_name}: {exc}")
        return {"status": "error", "error": str(exc)}


def main() -> None:
    cfg = load_models_config().get("transformers", {})
    for key in _KEYS:
        entry = cfg.get(key)
        if not entry:
            continue
        print(f"[download] {key}: {entry['hf_name']}")
        download_one(entry["hf_name"], entry["local_path"])


if __name__ == "__main__":
    main()
