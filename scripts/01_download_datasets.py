"""Download and adapt all configured datasets into data/raw/."""

from __future__ import annotations

import os

import _bootstrap  # noqa: F401

from safety_classifier.data.dataset_downloader import download_all, print_report

# ---------------------------------------------------------------------------
# Gated datasets (allenai/wildguardmix, sorry-bench, rogue-security) require a
# Hugging Face token. Set it in your environment before running — e.g. in Colab:
#     import os; os.environ["HF_TOKEN"] = "hf_..."
# or in a shell:
#     export HF_TOKEN=hf_...
# (No token is stored in the repo — GitHub push protection blocks that, and it
# would be a leaked credential anyway.) Whichever of the two standard env vars
# is set is mirrored to the other so every HF code path sees it.
# ---------------------------------------------------------------------------
_tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if _tok:
    os.environ.setdefault("HF_TOKEN", _tok)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", _tok)
else:
    print("[warn] No HF_TOKEN set — gated datasets (wildguardmix, sorry-bench, "
          "rogue-security) will be skipped. Export HF_TOKEN to include them.")


def main() -> None:
    metadata = download_all()
    print_report(metadata)


if __name__ == "__main__":
    main()
