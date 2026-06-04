"""Adapters that convert each dataset family into the unified record format.

Unified record::

    {"text": "...", "labels": ["prompt_injection", ...], "source": "dataset_name"}

Each adapter is tolerant of schema drift: it inspects available columns and falls
back gracefully. Labels here are *external* labels; the canonical mapping happens
later via :mod:`label_mapper` in the splitter. Some adapters emit already-canonical
labels when the dataset's intent is unambiguous (e.g. a pure jailbreak dataset).
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator

from .. import constants as C

UnifiedRecord = dict[str, Any]
Adapter = Callable[[Iterable[dict], str], Iterator[UnifiedRecord]]

_TEXT_KEYS = (
    "text",
    "prompt",
    "comment_text",
    "content",
    "sentence",
    "instruction",
    "question",
    "input",
    "message",
    "jailbreak_query",
    "goal",
)


def _first_text(row: dict) -> str | None:
    for key in _TEXT_KEYS:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return None


def _rec(text: str, labels: list[str], source: str) -> UnifiedRecord:
    return {"text": text, "labels": labels, "source": source}


# --------------------------------------------------------------------------- #
# Attack-style adapters
# --------------------------------------------------------------------------- #
def adapt_prompt_injection(rows: Iterable[dict], source: str) -> Iterator[UnifiedRecord]:
    """deepset/prompt-injections and Harelix mixed techniques.

    Label column is typically ``label`` with 1 = injection, 0 = safe.
    """
    for row in rows:
        text = _first_text(row)
        if not text:
            continue
        label = row.get("label")
        if label in (1, "1", "injection", "INJECTION", True):
            yield _rec(text, [C.PROMPT_INJECTION], source)
        elif label in (0, "0", "legit", "safe", False, None):
            yield _rec(text, [C.SAFE], source)
        else:
            yield _rec(text, [str(label)], source)


def adapt_jailbreak(rows: Iterable[dict], source: str) -> Iterator[UnifiedRecord]:
    """jackhhao/jailbreak-classification, in-the-wild jailbreak prompts."""
    for row in rows:
        text = _first_text(row)
        if not text:
            continue
        label = row.get("type", row.get("label"))
        if label is None:
            # in-the-wild prompts: every row is a jailbreak prompt.
            yield _rec(text, [C.JAILBREAK], source)
            continue
        sval = str(label).lower()
        if "jailbreak" in sval or sval in ("1", "true"):
            yield _rec(text, [C.JAILBREAK], source)
        elif sval in ("benign", "safe", "0", "false"):
            yield _rec(text, [C.SAFE], source)
        else:
            yield _rec(text, [sval], source)


def adapt_salad(rows: Iterable[dict], source: str) -> Iterator[UnifiedRecord]:
    """OpenSafetyLab/Salad-Data attack-enhanced set: attack prompts."""
    for row in rows:
        text = row.get("augq") or row.get("attack") or _first_text(row)
        if not text:
            continue
        yield _rec(text, [C.PROMPT_INJECTION, C.JAILBREAK], source)


def adapt_xstest(rows: Iterable[dict], source: str) -> Iterator[UnifiedRecord]:
    """XSTest: ``type`` starting with ``safe`` => safe; otherwise unsafe prompt."""
    for row in rows:
        text = _first_text(row)
        if not text:
            continue
        t = str(row.get("type", "")).lower()
        if t.startswith("safe") or t.startswith("contrast") is False and "safe" in t:
            yield _rec(text, [C.SAFE], source)
        else:
            yield _rec(text, [C.SAFE if t.startswith("safe") else C.UNKNOWN], source)


# --------------------------------------------------------------------------- #
# Toxicity / hate adapters
# --------------------------------------------------------------------------- #
_JIGSAW_COLS = (
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
)


def adapt_jigsaw(rows: Iterable[dict], source: str) -> Iterator[UnifiedRecord]:
    """Jigsaw toxic comment / unintended bias multi-label datasets."""
    for row in rows:
        text = row.get("comment_text") or _first_text(row)
        if not text:
            continue
        labels = [col for col in _JIGSAW_COLS if float(row.get(col, 0) or 0) >= 0.5]
        if not labels:
            labels = [C.SAFE]
        yield _rec(text, labels, source)


def adapt_toxicchat(rows: Iterable[dict], source: str) -> Iterator[UnifiedRecord]:
    """lmsys/toxic-chat: ``toxicity`` (0/1) and ``jailbreaking`` (0/1)."""
    for row in rows:
        text = row.get("user_input") or _first_text(row)
        if not text:
            continue
        labels: list[str] = []
        if int(row.get("toxicity", 0) or 0) == 1:
            labels.append(C.TOXICITY)
        if int(row.get("jailbreaking", 0) or 0) == 1:
            labels.append(C.JAILBREAK)
        if not labels:
            labels = [C.SAFE]
        yield _rec(text, labels, source)


def adapt_toxigen(rows: Iterable[dict], source: str) -> Iterator[UnifiedRecord]:
    """toxigen/toxigen-data: ``prompt_label`` 1 = toxic/hateful."""
    for row in rows:
        text = row.get("generation") or row.get("text") or _first_text(row)
        if not text:
            continue
        label = row.get("prompt_label", row.get("label"))
        if int(label or 0) == 1:
            yield _rec(text, [C.HATE, C.TOXICITY], source)
        else:
            yield _rec(text, [C.SAFE], source)


# --------------------------------------------------------------------------- #
# Broad moderation adapters
# --------------------------------------------------------------------------- #
def adapt_moderation_410k(rows: Iterable[dict], source: str) -> Iterator[UnifiedRecord]:
    """ifmain/text-moderation-410K: OpenAI-style category flags.

    Rows contain a ``text`` field and a ``moderation`` dict of category->bool, or
    flat boolean columns. We extract any truthy categories as external labels.
    """
    category_keys = (
        "harassment",
        "harassment/threatening",
        "hate",
        "hate/threatening",
        "self-harm",
        "self-harm/intent",
        "self-harm/instructions",
        "sexual",
        "sexual/minors",
        "violence",
        "violence/graphic",
    )
    for row in rows:
        text = _first_text(row)
        if not text:
            continue
        mod = row.get("moderation") or row
        if isinstance(mod, dict):
            cats = mod.get("categories", mod)
        else:
            cats = row
        labels = [k for k in category_keys if isinstance(cats, dict) and cats.get(k)]
        if not labels:
            labels = [C.SAFE]
        yield _rec(text, labels, source)


def adapt_wildguard(rows: Iterable[dict], source: str) -> Iterator[UnifiedRecord]:
    """allenai/wildguardmix: ``prompt_harm_label`` harmful/unharmful."""
    for row in rows:
        text = row.get("prompt") or _first_text(row)
        if not text:
            continue
        harm = str(row.get("prompt_harm_label", "")).lower()
        if harm == "harmful":
            yield _rec(text, [C.UNKNOWN], source)
        elif harm == "unharmful":
            yield _rec(text, [C.SAFE], source)
        else:
            yield _rec(text, [C.SAFE], source)


def adapt_beavertails(rows: Iterable[dict], source: str) -> Iterator[UnifiedRecord]:
    """PKU-Alignment/BeaverTails: ``is_safe`` + ``category`` dict."""
    cat_map = {
        "animal_abuse": C.VIOLENCE,
        "child_abuse": C.SEXUAL,
        "controversial_topics,politics": C.UNKNOWN,
        "discrimination,stereotype,injustice": C.HATE,
        "drug_abuse,weapons,banned_substance": C.DANGEROUS_INFORMATION,
        "financial_crime,property_crime,theft": C.ILLEGAL_ACTIVITY,
        "hate_speech,offensive_language": C.HATE,
        "misinformation_regarding_ethics,laws_and_safety": C.UNKNOWN,
        "non_violent_unethical_behavior": C.UNKNOWN,
        "privacy_violation": C.ILLEGAL_ACTIVITY,
        "self_harm": C.SELF_HARM,
        "sexually_explicit,adult_content": C.SEXUAL,
        "terrorism,organized_crime": C.ILLEGAL_ACTIVITY,
        "violence,aiding_and_abetting,incitement": C.VIOLENCE,
    }
    for row in rows:
        text = row.get("prompt") or _first_text(row)
        if not text:
            continue
        if row.get("is_safe"):
            yield _rec(text, [C.SAFE], source)
            continue
        category = row.get("category", {})
        labels = []
        if isinstance(category, dict):
            for k, v in category.items():
                if v and k in cat_map:
                    labels.append(cat_map[k])
        labels = list(dict.fromkeys(labels)) or [C.UNKNOWN]
        yield _rec(text, labels, source)


def adapt_generic(rows: Iterable[dict], source: str) -> Iterator[UnifiedRecord]:
    """Generic adapter for local CSV/JSONL with ``text`` and ``labels`` columns."""
    for row in rows:
        text = _first_text(row)
        if not text:
            continue
        labels = row.get("labels")
        if isinstance(labels, str):
            labels = [s.strip() for s in labels.split(",") if s.strip()]
        elif labels is None:
            label = row.get("label")
            labels = [str(label)] if label is not None else [C.SAFE]
        yield _rec(text, list(labels), source)


ADAPTERS: dict[str, Adapter] = {
    "prompt_injection": adapt_prompt_injection,
    "jailbreak": adapt_jailbreak,
    "salad": adapt_salad,
    "xstest": adapt_xstest,
    "jigsaw": adapt_jigsaw,
    "toxicchat": adapt_toxicchat,
    "toxigen": adapt_toxigen,
    "moderation_410k": adapt_moderation_410k,
    "wildguard": adapt_wildguard,
    "beavertails": adapt_beavertails,
    "generic": adapt_generic,
}


def get_adapter(kind: str) -> Adapter:
    return ADAPTERS.get(kind, adapt_generic)
