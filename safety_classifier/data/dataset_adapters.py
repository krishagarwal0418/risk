"""Adapters that convert each dataset family into the unified record format.

Unified record::

    {"text": "...", "labels": ["prompt_injection", ...], "source": "dataset_name"}

Each adapter is tolerant of schema drift: it inspects available columns and falls
back gracefully. Labels here are *external* labels; the canonical mapping happens
later via :mod:`label_mapper` in the splitter. Some adapters emit already-canonical
labels when the dataset's intent is unambiguous (e.g. a pure jailbreak dataset).
"""

from __future__ import annotations

import re
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
    """ifmain/text-moderation-410K.

    Real schema: each row has ``text`` and ``moder`` =
    ``{"categories": {cat: bool}, "category_scores": {cat: float}}``. We emit every
    category flagged ``True``; rows with nothing flagged are labelled safe. The
    older ``moderation`` key is also accepted for robustness. Raw category names
    (e.g. ``harassment``, ``self_harm_instructions``, ``hate/threatening``) are
    mapped to canonical labels downstream by the label mapper.
    """
    for row in rows:
        text = _first_text(row)
        if not text:
            continue
        mod = row.get("moder") or row.get("moderation") or {}
        cats = mod.get("categories", {}) if isinstance(mod, dict) else {}
        labels = [k for k, v in cats.items() if v is True]
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
    """PKU-Alignment/BeaverTails: ``is_safe`` + ``category`` dict.

    BeaverTails contains the SAME prompt twice: once with a safe response
    (is_safe=True) and once with a harmful response (is_safe=False). The
    splitter deduplicates by prompt-text hash and keeps the first occurrence.
    Since safe rows appear first in the iteration order, the harmful label was
    being silently discarded — causing dangerous_information to drop from
    20k raw to 843 processed and self_harm from 2.4k to 687.

    Fix: skip is_safe=True rows entirely. BeaverTails prompts are harmful
    requests by design; we have 400k+ safe examples from other sources.
    """
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
            continue  # skip — safe version of a harmful prompt poisons dedup
        category = row.get("category", {})
        labels = []
        if isinstance(category, dict):
            for k, v in category.items():
                if v and k in cat_map:
                    labels.append(cat_map[k])
        labels = list(dict.fromkeys(labels)) or [C.UNKNOWN]
        yield _rec(text, labels, source)


def adapt_aegis(rows: Iterable[dict], source: str) -> Iterator[UnifiedRecord]:
    """nvidia/Aegis-AI-Content-Safety-Dataset-1.0.

    Real schema (confirmed from data): num_annotations, id, text, text_type,
    labels_0..labels_4 where values are "Safe" / "Needs Caution" /
    "Needs Intervention" / "Other" / "None".  NO fine-grained harm categories.

    Strategy: majority "Needs Intervention" → skip (unknown category, can't
    assign a specific label); majority "Safe"/"Needs Caution" → safe.
    This avoids polluting the safe pool with borderline content and avoids
    emitting meaningless "unknown" rows that get dropped by the splitter anyway.
    """
    for row in rows:
        text = _first_text(row)
        if not text:
            continue
        votes = [
            str(row.get(f"labels_{i}", "") or "").strip().lower()
            for i in range(5)
            if str(row.get(f"labels_{i}", "") or "").strip().lower()
               not in ("none", "other", "")
        ]
        if not votes:
            continue
        intervention = sum(1 for v in votes if "intervention" in v)
        if intervention > len(votes) / 2:
            continue  # genuinely unsafe but no category → skip
        yield _rec(text, [C.SAFE], source)


def adapt_do_not_answer(rows: Iterable[dict], source: str) -> Iterator[UnifiedRecord]:
    """LibrAI/do-not-answer.

    Schema: ``question`` (the prompt), ``types_of_harm`` (free-text category),
    ``specific_harms`` (sub-category). We map the harm category string.
    """
    _cat_map = {
        "information hazards": "dangerous_information",
        "malicious uses": "illegal_activity",
        "discrimination, stereotype, injustice": "hate",
        "human trafficking": "illegal_activity",
        "financial crime": "illegal_activity",
        "privacy violation": "illegal_activity",
        "mental health or self-harm": "self_harm",
        "self-harm": "self_harm",
        "violence": "violence",
        "hate speech": "hate",
        "adult content": "sexual",
        "terrorism": "illegal_activity",
        "weapons": "dangerous_information",
    }
    for row in rows:
        text = row.get("question") or _first_text(row)
        if not text:
            continue
        harm = str(row.get("types_of_harm", "") or "").strip().lower()
        label = next(
            (v for k, v in _cat_map.items() if k in harm),
            None,
        )
        if label:
            yield _rec(text, [label], source)
        else:
            # Rows without a mapped harm type are benign refusal-test examples.
            yield _rec(text, [C.SAFE], source)


_HARMFUL_PROMPT_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        C.SELF_HARM,
        (
            r"\bself[-\s]?harm\b",
            r"\bsuicide\b",
            r"\bkill myself\b",
            r"\bharm myself\b",
            r"\boverdose\b",
            r"\beating disorder\b",
        ),
    ),
    (
        C.DANGEROUS_INFORMATION,
        (
            r"\bbomb\b",
            r"\bexplosive",
            r"\bweapon",
            r"\bfirearm",
            r"\bgun\b",
            r"\bmalware\b",
            r"\bvirus\b",
            r"\bransomware\b",
            r"\bphishing\b",
            r"\bhack(?:ing|er|ed|s)?\b",
            r"\bexploit\b",
            r"\bcyberattack\b",
            r"\bdrug",
            r"\bcocaine\b",
            r"\bfentanyl\b",
            r"\bmeth\b",
            r"\bpoison",
            r"\btoxin",
            r"\bchemical\b",
            r"\bbiological\b",
            r"\bnuclear\b",
        ),
    ),
    (
        C.ILLEGAL_ACTIVITY,
        (
            r"\billegal\b",
            r"\bfraud\b",
            r"\btheft\b",
            r"\bidentity theft\b",
            r"\bsteal",
            r"\bshoplift",
            r"\bscam\b",
            r"\bcounterfeit",
            r"\bmoney laundering\b",
            r"\bcredit card\b",
            r"\bfake id\b",
            r"\bblackmail\b",
            r"\bbribe\b",
            r"\bstalk",
        ),
    ),
    (
        C.VIOLENCE,
        (
            r"\bkill\b",
            r"\bmurder\b",
            r"\bassault\b",
            r"\btorture\b",
            r"\bterrorist attack\b",
            r"\bkidnap",
            r"\bstrangle\b",
            r"\bunconscious\b",
        ),
    ),
    (
        C.SEXUAL,
        (
            r"\bsexual\b",
            r"\bporn",
            r"\brape\b",
            r"\badult content\b",
            r"\bexplicit\b",
            r"\bnudity\b",
            r"\bprostitut",
        ),
    ),
    (
        C.HATE,
        (
            r"\bhate speech\b",
            r"\bracist\b",
            r"\bnazi\b",
            r"\bgenocide\b",
            r"\bslur\b",
        ),
    ),
    (
        C.HARASSMENT,
        (
            r"\bharass",
            r"\bcyberbully",
            r"\bcyberbullying\b",
            r"\bbully",
            r"\bstalk",
            r"\bdefam",
        ),
    ),
)


def _harmful_prompt_labels(text: str) -> list[str]:
    """Weak labels for harmful prompt datasets that lack category columns."""
    low = text.lower()
    labels: list[str] = []
    for label, patterns in _HARMFUL_PROMPT_PATTERNS:
        if any(re.search(pattern, low) for pattern in patterns):
            labels.append(label)
    return labels or [C.UNKNOWN]


def adapt_harmful_prompts(rows: Iterable[dict], source: str) -> Iterator[UnifiedRecord]:
    """Harmful prompt-only datasets with no category column.

    Example: ``LLM-LAT/harmful-dataset`` has ``prompt``, ``rejected`` and
    ``chosen`` but no fine-grained harm label. These are harmful prompts by
    construction, so emit conservative keyword-derived weak labels instead of
    sending every row to ``unknown``.
    """
    for row in rows:
        text = _first_text(row)
        if not text:
            continue
        yield _rec(text, _harmful_prompt_labels(text), source)


def adapt_sorry_bench(rows: Iterable[dict], source: str) -> Iterator[UnifiedRecord]:
    """sorry-bench/sorry-bench-202406.

    Schema: ``prompt``, ``category`` (int 1-45 mapping to harm sub-categories),
    ``category_original`` or ``category_name`` (string label if present).
    All SORRY-Bench prompts are harmful by design — we map the category.
    """
    # Category ID ranges → external label (based on published taxonomy).
    # 1-8:  weapons/CBRN/cybercrime → dangerous_information
    # 9-12: fraud/illegal → illegal_activity
    # 13-16: self harm / suicide → self_harm
    # 17-20: adult/sexual → sexual
    # 21-24: hate/discrimination → hate
    # 25-28: violence/threats → violence
    # 29+:   misc harmful → unknown
    _name_map = {
        "chemical": "dangerous_information",
        "biological": "dangerous_information",
        "radiological": "dangerous_information",
        "nuclear": "dangerous_information",
        "cybercrime": "dangerous_information",
        "weapon": "dangerous_information",
        "illegal": "illegal_activity",
        "fraud": "illegal_activity",
        "drug": "dangerous_information",
        "suicide": "self_harm",
        "self-harm": "self_harm",
        "self harm": "self_harm",
        "eating disorder": "self_harm",
        "sexual": "sexual",
        "adult": "sexual",
        "pornograph": "sexual",
        "hate": "hate",
        "discriminat": "hate",
        "harassment": "harassment",
        "stalk": "harassment",
        "violence": "violence",
        "threat": "violence",
        "murder": "violence",
        "human traffick": "illegal_activity",
    }
    for row in rows:
        text = row.get("prompt") or _first_text(row)
        if not text:
            continue
        cat_name = str(
            row.get("category_original") or row.get("category_name") or ""
        ).strip().lower()
        label = next((v for k, v in _name_map.items() if k in cat_name), None)
        if label is None and not cat_name and row.get("category") is None:
            yield _rec(text, _harmful_prompt_labels(text), source)
            continue
        if label is None:
            # Fall back to numeric range.
            try:
                cat_id = int(row.get("category") or 0)
            except (ValueError, TypeError):
                cat_id = 0
            if 1 <= cat_id <= 8:
                label = "dangerous_information"
            elif 9 <= cat_id <= 12:
                label = "illegal_activity"
            elif 13 <= cat_id <= 16:
                label = "self_harm"
            elif 17 <= cat_id <= 20:
                label = "sexual"
            elif 21 <= cat_id <= 24:
                label = "hate"
            elif 25 <= cat_id <= 28:
                label = "violence"
            else:
                label = C.UNKNOWN
        yield _rec(text, [label], source)


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
    "aegis": adapt_aegis,
    "do_not_answer": adapt_do_not_answer,
    "harmful_prompts": adapt_harmful_prompts,
    "sorry_bench": adapt_sorry_bench,
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
