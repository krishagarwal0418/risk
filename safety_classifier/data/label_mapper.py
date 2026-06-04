"""Map external dataset / model labels into the canonical taxonomy.

Unknown labels are mapped to :data:`constants.UNKNOWN` and never silently dropped;
callers are expected to preserve the original label in raw metadata.
"""

from __future__ import annotations

import re

from .. import constants as C

# Direct, exact (normalized) label -> canonical label mappings.
_DIRECT_MAP: dict[str, str] = {
    # safe / benign
    "safe": C.SAFE,
    "benign": C.SAFE,
    "clean": C.SAFE,
    "none": C.SAFE,
    "ok": C.SAFE,
    "negative": C.SAFE,
    "0": C.SAFE,
    "non_toxic": C.SAFE,
    "not_toxic": C.SAFE,
    # prompt injection
    "injection": C.PROMPT_INJECTION,
    "prompt_injection": C.PROMPT_INJECTION,
    "prompt-injection": C.PROMPT_INJECTION,
    "promptinjection": C.PROMPT_INJECTION,
    "malicious_instruction": C.PROMPT_INJECTION,
    "indirect_injection": C.PROMPT_INJECTION,
    "malicious": C.PROMPT_INJECTION,
    "attack": C.PROMPT_INJECTION,
    # jailbreak
    "jailbreak": C.JAILBREAK,
    "jailbreak_attempt": C.JAILBREAK,
    "jailbreaking": C.JAILBREAK,
    "dan": C.JAILBREAK,
    "bypass": C.JAILBREAK,
    # toxicity
    "toxic": C.TOXICITY,
    "toxicity": C.TOXICITY,
    "severe_toxic": C.TOXICITY,
    "severe_toxicity": C.TOXICITY,
    "insult": C.TOXICITY,
    "obscene": C.TOXICITY,
    "profanity": C.TOXICITY,
    # hate
    "hate": C.HATE,
    "identity_hate": C.HATE,
    "identity_attack": C.HATE,
    "hate_speech": C.HATE,
    "hatespeech": C.HATE,
    "discrimination": C.HATE,
    # harassment
    "harassment": C.HARASSMENT,
    "harass": C.HARASSMENT,
    "bullying": C.HARASSMENT,
    # sexual
    "sexual": C.SEXUAL,
    "sexual_content": C.SEXUAL,
    "sexual/minors": C.SEXUAL,
    "sexual_minors": C.SEXUAL,
    "porn": C.SEXUAL,
    "nsfw": C.SEXUAL,
    # violence
    "violence": C.VIOLENCE,
    "violent": C.VIOLENCE,
    "threat": C.VIOLENCE,
    "violence_graphic": C.VIOLENCE,
    "violence/graphic": C.VIOLENCE,
    "graphic_violence": C.VIOLENCE,
    "harassment_threatening": C.VIOLENCE,
    "harassment/threatening": C.VIOLENCE,
    "hate_threatening": C.VIOLENCE,
    "hate/threatening": C.VIOLENCE,
    # self harm
    "self_harm": C.SELF_HARM,
    "self-harm": C.SELF_HARM,
    "selfharm": C.SELF_HARM,
    "self_harm_intent": C.SELF_HARM,
    "self-harm/intent": C.SELF_HARM,
    "self_harm_instructions": C.SELF_HARM,
    "self-harm/instructions": C.SELF_HARM,
    "suicide": C.SELF_HARM,
    "suicidal": C.SELF_HARM,
    # dangerous information
    "dangerous": C.DANGEROUS_INFORMATION,
    "dangerous_information": C.DANGEROUS_INFORMATION,
    "weapons": C.DANGEROUS_INFORMATION,
    "cyber_abuse": C.DANGEROUS_INFORMATION,
    "cybercrime": C.DANGEROUS_INFORMATION,
    # illegal activity
    "illegal_activity": C.ILLEGAL_ACTIVITY,
    "illegal": C.ILLEGAL_ACTIVITY,
    "illegal_advice": C.ILLEGAL_ACTIVITY,
    "crime": C.ILLEGAL_ACTIVITY,
}

# A label can carry sexual context that overrides the default mapping.
_SEXUAL_CONTEXT_WORDS = ("sexual", "porn", "nsfw", "explicit")


def _norm(label: str) -> str:
    return re.sub(r"\s+", "_", str(label).strip().lower())


def map_label(label: str, *, context: str | None = None) -> str:
    """Map a single external label to a canonical label.

    ``context`` (optional) is the raw text or dataset hint used to disambiguate
    overloaded labels such as ``obscene`` (toxicity vs sexual).
    """
    norm = _norm(label)
    if norm in _DIRECT_MAP:
        canonical = _DIRECT_MAP[norm]
        # "obscene" maps to toxicity, but promote to sexual in a sexual context.
        if norm == "obscene" and context:
            ctx = context.lower()
            if any(w in ctx for w in _SEXUAL_CONTEXT_WORDS):
                return C.SEXUAL
        return canonical
    return C.UNKNOWN


def map_labels(labels: list[str], *, context: str | None = None) -> list[str]:
    """Map a list of external labels to a deduplicated list of canonical labels.

    ``safe`` is dropped if any risk label is present (an example labelled both
    safe and toxic is, in practice, unsafe).
    """
    mapped: list[str] = []
    for lab in labels:
        canonical = map_label(lab, context=context)
        if canonical not in mapped:
            mapped.append(canonical)
    risk = [m for m in mapped if m in C.RISK_LABELS or m == C.UNKNOWN]
    if risk and C.SAFE in mapped:
        mapped = [m for m in mapped if m != C.SAFE]
    if not mapped:
        mapped = [C.UNKNOWN]
    return mapped
