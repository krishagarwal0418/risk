"""Deterministic safety gate that runs BEFORE the probabilistic FastText router.

High-precision rules that catch *known* bad patterns with certainty and, crucially,
**veto the fast-allow shortcut** so an obvious attack can never be skipped just
because FastText scored it low. It can only ever make the system MORE cautious
(force a BERT, or block) — never less.

Design:
  * Each rule maps a match to a forced route ("attack" / "moderation") or a block.
  * Matching runs over the normalized detection text AND a de-obfuscated copy
    (de-leet, un-spaced, homoglyph-folded), so simple evasions don't bypass it.
  * Obfuscation signals (base64, zero-width, spaced-out, homoglyphs) themselves
    force a route — we don't trust the fuzzy model on mangled input.

This is a FLOOR, not the engine: near-zero recall on novel attacks (attackers
evade exact strings), but very high precision on the ones it covers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..normalizer import NormalizedText

ATTACK = "attack"
MODERATION = "moderation"


# --------------------------------------------------------------------------- #
# Rule sets (compiled regexes). Keep additions here — the rest is generic.
# --------------------------------------------------------------------------- #

# Prompt-injection / instruction-override.
_INJECTION = [
    r"ignore (all |any |the )?(previous|prior|above|earlier|preceding) (instruction|prompt|message|rule|direction)",
    r"disregard (all |any |the )?(previous|prior|above|earlier) (instruction|prompt|message|rule)",
    r"forget (all |everything |your |the )?(previous|prior|above|earlier) (instruction|prompt|rule|context)",
    r"forget (everything|all) (you|that) (were|was) (told|instructed)",
    r"do not (follow|obey) (the |your )?(previous|above|prior|system) (instruction|prompt|rule)",
    r"(override|bypass|circumvent|disable) (the |your |all )?(safety|guideline|rule|filter|restriction|guardrail)",
    r"ignore (your|the|all) (guideline|rule|policy|restriction|safety|guardrail)",
    r"new instruction(s)?:",
    r"the (real|actual|true) (instruction|task|prompt) (is|:)",
    r"(end|ignore) (of |the )?(above|previous) (text|prompt|instruction)",
    r"this is (a |an )?(test|drill|exercise).{0,40}(ignore|bypass)",
]

# System-prompt / data extraction.
_EXTRACTION = [
    r"(reveal|show|print|repeat|display|output|tell me|reproduce) (me )?(your |the )?(system|initial|original|hidden) (prompt|instruction|message|directive)",
    r"(what|repeat) (are|were|is) (your|the) (exact )?(instruction|prompt|rule|directive|guideline)s?",
    r"repeat (the |everything |all )?(word(s)? )?(above|before this|prior to)",
    r"print everything (above|before)",
    r"(what's|what is) (written |stated )?(above|in your prompt|before this)",
    r"summari[sz]e your (instruction|prompt|system message)",
    r"copy (the|your) (text|prompt|instruction) (above|verbatim)",
]

# Jailbreak personas / modes.
_JAILBREAK = [
    r"\b(dan|stan|dude|aim|jailbreak|do anything now)\b.{0,30}(mode|persona|prompt|enabled)?",
    r"you are (now |going to be )?(dan|stan|an? unrestricted|an? unfiltered|jailbroken)",
    r"(developer|debug|god|sudo|root|admin|unrestricted|unfiltered|uncensored) mode",
    r"(enable|activate|enter|turn on) (developer|debug|god|dan|jailbreak|unrestricted) mode",
    r"you (have|are) no (restriction|filter|limit|guideline|rule|guardrail)s?",
    r"without (any )?(restriction|filter|limit|censorship|moral|ethic|guideline)s?",
    r"you (are|will be) (free|unrestricted|unfiltered|uncensored|liberated)",
    r"pretend (you are|to be) (an? )?(ai|model|assistant)? ?(with no|without) (rule|restriction|filter|guideline)",
    r"(act|roleplay|role-play|behave) as (an? )?(unrestricted|unfiltered|evil|amoral|jailbroken)",
    r"evil (confidant|assistant|ai|trusted)",
    r"opposite day|do the opposite of",
    r"hypothetical(ly)?.{0,40}(no (rule|restriction)|ignore|bypass)",
]

# Chat/format-token injection.
_FORMAT_INJECTION = [
    r"<\|?(im_start|im_end|system|assistant|user|endoftext)\|?>",
    r"\[/?INST\]",
    r"\[/?SYS\]",
    r"### (system|instruction|response):",
    r"\bsystem\s*:\s*you (are|must|will)",
    r"<<SYS>>|<</SYS>>",
]

# Dangerous information (weapons / CBRN / cyber / drugs) -> moderation.
_DANGEROUS = [
    r"how (to|do i|can i) (make|build|create|construct|synthesi[sz]e|assemble) (a |an )?(bomb|explosive|grenade|ied|pipe bomb|molotov|detonator)",
    r"(build|make|create|construct) (a |an )?(nuclear|biological|chemical|dirty) (bomb|weapon|device)",
    r"(synthesi[sz]e|manufacture|produce|cook) (meth|methamphetamine|fentanyl|heroin|cocaine|mdma|lsd|nerve agent|sarin|ricin|vx)",
    r"how (to|do i) (acquire|buy|obtain|get) (a |an )?(illegal |unregistered )?(firearm|gun|weapon|explosive) (without|illegally|untraceable)",
    r"(anarchist cookbook|jihad|how to make napalm|how to make thermite)",
    r"how (to|do i) (hack|ddos|sql inject|exploit|breach|brute.?force) (into |a )?(server|database|account|network|system|wifi)",
    r"(write|create|generate) (a |me )?(virus|malware|ransomware|keylogger|trojan|worm|botnet|rootkit)",
    r"how (to|do i) (poison|kill) (someone|a person|people|my)",
]

# Self-harm / suicide -> moderation (route fast so the model confirms).
_SELF_HARM = [
    r"how (to|do i|can i) (kill|end|hurt|harm|cut) (myself|my self)",
    r"(ways|methods|easiest way|best way|painless way) to (kill myself|commit suicide|end (my|it) (life|all))",
    r"i (want|need|am going) to (kill myself|end my life|commit suicide|die|not (be|exist) (here|anymore))",
    r"(should|how) i (overdose|hang myself|slit my wrists)",
    r"(suicide|self.?harm) (method|technique|instruction|guide)",
    r"how much .{0,20}(to overdose|will kill me)",
]

# Sexual content involving minors -> BLOCK (the one unambiguous block).
_CSAM = [
    r"(child|minor|underage|teen|preteen|kid|loli|young (boy|girl))s?.{0,40}(sexual|nude|naked|porn|explicit|sex|fondl|molest)",
    r"(sexual|explicit|nude|porn).{0,40}(child|minor|underage|preteen|toddler|infant|loli)",
    r"cp\b.{0,20}(link|download|share|trade)",
]

# Explicit sexual content -> moderation.
_SEXUAL = [
    r"(write|generate|describe|create) (a |me |an )?(explicit|graphic|hardcore|pornographic) (sex|sexual|erotic|porn)",
    r"(describe|narrate) .{0,30}(having sex|sexual intercourse|gen[ie]tal)",
]

# Structural hate (group + dehumanization) -> moderation. Pattern-based, not a
# raw slur list, so it's robust + maintainable.
_HATE = [
    r"(all|every|those) (\w+ )?(people|men|women|jews|muslims|blacks|whites|asians|gays|immigrants|trans) (are|should be) (subhuman|vermin|animals|scum|inferior|exterminated|killed|gassed|deported|eliminated)",
    r"(\w+) (are|is) (subhuman|vermin|cockroach|animal|parasite|disease)s? (and|who) (should|must) (die|be (killed|exterminated|removed))",
    r"(gas|exterminate|genocide|ethnic(ally)? cleans|lynch) (the |all )?(\w+ )?(jews|muslims|blacks|immigrants|gays|people)",
    r"(kill|murder|hang|shoot) (all |every )?(the )?(jews|muslims|blacks|gays|immigrants|n[i1]ggers?|f[a4]ggots?)",
]


def _compile(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


_RULES: list[tuple[str, list[re.Pattern], str, bool]] = [
    # (name, compiled patterns, route, is_block)
    ("injection", _compile(_INJECTION), ATTACK, False),
    ("prompt_extraction", _compile(_EXTRACTION), ATTACK, False),
    ("jailbreak", _compile(_JAILBREAK), ATTACK, False),
    ("format_injection", _compile(_FORMAT_INJECTION), ATTACK, False),
    ("dangerous_information", _compile(_DANGEROUS), MODERATION, False),
    ("self_harm", _compile(_SELF_HARM), MODERATION, False),
    ("sexual_explicit", _compile(_SEXUAL), MODERATION, False),
    ("hate", _compile(_HATE), MODERATION, False),
    ("csam", _compile(_CSAM), MODERATION, True),  # also blocks
]


# --------------------------------------------------------------------------- #
# De-obfuscation: fold simple evasions so rules still match.
# --------------------------------------------------------------------------- #
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s",
                       "7": "t", "8": "b", "@": "a", "$": "s", "!": "i"})
_SPACED_RE = re.compile(r"(?:\b\w\b[\s._\-]+){5,}\w\b")          # "i g n o r e"
_HOMOGLYPH_RE = re.compile(r"[Ѐ-ӿͰ-Ͽ]")       # Cyrillic/Greek
_REPEAT_CHAR_RE = re.compile(r"(.)\1{4,}")


def _deobfuscate(text: str) -> str:
    t = text.lower()
    # collapse spaced-out letters: "i g n o r e" -> "ignore"
    if _SPACED_RE.search(t):
        t = re.sub(r"(?<=\b\w)[\s._\-]+(?=\w\b)", "", t)
    t = t.translate(_LEET)
    t = _REPEAT_CHAR_RE.sub(r"\1", t)        # "killlll" -> "kil" (then rules still hit "kill"? keep mild)
    return t


@dataclass
class GateResult:
    force_route: set[str] = field(default_factory=set)   # {"attack","moderation"}
    block: bool = False
    reasons: list[str] = field(default_factory=list)

    @property
    def allow_fast_skip(self) -> bool:
        """True only when nothing matched — i.e. eligible for fast-allow."""
        return not self.force_route and not self.block

    @property
    def matched(self) -> bool:
        return bool(self.force_route) or self.block


def evaluate(norm: "NormalizedText") -> GateResult:
    """Run all deterministic rules over a normalized input."""
    res = GateResult()

    # Texts to scan: normalized detection text (lowercased + base64-decoded copies)
    # plus a de-obfuscated copy to defeat simple evasions.
    detection = (norm.detection_text or "").lower()
    deob = _deobfuscate(norm.original_text or "")
    haystacks = (detection, deob)

    for name, patterns, route, is_block in _RULES:
        for pat in patterns:
            if any(pat.search(h) for h in haystacks):
                res.force_route.add(route)
                res.reasons.append(f"rule:{name}")
                if is_block:
                    res.block = True
                break  # one hit per rule is enough

    # Obfuscation signals -> never fast-allow; force the attack route (evasion
    # usually hides an injection). Reuse the normalizer's flags + own detectors.
    f = norm.flags or {}
    obf = []
    if f.get("excessive_zero_width"):
        obf.append("zero_width")
    if f.get("high_non_printable"):
        obf.append("non_printable")
    if f.get("suspicious_base64"):
        obf.append("base64")
    if f.get("excessive_separators"):
        obf.append("separators")
    if _SPACED_RE.search((norm.original_text or "")):
        obf.append("spaced_chars")
    if _HOMOGLYPH_RE.search((norm.original_text or "")):
        obf.append("homoglyphs")
    if obf:
        res.force_route.add(ATTACK)
        res.reasons.append("obfuscation:" + ",".join(obf))

    return res
