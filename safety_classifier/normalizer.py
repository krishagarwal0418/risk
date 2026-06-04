"""Text normalization for the safety classifier.

Produces a :class:`NormalizedText` object that separates:

* ``model_text``     -> fed to transformer models (keeps case + punctuation)
* ``detection_text`` -> lowercased; obfuscation-decoded copies appended

The normalization pipeline is intentionally *single-pass*: HTML and URL decoding
happen exactly once (never recursively) to avoid being tricked into infinite or
adversarial decode loops.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
import unicodedata
import urllib.parse
from dataclasses import dataclass, field
from html import unescape

# Zero-width / invisible characters that should be stripped entirely.
_ZERO_WIDTH = {
    "​",  # zero width space
    "‌",  # zero width non-joiner
    "‍",  # zero width joiner
    "﻿",  # zero width no-break space / BOM
}
_ZERO_WIDTH_RE = re.compile("[" + "".join(_ZERO_WIDTH) + "]")

# Unicode tag characters U+E0000 .. U+E007F (used in tag-based obfuscation).
_TAG_CHARS_RE = re.compile(r"[\U000E0000-\U000E007F]")

_WHITESPACE_RE = re.compile(r"\s+")

# Heuristic base64 span: long-ish, base64 alphabet, optional '=' padding.
# A trailing \b would exclude the '=' padding (since '=' is non-word), so we use
# explicit lookbehind/lookahead boundaries instead.
_BASE64_SPAN_RE = re.compile(
    r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{24,}={0,2}(?![A-Za-z0-9+/=])"
)

# Excessive repeated separator characters, e.g. "----------" or "==========".
_REPEAT_SEP_RE = re.compile(r"([\-=_*~#.])\1{9,}")

_MAX_BASE64_DECODED_CHARS = 512
_MAX_BASE64_SPANS = 4


@dataclass
class NormalizedText:
    """Normalized representation of an input string."""

    original_text: str
    model_text: str
    detection_text: str
    text_hash: str
    flags: dict[str, object] = field(default_factory=dict)


def _strip_invisible(text: str) -> tuple[str, int, int]:
    """Remove zero-width and tag characters. Returns (clean, zw_count, tag_count)."""
    zw_count = len(_ZERO_WIDTH_RE.findall(text))
    tag_count = len(_TAG_CHARS_RE.findall(text))
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _TAG_CHARS_RE.sub("", text)
    return text, zw_count, tag_count


def _non_printable_ratio(text: str) -> float:
    if not text:
        return 0.0
    non_printable = sum(
        1
        for ch in text
        if not ch.isprintable() and ch not in ("\n", "\t", "\r", " ")
    )
    return non_printable / len(text)


def _maybe_decode_base64(spans: list[str]) -> list[str]:
    """Decode high-confidence base64 spans, capping count and decoded length."""
    decoded: list[str] = []
    for span in spans[:_MAX_BASE64_SPANS]:
        if len(span) % 4 != 0:
            continue
        try:
            raw = base64.b64decode(span, validate=True)
        except (binascii.Error, ValueError):
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        # Require the decoded output to look like text (mostly printable).
        printable = sum(1 for ch in text if ch.isprintable() or ch in " \n\t")
        if not text or printable / len(text) < 0.8:
            continue
        decoded.append(text[:_MAX_BASE64_DECODED_CHARS])
    return decoded


def normalize(text: str) -> NormalizedText:
    """Normalize ``text`` into a :class:`NormalizedText`.

    Raises ``ValueError`` for empty / whitespace-only input.
    """
    if text is None or not str(text).strip():
        raise ValueError("Input text is empty or whitespace-only")

    original = str(text)
    flags: dict[str, object] = {}

    # 2. Unicode NFKC normalization.
    work = unicodedata.normalize("NFKC", original)

    # 3 + 4. Remove zero-width and tag characters.
    work, zw_count, tag_count = _strip_invisible(work)
    flags["zero_width_removed"] = zw_count
    flags["tag_chars_removed"] = tag_count
    flags["excessive_zero_width"] = zw_count >= 5

    # 5. Decode HTML entities exactly once.
    work = unescape(work)

    # 6. URL-decode exactly once.
    work = urllib.parse.unquote(work)

    # Non-printable ratio (computed before whitespace collapse).
    npr = _non_printable_ratio(work)
    flags["non_printable_ratio"] = round(npr, 4)
    flags["high_non_printable"] = npr > 0.15

    # Excessive repeated separators.
    flags["excessive_separators"] = bool(_REPEAT_SEP_RE.search(work))

    # 7. Collapse whitespace.
    work = _WHITESPACE_RE.sub(" ", work).strip()

    # 8. model_text keeps case + punctuation.
    model_text = work

    # 9. detection_text is lowercased.
    detection_text = work.lower()

    # Suspicious base64 spans -> decode only high-confidence ones; append to
    # detection_text only, never mutate model_text.
    spans = _BASE64_SPAN_RE.findall(work)
    flags["base64_span_count"] = len(spans)
    flags["suspicious_base64"] = len(spans) > 0
    decoded_spans = _maybe_decode_base64(spans)
    if decoded_spans:
        flags["base64_decoded_count"] = len(decoded_spans)
        detection_text = detection_text + " " + " ".join(s.lower() for s in decoded_spans)

    # 10. sha256 of model_text.
    text_hash = hashlib.sha256(model_text.encode("utf-8")).hexdigest()

    return NormalizedText(
        original_text=original,
        model_text=model_text,
        detection_text=detection_text,
        text_hash=text_hash,
        flags=flags,
    )
