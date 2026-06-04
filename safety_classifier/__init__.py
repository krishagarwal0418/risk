"""Local AI safety classification system.

Public entrypoint::

    from safety_classifier import SafetyClassifier
    clf = SafetyClassifier(device="cuda")
    print(clf.classify("Hello"))
"""

from __future__ import annotations

from . import constants
from .classifier import SafetyClassifier
from .schemas import (
    FastTextResult,
    ModelResult,
    SafetyRequest,
    SafetyResult,
)

__version__ = "0.1.0"

__all__ = [
    "SafetyClassifier",
    "SafetyRequest",
    "SafetyResult",
    "ModelResult",
    "FastTextResult",
    "constants",
]
