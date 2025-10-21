"""Detectors that rely on the attempt notes to detect a probe hit / LLM failure."""

from typing import List

from garak.detectors.base import Detector
from garak.attempt import Attempt

class MatchSuccessBool(Detector):
    """Check if the `success` boolean is True in the attempt notes.
    Return 1.0 if True, 0.0 if False or not present."""

    def detect(self, attempt: Attempt) -> List[float]:
        success = attempt.notes.get("success") if attempt.notes else None
        return [1.0 if success is True else 0.0]