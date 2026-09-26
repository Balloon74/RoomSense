"""Expire transient spatial state when reliable floor observations stop."""

from __future__ import annotations

import math


class SpatialObservationMonitor:
    def __init__(self, timeout_seconds: float = 1.0) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
        self.timeout_seconds = timeout_seconds
        self._missing_since: float | None = None
        self._expired = False

    def mark_valid(self) -> None:
        self._missing_since = None
        self._expired = False

    def mark_missing(self, timestamp: float) -> bool:
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if self._missing_since is None:
            self._missing_since = timestamp
        if not self._expired and timestamp - self._missing_since >= self.timeout_seconds:
            self._expired = True
            return True
        return False
