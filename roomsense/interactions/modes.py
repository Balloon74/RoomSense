"""NORMAL/COMMAND interaction mode state with inactivity timeout."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence

from roomsense.gestures.temporal_gestures import GestureTransition


class InteractionMode(str, Enum):
    NORMAL = "NORMAL"
    COMMAND = "COMMAND"


@dataclass(frozen=True)
class ModeTransition:
    previous: InteractionMode
    current: InteractionMode
    timestamp: float
    reason: str


class InteractionModeController:
    def __init__(self, command_timeout_seconds: float = 12.0) -> None:
        if not math.isfinite(command_timeout_seconds) or command_timeout_seconds <= 0.0:
            raise ValueError("command_timeout_seconds must be positive and finite")
        self.command_timeout_seconds = command_timeout_seconds
        self.mode = InteractionMode.NORMAL
        self._last_timestamp: float | None = None
        self._last_activity: float | None = None

    def update(
        self,
        gestures: Sequence[GestureTransition],
        timestamp: float,
        *,
        activity: bool = False,
    ) -> ModeTransition | None:
        if not math.isfinite(timestamp):
            raise ValueError("mode timestamp must be finite")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("mode timestamps must be non-decreasing")
        self._last_timestamp = timestamp
        hands_up = any(gesture.name == "BOTH_HANDS_UP" for gesture in gestures)

        if hands_up:
            if self.mode is InteractionMode.NORMAL:
                return self._transition(InteractionMode.COMMAND, timestamp, "BOTH_HANDS_UP")
            return self._transition(InteractionMode.NORMAL, timestamp, "BOTH_HANDS_UP")

        if self.mode is InteractionMode.COMMAND:
            if activity:
                self._last_activity = timestamp
            elif self._last_activity is not None \
                    and timestamp - self._last_activity >= self.command_timeout_seconds:
                return self._transition(InteractionMode.NORMAL, timestamp, "INACTIVITY_TIMEOUT")
        return None

    def _transition(self, next_mode: InteractionMode, timestamp: float, reason: str) -> ModeTransition:
        previous = self.mode
        self.mode = next_mode
        self._last_activity = timestamp if next_mode is InteractionMode.COMMAND else None
        return ModeTransition(previous, next_mode, timestamp, reason)
