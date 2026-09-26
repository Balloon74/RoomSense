"""Recent movement classification derived from smoothed position history."""

from __future__ import annotations

from collections import deque
from enum import Enum
from time import monotonic


class Movement(str, Enum):
    STILL = "STILL"
    MOVING_LEFT = "MOVING LEFT"
    MOVING_RIGHT = "MOVING RIGHT"
    MOVING_TOWARD = "MOVING TOWARD CAMERA"
    MOVING_AWAY = "MOVING AWAY FROM CAMERA"


class MovementTracker:
    def __init__(self, window_seconds: float = 0.7, horizontal_threshold: float = 0.08,
                 depth_threshold: float = 0.08) -> None:
        self.window_seconds = window_seconds
        self.horizontal_threshold = horizontal_threshold
        self.depth_threshold = depth_threshold
        self._history: deque[tuple[float, float, float]] = deque()
        self.current = Movement.STILL

    def update(self, x: float, y: float, z: float, now: float | None = None) -> Movement:
        timestamp = monotonic() if now is None else now
        self._history.append((timestamp, x, z))
        cutoff = timestamp - self.window_seconds
        while len(self._history) > 2 and self._history[1][0] < cutoff:
            self._history.popleft()
        if len(self._history) < 2:
            self.current = Movement.STILL
            return self.current
        _, old_x, old_z = self._history[0]
        delta_x, delta_z = x - old_x, z - old_z
        if abs(delta_x) >= self.horizontal_threshold:
            self.current = Movement.MOVING_RIGHT if delta_x > 0 else Movement.MOVING_LEFT
        elif abs(delta_z) >= self.depth_threshold:
            self.current = Movement.MOVING_TOWARD if delta_z > 0 else Movement.MOVING_AWAY
        else:
            self.current = Movement.STILL
        return self.current

    def reset(self) -> None:
        self._history.clear()
        self.current = Movement.STILL
