"""Recent normalized room positions and session movement metrics."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

from roomsense.calibration.camera_calibration import Point


@dataclass(frozen=True)
class PositionSample:
    timestamp: float
    x: float
    y: float


@dataclass(frozen=True)
class MovementMetrics:
    samples: tuple[PositionSample, ...]
    direction: Point
    speed: float
    distance: float


class PositionHistory:
    def __init__(self, retention_seconds: float = 4.0) -> None:
        if not math.isfinite(retention_seconds) or retention_seconds <= 0:
            raise ValueError("retention_seconds must be positive and finite")
        self.retention_seconds = retention_seconds
        self._samples: deque[PositionSample] = deque()
        self.distance = 0.0

    @property
    def samples(self) -> tuple[PositionSample, ...]:
        return tuple(self._samples)

    def update(self, timestamp: float, point: Point) -> MovementMetrics:
        x, y = point
        if not all(math.isfinite(value) for value in (timestamp, x, y)):
            raise ValueError("position timestamp and coordinates must be finite")
        if self._samples and timestamp < self._samples[-1].timestamp:
            raise ValueError("position timestamps must be non-decreasing")
        sample = PositionSample(float(timestamp), float(x), float(y))
        if self._samples:
            previous = self._samples[-1]
            self.distance += math.hypot(sample.x - previous.x, sample.y - previous.y)
        self._samples.append(sample)
        cutoff = timestamp - self.retention_seconds
        while len(self._samples) > 1 and self._samples[1].timestamp <= cutoff:
            self._samples.popleft()
        return self._metrics()

    def clear_tracking(self) -> MovementMetrics:
        self._samples.clear()
        return self._metrics()

    def reset_session(self) -> MovementMetrics:
        self._samples.clear()
        self.distance = 0.0
        return self._metrics()

    def _metrics(self) -> MovementMetrics:
        if len(self._samples) < 2:
            direction = (0.0, 0.0)
            speed = 0.0
        else:
            latest = self._samples[-1]
            window = [sample for sample in self._samples if latest.timestamp - sample.timestamp <= 1.0]
            if len(window) < 2:
                direction = (0.0, 0.0)
                speed = 0.0
            else:
                first = window[0]
                dx, dy = latest.x - first.x, latest.y - first.y
                elapsed = latest.timestamp - first.timestamp
                direction = (dx, dy)
                speed = math.hypot(dx, dy) / elapsed if elapsed > 0 else 0.0
        return MovementMetrics(tuple(self._samples), direction, speed, self.distance)
