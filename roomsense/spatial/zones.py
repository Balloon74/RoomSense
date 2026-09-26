"""Normalized virtual room zones and transition state."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from roomsense.calibration.camera_calibration import Point


@dataclass(frozen=True)
class Zone:
    name: str
    polygon: tuple[Point, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("zone name cannot be empty")
        if len(self.polygon) < 3:
            raise ValueError("zone polygon requires at least three points")
        if any(not math.isfinite(x) or not math.isfinite(y) or not 0 <= x <= 1 or not 0 <= y <= 1
               for x, y in self.polygon):
            raise ValueError("zone polygon points must be finite normalized coordinates")


@dataclass(frozen=True)
class ZoneUpdate:
    entered: tuple[str, ...]
    left: tuple[str, ...]
    current: tuple[str, ...]


class ZoneTracker:
    def __init__(self, zones: Iterable[Zone]) -> None:
        self.zones = tuple(zones)
        names = [zone.name for zone in self.zones]
        if len(set(names)) != len(names):
            raise ValueError("zone names must be unique")
        self._current: tuple[str, ...] = ()

    @property
    def current(self) -> tuple[str, ...]:
        return self._current

    def update(self, point: Point | None, in_bounds: bool = True) -> ZoneUpdate:
        next_current = tuple(
            zone.name for zone in self.zones
            if point is not None and in_bounds and _contains(zone.polygon, point)
        )
        previous = set(self._current)
        following = set(next_current)
        update = ZoneUpdate(
            entered=tuple(name for name in next_current if name not in previous),
            left=tuple(name for name in self._current if name not in following),
            current=next_current,
        )
        self._current = next_current
        return update

    def reset(self) -> None:
        self._current = ()


def _contains(polygon: tuple[Point, ...], point: Point) -> bool:
    x, y = point
    inside = False
    for index, first in enumerate(polygon):
        second = polygon[(index + 1) % len(polygon)]
        cross = (x - first[0]) * (second[1] - first[1]) - (y - first[1]) * (second[0] - first[0])
        if abs(cross) <= 1e-9 and min(first[0], second[0]) - 1e-9 <= x <= max(first[0], second[0]) + 1e-9 \
                and min(first[1], second[1]) - 1e-9 <= y <= max(first[1], second[1]) + 1e-9:
            return True
        if (first[1] > y) != (second[1] > y):
            crossing_x = (second[0] - first[0]) * (y - first[1]) / (second[1] - first[1]) + first[0]
            if x < crossing_x:
                inside = not inside
    return inside
