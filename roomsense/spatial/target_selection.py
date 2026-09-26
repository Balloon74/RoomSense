"""Stable object targeting from smoothed camera-image pointing vectors."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from roomsense.spatial.pointing import ArmPointing
from roomsense.spatial.room_objects import RoomObject, RoomObjectRegistry
from roomsense.calibration.camera_calibration import Point


@dataclass(frozen=True)
class TargetMatch:
    object: RoomObject
    side: str
    confidence: float
    ray_distance: float


@dataclass(frozen=True)
class TargetUpdate:
    candidate: TargetMatch | None
    confirmed: TargetMatch | None
    held: TargetMatch | None
    confirmed_now: bool = False
    held_now: bool = False


class TargetSelector:
    def __init__(
        self,
        registry: RoomObjectRegistry,
        stability_seconds: float = 0.35,
        hold_seconds: float = 0.8,
    ) -> None:
        if any(not math.isfinite(value) or value < 0 for value in (stability_seconds, hold_seconds)):
            raise ValueError("target durations must be finite and non-negative")
        self.registry = registry
        self.stability_seconds = stability_seconds
        self.hold_seconds = hold_seconds
        self._last_timestamp: float | None = None
        self._pending_id: str | None = None
        self._pending_since: float | None = None
        self._confirmed: TargetMatch | None = None
        self._confirmed_since: float | None = None
        self._held_emitted = False

    def update(self, pointing: Sequence[ArmPointing], timestamp: float) -> TargetUpdate:
        if not math.isfinite(timestamp):
            raise ValueError("target timestamp must be finite")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("target timestamps must be non-decreasing")
        self._last_timestamp = timestamp

        candidate = self._candidate(pointing)
        if candidate is None:
            self._clear()
            return TargetUpdate(None, None, None)

        if self._confirmed is not None and candidate.object.id == self._confirmed.object.id:
            self._confirmed = candidate
            self._pending_id = None
            self._pending_since = None
            assert self._confirmed_since is not None
            held_now = not self._held_emitted and timestamp - self._confirmed_since >= self.hold_seconds
            if held_now:
                self._held_emitted = True
            return TargetUpdate(candidate, self._confirmed, self._confirmed if self._held_emitted else None,
                                held_now=held_now)

        if candidate.object.id != self._pending_id:
            self._pending_id = candidate.object.id
            self._pending_since = timestamp
        confirmed_now = timestamp - (self._pending_since if self._pending_since is not None else timestamp) \
            >= self.stability_seconds
        if confirmed_now:
            self._confirmed = candidate
            self._confirmed_since = timestamp
            self._pending_id = None
            self._pending_since = None
            self._held_emitted = False
            held_now = self.hold_seconds == 0.0
            if held_now:
                self._held_emitted = True
            return TargetUpdate(candidate, candidate, candidate if held_now else None,
                                confirmed_now=True, held_now=held_now)
        return TargetUpdate(candidate, self._confirmed, None)

    def _candidate(self, pointing: Sequence[ArmPointing]) -> TargetMatch | None:
        candidates: list[TargetMatch] = []
        for arm in pointing:
            if not _valid_arm(arm):
                continue
            for item in self.registry.enabled_objects():
                distance = _ray_polygon_distance(arm.origin, arm.direction, item.image_region)
                radius = item.interaction_radius or 0.0
                if distance is None:
                    continue
                ray_distance, miss_distance = distance
                if miss_distance > radius:
                    continue
                confidence = arm.confidence
                if radius > 0.0 and miss_distance > 0.0:
                    confidence *= max(0.0, 1.0 - miss_distance / radius)
                candidates.append(TargetMatch(item, arm.side, confidence, ray_distance))
        return min(candidates, key=lambda match: (
            match.ray_distance, -match.confidence, match.object.id, match.side,
        )) if candidates else None

    def _clear(self) -> None:
        self._pending_id = None
        self._pending_since = None
        self._confirmed = None
        self._confirmed_since = None
        self._held_emitted = False


def _valid_arm(arm: ArmPointing) -> bool:
    values = (*arm.origin, *arm.direction, arm.confidence)
    length = math.hypot(*arm.direction)
    return (
        all(math.isfinite(value) for value in values)
        and 0.0 <= arm.confidence <= 1.0
        and length > 1e-9
    )


def _ray_polygon_distance(origin: Point, direction: Point, polygon: tuple[Point, ...]) -> tuple[float, float] | None:
    """Return nearest forward ray distance and miss distance, if within the image."""

    magnitude = math.hypot(*direction)
    dx, dy = direction[0] / magnitude, direction[1] / magnitude
    if _contains(polygon, origin):
        return 0.0, 0.0
    intersections: list[float] = []
    nearest: tuple[float, float] | None = None
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        sx, sy = end[0] - start[0], end[1] - start[1]
        ox, oy = start[0] - origin[0], start[1] - origin[1]
        denominator = dx * sy - dy * sx
        if abs(denominator) > 1e-10:
            ray_t = (ox * sy - oy * sx) / denominator
            segment_t = (ox * dy - oy * dx) / denominator
            if ray_t >= 0.0 and 0.0 <= segment_t <= 1.0:
                intersections.append(ray_t)

        # Closest polygon endpoint to the forward ray.
        for vertex in (start, end):
            projected = max(0.0, (vertex[0] - origin[0]) * dx + (vertex[1] - origin[1]) * dy)
            closest = (origin[0] + projected * dx, origin[1] + projected * dy)
            miss = math.dist(vertex, closest)
            if nearest is None or (miss, projected) < (nearest[1], nearest[0]):
                nearest = (projected, miss)
        # The ray origin may be closest to the middle of an edge.
        segment_length_sq = sx * sx + sy * sy
        if segment_length_sq > 1e-12:
            fraction = max(0.0, min(1.0, ((origin[0] - start[0]) * sx + (origin[1] - start[1]) * sy)
                                      / segment_length_sq))
            closest = (start[0] + fraction * sx, start[1] + fraction * sy)
            miss = math.dist(origin, closest)
            if nearest is None or (miss, 0.0) < (nearest[1], nearest[0]):
                nearest = (0.0, miss)
    if intersections:
        return min(intersections), 0.0
    return nearest


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
