"""Manually registered room objects with separate image/map coordinates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from roomsense.calibration.camera_calibration import Point


@dataclass(frozen=True)
class RoomObject:
    id: str
    name: str
    image_region: tuple[Point, ...]
    enabled: bool = True
    room_position: Point | None = None
    interaction_radius: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("room object id cannot be empty")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("room object name cannot be empty")
        try:
            points = tuple(tuple(point) for point in self.image_region)
            if any(len(point) != 2 for point in points):
                raise ValueError("image points must be two-dimensional")
            region = tuple((float(point[0]), float(point[1])) for point in points)
        except (TypeError, ValueError, IndexError, OverflowError) as exc:
            raise ValueError("image_region must contain normalized 2D points") from exc
        if len(region) < 3 or any(
            not math.isfinite(x) or not math.isfinite(y) or not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0
            for x, y in region
        ):
            raise ValueError("image_region requires at least three finite normalized points")
        if len(set(region)) < 3 or _area(region) <= 1e-8 or _self_intersects(region):
            raise ValueError("image_region must be a non-degenerate simple polygon")
        object.__setattr__(self, "image_region", region)

        if self.room_position is not None:
            try:
                raw_position = tuple(self.room_position)
                if len(raw_position) != 2:
                    raise ValueError("room position must be two-dimensional")
                position = (float(raw_position[0]), float(raw_position[1]))
            except (TypeError, ValueError, IndexError, OverflowError) as exc:
                raise ValueError("room_position must be a normalized 2D point") from exc
            if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in position):
                raise ValueError("room_position must be finite normalized coordinates")
            object.__setattr__(self, "room_position", position)

        if self.interaction_radius is not None:
            try:
                radius = float(self.interaction_radius)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("interaction_radius must be finite and between 0 and 1") from exc
            if not math.isfinite(radius) or not 0.0 <= radius <= 1.0:
                raise ValueError("interaction_radius must be finite and between 0 and 1")
            object.__setattr__(self, "interaction_radius", radius)
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean")


class RoomObjectRegistry:
    def __init__(self, objects: Iterable[RoomObject] = ()) -> None:
        self.objects = tuple(objects)
        if any(not isinstance(item, RoomObject) for item in self.objects):
            raise ValueError("room object registry accepts RoomObject entries only")
        ids = [item.id for item in self.objects]
        if len(set(ids)) != len(ids):
            raise ValueError("room object ids must be unique")

    def enabled_objects(self) -> tuple[RoomObject, ...]:
        return tuple(item for item in self.objects if item.enabled)


def _area(polygon: tuple[Point, ...]) -> float:
    return abs(sum(
        polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
        - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
        for index in range(len(polygon))
    )) * 0.5


def _self_intersects(polygon: tuple[Point, ...]) -> bool:
    count = len(polygon)
    for first_index in range(count):
        first_start = polygon[first_index]
        first_end = polygon[(first_index + 1) % count]
        for second_index in range(first_index + 1, count):
            if second_index in (first_index, (first_index + 1) % count) \
                    or (second_index + 1) % count in (first_index, (first_index + 1) % count):
                continue
            if _segments_intersect(first_start, first_end, polygon[second_index], polygon[(second_index + 1) % count]):
                return True
    return False


def _segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    def cross(first: Point, second: Point, third: Point) -> float:
        return (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (third[0] - first[0])

    def on_segment(first: Point, second: Point, point: Point) -> bool:
        return (min(first[0], second[0]) - 1e-9 <= point[0] <= max(first[0], second[0]) + 1e-9
                and min(first[1], second[1]) - 1e-9 <= point[1] <= max(first[1], second[1]) + 1e-9)

    c1, c2, c3, c4 = cross(a, b, c), cross(a, b, d), cross(c, d, a), cross(c, d, b)
    if ((c1 > 1e-9 and c2 < -1e-9 or c1 < -1e-9 and c2 > 1e-9)
            and (c3 > 1e-9 and c4 < -1e-9 or c3 < -1e-9 and c4 > 1e-9)):
        return True
    return (
        abs(c1) <= 1e-9 and on_segment(a, b, c)
        or abs(c2) <= 1e-9 and on_segment(a, b, d)
        or abs(c3) <= 1e-9 and on_segment(c, d, a)
        or abs(c4) <= 1e-9 and on_segment(c, d, b)
    )
