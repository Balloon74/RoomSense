"""Apply a saved floor-plane calibration to normalized image points."""

from __future__ import annotations

from dataclasses import dataclass

from roomsense.calibration.camera_calibration import (
    Calibration,
    Point,
    ProjectiveHorizonError,
    homography_for,
    transform_point,
)


@dataclass(frozen=True)
class RoomPoint:
    room_x: float
    room_y: float
    clamped_x: float
    clamped_y: float
    in_bounds: bool
    projectable: bool = True


class RoomTransform:
    def __init__(self, calibration: Calibration) -> None:
        self.calibration = calibration
        self._homography = homography_for(calibration)

    def transform(self, image_point: Point) -> RoomPoint:
        try:
            room_x, room_y = transform_point(image_point, self._homography)
        except ProjectiveHorizonError:
            return RoomPoint(0.0, 0.0, 0.5, 0.5, False, False)
        return RoomPoint(
            room_x=room_x,
            room_y=room_y,
            clamped_x=max(0.0, min(1.0, room_x)),
            clamped_y=max(0.0, min(1.0, room_y)),
            in_bounds=_contains(self.calibration.image_points, image_point),
            projectable=True,
        )


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
