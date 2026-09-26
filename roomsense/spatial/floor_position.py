"""Estimate a smoothed floor-contact point from visible ankle landmarks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from roomsense.spatial.room_transform import RoomTransform
from roomsense.tracking.pose_tracker import Landmark
from roomsense.utils.smoothing import VectorSmoother


@dataclass(frozen=True)
class FloorPosition:
    raw_image_x: float
    raw_image_y: float
    smoothed_image_x: float
    smoothed_image_y: float
    room_x: float
    room_y: float
    render_x: float
    render_y: float
    confidence: float
    in_bounds: bool
    projectable: bool


class FloorPositionTracker:
    def __init__(self, min_visibility: float = 0.5, smoothing_alpha: float = 0.35) -> None:
        if not 0.0 <= min_visibility <= 1.0:
            raise ValueError("min_visibility must be between 0 and 1")
        self.min_visibility = min_visibility
        self._smoother = VectorSmoother(smoothing_alpha)

    def update(
        self,
        landmarks: Mapping[str, Landmark],
        room_transform: RoomTransform,
    ) -> FloorPosition | None:
        feet = [
            landmarks.get(name)
            for name in ("left_ankle", "right_ankle")
        ]
        visible = [
            point for point in feet
            if point is not None
            and math.isfinite(point.visibility)
            and point.visibility > 0.0
            and point.visibility >= self.min_visibility
        ]
        if not visible:
            return None
        total_weight = sum(point.visibility for point in visible)
        raw_x = sum(point.x * point.visibility for point in visible) / total_weight
        raw_y = sum(point.y * point.visibility for point in visible) / total_weight
        smoothed_x, smoothed_y = self._smoother.update((raw_x, raw_y))
        room = room_transform.transform((smoothed_x, smoothed_y))
        confidence = total_weight / len(visible)
        return FloorPosition(
            raw_image_x=raw_x,
            raw_image_y=raw_y,
            smoothed_image_x=smoothed_x,
            smoothed_image_y=smoothed_y,
            room_x=room.room_x,
            room_y=room.room_y,
            render_x=room.clamped_x,
            render_y=room.clamped_y,
            confidence=confidence,
            in_bounds=room.in_bounds,
            projectable=room.projectable,
        )

    def reset(self) -> None:
        self._smoother.reset()
