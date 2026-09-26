"""Estimate normalized screen position and relative monocular depth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from roomsense.config import RoomSenseConfig
from roomsense.tracking.pose_tracker import Landmark
from roomsense.utils.smoothing import VectorSmoother


@dataclass(frozen=True)
class Position:
    x: float
    y: float
    z: float
    confidence: float


class PositionTracker:
    def __init__(self, config: RoomSenseConfig) -> None:
        self.config = config
        self._smoother = VectorSmoother(config.smoothing_alpha)

    def update(self, landmarks: Mapping[str, Landmark]) -> Position:
        shoulders = self._pair(landmarks, "left_shoulder", "right_shoulder")
        hips = self._pair(landmarks, "left_hip", "right_hip")
        if shoulders is None and hips is None:
            raise ValueError("At least one shoulder or hip pair is required")

        centers: list[tuple[float, float]] = []
        if shoulders is not None:
            centers.append(self._midpoint(shoulders))
        if hips is not None:
            centers.append(self._midpoint(hips))
        center_x = sum(point[0] for point in centers) / len(centers)
        center_y = sum(point[1] for point in centers) / len(centers)
        x = self._clamp(2.0 * center_x - 1.0)
        y = self._clamp(1.0 - 2.0 * center_y)

        scale = None
        reference = self.config.reference_shoulder_width
        if shoulders is not None:
            scale = abs(shoulders[1].x - shoulders[0].x)
        elif hips is not None:
            scale = abs(hips[1].x - hips[0].x) * 0.8
            reference *= 0.8
        depth = 0.0 if scale is None or scale <= 1e-4 else self._clamp(
            self.config.depth_sensitivity * (scale / reference - 1.0)
        )
        confidence = sum(point.visibility for pair in (shoulders, hips) if pair for point in pair) / max(
            1, sum(1 for pair in (shoulders, hips) if pair for _ in pair)
        )
        smoothed_x, smoothed_y, smoothed_z = self._smoother.update((x, y, depth))
        return Position(smoothed_x, smoothed_y, smoothed_z, confidence)

    def reset(self) -> None:
        self._smoother.reset()

    @staticmethod
    def _pair(landmarks: Mapping[str, Landmark], left: str, right: str) -> tuple[Landmark, Landmark] | None:
        if left not in landmarks or right not in landmarks:
            return None
        return landmarks[left], landmarks[right]

    @staticmethod
    def _midpoint(pair: tuple[Landmark, Landmark]) -> tuple[float, float]:
        return (pair[0].x + pair[1].x) / 2.0, (pair[0].y + pair[1].y) / 2.0

    @staticmethod
    def _clamp(value: float) -> float:
        return max(-1.0, min(1.0, value))
