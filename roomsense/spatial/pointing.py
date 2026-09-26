"""Confidence-gated, smoothed image-space arm pointing estimates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from roomsense.tracking.pose_tracker import Landmark
from roomsense.utils.smoothing import VectorSmoother


@dataclass(frozen=True)
class ArmPointing:
    side: str
    origin: tuple[float, float]
    direction: tuple[float, float]
    extension: float
    confidence: float


class PointingEstimator:
    """Estimate pointing from shoulder, elbow, and wrist image landmarks."""

    def __init__(
        self,
        min_visibility: float = 0.5,
        min_extension: float = 0.8,
        smoothing_alpha: float = 0.35,
    ) -> None:
        if not math.isfinite(min_visibility) or not 0.0 <= min_visibility <= 1.0:
            raise ValueError("min_visibility must be between 0 and 1")
        if not math.isfinite(min_extension) or not 0.0 <= min_extension <= 1.0:
            raise ValueError("min_extension must be between 0 and 1")
        # VectorSmoother validates its alpha when it first receives a value.
        self.min_visibility = min_visibility
        self.min_extension = min_extension
        self.smoothing_alpha = smoothing_alpha
        self._smoothers = {
            side: VectorSmoother(smoothing_alpha) for side in ("left", "right")
        }

    def update(self, landmarks: Mapping[str, Landmark]) -> tuple[ArmPointing, ...]:
        pointing: list[ArmPointing] = []
        for side in ("left", "right"):
            shoulder = landmarks.get(f"{side}_shoulder")
            elbow = landmarks.get(f"{side}_elbow")
            wrist = landmarks.get(f"{side}_wrist")
            if not self._reliable(shoulder, elbow, wrist):
                self._smoothers[side].reset()
                continue

            shoulder_xy = (shoulder.x, shoulder.y)
            elbow_xy = (elbow.x, elbow.y)
            wrist_xy = (wrist.x, wrist.y)
            upper_length = math.dist(shoulder_xy, elbow_xy)
            lower_length = math.dist(elbow_xy, wrist_xy)
            total_length = upper_length + lower_length
            endpoint_x, endpoint_y = wrist.x - shoulder.x, wrist.y - shoulder.y
            endpoint_length = math.hypot(endpoint_x, endpoint_y)
            if total_length <= 1e-9 or endpoint_length <= 1e-9:
                self._smoothers[side].reset()
                continue

            extension = min(1.0, endpoint_length / total_length)
            if extension < self.min_extension:
                self._smoothers[side].reset()
                continue
            raw_direction = (endpoint_x / endpoint_length, endpoint_y / endpoint_length)
            smoothed = self._smoothers[side].update(raw_direction)
            smooth_length = math.hypot(smoothed[0], smoothed[1])
            if smooth_length <= 1e-9 or not math.isfinite(smooth_length):
                self._smoothers[side].reset()
                continue
            direction = (smoothed[0] / smooth_length, smoothed[1] / smooth_length)
            visibility = (shoulder.visibility + elbow.visibility + wrist.visibility) / 3.0
            confidence = max(0.0, min(1.0, visibility * extension))
            pointing.append(ArmPointing(side, shoulder_xy, direction, extension, confidence))
        return tuple(pointing)

    def reset(self) -> None:
        for smoother in self._smoothers.values():
            smoother.reset()

    def _reliable(self, *points: Landmark | None) -> bool:
        return all(
            point is not None
            and math.isfinite(point.x)
            and math.isfinite(point.y)
            and math.isfinite(point.visibility)
            and point.visibility >= self.min_visibility
            for point in points
        )
