"""Timestamped V3 gesture transitions with short histories and cooldowns."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from roomsense.config import RoomSenseConfig
from roomsense.spatial.pointing import ArmPointing
from roomsense.tracking.pose_tracker import Landmark


@dataclass(frozen=True)
class GestureTransition:
    name: str
    timestamp: float
    confidence: float
    metadata: Mapping[str, object]


class TemporalGestureDetector:
    """Recognize swipes and held poses from multiple time-stamped frames."""

    def __init__(self, config: RoomSenseConfig) -> None:
        self.config = config
        self._last_timestamp: float | None = None
        self._wrist_history: dict[str, deque[tuple[float, float, float, float]]] = {
            side: deque() for side in ("left", "right")
        }
        self._cooldowns: dict[str, float] = {}
        self._hands_up_since: float | None = None
        self._hands_up_emitted = False
        self._point_since: dict[str, float] = {}
        self._point_emitted: set[str] = set()
        self._active_points: set[str] = set()

    @property
    def cooldowns(self) -> Mapping[str, float]:
        """Absolute monotonic timestamps at which each gesture may fire again."""
        return dict(self._cooldowns)

    @property
    def history(self) -> Mapping[str, tuple[tuple[float, float, float, float], ...]]:
        return {side: tuple(samples) for side, samples in self._wrist_history.items()}

    @property
    def active_gestures(self) -> tuple[str, ...]:
        active = ["BOTH_HANDS_UP"] if self._hands_up_emitted else []
        for side in sorted(self._active_points):
            active.append(f"{side.upper()}_POINT")
            if side in self._point_emitted:
                active.append(f"{side.upper()}_HOLD_POINT")
        return tuple(active)

    def update(
        self,
        landmarks: Mapping[str, Landmark] | None,
        pointing: Sequence[ArmPointing],
        timestamp: float,
    ) -> tuple[GestureTransition, ...]:
        if not math.isfinite(timestamp):
            raise ValueError("gesture timestamp must be finite")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("gesture timestamps must be non-decreasing")
        self._last_timestamp = timestamp
        if not landmarks:
            self._reset_continuity()
            return ()

        events: list[GestureTransition] = []
        self._update_swipe_history(landmarks, timestamp)
        swipe = self._swipe_candidate(timestamp)
        if swipe is not None:
            name, confidence, side = swipe
            event = self._emit(name, timestamp, confidence, {"side": side})
            if event is not None:
                events.append(event)

        hands_up = self._both_hands_up(landmarks)
        if not hands_up:
            self._hands_up_since = None
            self._hands_up_emitted = False
        else:
            if self._hands_up_since is None:
                self._hands_up_since = timestamp
            if not self._hands_up_emitted \
                    and timestamp - self._hands_up_since >= self.config.gesture_both_hands_hold_seconds:
                confidence = self._hands_up_confidence(landmarks)
                event = self._emit("BOTH_HANDS_UP", timestamp, confidence, {})
                if event is not None:
                    events.append(event)
                self._hands_up_emitted = True

        active: dict[str, ArmPointing] = {
            arm.side: arm for arm in pointing
            if arm.side in ("left", "right") and math.isfinite(arm.confidence)
            and 0.0 < arm.confidence <= 1.0
        }
        for side, arm in active.items():
            if side not in self._active_points:
                self._point_since[side] = timestamp
                self._point_emitted.discard(side)
                event = self._emit("POINT", timestamp, arm.confidence, {"side": side})
                if event is not None:
                    events.append(event)
            started = self._point_since.setdefault(side, timestamp)
            if side not in self._point_emitted \
                    and timestamp - started >= self.config.gesture_hold_point_seconds:
                event = self._emit("HOLD_POINT", timestamp, arm.confidence, {"side": side})
                if event is not None:
                    events.append(event)
                self._point_emitted.add(side)
        for side in self._active_points - set(active):
            self._point_since.pop(side, None)
            self._point_emitted.discard(side)
        self._active_points = set(active)
        return tuple(events)

    def reset(self) -> None:
        self._last_timestamp = None
        self._cooldowns.clear()
        self._reset_continuity()

    def _update_swipe_history(self, landmarks: Mapping[str, Landmark], timestamp: float) -> None:
        cutoff = timestamp - self.config.gesture_swipe_window_seconds
        for side in ("left", "right"):
            wrist = landmarks.get(f"{side}_wrist")
            samples = self._wrist_history[side]
            if self._valid_landmark(wrist):
                samples.append((timestamp, wrist.x, wrist.y, wrist.visibility))
            else:
                samples.clear()
                continue
            while len(samples) > 2 and samples[0][0] < cutoff:
                samples.popleft()

    def _swipe_candidate(self, timestamp: float) -> tuple[str, float, str] | None:
        candidates: list[tuple[str, float, str, float]] = []
        for side, samples in self._wrist_history.items():
            if len(samples) < 2:
                continue
            first, last = samples[0], samples[-1]
            elapsed = last[0] - first[0]
            displacement = last[1] - first[1]
            if elapsed <= 0.0 or elapsed > self.config.gesture_swipe_window_seconds:
                continue
            if abs(displacement) < self.config.gesture_swipe_distance:
                continue
            name = "SWIPE_RIGHT" if displacement > 0.0 else "SWIPE_LEFT"
            confidence = min(first[3], last[3]) * min(
                1.0, abs(displacement) / self.config.gesture_swipe_distance
            )
            candidates.append((name, confidence, side, abs(displacement)))
        if not candidates:
            return None
        name, confidence, side, _ = max(candidates, key=lambda item: (item[3], item[1], item[2]))
        return name, confidence, side

    def _both_hands_up(self, landmarks: Mapping[str, Landmark]) -> bool:
        for side in ("left", "right"):
            shoulder, wrist = landmarks.get(f"{side}_shoulder"), landmarks.get(f"{side}_wrist")
            if not self._valid_landmark(shoulder) or not self._valid_landmark(wrist):
                return False
            if wrist.y >= shoulder.y - self.config.raised_hand_margin:
                return False
        return True

    def _hands_up_confidence(self, landmarks: Mapping[str, Landmark]) -> float:
        return min(landmarks[f"{side}_{name}"].visibility for side in ("left", "right")
                   for name in ("shoulder", "wrist"))

    def _valid_landmark(self, landmark: Landmark | None) -> bool:
        return bool(
            landmark
            and math.isfinite(landmark.x)
            and math.isfinite(landmark.y)
            and math.isfinite(landmark.visibility)
            and landmark.visibility >= self.config.landmark_visibility
        )

    def _emit(
        self, name: str, timestamp: float, confidence: float, metadata: Mapping[str, object]
    ) -> GestureTransition | None:
        if timestamp < self._cooldowns.get(name, float("-inf")):
            return None
        self._cooldowns[name] = timestamp + self.config.gesture_cooldown_seconds
        return GestureTransition(name, timestamp, max(0.0, min(1.0, confidence)), dict(metadata))

    def _reset_continuity(self) -> None:
        for samples in self._wrist_history.values():
            samples.clear()
        self._hands_up_since = None
        self._hands_up_emitted = False
        self._point_since.clear()
        self._point_emitted.clear()
        self._active_points.clear()
