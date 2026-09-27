"""Temporal, camera-only recognition of hand commands for macOS controls."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence

from roomsense.actions.action_registry import ActionIntent, MacAction
from roomsense.config import RoomSenseConfig
from roomsense.tracking.hand_tracker import TrackedHand


class HandShape(str, Enum):
    OPEN_PALM = "OPEN_PALM"
    PINCH = "PINCH"
    FIST = "FIST"
    OTHER = "OTHER"


@dataclass(frozen=True)
class HandClassification:
    shape: HandShape
    confidence: float


@dataclass(frozen=True)
class GestureStatus:
    gesture: str | None = None
    confidence: float = 0.0
    cooldown_seconds: float = 0.0
    cancelled: bool = False
    actions: tuple[ActionIntent, ...] = ()


def _distance(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def classify_hand(hand: TrackedHand, config: RoomSenseConfig) -> HandClassification:
    """Classify normalized MediaPipe landmarks using palm-relative distances."""
    points = hand.landmarks
    wrist = points[0]
    palm_scale = _distance(wrist, points[9])
    if palm_scale < 1e-5:
        return HandClassification(HandShape.OTHER, 0.0)

    pairs = ((8, 6), (12, 10), (16, 14), (20, 18))
    extended = [
        _distance(wrist, points[tip]) > _distance(wrist, points[pip]) + 0.12 * palm_scale
        for tip, pip in pairs
    ]
    extended_count = sum(extended)
    pinch_ratio = _distance(points[4], points[8]) / palm_scale
    middle_fingers_extended = sum(extended[1:]) >= 2

    # A closed fist has most fingertips curled toward the palm. Give it priority
    # over pinch, whose index finger is intentionally folded inward.
    if extended_count <= 1:
        shape = HandShape.FIST
    elif pinch_ratio <= config.gesture_pinch_ratio and middle_fingers_extended:
        shape = HandShape.PINCH
    else:
        thumb_extended = _distance(points[4], points[5]) > 0.32 * palm_scale
        shape = HandShape.OPEN_PALM if extended_count == 4 and thumb_extended else HandShape.OTHER
    return HandClassification(shape, hand.confidence)


class HandGestureController:
    """Turn timestamped hand observations into debounced action intents."""

    def __init__(self, config: RoomSenseConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._last_timestamp: float | None = None
        self._owner: str | None = None
        self._shape: HandShape | None = None
        self._shape_started: float | None = None
        self._shape_samples = 0
        self._held_action_fired = False
        self._fist_cancelled = False
        self._last_action_time: float | None = None
        self._swipe: deque[tuple[float, float]] = deque()
        self._pinch_anchor_y: float | None = None
        self._last_volume_action_time: float | None = None
        self._last_confidence = 0.0

    def update(
        self, hands: Sequence[TrackedHand], timestamp: float, command_mode: bool
    ) -> GestureStatus:
        if not math.isfinite(timestamp):
            raise ValueError("gesture timestamp must be finite")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("gesture timestamps must be non-decreasing")
        self._last_timestamp = timestamp
        if not command_mode:
            self._clear_pending()
            self._owner = None
            return GestureStatus()

        active = {hand.handedness: hand for hand in hands
                  if hand.confidence >= self.config.gesture_min_confidence}
        hand = active.get(self._owner) if self._owner else None
        if hand is None:
            self._clear_pending()
            self._owner = None
            if active:
                # Stable deterministic ownership: strongest observation, then left.
                hand = sorted(active.values(), key=lambda item: (-item.confidence, item.handedness))[0]
                self._owner = hand.handedness
        if hand is None:
            return self._status(timestamp)

        classification = classify_hand(hand, self.config)
        if classification.confidence < self.config.gesture_min_confidence:
            self._clear_pending()
            self._owner = None
            return self._status(timestamp)
        shape = classification.shape
        self._last_confidence = classification.confidence

        if shape == HandShape.FIST:
            self._swipe.clear()
            self._pinch_anchor_y = None
            if self._shape != HandShape.FIST:
                self._set_shape(shape, timestamp)
                self._fist_cancelled = False
            self._shape_samples += 1
            cancelled = False
            if not self._fist_cancelled and self._duration(timestamp) >= self.config.gesture_fist_hold_seconds \
                    and self._shape_samples >= self.config.gesture_min_samples:
                self._fist_cancelled = True
                cancelled = True
            return self._status(timestamp, cancelled=cancelled)

        if self._shape == HandShape.FIST:
            self._fist_cancelled = False
        if shape != self._shape:
            self._set_shape(shape, timestamp)
        else:
            self._shape_samples += 1

        if shape == HandShape.PINCH:
            return self._update_pinch(hand, timestamp, classification.confidence)

        self._pinch_anchor_y = None
        if shape == HandShape.OPEN_PALM:
            wrist_x = hand.landmarks[0].x
            self._swipe.append((timestamp, wrist_x))
            while self._swipe and timestamp - self._swipe[0][0] > self.config.gesture_swipe_window_seconds:
                self._swipe.popleft()
            if len(self._swipe) >= self.config.gesture_min_samples:
                elapsed = self._swipe[-1][0] - self._swipe[0][0]
                distance = self._swipe[-1][1] - self._swipe[0][1]
                if elapsed >= self.config.gesture_swipe_min_duration_seconds \
                        and abs(distance) >= self.config.gesture_swipe_distance:
                    action = MacAction.NEXT_TRACK if distance > 0 else MacAction.PREVIOUS_TRACK
                    label = "SWIPE_RIGHT" if distance > 0 else "SWIPE_LEFT"
                    self._swipe.clear()
                    self._held_action_fired = True
                    actions = self._emit(action, label, timestamp, classification.confidence)
                    return self._status(timestamp, gesture=label, actions=actions)
            if not self._held_action_fired \
                    and self._duration(timestamp) >= self.config.gesture_open_palm_hold_seconds \
                    and self._shape_samples >= self.config.gesture_min_samples:
                self._held_action_fired = True
                actions = self._emit(MacAction.PLAY_PAUSE, "OPEN_PALM", timestamp,
                                     classification.confidence)
                return self._status(timestamp, actions=actions)
        else:
            self._swipe.clear()
        return self._status(timestamp)

    def _update_pinch(self, hand: TrackedHand, timestamp: float, confidence: float) -> GestureStatus:
        if self._shape_samples >= self.config.gesture_min_samples \
                and self._duration(timestamp) >= self.config.gesture_pinch_hold_seconds:
            wrist_y = hand.landmarks[0].y
            if self._pinch_anchor_y is None:
                self._pinch_anchor_y = wrist_y
            else:
                delta = wrist_y - self._pinch_anchor_y
                if abs(delta) >= self.config.gesture_volume_movement_threshold:
                    interval = self.config.gesture_volume_update_interval_seconds
                    ready = self._last_volume_action_time is None \
                        or timestamp - self._last_volume_action_time >= interval
                    if ready:
                        action = MacAction.VOLUME_UP if delta < 0 else MacAction.VOLUME_DOWN
                        actions = self._emit(action, "PINCH", timestamp, confidence)
                        if actions:
                            self._last_volume_action_time = timestamp
                            direction = -1 if delta < 0 else 1
                            self._pinch_anchor_y += direction * self.config.gesture_volume_movement_threshold
                        return self._status(timestamp, gesture="PINCH", actions=actions)
        return self._status(timestamp, gesture="PINCH")

    def _emit(self, action: MacAction, gesture: str, timestamp: float,
              confidence: float) -> tuple[ActionIntent, ...]:
        is_volume = action in (MacAction.VOLUME_UP, MacAction.VOLUME_DOWN)
        if not is_volume and self._last_action_time is not None \
                and timestamp - self._last_action_time < self.config.gesture_cooldown_seconds:
            return ()
        if not is_volume:
            self._last_action_time = timestamp
        return (ActionIntent(action, gesture, timestamp, confidence),)

    def _status(self, timestamp: float, gesture: str | None = None,
                actions: tuple[ActionIntent, ...] = (), cancelled: bool = False) -> GestureStatus:
        selected = gesture or (self._shape.value if self._shape else None)
        return GestureStatus(selected, self._last_confidence,
                             self._cooldown_remaining(timestamp), cancelled, actions)

    def _cooldown_remaining(self, timestamp: float) -> float:
        if self._last_action_time is None:
            return 0.0
        return max(0.0, self.config.gesture_cooldown_seconds - (timestamp - self._last_action_time))

    def _duration(self, timestamp: float) -> float:
        return 0.0 if self._shape_started is None else timestamp - self._shape_started

    def _set_shape(self, shape: HandShape, timestamp: float) -> None:
        self._shape = shape
        self._shape_started = timestamp
        self._shape_samples = 1
        self._held_action_fired = False
        self._swipe.clear()
        self._pinch_anchor_y = None

    def _clear_pending(self) -> None:
        self._shape = None
        self._shape_started = None
        self._shape_samples = 0
        self._held_action_fired = False
        self._fist_cancelled = False
        self._swipe.clear()
        self._pinch_anchor_y = None
        self._last_confidence = 0.0
