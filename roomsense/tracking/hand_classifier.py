"""Pure hand-state classification, temporal confirmation, and body-wrist association."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Mapping, Sequence

from roomsense.tracking.hand_geometry import FingerState, HandObservation, HandState
from roomsense.tracking.pose_tracker import Landmark


PINCH_DISTANCE_RATIO = 0.18
HANDEDNESS_MISMATCH_PENALTY = 0.45
DEFAULT_SHOULDER_WIDTH = 0.20
MIN_WRIST_VISIBILITY = 0.35


def classify_hand(observation: HandObservation) -> HandState | None:
    """Return the most specific supported state, or None for ambiguous geometry."""
    fingers = observation.finger_states
    if observation.pinch_distance is not None \
            and observation.pinch_distance <= PINCH_DISTANCE_RATIO \
            and fingers["thumb"] is not FingerState.UNCERTAIN \
            and fingers["index"] is not FingerState.UNCERTAIN \
            and (fingers["thumb"] is FingerState.EXTENDED or fingers["index"] is FingerState.EXTENDED):
        return HandState.PINCHING
    if all(fingers[name] is FingerState.EXTENDED for name in ("thumb", "index", "middle", "ring", "pinky")):
        return HandState.OPEN_PALM
    if all(fingers[name] is FingerState.CURLED for name in ("thumb", "index", "middle", "ring", "pinky")):
        return HandState.CLOSED_FIST
    if fingers["index"] is FingerState.EXTENDED \
            and all(fingers[name] is FingerState.CURLED for name in ("middle", "ring", "pinky")):
        return HandState.POINTING
    if fingers["index"] is FingerState.EXTENDED and fingers["middle"] is FingerState.EXTENDED \
            and fingers["ring"] is FingerState.CURLED and fingers["pinky"] is FingerState.CURLED:
        return HandState.PEACE_SIGN
    if fingers["thumb"] is FingerState.EXTENDED \
            and all(fingers[name] is FingerState.CURLED for name in ("index", "middle", "ring", "pinky")) \
            and _thumb_points_up(observation):
        return HandState.THUMBS_UP
    return None


def _thumb_points_up(observation: HandObservation) -> bool:
    cmc = observation.landmarks[1]
    tip = observation.landmarks[4]
    palm_size = _distance_2d(observation.landmarks[0], observation.landmarks[9])
    return palm_size > 1e-6 and cmc.y - tip.y >= palm_size * 0.25


@dataclass
class _TemporalState:
    current: HandState | None = None
    pending: HandState | None = None
    pending_count: int = 0


class TemporalHandClassifier:
    """Require consecutive observations before adopting or releasing a state."""

    def __init__(self, confirm_frames: int = 3) -> None:
        if isinstance(confirm_frames, bool) or not isinstance(confirm_frames, int) or confirm_frames <= 0:
            raise ValueError("confirm_frames must be a positive integer")
        self.confirm_frames = confirm_frames
        self._hands: dict[str, _TemporalState] = {}

    def update(self, hand_key: str, candidate: HandState | None) -> HandState | None:
        state = self._hands.setdefault(hand_key, _TemporalState())
        if candidate is state.current:
            state.pending = None
            state.pending_count = 0
            return state.current
        if candidate is state.pending:
            state.pending_count += 1
        else:
            state.pending = candidate
            state.pending_count = 1
        if state.pending_count >= self.confirm_frames:
            state.current = candidate
            state.pending = None
            state.pending_count = 0
        return state.current

    def reset(self, hand_key: str | None = None) -> None:
        if hand_key is None:
            self._hands.clear()
        else:
            self._hands.pop(hand_key, None)


def associate_hands(
    hands: Sequence[HandObservation],
    body_landmarks: Mapping[str, Landmark],
    max_distance_ratio: float = 0.75,
) -> tuple[HandObservation, ...]:
    """Match hands one-to-one to visible pose wrists using distance and handedness."""
    if not math.isfinite(max_distance_ratio) or max_distance_ratio <= 0.0:
        raise ValueError("max_distance_ratio must be finite and positive")
    if not hands:
        return ()

    left_shoulder = body_landmarks.get("left_shoulder")
    right_shoulder = body_landmarks.get("right_shoulder")
    shoulder_width = (
        _distance_2d(left_shoulder, right_shoulder)
        if _visible(left_shoulder) and _visible(right_shoulder) else DEFAULT_SHOULDER_WIDTH
    )
    if not math.isfinite(shoulder_width) or shoulder_width <= 1e-6:
        shoulder_width = DEFAULT_SHOULDER_WIDTH

    wrists = {
        side: body_landmarks.get(f"{side}_wrist")
        for side in ("left", "right")
    }
    ratios: list[dict[str, float]] = []
    for hand in hands:
        hand_wrist = hand.landmarks[0]
        choices: dict[str, float] = {}
        for side, wrist in wrists.items():
            if not _visible(wrist):
                continue
            ratio = _distance_2d(hand_wrist, wrist) / shoulder_width
            if math.isfinite(ratio) and ratio <= max_distance_ratio:
                choices[side] = ratio
        ratios.append(choices)

    assignments: tuple[str | None, ...] = tuple(None for _ in hands)
    best_score: tuple[int, float] | None = None
    options = [tuple([None, *choices.keys()]) for choices in ratios]
    for candidate in product(*options):
        matched = [side for side in candidate if side is not None]
        if len(matched) != len(set(matched)):
            continue
        mismatch_cost = sum(
            HANDEDNESS_MISMATCH_PENALTY
            for hand, side in zip(hands, candidate)
            if side is not None and hand.handedness.casefold() != side
        )
        distance_cost = sum(
            ratios[index][side] for index, side in enumerate(candidate) if side is not None
        )
        score = (-len(matched), distance_cost + mismatch_cost)
        if best_score is None or score < best_score:
            assignments = candidate
            best_score = score

    associated = []
    for hand, side, choices in zip(hands, assignments, ratios):
        if side is None:
            associated.append(hand)
            continue
        confidence = max(0.0, min(1.0, 1.0 - choices[side] / max_distance_ratio))
        if hand.handedness.casefold() != side:
            confidence *= 0.5
        associated.append(_with_association(hand, side, confidence))
    return tuple(associated)


def _with_association(hand: HandObservation, side: str, confidence: float) -> HandObservation:
    from dataclasses import replace

    return replace(hand, body_side=side, association_confidence=confidence)


def _visible(point: Landmark | None) -> bool:
    return point is not None and point.visibility >= MIN_WRIST_VISIBILITY \
        and math.isfinite(point.x) and math.isfinite(point.y)


def _distance_2d(first: Landmark | object, second: Landmark | object) -> float:
    return math.hypot(float(first.x) - float(second.x), float(first.y) - float(second.y))
