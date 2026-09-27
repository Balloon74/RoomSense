"""Camera-independent geometry and immutable values for MediaPipe hand landmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import math
from types import MappingProxyType
from typing import Mapping, Sequence


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
FINGERTIP_INDICES = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
FINGER_JOINT_INDICES = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
PALM_ANCHOR_INDICES = (0, 5, 9, 13, 17)

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

Point3 = tuple[float, float, float]


class FingerState(StrEnum):
    EXTENDED = "extended"
    CURLED = "curled"
    UNCERTAIN = "uncertain"


class HandState(StrEnum):
    OPEN_PALM = "OPEN PALM"
    CLOSED_FIST = "CLOSED FIST"
    POINTING = "POINTING"
    PEACE_SIGN = "PEACE SIGN"
    THUMBS_UP = "THUMBS UP"
    PINCHING = "PINCHING"


@dataclass(frozen=True)
class HandPoint:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


@dataclass(frozen=True)
class HandObservation:
    """One hand's model points and geometry, in normalized mirrored-frame coordinates."""

    landmarks: tuple[HandPoint, ...]
    handedness: str
    tracking_confidence: float
    palm_center: Point3 | None
    palm_angle: float | None
    palm_normal: Point3 | None
    fingertips: Mapping[str, HandPoint]
    finger_states: Mapping[str, FingerState]
    pinch_distance: float | None
    openness: float
    state: HandState | None = None
    body_side: str | None = None
    association_confidence: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "landmarks", tuple(self.landmarks))
        object.__setattr__(self, "fingertips", MappingProxyType(dict(self.fingertips)))
        object.__setattr__(self, "finger_states", MappingProxyType(dict(self.finger_states)))


def analyze_hand(
    points: Sequence[HandPoint], handedness: str, tracking_confidence: float,
) -> HandObservation:
    """Calculate palm, finger, pinch, and openness values from 21 landmarks."""
    if len(points) != 21:
        raise ValueError("a hand must contain exactly 21 landmarks")
    landmarks = tuple(points)
    fingertips = {name: landmarks[index] for name, index in FINGERTIP_INDICES.items()}
    invalid = any(
        not all(math.isfinite(value) for value in (point.x, point.y, point.z, point.visibility))
        or point.visibility <= 0.0
        for point in landmarks
    )
    palm_center = _mean_point(landmarks[index] for index in PALM_ANCHOR_INDICES) if not invalid else None
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    palm_size = _distance(wrist, middle_mcp) if not invalid else 0.0
    valid_palm = not invalid and math.isfinite(palm_size) and palm_size > 1e-6

    index_mcp = landmarks[5]
    pinky_mcp = landmarks[17]
    index_vector = _vector(wrist, index_mcp)
    pinky_vector = _vector(wrist, pinky_mcp)
    index_length = _magnitude(index_vector)
    pinky_length = _magnitude(pinky_vector)
    basis_area = _magnitude(_cross(index_vector, pinky_vector))
    valid_basis = (
        valid_palm
        and index_length > palm_size * 0.08
        and pinky_length > palm_size * 0.08
        and basis_area > index_length * pinky_length * 0.05
    )

    if not valid_basis:
        finger_states = {name: FingerState.UNCERTAIN for name in FINGER_NAMES}
        return HandObservation(
            landmarks, handedness, _confidence(tracking_confidence), None, None, None,
            fingertips, finger_states, None, 0.0,
        )

    palm_normal = _unit(_cross(index_vector, pinky_vector))
    palm_angle = math.atan2(middle_mcp.y - wrist.y, middle_mcp.x - wrist.x)

    states = {
        name: _finger_state(landmarks, name, palm_size)
        for name in FINGER_NAMES
    }
    extended_count = sum(state is FingerState.EXTENDED for state in states.values())
    openness = extended_count / len(FINGER_NAMES)
    pinch_distance = _distance(landmarks[4], landmarks[8]) / palm_size
    if not math.isfinite(pinch_distance):
        pinch_distance = None

    return HandObservation(
        landmarks=landmarks,
        handedness=handedness,
        tracking_confidence=_confidence(tracking_confidence),
        palm_center=palm_center,
        palm_angle=palm_angle,
        palm_normal=palm_normal,
        fingertips=fingertips,
        finger_states=states,
        pinch_distance=pinch_distance,
        openness=openness,
    )


def _finger_state(points: tuple[HandPoint, ...], name: str, palm_size: float) -> FingerState:
    if name == "thumb":
        base, _, ip, tip = (points[index] for index in FINGER_JOINT_INDICES[name])
        extension = _distance(tip, base) - _distance(ip, base)
    else:
        base, pip, _, tip = (points[index] for index in FINGER_JOINT_INDICES[name])
        extension = _distance(tip, points[0]) - _distance(pip, points[0])
    if not math.isfinite(extension):
        return FingerState.UNCERTAIN
    if extension >= palm_size * 0.08:
        return FingerState.EXTENDED
    if extension <= -palm_size * 0.04:
        return FingerState.CURLED
    return FingerState.UNCERTAIN


def _confidence(value: float) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return max(0.0, min(1.0, confidence)) if math.isfinite(confidence) else 0.0


def _mean_point(points: Sequence[HandPoint] | object) -> Point3:
    values = tuple(points)  # type: ignore[arg-type]
    count = len(values)
    return (
        sum(point.x for point in values) / count,
        sum(point.y for point in values) / count,
        sum(point.z for point in values) / count,
    )


def _distance(first: HandPoint, second: HandPoint) -> float:
    return math.sqrt((first.x - second.x) ** 2 + (first.y - second.y) ** 2 + (first.z - second.z) ** 2)


def _vector(start: HandPoint, end: HandPoint) -> Point3:
    return (end.x - start.x, end.y - start.y, end.z - start.z)


def _cross(first: Point3, second: Point3) -> Point3:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _magnitude(vector: Point3) -> float:
    return math.sqrt(sum(component * component for component in vector))


def _unit(vector: Point3) -> Point3 | None:
    length = _magnitude(vector)
    if not math.isfinite(length) or length <= 1e-8:
        return None
    return tuple(component / length for component in vector)  # type: ignore[return-value]
