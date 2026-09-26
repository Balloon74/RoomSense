"""Rule-based V1 posture and hand-raise recognition."""

from __future__ import annotations

from enum import Enum
from typing import Mapping

from roomsense.tracking.pose_tracker import Landmark


class PoseState(str, Enum):
    STANDING = "STANDING"
    CROUCHING = "CROUCHING"
    SITTING = "SITTING"
    LEFT_HAND_RAISED = "LEFT HAND RAISED"
    RIGHT_HAND_RAISED = "RIGHT HAND RAISED"
    BOTH_HANDS_RAISED = "BOTH HANDS RAISED"


class GestureDetector:
    def __init__(self, raised_margin: float = 0.04, standing_leg_ratio: float = 0.42,
                 sitting_leg_ratio: float = 0.18) -> None:
        self.raised_margin = raised_margin
        self.standing_leg_ratio = standing_leg_ratio
        self.sitting_leg_ratio = sitting_leg_ratio

    def detect(self, landmarks: Mapping[str, Landmark]) -> list[PoseState]:
        states: list[PoseState] = []
        left_raised = self._hand_raised(landmarks, "left")
        right_raised = self._hand_raised(landmarks, "right")
        if left_raised and right_raised:
            states.append(PoseState.BOTH_HANDS_RAISED)
        elif left_raised:
            states.append(PoseState.LEFT_HAND_RAISED)
        elif right_raised:
            states.append(PoseState.RIGHT_HAND_RAISED)

        posture = self._posture(landmarks)
        if posture is not None:
            states.insert(0, posture)
        return states

    def _hand_raised(self, landmarks: Mapping[str, Landmark], side: str) -> bool:
        wrist, shoulder = landmarks.get(f"{side}_wrist"), landmarks.get(f"{side}_shoulder")
        return bool(wrist and shoulder and wrist.y < shoulder.y - self.raised_margin)

    def _posture(self, landmarks: Mapping[str, Landmark]) -> PoseState | None:
        hips = [landmarks.get(f"{side}_hip") for side in ("left", "right")]
        knees = [landmarks.get(f"{side}_knee") for side in ("left", "right")]
        ankles = [landmarks.get(f"{side}_ankle") for side in ("left", "right")]
        valid = [(hip, knee, ankle) for hip, knee, ankle in zip(hips, knees, ankles) if hip and knee and ankle]
        if not valid:
            return None
        hip_y = sum(hip.y for hip, _, _ in valid) / len(valid)
        knee_y = sum(knee.y for _, knee, _ in valid) / len(valid)
        ankle_y = sum(ankle.y for _, _, ankle in valid) / len(valid)
        leg_span = max(0.01, ankle_y - hip_y)
        knee_to_hip = abs(knee_y - hip_y)
        if knee_to_hip / leg_span < self.sitting_leg_ratio:
            return PoseState.SITTING
        if knee_to_hip / leg_span < self.standing_leg_ratio:
            return PoseState.CROUCHING
        return PoseState.STANDING
