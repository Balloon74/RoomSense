import unittest

from roomsense.config import RoomSenseConfig
from roomsense.tracking.position_tracker import PositionTracker
from roomsense.tracking.pose_tracker import Landmark
from roomsense.tracking.person_state import Movement, MovementTracker
from roomsense.gestures.gesture_detector import GestureDetector, PoseState
from roomsense.utils.smoothing import ExponentialSmoother


def landmark(x: float, y: float, visibility: float = 1.0) -> Landmark:
    return Landmark(x=x, y=y, z=0.0, visibility=visibility)


class SmoothingTests(unittest.TestCase):
    def test_exponential_smoother_blends_new_measurement(self) -> None:
        smoother = ExponentialSmoother(alpha=0.5)
        self.assertEqual(smoother.update(0.0), 0.0)
        self.assertEqual(smoother.update(1.0), 0.5)


class PositionTests(unittest.TestCase):
    def test_position_uses_torso_center_and_relative_shoulder_scale(self) -> None:
        tracker = PositionTracker(RoomSenseConfig(smoothing_alpha=1.0, reference_shoulder_width=0.2))
        landmarks = {
            "left_shoulder": landmark(0.4, 0.35),
            "right_shoulder": landmark(0.6, 0.35),
            "left_hip": landmark(0.43, 0.62),
            "right_hip": landmark(0.57, 0.62),
        }
        position = tracker.update(landmarks)
        self.assertAlmostEqual(position.x, 0.0)
        self.assertAlmostEqual(position.y, 0.03)  # Positive Y is up; torso center is y=0.485.
        self.assertAlmostEqual(position.z, 0.0)

    def test_position_depth_increases_when_person_appears_larger(self) -> None:
        tracker = PositionTracker(RoomSenseConfig(smoothing_alpha=1.0, reference_shoulder_width=0.2))
        landmarks = {
            "left_shoulder": landmark(0.35, 0.35),
            "right_shoulder": landmark(0.65, 0.35),
            "left_hip": landmark(0.4, 0.62),
            "right_hip": landmark(0.6, 0.62),
        }
        self.assertGreater(tracker.update(landmarks).z, 0.0)


class MovementTests(unittest.TestCase):
    def test_movement_uses_recent_history(self) -> None:
        tracker = MovementTracker(window_seconds=1.0, horizontal_threshold=0.08, depth_threshold=0.08)
        tracker.update(0.0, 0.0, 0.0, now=0.0)
        tracker.update(0.2, 0.0, 0.0, now=0.5)
        self.assertEqual(tracker.current, Movement.MOVING_RIGHT)

    def test_movement_returns_still_when_recent_delta_is_small(self) -> None:
        tracker = MovementTracker(window_seconds=1.0, horizontal_threshold=0.08, depth_threshold=0.08)
        tracker.update(0.0, 0.0, 0.0, now=0.0)
        tracker.update(0.02, 0.0, 0.01, now=0.5)
        self.assertEqual(tracker.current, Movement.STILL)

    def test_movement_uses_depth_direction_for_toward_and_away(self) -> None:
        tracker = MovementTracker(window_seconds=1.0, horizontal_threshold=0.08, depth_threshold=0.08)
        tracker.update(0.0, 0.0, -0.2, now=0.0)
        self.assertEqual(tracker.update(0.0, 0.0, 0.1, now=0.5), Movement.MOVING_TOWARD)
        tracker.reset()
        tracker.update(0.0, 0.0, 0.2, now=1.0)
        self.assertEqual(tracker.update(0.0, 0.0, -0.1, now=1.5), Movement.MOVING_AWAY)


class GestureTests(unittest.TestCase):
    def test_detects_both_hands_raised_and_standing(self) -> None:
        detector = GestureDetector()
        pose = {
            "nose": landmark(0.5, 0.12),
            "left_shoulder": landmark(0.4, 0.3),
            "right_shoulder": landmark(0.6, 0.3),
            "left_elbow": landmark(0.3, 0.2),
            "right_elbow": landmark(0.7, 0.2),
            "left_wrist": landmark(0.3, 0.08),
            "right_wrist": landmark(0.7, 0.08),
            "left_hip": landmark(0.43, 0.58),
            "right_hip": landmark(0.57, 0.58),
            "left_knee": landmark(0.43, 0.78),
            "right_knee": landmark(0.57, 0.78),
            "left_ankle": landmark(0.43, 0.98),
            "right_ankle": landmark(0.57, 0.98),
        }
        states = detector.detect(pose)
        self.assertIn(PoseState.STANDING, states)
        self.assertIn(PoseState.BOTH_HANDS_RAISED, states)

    def test_distinguishes_crouching_and_sitting_from_leg_proportions(self) -> None:
        detector = GestureDetector()
        crouching = {
            "left_hip": landmark(0.43, 0.50), "right_hip": landmark(0.57, 0.50),
            "left_knee": landmark(0.40, 0.65), "right_knee": landmark(0.60, 0.65),
            "left_ankle": landmark(0.40, 0.90), "right_ankle": landmark(0.60, 0.90),
        }
        sitting = {
            "left_hip": landmark(0.43, 0.55), "right_hip": landmark(0.57, 0.55),
            "left_knee": landmark(0.40, 0.57), "right_knee": landmark(0.60, 0.57),
            "left_ankle": landmark(0.40, 0.85), "right_ankle": landmark(0.60, 0.85),
        }
        self.assertIn(PoseState.CROUCHING, detector.detect(crouching))
        self.assertIn(PoseState.SITTING, detector.detect(sitting))


if __name__ == "__main__":
    unittest.main()
