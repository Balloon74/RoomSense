import unittest

from roomsense.config import RoomSenseConfig
from roomsense.spatial.pointing import ArmPointing
from roomsense.gestures.temporal_gestures import TemporalGestureDetector
from roomsense.tracking.pose_tracker import Landmark


def point(x, y, visibility=1.0):
    return Landmark(x, y, 0.0, visibility)


def one_hand(x, side="left"):
    return {f"{side}_wrist": point(x, 0.4)}


def both_hands_up():
    return {
        "left_shoulder": point(0.4, 0.5), "left_wrist": point(0.3, 0.3),
        "right_shoulder": point(0.6, 0.5), "right_wrist": point(0.7, 0.3),
    }


def arm(side="left", confidence=0.9):
    return ArmPointing(side, (0.3, 0.5), (1.0, 0.0), 0.9, confidence)


class TemporalGestureTests(unittest.TestCase):
    def setUp(self):
        self.config = RoomSenseConfig(
            gesture_swipe_distance=0.15,
            gesture_swipe_window_seconds=0.4,
            gesture_both_hands_hold_seconds=0.5,
            gesture_hold_point_seconds=0.4,
            gesture_cooldown_seconds=0.8,
        )
        self.detector = TemporalGestureDetector(self.config)

    def test_swipe_requires_multiple_timestamped_samples_and_reports_direction(self):
        self.assertEqual(self.detector.update(one_hand(0.2), (), 0.0), ())
        self.assertEqual(self.detector.update(one_hand(0.3), (), 0.1), ())
        right = self.detector.update(one_hand(0.4), (), 0.2)
        self.assertEqual([event.name for event in right], ["SWIPE_RIGHT"])
        self.detector.reset()
        self.detector.update(one_hand(0.5), (), 1.0)
        self.assertEqual(self.detector.update(one_hand(0.3), (), 1.2)[0].name, "SWIPE_LEFT")

    def test_swipe_is_not_classified_from_a_single_observation(self):
        events = self.detector.update(one_hand(0.9), (), 0.0)
        self.assertEqual(events, ())

    def test_both_hands_up_requires_a_hold_and_emits_once(self):
        self.assertEqual(self.detector.update(both_hands_up(), (), 0.0), ())
        self.assertEqual(self.detector.update(both_hands_up(), (), 0.49), ())
        raised = self.detector.update(both_hands_up(), (), 0.51)
        self.assertEqual([event.name for event in raised], ["BOTH_HANDS_UP"])
        self.assertEqual(self.detector.update(both_hands_up(), (), 0.7), ())

    def test_point_and_hold_point_are_transition_events_with_confidence(self):
        tracked_pose = {"nose": point(0.5, 0.2)}
        first = self.detector.update(tracked_pose, (arm(),), 0.0)
        continuing = self.detector.update(tracked_pose, (arm(),), 0.2)
        held = self.detector.update(tracked_pose, (arm(),), 0.41)
        self.assertEqual([event.name for event in first], ["POINT"])
        self.assertAlmostEqual(first[0].confidence, 0.9)
        self.assertEqual(continuing, ())
        self.assertEqual([event.name for event in held], ["HOLD_POINT"])
        self.assertEqual(self.detector.update(tracked_pose, (arm(),), 0.6), ())
        self.assertEqual(self.detector.active_gestures, ("LEFT_POINT", "LEFT_HOLD_POINT"))

    def test_point_confidence_is_not_mistaken_for_landmark_visibility(self):
        events = self.detector.update({"nose": point(0.5, 0.2)}, (arm(confidence=0.43),), 0.0)
        self.assertEqual([event.name for event in events], ["POINT"])

    def test_cooldown_suppresses_repeats_then_allows_a_later_swipe(self):
        self.detector.update(one_hand(0.1), (), 0.0)
        self.detector.update(one_hand(0.2), (), 0.1)
        self.assertEqual(self.detector.update(one_hand(0.3), (), 0.2)[0].name, "SWIPE_RIGHT")
        self.assertEqual(self.detector.update(one_hand(0.4), (), 0.3), ())
        self.detector.update(one_hand(0.5), (), 1.1)
        self.assertEqual(self.detector.update(one_hand(0.7), (), 1.2)[0].name, "SWIPE_RIGHT")

    def test_missing_landmarks_break_continuity_and_cooldown_state_is_exposed(self):
        self.detector.update(one_hand(0.1), (), 0.0)
        self.assertEqual(self.detector.update(None, (), 0.1), ())
        self.assertEqual(self.detector.update(one_hand(0.8), (), 0.2), ())
        events = self.detector.update(one_hand(0.6), (), 0.3)
        self.assertEqual([event.name for event in events], ["SWIPE_LEFT"])
        self.assertIn("SWIPE_LEFT", self.detector.cooldowns)
        self.assertGreater(self.detector.cooldowns["SWIPE_LEFT"], 0.3)

    def test_timestamp_order_is_validated(self):
        self.detector.update({}, (), 1.0)
        with self.assertRaises(ValueError):
            self.detector.update({}, (), 0.9)


if __name__ == "__main__":
    unittest.main()
