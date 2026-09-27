import unittest

from roomsense.actions.action_registry import MacAction
from roomsense.config import RoomSenseConfig
from roomsense.gestures.mac_controls import (
    HandGestureController,
    HandShape,
    classify_hand,
)
from roomsense.tracking.hand_tracker import HandPoint, TrackedHand


def make_hand(shape="open", *, side="right", confidence=0.94, x_offset=0.0, y_offset=0.0):
    points = [HandPoint(0.5, 0.9, 0.0) for _ in range(21)]
    coords = {
        0: (0.50, 0.90),
        1: (0.35, 0.78), 2: (0.28, 0.70), 3: (0.23, 0.63), 4: (0.18, 0.56),
        5: (0.30, 0.64), 6: (0.30, 0.43), 7: (0.30, 0.31), 8: (0.30, 0.19),
        9: (0.43, 0.62), 10: (0.43, 0.40), 11: (0.43, 0.27), 12: (0.43, 0.15),
        13: (0.56, 0.64), 14: (0.56, 0.43), 15: (0.56, 0.31), 16: (0.56, 0.20),
        17: (0.69, 0.66), 18: (0.69, 0.47), 19: (0.69, 0.36), 20: (0.69, 0.25),
    }
    if shape == "pinch":
        coords[8] = coords[4]
    elif shape == "other":
        coords[8] = (0.34, 0.73)
        coords[12] = (0.45, 0.73)
    elif shape == "fist":
        for tip, x in ((8, 0.34), (12, 0.45), (16, 0.55), (20, 0.64)):
            coords[tip] = (x, 0.73)
        coords[4] = (0.36, 0.72)
    for index, (x, y) in coords.items():
        points[index] = HandPoint(x + x_offset, y + y_offset, 0.0)
    return TrackedHand(side, tuple(points), confidence)


def move(hand, *, x=0.0, y=0.0, confidence=None):
    return TrackedHand(
        hand.handedness,
        tuple(HandPoint(point.x + x, point.y + y, point.z) for point in hand.landmarks),
        hand.confidence if confidence is None else confidence,
    )


class MacControlsTests(unittest.TestCase):
    def setUp(self):
        self.config = RoomSenseConfig(
            gesture_min_confidence=0.75,
            gesture_open_palm_hold_seconds=0.5,
            gesture_pinch_hold_seconds=0.15,
            gesture_pinch_ratio=0.30,
            gesture_fist_hold_seconds=0.35,
            gesture_min_samples=3,
            gesture_swipe_min_duration_seconds=0.08,
            gesture_swipe_distance=0.16,
            gesture_swipe_window_seconds=0.6,
            gesture_cooldown_seconds=0.8,
            gesture_volume_movement_threshold=0.08,
            gesture_volume_step_percent=5,
            gesture_volume_update_interval_seconds=0.25,
        )

    def test_shape_classifier_distinguishes_open_palm_pinch_and_fist(self):
        self.assertEqual(classify_hand(make_hand("open"), self.config).shape, HandShape.OPEN_PALM)
        self.assertEqual(classify_hand(make_hand("pinch"), self.config).shape, HandShape.PINCH)
        self.assertEqual(classify_hand(make_hand("fist"), self.config).shape, HandShape.FIST)

    def test_open_palm_requires_configured_hold_and_fires_once_until_release(self):
        controller = HandGestureController(self.config)
        for timestamp in (0.0, 0.25, 0.49):
            status = controller.update((make_hand(),), timestamp, command_mode=True)
            self.assertEqual(status.actions, ())
        fired = controller.update((make_hand(),), 0.51, command_mode=True)
        self.assertEqual([action.action for action in fired.actions], [MacAction.PLAY_PAUSE])
        self.assertEqual(controller.update((make_hand(),), 0.75, command_mode=True).actions, ())
        controller.update((make_hand("other"),), 0.8, command_mode=True)
        again = controller.update((make_hand(),), 0.9, command_mode=True)
        self.assertEqual(again.actions, ())
        controller.update((make_hand(),), 1.2, command_mode=True)
        refired = controller.update((make_hand(),), 1.42, command_mode=True)
        self.assertEqual([action.action for action in refired.actions], [MacAction.PLAY_PAUSE])

    def test_swipe_requires_multiple_samples_and_reports_direction(self):
        controller = HandGestureController(self.config)
        self.assertEqual(controller.update((make_hand(x_offset=0.2),), 0.0, True).actions, ())
        self.assertEqual(controller.update((make_hand(x_offset=0.28),), 0.05, True).actions, ())
        right = controller.update((make_hand(x_offset=0.37),), 0.12, True)
        self.assertEqual([action.action for action in right.actions], [MacAction.NEXT_TRACK])
        self.assertEqual(right.gesture, "SWIPE_RIGHT")

        controller.reset()
        controller.update((make_hand(x_offset=0.5),), 1.0, True)
        controller.update((make_hand(x_offset=0.42),), 1.05, True)
        left = controller.update((make_hand(x_offset=0.3),), 1.12, True)
        self.assertEqual([action.action for action in left.actions], [MacAction.PREVIOUS_TRACK])

    def test_pinch_start_movement_and_release(self):
        controller = HandGestureController(self.config)
        pinch = make_hand("pinch")
        for timestamp in (0.0, 0.08, 0.16):
            status = controller.update((pinch,), timestamp, True)
        self.assertEqual(status.gesture, "PINCH")
        self.assertEqual(status.actions, ())
        moved = controller.update((move(pinch, y=-0.09),), 0.20, True)
        self.assertEqual([action.action for action in moved.actions], [MacAction.VOLUME_UP])
        released = controller.update((make_hand(),), 0.3, True)
        self.assertNotEqual(released.gesture, "PINCH")
        self.assertEqual(released.actions, ())

    def test_volume_quantizes_each_threshold_crossing_and_rate_limits(self):
        controller = HandGestureController(self.config)
        pinch = make_hand("pinch")
        for timestamp in (0.0, 0.08, 0.16):
            controller.update((pinch,), timestamp, True)
        first = controller.update((move(pinch, y=-0.09),), 0.20, True)
        too_soon = controller.update((move(pinch, y=-0.18),), 0.30, True)
        second = controller.update((move(pinch, y=-0.18),), 0.46, True)
        self.assertEqual([action.action for action in first.actions], [MacAction.VOLUME_UP])
        self.assertEqual(too_soon.actions, ())
        self.assertEqual([action.action for action in second.actions], [MacAction.VOLUME_UP])

    def test_fist_cancels_pinch_and_requires_release_to_rearm(self):
        controller = HandGestureController(self.config)
        pinch = make_hand("pinch")
        for timestamp in (0.0, 0.08, 0.16):
            controller.update((pinch,), timestamp, True)
        fist = make_hand("fist")
        controller.update((fist,), 0.2, True)
        controller.update((fist,), 0.38, True)
        canceled = controller.update((fist,), 0.56, True)
        self.assertTrue(canceled.cancelled)
        self.assertEqual(canceled.actions, ())
        self.assertFalse(controller.update((fist,), 0.7, True).cancelled)
        controller.update((make_hand(),), 0.8, True)
        controller.update((pinch,), 0.9, True)
        controller.update((pinch,), 1.0, True)
        rearmed = controller.update((pinch,), 1.06, True)
        self.assertEqual(rearmed.gesture, "PINCH")
        self.assertEqual(rearmed.actions, ())
        volume = controller.update((move(pinch, y=-0.09),), 1.1, True)
        self.assertEqual([action.action for action in volume.actions], [MacAction.VOLUME_UP])

    def test_low_confidence_or_missing_hand_breaks_continuity(self):
        controller = HandGestureController(self.config)
        hand = make_hand()
        controller.update((hand,), 0.0, True)
        controller.update((hand,), 0.3, True)
        controller.update((move(hand, confidence=0.4),), 0.4, True)
        self.assertEqual(controller.update((), 0.6, True).actions, ())
        self.assertEqual(controller.update((hand,), 0.7, True).actions, ())
        self.assertEqual(controller.update((hand,), 0.9, True).actions, ())
        fired = controller.update((hand,), 1.21, True)
        self.assertEqual([action.action for action in fired.actions], [MacAction.PLAY_PAUSE])

    def test_mode_off_never_emits_action_and_restarts_holds_on_activation(self):
        controller = HandGestureController(self.config)
        for timestamp in (0.0, 0.3, 0.6):
            self.assertEqual(controller.update((make_hand(),), timestamp, False).actions, ())
        self.assertEqual(controller.update((make_hand(),), 0.7, True).actions, ())
        self.assertEqual(controller.update((make_hand(),), 1.0, True).actions, ())
        activated = controller.update((make_hand(),), 1.21, True)
        self.assertEqual([action.action for action in activated.actions], [MacAction.PLAY_PAUSE])

    def test_two_hands_keep_one_stable_gesture_owner(self):
        controller = HandGestureController(self.config)
        left_open = make_hand(side="left", confidence=0.85)
        right_swipe = make_hand(side="right", confidence=0.99, x_offset=0.0)
        self.assertEqual(controller.update((left_open,), 0.0, True).gesture, "OPEN_PALM")
        controller.update((left_open, move(right_swipe, x=0.2)), 0.05, True)
        still_left = controller.update((left_open, move(right_swipe, x=0.3)), 0.12, True)
        self.assertEqual(still_left.gesture, "OPEN_PALM")
        self.assertEqual(still_left.actions, ())
        first_right = controller.update((move(right_swipe, x=0.4),), 0.2, True)
        self.assertEqual(first_right.actions, ())
        controller.update((move(right_swipe, x=0.48),), 0.25, True)
        next_track = controller.update((move(right_swipe, x=0.58),), 0.32, True)
        self.assertEqual([action.action for action in next_track.actions], [MacAction.NEXT_TRACK])

    def test_cooldown_and_debounce_suppress_repeated_swipes(self):
        controller = HandGestureController(self.config)
        for timestamp, x in ((0.0, 0.1), (0.05, 0.2), (0.1, 0.3)):
            status = controller.update((make_hand(x_offset=x),), timestamp, True)
        self.assertEqual([action.action for action in status.actions], [MacAction.NEXT_TRACK])
        for timestamp, x in ((0.2, 0.5), (0.25, 0.4), (0.3, 0.3)):
            self.assertEqual(controller.update((make_hand(x_offset=x),), timestamp, True).actions, ())
        for timestamp, x in ((0.95, 0.5), (1.0, 0.4), (1.05, 0.3)):
            status = controller.update((make_hand(x_offset=x),), timestamp, True)
        self.assertEqual([action.action for action in status.actions], [MacAction.PREVIOUS_TRACK])

    def test_non_finite_or_decreasing_timestamps_are_rejected(self):
        controller = HandGestureController(self.config)
        with self.assertRaises(ValueError):
            controller.update((make_hand(),), float("nan"), True)
        controller.update((make_hand(),), 1.0, True)
        with self.assertRaises(ValueError):
            controller.update((make_hand(),), 0.9, True)


if __name__ == "__main__":
    unittest.main()
