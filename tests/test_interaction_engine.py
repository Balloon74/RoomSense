import unittest

from roomsense.actions.action_registry import ActionRegistry, MacAction
from roomsense.config import RoomSenseConfig
from roomsense.interactions.engine import InteractionEngine, InteractionObservation
from roomsense.interactions.events import EventType
from roomsense.interactions.modes import InteractionMode
from roomsense.spatial.room_objects import RoomObject, RoomObjectRegistry
from roomsense.tracking.pose_tracker import Landmark
from roomsense.tracking.hand_tracker import HandPoint, TrackedHand


_DEFAULT_POSE = object()


def pose():
    return {
        "nose": Landmark(0.5, 0.2),
        "left_shoulder": Landmark(0.2, 0.5),
        "left_elbow": Landmark(0.4, 0.5),
        "left_wrist": Landmark(0.6, 0.5),
    }


def open_hand(*, y_offset=0.0):
    coords = {
        0: (0.50, 0.90), 1: (0.35, 0.78), 2: (0.28, 0.70), 3: (0.23, 0.63), 4: (0.18, 0.56),
        5: (0.30, 0.64), 6: (0.30, 0.43), 7: (0.30, 0.31), 8: (0.30, 0.19),
        9: (0.43, 0.62), 10: (0.43, 0.40), 11: (0.43, 0.27), 12: (0.43, 0.15),
        13: (0.56, 0.64), 14: (0.56, 0.43), 15: (0.56, 0.31), 16: (0.56, 0.20),
        17: (0.69, 0.66), 18: (0.69, 0.47), 19: (0.69, 0.36), 20: (0.69, 0.25),
    }
    points = tuple(HandPoint(x, y + y_offset, 0.0) for x, y in
                   (coords[index] for index in range(21)))
    return TrackedHand("right", points, 0.94)


def both_hands_up_pose():
    return {
        "left_shoulder": Landmark(0.4, 0.5), "left_wrist": Landmark(0.3, 0.2),
        "right_shoulder": Landmark(0.6, 0.5), "right_wrist": Landmark(0.7, 0.2),
    }


def observation(timestamp, *, zones=(), movement="STILL", room_position=(0.25, 0.35), landmarks=_DEFAULT_POSE):
    current_landmarks = pose() if landmarks is _DEFAULT_POSE else landmarks
    return InteractionObservation(timestamp, current_landmarks, room_position,
                                  tuple(zones), (), movement)


class InteractionEngineTests(unittest.TestCase):
    def setUp(self):
        self.config = RoomSenseConfig(
            pointing_min_visibility=0.5,
            pointing_min_extension=0.7,
            pointing_smoothing_alpha=1.0,
            target_stability_seconds=0.2,
            target_hold_seconds=0.3,
            gesture_hold_point_seconds=0.5,
            gesture_cooldown_seconds=0.2,
        )
        monitor = RoomObject("monitor", "MONITOR", ((0.65, 0.3), (0.9, 0.3), (0.9, 0.7), (0.65, 0.7)),
                             room_position=(0.8, 0.2), enabled=True)
        self.engine = InteractionEngine(self.config, RoomObjectRegistry((monitor,)))

    def test_emits_zone_target_gesture_and_movement_transition_events_once(self):
        first = self.engine.update(observation(1.0, zones=("DESK",)))
        target = self.engine.update(observation(1.3))
        moving = self.engine.update(observation(1.4, movement="MOVING_RIGHT"))
        stopped = self.engine.update(observation(1.5))

        first_types = [event.type for event in first.events]
        target_types = [event.type for event in target.events]
        self.assertIn(EventType.ZONE_ENTERED, first_types)
        self.assertIn(EventType.GESTURE_DETECTED, first_types)
        self.assertIn(EventType.OBJECT_POINTED, target_types)
        self.assertIn(EventType.PERSON_STARTED_MOVING, [event.type for event in moving.events])
        self.assertIn(EventType.PERSON_STOPPED_MOVING, [event.type for event in stopped.events])
        self.assertEqual(target.target.confirmed.object.name, "MONITOR")
        self.assertIn("DESK MODE", first.demo_status)
        self.assertIn("POINTING MONITOR", target.demo_status)

    def test_event_feed_is_bounded_and_debug_state_contains_interaction_values(self):
        config = RoomSenseConfig(event_feed_size=2, target_stability_seconds=0.0,
                                 pointing_min_visibility=0.5, pointing_min_extension=0.7,
                                 pointing_smoothing_alpha=1.0)
        engine = InteractionEngine(config, self.engine.room_objects)
        engine.update(observation(0.0, zones=("DESK",)))
        update = engine.update(observation(0.1, zones=("BED",)))
        update = engine.update(observation(0.2, zones=("DOOR",)))
        self.assertLessEqual(len(update.recent_events), 2)
        self.assertIn("pointing", update.debug_state)
        self.assertIn("cooldowns", update.debug_state)

    def test_missing_pose_clears_point_target_and_does_not_emit_frame_events(self):
        self.engine.update(observation(0.0))
        self.engine.update(observation(0.3))
        lost = self.engine.update(observation(0.4, landmarks=None))
        self.assertIsNone(lost.target.confirmed)
        self.assertEqual(lost.pointing, ())

    def test_debug_state_explains_why_pointing_or_target_is_unavailable(self):
        no_person = self.engine.update(observation(0.0, landmarks=None))
        self.assertEqual(no_person.debug_state["pointing_status"], "NO PERSON DETECTED")

        missing_arm = self.engine.update(observation(0.1, landmarks={"nose": Landmark(0.5, 0.2)}))
        self.assertEqual(missing_arm.debug_state["pointing_status"], "ARM LANDMARKS MISSING")

        low_visibility = pose()
        low_visibility["left_wrist"] = Landmark(0.6, 0.5, visibility=0.2)
        low = self.engine.update(observation(0.2, landmarks=low_visibility))
        self.assertEqual(low.debug_state["pointing_status"], "LOW ARM VISIBILITY")

        bent = pose()
        bent["left_elbow"] = Landmark(0.4, 0.5)
        bent["left_wrist"] = Landmark(0.3, 0.4)
        bent_result = self.engine.update(observation(0.3, landmarks=bent))
        self.assertEqual(bent_result.debug_state["pointing_status"], "ARM NOT EXTENDED")

    def test_command_mode_transitions_and_timeout_are_reported(self):
        config = RoomSenseConfig(
            pointing_min_visibility=0.5,
            pointing_min_extension=0.7,
            command_mode_timeout_seconds=1.0,
            gesture_both_hands_hold_seconds=0.2,
        )
        engine = InteractionEngine(config, RoomObjectRegistry())
        up = {
            "left_shoulder": Landmark(0.4, 0.5), "left_wrist": Landmark(0.3, 0.2),
            "right_shoulder": Landmark(0.6, 0.5), "right_wrist": Landmark(0.7, 0.2),
        }
        engine.update(observation(0.0, landmarks=up))
        entered = engine.update(observation(0.21, landmarks=up))
        self.assertEqual(entered.mode.value, "COMMAND")
        self.assertIn(EventType.MODE_CHANGED, [event.type for event in entered.events])
        timed_out = engine.update(observation(1.22, landmarks=up))
        self.assertEqual(timed_out.mode.value, "NORMAL")

    def test_mode_off_cannot_dispatch_completed_hand_gesture(self):
        registry = ActionRegistry()
        engine = InteractionEngine(self.config, RoomObjectRegistry(), mac_action_registry=registry)
        for stamp in (0.0, 0.3, 0.6, 0.9):
            update = engine.update(InteractionObservation(stamp, None, None, (), (), "STILL", (open_hand(),)))
        self.assertEqual(update.mode.value, "NORMAL")
        self.assertEqual(update.action_results, ())
        self.assertEqual(registry.history, ())

    def test_both_hands_up_is_required_to_enter_command_mode(self):
        engine = InteractionEngine(self.config, RoomObjectRegistry())
        one_up = {"left_shoulder": Landmark(0.4, 0.5), "left_wrist": Landmark(0.3, 0.2)}
        engine.update(observation(0.0, landmarks=one_up))
        still_normal = engine.update(observation(0.3, landmarks=one_up))
        self.assertEqual(still_normal.mode.value, "NORMAL")
        engine.update(observation(0.4, landmarks=both_hands_up_pose()))
        entered = engine.update(observation(1.0, landmarks=both_hands_up_pose()))
        self.assertEqual(entered.mode.value, "COMMAND")

    def test_only_accepted_command_actions_refresh_timeout(self):
        config = RoomSenseConfig(
            command_mode_timeout_seconds=0.8, gesture_both_hands_hold_seconds=0.2,
            gesture_min_samples=3, gesture_open_palm_hold_seconds=0.3,
            target_stability_seconds=0.0,
        )
        registry = ActionRegistry()
        engine = InteractionEngine(config, RoomObjectRegistry(), mac_action_registry=registry)
        engine.update(observation(0.0, landmarks=both_hands_up_pose()))
        entered = engine.update(observation(0.21, landmarks=both_hands_up_pose()))
        self.assertEqual(entered.mode.value, "COMMAND")
        for stamp in (0.3, 0.5, 0.72):
            update = engine.update(InteractionObservation(stamp, pose(), (0.2, 0.2), (), (), "STILL",
                                                          (open_hand(),)))
        self.assertEqual([result.action for result in update.action_results], [MacAction.PLAY_PAUSE])
        self.assertEqual(engine.update(observation(1.3, landmarks=None)).mode.value, "COMMAND")
        self.assertEqual(engine.update(observation(1.53, landmarks=None)).mode.value, "NORMAL")

    def test_timeout_clears_pending_gesture_and_exits_once(self):
        config = RoomSenseConfig(command_mode_timeout_seconds=0.35, gesture_both_hands_hold_seconds=0.2,
                                 gesture_open_palm_hold_seconds=0.5)
        registry = ActionRegistry()
        engine = InteractionEngine(config, RoomObjectRegistry(), mac_action_registry=registry)
        engine.update(observation(0.0, landmarks=both_hands_up_pose()))
        engine.update(observation(0.21, landmarks=both_hands_up_pose()))
        for stamp in (0.25, 0.4, 0.56):
            update = engine.update(InteractionObservation(stamp, None, None, (), (), "STILL", (open_hand(),)))
        self.assertEqual(update.mode.value, "NORMAL")
        self.assertEqual(update.action_results, ())
        self.assertEqual(engine.update(observation(0.7, landmarks=None)).mode.value, "NORMAL")

    def test_fist_cancel_is_visible_without_dispatching_action(self):
        registry = ActionRegistry()
        engine = InteractionEngine(self.config, RoomObjectRegistry(), mac_action_registry=registry)
        engine.mode_controller._transition(InteractionMode.COMMAND, 0.0, "test")
        fist_coords = {
            0: (0.50, 0.90), 1: (0.35, 0.78), 2: (0.28, 0.70), 3: (0.23, 0.63), 4: (0.36, 0.72),
            5: (0.30, 0.64), 6: (0.30, 0.43), 7: (0.30, 0.31), 8: (0.34, 0.73),
            9: (0.43, 0.62), 10: (0.43, 0.40), 11: (0.43, 0.27), 12: (0.45, 0.73),
            13: (0.56, 0.64), 14: (0.56, 0.43), 15: (0.56, 0.31), 16: (0.55, 0.73),
            17: (0.69, 0.66), 18: (0.69, 0.47), 19: (0.69, 0.36), 20: (0.64, 0.73),
        }
        fist = TrackedHand("right", tuple(HandPoint(x, y, 0.0) for x, y in
                                           (fist_coords[index] for index in range(21))), 0.94)
        engine.update(InteractionObservation(0.0, None, None, (), (), "STILL", (fist,)))
        engine.update(InteractionObservation(0.2, None, None, (), (), "STILL", (fist,)))
        update = engine.update(InteractionObservation(0.4, None, None, (), (), "STILL", (fist,)))
        self.assertTrue(update.command_gesture.cancelled)
        self.assertEqual(update.action_results, ())


if __name__ == "__main__":
    unittest.main()
