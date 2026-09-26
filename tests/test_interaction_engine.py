import unittest

from roomsense.config import RoomSenseConfig
from roomsense.interactions.engine import InteractionEngine, InteractionObservation
from roomsense.interactions.events import EventType
from roomsense.spatial.room_objects import RoomObject, RoomObjectRegistry
from roomsense.tracking.pose_tracker import Landmark


_DEFAULT_POSE = object()


def pose():
    return {
        "nose": Landmark(0.5, 0.2),
        "left_shoulder": Landmark(0.2, 0.5),
        "left_elbow": Landmark(0.4, 0.5),
        "left_wrist": Landmark(0.6, 0.5),
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


if __name__ == "__main__":
    unittest.main()
