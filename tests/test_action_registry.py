import unittest

from roomsense.interactions.actions import ActionRegistry, create_demo_action_registry
from roomsense.interactions.events import EventType, RoomSenseEvent


class ActionRegistryTests(unittest.TestCase):
    def test_registered_handler_receives_event_and_context(self):
        registry = ActionRegistry()
        registry.register("test.message", lambda event, context: "NEXT" if context.get("mode") == "COMMAND" else None)
        event = RoomSenseEvent(EventType.GESTURE_DETECTED, 1.0, {"gesture": "SWIPE_RIGHT"})
        self.assertEqual(registry.dispatch(event, {"mode": "COMMAND"}), ("NEXT",))
        self.assertEqual(registry.dispatch(event, {"mode": "NORMAL"}), ())

    def test_demo_registry_only_maps_command_monitor_swipes_and_ui_events(self):
        registry = create_demo_action_registry()
        swipe = RoomSenseEvent(EventType.GESTURE_DETECTED, 1.0, {"gesture": "SWIPE_RIGHT"})
        self.assertEqual(registry.dispatch(swipe, {"mode": "COMMAND", "target_id": "monitor"}), ("NEXT",))
        self.assertEqual(registry.dispatch(swipe, {"mode": "NORMAL", "target_id": "monitor"}), ())
        self.assertEqual(registry.dispatch(swipe, {"mode": "COMMAND", "target_id": "bed"}), ())
        zone = RoomSenseEvent(EventType.ZONE_ENTERED, 2.0, {"zone": "DESK"})
        self.assertEqual(registry.dispatch(zone, {}), ("DESK MODE",))

    def test_unknown_event_has_no_action_and_duplicate_ids_are_rejected(self):
        registry = ActionRegistry()
        event = RoomSenseEvent(EventType.OBJECT_POINTED, 1.0, {"object_id": "lamp"})
        self.assertEqual(registry.dispatch(event, {}), ())
        registry.register("one", lambda _event, _context: None)
        with self.assertRaises(ValueError):
            registry.register("one", lambda _event, _context: "UNSAFE")

    def test_unknown_action_is_not_run_and_reports_configuration_error(self):
        registry = ActionRegistry()
        event = RoomSenseEvent(EventType.GESTURE_DETECTED, 1.0, {"gesture": "SWIPE_RIGHT"})
        with self.assertLogs("roomsense.interactions.actions", level="ERROR") as logs:
            self.assertEqual(registry.dispatch_action("missing.action", event, {}), ())
        self.assertIn("Unknown RoomSense action", logs.output[0])


if __name__ == "__main__":
    unittest.main()
