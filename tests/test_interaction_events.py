import unittest

from roomsense.interactions.events import EventCollector, EventType, RoomSenseEvent


class InteractionEventTests(unittest.TestCase):
    def test_event_has_type_timestamp_metadata_and_optional_confidence(self):
        event = RoomSenseEvent(EventType.OBJECT_POINTED, 12.5, {"object_id": "monitor"}, 0.87)
        self.assertEqual(event.type, EventType.OBJECT_POINTED)
        self.assertEqual(event.timestamp, 12.5)
        self.assertEqual(event.metadata["object_id"], "monitor")
        self.assertEqual(event.confidence, 0.87)
        self.assertRegex(event.display_time, r"^\d{2}:\d{2}:\d{2}$")

    def test_collector_retains_bounded_events_in_chronological_order(self):
        collector = EventCollector(max_events=2)
        first = RoomSenseEvent(EventType.ZONE_ENTERED, 1.0, {"zone": "DESK"})
        second = RoomSenseEvent(EventType.GESTURE_DETECTED, 2.0, {"gesture": "SWIPE_RIGHT"})
        third = RoomSenseEvent(EventType.MODE_CHANGED, 3.0, {"mode": "COMMAND"})
        collector.publish(first)
        collector.publish(second)
        collector.publish(third)
        self.assertEqual(collector.recent, (second, third))

    def test_rejects_invalid_confidence_and_out_of_order_events(self):
        with self.assertRaises(ValueError):
            RoomSenseEvent(EventType.OBJECT_POINTED, 1.0, {}, 1.1)
        collector = EventCollector()
        collector.publish(RoomSenseEvent(EventType.ZONE_ENTERED, 2.0, {}))
        with self.assertRaises(ValueError):
            collector.publish(RoomSenseEvent(EventType.ZONE_LEFT, 1.0, {}))


if __name__ == "__main__":
    unittest.main()
