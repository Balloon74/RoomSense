import json
import tempfile
import unittest
from pathlib import Path

from roomsense.interactions.events import EventType, RoomSenseEvent
from roomsense.recording.session_recorder import SessionRecorder, read_session


class SessionRecordingTests(unittest.TestCase):
    def test_writes_schema_versioned_event_state_and_position_without_video(self):
        now = [10.0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.jsonl"
            recorder = SessionRecorder(path, clock=lambda: now[0])
            recorder.start()
            now[0] = 10.1
            event = RoomSenseEvent(EventType.OBJECT_POINTED, 10.1, {"object_id": "monitor"}, 0.87)
            recorder.record(event, {
                "timestamp": 10.1,
                "room_position": (0.3, 0.4),
                "movement": "STILL",
                "mode": "NORMAL",
                "target": "monitor",
            })
            now[0] = 10.2
            recorder.close()
            lines = path.read_text(encoding="utf-8").splitlines()
            records = list(read_session(path))

        self.assertEqual([record.kind for record in records], ["session_start", "event", "position", "session_end"])
        self.assertTrue(all(record.schema_version == 1 for record in records))
        self.assertEqual(records[1].payload["event"]["type"], "OBJECT_POINTED")
        self.assertEqual(records[1].payload["state"]["mode"], "NORMAL")
        self.assertEqual(records[2].payload["room_position"], [0.3, 0.4])
        for line in lines:
            parsed = json.loads(line, parse_constant=lambda value: self.fail(f"non-standard JSON constant: {value}"))
            self.assertNotIn("video", parsed)
            self.assertNotIn("image", parsed)

    def test_rate_limits_position_samples_and_records_state_changes(self):
        now = [0.0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.jsonl"
            recorder = SessionRecorder(path, position_sample_interval=0.5, clock=lambda: now[0])
            recorder.start()
            for timestamp, x, movement in (
                (0.0, 0.1, "STILL"),
                (0.2, 0.2, "STILL"),
                (0.6, 0.3, "STILL"),
                (0.7, 0.4, "MOVING_RIGHT"),
            ):
                now[0] = timestamp
                recorder.record(None, {
                    "timestamp": timestamp,
                    "room_position": (x, 0.5),
                    "movement": movement,
                    "mode": "NORMAL",
                })
            recorder.close()
            records = list(read_session(path))

        positions = [record for record in records if record.kind == "position"]
        states = [record for record in records if record.kind == "state"]
        self.assertEqual([record.timestamp for record in positions], [0.0, 0.6])
        self.assertEqual(len(states), 2)
        self.assertEqual(states[-1].payload["movement"], "MOVING_RIGHT")

    def test_missing_room_position_is_null_and_malformed_lines_are_line_numbered(self):
        now = [1.0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.jsonl"
            recorder = SessionRecorder(path, clock=lambda: now[0])
            recorder.start()
            now[0] = 1.1
            recorder.record(None, {"timestamp": 1.1, "room_position": None, "movement": "UNKNOWN", "mode": "NORMAL"})
            now[0] = 1.2
            recorder.close()
            records = list(read_session(path))
            self.assertTrue(any(record.kind == "position" and record.payload["room_position"] is None
                                for record in records))
            path.write_text(lines := path.read_text(encoding="utf-8") + "{bad json}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"line 5"):
                list(read_session(path))

    def test_reader_rejects_bad_schema_and_out_of_order_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.jsonl"
            path.write_text(
                '{"schema_version": 2, "timestamp": 1, "kind": "state", "payload": {}}\n',
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                list(read_session(path))
            path.write_text(
                '{"schema_version": 1, "timestamp": 2, "kind": "state", "payload": {}}\n'
                '{"schema_version": 1, "timestamp": 1, "kind": "state", "payload": {}}\n',
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                list(read_session(path))


if __name__ == "__main__":
    unittest.main()
