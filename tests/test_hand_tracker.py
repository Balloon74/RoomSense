import sys
import tempfile
import types
import unittest
from pathlib import Path

from roomsense.config import RoomSenseConfig


class FakeImage:
    def __init__(self, *, image_format, data):
        self.image_format = image_format
        self.data = data


class FakeTask:
    def __init__(self, results):
        self.results = list(results)
        self.timestamps = []
        self.closed = False

    def detect_for_video(self, image, timestamp_ms):
        self.timestamps.append(timestamp_ms)
        self.last_image = image
        return self.results.pop(0)

    def close(self):
        self.closed = True


def fake_mediapipe(task):
    class FakeHandLandmarker:
        @staticmethod
        def create_from_options(options):
            task.options = options
            return task

    class FakeHandLandmarkerOptions:
        def __init__(self, **kwargs):
            self.values = kwargs

    vision = types.SimpleNamespace(
        HandLandmarker=FakeHandLandmarker,
        HandLandmarkerOptions=FakeHandLandmarkerOptions,
        RunningMode=types.SimpleNamespace(VIDEO="VIDEO"),
    )
    tasks = types.SimpleNamespace(BaseOptions=lambda **kwargs: kwargs, vision=vision)
    return types.SimpleNamespace(
        Image=FakeImage,
        ImageFormat=types.SimpleNamespace(SRGB="SRGB"),
        tasks=tasks,
    )


def result_with_hand():
    points = [types.SimpleNamespace(x=index / 25, y=0.25 + index / 100, z=-0.01 * index)
              for index in range(21)]
    category = types.SimpleNamespace(category_name="Right", score=0.94)
    return types.SimpleNamespace(hand_landmarks=[points], handedness=[[category]])


class HandTrackerTests(unittest.TestCase):
    def make_tracker(self, directory, task):
        model_path = Path(directory) / "hand.task"
        model_path.write_bytes(b"test model placeholder")
        fake_mp = fake_mediapipe(task)
        previous_mediapipe = sys.modules.get("mediapipe")
        had_mediapipe = "mediapipe" in sys.modules
        sys.modules["mediapipe"] = fake_mp

        def restore_mediapipe():
            if had_mediapipe:
                sys.modules["mediapipe"] = previous_mediapipe
            else:
                sys.modules.pop("mediapipe", None)

        self.addCleanup(restore_mediapipe)
        config = RoomSenseConfig(
            hand_model_path=str(model_path),
            hand_detection_confidence=0.42,
            hand_tracking_confidence=0.63,
        )
        from roomsense.tracking.hand_tracker import HandTracker
        return HandTracker(config)

    def test_tracker_maps_21_normalized_landmarks_and_handedness(self):
        task = FakeTask([result_with_hand()])
        with tempfile.TemporaryDirectory() as directory:
            tracker = self.make_tracker(directory, task)
            hands = tracker.process("rgb-frame", timestamp_ms=45)

        self.assertEqual(len(hands), 1)
        self.assertEqual(hands[0].handedness, "right")
        self.assertEqual(hands[0].confidence, 0.94)
        self.assertEqual(len(hands[0].landmarks), 21)
        self.assertEqual(hands[0].landmarks[8].x, 8 / 25)
        self.assertEqual(hands[0].landmarks[8].y, 0.33)
        self.assertEqual(hands[0].landmarks[8].z, -0.08)
        self.assertEqual(task.timestamps, [45])
        self.assertEqual(task.last_image.image_format, "SRGB")
        self.assertEqual(task.options.values["running_mode"], "VIDEO")
        self.assertEqual(task.options.values["min_hand_detection_confidence"], 0.42)
        self.assertEqual(task.options.values["min_tracking_confidence"], 0.63)

    def test_tracker_returns_empty_tuple_when_no_hand_is_found(self):
        task = FakeTask([types.SimpleNamespace(hand_landmarks=[], handedness=[])])
        with tempfile.TemporaryDirectory() as directory:
            tracker = self.make_tracker(directory, task)
            self.assertEqual(tracker.process("rgb-frame", timestamp_ms=5), ())

    def test_tracker_close_releases_task(self):
        task = FakeTask([])
        with tempfile.TemporaryDirectory() as directory:
            tracker = self.make_tracker(directory, task)
            tracker.close()
        self.assertTrue(task.closed)


if __name__ == "__main__":
    unittest.main()
