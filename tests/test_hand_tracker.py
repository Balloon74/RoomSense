import sys
import tempfile
import types
import unittest
import math
from pathlib import Path
from unittest.mock import patch

from roomsense.config import RoomSenseConfig
from roomsense.tracking.hand_tracker import HandTracker
from roomsense.tracking.hand_geometry import HandPoint
from test_hand_geometry import open_hand_points


class FakeLandmarker:
    results = []
    options = None
    instance = None

    def __init__(self, options):
        type(self).options = options
        type(self).instance = self
        self.timestamps = []
        self.closed = False

    def detect_for_video(self, image, timestamp_ms):
        self.timestamps.append(timestamp_ms)
        return type(self).results.pop(0)

    def close(self):
        self.closed = True


class FakeLandmarkerOptions:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeBaseOptions:
    def __init__(self, model_asset_path):
        self.model_asset_path = model_asset_path


class FakeImage:
    def __init__(self, image_format, data):
        self.image_format = image_format
        self.data = data


def fake_mediapipe():
    vision = types.SimpleNamespace(
        HandLandmarkerOptions=FakeLandmarkerOptions,
        HandLandmarker=types.SimpleNamespace(create_from_options=FakeLandmarker),
        RunningMode=types.SimpleNamespace(VIDEO="VIDEO"),
    )
    return types.SimpleNamespace(
        tasks=types.SimpleNamespace(BaseOptions=FakeBaseOptions, vision=vision),
        Image=FakeImage,
        ImageFormat=types.SimpleNamespace(SRGB="SRGB"),
    )


def model_hand(points):
    return [types.SimpleNamespace(x=p.x, y=p.y, z=p.z, visibility=p.visibility) for p in points]


def result(*hands):
    names = [name for name, _ in hands]
    return types.SimpleNamespace(
        hand_landmarks=[model_hand(points) for _, points in hands],
        handedness=[
            [types.SimpleNamespace(category_name=name, score=0.9)]
            for name in names
        ],
    )


class HandTrackerConfigTests(unittest.TestCase):
    def test_hand_tracking_settings_have_safe_defaults(self):
        config = RoomSenseConfig()

        self.assertEqual(config.hand_detection_confidence, 0.5)
        self.assertEqual(config.hand_presence_confidence, 0.5)
        self.assertEqual(config.hand_tracking_confidence, 0.5)
        self.assertEqual(config.hand_smoothing_alpha, 0.45)
        self.assertEqual(config.hand_max_wrist_distance_ratio, 0.75)
        self.assertEqual(config.hand_gesture_confirm_frames, 3)

    def test_hand_tracking_settings_reject_out_of_range_and_invalid_values(self):
        invalid = (
            {"hand_detection_confidence": -0.1},
            {"hand_presence_confidence": 1.1},
            {"hand_tracking_confidence": float("nan")},
            {"hand_smoothing_alpha": 0.0},
            {"hand_max_wrist_distance_ratio": 0.0},
            {"hand_gesture_confirm_frames": 0},
            {"hand_gesture_confirm_frames": True},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                RoomSenseConfig(**values)


class HandTrackerVideoTests(unittest.TestCase):
    def make_tracker(self, *results):
        FakeLandmarker.results = list(results)
        FakeLandmarker.options = None
        FakeLandmarker.instance = None
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        model_path = Path(temp_dir.name) / "hand_landmarker.task"
        model_path.touch()
        media_pipe = fake_mediapipe()
        with patch.dict(sys.modules, {"mediapipe": media_pipe}), \
                patch.object(HandTracker, "_cached_model_path", return_value=model_path):
            tracker = HandTracker(RoomSenseConfig())
        return tracker, media_pipe

    def test_no_detected_hands_returns_empty_tuple(self):
        tracker, _ = self.make_tracker(result())

        self.assertEqual(tracker.process(object(), 100), ())

    def test_converts_left_and_right_with_all_21_points(self):
        points = open_hand_points()
        tracker, _ = self.make_tracker(result(("Left", points), ("Right", points)))

        hands = tracker.process(object(), 100)

        self.assertEqual([hand.handedness for hand in hands], ["Left", "Right"])
        self.assertEqual([len(hand.landmarks) for hand in hands], [21, 21])
        self.assertAlmostEqual(hands[0].tracking_confidence, 0.9)
        self.assertEqual(hands[0].fingertips["pinky"], hands[0].landmarks[20])

    def test_task_is_configured_for_two_hands_video_and_confidence_thresholds(self):
        tracker, _ = self.make_tracker(result())

        options = FakeLandmarker.options
        self.assertEqual(options.running_mode, "VIDEO")
        self.assertEqual(options.num_hands, 2)
        self.assertEqual(options.min_hand_detection_confidence, 0.5)
        self.assertEqual(options.min_hand_presence_confidence, 0.5)
        self.assertEqual(options.min_tracking_confidence, 0.5)
        self.assertTrue(Path(options.base_options.model_asset_path).exists())
        tracker.close()

    def test_video_timestamps_are_monotonic_when_input_time_repeats_or_rewinds(self):
        tracker, _ = self.make_tracker(result(), result())

        tracker.process(object(), 100)
        tracker.process(object(), 90)

        self.assertEqual(FakeLandmarker.instance.timestamps, [100, 101])

    def test_nonfinite_landmark_does_not_poison_smoothing_after_recovery(self):
        invalid_points = open_hand_points()
        invalid_points[8] = HandPoint(float("nan"), 0.3, 0.0)
        tracker, _ = self.make_tracker(result(("Left", invalid_points)), result(("Left", open_hand_points())))

        invalid_observation = tracker.process(object(), 100)[0]
        recovered = tracker.process(object(), 101)[0]

        self.assertIsNone(invalid_observation.palm_center)
        self.assertTrue(math.isfinite(recovered.landmarks[8].x))
        self.assertAlmostEqual(recovered.landmarks[8].x, 0.42)

    def test_close_and_reset_release_model_and_per_hand_state(self):
        tracker, _ = self.make_tracker(result(("Left", open_hand_points())))
        tracker.process(object(), 100)
        self.assertIn("Left", tracker._smoothers)

        tracker.reset_smoothing()
        self.assertEqual(tracker._smoothers, {})
        tracker.close()
        self.assertTrue(FakeLandmarker.instance.closed)


if __name__ == "__main__":
    unittest.main()
