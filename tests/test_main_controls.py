import builtins
import unittest
from unittest.mock import patch

from roomsense.config import RoomSenseConfig
from roomsense.main import _resolve_mac_controls_config, run


class Frame:
    shape = (720, 1280, 3)

    def copy(self):
        return self


class FakeCamera:
    def __init__(self, *args):
        self.frames = 0
        self.released = False

    def isOpened(self):
        return True

    def set(self, *_args):
        return True

    def read(self):
        self.frames += 1
        return True, Frame()

    def release(self):
        self.released = True


class FakeCV2:
    CAP_ANY = 0
    CAP_AVFOUNDATION = 1
    CAP_PROP_FRAME_WIDTH = 2
    CAP_PROP_FRAME_HEIGHT = 3
    CAP_PROP_FPS = 4
    CAP_PROP_BUFFERSIZE = 5
    WINDOW_NORMAL = 6
    COLOR_BGR2RGB = 7

    def VideoCapture(self, *_args):
        self.camera = FakeCamera()
        return self.camera

    def flip(self, frame, _axis):
        return frame

    def cvtColor(self, frame, _conversion):
        return frame

    def namedWindow(self, *_args):
        pass

    def imshow(self, *_args):
        pass

    def waitKey(self, _delay):
        return ord("q")

    def destroyAllWindows(self):
        pass


class FakePoseTracker:
    instances = []

    def __init__(self, _settings):
        self.processed = 0
        self.closed = False
        self.instances.append(self)

    def process(self, *_args):
        self.processed += 1
        return None

    def close(self):
        self.closed = True


class FakeHandTracker:
    def __init__(self):
        self.processed = []
        self.closed = False

    def process(self, frame, timestamp):
        self.processed.append((frame, timestamp))
        return ()

    def close(self):
        self.closed = True


class FakeOverlay:
    def draw(self, frame, *_args, **_kwargs):
        return frame


class FakeRecordingController:
    active = False

    def __init__(self, *_args):
        pass

    def record(self, *_args):
        pass

    def close(self):
        pass


class MainControlsTests(unittest.TestCase):
    def test_mac_control_flags_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            _resolve_mac_controls_config(RoomSenseConfig(), ["--enable-mac-controls", "--dry-run"])

    def test_cli_flag_overrides_config_and_default_is_dry_run(self):
        self.assertFalse(_resolve_mac_controls_config(RoomSenseConfig(), []).mac_controls_enabled)
        configured = RoomSenseConfig(mac_controls_enabled=True)
        self.assertTrue(_resolve_mac_controls_config(configured, []).mac_controls_enabled)
        self.assertTrue(_resolve_mac_controls_config(RoomSenseConfig(), ["--enable-mac-controls"])
                        .mac_controls_enabled)
        self.assertFalse(_resolve_mac_controls_config(configured, ["--dry-run"]).mac_controls_enabled)

    def _run_with_hand_setup(self, factory, injected_tracker=None):
        import numpy

        fake_cv2 = FakeCV2()
        fake_numpy = type("NumpyProxy", (), {"ascontiguousarray": staticmethod(lambda value: value)})
        original_import = builtins.__import__

        def import_proxy(name, *args, **kwargs):
            if name == "cv2":
                return fake_cv2
            if name == "numpy":
                return fake_numpy
            return original_import(name, *args, **kwargs)

        FakePoseTracker.instances.clear()
        with patch("builtins.__import__", side_effect=import_proxy), \
             patch("roomsense.main.PoseTracker", FakePoseTracker), \
             patch("roomsense.main.TrackingOverlay", FakeOverlay), \
             patch("roomsense.main.RecordingController", FakeRecordingController):
            result = run(RoomSenseConfig(camera_index=0, calibration_path="/tmp/roomsense-no-calibration.json"),
                         argv=[], hand_tracker=injected_tracker, hand_tracker_factory=factory)
        return result, fake_cv2

    def test_run_with_injected_hand_tracker_preserves_pose_pipeline(self):
        tracker = FakeHandTracker()
        result, _camera = self._run_with_hand_setup(lambda _settings: None, tracker)
        self.assertEqual(result, 0)
        self.assertEqual(FakePoseTracker.instances[0].processed, 1)
        self.assertTrue(FakePoseTracker.instances[0].closed)
        self.assertEqual(len(tracker.processed), 1)
        self.assertTrue(tracker.closed)

    def test_hand_tracker_unavailable_leaves_pose_tracking_running(self):
        def unavailable(_settings):
            raise RuntimeError("model unavailable")

        result, _camera = self._run_with_hand_setup(unavailable)
        self.assertEqual(result, 0)
        self.assertEqual(FakePoseTracker.instances[0].processed, 1)
        self.assertTrue(FakePoseTracker.instances[0].closed)

    def test_real_controls_are_rejected_before_camera_opens_off_macos(self):
        camera = FakeCV2()
        original_import = builtins.__import__

        def import_proxy(name, *args, **kwargs):
            if name == "cv2":
                return camera
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_proxy), \
             patch("roomsense.main.sys.platform", "linux"):
            result = run(RoomSenseConfig(camera_index=0), argv=["--enable-mac-controls"])
        self.assertEqual(result, 1)
        self.assertFalse(hasattr(camera, "camera"))


if __name__ == "__main__":
    unittest.main()
