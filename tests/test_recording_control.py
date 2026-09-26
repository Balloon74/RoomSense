import unittest

from roomsense.recording.recording_control import RecordingController


class FakeRecorder:
    def __init__(self):
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def close(self):
        self.closed = True


class RecordingControllerTests(unittest.TestCase):
    def test_starts_off_and_toggles_start_and_stop(self):
        made = []

        def factory():
            recorder = FakeRecorder()
            made.append(recorder)
            return recorder

        controller = RecordingController(factory)
        self.assertFalse(controller.active)
        self.assertTrue(controller.toggle())
        self.assertTrue(controller.active)
        self.assertTrue(made[0].started)
        self.assertFalse(controller.toggle())
        self.assertFalse(controller.active)
        self.assertTrue(made[0].closed)

    def test_close_stops_active_recorder_and_is_safe_when_off(self):
        made = []
        controller = RecordingController(lambda: made.append(FakeRecorder()) or made[-1])
        controller.close()
        controller.toggle()
        controller.close()
        self.assertFalse(controller.active)
        self.assertTrue(made[0].closed)


if __name__ == "__main__":
    unittest.main()
