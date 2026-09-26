import unittest

from roomsense.gestures.temporal_gestures import GestureTransition
from roomsense.interactions.modes import InteractionMode, InteractionModeController


def hands_up(timestamp):
    return (GestureTransition("BOTH_HANDS_UP", timestamp, 0.95, {}),)


class InteractionModeTests(unittest.TestCase):
    def test_starts_in_normal_and_held_both_hands_enters_command_once(self):
        controller = InteractionModeController(command_timeout_seconds=2.0)
        self.assertEqual(controller.mode, InteractionMode.NORMAL)
        transition = controller.update(hands_up(0.5), 0.5)
        self.assertEqual((transition.previous, transition.current), (InteractionMode.NORMAL, InteractionMode.COMMAND))
        self.assertEqual(transition.reason, "BOTH_HANDS_UP")
        self.assertIsNone(controller.update((), 0.6))

    def test_held_both_hands_exits_command_once(self):
        controller = InteractionModeController(command_timeout_seconds=2.0)
        controller.update(hands_up(0.0), 0.0)
        transition = controller.update(hands_up(0.5), 0.5)
        self.assertEqual((transition.previous, transition.current), (InteractionMode.COMMAND, InteractionMode.NORMAL))
        self.assertEqual(transition.reason, "BOTH_HANDS_UP")
        self.assertIsNone(controller.update((), 0.6))

    def test_command_mode_times_out_once_after_inactivity(self):
        controller = InteractionModeController(command_timeout_seconds=1.0)
        controller.update(hands_up(0.0), 0.0)
        self.assertIsNone(controller.update((), 0.99))
        transition = controller.update((), 1.0)
        self.assertEqual(transition.current, InteractionMode.NORMAL)
        self.assertEqual(transition.reason, "INACTIVITY_TIMEOUT")
        self.assertIsNone(controller.update((), 1.1))

    def test_activity_refreshes_timeout_and_unchanged_mode_emits_nothing(self):
        controller = InteractionModeController(command_timeout_seconds=1.0)
        controller.update(hands_up(0.0), 0.0)
        self.assertIsNone(controller.update((), 0.9, activity=True))
        self.assertIsNone(controller.update((), 1.8))
        self.assertEqual(controller.update((), 1.91).current, InteractionMode.NORMAL)

    def test_rejects_non_finite_or_decreasing_timestamps(self):
        controller = InteractionModeController(command_timeout_seconds=1.0)
        with self.assertRaises(ValueError):
            controller.update((), float("nan"))
        controller.update((), 1.0)
        with self.assertRaises(ValueError):
            controller.update((), 0.0)


if __name__ == "__main__":
    unittest.main()
