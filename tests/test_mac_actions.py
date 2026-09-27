import subprocess
import sys
import unittest
from unittest.mock import call, patch

from roomsense.actions.action_registry import ActionIntent, ActionRegistry, MacAction
from roomsense.actions.macos_controller import MacOSController


class SpyController:
    def __init__(self):
        self.intents = []

    def execute(self, intent):
        self.intents.append(intent)


def intent(action=MacAction.NEXT_TRACK, timestamp=1.0):
    return ActionIntent(action, "SWIPE_RIGHT", timestamp, 0.91)


class MacActionTests(unittest.TestCase):
    def test_default_registry_returns_dry_run_message_without_calling_controller(self):
        controller = SpyController()
        registry = ActionRegistry(controller=controller)
        result = registry.dispatch(intent())
        self.assertEqual(result.message, "ACTION: NEXT TRACK")
        self.assertTrue(result.dry_run)
        self.assertTrue(result.succeeded)
        self.assertEqual(controller.intents, [])

    def test_enabled_registry_dispatches_to_injected_controller(self):
        controller = SpyController()
        with patch("roomsense.actions.action_registry.sys.platform", "darwin"):
            registry = ActionRegistry(mac_controls_enabled=True, controller=controller)
            result = registry.dispatch(intent(MacAction.PLAY_PAUSE))
        self.assertEqual(controller.intents, [intent(MacAction.PLAY_PAUSE)])
        self.assertEqual(result.message, "EXECUTED: PLAY/PAUSE")
        self.assertFalse(result.dry_run)
        self.assertTrue(result.succeeded)

    def test_registry_keeps_bounded_action_history(self):
        registry = ActionRegistry(history_size=2)
        results = [registry.dispatch(intent(timestamp=value)) for value in (1.0, 2.0, 3.0)]
        self.assertEqual(registry.history, tuple(results[-2:]))
        self.assertEqual([item.timestamp for item in registry.history], [2.0, 3.0])

    def test_macos_controller_maps_media_actions_to_osascript_keys(self):
        expected = (
            (MacAction.PREVIOUS_TRACK, "key code 98"),
            (MacAction.PLAY_PAUSE, "key code 100"),
            (MacAction.NEXT_TRACK, "key code 101"),
        )
        for action, expected_script in expected:
            with self.subTest(action=action), patch("roomsense.actions.macos_controller.subprocess.run") as run:
                MacOSController().execute(intent(action))
                self.assertEqual(run.call_args.args[0][:2], ["osascript", "-e"])
                self.assertIn(expected_script, run.call_args.args[0][2])
                self.assertTrue(run.call_args.kwargs["check"])

    def test_macos_controller_quantizes_and_clamps_volume(self):
        readings = iter(("98\n", "2\n"))

        def run_osascript(args, **_kwargs):
            output = next(readings) if args[2] == "output volume of (get volume settings)" else ""
            return subprocess.CompletedProcess(args=args, returncode=0, stdout=output, stderr="")

        with patch("roomsense.actions.macos_controller.subprocess.run", side_effect=run_osascript) as run:
            controller = MacOSController(volume_step_percent=5)
            controller.execute(intent(MacAction.VOLUME_UP))
            controller.execute(intent(MacAction.VOLUME_DOWN, timestamp=2.0))

        scripts = [entry.args[0][2] for entry in run.call_args_list]
        self.assertEqual(scripts[0], "output volume of (get volume settings)")
        self.assertIn("set volume output volume 100", scripts[1])
        self.assertIn("set volume output volume 0", scripts[3])
        self.assertTrue(all(entry.args[0][0] == "osascript" for entry in run.call_args_list))

    def test_macos_controller_reports_subprocess_failure(self):
        failure = subprocess.CalledProcessError(1, ["osascript"], stderr="denied")
        with patch("roomsense.actions.macos_controller.subprocess.run", side_effect=failure):
            registry = ActionRegistry(mac_controls_enabled=True, controller=MacOSController())
            result = registry.dispatch(intent())
        self.assertFalse(result.succeeded)
        self.assertFalse(result.dry_run)
        self.assertTrue(result.message.startswith("ERROR:"))

    def test_non_macos_real_control_activation_is_rejected(self):
        with patch("roomsense.actions.action_registry.sys.platform", "linux"):
            with self.assertRaisesRegex(RuntimeError, "macOS"):
                ActionRegistry(mac_controls_enabled=True, controller=SpyController())


if __name__ == "__main__":
    unittest.main()
