import unittest
from unittest.mock import patch

import numpy as np

from roomsense.calibration.camera_calibration import create_calibration
from roomsense.actions.action_registry import ActionResult, MacAction
from roomsense.gestures.mac_controls import GestureStatus
from roomsense.interactions.events import EventType, RoomSenseEvent
from roomsense.interactions.modes import InteractionMode
from roomsense.spatial.room_objects import RoomObject
from roomsense.spatial.target_selection import TargetMatch, TargetUpdate
from roomsense.visualization.overlay import TrackingOverlay


class OverlayControlHintTests(unittest.TestCase):
    def test_live_overlay_displays_calibration_debug_reset_and_quit_controls(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        display = TrackingOverlay().draw(frame, None, None, "STILL", (), 0.0)
        control_band = display[620:644, 0:560]
        self.assertGreater(np.count_nonzero(control_band), 0)

    def test_live_hud_displays_mode_target_confidence_event_and_recording_status(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        item = RoomObject("monitor", "MONITOR", ((0.6, 0.2), (0.9, 0.2), (0.9, 0.5), (0.6, 0.5)),
                          enabled=True, room_position=(0.8, 0.2))
        target = TargetUpdate(TargetMatch(item, "right", 0.87, 0.3),
                              TargetMatch(item, "right", 0.87, 0.3), None)
        events = (RoomSenseEvent(EventType.ZONE_ENTERED, 1.0, {"zone": "DESK"}),)
        texts = []
        import cv2
        put_text = cv2.putText
        with patch("cv2.putText", wraps=put_text) as draw_text:
            TrackingOverlay().draw(
                frame, None, None, "STILL", (), 0.0,
                target=target, mode=InteractionMode.COMMAND, recent_events=events,
                recorder_active=True,
            )
        texts = [call.args[1] for call in draw_text.call_args_list]
        self.assertTrue(any("COMMAND" in text for text in texts))
        self.assertTrue(any("MONITOR" in text for text in texts))
        self.assertTrue(any("0.87" in text for text in texts))
        self.assertTrue(any("DESK" in text for text in texts))
        self.assertTrue(any("RECORDING" in text for text in texts))

    def test_compact_frame_renders_without_error(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        import cv2
        with patch("cv2.putText", wraps=cv2.putText) as draw_text:
            TrackingOverlay().draw(frame, None, None, "STILL", (), 0.0)
        self.assertTrue(any("MAP HIDDEN: ENLARGE" in call.args[1] for call in draw_text.call_args_list))
        self.assertGreater(np.count_nonzero(frame), 0)

    def test_room_map_accepts_registered_and_selected_objects(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        item = RoomObject("monitor", "MONITOR", ((0.6, 0.2), (0.9, 0.2), (0.9, 0.5), (0.6, 0.5)),
                          enabled=True, room_position=(0.8, 0.2))
        texts = []
        import cv2
        with patch("cv2.putText", wraps=cv2.putText) as draw_text:
            TrackingOverlay().draw(
                frame, None, None, "STILL", (), 0.0,
                calibration=create_calibration(((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)), (1280, 720)),
                objects=(item,), target=TargetUpdate(
                    TargetMatch(item, "left", 0.9, 0.3), TargetMatch(item, "left", 0.9, 0.3), None
                ),
            )
        texts = [call.args[1] for call in draw_text.call_args_list]
        self.assertTrue(any("MONITOR" in text for text in texts))
        self.assertTrue(any("MANUAL MAP CUE" in text for text in texts))

    def test_debug_panel_includes_smoothed_vector_and_active_gestures(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        texts = []
        import cv2
        with patch("cv2.putText", wraps=cv2.putText) as draw_text:
            TrackingOverlay().draw(
                frame, None, None, "STILL", (), 0.0, debug=True,
                debug_state={
                    "pointing": ({"direction": (0.8, 0.6)},),
                    "active_gestures": ("RIGHT_HOLD_POINT",),
                    "gesture_history": {"right": ((1.0, 0.5, 0.4, 0.9),)},
                    "cooldowns": {"POINT": 2.0},
                },
            )
        texts = [call.args[1] for call in draw_text.call_args_list]
        self.assertTrue(any("SMOOTH" in text and "0.8" in text for text in texts))
        self.assertTrue(any("RIGHT_HOLD_POINT" in text for text in texts))

    def test_hud_shows_command_mode_off_and_dry_run(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        import cv2
        with patch("cv2.putText", wraps=cv2.putText) as draw_text:
            TrackingOverlay().draw(frame, None, None, "STILL", (), 0.0,
                                   mode=InteractionMode.NORMAL, mac_controls_enabled=False)
        texts = [call.args[1] for call in draw_text.call_args_list]
        self.assertIn("COMMAND MODE: OFF", texts)
        self.assertIn("CONTROL: DRY RUN", texts)

    def test_hud_shows_gesture_action_confidence_and_cooldown(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        status = GestureStatus("PINCH", 0.91, 0.4, False, ())
        results = (ActionResult(MacAction.VOLUME_UP, 2.0, "ACTION: VOLUME UP", True, True),)
        import cv2
        with patch("cv2.putText", wraps=cv2.putText) as draw_text:
            TrackingOverlay().draw(frame, None, None, "STILL", (), 0.0,
                                   mode=InteractionMode.COMMAND, command_gesture=status,
                                   action_results=results)
        texts = [call.args[1] for call in draw_text.call_args_list]
        self.assertIn("COMMAND MODE: ON", texts)
        self.assertIn("GESTURE: PINCH", texts)
        self.assertIn("ACTION: VOLUME UP", texts)
        self.assertIn("COOLDOWN: 0.4s", texts)
        self.assertIn("CONFIDENCE: 0.91", texts)

    def test_action_log_is_visible_and_bounded(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        actions = tuple(ActionResult(MacAction.NEXT_TRACK, stamp, f"ACTION: TRACK {stamp}", True, True)
                        for stamp in range(6))
        import cv2
        with patch("cv2.putText", wraps=cv2.putText) as draw_text:
            TrackingOverlay().draw(frame, None, None, "STILL", (), 0.0,
                                   action_results=actions)
        texts = [call.args[1] for call in draw_text.call_args_list]
        action_rows = [text for text in texts if text.startswith("2.0s") or text.startswith("3.0s")
                       or text.startswith("4.0s") or text.startswith("5.0s")]
        self.assertEqual(len(action_rows), 3)
        self.assertTrue(any("ACTION FEED" in text for text in texts))
        self.assertFalse(any("0.0s ACTION" in text or "1.0s ACTION" in text for text in texts))


if __name__ == "__main__":
    unittest.main()
