import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from roomsense.calibration.camera_calibration import create_calibration
from roomsense.interactions.events import EventType, RoomSenseEvent
from roomsense.interactions.modes import InteractionMode
from roomsense.spatial.room_objects import RoomObject
from roomsense.spatial.target_selection import TargetMatch, TargetUpdate
from roomsense.tracking.hand_geometry import HAND_CONNECTIONS, HandState, analyze_hand
from roomsense.tracking.pose_tracker import Landmark
from roomsense.visualization.overlay import TrackingOverlay
from test_hand_geometry import open_hand_points


class OverlayControlHintTests(unittest.TestCase):
    def test_hand_landmarks_connect_to_associated_body_wrist_and_show_state(self):
        frame = np.zeros((600, 900, 3), dtype=np.uint8)
        hand = replace(analyze_hand(open_hand_points(), "Left", 0.9),
                       state=HandState.OPEN_PALM, body_side="left", association_confidence=0.8)
        body = {"left_wrist": Landmark(0.25, 0.75)}
        import cv2
        with patch("cv2.line", wraps=cv2.line) as draw_line, \
                patch("cv2.putText", wraps=cv2.putText) as draw_text:
            TrackingOverlay().draw(frame, body, None, "STILL", (), 0.0, hands=(hand,))

        hand_wrist = (int(0.5 * 900), int(0.8 * 600))
        body_wrist = (int(0.25 * 900), int(0.75 * 600))
        drawn_lines = [call.args[1:3] for call in draw_line.call_args_list]
        self.assertTrue(any(set(pair) == {hand_wrist, body_wrist} for pair in drawn_lines))
        self.assertGreaterEqual(draw_line.call_count, len(HAND_CONNECTIONS) + 1)
        self.assertTrue(any("LEFT OPEN PALM" in call.args[1] for call in draw_text.call_args_list))
        self.assertGreater(np.count_nonzero(frame), 0)

    def test_debug_overlay_exposes_hand_fingers_pinch_confidence_orientation_and_association(self):
        frame = np.zeros((600, 900, 3), dtype=np.uint8)
        hand = replace(analyze_hand(open_hand_points(), "Right", 0.86),
                       state=HandState.PEACE_SIGN, body_side="right", association_confidence=0.72)
        import cv2
        with patch("cv2.putText", wraps=cv2.putText) as draw_text:
            TrackingOverlay().draw(frame, None, None, "STILL", (), 0.0, debug=True, hands=(hand,))

        texts = [call.args[1] for call in draw_text.call_args_list]
        self.assertTrue(any("FINGERS" in text for text in texts))
        self.assertTrue(any("PINKY:EXTENDED" in text for text in texts))
        self.assertTrue(any("PINCH" in text for text in texts))
        self.assertTrue(any("CONF" in text for text in texts))
        self.assertTrue(any("PALM" in text and "NORMAL" in text for text in texts))
        self.assertTrue(any("BODY RIGHT" in text for text in texts))

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

    def test_hud_shows_evaluation_state_and_pointing_diagnostic(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        import cv2
        with patch("cv2.putText", wraps=cv2.putText) as draw_text:
            TrackingOverlay().draw(
                frame, None, None, "STILL", (), 0.0,
                evaluation_active=True,
                pointing_status="LOW ARM VISIBILITY",
            )
        texts = [call.args[1] for call in draw_text.call_args_list]
        self.assertTrue(any("EVALUATION CAPTURE" in text for text in texts))
        self.assertTrue(any("LOW ARM VISIBILITY" in text for text in texts))


if __name__ == "__main__":
    unittest.main()
