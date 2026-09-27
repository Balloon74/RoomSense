import unittest
from unittest.mock import patch

import numpy as np

from roomsense.calibration.camera_calibration import create_calibration
from roomsense.interactions.events import EventType, RoomSenseEvent
from roomsense.interactions.modes import InteractionMode
from roomsense.spatial.room_objects import RoomObject
from roomsense.spatial.target_selection import TargetMatch, TargetUpdate
from roomsense.tracking.reidentification import CandidateScore
from roomsense.visualization.overlay import TrackingOverlay


class OverlayControlHintTests(unittest.TestCase):
    def test_person_id_state_and_match_factors_are_visible_in_hud_and_debug(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        candidate = CandidateScore(
            "PERSON_001",
            {"trajectory": 0.91, "reentry_direction": 0.84,
             "elapsed_time": 0.95, "torso_geometry": 0.88},
            0.89,
            "ambiguous",
        )
        second_candidate = CandidateScore(
            "PERSON_002",
            {"trajectory": 0.88, "reentry_direction": 0.82,
             "elapsed_time": 0.94, "torso_geometry": 0.86},
            0.86,
            "ambiguous",
        )
        import cv2
        with patch("cv2.putText", wraps=cv2.putText) as draw_text:
            TrackingOverlay().draw(
                frame, None, None, "STILL", (), 0.0, debug=True,
                person_id="PERSON_001", person_state="REACQUIRED",
                reidentification_debug={
                    "reason": "ambiguous_candidates",
                    "threshold": 0.72,
                    "ambiguity_margin": 0.12,
                    "candidates": (candidate, second_candidate),
                },
            )
        texts = [call.args[1] for call in draw_text.call_args_list]
        self.assertTrue(any("PERSON_001" in text and "REACQUIRED" in text for text in texts))
        self.assertTrue(any("TRAJ" in text and "0.91" in text and "0.89" in text for text in texts))
        self.assertTrue(any("PERSON_002" in text and "0.86" in text for text in texts))
        self.assertTrue(any("0.72" in text and "0.12" in text and "AMBIGUOUS" in text for text in texts))
        for state in ("NEW", "LOST", "REACQUIRED"):
            with self.subTest(state=state), patch("cv2.putText", wraps=cv2.putText) as draw_status:
                TrackingOverlay().draw(
                    frame.copy(), None, None, "STILL", (), 0.0,
                    person_id="PERSON_001", person_state=state,
                )
            status_text = [call.args[1] for call in draw_status.call_args_list]
            self.assertTrue(any("PERSON_001" in text and state in text for text in status_text))

    def test_compact_reidentification_debug_panel_does_not_cover_person_hud(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        candidate = CandidateScore("PERSON_001", {"trajectory": 0.8}, 0.8, "selected")
        import cv2
        with patch("cv2.rectangle", wraps=cv2.rectangle) as draw_rect:
            TrackingOverlay().draw(
                frame, None, None, "STILL", (), 0.0, debug=True,
                person_id="PERSON_001", person_state="REACQUIRED",
                reidentification_debug={"reason": "reacquired", "candidates": (candidate,)},
            )
        reid_panel_top_left, reid_panel_bottom_right = draw_rect.call_args_list[-1].args[1:3]
        self.assertGreaterEqual(reid_panel_top_left[1], 162)
        self.assertLess(reid_panel_top_left[0], reid_panel_bottom_right[0])

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


if __name__ == "__main__":
    unittest.main()
