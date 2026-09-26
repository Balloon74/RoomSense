"""Readable HUD and pose skeleton for the camera preview."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from roomsense.calibration.camera_calibration import Calibration
from roomsense.interactions.events import RoomSenseEvent
from roomsense.interactions.modes import InteractionMode
from roomsense.spatial.floor_position import FloorPosition
from roomsense.spatial.pointing import ArmPointing
from roomsense.spatial.position_history import PositionSample
from roomsense.spatial.room_objects import RoomObject
from roomsense.spatial.target_selection import TargetUpdate
from roomsense.spatial.zones import Zone
from roomsense.tracking.position_tracker import Position
from roomsense.tracking.pose_tracker import Landmark, SKELETON_CONNECTIONS
from roomsense.visualization.room_map import RoomMap


class TrackingOverlay:
    def __init__(self) -> None:
        self.room_map = RoomMap()

    def draw(self, frame: Any, landmarks: Mapping[str, Landmark] | None, position: Position | None,
             movement: str, states: Sequence[str], fps: float, lost: bool = False, *,
             calibration: Calibration | None = None, floor_position: FloorPosition | None = None,
             zones: Sequence[Zone] = (), current_zones: Sequence[str] = (),
             history: Sequence[PositionSample] = (), direction: tuple[float, float] = (0.0, 0.0),
             speed: float = 0.0, distance: float = 0.0, debug: bool = False,
             zone_transition: str | None = None, pointing: Sequence[ArmPointing] = (),
             target: TargetUpdate | None = None, mode: InteractionMode | str = InteractionMode.NORMAL,
             recent_events: Sequence[RoomSenseEvent] = (), demo_status: str | None = None,
             recorder_active: bool = False, objects: Sequence[RoomObject] = (),
             debug_state: Mapping[str, object] | None = None) -> Any:
        import cv2

        height, width = frame.shape[:2]
        if landmarks:
            self._draw_skeleton(frame, landmarks)
            points = [self._pixel_point(frame, point) for point in landmarks.values()]
            xs, ys = [point[0] for point in points], [point[1] for point in points]
            x0, y0 = max(0, min(xs) - 18), max(0, min(ys) - 24)
            x1, y1 = min(width - 1, max(xs) + 18), min(height - 1, max(ys) + 18)
            self._corner_box(frame, x0, y0, x1, y1)
        self._draw_pointing(frame, pointing)

        panel_h = min(162 if calibration is None else 184, max(40, height - 12))
        panel_right = max(16, min(370, width - 15))
        cv2.rectangle(frame, (15, 15), (panel_right, panel_h), (13, 22, 30), -1)
        cv2.rectangle(frame, (15, 15), (panel_right, panel_h), (58, 176, 194), 1)
        mode_name = mode.value if isinstance(mode, InteractionMode) else str(mode)
        cv2.putText(frame, f"ROOMSENSE  /  {mode_name}", (29, 39), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (113, 224, 234), 1, cv2.LINE_AA)
        color = (80, 230, 140) if landmarks else (60, 160, 240)
        status = "TRACKING  /  1 PERSON" if landmarks else ("SEARCHING FOR PERSON" if not lost else "TRACK LOST")
        cv2.circle(frame, (31, 61), 5, color, -1, cv2.LINE_AA)
        cv2.putText(frame, status, (44, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)
        if position:
            values = f"X {position.x:+.2f}    Y {position.y:+.2f}    Z {position.z:+.2f}*"
            confidence = f"CONFIDENCE {position.confidence * 100:.0f}%"
        else:
            values, confidence = "X --    Y --    Z --*", "CONFIDENCE --"
        cv2.putText(frame, values, (29, 91), cv2.FONT_HERSHEY_SIMPLEX, 0.49, (225, 238, 240), 1, cv2.LINE_AA)
        cv2.putText(frame, confidence, (29, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (155, 185, 192), 1, cv2.LINE_AA)
        if calibration is None:
            cv2.putText(frame, "CALIBRATION OFF  /  PRESS C", (29, 141), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (170, 190, 195), 1, cv2.LINE_AA)
        else:
            if floor_position is None:
                room_values = "ROOM X --  Y --"
            elif not floor_position.projectable:
                room_values = "ROOM POINT UNPROJECTABLE"
            else:
                room_values = (f"ROOM X {floor_position.render_x:.2f}  Y {floor_position.render_y:.2f}"
                               + ("  OUTSIDE FLOOR" if not floor_position.in_bounds else ""))
            cv2.putText(frame, room_values, (29, 141), cv2.FONT_HERSHEY_SIMPLEX,
                        0.40, (135, 220, 215), 1, cv2.LINE_AA)
            zone_line = "IN: " + (", ".join(current_zones) if current_zones else "NO ZONE")
            cv2.putText(frame, zone_line[:43], (29, 162), cv2.FONT_HERSHEY_SIMPLEX,
                        0.36, (185, 208, 210), 1, cv2.LINE_AA)
            if zone_transition:
                cv2.putText(frame, zone_transition[:42], (29, 180), cv2.FONT_HERSHEY_SIMPLEX,
                            0.34, (80, 220, 170), 1, cv2.LINE_AA)
        if debug and calibration is not None:
            self._draw_spatial_debug(frame, landmarks, calibration, floor_position)
        cv2.putText(frame, f"FPS {fps:04.1f}", (width - 112, height - 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.56, (125, 226, 230), 1, cv2.LINE_AA)
        movement_text = f"MOVEMENT: {movement}"
        pose_text = "POSE: " + (" / ".join(states) if states else "UNKNOWN")
        self._bottom_status(frame, movement_text, pose_text)
        self._draw_interaction_panel(frame, pointing, target, recent_events, demo_status, recorder_active)
        if debug and debug_state is not None:
            self._draw_interaction_debug(frame, debug_state)
        cv2.putText(frame, "C CALIBRATE   D DEBUG   R RESET   V RECORD   Q/ESC QUIT", (15, height - 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (165, 190, 195), 1, cv2.LINE_AA)
        if width >= 500 and height >= 350:
            self.room_map.draw(
                frame, position, calibrated=calibration is not None,
                room_point=(floor_position.render_x, floor_position.render_y)
                if floor_position and floor_position.projectable else None,
                in_bounds=floor_position.in_bounds if floor_position else True,
                zones=zones, current_zones=current_zones, history=history, direction=direction,
                speed=speed, distance=distance,
                objects=objects,
                selected_object=target.confirmed.object if target and target.confirmed else None,
            )
        else:
            cv2.putText(frame, "MAP HIDDEN: ENLARGE WINDOW", (max(15, width - 170), height - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (115, 180, 190), 1, cv2.LINE_AA)
        return frame

    def _draw_pointing(self, frame: Any, pointing: Sequence[ArmPointing]) -> None:
        import cv2

        height, width = frame.shape[:2]
        length = max(36, min(width, height) * 0.17)
        for arm in pointing:
            origin = (int(arm.origin[0] * width), int(arm.origin[1] * height))
            endpoint = (int(origin[0] + arm.direction[0] * length),
                        int(origin[1] + arm.direction[1] * length))
            color = (50, 235, 255) if arm.side == "left" else (255, 180, 70)
            cv2.arrowedLine(frame, origin, endpoint, color, 3, cv2.LINE_AA, tipLength=0.25)

    def _draw_interaction_panel(
        self, frame: Any, pointing: Sequence[ArmPointing], target: TargetUpdate | None,
        events: Sequence[RoomSenseEvent], demo_status: str | None, recorder_active: bool,
    ) -> None:
        import cv2

        height, width = frame.shape[:2]
        left, top = 15, 190
        if height < 360:
            top = max(18, height - 66)
            bottom = height - 18
            panel_width = min(270, max(150, width - 30))
            cv2.rectangle(frame, (left, top), (min(width - 1, left + panel_width), bottom), (13, 22, 30), -1)
            cv2.rectangle(frame, (left, top), (min(width - 1, left + panel_width), bottom), (58, 176, 194), 1)
            chosen = (target.confirmed or target.candidate) if target else None
            label = chosen.object.name if chosen else ",".join(arm.side.upper() for arm in pointing) or "--"
            cv2.putText(frame, f"POINTING: {label}"[:32], (left + 9, top + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (245, 225, 120), 1, cv2.LINE_AA)
            session = "REC" if recorder_active else "NOT REC"
            cv2.putText(frame, f"SESSION: {session}  EVENTS: {len(events)}", (left + 9, top + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (145, 205, 190), 1, cv2.LINE_AA)
            return
        panel_width = min(430, max(150, width - 30))
        bottom = min(height - 18, 360)
        if bottom > top:
            cv2.rectangle(frame, (left, top), (left + panel_width, bottom), (13, 22, 30), -1)
            cv2.rectangle(frame, (left, top), (left + panel_width, bottom), (58, 176, 194), 1)
        confirmed = target.confirmed if target else None
        candidate = target.candidate if target else None
        chosen = confirmed or candidate
        if chosen:
            pointing_line = f"POINTING: {chosen.object.name}"
            confidence = f"CONFIDENCE: {chosen.confidence:.2f}"
        else:
            sides = ",".join(arm.side.upper() for arm in pointing)
            pointing_line = f"POINTING: {sides}" if sides else "POINTING: --"
            confidence = (f"CONFIDENCE: {max((arm.confidence for arm in pointing), default=0.0):.2f}"
                          if pointing else "CONFIDENCE: --")
        cv2.putText(frame, pointing_line, (left + 12, top + 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (245, 225, 120), 1, cv2.LINE_AA)
        cv2.putText(frame, confidence, (left + 12, top + 47), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (190, 220, 225), 1, cv2.LINE_AA)
        if demo_status:
            cv2.putText(frame, f"ACTION: {demo_status}"[:48], (left + 12, top + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (90, 235, 155), 1, cv2.LINE_AA)
        status_y = top + 91 if demo_status else top + 70
        cv2.putText(frame, "SESSION: RECORDING" if recorder_active else "SESSION: NOT RECORDING",
                    (left + 12, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    (80, 220, 160) if recorder_active else (145, 165, 170), 1, cv2.LINE_AA)
        feed_y = status_y + 22
        cv2.putText(frame, "RECENT EVENTS", (left + 12, feed_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.34, (115, 205, 215), 1, cv2.LINE_AA)
        visible_rows = max(0, (bottom - (feed_y + 5)) // 17) if bottom > top else 4
        for index, event in enumerate(reversed(events[-min(4, visible_rows):])):
            line = f"{event.display_time} {event.type.value.replace('_', ' ')}"
            detail = event.metadata.get("name") or event.metadata.get("zone") or event.metadata.get("gesture")
            if isinstance(detail, str):
                line += f" {detail}"
            cv2.putText(frame, line[:54], (left + 12, feed_y + 18 + index * 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.31, (200, 215, 215), 1, cv2.LINE_AA)

    def _draw_interaction_debug(self, frame: Any, debug_state: Mapping[str, object]) -> None:
        import cv2

        height, _ = frame.shape[:2]
        raw = str(debug_state.get("raw_arm_vectors", {}))[:65]
        pointing_values = debug_state.get("pointing", ())
        smooth = str(pointing_values[0].get("direction"))[:32] if pointing_values else "--"
        candidate = debug_state.get("candidate_target") or "--"
        confirmed = debug_state.get("confirmed_target") or "--"
        confidence = debug_state.get("pointing_confidence", 0.0)
        history = str(debug_state.get("gesture_history", ()))[:48]
        active_gestures = str(debug_state.get("active_gestures", ()))[:32]
        cooldowns = str(debug_state.get("cooldowns", {}))[:42]
        cv2.putText(frame, f"ARM {raw}  SMOOTH {smooth}  CONF {confidence}  CAND {candidate}  TARGET {confirmed}",
                    (15, max(24, height - 108)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.28, (220, 235, 230), 1, cv2.LINE_AA)
        cv2.putText(frame, f"GESTURE HISTORY {history} ACTIVE {active_gestures} COOLDOWNS {cooldowns}",
                    (15, max(36, height - 94)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.28, (220, 235, 230), 1, cv2.LINE_AA)

    def _draw_spatial_debug(self, frame: Any, landmarks: Mapping[str, Landmark] | None,
                            calibration: Calibration, floor_position: FloorPosition | None) -> None:
        import cv2

        height, width = frame.shape[:2]
        polygon = [(int(x * width), int(y * height)) for x, y in calibration.image_points]
        cv2.polylines(frame, [__import__("numpy").array(polygon, dtype="int32")], True,
                      (185, 85, 230), 2, cv2.LINE_AA)
        for side in ("left", "right"):
            point = landmarks.get(f"{side}_ankle") if landmarks else None
            if point:
                cv2.circle(frame, self._pixel_point(frame, point), 7, (245, 110, 70), 2, cv2.LINE_AA)
        if floor_position:
            raw = (int(floor_position.raw_image_x * width), int(floor_position.raw_image_y * height))
            smooth = (int(floor_position.smoothed_image_x * width), int(floor_position.smoothed_image_y * height))
            cv2.circle(frame, raw, 8, (40, 90, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, smooth, 8, (40, 255, 100), -1, cv2.LINE_AA)
            room_debug = (f"ROOM {floor_position.room_x:.2f},{floor_position.room_y:.2f} "
                          f"MAP {floor_position.render_x:.2f},{floor_position.render_y:.2f}"
                          if floor_position.projectable else "ROOM UNPROJECTABLE")
            text = (f"RAW {floor_position.raw_image_x:.2f},{floor_position.raw_image_y:.2f}  "
                    f"SMOOTH {floor_position.smoothed_image_x:.2f},{floor_position.smoothed_image_y:.2f}  "
                    f"{room_debug}  CONF {floor_position.confidence:.2f}")
            cv2.putText(frame, text, (20, height - 88), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (230, 240, 240), 1, cv2.LINE_AA)

    def _draw_skeleton(self, frame: Any, landmarks: Mapping[str, Landmark]) -> None:
        import cv2

        for start, end in SKELETON_CONNECTIONS:
            if start in landmarks and end in landmarks:
                cv2.line(frame, self._pixel_point(frame, landmarks[start]), self._pixel_point(frame, landmarks[end]),
                         (74, 222, 209), 3, cv2.LINE_AA)
        for point in landmarks.values():
            cv2.circle(frame, self._pixel_point(frame, point), 5, (40, 245, 255), -1, cv2.LINE_AA)

    @staticmethod
    def _pixel_point(frame: Any, landmark: Landmark) -> tuple[int, int]:
        height, width = frame.shape[:2]
        return int(landmark.x * width), int(landmark.y * height)

    @staticmethod
    def _corner_box(frame: Any, x0: int, y0: int, x1: int, y1: int) -> None:
        import cv2

        color, size, thickness = (0, 235, 245), 18, 3
        for x, y, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            cv2.line(frame, (x, y), (x + dx * size, y), color, thickness, cv2.LINE_AA)
            cv2.line(frame, (x, y), (x, y + dy * size), color, thickness, cv2.LINE_AA)

    @staticmethod
    def _bottom_status(frame: Any, movement: str, pose: str) -> None:
        import cv2

        height, width = frame.shape[:2]
        cv2.rectangle(frame, (15, height - 74), (min(width - 15, 545), height - 15), (13, 22, 30), -1)
        cv2.putText(frame, movement, (29, height - 49), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (231, 238, 238), 1, cv2.LINE_AA)
        cv2.putText(frame, pose[:48], (29, height - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (121, 216, 221), 1, cv2.LINE_AA)
