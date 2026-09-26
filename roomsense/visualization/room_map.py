"""Compact top-down X / estimated-depth room map."""

from __future__ import annotations

from typing import Any, Sequence

from roomsense.spatial.position_history import PositionSample
from roomsense.spatial.room_objects import RoomObject
from roomsense.spatial.zones import Zone
from roomsense.tracking.position_tracker import Position


class RoomMap:
    def __init__(self, width: int = 260, height: int = 190) -> None:
        self.width = width
        self.height = height

    def draw(
        self,
        frame: Any,
        position: Position | None,
        *,
        calibrated: bool = False,
        room_point: tuple[float, float] | None = None,
        in_bounds: bool = True,
        zones: Sequence[Zone] = (),
        current_zones: Sequence[str] = (),
        history: Sequence[PositionSample] = (),
        direction: tuple[float, float] = (0.0, 0.0),
        speed: float = 0.0,
        distance: float = 0.0,
        objects: Sequence[RoomObject] = (),
        selected_object: RoomObject | None = None,
    ) -> None:
        import cv2

        height, width = frame.shape[:2]
        map_width = min(self.width, max(120, width - 24))
        map_height = min(self.height, max(100, height - 44))
        x0, y0 = max(12, width - map_width - 22), 22
        x1, y1 = min(width - 1, x0 + map_width), min(height - 1, y0 + map_height)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (16, 25, 34), -1)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (68, 190, 210), 1)
        title = "ROOM MAP  /  NORMALIZED" if calibrated else "ROOM MAP  /  TOP VIEW"
        cv2.putText(frame, title, (x0 + 12, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (118, 220, 230), 1, cv2.LINE_AA)
        left, right = x0 + 28, x1 - 18
        top, bottom = y0 + 42, y1 - 32
        cv2.rectangle(frame, (left, top), (right, bottom), (28, 41, 50), 1)
        if calibrated:
            cv2.putText(frame, "BACK", (left + 4, top + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                        (130, 155, 164), 1, cv2.LINE_AA)
            cv2.putText(frame, "FRONT", (left + 4, bottom - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                        (130, 155, 164), 1, cv2.LINE_AA)
            self._draw_calibrated(frame, (left, top, right, bottom), zones, current_zones,
                                  history, room_point, in_bounds, direction, objects, selected_object)
            cv2.putText(frame, f"SPEED {speed:.2f} u/s   DIST {distance:.2f} u",
                        (x0 + 12, y1 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                        (115, 180, 190), 1, cv2.LINE_AA)
            return
        center_x = (left + right) // 2
        cv2.line(frame, (center_x, top), (center_x, bottom), (44, 62, 71), 1)
        cv2.putText(frame, "FAR", (left + 4, top + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    (130, 155, 164), 1, cv2.LINE_AA)
        cv2.putText(frame, "CAMERA", (left + 4, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    (130, 155, 164), 1, cv2.LINE_AA)
        cv2.putText(frame, "X  -1          0          +1", (left, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.34, (148, 173, 180), 1, cv2.LINE_AA)
        if position is None:
            cv2.putText(frame, "NO TRACK", (left + 12, (top + bottom) // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (120, 150, 160), 1, cv2.LINE_AA)
            return
        px = int(left + (position.x + 1.0) * 0.5 * (right - left))
        py = int(top + ((position.z + 1.0) * 0.5) * (bottom - top))
        cv2.circle(frame, (px, py), 8, (0, 245, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 13, (0, 145, 175), 1, cv2.LINE_AA)
        cv2.putText(frame, "DEPTH ESTIMATED", (x0 + 12, y1 + 17), cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, (115, 180, 190), 1, cv2.LINE_AA)

    def _draw_calibrated(
        self, frame: Any, bounds: tuple[int, int, int, int], zones: Sequence[Zone],
        current_zones: Sequence[str], history: Sequence[PositionSample],
        room_point: tuple[float, float] | None, in_bounds: bool,
        direction: tuple[float, float],
        objects: Sequence[RoomObject], selected_object: RoomObject | None,
    ) -> None:
        import cv2

        left, top, right, bottom = bounds
        width, height = right - left, bottom - top
        to_pixel = lambda point: (int(left + point[0] * width), int(top + point[1] * height))
        cv2.rectangle(frame, (left, top), (right, bottom), (90, 145, 150), 1)
        colors = ((78, 120, 145), (120, 100, 76), (88, 122, 98), (92, 99, 128))
        current = set(current_zones)
        for index, zone in enumerate(zones):
            polygon = [to_pixel(point) for point in zone.polygon]
            color = (40, 160, 190) if zone.name in current else colors[index % len(colors)]
            cv2.polylines(frame, [__import__("numpy").array(polygon, dtype="int32")], True, color, 1, cv2.LINE_AA)
            center = (int(sum(point[0] for point in polygon) / len(polygon)),
                      int(sum(point[1] for point in polygon) / len(polygon)))
            cv2.putText(frame, zone.name, center, cv2.FONT_HERSHEY_SIMPLEX, 0.31, color, 1, cv2.LINE_AA)
        for item in objects:
            if not item.enabled or item.room_position is None:
                continue
            object_pixel = to_pixel(item.room_position)
            selected = selected_object is not None and item.id == selected_object.id
            color = (20, 200, 255) if selected else (155, 185, 195)
            cv2.circle(frame, object_pixel, 7 if selected else 5, color, -1 if selected else 1, cv2.LINE_AA)
            cv2.putText(frame, item.name[:12], (object_pixel[0] + 6, object_pixel[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.29, color, 1, cv2.LINE_AA)
        if selected_object is not None and selected_object.enabled and selected_object.room_position is not None:
            cv2.putText(frame, "OBJECT ARROW: MANUAL MAP CUE", (left + 2, top + 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.27, (20, 200, 255), 1, cv2.LINE_AA)
        samples = list(history)
        for index in range(1, len(samples)):
            fade = index / max(1, len(samples) - 1)
            color = (int(25 + 80 * fade), int(75 + 125 * fade), int(110 + 125 * fade))
            cv2.line(frame, to_pixel((samples[index - 1].x, samples[index - 1].y)),
                     to_pixel((samples[index].x, samples[index].y)), color, 2, cv2.LINE_AA)
        if room_point is None:
            cv2.putText(frame, "NO TRACK", (left + 12, (top + bottom) // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (120, 150, 160), 1, cv2.LINE_AA)
            return
        px, py = to_pixel(room_point)
        cv2.circle(frame, (px, py), 8, (0, 245, 255) if in_bounds else (80, 140, 240), -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 13, (0, 145, 175), 1, cv2.LINE_AA)
        length = (direction[0] ** 2 + direction[1] ** 2) ** 0.5
        if length > 1e-4:
            dx, dy = direction[0] / length, direction[1] / length
            cv2.arrowedLine(frame, (px, py), (int(px + dx * 20), int(py + dy * 20)),
                            (145, 235, 238), 2, cv2.LINE_AA, tipLength=0.35)
        if selected_object is not None and selected_object.enabled and selected_object.room_position is not None:
            cv2.arrowedLine(frame, (px, py), to_pixel(selected_object.room_position),
                            (20, 200, 255), 1, cv2.LINE_AA, tipLength=0.2)
