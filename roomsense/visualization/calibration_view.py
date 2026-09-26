"""Frozen-frame corner selection UI and its camera-free click state."""

from __future__ import annotations

import sys

from roomsense.calibration.camera_calibration import Calibration, Point, create_calibration


_CORNER_NAMES = ("back-left", "back-right", "front-right", "front-left")


def _frame_coordinates(
    x: int, y: int, image_size: tuple[int, int], display_size: tuple[int, int],
    platform: str | None = None,
) -> tuple[int, int]:
    """Convert HighGUI callback coordinates to the frozen frame's pixel space."""

    if (platform or sys.platform) == "darwin":
        # The Cocoa backend scales coordinates to the source image before calling
        # the registered callback. Applying the window-size ratio again is wrong.
        return int(x), int(y)
    image_width, image_height = image_size
    display_width, display_height = display_size
    if display_width <= 0 or display_height <= 0:
        return int(x), int(y)
    return round(x * image_width / display_width), round(y * image_height / display_height)


class CalibrationSession:
    """Collect an ordered set of floor-corner clicks on one frozen frame."""

    def __init__(self, frame_width: int, frame_height: int) -> None:
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame dimensions must be positive")
        self.frame_width = frame_width
        self.frame_height = frame_height
        self._points: list[Point] = []
        self.cancelled = False

    @property
    def image_points(self) -> tuple[Point, ...]:
        return tuple(self._points)

    @property
    def next_corner(self) -> str | None:
        return _CORNER_NAMES[len(self._points)] if len(self._points) < 4 else None

    def add_point(self, x: int, y: int) -> None:
        if self.cancelled:
            raise ValueError("calibration session was cancelled")
        if len(self._points) >= 4:
            return
        if not 0 <= x <= self.frame_width or not 0 <= y <= self.frame_height:
            raise ValueError("click must be inside the frozen camera frame")
        self._points.append((x / self.frame_width, y / self.frame_height))

    def reset(self) -> None:
        self._points.clear()
        self.cancelled = False

    def confirm(self) -> Calibration:
        if self.cancelled:
            raise ValueError("calibration session was cancelled")
        if len(self._points) != 4:
            raise ValueError("select all four floor corners before confirming")
        return create_calibration(self._points, (self.frame_width, self.frame_height))

    def cancel(self) -> None:
        self.cancelled = True


class CalibrationView:
    """Draw corner-selection instructions over a frozen camera frame."""

    def run(self, frame, window_name: str) -> tuple[Calibration | None, bool]:
        """Return (calibration, quit_requested); cancellation returns (None, False)."""

        import cv2
        import numpy as np

        frozen = frame.copy()
        height, width = frozen.shape[:2]
        session = CalibrationSession(width, height)
        notice = "Click the four floor corners in the requested order"

        def on_mouse(event, x, y, _flags, _parameter):
            nonlocal notice
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            try:
                rect = cv2.getWindowImageRect(window_name)
                x, y = _frame_coordinates(x, y, (width, height), (rect[2], rect[3]))
                session.add_point(x, y)
                notice = ""
            except ValueError as exc:
                notice = str(exc)

        cv2.setMouseCallback(window_name, on_mouse)
        confirmed: Calibration | None = None
        quit_requested = False
        while True:
            display = frozen.copy()
            prompt = f"Next: {session.next_corner}" if session.next_corner else "Press Enter to confirm"
            cv2.rectangle(display, (12, 12), (min(width - 12, 650), 86), (13, 22, 30), -1)
            cv2.putText(display, prompt, (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (113, 224, 234), 1, cv2.LINE_AA)
            cv2.putText(display, notice or "R reset   Enter confirm   Esc cancel   Q quit",
                        (25, 69), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 230, 230), 1, cv2.LINE_AA)
            points = [(int(x * width), int(y * height)) for x, y in session.image_points]
            for index, point in enumerate(points):
                cv2.circle(display, point, 8, (0, 240, 255), -1, cv2.LINE_AA)
                cv2.putText(display, str(index + 1), (point[0] + 9, point[1] - 9),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 240, 255), 2, cv2.LINE_AA)
            if len(points) >= 2:
                cv2.polylines(display, [np.asarray(points, dtype="int32")], len(points) == 4,
                              (0, 200, 230), 2, cv2.LINE_AA)
            cv2.imshow(window_name, display)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("r"), ord("R")):
                session.reset()
                notice = ""
            elif key in (10, 13):
                try:
                    confirmed = session.confirm()
                    break
                except ValueError as exc:
                    notice = str(exc)
            elif key == 27:
                session.cancel()
                break
            elif key in (ord("q"), ord("Q")):
                session.cancel()
                quit_requested = True
                break
        cv2.setMouseCallback(window_name, lambda *_args: None)
        return confirmed, quit_requested
