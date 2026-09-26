import unittest
import roomsense.visualization.calibration_view as calibration_view

from roomsense.visualization.calibration_view import CalibrationSession


class CalibrationSessionTests(unittest.TestCase):
    def setUp(self):
        self.session = CalibrationSession(frame_width=1000, frame_height=500)

    def test_records_four_clicks_in_requested_order(self):
        clicks = ((200, 50), (800, 50), (800, 450), (200, 450))
        for click in clicks:
            self.session.add_point(*click)
        self.assertEqual(self.session.image_points, ((0.2, 0.1), (0.8, 0.1), (0.8, 0.9), (0.2, 0.9)))
        self.assertEqual(self.session.next_corner, None)

    def test_reset_clears_points_and_starts_at_back_left(self):
        self.session.add_point(200, 50)
        self.session.reset()
        self.assertEqual(self.session.image_points, ())
        self.assertEqual(self.session.next_corner, "back-left")

    def test_confirm_requires_four_valid_points(self):
        self.session.add_point(200, 50)
        with self.assertRaises(ValueError):
            self.session.confirm()

    def test_confirm_returns_calibration_and_cancel_marks_session(self):
        for click in ((200, 50), (800, 50), (800, 450), (200, 450)):
            self.session.add_point(*click)
        calibration = self.session.confirm()
        self.assertEqual(calibration.source_width, 1000)
        self.assertEqual(calibration.image_points[2], (0.8, 0.9))
        self.session.cancel()
        self.assertTrue(self.session.cancelled)

    def test_rejects_clicks_outside_frozen_frame(self):
        with self.assertRaises(ValueError):
            self.session.add_point(1001, 200)

    def test_resized_cocoa_window_mouse_coordinates_are_already_image_coordinates(self):
        self.assertEqual(
            calibration_view._frame_coordinates(
                800, 450, image_size=(1280, 720), display_size=(640, 360), platform="darwin"
            ),
            (800, 450),
        )

    def test_resized_non_cocoa_window_coordinates_scale_to_image_size(self):
        self.assertEqual(
            calibration_view._frame_coordinates(
                400, 225, image_size=(1280, 720), display_size=(640, 360), platform="linux"
            ),
            (800, 450),
        )


if __name__ == "__main__":
    unittest.main()
