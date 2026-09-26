import json
import math
import tempfile
import unittest
from pathlib import Path

from roomsense.calibration.camera_calibration import (
    Calibration,
    create_calibration,
    homography_for,
    transform_point,
    validate_calibration,
)
from roomsense.calibration.calibration_store import load_calibration, save_calibration


RECTANGLE = ((0.2, 0.1), (0.8, 0.1), (0.8, 0.9), (0.2, 0.9))


class CalibrationGeometryTests(unittest.TestCase):
    def test_homography_maps_ordered_corners_to_normalized_room_corners(self):
        calibration = create_calibration(RECTANGLE, (1280, 720))
        matrix = homography_for(calibration)
        actual = [transform_point(point, matrix) for point in RECTANGLE]
        expected = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
        for got, want in zip(actual, expected):
            self.assertAlmostEqual(got[0], want[0], places=5)
            self.assertAlmostEqual(got[1], want[1], places=5)

    def test_perspective_homography_maps_an_interior_diagonal_intersection(self):
        trapezoid = ((0.2, 0.1), (0.8, 0.1), (0.95, 0.9), (0.05, 0.9))
        calibration = create_calibration(trapezoid, (1280, 720))
        mapped = transform_point((0.5, 0.42), homography_for(calibration))
        self.assertAlmostEqual(mapped[0], 0.5, places=5)
        self.assertAlmostEqual(mapped[1], 0.5, places=5)

    def test_rejects_invalid_quadrilaterals(self):
        invalid_points = (
            ((0.1, 0.1), (0.9, 0.1), (0.5, 0.5)),
            ((-0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)),
            ((0.1, 0.1), (0.9, 0.1), (0.5, 0.5), (0.1, 0.9)),
            ((0.1, 0.1), (0.9, 0.9), (0.9, 0.1), (0.1, 0.9)),
            ((0.1, 0.1), (0.9, 0.1), (0.9, 0.1), (0.1, 0.9)),
            ((0.1, 0.1), (0.10001, 0.1), (0.10001, 0.10001), (0.1, 0.10001)),
        )
        for points in invalid_points:
            with self.subTest(points=points), self.assertRaises(ValueError):
                create_calibration(points, (640, 480))

    def test_rejects_non_finite_points_and_unsupported_version(self):
        points = ((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9))
        bad_points = ((math.nan, 0.1), *points[1:])
        with self.assertRaises(ValueError):
            create_calibration(bad_points, (640, 480))

        calibration = create_calibration(points, (640, 480))
        unsupported = Calibration(
            image_points=calibration.image_points,
            source_width=calibration.source_width,
            source_height=calibration.source_height,
            map_width=calibration.map_width,
            map_height=calibration.map_height,
            version=99,
        )
        with self.assertRaises(ValueError):
            validate_calibration(unsupported)

    def test_rejects_mismatched_aspect_ratio(self):
        calibration = create_calibration(RECTANGLE, (1000, 500))
        with self.assertRaises(ValueError):
            validate_calibration(calibration, expected_size=(640, 480))

    def test_rejects_unrepresentably_large_source_dimensions(self):
        calibration = create_calibration(RECTANGLE, (10**400, 10**400))
        with self.assertRaises(ValueError):
            validate_calibration(calibration, expected_size=(640, 480))


class CalibrationStoreTests(unittest.TestCase):
    def test_round_trips_calibration_json(self):
        calibration = create_calibration(RECTANGLE, (1280, 720))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            save_calibration(path, calibration)
            loaded = load_calibration(path, expected_size=(640, 360))

        self.assertEqual(loaded, calibration)

    def test_missing_file_returns_none(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(load_calibration(Path(directory) / "missing.json"))

    def test_saved_calibration_rejects_aspect_mismatch_for_current_camera(self):
        calibration = create_calibration(RECTANGLE, (1280, 720))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            save_calibration(path, calibration)
            with self.assertRaises(ValueError):
                load_calibration(path, expected_size=(640, 480))

    def test_invalid_json_and_unsupported_schema_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_calibration(path)

    def test_unrepresentably_large_numeric_point_is_rejected_as_invalid_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            payload = {
                "version": 1,
                "image_points": [[10**400, 0], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
                "source_size": [640, 480],
                "map_size": [1.0, 1.0],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_calibration(path)

            path.write_text(json.dumps({"version": 88}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_calibration(path)


if __name__ == "__main__":
    unittest.main()
