import unittest
import importlib

from roomsense.calibration.camera_calibration import create_calibration, homography_for
from roomsense.spatial.floor_position import FloorPositionTracker
from roomsense.spatial.position_history import PositionHistory
from roomsense.spatial.room_transform import RoomTransform
from roomsense.spatial.zones import Zone, ZoneTracker
from roomsense.tracking.pose_tracker import Landmark


CALIBRATION = create_calibration(
    ((0.2, 0.1), (0.8, 0.1), (0.8, 0.9), (0.2, 0.9)), (1000, 500)
)


def ankle(x, y, visibility=1.0):
    return Landmark(x, y, 0.0, visibility)


class RoomTransformTests(unittest.TestCase):
    def test_maps_floor_corners_and_marks_outside_image_points(self):
        transform = RoomTransform(CALIBRATION)
        self.assertEqual(transform.transform((0.2, 0.1)).in_bounds, True)
        mapped = transform.transform((0.8, 0.9))
        self.assertAlmostEqual(mapped.room_x, 1.0)
        self.assertAlmostEqual(mapped.room_y, 1.0)
        self.assertTrue(mapped.in_bounds)
        outside = transform.transform((0.05, 0.5))
        self.assertFalse(outside.in_bounds)
        self.assertLess(outside.room_x, 0.0)
        self.assertEqual(outside.clamped_x, 0.0)

    def test_projective_horizon_observation_is_invalid_instead_of_raising(self):
        horizon_calibration = create_calibration(
            ((0.375, 0.5), (0.625, 0.5), (0.875, 1.0), (0.125, 1.0)), (1000, 500)
        )
        mapped = RoomTransform(horizon_calibration).transform((0.5, 0.25))
        self.assertFalse(mapped.projectable)
        self.assertFalse(mapped.in_bounds)


class FloorPositionTests(unittest.TestCase):
    def test_uses_visibility_weighted_midpoint_of_reliable_feet(self):
        tracker = FloorPositionTracker(min_visibility=0.5, smoothing_alpha=1.0)
        result = tracker.update(
            {"left_ankle": ankle(0.3, 0.5, 0.5), "right_ankle": ankle(0.7, 0.5, 1.0)},
            RoomTransform(CALIBRATION),
        )
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.raw_image_x, 0.5666666667)
        self.assertAlmostEqual(result.room_x, 0.6111111111)
        self.assertTrue(result.in_bounds)

    def test_falls_back_to_one_reliable_foot_and_ignores_unreliable_other_foot(self):
        tracker = FloorPositionTracker(min_visibility=0.5, smoothing_alpha=1.0)
        result = tracker.update(
            {"left_ankle": ankle(0.3, 0.5, 0.9), "right_ankle": ankle(0.7, 0.5, 0.1)},
            RoomTransform(CALIBRATION),
        )
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.raw_image_x, 0.3)

    def test_returns_no_position_when_neither_foot_is_reliable(self):
        tracker = FloorPositionTracker(min_visibility=0.5)
        self.assertIsNone(
            tracker.update({"left_ankle": ankle(0.3, 0.5, 0.2)}, RoomTransform(CALIBRATION))
        )

    def test_zero_visibility_ankles_are_rejected_when_threshold_is_zero(self):
        tracker = FloorPositionTracker(min_visibility=0.0)
        result = tracker.update(
            {"left_ankle": ankle(0.3, 0.5, 0.0), "right_ankle": ankle(0.7, 0.5, 0.0)},
            RoomTransform(CALIBRATION),
        )
        self.assertIsNone(result)

    def test_smooths_foot_contact_measurements(self):
        tracker = FloorPositionTracker(min_visibility=0.5, smoothing_alpha=0.5)
        transform = RoomTransform(CALIBRATION)
        first = tracker.update({"left_ankle": ankle(0.3, 0.5)}, transform)
        second = tracker.update({"left_ankle": ankle(0.5, 0.5)}, transform)
        self.assertAlmostEqual(first.smoothed_image_x, 0.3)
        self.assertAlmostEqual(second.smoothed_image_x, 0.4)


class ZoneTests(unittest.TestCase):
    def setUp(self):
        self.desk = Zone("DESK", ((0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)))
        self.bed = Zone("BED", ((0.4, 0.4), (0.8, 0.4), (0.8, 0.8), (0.4, 0.8)))
        self.tracker = ZoneTracker((self.desk, self.bed))

    def test_detects_polygon_containment_and_entry_only_once(self):
        first = self.tracker.update((0.2, 0.2))
        repeated = self.tracker.update((0.25, 0.25))
        self.assertEqual(first.entered, ("DESK",))
        self.assertEqual(repeated.entered, ())
        self.assertEqual(repeated.current, ("DESK",))

    def test_supports_overlapping_zones_and_leaves_once(self):
        overlap = self.tracker.update((0.45, 0.45))
        self.assertEqual(overlap.entered, ("DESK", "BED"))
        left = self.tracker.update((0.9, 0.9))
        repeated = self.tracker.update((0.95, 0.95))
        self.assertEqual(left.left, ("DESK", "BED"))
        self.assertEqual(repeated.left, ())

    def test_outside_floor_clears_membership(self):
        self.tracker.update((0.2, 0.2))
        update = self.tracker.update((0.2, 0.2), in_bounds=False)
        self.assertEqual(update.left, ("DESK",))
        self.assertEqual(update.current, ())

    def test_none_position_clears_membership(self):
        self.tracker.update((0.2, 0.2))
        self.assertEqual(self.tracker.update(None).left, ("DESK",))


class PositionHistoryTests(unittest.TestCase):
    def test_calculates_direction_speed_distance_and_retains_recent_trail(self):
        history = PositionHistory(retention_seconds=1.0)
        history.update(0.0, (0.0, 0.0))
        middle = history.update(0.5, (0.3, 0.4))
        end = history.update(1.5, (0.6, 0.8))
        self.assertAlmostEqual(middle.speed, 1.0)
        self.assertEqual(middle.direction, (0.3, 0.4))
        self.assertAlmostEqual(end.speed, 0.5)
        self.assertAlmostEqual(end.distance, 1.0)
        self.assertEqual([sample.timestamp for sample in end.samples], [0.5, 1.5])

    def test_tracking_loss_clears_transient_metrics_and_keeps_session_distance(self):
        history = PositionHistory()
        history.update(0.0, (0.0, 0.0))
        before = history.update(1.0, (0.3, 0.4))
        history.clear_tracking()
        after = history.update(3.0, (0.8, 0.8))
        self.assertEqual(after.distance, before.distance)
        self.assertEqual(after.direction, (0.0, 0.0))
        self.assertEqual(after.speed, 0.0)

    def test_session_reset_clears_path_distance(self):
        history = PositionHistory()
        history.update(0.0, (0.0, 0.0))
        history.update(1.0, (0.3, 0.4))
        history.reset_session()
        self.assertEqual(history.distance, 0.0)
        self.assertEqual(history.samples, ())


class SpatialObservationMonitorTests(unittest.TestCase):
    def test_ankle_dropout_expires_once_and_valid_position_resets_timeout(self):
        monitor_module = importlib.import_module("roomsense.spatial.observation_state")
        monitor = monitor_module.SpatialObservationMonitor(timeout_seconds=1.0)
        self.assertFalse(monitor.mark_missing(2.0))
        self.assertFalse(monitor.mark_missing(2.9))
        self.assertTrue(monitor.mark_missing(3.0))
        self.assertFalse(monitor.mark_missing(4.0))
        monitor.mark_valid()
        self.assertFalse(monitor.mark_missing(5.0))


if __name__ == "__main__":
    unittest.main()
