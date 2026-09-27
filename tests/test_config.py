import unittest

from roomsense.config import DEFAULT_ROOM_OBJECTS, DEFAULT_ZONE_POLYGONS, RoomObjectConfig, RoomSenseConfig


class RoomSenseV2ConfigTests(unittest.TestCase):
    def test_configuration_has_local_calibration_and_spatial_defaults(self):
        config = RoomSenseConfig()
        self.assertEqual(config.calibration_path, "calibration.json")
        self.assertEqual(config.floor_landmark_visibility, 0.5)
        self.assertEqual(config.history_retention_seconds, 4.0)

    def test_default_zones_are_named_normalized_polygons(self):
        zones = dict(DEFAULT_ZONE_POLYGONS)
        self.assertEqual(set(zones), {"DESK", "BED", "DOOR", "CENTER"})
        for polygon in zones.values():
            self.assertGreaterEqual(len(polygon), 3)
            for x, y in polygon:
                self.assertTrue(0.0 <= x <= 1.0)
                self.assertTrue(0.0 <= y <= 1.0)

    def test_v1_defaults_are_preserved(self):
        config = RoomSenseConfig()
        self.assertIsNone(config.camera_index)  # Current V1 default selects the built-in Mac camera.
        self.assertEqual(RoomSenseConfig(camera_index=2).camera_index, 2)
        self.assertEqual(config.smoothing_alpha, 0.35)
        self.assertEqual(config.reference_shoulder_width, 0.20)

    def test_v3_defaults_include_safe_thresholds_and_disabled_object_examples(self):
        config = RoomSenseConfig()
        self.assertEqual(config.pointing_min_visibility, 0.55)
        self.assertEqual(config.pointing_min_extension, 0.78)
        self.assertEqual(config.target_stability_seconds, 0.35)
        self.assertEqual(config.target_hold_seconds, 0.8)
        self.assertEqual(config.command_mode_timeout_seconds, 12.0)
        self.assertEqual(config.event_feed_size, 8)
        objects = {item.id: item for item in DEFAULT_ROOM_OBJECTS}
        self.assertIn("monitor", objects)
        self.assertEqual(objects["monitor"].name, "MONITOR")
        self.assertFalse(any(item.enabled for item in objects.values()))
        for item in objects.values():
            self.assertGreaterEqual(len(item.image_region), 3)
            self.assertTrue(all(0.0 <= value <= 1.0 for point in item.image_region for value in point))
            if item.room_position is not None:
                self.assertTrue(all(0.0 <= value <= 1.0 for value in item.room_position))

    def test_v3_timing_and_visibility_configuration_is_validated(self):
        with self.assertRaises(ValueError):
            RoomSenseConfig(pointing_min_visibility=float("nan"))

    def test_evaluation_recording_path_defaults_to_separate_output_and_is_validated(self):
        config = RoomSenseConfig()
        self.assertEqual(config.evaluation_recording_path, "roomsense-evaluation.jsonl")
        self.assertNotEqual(config.evaluation_recording_path, config.session_recording_path)
        with self.assertRaisesRegex(ValueError, "evaluation_recording_path"):
            RoomSenseConfig(evaluation_recording_path=" ")
        with self.assertRaises(ValueError):
            RoomSenseConfig(target_stability_seconds=-0.1)
        with self.assertRaises(ValueError):
            RoomSenseConfig(gesture_swipe_window_seconds=0.0)
        with self.assertRaises(ValueError):
            RoomSenseConfig(event_feed_size=0)

    def test_room_object_configuration_is_validated_when_config_is_created(self):
        with self.assertRaisesRegex(ValueError, "RoomObjectConfig"):
            RoomSenseConfig(room_objects=("invalid",))
        invalid = RoomObjectConfig("monitor", "MONITOR", ((0.1, 0.1, 0.5), (0.8, 0.1), (0.8, 0.8)))
        with self.assertRaisesRegex(ValueError, "image_region"):
            RoomSenseConfig(room_objects=(invalid,))


if __name__ == "__main__":
    unittest.main()
