import unittest

from roomsense.spatial.room_objects import RoomObject, RoomObjectRegistry


REGION = ((0.5, 0.3), (0.7, 0.3), (0.7, 0.7), (0.5, 0.7))


class RoomObjectTests(unittest.TestCase):
    def test_accepts_image_region_optional_room_position_and_radius(self):
        item = RoomObject("monitor", "MONITOR", REGION, room_position=(0.8, 0.2), interaction_radius=0.03)
        self.assertEqual(item.id, "monitor")
        self.assertEqual(item.image_region, REGION)
        self.assertEqual(item.room_position, (0.8, 0.2))
        self.assertEqual(item.interaction_radius, 0.03)
        self.assertTrue(item.enabled)

    def test_rejects_empty_identity_and_invalid_image_regions(self):
        with self.assertRaises(ValueError):
            RoomObject("", "MONITOR", REGION)
        for polygon in (
            ((0.1, 0.1), (0.2, 0.2)),
            ((-0.1, 0.1), (0.7, 0.1), (0.7, 0.7)),
            ((0.1, 0.1), (0.7, 0.7), (0.1, 0.7), (0.7, 0.1)),
            ((0.1, 0.1, 0.5), (0.7, 0.1), (0.7, 0.7)),
        ):
            with self.subTest(polygon=polygon), self.assertRaises(ValueError):
                RoomObject("invalid", "INVALID", polygon)

    def test_rejects_invalid_map_position_and_radius(self):
        with self.assertRaises(ValueError):
            RoomObject("monitor", "MONITOR", REGION, room_position=(1.1, 0.5))
        with self.assertRaises(ValueError):
            RoomObject("monitor", "MONITOR", REGION, interaction_radius=float("nan"))
        with self.assertRaises(ValueError):
            RoomObject("monitor", "MONITOR", REGION, interaction_radius=-0.1)
        with self.assertRaises(ValueError):
            RoomObject("monitor", "MONITOR", REGION, room_position=(0.5, 0.5, 0.5))

    def test_registry_requires_unique_ids_and_returns_only_enabled_objects(self):
        enabled = RoomObject("monitor", "MONITOR", REGION)
        disabled = RoomObject("bed", "BED", REGION, enabled=False)
        registry = RoomObjectRegistry((enabled, disabled))
        self.assertEqual(registry.enabled_objects(), (enabled,))
        with self.assertRaises(ValueError):
            RoomObjectRegistry((enabled, RoomObject("monitor", "OTHER", REGION)))


if __name__ == "__main__":
    unittest.main()
