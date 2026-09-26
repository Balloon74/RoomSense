import unittest

from roomsense.spatial.pointing import ArmPointing
from roomsense.spatial.room_objects import RoomObject, RoomObjectRegistry
from roomsense.spatial.target_selection import TargetSelector


def pointing(y=0.5, confidence=0.9, side="left"):
    return ArmPointing(side, (0.1, y), (1.0, 0.0), 0.95, confidence)


def target(object_id, start, end, *, enabled=True, radius=None):
    return RoomObject(object_id, object_id.upper(), ((start, 0.3), (end, 0.3), (end, 0.7), (start, 0.7)),
                      enabled=enabled, interaction_radius=radius)


class TargetSelectorTests(unittest.TestCase):
    def test_selects_nearest_region_intersected_by_forward_ray(self):
        near, far = target("near", 0.4, 0.5), target("far", 0.7, 0.9)
        selector = TargetSelector(RoomObjectRegistry((far, near)), stability_seconds=0.2, hold_seconds=0.5)
        result = selector.update((pointing(),), 1.0)
        self.assertEqual(result.candidate.object, near)
        self.assertAlmostEqual(result.candidate.ray_distance, 0.3)
        self.assertIsNone(result.confirmed)

    def test_near_miss_requires_configured_interaction_radius(self):
        item = RoomObject("monitor", "MONITOR", ((0.5, 0.54), (0.7, 0.54), (0.7, 0.7), (0.5, 0.7)),
                          interaction_radius=0.05)
        selector = TargetSelector(RoomObjectRegistry((item,)), stability_seconds=0.0, hold_seconds=1.0)
        result = selector.update((pointing(y=0.5),), 0.0)
        self.assertEqual(result.confirmed.object, item)

    def test_disabled_and_non_intersected_objects_are_not_candidates(self):
        disabled = target("disabled", 0.3, 0.4, enabled=False)
        miss = target("miss", 0.5, 0.7)
        selector = TargetSelector(RoomObjectRegistry((disabled, miss)), stability_seconds=0.0, hold_seconds=1.0)
        self.assertIsNone(selector.update((pointing(y=0.9),), 0.0).candidate)

    def test_requires_stable_candidate_then_emits_one_confirmation_transition(self):
        item = target("monitor", 0.5, 0.7)
        selector = TargetSelector(RoomObjectRegistry((item,)), stability_seconds=0.3, hold_seconds=1.0)
        first = selector.update((pointing(),), 0.0)
        stable = selector.update((pointing(),), 0.29)
        confirmed = selector.update((pointing(),), 0.3)
        repeated = selector.update((pointing(),), 0.4)
        self.assertIsNone(first.confirmed)
        self.assertIsNone(stable.confirmed)
        self.assertEqual(confirmed.confirmed.object, item)
        self.assertTrue(confirmed.confirmed_now)
        self.assertFalse(repeated.confirmed_now)

    def test_candidate_change_must_be_stable_before_replacing_confirmation(self):
        monitor, bed = target("monitor", 0.4, 0.5), target("bed", 0.7, 0.9)
        selector = TargetSelector(RoomObjectRegistry((monitor, bed)), stability_seconds=0.2, hold_seconds=2.0)
        selector.update((pointing(),), 0.0)
        selector.update((pointing(),), 0.2)
        changing = selector.update((pointing(y=0.5),), 0.3)
        # Move the ray origin beyond MONITOR so it only intersects BED.
        bed_pointing = ArmPointing("right", (0.6, 0.5), (1.0, 0.0), 0.95, 0.8)
        waiting = selector.update((bed_pointing,), 0.4)
        switched = selector.update((bed_pointing,), 0.61)
        self.assertEqual(changing.confirmed.object, monitor)
        self.assertEqual(waiting.confirmed.object, monitor)
        self.assertEqual(switched.confirmed.object, bed)
        self.assertTrue(switched.confirmed_now)

    def test_pointing_loss_clears_target_and_held_event_fires_once(self):
        item = target("monitor", 0.5, 0.7)
        selector = TargetSelector(RoomObjectRegistry((item,)), stability_seconds=0.2, hold_seconds=0.3)
        selector.update((pointing(),), 0.0)
        selector.update((pointing(),), 0.2)
        before_hold = selector.update((pointing(),), 0.49)
        held = selector.update((pointing(),), 0.5)
        repeated = selector.update((pointing(),), 0.6)
        lost = selector.update((), 0.7)
        self.assertFalse(before_hold.held_now)
        self.assertTrue(held.held_now)
        self.assertFalse(repeated.held_now)
        self.assertIsNone(lost.confirmed)
        self.assertIsNone(lost.held)

    def test_rejects_non_finite_or_decreasing_timestamps(self):
        selector = TargetSelector(RoomObjectRegistry((target("monitor", 0.5, 0.7),)))
        with self.assertRaises(ValueError):
            selector.update((), float("nan"))
        selector.update((), 1.0)
        with self.assertRaises(ValueError):
            selector.update((), 0.0)


if __name__ == "__main__":
    unittest.main()
