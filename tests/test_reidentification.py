import unittest

from roomsense.tracking.reidentification import (
    PersonReidentifier,
    ReidentificationState,
    TrackObservation,
)


def observation(timestamp: float, x: float, y: float = 0.5, z: float = 0.0) -> TrackObservation:
    return TrackObservation(
        timestamp=timestamp,
        x=x,
        y=y,
        z=z,
        shoulder_width=0.2,
        torso_ratio=1.0,
    )


class ReidentificationLifecycleTests(unittest.TestCase):
    def test_immediate_return_reuses_anonymous_id(self):
        manager = PersonReidentifier()

        first = manager.update(observation(0.0, 0.5))
        continuing = manager.update(observation(0.2, 0.51))
        lost = manager.update(None, timestamp=0.21)
        returned = manager.update(observation(0.35, 0.52))
        continuing_again = manager.update(observation(0.5, 0.53))

        self.assertEqual((first.person_id, first.state), ("PERSON_001", ReidentificationState.NEW))
        self.assertEqual((continuing.person_id, continuing.state), ("PERSON_001", ReidentificationState.TRACKED))
        self.assertEqual((lost.person_id, lost.state), ("PERSON_001", ReidentificationState.LOST))
        self.assertEqual((returned.person_id, returned.state), ("PERSON_001", ReidentificationState.REACQUIRED))
        self.assertEqual((continuing_again.person_id, continuing_again.state),
                         ("PERSON_001", ReidentificationState.TRACKED))

    def test_return_near_predicted_exit_side_has_high_direction_score(self):
        manager = PersonReidentifier()
        manager.update(observation(0.0, 0.75))
        manager.update(observation(0.25, 0.85))  # Last motion is toward the right frame edge.
        manager.update(None, timestamp=0.26)

        returned = manager.update(observation(0.5, 0.98))

        self.assertEqual(returned.person_id, "PERSON_001")
        self.assertEqual(returned.state, ReidentificationState.REACQUIRED)
        self.assertGreater(returned.candidates[0].factors["reentry_direction"], 0.8)

    def test_return_after_timeout_gets_a_new_id(self):
        manager = PersonReidentifier(timeout_seconds=0.5)
        manager.update(observation(0.0, 0.5))
        manager.update(None, timestamp=0.1)

        returned = manager.update(observation(0.6, 0.5))

        self.assertEqual(returned.person_id, "PERSON_002")
        self.assertEqual(returned.state, ReidentificationState.NEW)
        self.assertEqual(returned.reason, "expired_candidate")
        self.assertEqual(returned.candidates, ())

    def test_lost_registry_discards_person_after_timeout_without_a_return(self):
        manager = PersonReidentifier(timeout_seconds=0.5)
        manager.update(observation(0.0, 0.5))
        manager.update(None, timestamp=0.1)

        still_lost = manager.update(None, timestamp=0.6)

        self.assertIsNone(still_lost.person_id)
        self.assertEqual(still_lost.state, ReidentificationState.LOST)
        self.assertEqual(still_lost.reason, "expired")

    def test_clearly_different_track_is_not_reconnected(self):
        manager = PersonReidentifier()
        manager.update(observation(0.0, 0.2, 0.2))
        manager.update(None, timestamp=0.1)

        returned = manager.update(TrackObservation(0.2, 0.9, 0.9, 0.8, 0.4, 2.0))

        self.assertEqual(returned.person_id, "PERSON_002")
        self.assertEqual(returned.state, ReidentificationState.NEW)
        self.assertEqual(returned.reason, "below_threshold")
        self.assertEqual(returned.candidates[0].reason, "below_threshold")

    def test_missing_geometry_does_not_raise_a_candidate_over_threshold(self):
        manager = PersonReidentifier(confidence_threshold=0.73)
        manager.update(observation(0.0, 0.5))
        manager.update(None, timestamp=0.01)

        returned = manager.update(TrackObservation(0.02, 0.5, 0.5, 0.0))

        self.assertEqual(returned.person_id, "PERSON_002")
        self.assertEqual(returned.reason, "below_threshold")
        self.assertEqual(returned.candidates[0].factors["torso_geometry"], 0.0)

    def test_ambiguous_candidates_are_not_resolved_arbitrarily(self):
        manager = PersonReidentifier()
        manager.update(observation(0.0, 0.4))
        manager.update(observation(0.1, 0.3))
        manager.update(observation(0.2, 0.2))
        manager.update(None, timestamp=0.21)
        second_person = manager.update(observation(0.3, 0.9))
        manager.update(observation(0.4, 0.55))
        manager.update(observation(0.5, 0.2))
        manager.update(None, timestamp=0.51)

        returned = manager.update(observation(0.6, 0.1))

        self.assertEqual(second_person.person_id, "PERSON_002")
        self.assertEqual(returned.person_id, "PERSON_003")
        self.assertEqual(returned.reason, "ambiguous_candidates")
        self.assertEqual(returned.state, ReidentificationState.NEW)
        self.assertEqual({candidate.reason for candidate in returned.candidates}, {"ambiguous"})
        self.assertLess(abs(returned.candidates[0].score - returned.candidates[1].score), 0.12)

    def test_best_of_multiple_lost_tracks_is_reacquired_with_factor_details(self):
        manager = PersonReidentifier()
        manager.update(observation(0.0, 0.2))
        manager.update(None, timestamp=0.01)
        manager.update(observation(0.2, 0.95))
        manager.update(observation(0.3, 0.92))
        manager.update(None, timestamp=0.31)

        returned = manager.update(observation(0.5, 0.91))

        self.assertEqual(returned.person_id, "PERSON_002")
        self.assertEqual(returned.state, ReidentificationState.REACQUIRED)
        self.assertEqual({candidate.person_id for candidate in returned.candidates},
                         {"PERSON_001", "PERSON_002"})
        self.assertEqual(returned.candidates[0].person_id, "PERSON_002")
        self.assertEqual(returned.candidates[0].reason, "selected")
        self.assertEqual(set(returned.candidates[0].factors),
                         {"trajectory", "reentry_direction", "elapsed_time", "torso_geometry"})


if __name__ == "__main__":
    unittest.main()
