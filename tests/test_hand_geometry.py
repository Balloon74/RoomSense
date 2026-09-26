import math
import unittest

from roomsense.tracking.hand_geometry import (
    FINGER_NAMES,
    FingerState,
    HandPoint,
    HandState,
    analyze_hand,
)


def open_hand_points():
    points = [HandPoint(0.5, 0.8, 0.0) for _ in range(21)]
    coordinates = {
        0: (0.50, 0.80),
        1: (0.47, 0.75), 2: (0.44, 0.72), 3: (0.40, 0.68), 4: (0.37, 0.64),
        5: (0.45, 0.60), 6: (0.43, 0.50), 7: (0.42, 0.40), 8: (0.42, 0.30),
        9: (0.50, 0.58), 10: (0.50, 0.47), 11: (0.50, 0.36), 12: (0.50, 0.25),
        13: (0.55, 0.60), 14: (0.57, 0.51), 15: (0.58, 0.42), 16: (0.58, 0.32),
        17: (0.60, 0.63), 18: (0.64, 0.56), 19: (0.65, 0.49), 20: (0.66, 0.42),
    }
    for index, (x, y) in coordinates.items():
        points[index] = HandPoint(x, y, 0.0)
    return points


class HandGeometryTests(unittest.TestCase):
    def test_palm_center_and_image_orientation_use_the_five_palm_anchors(self):
        observation = analyze_hand(open_hand_points(), "Left", 0.9)

        self.assertAlmostEqual(observation.palm_center[0], 0.52)
        self.assertAlmostEqual(observation.palm_center[1], 0.642)
        self.assertAlmostEqual(observation.palm_angle, -math.pi / 2)
        self.assertAlmostEqual(observation.palm_normal[0], 0.0)
        self.assertAlmostEqual(observation.palm_normal[1], 0.0)
        self.assertAlmostEqual(observation.palm_normal[2], 1.0)

    def test_observation_preserves_all_model_points_and_named_fingertips(self):
        points = open_hand_points()
        observation = analyze_hand(points, "Right", 0.8)

        self.assertEqual(observation.landmarks, tuple(points))
        self.assertEqual(observation.handedness, "Right")
        self.assertAlmostEqual(observation.tracking_confidence, 0.8)
        self.assertEqual(set(observation.fingertips), set(FINGER_NAMES))
        self.assertEqual(observation.fingertips["index"], points[8])
        self.assertEqual(observation.fingertips["thumb"], points[4])
        with self.assertRaises(TypeError):
            observation.finger_states["index"] = FingerState.CURLED

    def test_extended_fingers_produce_full_openness_and_normalized_pinch_distance(self):
        observation = analyze_hand(open_hand_points(), "Left", 1.0)

        self.assertTrue(all(state is FingerState.EXTENDED for state in observation.finger_states.values()))
        self.assertEqual(observation.openness, 1.0)
        self.assertAlmostEqual(observation.pinch_distance, math.dist((0.37, 0.64), (0.42, 0.30)) / 0.22)
        self.assertIsNone(observation.state)

    def test_curled_finger_is_classified_from_joint_geometry(self):
        points = open_hand_points()
        points[7] = HandPoint(0.45, 0.59, 0.0)
        points[8] = HandPoint(0.46, 0.57, 0.0)

        observation = analyze_hand(points, "Right", 0.9)

        self.assertIs(observation.finger_states["index"], FingerState.CURLED)
        self.assertAlmostEqual(observation.openness, 0.8)

    def test_close_thumb_and_index_tips_produce_small_palm_relative_pinch_distance(self):
        points = open_hand_points()
        points[4] = HandPoint(0.42, 0.305, 0.0)

        observation = analyze_hand(points, "Left", 0.9)

        self.assertLess(observation.pinch_distance, 0.04)

    def test_degenerate_or_nonfinite_palm_geometry_is_uncertain(self):
        degenerate = [HandPoint(0.5, 0.5, 0.0) for _ in range(21)]
        invalid = open_hand_points()
        invalid[9] = HandPoint(float("nan"), 0.58, 0.0)

        for points in (degenerate, invalid):
            with self.subTest(points=points is degenerate):
                observation = analyze_hand(points, "Left", 0.7)
                self.assertTrue(all(state is FingerState.UNCERTAIN for state in observation.finger_states.values()))
                self.assertEqual(observation.openness, 0.0)
                self.assertIsNone(observation.pinch_distance)
                self.assertIsNone(observation.palm_angle)
                self.assertIsNone(observation.palm_normal)

    def test_landmark_count_must_match_the_media_pipe_hand_topology(self):
        with self.assertRaisesRegex(ValueError, "21"):
            analyze_hand(open_hand_points()[:-1], "Left", 0.9)

    def test_state_enum_includes_required_hand_states(self):
        self.assertEqual(
            {state.value for state in HandState},
            {"OPEN PALM", "CLOSED FIST", "POINTING", "PEACE SIGN", "THUMBS UP", "PINCHING"},
        )


if __name__ == "__main__":
    unittest.main()
