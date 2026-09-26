import math
import unittest

from roomsense.spatial.pointing import PointingEstimator
from roomsense.tracking.pose_tracker import Landmark


def point(x, y, visibility=1.0):
    return Landmark(x, y, 0.0, visibility)


class PointingEstimatorTests(unittest.TestCase):
    def test_reports_left_pointing_direction_and_full_extension(self):
        estimator = PointingEstimator(min_visibility=0.5, min_extension=0.8, smoothing_alpha=1.0)
        result = estimator.update({
            "left_shoulder": point(0.2, 0.4),
            "left_elbow": point(0.4, 0.4),
            "left_wrist": point(0.6, 0.4),
        })

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].side, "left")
        self.assertEqual(result[0].origin, (0.2, 0.4))
        self.assertEqual(result[0].direction, (1.0, 0.0))
        self.assertAlmostEqual(result[0].extension, 1.0)
        self.assertAlmostEqual(result[0].confidence, 1.0)

    def test_reports_right_pointing_with_diagonal_unit_vector(self):
        estimator = PointingEstimator(smoothing_alpha=1.0)
        result = estimator.update({
            "right_shoulder": point(0.5, 0.5),
            "right_elbow": point(0.6, 0.4),
            "right_wrist": point(0.7, 0.3),
        })
        self.assertEqual(result[0].side, "right")
        self.assertAlmostEqual(result[0].direction[0], math.sqrt(0.5))
        self.assertAlmostEqual(result[0].direction[1], -math.sqrt(0.5))

    def test_rejects_bent_arm_below_extension_threshold(self):
        estimator = PointingEstimator(min_extension=0.8)
        result = estimator.update({
            "left_shoulder": point(0.2, 0.4),
            "left_elbow": point(0.4, 0.4),
            "left_wrist": point(0.2, 0.4),
        })
        self.assertEqual(result, ())

    def test_uses_landmark_visibility_in_confidence_and_threshold(self):
        estimator = PointingEstimator(min_visibility=0.5, min_extension=0.5, smoothing_alpha=1.0)
        pose = {
            "left_shoulder": point(0.2, 0.4, 0.8),
            "left_elbow": point(0.4, 0.4, 0.6),
            "left_wrist": point(0.6, 0.4, 0.7),
        }
        result = estimator.update(pose)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].confidence, 0.7)
        pose["left_elbow"] = point(0.4, 0.4, 0.49)
        self.assertEqual(estimator.update(pose), ())

    def test_rejects_missing_landmarks_and_degenerate_vectors(self):
        estimator = PointingEstimator(min_extension=0.0)
        self.assertEqual(estimator.update({}), ())
        self.assertEqual(estimator.update({
            "left_shoulder": point(0.3, 0.3),
            "left_elbow": point(0.3, 0.3),
            "left_wrist": point(0.3, 0.3),
        }), ())

    def test_smooths_direction_and_reset_clears_history(self):
        estimator = PointingEstimator(smoothing_alpha=0.5)
        right = {
            "left_shoulder": point(0.2, 0.4),
            "left_elbow": point(0.4, 0.4),
            "left_wrist": point(0.6, 0.4),
        }
        first = estimator.update(right)[0]
        up = {
            "left_shoulder": point(0.2, 0.4),
            "left_elbow": point(0.2, 0.2),
            "left_wrist": point(0.2, 0.0),
        }
        second = estimator.update(up)[0]
        self.assertAlmostEqual(second.direction[0], math.sqrt(0.5))
        self.assertAlmostEqual(second.direction[1], -math.sqrt(0.5))
        estimator.reset()
        reset = estimator.update(up)[0]
        self.assertEqual(reset.direction, (0.0, -1.0))
        self.assertEqual(first.direction, (1.0, 0.0))


if __name__ == "__main__":
    unittest.main()
