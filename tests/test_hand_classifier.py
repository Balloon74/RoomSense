import unittest

from roomsense.tracking.hand_classifier import TemporalHandClassifier, associate_hands, classify_hand
from roomsense.tracking.hand_geometry import FingerState, HandPoint, HandState, analyze_hand
from roomsense.tracking.pose_tracker import Landmark
from test_hand_geometry import open_hand_points


def curled(points, finger):
    if finger == "thumb":
        points[4] = HandPoint(0.50, 0.72, 0.0)
    else:
        tip = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}[finger]
        points[tip] = HandPoint(0.50, 0.70, 0.0)


def observation(extended=(), handedness="Left", pinch=False):
    points = open_hand_points()
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        if finger not in extended:
            curled(points, finger)
    if pinch:
        points[4] = points[8]
    return analyze_hand(points, handedness, 0.9)


def body_landmarks(left_wrist=(0.20, 0.80), right_wrist=(0.80, 0.80)):
    return {
        "left_shoulder": Landmark(0.30, 0.30),
        "right_shoulder": Landmark(0.70, 0.30),
        "left_wrist": Landmark(*left_wrist),
        "right_wrist": Landmark(*right_wrist),
    }


class HandGestureClassificationTests(unittest.TestCase):
    def test_recognizes_each_supported_hand_state(self):
        examples = (
            (observation(("thumb", "index", "middle", "ring", "pinky")), HandState.OPEN_PALM),
            (observation(()), HandState.CLOSED_FIST),
            (observation(("index",)), HandState.POINTING),
            (observation(("index", "middle")), HandState.PEACE_SIGN),
            (observation(("thumb",)), HandState.THUMBS_UP),
            (observation(("thumb", "index", "middle", "ring", "pinky"), pinch=True), HandState.PINCHING),
        )

        for hand, expected in examples:
            with self.subTest(expected=expected):
                self.assertEqual(classify_hand(hand), expected)

    def test_uncertain_finger_geometry_does_not_form_a_gesture(self):
        points = open_hand_points()
        points[8] = HandPoint(0.50, 0.492, 0.0)
        hand = analyze_hand(points, "Left", 0.9)

        self.assertIs(hand.finger_states["index"], FingerState.UNCERTAIN)
        self.assertIsNone(classify_hand(hand))

    def test_pinching_requires_both_tip_distance_and_confident_finger_geometry(self):
        uncertain = observation(("index", "middle", "ring", "pinky"), pinch=True)

        self.assertLess(uncertain.pinch_distance, 0.18)
        self.assertIs(uncertain.finger_states["index"], FingerState.EXTENDED)
        self.assertEqual(classify_hand(uncertain), HandState.PINCHING)

    def test_pinch_enter_and_exit_margins_are_configurable(self):
        points = open_hand_points()
        points[4] = HandPoint(0.42, 0.337, 0.0)
        hand = analyze_hand(points, "Left", 0.9)
        self.assertGreater(hand.pinch_distance, 0.16)
        self.assertLess(hand.pinch_distance, 0.18)

        self.assertIs(classify_hand(hand), HandState.OPEN_PALM)
        self.assertIs(classify_hand(hand, HandState.PINCHING), HandState.PINCHING)
        self.assertIs(classify_hand(hand, enter_margin=0.04), HandState.OPEN_PALM)
        self.assertIs(classify_hand(hand, enter_margin=0.0), HandState.PINCHING)

    def test_uncertain_thumb_does_not_form_pinch_and_exit_margin_boundary_is_inclusive(self):
        points = open_hand_points()
        points[1] = HandPoint(0.43, 0.31)
        points[3] = HandPoint(0.44, 0.32)
        points[4] = HandPoint(0.42, 0.30)
        points[12] = HandPoint(0.5, 0.8)
        points[16] = HandPoint(0.5, 0.8)
        points[20] = points[18]
        uncertain_thumb = analyze_hand(points, "Left", 0.9)

        self.assertIs(uncertain_thumb.finger_states["thumb"], FingerState.UNCERTAIN)
        self.assertLess(uncertain_thumb.pinch_distance, 0.18)
        self.assertIsNone(classify_hand(uncertain_thumb))

        boundary = observation(("thumb", "index", "middle", "ring", "pinky"))
        from dataclasses import replace
        at_exit = replace(boundary, pinch_distance=0.21)
        beyond_exit = replace(boundary, pinch_distance=0.21001)
        self.assertIs(classify_hand(at_exit, HandState.PINCHING), HandState.PINCHING)
        self.assertIs(classify_hand(beyond_exit, HandState.PINCHING), HandState.OPEN_PALM)


class TemporalHandClassifierTests(unittest.TestCase):
    def test_new_state_requires_three_consecutive_observations(self):
        classifier = TemporalHandClassifier(confirm_frames=3)

        self.assertIsNone(classifier.update("left", HandState.OPEN_PALM))
        self.assertIsNone(classifier.update("left", HandState.OPEN_PALM))
        self.assertIs(classifier.update("left", HandState.OPEN_PALM), HandState.OPEN_PALM)

    def test_one_frame_noise_does_not_change_state_and_new_state_can_stabilize(self):
        classifier = TemporalHandClassifier(confirm_frames=3)
        for _ in range(3):
            current = classifier.update("left", HandState.OPEN_PALM)
        self.assertIs(current, HandState.OPEN_PALM)

        self.assertIs(classifier.update("left", HandState.POINTING), HandState.OPEN_PALM)
        self.assertIs(classifier.update("left", HandState.OPEN_PALM), HandState.OPEN_PALM)
        self.assertIs(classifier.update("left", HandState.POINTING), HandState.OPEN_PALM)
        self.assertIs(classifier.update("left", HandState.POINTING), HandState.OPEN_PALM)
        self.assertIs(classifier.update("left", HandState.POINTING), HandState.POINTING)

    def test_three_uncertain_observations_release_the_previous_state(self):
        classifier = TemporalHandClassifier(confirm_frames=3)
        for _ in range(3):
            classifier.update("right", HandState.CLOSED_FIST)

        self.assertIs(classifier.update("right", None), HandState.CLOSED_FIST)
        self.assertIs(classifier.update("right", None), HandState.CLOSED_FIST)
        self.assertIsNone(classifier.update("right", None))

    def test_each_hand_keeps_an_independent_temporal_state(self):
        classifier = TemporalHandClassifier(confirm_frames=2)

        self.assertIsNone(classifier.update("left", HandState.OPEN_PALM))
        self.assertIsNone(classifier.update("right", HandState.POINTING))
        self.assertIs(classifier.update("right", HandState.POINTING), HandState.POINTING)
        self.assertIs(classifier.update("left", HandState.OPEN_PALM), HandState.OPEN_PALM)


class HandBodyAssociationTests(unittest.TestCase):
    def test_associates_each_hand_to_the_nearest_consistent_body_wrist(self):
        left_points = open_hand_points()
        left_points[0] = HandPoint(0.21, 0.80)
        left = analyze_hand(left_points, "Left", 0.9)
        right_points = open_hand_points()
        right_points = [HandPoint(point.x, point.y, point.z, point.visibility) for point in right_points]
        right_points[0] = HandPoint(0.79, 0.80)
        right = analyze_hand(right_points, "Right", 0.8)

        associated = associate_hands((left, right), body_landmarks())

        self.assertEqual([hand.body_side for hand in associated], ["left", "right"])
        self.assertGreater(associated[0].association_confidence, 0.9)
        self.assertGreater(associated[1].association_confidence, 0.9)

    def test_handedness_breaks_an_ambiguous_equal_distance_wrist_match(self):
        left = observation(("index",), "Left")
        body = body_landmarks(left_wrist=(0.5, 0.8), right_wrist=(0.5, 0.8))

        associated = associate_hands((left,), body)

        self.assertEqual(associated[0].body_side, "left")

    def test_unknown_handedness_remains_unassociated_when_wrist_scores_tie(self):
        hand = observation(("index",), "Unknown")
        body = body_landmarks(left_wrist=(0.5, 0.8), right_wrist=(0.5, 0.8))

        associated = associate_hands((hand,), body)

        self.assertIsNone(associated[0].body_side)

    def test_association_is_one_to_one_even_when_hands_share_one_nearby_wrist(self):
        first_points = open_hand_points()
        first_points[0] = HandPoint(0.21, 0.80)
        first = analyze_hand(first_points, "Left", 0.9)
        second_points = open_hand_points()
        second_points[0] = HandPoint(0.21, 0.80)
        second = analyze_hand(second_points, "Right", 0.8)

        associated = associate_hands((first, second), body_landmarks(right_wrist=(0.95, 0.95)))

        self.assertEqual(sum(hand.body_side is not None for hand in associated), 1)
        self.assertEqual(associated[0].body_side, "left")
        self.assertIsNone(associated[1].body_side)

    def test_missing_or_distant_body_landmarks_leave_hand_unassociated(self):
        hand = observation(("index",), "Left")

        self.assertIsNone(associate_hands((hand,), {})[0].body_side)
        distant = associate_hands((hand,), body_landmarks(left_wrist=(0.95, 0.95), right_wrist=(0.9, 0.9)))
        self.assertIsNone(distant[0].body_side)


if __name__ == "__main__":
    unittest.main()
