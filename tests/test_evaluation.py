import json
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO

from roomsense.config import RoomObjectConfig, RoomSenseConfig
from roomsense.evaluation import (
    EvaluationLabel,
    EvaluationRecorder,
    evaluate_session,
    annotate_interactively,
    load_evaluation,
    load_labels,
    main,
    save_labels,
)
from roomsense.recording.evaluation_control import EvaluationCaptureController
from roomsense.tracking.pose_tracker import Landmark


def pointing_pose(wrist_x=0.45):
    return {
        "right_shoulder": Landmark(0.25, 0.5, visibility=0.95),
        "right_elbow": Landmark(0.35, 0.5, visibility=0.95),
        "right_wrist": Landmark(wrist_x, 0.5, visibility=0.95),
    }


class EvaluationCaptureTests(unittest.TestCase):
    def test_capture_is_opt_in_and_records_only_landmarks_and_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.jsonl"
            config = RoomSenseConfig(
                pointing_min_visibility=0.61,
                target_stability_seconds=0.12,
                raised_hand_margin=0.09,
                room_objects=(RoomObjectConfig(
                    "lamp", "LAMP", ((0.55, 0.4), (0.8, 0.4), (0.8, 0.6), (0.55, 0.6)),
                    enabled=True,
                ),),
            )
            controller = EvaluationCaptureController(lambda: EvaluationRecorder(path, config))
            self.assertFalse(controller.active)
            controller.record(1.0, pointing_pose())
            self.assertFalse(path.exists())

            self.assertTrue(controller.toggle())
            controller.record(1.0, pointing_pose())
            controller.record(1.1, None)
            self.assertFalse(controller.toggle())

            session = load_evaluation(path)
            self.assertEqual(session.settings["pointing_min_visibility"], 0.61)
            self.assertEqual(session.settings["raised_hand_margin"], 0.09)
            self.assertEqual(session.room_objects[0].id, "lamp")
            self.assertEqual(len(session.samples), 2)
            self.assertEqual(set(session.samples[0].landmarks), {
                "right_shoulder", "right_elbow", "right_wrist",
            })
            self.assertEqual(session.samples[1].landmarks, {})
            content = path.read_text()
            self.assertNotIn('"frame"', content)
            self.assertNotIn('"video"', content)

    def test_capture_rejects_invalid_landmarks_and_reader_rejects_incomplete_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.jsonl"
            recorder = EvaluationRecorder(path, RoomSenseConfig())
            recorder.start()
            with self.assertRaises(ValueError):
                recorder.record(1.0, {"right_wrist": Landmark(0.5, 0.4, visibility=float("nan"))})
            recorder.close()
            path.write_text(path.read_text() + json.dumps({
                "schema_version": 1,
                "kind": "sample",
                "timestamp": 1.0,
                "landmarks": {"right_wrist": {"x": 0.5, "visibility": 0.8}},
            }) + "\n")
            with self.assertRaisesRegex(ValueError, "line 2"):
                load_evaluation(path)

    def test_annotations_round_trip_and_replay_reports_gesture_target_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory) / "evaluation.jsonl"
            labels_path = Path(directory) / "evaluation.labels.json"
            config = RoomSenseConfig(
                target_stability_seconds=0.1,
                target_hold_seconds=0.5,
                room_objects=(RoomObjectConfig(
                    "lamp", "LAMP", ((0.55, 0.4), (0.8, 0.4), (0.8, 0.6), (0.55, 0.6)),
                    enabled=True,
                ),),
            )
            recorder = EvaluationRecorder(capture, config)
            recorder.start()
            for index in range(12):
                recorder.record(10.0 + index * 0.1, pointing_pose())
            recorder.close()
            labels = (
                EvaluationLabel(0.0, 1.0, gesture="POINT"),
                EvaluationLabel(0.5, 1.0, gesture="HOLD_POINT"),
                EvaluationLabel(0.1, 1.0, target_id="lamp"),
                EvaluationLabel(1.0, 1.1, no_action=True),
            )
            save_labels(labels_path, capture, labels)
            loaded = load_labels(labels_path, capture)
            report = evaluate_session(load_evaluation(capture), loaded)
            self.assertEqual(report.gestures["POINT"].true_positives, 1)
            self.assertEqual(report.gestures["POINT"].false_negatives, 0)
            self.assertGreaterEqual(report.target_accuracy, 0.8)
            self.assertEqual(report.false_gesture_activations_per_minute, 0.0)
            self.assertGreaterEqual(report.gesture_latency_seconds["POINT"], 0.0)

    def test_labels_reject_unknown_gestures_and_out_of_capture_intervals(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory) / "evaluation.jsonl"
            recorder = EvaluationRecorder(capture, RoomSenseConfig())
            recorder.start()
            recorder.record(1.0, None)
            recorder.record(2.0, None)
            recorder.close()
            session = load_evaluation(capture)
            with self.assertRaisesRegex(ValueError, "gesture"):
                save_labels(Path(directory) / "bad.json", capture, (
                    EvaluationLabel(0.0, 0.5, gesture="WAVE"),
                ), session=session)
            with self.assertRaisesRegex(ValueError, "duration"):
                save_labels(Path(directory) / "bad.json", capture, (
                    EvaluationLabel(0.5, 2.0, no_action=True),
                ), session=session)
            with self.assertRaisesRegex(ValueError, "no_action"):
                save_labels(Path(directory) / "bad.json", capture, (
                    EvaluationLabel(0.0, 0.5, no_action="yes"),
                ), session=session)

    def test_unlabeled_swipe_is_counted_as_a_false_activation(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory) / "swipe.jsonl"
            recorder = EvaluationRecorder(capture, RoomSenseConfig(room_objects=()))
            recorder.start()
            for index, wrist_x in enumerate((0.8, 0.8, 0.75, 0.45, 0.45, 0.45)):
                timestamp = index * 0.1
                recorder.record(timestamp, pointing_pose(wrist_x))
            recorder.close()
            session = load_evaluation(capture)
            report = evaluate_session(session, (EvaluationLabel(0.0, 0.5, no_action=True),))
            self.assertEqual(report.gestures["SWIPE_LEFT"].false_positives, 1)
            self.assertGreater(report.false_gesture_activations_per_minute, 0.0)

    def test_candidate_threshold_override_replays_same_capture_for_tuning(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory) / "point.jsonl"
            recorder = EvaluationRecorder(capture, RoomSenseConfig(room_objects=()))
            recorder.start()
            bent = {
                "right_shoulder": Landmark(0.25, 0.5),
                "right_elbow": Landmark(0.35, 0.5),
                "right_wrist": Landmark(0.4, 0.55),
            }
            for index in range(11):
                recorder.record(index * 0.1, bent)
            recorder.close()
            session = load_evaluation(capture)
            labels = (EvaluationLabel(0.0, 1.0, no_action=True),)
            baseline = evaluate_session(session, labels)
            tuned = evaluate_session(session, labels, overrides={"pointing_min_extension": 0.99})
            self.assertEqual(baseline.gestures["POINT"].false_positives, 1)
            self.assertEqual(tuned.gestures["POINT"].false_positives, 0)
            self.assertEqual(tuned.settings["pointing_min_extension"], 0.99)
            with self.assertRaisesRegex(ValueError, "cannot lower landmark_visibility"):
                evaluate_session(session, labels, overrides={"landmark_visibility": 0.1})

    def test_missing_landmark_samples_break_gesture_continuity_during_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory) / "dropout.jsonl"
            recorder = EvaluationRecorder(capture, RoomSenseConfig(room_objects=()))
            recorder.start()
            recorder.record(0.0, pointing_pose(0.8))
            recorder.record(0.1, None)
            recorder.record(0.2, pointing_pose(0.3))
            recorder.record(0.3, pointing_pose(0.28))
            recorder.close()
            session = load_evaluation(capture)
            report = evaluate_session(session, (EvaluationLabel(0.0, 0.3, no_action=True),))
            self.assertEqual(report.gestures["SWIPE_LEFT"].false_positives, 0)

    def test_cli_evaluates_a_labeled_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory) / "capture.jsonl"
            labels_path = Path(directory) / "labels.json"
            recorder = EvaluationRecorder(capture, RoomSenseConfig(room_objects=()))
            recorder.start()
            recorder.record(1.0, None)
            recorder.record(2.0, None)
            recorder.close()
            save_labels(labels_path, capture, (EvaluationLabel(0.0, 1.0, no_action=True),))
            output = StringIO()
            with redirect_stdout(output):
                result = main(["evaluate", str(capture), "--labels", str(labels_path)])
            self.assertEqual(result, 0)
            result_json = json.loads(output.getvalue())
            self.assertEqual(result_json["duration_seconds"], 1.0)
            self.assertIsNone(result_json["gestures"]["BOTH_HANDS_UP"]["precision"])
            self.assertIsNone(result_json["gestures"]["BOTH_HANDS_UP"]["recall"])

    def test_after_capture_annotation_accepts_gestures_targets_and_no_action(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory) / "capture.jsonl"
            config = RoomSenseConfig(room_objects=(RoomObjectConfig(
                "lamp", "LAMP", ((0.55, 0.4), (0.8, 0.4), (0.8, 0.6), (0.55, 0.6)), enabled=True,
            ),))
            recorder = EvaluationRecorder(capture, config)
            recorder.start()
            for index in range(13):
                recorder.record(10.0 + index * 0.1, pointing_pose())
            recorder.close()
            answers = iter(("gesture 0 0.5 POINT", "target 0.1 1.1 lamp", "none 1.1 1.2", "done"))
            annotate_interactively(capture, input_fn=lambda _prompt: next(answers), output_fn=lambda _text: None)
            labels = load_labels(Path(directory) / "capture.labels.json", capture)
            self.assertEqual([label.gesture for label in labels if label.gesture], ["POINT"])
            self.assertEqual([label.target_id for label in labels if label.target_id], ["lamp"])
            self.assertTrue(any(label.no_action for label in labels))

    def test_annotation_prompt_keeps_prior_labels_after_invalid_input(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory) / "capture.jsonl"
            recorder = EvaluationRecorder(capture, RoomSenseConfig(room_objects=()))
            recorder.start()
            recorder.record(1.0, None)
            recorder.record(2.0, None)
            recorder.close()
            answers = iter(("gesture 0 3 WAVE", "none 0 1", "done"))
            messages = []
            annotate_interactively(capture, input_fn=lambda _prompt: next(answers), output_fn=messages.append)
            labels = load_labels(Path(directory) / "capture.labels.json", capture)
            self.assertEqual(len(labels), 1)
            self.assertTrue(labels[0].no_action)
            self.assertTrue(any("Invalid label" in message for message in messages))

    def test_labels_reject_overlapping_no_action_and_conflicting_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory) / "capture.jsonl"
            objects = (
                RoomObjectConfig("lamp", "LAMP", ((0.1, 0.1), (0.2, 0.1), (0.2, 0.2), (0.1, 0.2)), enabled=True),
                RoomObjectConfig("tv", "TV", ((0.3, 0.3), (0.4, 0.3), (0.4, 0.4), (0.3, 0.4)), enabled=True),
            )
            recorder = EvaluationRecorder(capture, RoomSenseConfig(room_objects=objects))
            recorder.start()
            recorder.record(1.0, None)
            recorder.record(2.0, None)
            recorder.close()
            session = load_evaluation(capture)
            with self.assertRaisesRegex(ValueError, "overlap"):
                save_labels(Path(directory) / "bad.json", capture, (
                    EvaluationLabel(0.0, 0.7, gesture="POINT"),
                    EvaluationLabel(0.5, 1.0, no_action=True),
                ), session=session)
            with self.assertRaisesRegex(ValueError, "overlap"):
                save_labels(Path(directory) / "bad.json", capture, (
                    EvaluationLabel(0.0, 0.7, target_id="lamp"),
                    EvaluationLabel(0.5, 1.0, target_id="tv"),
                ), session=session)


if __name__ == "__main__":
    unittest.main()
