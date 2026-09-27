"""Record landmark-only evaluation traces and compare replay to hand labels."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shlex
import sys
from typing import Callable, Iterable, Mapping

from roomsense.config import RoomObjectConfig, RoomSenseConfig
from roomsense.interactions.engine import InteractionEngine, InteractionObservation
from roomsense.interactions.events import EventType
from roomsense.spatial.room_objects import RoomObject
from roomsense.tracking.pose_tracker import Landmark


EVALUATION_SCHEMA_VERSION = 1
LABEL_SCHEMA_VERSION = 1
CAPTURED_LANDMARKS = frozenset({
    "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
})
GESTURE_NAMES = frozenset({
    "SWIPE_LEFT", "SWIPE_RIGHT", "BOTH_HANDS_UP", "POINT", "HOLD_POINT",
})
_SETTING_NAMES = (
    "landmark_visibility", "smoothing_alpha", "pointing_min_visibility", "pointing_min_extension",
    "pointing_smoothing_alpha", "target_stability_seconds", "target_hold_seconds",
    "gesture_swipe_distance", "gesture_swipe_window_seconds",
    "gesture_both_hands_hold_seconds", "gesture_hold_point_seconds",
    "gesture_cooldown_seconds", "command_mode_timeout_seconds", "raised_hand_margin",
)


@dataclass(frozen=True)
class EvaluationSample:
    timestamp: float
    landmarks: Mapping[str, Landmark]


@dataclass(frozen=True)
class EvaluationSession:
    settings: Mapping[str, object]
    room_objects: tuple[RoomObjectConfig, ...]
    samples: tuple[EvaluationSample, ...]

    @property
    def duration(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        return self.samples[-1].timestamp - self.samples[0].timestamp


@dataclass(frozen=True)
class EvaluationLabel:
    """An intended gesture, target, or no-action interval in relative seconds."""

    start: float
    end: float
    gesture: str | None = None
    target_id: str | None = None
    no_action: bool = False


@dataclass(frozen=True)
class GestureMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float | None
    recall: float | None


@dataclass(frozen=True)
class EvaluationReport:
    gestures: Mapping[str, GestureMetrics]
    false_gesture_activations_per_minute: float
    false_target_activations_per_minute: float
    target_accuracy: float | None
    gesture_latency_seconds: Mapping[str, float]
    duration_seconds: float
    settings: Mapping[str, object]


class EvaluationRecorder:
    """Write explicitly enabled pose samples; image and video data are excluded."""

    def __init__(self, path: str | Path, config: RoomSenseConfig) -> None:
        self.path = Path(path).expanduser()
        self.config = config
        self._stream = None
        self._last_timestamp: float | None = None

    @property
    def active(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        if self.active:
            raise RuntimeError("evaluation capture is already active")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("w", encoding="utf-8")
        self._last_timestamp = None
        payload = {
            "settings": {name: getattr(self.config, name) for name in _SETTING_NAMES},
            "room_objects": [
                {
                    "id": item.id,
                    "name": item.name,
                    "image_region": [list(point) for point in item.image_region],
                    "enabled": True,
                    "room_position": list(item.room_position) if item.room_position is not None else None,
                    "interaction_radius": item.interaction_radius,
                }
                for item in self.config.room_objects if item.enabled
            ],
            "landmarks_are_pose_tracker_outputs": True,
        }
        self._write({"schema_version": EVALUATION_SCHEMA_VERSION, "kind": "metadata", "payload": payload})

    def record(self, timestamp: float, landmarks: Mapping[str, Landmark] | None) -> None:
        if not self.active:
            raise RuntimeError("evaluation capture is not active")
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            raise ValueError("evaluation timestamp must be finite")
        timestamp = float(timestamp)
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("evaluation timestamps must be non-decreasing")
        values: dict[str, dict[str, float]] = {}
        if landmarks is not None:
            for name in CAPTURED_LANDMARKS:
                point = landmarks.get(name)
                if point is None:
                    continue
                if not all(math.isfinite(value) for value in (point.x, point.y, point.visibility)) \
                        or not 0.0 <= point.visibility <= 1.0:
                    raise ValueError(f"landmark {name} must have finite coordinates and visibility in [0, 1]")
                values[name] = {"x": float(point.x), "y": float(point.y), "visibility": float(point.visibility)}
        self._write({
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "kind": "sample",
            "timestamp": timestamp,
            "landmarks": values,
        })
        self._last_timestamp = timestamp

    def close(self) -> None:
        stream = self._stream
        if stream is None:
            return
        try:
            stream.flush()
        finally:
            stream.close()
            self._stream = None

    def _write(self, record: Mapping[str, object]) -> None:
        if not self.active:
            raise RuntimeError("evaluation capture is not active")
        self._stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")


def load_evaluation(path: str | Path) -> EvaluationSession:
    """Read and validate an evaluation capture, with line-numbered errors."""

    settings: Mapping[str, object] | None = None
    objects: tuple[RoomObjectConfig, ...] = ()
    samples: list[EvaluationSample] = []
    previous_timestamp: float | None = None
    metadata_seen = False
    with Path(path).expanduser().open("r", encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            try:
                record = json.loads(raw_line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                if not isinstance(record, dict) or record.get("schema_version") != EVALUATION_SCHEMA_VERSION:
                    raise ValueError("unsupported or malformed evaluation record")
                kind = record.get("kind")
                if kind == "metadata":
                    if metadata_seen or samples:
                        raise ValueError("metadata must appear exactly once at the start")
                    payload = record.get("payload")
                    if not isinstance(payload, dict):
                        raise ValueError("metadata payload must be an object")
                    settings_value = payload.get("settings")
                    object_values = payload.get("room_objects")
                    if not isinstance(settings_value, dict) or not isinstance(object_values, list):
                        raise ValueError("metadata must include settings and room_objects")
                    settings = _validate_settings(settings_value)
                    objects = tuple(_object_from_record(value) for value in object_values)
                    if len({item.id for item in objects}) != len(objects):
                        raise ValueError("room object ids must be unique")
                    metadata_seen = True
                elif kind == "sample":
                    if not metadata_seen:
                        raise ValueError("metadata must precede samples")
                    if any(key in record for key in ("frame", "video", "image_data", "pixels")):
                        raise ValueError("evaluation samples cannot contain image or video data")
                    timestamp = record.get("timestamp")
                    landmarks_value = record.get("landmarks")
                    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) \
                            or not math.isfinite(timestamp):
                        raise ValueError("sample timestamp must be finite")
                    if previous_timestamp is not None and timestamp < previous_timestamp:
                        raise ValueError("sample timestamps must be non-decreasing")
                    if not isinstance(landmarks_value, dict):
                        raise ValueError("sample landmarks must be an object")
                    landmarks: dict[str, Landmark] = {}
                    for name, point in landmarks_value.items():
                        if name not in CAPTURED_LANDMARKS or not isinstance(point, dict):
                            raise ValueError(f"unsupported or malformed landmark: {name}")
                        if set(point) != {"x", "y", "visibility"}:
                            raise ValueError(f"landmark {name} requires x, y, and visibility")
                        x, y, visibility = point["x"], point["y"], point["visibility"]
                        if any(isinstance(value, bool) or not isinstance(value, (int, float))
                               or not math.isfinite(value) for value in (x, y, visibility)):
                            raise ValueError(f"landmark {name} values must be finite numbers")
                        if not 0.0 <= visibility <= 1.0:
                            raise ValueError(f"landmark {name} visibility must be in [0, 1]")
                        landmarks[name] = Landmark(float(x), float(y), visibility=float(visibility))
                    samples.append(EvaluationSample(float(timestamp), landmarks))
                    previous_timestamp = float(timestamp)
                else:
                    raise ValueError(f"unknown evaluation record kind: {kind}")
            except (json.JSONDecodeError, TypeError, KeyError, ValueError, OverflowError) as exc:
                raise ValueError(f"invalid evaluation record at line {line_number}: {exc}") from exc
    if not metadata_seen or not samples or settings is None:
        raise ValueError("evaluation capture must contain metadata and at least one sample")
    return EvaluationSession(settings, objects, tuple(samples))


def save_labels(
    path: str | Path,
    capture_path: str | Path,
    labels: Iterable[EvaluationLabel],
    *,
    session: EvaluationSession | None = None,
) -> None:
    capture = session or load_evaluation(capture_path)
    values = tuple(labels)
    _validate_labels(values, capture)
    payload = {
        "schema_version": LABEL_SCHEMA_VERSION,
        "capture": Path(capture_path).name,
        "intervals": [
            {"start": label.start, "end": label.end, "gesture": label.gesture,
             "target_id": label.target_id, "no_action": label.no_action}
            for label in values
        ],
    }
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_labels(path: str | Path, capture_path: str | Path, *,
                session: EvaluationSession | None = None) -> tuple[EvaluationLabel, ...]:
    capture = session or load_evaluation(capture_path)
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != LABEL_SCHEMA_VERSION:
            raise ValueError("unsupported labels schema")
        if payload.get("capture") != Path(capture_path).name:
            raise ValueError("labels file belongs to a different capture")
        intervals = payload.get("intervals")
        if not isinstance(intervals, list):
            raise ValueError("labels intervals must be a list")
        labels = tuple(EvaluationLabel(
            start=item["start"], end=item["end"], gesture=item.get("gesture"),
            target_id=item.get("target_id"), no_action=item.get("no_action", False),
        ) for item in intervals if isinstance(item, dict))
        if len(labels) != len(intervals):
            raise ValueError("each labels interval must be an object")
        _validate_labels(labels, capture)
        return labels
    except (OSError, json.JSONDecodeError, TypeError, KeyError) as exc:
        raise ValueError(f"could not read labels: {exc}") from exc


def evaluate_session(
    session: EvaluationSession,
    labels: Iterable[EvaluationLabel],
    *,
    overrides: Mapping[str, float] | None = None,
) -> EvaluationReport:
    label_values = tuple(labels)
    _validate_labels(label_values, session)
    if not session.samples:
        raise ValueError("evaluation session has no samples")
    first_timestamp = session.samples[0].timestamp
    objects = tuple(RoomObject(
        item.id, item.name, item.image_region, item.enabled, item.room_position, item.interaction_radius,
    ) for item in session.room_objects)
    config_values = dict(session.settings)
    for name, value in (overrides or {}).items():
        if name not in _SETTING_NAMES:
            raise ValueError(f"unsupported tuning setting: {name}")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"tuning setting {name} must be finite")
        if name == "smoothing_alpha":
            raise ValueError("cannot tune smoothing_alpha because captured landmarks are already smoothed")
        if name == "landmark_visibility" and value < session.settings["landmark_visibility"]:
            raise ValueError("cannot lower landmark_visibility because filtered landmarks are not in the capture")
        config_values[name] = float(value)
    config_values["room_objects"] = session.room_objects
    config = RoomSenseConfig(**config_values)
    engine = InteractionEngine(config, objects)
    gesture_predictions: list[tuple[float, str]] = []
    target_predictions: list[tuple[float, str | None]] = []
    target_stream: list[str | None] = []
    for sample in session.samples:
        replay_landmarks = {
            name: point for name, point in sample.landmarks.items()
            if point.visibility >= config.landmark_visibility
        }
        result = engine.update(InteractionObservation(
            timestamp=sample.timestamp,
            landmarks=replay_landmarks or None,
            room_position=None,
            zone_entered=(),
            zone_left=(),
            movement_state="STILL",
        ))
        for event in result.events:
            if event.type is EventType.GESTURE_DETECTED:
                name = event.metadata.get("gesture")
                if isinstance(name, str):
                    gesture_predictions.append((sample.timestamp, name))
            elif event.type is EventType.OBJECT_POINTED:
                object_id = event.metadata.get("object_id")
                target_predictions.append((sample.timestamp, object_id if isinstance(object_id, str) else None))
        target_stream.append(result.target.confirmed.object.id if result.target.confirmed else None)

    gesture_metrics: dict[str, GestureMetrics] = {}
    latencies: dict[str, float] = {}
    false_gestures = 0
    for name in sorted(GESTURE_NAMES):
        intervals = [label for label in label_values if label.gesture == name]
        predicted = [(stamp, event_name) for stamp, event_name in gesture_predictions if event_name == name]
        matches, false_positive, false_negative, delays = _match_intervals(predicted, intervals, first_timestamp)
        precision = matches / (matches + false_positive) if matches + false_positive else None
        recall = matches / (matches + false_negative) if matches + false_negative else None
        gesture_metrics[name] = GestureMetrics(matches, false_positive, false_negative, precision, recall)
        false_gestures += false_positive
        if delays:
            latencies[name] = sum(delays) / len(delays)

    target_intervals = [label for label in label_values if label.target_id is not None]
    correct_targets = 0
    target_samples = 0
    for sample, object_id in zip(session.samples, target_stream):
        offset = sample.timestamp - first_timestamp
        target_label = next((label for label in target_intervals
                             if label.start <= offset < label.end), None)
        if target_label is not None:
            target_samples += 1
            if object_id == target_label.target_id:
                correct_targets += 1
    false_target_count = sum(
        not any(label.target_id == object_id and label.start <= stamp - first_timestamp < label.end
                for label in target_intervals)
        for stamp, object_id in target_predictions
    )
    minutes = session.duration / 60.0
    return EvaluationReport(
        gestures=gesture_metrics,
        false_gesture_activations_per_minute=false_gestures / minutes if minutes > 0 else 0.0,
        false_target_activations_per_minute=false_target_count / minutes if minutes > 0 else 0.0,
        target_accuracy=correct_targets / target_samples if target_samples else None,
        gesture_latency_seconds=latencies,
        duration_seconds=session.duration,
        settings={name: getattr(config, name) for name in _SETTING_NAMES},
    )


def annotate_interactively(
    capture_path: str | Path,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> Path:
    session = load_evaluation(capture_path)
    output_fn(
        f"Capture duration: {session.duration:.2f}s. Add labels as: gesture START END NAME, "
        "target START END ID, or none START END. Times are seconds from the first sample; type done to save."
    )
    labels: list[EvaluationLabel] = []
    while True:
        line = input_fn("label> ").strip()
        if line.lower() in ("done", "quit", "q"):
            break
        if not line:
            continue
        try:
            fields = shlex.split(line)
            if fields and fields[0].lower() == "none" and len(fields) == 3:
                kind, start_text, end_text = fields
                label = EvaluationLabel(float(start_text), float(end_text), no_action=True)
            elif len(fields) == 4:
                kind, start_text, end_text, value = fields
                start, end = float(start_text), float(end_text)
                if kind.lower() == "gesture":
                    label = EvaluationLabel(start, end, gesture=value.upper())
                elif kind.lower() == "target":
                    label = EvaluationLabel(start, end, target_id=value)
                else:
                    raise ValueError("label kind must be gesture, target, or none")
            else:
                raise ValueError("enter gesture START END NAME, target START END ID, or none START END")
            _validate_labels(tuple(labels) + (label,), session)
            labels.append(label)
        except (TypeError, ValueError) as exc:
            output_fn(f"Invalid label: {exc}")
    label_path = _default_labels_path(capture_path)
    save_labels(label_path, capture_path, labels, session=session)
    output_fn(f"Saved labels to {label_path}")
    return label_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Annotate and evaluate landmark-only RoomSense captures.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    annotate_parser = subparsers.add_parser("annotate", help="label a capture after recording")
    annotate_parser.add_argument("capture", type=Path)
    evaluate_parser = subparsers.add_parser("evaluate", help="replay a capture against its labels")
    evaluate_parser.add_argument("capture", type=Path)
    evaluate_parser.add_argument("--labels", type=Path)
    evaluate_parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="SETTING=VALUE",
        help="replay with a candidate detector setting; may be repeated",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "annotate":
            annotate_interactively(args.capture)
            return 0
        session = load_evaluation(args.capture)
        label_path = args.labels or _default_labels_path(args.capture)
        labels = load_labels(label_path, args.capture, session=session)
        overrides: dict[str, float] = {}
        for item in args.overrides:
            name, separator, value = item.partition("=")
            if not separator:
                raise ValueError("--set values must use SETTING=VALUE")
            try:
                overrides[name] = float(value)
            except ValueError as exc:
                raise ValueError(f"--set value for {name} must be a number") from exc
        report = evaluate_session(session, labels, overrides=overrides)
        print(json.dumps(_report_dict(report), indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ValueError) as exc:
        print(f"could not evaluate capture: {exc}", file=sys.stderr)
        return 2


def _validate_settings(value: Mapping[str, object]) -> Mapping[str, object]:
    if set(value) != set(_SETTING_NAMES):
        raise ValueError("metadata settings are incomplete or unsupported")
    settings: dict[str, object] = {}
    for name in _SETTING_NAMES:
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise ValueError(f"setting {name} must be a finite number")
        settings[name] = item
    return settings


def _object_from_record(value: object) -> RoomObjectConfig:
    if not isinstance(value, dict):
        raise ValueError("room object must be an object")
    try:
        obj = RoomObjectConfig(
            id=value["id"], name=value["name"],
            image_region=tuple(tuple(point) for point in value["image_region"]),
            enabled=value["enabled"],
            room_position=tuple(value["room_position"]) if value.get("room_position") is not None else None,
            interaction_radius=value.get("interaction_radius"),
        )
        RoomObject(obj.id, obj.name, obj.image_region, obj.enabled, obj.room_position, obj.interaction_radius)
        if not obj.enabled:
            raise ValueError("evaluation capture room objects must be enabled")
        return obj
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid room object: {exc}") from exc


def _validate_labels(labels: tuple[EvaluationLabel, ...], session: EvaluationSession) -> None:
    duration = session.duration
    object_ids = {item.id for item in session.room_objects}
    for label in labels:
        if not isinstance(label, EvaluationLabel):
            raise ValueError("labels must be EvaluationLabel entries")
        if not isinstance(label.no_action, bool):
            raise ValueError("no_action must be a boolean")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in (label.start, label.end)) \
                or label.start < 0.0 or label.end <= label.start or label.end > duration + 1e-9:
            raise ValueError("label interval must have positive duration within capture duration")
        if label.no_action:
            if label.gesture is not None or label.target_id is not None:
                raise ValueError("no-action labels cannot include a gesture or target")
        elif label.gesture is None and label.target_id is None:
            raise ValueError("label must include a gesture, target, or no-action interval")
        if label.gesture is not None and label.gesture not in GESTURE_NAMES:
            raise ValueError(f"unsupported gesture: {label.gesture}")
        if label.target_id is not None and label.target_id not in object_ids:
            raise ValueError(f"target is not enabled in capture: {label.target_id}")
    for index, first in enumerate(labels):
        for second in labels[index + 1:]:
            overlaps = max(first.start, second.start) < min(first.end, second.end)
            if not overlaps:
                continue
            if first.no_action != second.no_action and (first.no_action or second.no_action):
                raise ValueError("no-action labels cannot overlap gesture or target labels")
            if first.gesture is not None and first.gesture == second.gesture:
                raise ValueError(f"overlapping intervals for gesture {first.gesture}")
            if first.target_id is not None and second.target_id is not None \
                    and first.target_id != second.target_id:
                raise ValueError("target labels for different objects cannot overlap")


def _match_intervals(
    predictions: list[tuple[float, str]],
    intervals: list[EvaluationLabel],
    first_timestamp: float,
) -> tuple[int, int, int, list[float]]:
    used: set[int] = set()
    delays: list[float] = []
    true_positive = 0
    for interval in intervals:
        candidates = [
            (index, timestamp) for index, (timestamp, _name) in enumerate(predictions)
            if index not in used and interval.start <= timestamp - first_timestamp < interval.end
        ]
        if candidates:
            index, timestamp = min(candidates, key=lambda item: item[1])
            used.add(index)
            true_positive += 1
            delays.append(max(0.0, timestamp - first_timestamp - interval.start))
    false_positive = len(predictions) - len(used)
    false_negative = len(intervals) - true_positive
    return true_positive, false_positive, false_negative, delays


def _default_labels_path(capture_path: str | Path) -> Path:
    path = Path(capture_path)
    return path.with_name(path.stem + ".labels.json")


def _report_dict(report: EvaluationReport) -> dict[str, object]:
    return {
        "duration_seconds": report.duration_seconds,
        "false_gesture_activations_per_minute": report.false_gesture_activations_per_minute,
        "false_target_activations_per_minute": report.false_target_activations_per_minute,
        "target_accuracy": report.target_accuracy,
        "gesture_latency_seconds": dict(report.gesture_latency_seconds),
        "settings": dict(report.settings),
        "gestures": {
            name: {
                "true_positives": metrics.true_positives,
                "false_positives": metrics.false_positives,
                "false_negatives": metrics.false_negatives,
                "precision": metrics.precision,
                "recall": metrics.recall,
            }
            for name, metrics in report.gestures.items()
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
