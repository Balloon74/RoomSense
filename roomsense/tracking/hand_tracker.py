"""MediaPipe Tasks video adapter for two-hand landmark tracking."""

from __future__ import annotations

from dataclasses import replace
from itertools import product
import math
from pathlib import Path
from typing import Any, Mapping
from urllib.request import urlopen

from roomsense.config import RoomSenseConfig
from roomsense.tracking.hand_classifier import TemporalHandClassifier, associate_hands, classify_hand
from roomsense.tracking.hand_geometry import HandObservation, HandPoint, analyze_hand
from roomsense.tracking.pose_tracker import Landmark
from roomsense.utils.smoothing import VectorSmoother


HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/"
    "float16/1/hand_landmarker.task"
)


class HandTracker:
    """Find up to two hands from RGB video frames and return smoothed observations."""

    def __init__(self, config: RoomSenseConfig) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("MediaPipe is missing. Install requirements.txt first.") from exc

        self.config = config
        self._mp = mp
        model_path = self._cached_model_path()
        if not model_path.exists():
            self._download_model(model_path)
        vision = mp.tasks.vision
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=config.hand_detection_confidence,
            min_hand_presence_confidence=config.hand_presence_confidence,
            min_tracking_confidence=config.hand_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._smoothers: dict[int, tuple[VectorSmoother, ...]] = {}
        self._track_wrist: dict[int, tuple[float, float]] = {}
        self._track_handedness: dict[int, str] = {}
        self._classifier = TemporalHandClassifier(config.hand_gesture_confirm_frames)
        self._last_timestamp_ms = 0

    def process(
        self,
        rgb_frame: Any,
        timestamp_ms: int,
        body_landmarks: Mapping[str, Landmark] | None = None,
    ) -> tuple[HandObservation, ...]:
        timestamp = max(int(timestamp_ms), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(image, timestamp)
        model_hands = tuple(getattr(result, "hand_landmarks", ()) or ())[:2]
        handedness_results = tuple(getattr(result, "handedness", ()) or ())
        detections = []

        for index, model_points in enumerate(model_hands):
            side, handedness_score = _handedness(handedness_results, index)
            points, validity = _convert_points(model_points)
            detections.append((side, handedness_score, points, validity))

        track_ids = _assign_tracks(
            detections,
            self._track_wrist,
            self._track_handedness,
        )
        observations = []
        seen_tracks: set[int] = set()
        for (side, handedness_score, points, validity), track_id in zip(detections, track_ids):
            seen_tracks.add(track_id)
            wrist = points[0]
            if math.isfinite(wrist.x) and math.isfinite(wrist.y):
                self._track_wrist[track_id] = (wrist.x, wrist.y)
            self._track_handedness[track_id] = side
            smoothers = self._smoothers.setdefault(
                track_id,
                tuple(VectorSmoother(self.config.hand_smoothing_alpha) for _ in range(21)),
            )
            if validity < 1.0:
                for smoother in smoothers:
                    smoother.reset()
                smoothed = list(points)
            else:
                smoothed = []
                for point, smoother in zip(points, smoothers):
                    x, y, z = smoother.update((point.x, point.y, point.z))
                    smoothed.append(HandPoint(x, y, z, point.visibility))

            confidence = _clamp(handedness_score * validity)
            observation = analyze_hand(smoothed, side, confidence)
            track_key = str(track_id)
            candidate = classify_hand(
                observation,
                self._classifier.current(track_key),
                enter_margin=self.config.hand_gesture_enter_margin,
                exit_margin=self.config.hand_gesture_exit_margin,
            ) if confidence >= self.config.hand_detection_confidence else None
            stable_state = self._classifier.update(track_key, candidate)
            observations.append(replace(observation, state=stable_state))

        for track_id in tuple(self._smoothers):
            if track_id not in seen_tracks:
                del self._smoothers[track_id]
                self._track_wrist.pop(track_id, None)
                self._track_handedness.pop(track_id, None)
                self._classifier.reset(str(track_id))

        return associate_hands(
            tuple(observations), body_landmarks or {}, self.config.hand_max_wrist_distance_ratio,
        )

    def close(self) -> None:
        self._landmarker.close()

    def reset_smoothing(self) -> None:
        self._smoothers.clear()
        self._track_wrist.clear()
        self._track_handedness.clear()
        self._classifier.reset()

    @staticmethod
    def _cached_model_path() -> Path:
        return Path.home() / ".cache" / "roomsense" / "hand_landmarker.task"

    @staticmethod
    def _download_model(destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".download")
        try:
            with urlopen(HAND_MODEL_URL, timeout=45) as response, temporary.open("wb") as model_file:
                while chunk := response.read(1024 * 1024):
                    model_file.write(chunk)
            temporary.replace(destination)
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                "Could not download the MediaPipe hand model. Check your internet connection, "
                "or verify access to the configured hand model cache."
            ) from exc


def _handedness(results: tuple[Any, ...], index: int) -> tuple[str, float]:
    categories = results[index] if index < len(results) else ()
    category = categories[0] if categories else None
    if category is None:
        return "Unknown", 0.0
    name = str(getattr(category, "category_name", "Unknown"))
    side = name.title() if name.casefold() in {"left", "right"} else "Unknown"
    try:
        confidence = float(getattr(category, "score", 0.0))
    except (TypeError, ValueError, OverflowError):
        confidence = 0.0
    return side, confidence if math.isfinite(confidence) else 0.0


def _convert_points(model_points: Any) -> tuple[tuple[HandPoint, ...], float]:
    if len(model_points) != 21:
        raise ValueError("MediaPipe hand results must contain exactly 21 landmarks")
    points = []
    valid_count = 0
    for point in model_points:
        try:
            x, y, z = float(point.x), float(point.y), float(point.z)
        except (TypeError, ValueError, OverflowError, AttributeError):
            x = y = z = float("nan")
        valid = all(math.isfinite(value) for value in (x, y, z))
        visibility = 1.0 if valid else 0.0
        points.append(HandPoint(x, y, z, visibility))
        valid_count += int(valid)
    return tuple(points), valid_count / 21


def _assign_tracks(
    detections: list[tuple[str, float, tuple[HandPoint, ...], float]],
    previous_wrists: Mapping[int, tuple[float, float]],
    previous_handedness: Mapping[int, str],
) -> tuple[int, ...]:
    """Match detections to prior per-hand identities before smoothing or classifying."""
    if not detections:
        return ()
    track_ids = tuple(sorted(previous_wrists))
    options = [tuple([None, *track_ids]) for _ in detections]
    best_assignment: tuple[int | None, ...] = tuple(None for _ in detections)
    best_score: tuple[int, float] | None = None
    for assignment in product(*options):
        matched = [track_id for track_id in assignment if track_id is not None]
        if len(matched) != len(set(matched)):
            continue
        mismatch_cost = sum(
            0.15 for (side, _, _, _), track_id in zip(detections, assignment)
            if track_id is not None and side in {"Left", "Right"}
            and previous_handedness.get(track_id) in {"Left", "Right"}
            and side != previous_handedness.get(track_id)
        )
        distance_cost = 0.0
        for (_, _, points, _), track_id in zip(detections, assignment):
            if track_id is None:
                continue
            point = points[0]
            old_x, old_y = previous_wrists[track_id]
            distance = math.hypot(point.x - old_x, point.y - old_y)
            distance_cost += distance if math.isfinite(distance) else 1e6
        score = (-len(matched), distance_cost + mismatch_cost)
        if best_score is None or score < best_score:
            best_assignment = assignment
            best_score = score

    assigned = list(best_assignment)
    free_ids = [track_id for track_id in range(2) if track_id not in assigned]
    for index, track_id in enumerate(assigned):
        if track_id is None:
            assigned[index] = free_ids.pop(0)
    return tuple(int(track_id) for track_id in assigned)


def _clamp(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))
