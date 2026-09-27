"""MediaPipe Tasks video adapter for two-hand landmark tracking."""

from __future__ import annotations

from dataclasses import replace
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
        self._smoothers: dict[str, tuple[VectorSmoother, ...]] = {}
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
        observations = []
        seen_sides: set[str] = set()

        for index, model_points in enumerate(model_hands):
            side, handedness_score = _handedness(handedness_results, index)
            seen_sides.add(side)
            points, validity = _convert_points(model_points)
            smoothers = self._smoothers.setdefault(
                side,
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
            candidate = classify_hand(observation) if confidence >= self.config.hand_detection_confidence else None
            stable_state = self._classifier.update(side, candidate)
            observations.append(replace(observation, state=stable_state))

        for side in tuple(self._smoothers):
            if side not in seen_sides:
                del self._smoothers[side]
                self._classifier.reset(side)

        return associate_hands(
            tuple(observations), body_landmarks or {}, self.config.hand_max_wrist_distance_ratio,
        )

    def close(self) -> None:
        self._landmarker.close()

    def reset_smoothing(self) -> None:
        self._smoothers.clear()
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


def _clamp(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))
