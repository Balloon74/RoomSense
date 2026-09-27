"""MediaPipe hand landmark inference, separate from pose tracking and drawing."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from time import monotonic_ns
from typing import Any
from urllib.request import urlopen

from roomsense.config import RoomSenseConfig


HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


@dataclass(frozen=True)
class HandPoint:
    """One normalized MediaPipe hand point in image coordinates."""

    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.x, self.y, self.z)):
            raise ValueError("hand landmark coordinates must be finite")


@dataclass(frozen=True)
class TrackedHand:
    """A handedness label, its 21 normalized landmarks, and classifier confidence."""

    handedness: str
    landmarks: tuple[HandPoint, ...]
    confidence: float

    def __post_init__(self) -> None:
        handedness = self.handedness.lower() if isinstance(self.handedness, str) else ""
        if handedness not in ("left", "right"):
            raise ValueError("handedness must be left or right")
        object.__setattr__(self, "handedness", handedness)
        if not isinstance(self.landmarks, tuple) or len(self.landmarks) != 21 \
                or any(not isinstance(point, HandPoint) for point in self.landmarks):
            raise ValueError("tracked hand must contain exactly 21 HandPoint landmarks")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("hand confidence must be between 0 and 1")


class HandTracker:
    """Find up to two hands in RGB frames using MediaPipe's VIDEO task."""

    def __init__(self, config: RoomSenseConfig) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("MediaPipe is missing. Install requirements.txt first.") from exc

        self._mp = mp
        model_path = Path(config.hand_model_path).expanduser() if config.hand_model_path else self._cached_model_path()
        if not model_path.exists():
            self._download_model(model_path)
        vision = mp.tasks.vision
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=config.hand_detection_confidence,
            min_hand_presence_confidence=config.hand_detection_confidence,
            min_tracking_confidence=config.hand_tracking_confidence,
        )
        self._hands = vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = 0

    def process(self, rgb_frame: Any, timestamp_ms: int | None = None) -> tuple[TrackedHand, ...]:
        if timestamp_ms is not None and (
            isinstance(timestamp_ms, bool) or not isinstance(timestamp_ms, int) or timestamp_ms < 0
        ):
            raise ValueError("hand tracker timestamp_ms must be a non-negative integer")
        candidate = monotonic_ns() // 1_000_000 if timestamp_ms is None else timestamp_ms
        timestamp = max(candidate, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._hands.detect_for_video(image, timestamp)
        detected = getattr(result, "hand_landmarks", ()) or ()
        categories = getattr(result, "handedness", ()) or ()
        tracked: list[TrackedHand] = []
        for index, raw_points in enumerate(detected):
            if len(raw_points) != 21 or index >= len(categories) or not categories[index]:
                continue
            category = categories[index][0]
            label = str(getattr(category, "category_name", "")).lower()
            confidence = float(getattr(category, "score", 0.0))
            if label not in ("left", "right"):
                continue
            points = tuple(HandPoint(float(point.x), float(point.y), float(point.z)) for point in raw_points)
            tracked.append(TrackedHand(label, points, confidence))
        return tuple(tracked)

    def close(self) -> None:
        self._hands.close()

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
                "or set RoomSenseConfig.hand_model_path to a local .task model file."
            ) from exc
