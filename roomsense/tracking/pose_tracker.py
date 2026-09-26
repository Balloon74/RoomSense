"""MediaPipe pose extraction, kept independent from drawing and app state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import monotonic_ns
from typing import Any
from urllib.request import urlopen

from roomsense.config import RoomSenseConfig
from roomsense.utils.smoothing import VectorSmoother


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


LANDMARK_NAMES: dict[str, int] = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

SKELETON_CONNECTIONS: tuple[tuple[str, str], ...] = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
)

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)


class PoseTracker:
    """Find one full-body pose and return normalized, lightly smoothed landmarks."""

    def __init__(self, config: RoomSenseConfig) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("MediaPipe is missing. Install requirements.txt first.") from exc

        self.config = config
        self._mp = mp
        model_path = Path(config.pose_model_path).expanduser() if config.pose_model_path else self._cached_model_path()
        if not model_path.exists():
            self._download_model(model_path)
        vision = mp.tasks.vision
        options = vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=config.detection_confidence,
            min_pose_presence_confidence=config.detection_confidence,
            min_tracking_confidence=config.tracking_confidence,
            output_segmentation_masks=False,
        )
        self._pose = vision.PoseLandmarker.create_from_options(options)
        self._smoothers = {name: VectorSmoother(config.smoothing_alpha) for name in LANDMARK_NAMES}
        self._last_timestamp_ms = 0

    def process(self, rgb_frame: Any, timestamp_ms: int | None = None) -> dict[str, Landmark] | None:
        timestamp = max(timestamp_ms or monotonic_ns() // 1_000_000, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._pose.detect_for_video(mp_image, timestamp)
        if not result.pose_landmarks:
            return None
        points = result.pose_landmarks[0]
        tracked: dict[str, Landmark] = {}
        for name, index in LANDMARK_NAMES.items():
            point = points[index]
            visibility = float(getattr(point, "visibility", 1.0))
            if visibility < self.config.landmark_visibility:
                continue
            x, y, z = self._smoothers[name].update((float(point.x), float(point.y), float(point.z)))
            tracked[name] = Landmark(x, y, z, visibility)
        return tracked or None

    def close(self) -> None:
        self._pose.close()

    def reset_smoothing(self) -> None:
        for smoother in self._smoothers.values():
            smoother.reset()

    @staticmethod
    def _cached_model_path() -> Path:
        return Path.home() / ".cache" / "roomsense" / "pose_landmarker_lite.task"

    @staticmethod
    def _download_model(destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".download")
        try:
            with urlopen(POSE_MODEL_URL, timeout=45) as response, temporary.open("wb") as model_file:
                while chunk := response.read(1024 * 1024):
                    model_file.write(chunk)
            temporary.replace(destination)
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                "Could not download the MediaPipe pose model. Check your internet connection, "
                "or set RoomSenseConfig.pose_model_path to a local .task model file."
            ) from exc
