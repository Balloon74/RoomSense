"""User-tunable settings for RoomSense."""

from dataclasses import dataclass
import math


DEFAULT_ZONE_POLYGONS = (
    ("DESK", ((0.03, 0.03), (0.36, 0.03), (0.36, 0.36), (0.03, 0.36))),
    ("BED", ((0.64, 0.03), (0.97, 0.03), (0.97, 0.36), (0.64, 0.36))),
    ("DOOR", ((0.78, 0.68), (0.97, 0.68), (0.97, 0.97), (0.78, 0.97))),
    ("CENTER", ((0.37, 0.37), (0.63, 0.37), (0.63, 0.63), (0.37, 0.63))),
)


@dataclass(frozen=True)
class RoomObjectConfig:
    """Plain-data room object definition with separate image/map coordinates."""

    id: str
    name: str
    image_region: tuple[tuple[float, float], ...]
    enabled: bool = False
    room_position: tuple[float, float] | None = None
    interaction_radius: float | None = None


DEFAULT_ROOM_OBJECTS = (
    RoomObjectConfig("monitor", "MONITOR", ((0.68, 0.18), (0.94, 0.18), (0.94, 0.48), (0.68, 0.48)),
                     room_position=(0.82, 0.18), interaction_radius=0.025),
    RoomObjectConfig("desk", "DESK", ((0.12, 0.48), (0.46, 0.48), (0.46, 0.82), (0.12, 0.82)),
                     room_position=(0.22, 0.22), interaction_radius=0.025),
    RoomObjectConfig("bed", "BED", ((0.54, 0.50), (0.90, 0.50), (0.90, 0.92), (0.54, 0.92)),
                     room_position=(0.78, 0.80), interaction_radius=0.025),
    RoomObjectConfig("door", "DOOR", ((0.02, 0.16), (0.20, 0.16), (0.20, 0.70), (0.02, 0.70)),
                     room_position=(0.92, 0.82), interaction_radius=0.025),
    RoomObjectConfig("lamp", "LAMP", ((0.36, 0.12), (0.50, 0.12), (0.50, 0.35), (0.36, 0.35)),
                     room_position=(0.55, 0.20), interaction_radius=0.02),
    RoomObjectConfig("tv", "TV", ((0.72, 0.52), (0.98, 0.52), (0.98, 0.78), (0.72, 0.78)),
                     room_position=(0.50, 0.12), interaction_radius=0.025),
)


@dataclass(frozen=True)
class RoomSenseConfig:
    # On macOS, None selects the built-in camera by device type. Set an index
    # explicitly only when you want a different camera.
    camera_index: int | None = None
    camera_width: int = 1280
    camera_height: int = 720
    max_fps: int = 30
    detection_confidence: float = 0.55
    tracking_confidence: float = 0.55
    landmark_visibility: float = 0.35
    smoothing_alpha: float = 0.35
    reference_shoulder_width: float = 0.20
    depth_sensitivity: float = 0.8
    movement_window_seconds: float = 0.7
    movement_horizontal_threshold: float = 0.08
    movement_depth_threshold: float = 0.08
    tracking_lost_seconds: float = 1.0
    pose_model_path: str | None = None
    raised_hand_margin: float = 0.04
    sitting_leg_ratio: float = 0.18
    standing_leg_ratio: float = 0.42
    calibration_path: str = "calibration.json"
    floor_landmark_visibility: float = 0.5
    history_retention_seconds: float = 4.0
    zone_polygons: tuple[tuple[str, tuple[tuple[float, float], ...]], ...] = DEFAULT_ZONE_POLYGONS
    room_objects: tuple[RoomObjectConfig, ...] = DEFAULT_ROOM_OBJECTS
    pointing_min_visibility: float = 0.55
    pointing_min_extension: float = 0.78
    pointing_smoothing_alpha: float = 0.4
    target_stability_seconds: float = 0.35
    target_hold_seconds: float = 0.8
    gesture_swipe_distance: float = 0.16
    gesture_swipe_window_seconds: float = 0.6
    gesture_both_hands_hold_seconds: float = 0.55
    gesture_hold_point_seconds: float = 0.75
    gesture_cooldown_seconds: float = 0.8
    command_mode_timeout_seconds: float = 12.0
    event_feed_size: int = 8
    session_recording_path: str = "roomsense-session.jsonl"
    record_position_interval_seconds: float = 0.5

    def __post_init__(self) -> None:
        for name in ("pointing_min_visibility", "pointing_min_extension"):
            value = _finite(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        smoothing = _finite(self.pointing_smoothing_alpha, "pointing_smoothing_alpha")
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("pointing_smoothing_alpha must be in (0, 1]")
        for name in ("target_stability_seconds", "target_hold_seconds", "gesture_cooldown_seconds"):
            if _finite(getattr(self, name), name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        for name in (
            "gesture_swipe_window_seconds", "gesture_both_hands_hold_seconds",
            "gesture_hold_point_seconds", "command_mode_timeout_seconds",
            "record_position_interval_seconds",
        ):
            if _finite(getattr(self, name), name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        swipe_distance = _finite(self.gesture_swipe_distance, "gesture_swipe_distance")
        if not 0.0 < swipe_distance <= 1.0:
            raise ValueError("gesture_swipe_distance must be in (0, 1]")
        if isinstance(self.event_feed_size, bool) or not isinstance(self.event_feed_size, int) \
                or self.event_feed_size <= 0:
            raise ValueError("event_feed_size must be a positive integer")
        if not isinstance(self.session_recording_path, str) or not self.session_recording_path.strip():
            raise ValueError("session_recording_path cannot be empty")
        if not isinstance(self.room_objects, (tuple, list)) or any(
            not isinstance(item, RoomObjectConfig) for item in self.room_objects
        ):
            raise ValueError("room_objects must contain RoomObjectConfig entries")
        from roomsense.spatial.room_objects import RoomObject

        for item in self.room_objects:
            RoomObject(item.id, item.name, item.image_region, item.enabled,
                       item.room_position, item.interaction_radius)
        ids = [item.id for item in self.room_objects]
        if len(ids) != len(set(ids)):
            raise ValueError("room object ids must be unique")


def _finite(value: float, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number
