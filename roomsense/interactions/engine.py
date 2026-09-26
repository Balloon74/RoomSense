"""Compose camera-free gesture, target, mode, event, and demo-action state."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from roomsense.config import RoomObjectConfig, RoomSenseConfig
from roomsense.gestures.temporal_gestures import GestureTransition, TemporalGestureDetector
from roomsense.interactions.actions import ActionRegistry, create_demo_action_registry
from roomsense.interactions.events import EventCollector, EventType, RoomSenseEvent
from roomsense.interactions.modes import InteractionMode, InteractionModeController
from roomsense.spatial.pointing import ArmPointing, PointingEstimator
from roomsense.spatial.room_objects import RoomObject, RoomObjectRegistry
from roomsense.spatial.target_selection import TargetMatch, TargetSelector, TargetUpdate
from roomsense.tracking.pose_tracker import Landmark

Point = tuple[float, float]


@dataclass(frozen=True)
class InteractionObservation:
    timestamp: float
    landmarks: Mapping[str, Landmark] | None
    room_position: Point | None
    zone_entered: tuple[str, ...]
    zone_left: tuple[str, ...]
    movement_state: str


@dataclass(frozen=True)
class InteractionUpdate:
    pointing: tuple[ArmPointing, ...]
    target: TargetUpdate
    mode: InteractionMode
    gestures: tuple[GestureTransition, ...]
    events: tuple[RoomSenseEvent, ...]
    recent_events: tuple[RoomSenseEvent, ...]
    demo_status: str | None
    debug_state: Mapping[str, object]


class InteractionEngine:
    def __init__(
        self,
        config: RoomSenseConfig,
        room_objects: RoomObjectRegistry | tuple[RoomObject, ...] | tuple[RoomObjectConfig, ...],
        action_registry: ActionRegistry | None = None,
    ) -> None:
        if isinstance(room_objects, RoomObjectRegistry):
            registry = room_objects
        else:
            configured = tuple(
                item if isinstance(item, RoomObject) else RoomObject(
                    item.id, item.name, item.image_region, item.enabled, item.room_position,
                    item.interaction_radius,
                )
                for item in room_objects
            )
            registry = RoomObjectRegistry(configured)
        self.config = config
        self.room_objects = registry
        self.pointing_estimator = PointingEstimator(
            config.pointing_min_visibility, config.pointing_min_extension, config.pointing_smoothing_alpha,
        )
        self.target_selector = TargetSelector(
            registry, config.target_stability_seconds, config.target_hold_seconds,
        )
        self.gesture_detector = TemporalGestureDetector(config)
        self.mode_controller = InteractionModeController(config.command_mode_timeout_seconds)
        self.event_collector = EventCollector(config.event_feed_size)
        self.action_registry = action_registry or create_demo_action_registry()
        self._was_moving = False
        self._demo_status: str | None = None

    def update(self, observation: InteractionObservation) -> InteractionUpdate:
        timestamp = observation.timestamp
        if not math.isfinite(timestamp):
            raise ValueError("interaction timestamp must be finite")
        landmarks = observation.landmarks
        if landmarks:
            pointing = self.pointing_estimator.update(landmarks)
            gestures = self.gesture_detector.update(landmarks, pointing, timestamp)
        else:
            self.pointing_estimator.reset()
            pointing = ()
            gestures = self.gesture_detector.update(None, (), timestamp)
        target = self.target_selector.update(pointing, timestamp)

        current_moving = _is_moving(observation.movement_state)
        movement_event: tuple[EventType, str] | None = None
        if current_moving and not self._was_moving:
            movement_event = (EventType.PERSON_STARTED_MOVING, observation.movement_state)
        elif self._was_moving and not current_moving:
            movement_event = (EventType.PERSON_STOPPED_MOVING, observation.movement_state)
        self._was_moving = current_moving

        active = bool(
            pointing or target.confirmed or gestures or observation.zone_entered or observation.zone_left
            or movement_event
        )
        mode_transition = self.mode_controller.update(gestures, timestamp, activity=active)
        context = {
            "mode": self.mode_controller.mode.value,
            "target_id": target.confirmed.object.id if target.confirmed else None,
            "target_name": target.confirmed.object.name if target.confirmed else None,
            "room_position": observation.room_position,
        }
        produced: list[RoomSenseEvent] = []
        for name in observation.zone_entered:
            produced.append(RoomSenseEvent(EventType.ZONE_ENTERED, timestamp, {"zone": name}))
        for name in observation.zone_left:
            produced.append(RoomSenseEvent(EventType.ZONE_LEFT, timestamp, {"zone": name}))
        for gesture in gestures:
            metadata = {"gesture": gesture.name, **dict(gesture.metadata)}
            produced.append(RoomSenseEvent(EventType.GESTURE_DETECTED, timestamp, metadata, gesture.confidence))
        if target.confirmed_now and target.confirmed:
            produced.append(_target_event(EventType.OBJECT_POINTED, target.confirmed, timestamp))
        if target.held_now and target.held:
            produced.append(_target_event(EventType.OBJECT_POINT_HELD, target.held, timestamp))
        if movement_event:
            event_type, movement = movement_event
            produced.append(RoomSenseEvent(event_type, timestamp, {"movement": movement}))
        if mode_transition:
            produced.append(RoomSenseEvent(EventType.MODE_CHANGED, timestamp, {
                "from": mode_transition.previous.value,
                "to": mode_transition.current.value,
                "reason": mode_transition.reason,
            }))

        messages: list[str] = []
        for event in produced:
            self.event_collector.publish(event)
            messages.extend(self.action_registry.dispatch(event, context))
        if messages:
            self._demo_status = messages[-1]

        candidate = target.candidate
        confirmed = target.confirmed
        debug = {
            "raw_arm_vectors": _raw_arm_vectors(landmarks),
            "pointing": tuple({
                "side": arm.side, "origin": arm.origin, "direction": arm.direction,
                "confidence": arm.confidence, "extension": arm.extension,
            } for arm in pointing),
            "pointing_confidence": max((arm.confidence for arm in pointing), default=0.0),
            "candidate_target": candidate.object.id if candidate else None,
            "confirmed_target": confirmed.object.id if confirmed else None,
            "gesture_history": self.gesture_detector.history,
            "active_gestures": self.gesture_detector.active_gestures,
            "cooldowns": self.gesture_detector.cooldowns,
            "mode": self.mode_controller.mode.value,
        }
        return InteractionUpdate(
            pointing=pointing,
            target=target,
            mode=self.mode_controller.mode,
            gestures=gestures,
            events=tuple(produced),
            recent_events=self.event_collector.recent,
            demo_status=self._demo_status,
            debug_state=debug,
        )


def _target_event(event_type: EventType, match: TargetMatch, timestamp: float) -> RoomSenseEvent:
    return RoomSenseEvent(event_type, timestamp, {
        "object_id": match.object.id,
        "name": match.object.name,
        "side": match.side,
        "ray_distance": match.ray_distance,
    }, match.confidence)


def _is_moving(state: str) -> bool:
    value = getattr(state, "value", state)
    return isinstance(value, str) and value.upper().startswith("MOVING")


def _raw_arm_vectors(landmarks: Mapping[str, Landmark] | None) -> Mapping[str, tuple[Point, Point]]:
    vectors = {}
    if not landmarks:
        return vectors
    for side in ("left", "right"):
        shoulder = landmarks.get(f"{side}_shoulder")
        elbow = landmarks.get(f"{side}_elbow")
        wrist = landmarks.get(f"{side}_wrist")
        if shoulder and elbow and wrist:
            vectors[side] = (
                (elbow.x - shoulder.x, elbow.y - shoulder.y),
                (wrist.x - elbow.x, wrist.y - elbow.y),
            )
    return vectors
