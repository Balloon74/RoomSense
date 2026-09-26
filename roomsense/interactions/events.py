"""Typed interaction events and a bounded recent-event collector."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import math
from types import MappingProxyType
from typing import Mapping


class EventType(str, Enum):
    ZONE_ENTERED = "ZONE_ENTERED"
    ZONE_LEFT = "ZONE_LEFT"
    GESTURE_DETECTED = "GESTURE_DETECTED"
    OBJECT_POINTED = "OBJECT_POINTED"
    OBJECT_POINT_HELD = "OBJECT_POINT_HELD"
    PERSON_STARTED_MOVING = "PERSON_STARTED_MOVING"
    PERSON_STOPPED_MOVING = "PERSON_STOPPED_MOVING"
    MODE_CHANGED = "MODE_CHANGED"


@dataclass(frozen=True)
class RoomSenseEvent:
    type: EventType
    timestamp: float
    metadata: Mapping[str, object]
    confidence: float | None = None
    display_time: str = field(default_factory=lambda: datetime.now().astimezone().strftime("%H:%M:%S"))

    def __post_init__(self) -> None:
        try:
            event_type = self.type if isinstance(self.type, EventType) else EventType(self.type)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unsupported RoomSense event type: {self.type}") from exc
        object.__setattr__(self, "type", event_type)
        if not math.isfinite(self.timestamp):
            raise ValueError("event timestamp must be finite")
        if self.confidence is not None and (
            not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("event confidence must be between 0 and 1")
        if not isinstance(self.metadata, Mapping) or any(not isinstance(key, str) for key in self.metadata):
            raise ValueError("event metadata must be a string-keyed mapping")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        if len(self.display_time) != 8 or self.display_time[2] != ":" or self.display_time[5] != ":":
            raise ValueError("display_time must use HH:MM:SS format")


class EventCollector:
    def __init__(self, max_events: int = 8) -> None:
        if isinstance(max_events, bool) or not isinstance(max_events, int) or max_events <= 0:
            raise ValueError("max_events must be a positive integer")
        self._events: deque[RoomSenseEvent] = deque(maxlen=max_events)
        self._last_timestamp: float | None = None

    @property
    def recent(self) -> tuple[RoomSenseEvent, ...]:
        return tuple(self._events)

    def publish(self, event: RoomSenseEvent) -> None:
        if not isinstance(event, RoomSenseEvent):
            raise TypeError("event collector accepts RoomSenseEvent values only")
        if self._last_timestamp is not None and event.timestamp < self._last_timestamp:
            raise ValueError("event timestamps must be non-decreasing")
        self._last_timestamp = event.timestamp
        self._events.append(event)
