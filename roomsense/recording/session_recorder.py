"""Versioned JSONL recording for structured RoomSense interaction state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
import os
from pathlib import Path
import time
from typing import Callable, Iterator, Mapping

from roomsense.interactions.events import RoomSenseEvent


SESSION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SessionRecord:
    schema_version: int
    timestamp: float
    kind: str
    payload: Mapping[str, object]


class SessionRecorder:
    """Write events and bounded-rate position samples after explicit start()."""

    def __init__(
        self,
        path: str | Path,
        position_sample_interval: float = 0.5,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(position_sample_interval) or position_sample_interval <= 0.0:
            raise ValueError("position_sample_interval must be positive and finite")
        self.path = Path(path).expanduser()
        self.position_sample_interval = position_sample_interval
        self._clock = clock
        self._stream = None
        self._last_timestamp: float | None = None
        self._last_state: object | None = None
        self._last_position_sample: float | None = None

    @property
    def active(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        if self.active:
            raise RuntimeError("session recorder is already active")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("w", encoding="utf-8")
        self._last_timestamp = None
        self._last_state = None
        self._last_position_sample = None
        self._write(self._now(), "session_start", {})

    def record(self, event: RoomSenseEvent | None, state: Mapping[str, object]) -> None:
        if not self.active:
            raise RuntimeError("session recorder is not active")
        state_json = _jsonable(dict(state))
        timestamp_value = event.timestamp if event is not None else state.get("timestamp")
        if not isinstance(timestamp_value, (int, float)) or isinstance(timestamp_value, bool) \
                or not math.isfinite(timestamp_value):
            raise ValueError("record state must provide a finite timestamp")
        timestamp = float(timestamp_value)
        fingerprint = {key: value for key, value in state_json.items() if key not in ("timestamp", "room_position")}
        state_changed = fingerprint != self._last_state
        if event is not None:
            event_payload = {
                "type": event.type.value,
                "timestamp": event.timestamp,
                "display_time": event.display_time,
                "metadata": _jsonable(dict(event.metadata)),
                "confidence": event.confidence,
            }
            self._write(timestamp, "event", {"event": event_payload, "state": state_json})
        elif state_changed:
            self._write(timestamp, "state", state_json)
        self._last_state = fingerprint

        if timestamp - (self._last_position_sample if self._last_position_sample is not None else float("-inf")) \
                >= self.position_sample_interval:
            position_payload = {
                "room_position": state_json.get("room_position"),
                "movement": state_json.get("movement"),
                "mode": state_json.get("mode"),
            }
            self._write(timestamp, "position", position_payload)
            self._last_position_sample = timestamp

    def close(self) -> None:
        if not self.active:
            return
        stream = self._stream
        try:
            self._write(self._now(), "session_end", {})
            stream.flush()
            os.fsync(stream.fileno())
        finally:
            stream.close()
            self._stream = None

    def _write(self, timestamp: float, kind: str, payload: Mapping[str, object]) -> None:
        if not math.isfinite(timestamp):
            raise ValueError("record timestamp must be finite")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("session record timestamps must be non-decreasing")
        if not self.active:
            raise RuntimeError("session recorder is not active")
        record = {
            "schema_version": SESSION_SCHEMA_VERSION,
            "timestamp": float(timestamp),
            "kind": kind,
            "payload": _jsonable(dict(payload)),
        }
        self._stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
        self._last_timestamp = float(timestamp)

    def _now(self) -> float:
        value = float(self._clock())
        if not math.isfinite(value):
            raise ValueError("recorder clock must be finite")
        return value


def read_session(path: str | Path) -> Iterator[SessionRecord]:
    """Read strict schema-v1 JSONL, raising a line-numbered error on bad data."""

    previous_timestamp: float | None = None
    with Path(path).expanduser().open("r", encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            try:
                parsed = json.loads(raw_line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                if not isinstance(parsed, dict):
                    raise ValueError("record must be an object")
                version = parsed.get("schema_version")
                timestamp = parsed.get("timestamp")
                kind = parsed.get("kind")
                payload = parsed.get("payload")
                if version != SESSION_SCHEMA_VERSION:
                    raise ValueError(f"unsupported schema version: {version}")
                if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool) or not math.isfinite(timestamp):
                    raise ValueError("timestamp must be finite")
                if not isinstance(kind, str) or not kind:
                    raise ValueError("kind must be a non-empty string")
                if not isinstance(payload, dict):
                    raise ValueError("payload must be an object")
                if previous_timestamp is not None and timestamp < previous_timestamp:
                    raise ValueError("timestamps must be non-decreasing")
                record = SessionRecord(version, float(timestamp), kind, payload)
            except (json.JSONDecodeError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"invalid session record at line {line_number}: {exc}") from exc
            previous_timestamp = record.timestamp
            yield record


def _jsonable(value: object) -> object:
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, RoomSenseEvent):
        return {
            "type": value.type.value,
            "timestamp": value.timestamp,
            "display_time": value.display_time,
            "metadata": _jsonable(dict(value.metadata)),
            "confidence": value.confidence,
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON record mappings must use string keys")
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("record values must be finite")
        return value
    raise TypeError(f"unsupported JSON record value: {type(value).__name__}")
