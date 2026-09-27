"""In-memory anonymous person track continuity, independent of pose inference."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from math import exp, isfinite, log, sqrt
from typing import Mapping


@dataclass(frozen=True)
class TrackObservation:
    """A compact, normalized observation for one live track."""

    timestamp: float
    x: float
    y: float
    z: float
    shoulder_width: float | None = None
    torso_ratio: float | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError("timestamp must be finite and non-negative")
        for name in ("x", "y", "z"):
            value = getattr(self, name)
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.x <= 1.0 or not 0.0 <= self.y <= 1.0:
            raise ValueError("x and y must be normalized to [0, 1]")
        for name in ("shoulder_width", "torso_ratio"):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be positive and finite when provided")


class ReidentificationState(str, Enum):
    NEW = "NEW"
    TRACKED = "TRACKED"
    LOST = "LOST"
    REACQUIRED = "REACQUIRED"


@dataclass(frozen=True)
class CandidateScore:
    person_id: str
    factors: Mapping[str, float]
    score: float
    reason: str


@dataclass(frozen=True)
class ReidentificationUpdate:
    person_id: str | None
    state: ReidentificationState
    confidence: float | None
    candidates: tuple[CandidateScore, ...]
    reason: str


def observation_from_position(timestamp: float, position: object,
                              landmarks: Mapping[str, object]) -> TrackObservation:
    """Build a safe continuity observation from a normalized position and pose geometry."""
    shoulder_width = None
    torso_ratio = None
    left_shoulder = landmarks.get("left_shoulder")
    right_shoulder = landmarks.get("right_shoulder")
    left_hip = landmarks.get("left_hip")
    right_hip = landmarks.get("right_hip")
    if left_shoulder is not None and right_shoulder is not None:
        shoulder_dx = float(getattr(right_shoulder, "x")) - float(getattr(left_shoulder, "x"))
        shoulder_dy = float(getattr(right_shoulder, "y")) - float(getattr(left_shoulder, "y"))
        measured_width = sqrt(shoulder_dx ** 2 + shoulder_dy ** 2)
        if isfinite(measured_width) and measured_width > 1e-4:
            shoulder_width = measured_width
            if left_hip is not None and right_hip is not None:
                shoulder_center = (
                    (float(getattr(left_shoulder, "x")) + float(getattr(right_shoulder, "x"))) / 2.0,
                    (float(getattr(left_shoulder, "y")) + float(getattr(right_shoulder, "y"))) / 2.0,
                )
                hip_center = (
                    (float(getattr(left_hip, "x")) + float(getattr(right_hip, "x"))) / 2.0,
                    (float(getattr(left_hip, "y")) + float(getattr(right_hip, "y"))) / 2.0,
                )
                torso_length = sqrt(
                    (hip_center[0] - shoulder_center[0]) ** 2
                    + (hip_center[1] - shoulder_center[1]) ** 2
                )
                ratio = torso_length / measured_width
                if isfinite(ratio) and ratio > 0.0:
                    torso_ratio = ratio
    x = (float(getattr(position, "x")) + 1.0) / 2.0
    y = (1.0 - float(getattr(position, "y"))) / 2.0
    return TrackObservation(
        timestamp=timestamp,
        x=x,
        y=y,
        z=float(getattr(position, "z")),
        shoulder_width=shoulder_width,
        torso_ratio=torso_ratio,
    )


@dataclass(frozen=True)
class _LostTrack:
    person_id: str
    observation: TrackObservation
    velocity: tuple[float, float, float]
    lost_at: float


class PersonReidentifier:
    """Assign per-process anonymous IDs and cautiously reconnect lost tracks."""

    def __init__(self, timeout_seconds: float = 5.0, confidence_threshold: float = 0.72,
                 ambiguity_margin: float = 0.12) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self.confidence_threshold = float(confidence_threshold)
        self.ambiguity_margin = float(ambiguity_margin)
        if not isfinite(self.timeout_seconds) or self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive and finite")
        if not isfinite(self.confidence_threshold) or not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not isfinite(self.ambiguity_margin) or not 0.0 <= self.ambiguity_margin <= 1.0:
            raise ValueError("ambiguity_margin must be between 0 and 1")
        self._next_id = 1
        self._active_id: str | None = None
        self._active_history: deque[TrackObservation] = deque(maxlen=2)
        self._lost: dict[str, _LostTrack] = {}
        self._last_timestamp: float | None = None
        self._last_decision: tuple[float | None, tuple[CandidateScore, ...], str] = (
            None, (), "new_identity"
        )

    def update(self, observation: TrackObservation | None,
               timestamp: float | None = None) -> ReidentificationUpdate:
        current_time = observation.timestamp if observation is not None else timestamp
        if current_time is None or not isfinite(current_time) or current_time < 0.0:
            raise ValueError("a finite non-negative timestamp is required")
        if self._last_timestamp is not None and current_time < self._last_timestamp:
            raise ValueError("timestamps must be monotonic")
        if observation is not None and abs(observation.timestamp - current_time) > 1e-9:
            raise ValueError("timestamp must match observation.timestamp")
        self._last_timestamp = current_time

        if observation is None:
            return self._mark_lost(current_time)
        if self._active_id is not None:
            self._discard_expired(current_time)
            self._active_history.append(observation)
            confidence, candidates, reason = self._last_decision
            return ReidentificationUpdate(
                self._active_id, ReidentificationState.TRACKED, confidence, candidates, reason
            )
        return self._start_track(observation, current_time)

    def _mark_lost(self, timestamp: float) -> ReidentificationUpdate:
        expired = self._discard_expired(timestamp)
        if self._active_id is not None:
            person_id = self._active_id
            history = tuple(self._active_history)
            velocity = self._velocity(history)
            self._lost[person_id] = _LostTrack(person_id, history[-1], velocity, timestamp)
            self._active_id = None
            self._active_history.clear()
            return ReidentificationUpdate(person_id, ReidentificationState.LOST, None, (), "track_lost")
        latest = next(reversed(self._lost.values()), None)
        return ReidentificationUpdate(
            latest.person_id if latest else None, ReidentificationState.LOST, None, (),
            "awaiting_return" if latest else ("expired" if expired else "awaiting_return")
        )

    def _start_track(self, observation: TrackObservation, timestamp: float) -> ReidentificationUpdate:
        expired = self._discard_expired(timestamp)
        scored = sorted(
            (self._score(candidate, observation, timestamp) for candidate in self._lost.values()),
            key=lambda candidate: candidate.score,
            reverse=True,
        )
        if scored:
            selected = scored[0]
            second_score = scored[1].score if len(scored) > 1 else None
            if selected.score < self.confidence_threshold:
                reason = "below_threshold"
                candidate_reason = "below_threshold"
            elif second_score is not None and selected.score - second_score < self.ambiguity_margin:
                reason = "ambiguous_candidates"
                candidate_reason = "ambiguous"
            else:
                self._lost.pop(selected.person_id)
                self._activate(selected.person_id, observation)
                marked = tuple(
                    CandidateScore(candidate.person_id, candidate.factors, candidate.score,
                                   "selected" if candidate.person_id == selected.person_id else "not_selected")
                    for candidate in scored
                )
                result = ReidentificationUpdate(
                    selected.person_id, ReidentificationState.REACQUIRED, selected.score, marked, "reacquired"
                )
                self._last_decision = (result.confidence, result.candidates, result.reason)
                return result
            marked = tuple(
                CandidateScore(candidate.person_id, candidate.factors, candidate.score,
                               candidate_reason if candidate.score >= selected.score - self.ambiguity_margin
                               else "lower_score")
                for candidate in scored
            )
            person_id = self._allocate_id()
            self._activate(person_id, observation)
            result = ReidentificationUpdate(
                person_id, ReidentificationState.NEW, selected.score, marked, reason
            )
            self._last_decision = (result.confidence, result.candidates, result.reason)
            return result
        person_id = self._allocate_id()
        self._activate(person_id, observation)
        reason = "expired_candidate" if expired else ("new_identity" if person_id == "PERSON_001" else "no_candidate")
        result = ReidentificationUpdate(person_id, ReidentificationState.NEW, None, (), reason)
        self._last_decision = (result.confidence, result.candidates, result.reason)
        return result

    def _discard_expired(self, timestamp: float) -> bool:
        expired_ids = [
            person_id for person_id, candidate in self._lost.items()
            if timestamp - candidate.lost_at >= self.timeout_seconds
        ]
        for person_id in expired_ids:
            del self._lost[person_id]
        return bool(expired_ids)

    def _score(self, candidate: _LostTrack, observation: TrackObservation,
               timestamp: float) -> CandidateScore:
        elapsed = max(0.0, timestamp - candidate.lost_at)
        vx, vy, vz = candidate.velocity
        prediction_time = min(elapsed, 1.0)
        predicted_x = min(1.0, max(0.0, candidate.observation.x + vx * prediction_time))
        predicted_y = min(1.0, max(0.0, candidate.observation.y + vy * prediction_time))
        predicted_z = candidate.observation.z + vz * prediction_time
        distance = sqrt(
            (observation.x - predicted_x) ** 2
            + (observation.y - predicted_y) ** 2
            + (0.5 * (observation.z - predicted_z)) ** 2
        )
        trajectory = exp(-distance / 0.35)
        direction = self._reentry_score(candidate, observation)
        time_score = exp(-elapsed / self.timeout_seconds)
        geometry_scores = [
            exp(-abs(log(returned / stored)) / 0.35)
            for returned, stored in (
                (observation.shoulder_width, candidate.observation.shoulder_width),
                (observation.torso_ratio, candidate.observation.torso_ratio),
            )
            if returned is not None and stored is not None
        ]
        geometry = sum(geometry_scores) / len(geometry_scores) if geometry_scores else 0.0
        factors = {
            "trajectory": trajectory,
            "reentry_direction": direction,
            "elapsed_time": time_score,
            "torso_geometry": geometry,
        }
        score = (0.45 * trajectory + 0.25 * direction + 0.15 * time_score + 0.15 * geometry)
        return CandidateScore(candidate.person_id, factors, score, "scored")

    @staticmethod
    def _reentry_score(candidate: _LostTrack, observation: TrackObservation) -> float:
        vx, vy, _ = candidate.velocity
        projected_x = candidate.observation.x + vx * 0.75
        projected_y = candidate.observation.y + vy * 0.75
        boundaries: list[float] = []
        if projected_x < 0.0:
            boundaries.append(observation.x)
        elif projected_x > 1.0:
            boundaries.append(1.0 - observation.x)
        if projected_y < 0.0:
            boundaries.append(observation.y)
        elif projected_y > 1.0:
            boundaries.append(1.0 - observation.y)
        if not boundaries:
            return 0.5
        distance_to_boundary = min(boundaries)
        return max(0.0, 1.0 - distance_to_boundary / 0.20)

    @staticmethod
    def _velocity(history: tuple[TrackObservation, ...]) -> tuple[float, float, float]:
        if len(history) < 2:
            return 0.0, 0.0, 0.0
        previous, latest = history
        elapsed = latest.timestamp - previous.timestamp
        if elapsed <= 0.0:
            return 0.0, 0.0, 0.0
        return (
            (latest.x - previous.x) / elapsed,
            (latest.y - previous.y) / elapsed,
            (latest.z - previous.z) / elapsed,
        )

    def _activate(self, person_id: str, observation: TrackObservation) -> None:
        self._active_id = person_id
        self._active_history.clear()
        self._active_history.append(observation)

    def _allocate_id(self) -> str:
        person_id = f"PERSON_{self._next_id:03d}"
        self._next_id += 1
        return person_id
