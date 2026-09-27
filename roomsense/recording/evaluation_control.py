"""Explicit user-controlled lifecycle for landmark evaluation capture."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from roomsense.evaluation import EvaluationRecorder
from roomsense.tracking.pose_tracker import Landmark


class EvaluationCaptureController:
    def __init__(self, recorder_factory: Callable[[], EvaluationRecorder]) -> None:
        self._recorder_factory = recorder_factory
        self._recorder: EvaluationRecorder | None = None

    @property
    def active(self) -> bool:
        return self._recorder is not None

    def toggle(self) -> bool:
        if self.active:
            self.close()
            return False
        recorder = self._recorder_factory()
        recorder.start()
        self._recorder = recorder
        return True

    def record(self, timestamp: float, landmarks: Mapping[str, Landmark] | None) -> None:
        if self._recorder is not None:
            self._recorder.record(timestamp, landmarks)

    def close(self) -> None:
        recorder = self._recorder
        self._recorder = None
        if recorder is not None:
            recorder.close()
