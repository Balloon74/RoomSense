"""User-controlled lifecycle for an optional session recorder."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from roomsense.interactions.events import RoomSenseEvent
from roomsense.recording.session_recorder import SessionRecorder


class RecordingController:
    def __init__(self, recorder_factory: Callable[[], SessionRecorder]) -> None:
        self._recorder_factory = recorder_factory
        self._recorder: SessionRecorder | None = None

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

    def record(self, events: Sequence[RoomSenseEvent], state: Mapping[str, object]) -> None:
        if not self.active:
            return
        if events:
            for event in events:
                self._recorder.record(event, state)
        else:
            self._recorder.record(None, state)

    def close(self) -> None:
        recorder = self._recorder
        self._recorder = None
        if recorder is not None:
            recorder.close()
