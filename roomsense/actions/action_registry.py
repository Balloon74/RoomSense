"""Explicit dry-run or macOS action dispatch for gesture commands."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import math
import subprocess
import sys
from typing import Protocol


class MacAction(str, Enum):
    NEXT_TRACK = "NEXT_TRACK"
    PREVIOUS_TRACK = "PREVIOUS_TRACK"
    PLAY_PAUSE = "PLAY_PAUSE"
    VOLUME_UP = "VOLUME_UP"
    VOLUME_DOWN = "VOLUME_DOWN"


@dataclass(frozen=True)
class ActionIntent:
    action: MacAction
    gesture: str
    timestamp: float
    confidence: float

    def __post_init__(self) -> None:
        try:
            action = self.action if isinstance(self.action, MacAction) else MacAction(self.action)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unsupported macOS action: {self.action}") from exc
        object.__setattr__(self, "action", action)
        if not isinstance(self.gesture, str) or not self.gesture.strip():
            raise ValueError("gesture cannot be empty")
        if not math.isfinite(self.timestamp):
            raise ValueError("action timestamp must be finite")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("action confidence must be between 0 and 1")


@dataclass(frozen=True)
class ActionResult:
    action: MacAction
    timestamp: float
    message: str
    dry_run: bool
    succeeded: bool


class ActionController(Protocol):
    def execute(self, intent: ActionIntent) -> None: ...


class ActionRegistry:
    """Dispatch intents to dry-run messages unless real controls are opted in."""

    def __init__(
        self,
        mac_controls_enabled: bool = False,
        controller: ActionController | None = None,
        history_size: int = 8,
    ) -> None:
        if not isinstance(mac_controls_enabled, bool):
            raise ValueError("mac_controls_enabled must be a boolean")
        if isinstance(history_size, bool) or not isinstance(history_size, int) or history_size <= 0:
            raise ValueError("history_size must be a positive integer")
        self.mac_controls_enabled = mac_controls_enabled
        self._history: deque[ActionResult] = deque(maxlen=history_size)
        self.controller = controller
        if mac_controls_enabled:
            if sys.platform != "darwin":
                raise RuntimeError("Real RoomSense Mac controls are available only on macOS")
            if self.controller is None:
                from roomsense.actions.macos_controller import MacOSController

                self.controller = MacOSController()

    @property
    def history(self) -> tuple[ActionResult, ...]:
        return tuple(self._history)

    def dispatch(self, intent: ActionIntent) -> ActionResult:
        if not isinstance(intent, ActionIntent):
            raise TypeError("action registry accepts ActionIntent values only")
        if not self.mac_controls_enabled:
            result = ActionResult(
                intent.action, intent.timestamp, _action_label(intent.action), True, True,
            )
        else:
            try:
                assert self.controller is not None
                self.controller.execute(intent)
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                detail = str(exc).strip()
                result = ActionResult(
                    intent.action,
                    intent.timestamp,
                    f"ERROR: {detail or type(exc).__name__}",
                    False,
                    False,
                )
            else:
                result = ActionResult(
                    intent.action,
                    intent.timestamp,
                    f"EXECUTED: {_action_label(intent.action).removeprefix('ACTION: ')}",
                    False,
                    True,
                )
        self._history.append(result)
        return result


def _action_label(action: MacAction) -> str:
    return {
        MacAction.NEXT_TRACK: "ACTION: NEXT TRACK",
        MacAction.PREVIOUS_TRACK: "ACTION: PREVIOUS TRACK",
        MacAction.PLAY_PAUSE: "ACTION: PLAY/PAUSE",
        MacAction.VOLUME_UP: "ACTION: VOLUME UP",
        MacAction.VOLUME_DOWN: "ACTION: VOLUME DOWN",
    }[action]
