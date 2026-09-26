"""Safe, in-process action registry for RoomSense interaction events."""

from __future__ import annotations

import logging
from typing import Callable, Mapping

from roomsense.interactions.events import EventType, RoomSenseEvent


ActionHandler = Callable[[RoomSenseEvent, Mapping[str, object]], str | None]


class ActionRegistry:
    """Dispatch events to explicitly registered local handlers."""

    def __init__(self) -> None:
        self._handlers: dict[str, ActionHandler] = {}

    @property
    def action_ids(self) -> tuple[str, ...]:
        return tuple(self._handlers)

    def register(self, action_id: str, handler: ActionHandler) -> None:
        if not isinstance(action_id, str) or not action_id.strip():
            raise ValueError("action_id cannot be empty")
        if not callable(handler):
            raise TypeError("action handler must be callable")
        if action_id in self._handlers:
            raise ValueError(f"action is already registered: {action_id}")
        self._handlers[action_id] = handler

    def dispatch(self, event: RoomSenseEvent, context: Mapping[str, object]) -> tuple[str, ...]:
        messages: list[str] = []
        for handler in self._handlers.values():
            message = handler(event, context)
            if message is not None:
                if not isinstance(message, str):
                    raise TypeError("action handlers must return a string or None")
                if message:
                    messages.append(message)
        return tuple(messages)

    def dispatch_action(
        self, action_id: str, event: RoomSenseEvent, context: Mapping[str, object]
    ) -> tuple[str, ...]:
        """Run one named action, reporting unknown configuration without acting."""
        handler = self._handlers.get(action_id)
        if handler is None:
            logging.getLogger(__name__).error("Unknown RoomSense action %r; no action was run", action_id)
            return ()
        message = handler(event, context)
        if message is None or message == "":
            return ()
        if not isinstance(message, str):
            raise TypeError("action handlers must return a string or None")
        return (message,)


def create_demo_action_registry() -> ActionRegistry:
    """Create only harmless UI-message actions for the initial RoomSense build."""

    registry = ActionRegistry()

    def zone_mode(event: RoomSenseEvent, _context: Mapping[str, object]) -> str | None:
        if event.type is EventType.ZONE_ENTERED:
            zone = event.metadata.get("zone")
            return f"{zone} MODE" if isinstance(zone, str) else None
        return None

    def pointing_message(event: RoomSenseEvent, _context: Mapping[str, object]) -> str | None:
        if event.type is EventType.OBJECT_POINTED:
            name = event.metadata.get("name")
            return f"POINTING {name}" if isinstance(name, str) else None
        return None

    def command_mode(event: RoomSenseEvent, _context: Mapping[str, object]) -> str | None:
        if event.type is not EventType.MODE_CHANGED:
            return None
        return "COMMAND MODE" if event.metadata.get("to") == "COMMAND" else "COMMAND MODE EXIT"

    def monitor_swipe(event: RoomSenseEvent, context: Mapping[str, object]) -> str | None:
        if event.type is not EventType.GESTURE_DETECTED:
            return None
        if context.get("mode") != "COMMAND" or context.get("target_id") != "monitor":
            return None
        gesture = event.metadata.get("gesture")
        if gesture == "SWIPE_RIGHT":
            return "NEXT"
        if gesture == "SWIPE_LEFT":
            return "PREVIOUS"
        return None

    registry.register("demo.zone_mode", zone_mode)
    registry.register("demo.pointing", pointing_message)
    registry.register("demo.command_mode", command_mode)
    registry.register("demo.monitor_swipe", monitor_swipe)
    return registry
