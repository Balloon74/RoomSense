"""Native macOS media-key and output-volume control via osascript."""

from __future__ import annotations

import subprocess

from roomsense.actions.action_registry import ActionIntent, MacAction


_MEDIA_KEY_CODES = {
    MacAction.PREVIOUS_TRACK: 98,  # F7
    MacAction.PLAY_PAUSE: 100,  # F8
    MacAction.NEXT_TRACK: 101,  # F9
}


class MacOSController:
    """Send standard macOS media keys and discrete output-volume steps."""

    def __init__(self, volume_step_percent: int = 5) -> None:
        if isinstance(volume_step_percent, bool) or not isinstance(volume_step_percent, int) \
                or not 1 <= volume_step_percent <= 100:
            raise ValueError("volume_step_percent must be an integer between 1 and 100")
        self.volume_step_percent = volume_step_percent

    def execute(self, intent: ActionIntent) -> None:
        if intent.action in _MEDIA_KEY_CODES:
            key_code = _MEDIA_KEY_CODES[intent.action]
            script = f'tell application "System Events" to key code {key_code}'
            subprocess.run(["osascript", "-e", script], check=True)
            return
        if intent.action in (MacAction.VOLUME_UP, MacAction.VOLUME_DOWN):
            self._adjust_volume(1 if intent.action is MacAction.VOLUME_UP else -1)
            return
        raise ValueError(f"unsupported macOS action: {intent.action}")

    def _adjust_volume(self, direction: int) -> None:
        current = subprocess.run(
            ["osascript", "-e", "output volume of (get volume settings)"],
            check=True,
            capture_output=True,
            text=True,
        )
        current_volume = int(current.stdout.strip())
        volume = max(0, min(100, current_volume + direction * self.volume_step_percent))
        subprocess.run(
            ["osascript", "-e", f"set volume output volume {volume}"],
            check=True,
        )
