"""Versioned JSON calibration persistence."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

from roomsense.calibration.camera_calibration import Calibration, create_calibration


def save_calibration(path: str | Path, calibration: Calibration) -> None:
    """Validate and atomically save calibration JSON."""

    from roomsense.calibration.camera_calibration import validate_calibration

    validate_calibration(calibration)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": calibration.version,
        "image_points": [list(point) for point in calibration.image_points],
        "source_size": [calibration.source_width, calibration.source_height],
        "map_size": [calibration.map_width, calibration.map_height],
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".tmp", delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, indent=2, allow_nan=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_calibration(
    path: str | Path,
    expected_size: tuple[int, int] | None = None,
) -> Calibration | None:
    """Load validated JSON; return ``None`` when the file is absent."""

    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as calibration_file:
            payload: Any = json.load(calibration_file)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read calibration file: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("calibration JSON must contain an object")
    try:
        if payload.get("version") != 1:
            raise ValueError(f"unsupported calibration version: {payload.get('version')}")
        source_size = payload["source_size"]
        map_size = payload["map_size"]
        calibration = create_calibration(payload["image_points"], tuple(source_size), tuple(map_size))
        # Keep the version check explicit even if a future decoder changes.
        from roomsense.calibration.camera_calibration import validate_calibration

        validate_calibration(calibration, expected_size)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid calibration file: {exc}") from exc
    return calibration
