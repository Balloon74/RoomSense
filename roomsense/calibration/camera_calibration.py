"""Camera-image to normalized floor-map projective calibration."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


Point = tuple[float, float]
CALIBRATION_VERSION = 1
_MIN_POLYGON_AREA = 1e-3
_MIN_POINT_SEPARATION = 1e-4


class ProjectiveHorizonError(ValueError):
    """Raised when the homogeneous projection denominator is zero."""


@dataclass(frozen=True)
class Calibration:
    """Validated normalized image corners and their source/map dimensions."""

    image_points: tuple[Point, Point, Point, Point]
    source_width: int
    source_height: int
    map_width: float = 1.0
    map_height: float = 1.0
    version: int = CALIBRATION_VERSION


def create_calibration(
    image_points: Iterable[Iterable[float]],
    source_size: tuple[int, int],
    map_size: tuple[float, float] = (1.0, 1.0),
) -> Calibration:
    """Create and validate calibration from normalized image points."""

    try:
        points = tuple(tuple(float(component) for component in point) for point in image_points)
        width, height = source_size
        map_width, map_height = map_size
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("calibration points and dimensions are malformed") from exc
    if len(points) != 4 or any(len(point) != 2 for point in points):
        raise ValueError("calibration requires exactly four 2D image points")
    try:
        map_width, map_height = float(map_width), float(map_height)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("normalized map dimensions must be finite numbers") from exc
    calibration = Calibration(
        image_points=points,  # type: ignore[arg-type]
        source_width=width,
        source_height=height,
        map_width=map_width,
        map_height=map_height,
    )
    validate_calibration(calibration)
    return calibration


def validate_calibration(
    calibration: Calibration,
    expected_size: tuple[int, int] | None = None,
) -> None:
    """Raise ``ValueError`` unless calibration is safe to use for a frame."""

    if calibration.version != CALIBRATION_VERSION:
        raise ValueError(f"unsupported calibration version: {calibration.version}")
    if (not isinstance(calibration.source_width, int) or not isinstance(calibration.source_height, int)
            or calibration.source_width <= 0 or calibration.source_height <= 0):
        raise ValueError("source image dimensions must be positive integers")
    try:
        map_width, map_height = float(calibration.map_width), float(calibration.map_height)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("normalized map dimensions must be positive and finite") from exc
    if (not math.isfinite(map_width) or not math.isfinite(map_height)
            or map_width <= 0 or map_height <= 0):
        raise ValueError("normalized map dimensions must be positive and finite")
    if len(calibration.image_points) != 4:
        raise ValueError("calibration requires exactly four image points")
    points = np.asarray(calibration.image_points, dtype=np.float64)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("calibration image points must be four finite 2D points")
    if np.any(points < 0.0) or np.any(points > 1.0):
        raise ValueError("calibration image points must be normalized within the frame")
    for first in range(4):
        for second in range(first + 1, 4):
            if np.linalg.norm(points[first] - points[second]) < _MIN_POINT_SEPARATION:
                raise ValueError("calibration image points must be distinct")
    if not _is_convex(points):
        raise ValueError("calibration points must form a convex polygon in the specified order")
    if _polygon_area(points) < _MIN_POLYGON_AREA:
        raise ValueError("calibration floor polygon is too small")
    if expected_size is not None:
        try:
            expected_width, expected_height = expected_size
        except (TypeError, ValueError) as exc:
            raise ValueError("expected image size must be a width-height pair") from exc
        if not isinstance(expected_width, int) or not isinstance(expected_height, int):
            raise ValueError("expected image dimensions must be integers")
        if expected_width <= 0 or expected_height <= 0:
            raise ValueError("expected image dimensions must be positive")
        # Compare integer cross-products so maliciously large JSON dimensions
        # cannot overflow while converting to a floating-point aspect ratio.
        saved_scaled = calibration.source_width * expected_height
        current_scaled = expected_width * calibration.source_height
        if abs(saved_scaled - current_scaled) * 1000 > max(saved_scaled, current_scaled):
            raise ValueError("camera aspect ratio differs from the saved calibration; recalibrate")
    matrix = homography_for(calibration)
    if not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-10:
        raise ValueError("calibration transform is singular or non-finite")


def homography_for(calibration: Calibration) -> np.ndarray:
    """Return the image-point to normalized top-down-map homography."""

    import cv2

    source = np.asarray(calibration.image_points, dtype=np.float32)
    destination = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float32)
    return cv2.getPerspectiveTransform(source, destination)


def transform_point(point: Point, homography: np.ndarray) -> Point:
    """Project a normalized image point through a 3x3 homography."""

    matrix = np.asarray(homography, dtype=np.float64)
    source = np.asarray(point, dtype=np.float64)
    if matrix.shape != (3, 3) or source.shape != (2,) or not np.isfinite(matrix).all() or not np.isfinite(source).all():
        raise ValueError("point and homography must contain finite 2D and 3x3 values")
    projected = matrix @ np.asarray((source[0], source[1], 1.0))
    if abs(projected[2]) < 1e-12:
        raise ProjectiveHorizonError("point projects to infinity")
    return float(projected[0] / projected[2]), float(projected[1] / projected[2])


def _cross(first: np.ndarray, second: np.ndarray, third: np.ndarray) -> float:
    a, b = second - first, third - second
    return float(a[0] * b[1] - a[1] * b[0])


def _is_convex(points: np.ndarray) -> bool:
    crosses = [_cross(points[index], points[(index + 1) % 4], points[(index + 2) % 4]) for index in range(4)]
    return all(value > 1e-9 for value in crosses) or all(value < -1e-9 for value in crosses)


def _polygon_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))) * 0.5
