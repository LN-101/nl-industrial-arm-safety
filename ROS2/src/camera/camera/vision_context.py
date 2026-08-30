"""Helpers for ROS2 vision context snapshot Trigger responses."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_VISION_SERVICE_NAME = '/vision/capture_snapshot'
DEFAULT_VISION_OUTPUT_DIR = Path(
    '/home/inteldk/ROS2/.runtime/vision_snapshots'
)
DEFAULT_VISION_FRAME_ID = 'camera_frame'
DEFAULT_VISION_SOURCE = 'min_dis'
SUPPORTED_IMAGE_EXTENSIONS = frozenset({'jpg', 'jpeg', 'png'})


def normalize_image_extension(value: str) -> str:
    """Return a supported lowercase image extension without a leading dot."""
    extension = str(value or '').strip().lower().lstrip('.')
    if extension not in SUPPORTED_IMAGE_EXTENSIONS:
        supported = ', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        raise ValueError(
            f'Unsupported vision image extension {value!r}; '
            f'expected one of: {supported}'
        )
    return extension


def finite_float_or_none(value: Any) -> float | None:
    """Convert finite numeric values to float and reject NaN/inf as None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def point_to_payload(point: Any) -> dict[str, float] | None:
    """Convert an XYZ point tuple/list into a JSON-safe object."""
    if point is None:
        return None
    try:
        x, y, z = point
    except (TypeError, ValueError):
        return None
    return {
        'x': float(x),
        'y': float(y),
        'z': float(z),
    }


def snapshot_timestamp(wall_time_seconds: float | None = None) -> str:
    """Return an ISO-8601 UTC timestamp with millisecond precision."""
    if wall_time_seconds is None:
        timestamp = datetime.now(timezone.utc)
    else:
        timestamp = datetime.fromtimestamp(wall_time_seconds, tz=timezone.utc)
    return timestamp.isoformat(timespec='milliseconds').replace('+00:00', 'Z')


def build_context_payload(
    *,
    image_path: Path,
    stamp: str,
    frame_id: str,
    min_distance_m: Any,
    human_closest_point: Any,
    arm_closest_point: Any,
    emergency_stop: bool,
    fresh: bool,
    age_ms: float,
    source: str,
    width: int,
    height: int,
    reason: str = '',
    threshold_m: Any = None,
) -> str:
    """Build the Trigger response JSON payload for one vision context."""
    payload: dict[str, Any] = {
        'image_path': str(image_path),
        'stamp': stamp,
        'frame_id': frame_id,
        'min_distance_m': finite_float_or_none(min_distance_m),
        'human_closest_point': point_to_payload(human_closest_point),
        'arm_closest_point': point_to_payload(arm_closest_point),
        'emergency_stop': bool(emergency_stop),
        'fresh': bool(fresh),
        'age_ms': round(float(age_ms), 3),
        'source': str(source),
        'width': int(width),
        'height': int(height),
    }
    threshold = finite_float_or_none(threshold_m)
    if threshold is not None:
        payload['threshold_m'] = threshold
    if reason:
        payload['reason'] = str(reason)
    return json.dumps(payload, ensure_ascii=False, allow_nan=False)


def write_rgb_snapshot(
    rgb_image: Any,
    output_path: Path,
    *,
    image_extension: str,
    jpeg_quality: int = 92,
) -> Path:
    """Write one RGB uint8 image as jpg/jpeg/png and return the final path."""
    import cv2
    import numpy as np

    extension = normalize_image_extension(image_extension)
    image = np.asarray(rgb_image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f'RGB snapshot must have shape HxWx3; got {image.shape!r}'
        )
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    output_path = output_path.with_suffix(f'.{extension}')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    params: list[int] = []
    if extension in {'jpg', 'jpeg'}:
        params = [
            int(cv2.IMWRITE_JPEG_QUALITY),
            int(max(1, min(jpeg_quality, 100))),
        ]
    ok = cv2.imwrite(str(output_path), bgr_image, params)
    if not ok or not output_path.is_file():
        raise RuntimeError(f'Failed to write vision snapshot: {output_path}')
    return output_path
