from dataclasses import dataclass
from enum import Enum
import math

import numpy as np


PERSON_ROLE = 'person'
ARM_ROLE = 'arm'

COCO_SKELETON_EDGES = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)
ARM_SKELETON_EDGES = tuple((index, index + 1) for index in range(5))


class DepthQuality(str, Enum):
    DIRECT = 'direct'
    RECOVERED = 'recovered'
    INVALID = 'invalid'


@dataclass(frozen=True)
class PoseKeypoint:
    role: str
    class_id: int
    detection_id: int
    keypoint_id: int
    pixel: tuple[int, int]


@dataclass(frozen=True)
class DepthRecoveryConfig:
    depth_max_mm: float = 10000.0
    absolute_tolerance_mm: float = 50.0
    relative_tolerance: float = 0.05
    min_candidates: int = 3
    max_mad_mm: float = 20.0
    direct_uncertainty_mm: float = 20.0
    recovered_uncertainty_mm: float = 50.0
    history_max_age_seconds: float = 0.15
    history_pixel_gate_px: float = 60.0
    association_max_center_distance_ratio: float = 0.5
    association_min_center_distance_px: float = 25.0
    window_sizes: tuple[int, ...] = (3, 5, 7)

    def __post_init__(self):
        positive_values = {
            'depth_max_mm': self.depth_max_mm,
            'absolute_tolerance_mm': self.absolute_tolerance_mm,
            'min_candidates': self.min_candidates,
            'direct_uncertainty_mm': self.direct_uncertainty_mm,
            'recovered_uncertainty_mm': self.recovered_uncertainty_mm,
            'history_max_age_seconds': self.history_max_age_seconds,
            'history_pixel_gate_px': self.history_pixel_gate_px,
            'association_max_center_distance_ratio': (
                self.association_max_center_distance_ratio
            ),
            'association_min_center_distance_px': (
                self.association_min_center_distance_px
            ),
        }
        for name, value in positive_values.items():
            if float(value) <= 0:
                raise ValueError(f'{name} must be greater than 0')
        if self.relative_tolerance < 0:
            raise ValueError('relative_tolerance must not be negative')
        if self.max_mad_mm < 0:
            raise ValueError('max_mad_mm must not be negative')
        if not self.window_sizes:
            raise ValueError('window_sizes must not be empty')
        if any(size <= 0 or size % 2 == 0 for size in self.window_sizes):
            raise ValueError('window_sizes must contain positive odd integers')
        if tuple(sorted(self.window_sizes)) != self.window_sizes:
            raise ValueError('window_sizes must be sorted smallest to largest')


DEFAULT_DEPTH_RECOVERY_CONFIG = DepthRecoveryConfig()


@dataclass(frozen=True)
class DepthMeasurement:
    depth_mm: float | None
    quality: DepthQuality
    uncertainty_mm: float
    recovery_window_size: int | None = None

    @property
    def valid(self):
        return self.depth_mm is not None and self.quality != DepthQuality.INVALID


@dataclass(frozen=True)
class PosePoint3D:
    keypoint: PoseKeypoint
    xyz_m: tuple[float, float, float]
    measurement: DepthMeasurement

    @property
    def pixel(self):
        return self.keypoint.pixel


@dataclass(frozen=True)
class _TrackedDetection:
    class_id: int
    detection_id: int
    box_xyxy: tuple[float, float, float, float]


class DetectionInstanceAssociator:
    """Associate detections only with the immediately preceding nearby frame."""

    def __init__(
        self,
        max_age_seconds=DEFAULT_DEPTH_RECOVERY_CONFIG.history_max_age_seconds,
        max_center_distance_ratio=(
            DEFAULT_DEPTH_RECOVERY_CONFIG.association_max_center_distance_ratio
        ),
        min_center_distance_px=(
            DEFAULT_DEPTH_RECOVERY_CONFIG.association_min_center_distance_px
        ),
    ):
        if max_age_seconds <= 0:
            raise ValueError('max_age_seconds must be greater than 0')
        if max_center_distance_ratio <= 0:
            raise ValueError('max_center_distance_ratio must be greater than 0')
        if min_center_distance_px <= 0:
            raise ValueError('min_center_distance_px must be greater than 0')
        self.max_age_seconds = float(max_age_seconds)
        self.max_center_distance_ratio = float(max_center_distance_ratio)
        self.min_center_distance_px = float(min_center_distance_px)
        self._next_detection_id = 0
        self._previous_timestamp = None
        self._previous = []

    def assign(self, class_ids, boxes_xyxy, timestamp_seconds):
        class_ids = np.asarray(class_ids, dtype=np.int64).reshape(-1)
        boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float64)
        if boxes_xyxy.shape != (class_ids.size, 4):
            raise ValueError('boxes_xyxy must have shape (detections, 4)')

        timestamp_seconds = float(timestamp_seconds)
        can_reuse_previous = (
            self._previous_timestamp is not None
            and 0.0
            <= timestamp_seconds - self._previous_timestamp
            <= self.max_age_seconds
        )
        assigned_ids = [None] * class_ids.size

        if can_reuse_previous:
            candidates = []
            for current_index, (class_id, box) in enumerate(
                zip(class_ids, boxes_xyxy)
            ):
                for previous_index, previous in enumerate(self._previous):
                    if int(class_id) != previous.class_id:
                        continue
                    distance, gate = _box_center_distance_and_gate(
                        box,
                        previous.box_xyxy,
                        self.max_center_distance_ratio,
                        self.min_center_distance_px,
                    )
                    if distance <= gate:
                        candidates.append(
                            (distance / gate, current_index, previous_index)
                        )

            used_current = set()
            used_previous = set()
            for _, current_index, previous_index in sorted(candidates):
                if (
                    current_index in used_current
                    or previous_index in used_previous
                ):
                    continue
                assigned_ids[current_index] = self._previous[
                    previous_index
                ].detection_id
                used_current.add(current_index)
                used_previous.add(previous_index)

        for index, detection_id in enumerate(assigned_ids):
            if detection_id is None:
                assigned_ids[index] = self._next_detection_id
                self._next_detection_id += 1

        self._previous_timestamp = timestamp_seconds
        self._previous = [
            _TrackedDetection(
                class_id=int(class_id),
                detection_id=int(detection_id),
                box_xyxy=tuple(float(value) for value in box),
            )
            for class_id, detection_id, box in zip(
                class_ids,
                assigned_ids,
                boxes_xyxy,
            )
        ]
        return np.asarray(assigned_ids, dtype=np.int64)


def _box_center_distance_and_gate(
    current_box,
    previous_box,
    max_center_distance_ratio,
    min_center_distance_px,
):
    current_box = np.asarray(current_box, dtype=np.float64)
    previous_box = np.asarray(previous_box, dtype=np.float64)
    if not np.isfinite(current_box).all() or not np.isfinite(previous_box).all():
        return float('inf'), 0.0

    current_center = (current_box[:2] + current_box[2:]) / 2.0
    previous_center = (previous_box[:2] + previous_box[2:]) / 2.0
    distance = float(np.linalg.norm(current_center - previous_center))
    current_diagonal = float(np.linalg.norm(current_box[2:] - current_box[:2]))
    previous_diagonal = float(
        np.linalg.norm(previous_box[2:] - previous_box[:2])
    )
    gate = max(
        min_center_distance_px,
        max_center_distance_ratio * max(current_diagonal, previous_diagonal),
    )
    return distance, gate


def select_pose_keypoints_by_class(
    class_ids,
    keypoint_xy,
    keypoint_confidence,
    keypoint_counts_by_class,
    person_class_id,
    arm_class_ids,
    confidence_threshold,
    detection_instance_ids=None,
    image_shape=None,
):
    """Return pose keypoints while retaining detection and keypoint identity."""
    class_ids = np.asarray(class_ids, dtype=np.int64).reshape(-1)
    keypoint_xy = np.asarray(keypoint_xy, dtype=np.float32)
    if keypoint_xy.ndim != 3 or keypoint_xy.shape[-1] != 2:
        raise ValueError('keypoint_xy must have shape (detections, keypoints, 2)')
    if keypoint_xy.shape[0] != class_ids.shape[0]:
        raise ValueError('detection classes and keypoints must have equal length')

    if detection_instance_ids is None:
        detection_instance_ids = np.arange(class_ids.size, dtype=np.int64)
    else:
        detection_instance_ids = np.asarray(
            detection_instance_ids,
            dtype=np.int64,
        ).reshape(-1)
        if detection_instance_ids.shape != class_ids.shape:
            raise ValueError('detection_instance_ids must match class_ids')

    if keypoint_confidence is not None:
        keypoint_confidence = np.asarray(
            keypoint_confidence,
            dtype=np.float32,
        )
        if keypoint_confidence.shape != keypoint_xy.shape[:2]:
            raise ValueError(
                'keypoint_confidence must match the first two keypoint_xy axes'
            )

    image_height = None
    image_width = None
    if image_shape is not None:
        if len(image_shape) < 2:
            raise ValueError('image_shape must contain height and width')
        image_height = int(image_shape[0])
        image_width = int(image_shape[1])
        if image_height <= 0 or image_width <= 0:
            raise ValueError('image_shape height and width must be positive')

    person_keypoints = []
    arm_keypoints = []
    arm_class_ids = {int(class_id) for class_id in arm_class_ids}

    for detection_index, class_id_value in enumerate(class_ids):
        class_id = int(class_id_value)
        if class_id != person_class_id and class_id not in arm_class_ids:
            continue

        keypoint_count = min(
            int(keypoint_counts_by_class.get(class_id, 0)),
            keypoint_xy.shape[1],
        )
        role = PERSON_ROLE if class_id == person_class_id else ARM_ROLE
        destination = (
            person_keypoints if role == PERSON_ROLE else arm_keypoints
        )
        for keypoint_index in range(keypoint_count):
            x, y = keypoint_xy[detection_index, keypoint_index]
            confidence = (
                1.0
                if keypoint_confidence is None
                else float(
                    keypoint_confidence[detection_index, keypoint_index]
                )
            )
            if confidence < confidence_threshold:
                continue
            if not np.isfinite((x, y)).all() or x <= 0 or y <= 0:
                continue
            if (
                image_shape is not None
                and (x >= image_width or y >= image_height)
            ):
                continue
            destination.append(
                PoseKeypoint(
                    role=role,
                    class_id=class_id,
                    detection_id=int(detection_instance_ids[detection_index]),
                    keypoint_id=keypoint_index,
                    pixel=(int(round(float(x))), int(round(float(y)))),
                )
            )

    return person_keypoints, arm_keypoints


def select_keypoints_by_class(
    class_ids,
    keypoint_xy,
    keypoint_confidence,
    keypoint_counts_by_class,
    person_class_id,
    arm_class_ids,
    confidence_threshold,
    image_shape=None,
):
    """Return valid person and arm pose keypoints as integer pixel pairs."""
    person_keypoints, arm_keypoints = select_pose_keypoints_by_class(
        class_ids=class_ids,
        keypoint_xy=keypoint_xy,
        keypoint_confidence=keypoint_confidence,
        keypoint_counts_by_class=keypoint_counts_by_class,
        person_class_id=person_class_id,
        arm_class_ids=arm_class_ids,
        confidence_threshold=confidence_threshold,
        image_shape=image_shape,
    )
    return (
        [keypoint.pixel for keypoint in person_keypoints],
        [keypoint.pixel for keypoint in arm_keypoints],
    )


def map_pixel_between_image_shapes(pixel, source_shape, target_shape):
    """Map a pixel between image resolutions without clamping invalid input."""
    source_height, source_width = _validated_image_shape(
        source_shape,
        'source_shape',
    )
    target_height, target_width = _validated_image_shape(
        target_shape,
        'target_shape',
    )
    x, y = (int(pixel[0]), int(pixel[1]))
    if not 0 <= x < source_width or not 0 <= y < source_height:
        raise ValueError('pixel must be inside source_shape')

    mapped_x = min(target_width - 1, int(round(x * target_width / source_width)))
    mapped_y = min(
        target_height - 1,
        int(round(y * target_height / source_height)),
    )
    return mapped_x, mapped_y


def _validated_image_shape(image_shape, name):
    if len(image_shape) < 2:
        raise ValueError(f'{name} must contain height and width')
    height = int(image_shape[0])
    width = int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f'{name} height and width must be positive')
    return height, width


def measure_depth_at_keypoint(
    depth_image,
    pixel,
    reference_depth_mm,
    config=DEFAULT_DEPTH_RECOVERY_CONFIG,
):
    """Measure direct depth or recover it from reference-consistent neighbors."""
    depth_image = np.asarray(depth_image)
    if depth_image.ndim != 2:
        raise ValueError('depth_image must be a two-dimensional array')
    if not _pixel_in_image(pixel, depth_image.shape):
        return _invalid_depth_measurement()
    x, y = (int(pixel[0]), int(pixel[1]))
    center_depth = float(depth_image[y, x])
    if _valid_depth(center_depth, config.depth_max_mm):
        return DepthMeasurement(
            depth_mm=center_depth,
            quality=DepthQuality.DIRECT,
            uncertainty_mm=config.direct_uncertainty_mm,
        )

    if not _valid_depth(reference_depth_mm, config.depth_max_mm):
        return _invalid_depth_measurement()

    candidate_tolerance = max(
        config.absolute_tolerance_mm,
        float(reference_depth_mm) * config.relative_tolerance,
    )
    for window_size in config.window_sizes:
        radius = window_size // 2
        x_start = max(0, x - radius)
        x_stop = min(depth_image.shape[1], x + radius + 1)
        y_start = max(0, y - radius)
        y_stop = min(depth_image.shape[0], y + radius + 1)
        candidates = depth_image[y_start:y_stop, x_start:x_stop].reshape(-1)
        candidates = candidates[
            np.isfinite(candidates)
            & (candidates > 0)
            & (candidates < config.depth_max_mm)
            & (np.abs(candidates - reference_depth_mm) <= candidate_tolerance)
        ].astype(np.float64, copy=False)
        if candidates.size < config.min_candidates:
            continue

        median_depth = float(np.median(candidates))
        mad_mm = float(np.median(np.abs(candidates - median_depth)))
        if mad_mm > config.max_mad_mm:
            continue
        return DepthMeasurement(
            depth_mm=median_depth,
            quality=DepthQuality.RECOVERED,
            uncertainty_mm=config.recovered_uncertainty_mm,
            recovery_window_size=window_size,
        )

    return _invalid_depth_measurement()


def _pixel_in_image(pixel, image_shape):
    x, y = (int(pixel[0]), int(pixel[1]))
    return 0 <= x < image_shape[1] and 0 <= y < image_shape[0]


def _valid_depth(depth_mm, depth_max_mm):
    if depth_mm is None:
        return False
    try:
        depth_mm = float(depth_mm)
    except (TypeError, ValueError):
        return False
    return math.isfinite(depth_mm) and 0.0 < depth_mm < depth_max_mm


def _invalid_depth_measurement():
    return DepthMeasurement(
        depth_mm=None,
        quality=DepthQuality.INVALID,
        uncertainty_mm=float('inf'),
    )


def _history_key(keypoint):
    return (
        keypoint.role,
        keypoint.class_id,
        keypoint.detection_id,
        keypoint.keypoint_id,
    )


def _neighbor_ids(role, keypoint_id):
    edges = (
        COCO_SKELETON_EDGES if role == PERSON_ROLE else ARM_SKELETON_EDGES
    )
    neighbors = []
    for first, second in edges:
        if first == keypoint_id:
            neighbors.append(second)
        elif second == keypoint_id:
            neighbors.append(first)
    return neighbors


class PoseDepthResolver:
    """Resolve pose depths with per-instance history and skeleton references."""

    def __init__(self, config=DEFAULT_DEPTH_RECOVERY_CONFIG):
        self.config = config
        self._history = {}

    def resolve(
        self,
        depth_image,
        keypoints,
        timestamp_seconds,
        source_image_shape=None,
    ):
        keypoints = list(keypoints)
        timestamp_seconds = float(timestamp_seconds)
        self._prune_history(timestamp_seconds)
        depth_shape = np.asarray(depth_image).shape
        if source_image_shape is None:
            source_image_shape = depth_shape

        sample_pixels = []
        for keypoint in keypoints:
            try:
                sample_pixels.append(
                    map_pixel_between_image_shapes(
                        keypoint.pixel,
                        source_image_shape,
                        depth_shape,
                    )
                )
            except ValueError:
                sample_pixels.append(None)

        direct_measurements = {}
        for keypoint, sample_pixel in zip(keypoints, sample_pixels):
            measurement = (
                _invalid_depth_measurement()
                if sample_pixel is None
                else measure_depth_at_keypoint(
                    depth_image,
                    sample_pixel,
                    reference_depth_mm=None,
                    config=self.config,
                )
            )
            if measurement.quality == DepthQuality.DIRECT:
                direct_measurements[_history_key(keypoint)] = measurement

        resolved = []
        for keypoint, sample_pixel in zip(keypoints, sample_pixels):
            key = _history_key(keypoint)
            measurement = direct_measurements.get(key)
            if measurement is None:
                reference_depth_mm = self._history_reference(
                    keypoint,
                    timestamp_seconds,
                )
                if reference_depth_mm is None:
                    reference_depth_mm = self._skeleton_reference(
                        keypoint,
                        direct_measurements,
                    )
                measurement = (
                    _invalid_depth_measurement()
                    if sample_pixel is None
                    else measure_depth_at_keypoint(
                        depth_image,
                        sample_pixel,
                        reference_depth_mm=reference_depth_mm,
                        config=self.config,
                    )
                )
            resolved.append((keypoint, measurement))

        for keypoint, measurement in resolved:
            if measurement.valid:
                self._history[_history_key(keypoint)] = (
                    float(measurement.depth_mm),
                    timestamp_seconds,
                    keypoint.pixel,
                )
        return resolved

    def _history_reference(self, keypoint, timestamp_seconds):
        history = self._history.get(_history_key(keypoint))
        if history is None:
            return None
        depth_mm, stored_at, stored_pixel = history
        age_seconds = timestamp_seconds - stored_at
        if not 0.0 <= age_seconds <= self.config.history_max_age_seconds:
            return None
        pixel_distance = math.dist(keypoint.pixel, stored_pixel)
        if pixel_distance > self.config.history_pixel_gate_px:
            return None
        return depth_mm

    def _skeleton_reference(self, keypoint, direct_measurements):
        neighbor_depths = []
        for neighbor_id in _neighbor_ids(keypoint.role, keypoint.keypoint_id):
            neighbor_key = (
                keypoint.role,
                keypoint.class_id,
                keypoint.detection_id,
                neighbor_id,
            )
            measurement = direct_measurements.get(neighbor_key)
            if measurement is not None:
                neighbor_depths.append(measurement.depth_mm)
        if not neighbor_depths:
            return None
        return float(np.median(neighbor_depths))

    def _prune_history(self, timestamp_seconds):
        oldest_allowed = timestamp_seconds - self.config.history_max_age_seconds
        self._history = {
            key: value
            for key, value in self._history.items()
            if oldest_allowed <= value[1] <= timestamp_seconds
        }


def conservative_distance_m(
    first_xyz_m,
    second_xyz_m,
    first_uncertainty_mm=0.0,
    second_uncertainty_mm=0.0,
):
    """Return Euclidean distance reduced by both endpoint uncertainties."""
    if first_uncertainty_mm < 0 or second_uncertainty_mm < 0:
        raise ValueError('uncertainty must not be negative')
    euclidean_distance = float(
        np.linalg.norm(
            np.asarray(first_xyz_m, dtype=np.float64)
            - np.asarray(second_xyz_m, dtype=np.float64)
        )
    )
    uncertainty_m = (first_uncertainty_mm + second_uncertainty_mm) / 1000.0
    return max(0.0, euclidean_distance - uncertainty_m)


def _point_xyz(point):
    return point.xyz_m if isinstance(point, PosePoint3D) else point[1]


def compute_min_distance_between_sets(person_points, arm_points):
    """Return the closest raw Euclidean distance and its endpoint pair."""
    if not person_points or not arm_points:
        return float('inf'), None, None

    person_xyz = np.asarray(
        [_point_xyz(point) for point in person_points],
        dtype=np.float64,
    )
    arm_xyz = np.asarray(
        [_point_xyz(point) for point in arm_points],
        dtype=np.float64,
    )
    distances = np.linalg.norm(
        person_xyz[:, np.newaxis, :] - arm_xyz[np.newaxis, :, :],
        axis=2,
    )
    person_index, arm_index = np.unravel_index(
        int(np.argmin(distances)),
        distances.shape,
    )
    return (
        float(distances[person_index, arm_index]),
        person_points[person_index],
        arm_points[arm_index],
    )
