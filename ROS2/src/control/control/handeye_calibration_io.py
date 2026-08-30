from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import os
from pathlib import Path
import tempfile

import numpy as np
import yaml


CALIBRATION_SCHEMA_VERSION = 1
DEFAULT_MAX_FIT_RMSE_M = 0.003
DEFAULT_MAX_CONDITION_NUMBER = 1e4
MIN_LAYER_SAMPLES = 5
DEFAULT_RUNTIME_WORKSPACE_MIN = np.array([-0.32, -0.32, 0.05])
DEFAULT_RUNTIME_WORKSPACE_MAX = np.array([0.38, 0.38, 0.30])
WS712_OUTPUT_ANCHOR_XY = np.array([0.08, 0.33])
WS712_OUTPUT_CORRECTION_MATRIX = np.array([
    [2.22035638, -0.55975371],
    [4.71126038, -3.28855304],
])
WS712_OUTPUT_OFFSET_XY = np.array([-0.0112, 0.0783])
DEFAULT_OUTPUT_CORRECTION_MATRIX = np.eye(2, dtype=np.float64)
DEFAULT_OUTPUT_OFFSET_XY = np.zeros(2, dtype=np.float64)


@dataclass(frozen=True)
class LayerFit:
    matrix: np.ndarray
    offset: np.ndarray
    rmse_m: float
    condition_number: float
    inlier_indices: tuple[int, ...]
    rejected_indices: tuple[int, ...]


@dataclass(frozen=True)
class HandeyeCalibration:
    z_layers: np.ndarray
    layer_matrices: np.ndarray
    layer_offsets: np.ndarray
    image_center: np.ndarray
    tool_to_camera_xy: np.ndarray
    workspace_min: np.ndarray
    workspace_max: np.ndarray
    layer_quality: tuple[dict, ...]
    metadata: dict
    verified_pixel_to_base_affine: np.ndarray | None = None
    pixel_to_base_delta: np.ndarray | None = None
    base_xy_offset: np.ndarray | None = None
    known_target_xy: np.ndarray = field(
        default_factory=lambda: WS712_OUTPUT_ANCHOR_XY.copy())
    output_anchor_xy: np.ndarray = field(
        default_factory=lambda: WS712_OUTPUT_ANCHOR_XY.copy())
    output_correction_matrix: np.ndarray = field(
        default_factory=lambda: DEFAULT_OUTPUT_CORRECTION_MATRIX.copy())
    output_offset_xy: np.ndarray = field(
        default_factory=lambda: DEFAULT_OUTPUT_OFFSET_XY.copy())
    calibration_workspace: dict = field(default_factory=dict)
    sampled_layer_y_ranges: tuple[tuple[float, float], ...] = ()
    fit_statistics: dict = field(default_factory=dict)

    def mapping_at_z(self, z_value):
        if z_value <= self.z_layers[0]:
            return self.layer_matrices[0], self.layer_offsets[0]
        if z_value >= self.z_layers[-1]:
            return self.layer_matrices[-1], self.layer_offsets[-1]
        upper = int(np.searchsorted(self.z_layers, z_value))
        lower = upper - 1
        alpha = (
            (z_value - self.z_layers[lower])
            / (self.z_layers[upper] - self.z_layers[lower])
        )
        matrix = (
            (1.0 - alpha) * self.layer_matrices[lower]
            + alpha * self.layer_matrices[upper]
        )
        offset = (
            (1.0 - alpha) * self.layer_offsets[lower]
            + alpha * self.layer_offsets[upper]
        )
        return matrix, offset

    def pixel_to_base(self, pixel, end_effector_position):
        pixel = _finite_array(pixel, (2,), 'pixel')
        if self.verified_pixel_to_base_affine is not None:
            return self.verified_pixel_to_base_affine @ np.array([
                pixel[0], pixel[1], 1.0])
        end_effector_position = _finite_array(
            end_effector_position,
            (3,),
            'end_effector_position',
        )
        matrix, offset = self.mapping_at_z(end_effector_position[2])
        camera_xy = end_effector_position[:2] + self.tool_to_camera_xy
        uncorrected_position = (
            camera_xy - matrix @ (pixel - self.image_center) + offset
        )
        return self.output_anchor_xy + self.output_correction_matrix @ (
            uncorrected_position - self.output_anchor_xy
        ) + self.output_offset_xy

    def contains(self, point):
        point = _finite_array(point, (3,), 'point')
        return bool(
            np.all(point >= self.workspace_min)
            and np.all(point <= self.workspace_max)
        )

    def to_document(self):
        middle_index = len(self.layer_matrices) // 2
        document = {
            'schema_version': CALIBRATION_SCHEMA_VERSION,
            'calibration_type': 'eye_in_hand_layered_xy',
            'pixel_to_base_delta': (
                self.pixel_to_base_delta
                if self.pixel_to_base_delta is not None
                else self.layer_matrices[middle_index]
            ).tolist(),
            'z_layers': self.z_layers.tolist(),
            'layer_matrices': self.layer_matrices.tolist(),
            'base_xy_offset': (
                self.base_xy_offset
                if self.base_xy_offset is not None
                else self.layer_offsets[middle_index]
            ).tolist(),
            'layer_offsets': self.layer_offsets.tolist(),
            'known_target_xy': self.known_target_xy.tolist(),
            'output_anchor_xy': self.output_anchor_xy.tolist(),
            'output_correction_matrix': self.output_correction_matrix.tolist(),
            'output_offset_xy': self.output_offset_xy.tolist(),
            'image_center': self.image_center.tolist(),
            'tool_to_camera_xy': self.tool_to_camera_xy.tolist(),
            'workspace': deepcopy(self.calibration_workspace) or {
                'min': self.workspace_min.tolist(),
                'max': self.workspace_max.tolist(),
            },
            'runtime_workspace': {
                'min': self.workspace_min.tolist(),
                'max': self.workspace_max.tolist(),
            },
            'sampled_layer_y_ranges': [
                list(values) for values in self.sampled_layer_y_ranges
            ],
            'layer_quality': deepcopy(list(self.layer_quality)),
            'metadata': deepcopy(self.metadata),
        }
        if self.verified_pixel_to_base_affine is not None:
            document['verified_pixel_to_base_affine'] = (
                self.verified_pixel_to_base_affine.tolist())
        document.update(deepcopy(self.fit_statistics))
        return document


def _finite_array(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f'{name} must have shape {shape}, got {array.shape}')
    if not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must contain only finite values')
    return array


def _finite_range(value, name, allow_equal=True):
    result = _finite_array(value, (2,), name)
    valid = result[0] <= result[1] if allow_equal else result[0] < result[1]
    if not valid:
        raise ValueError(f'{name} lower bound must not exceed upper bound')
    return result


def fit_layer(ee_xy, pixels):
    ee_xy = np.asarray(ee_xy, dtype=np.float64)
    pixels = np.asarray(pixels, dtype=np.float64)
    if ee_xy.shape != pixels.shape or ee_xy.ndim != 2 or ee_xy.shape[1] != 2:
        raise ValueError('ee_xy and pixels must have matching Nx2 shapes')
    if len(ee_xy) < MIN_LAYER_SAMPLES:
        return None
    if not np.all(np.isfinite(ee_xy)) or not np.all(np.isfinite(pixels)):
        raise ValueError('calibration samples must be finite')
    delta_ee = ee_xy - ee_xy[0]
    delta_pixels = pixels - pixels[0]
    design = np.column_stack([delta_pixels, np.ones(len(delta_pixels))])
    coefficients, _, rank, _ = np.linalg.lstsq(design, delta_ee, rcond=None)
    if rank < 3:
        return None
    matrix = coefficients[:2, :].T
    predicted = design @ coefficients
    residuals = np.linalg.norm(predicted - delta_ee, axis=1)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    condition = float(np.linalg.cond(delta_pixels.T @ delta_pixels))
    offset = np.mean(ee_xy - (matrix @ pixels.T).T, axis=0)
    return matrix, offset, rmse, condition, residuals


def fit_layer_robust(
        ee_xy,
        pixels,
        max_rmse=DEFAULT_MAX_FIT_RMSE_M,
        min_samples=MIN_LAYER_SAMPLES):
    if max_rmse <= 0.0 or min_samples < MIN_LAYER_SAMPLES:
        raise ValueError('fit thresholds are invalid')
    if len(ee_xy) < min_samples:
        return None
    active_indices = list(range(len(ee_xy)))
    rejected_indices = []
    while len(active_indices) >= min_samples:
        result = fit_layer(
            np.asarray(ee_xy)[active_indices],
            np.asarray(pixels)[active_indices],
        )
        if result is None:
            return None
        matrix, offset, rmse, condition, _ = result
        if rmse <= max_rmse or len(active_indices) == min_samples:
            return LayerFit(
                matrix=matrix,
                offset=offset,
                rmse_m=rmse,
                condition_number=condition,
                inlier_indices=tuple(active_indices),
                rejected_indices=tuple(rejected_indices),
            )
        removal_scores = []
        for local_index in range(len(active_indices)):
            candidate_indices = active_indices[:local_index] + active_indices[local_index + 1:]
            candidate_fit = fit_layer(
                np.asarray(ee_xy)[candidate_indices],
                np.asarray(pixels)[candidate_indices],
            )
            candidate_rmse = float('inf') if candidate_fit is None else candidate_fit[2]
            removal_scores.append(candidate_rmse)
        rejected_local_index = int(np.argmin(removal_scores))
        rejected_indices.append(active_indices.pop(rejected_local_index))
    return None


def calibration_from_document(
        document,
        max_fit_rmse=DEFAULT_MAX_FIT_RMSE_M,
        max_condition_number=DEFAULT_MAX_CONDITION_NUMBER):
    if not isinstance(document, dict):
        raise ValueError('calibration document must be a mapping')
    if document.get('schema_version', CALIBRATION_SCHEMA_VERSION) != CALIBRATION_SCHEMA_VERSION:
        raise ValueError('unsupported calibration schema_version')
    if document.get('calibration_type') != 'eye_in_hand_layered_xy':
        raise ValueError('unsupported calibration_type')
    z_layers = np.asarray(document['z_layers'], dtype=np.float64)
    if z_layers.ndim != 1 or len(z_layers) == 0 or not np.all(np.isfinite(z_layers)):
        raise ValueError('z_layers must be a non-empty finite vector')
    if len(z_layers) > 1 and not np.all(np.diff(z_layers) > 0.0):
        raise ValueError('z_layers must be strictly increasing')
    layer_matrices = _finite_array(
        document['layer_matrices'],
        (len(z_layers), 2, 2),
        'layer_matrices',
    )
    layer_offsets = _finite_array(
        document['layer_offsets'],
        (len(z_layers), 2),
        'layer_offsets',
    )
    middle_index = len(z_layers) // 2
    pixel_to_base_delta = _finite_array(
        document.get('pixel_to_base_delta', layer_matrices[middle_index]),
        (2, 2),
        'pixel_to_base_delta',
    )
    base_xy_offset = _finite_array(
        document.get('base_xy_offset', layer_offsets[middle_index]),
        (2,),
        'base_xy_offset',
    )
    verified_affine_data = document.get('verified_pixel_to_base_affine')
    verified_affine = (
        None
        if verified_affine_data is None
        else _finite_array(
            verified_affine_data,
            (2, 3),
            'verified_pixel_to_base_affine',
        )
    )
    image_center = _finite_array(document['image_center'], (2,), 'image_center')
    tool_offset = _finite_array(
        document.get('tool_to_camera_xy', [0.0, 0.0]),
        (2,),
        'tool_to_camera_xy',
    )
    known_target_xy = _finite_array(
        document.get('known_target_xy', WS712_OUTPUT_ANCHOR_XY),
        (2,),
        'known_target_xy',
    )
    output_anchor_xy = _finite_array(
        document.get('output_anchor_xy', WS712_OUTPUT_ANCHOR_XY),
        (2,),
        'output_anchor_xy',
    )
    output_correction_matrix = _finite_array(
        document.get(
            'output_correction_matrix',
            DEFAULT_OUTPUT_CORRECTION_MATRIX,
        ),
        (2, 2),
        'output_correction_matrix',
    )
    output_offset_xy = _finite_array(
        document.get('output_offset_xy', DEFAULT_OUTPUT_OFFSET_XY),
        (2,),
        'output_offset_xy',
    )
    workspace = document['workspace']
    if not isinstance(workspace, dict):
        raise ValueError('workspace must be a mapping')
    if 'min' in workspace and 'max' in workspace:
        legacy_min = _finite_array(workspace['min'], (3,), 'workspace.min')
        legacy_max = _finite_array(workspace['max'], (3,), 'workspace.max')
        if not np.all(legacy_min < legacy_max):
            raise ValueError('workspace min must be lower than max')
        calibration_workspace = {
            'min': legacy_min.tolist(),
            'max': legacy_max.tolist(),
        }
        runtime_workspace = document.get(
            'runtime_workspace',
            calibration_workspace,
        )
    else:
        calibration_workspace = {}
        for axis in ('x', 'y', 'z'):
            calibration_workspace[axis] = _finite_range(
                workspace[axis],
                f'workspace.{axis}',
            ).tolist()
        runtime_workspace = document.get('runtime_workspace', {
            'min': DEFAULT_RUNTIME_WORKSPACE_MIN,
            'max': DEFAULT_RUNTIME_WORKSPACE_MAX,
        })
    if not isinstance(runtime_workspace, dict):
        raise ValueError('runtime_workspace must be a mapping')
    workspace_min = _finite_array(
        runtime_workspace['min'],
        (3,),
        'runtime_workspace.min',
    )
    workspace_max = _finite_array(
        runtime_workspace['max'],
        (3,),
        'runtime_workspace.max',
    )
    if not np.all(workspace_min < workspace_max):
        raise ValueError('runtime workspace min must be lower than max')
    sampled_layer_y_ranges_data = document.get('sampled_layer_y_ranges', [])
    if not isinstance(sampled_layer_y_ranges_data, list):
        raise ValueError('sampled_layer_y_ranges must be a list')
    sampled_layer_y_ranges = tuple(
        tuple(
            float(value)
            for value in _finite_range(
                values,
                'sampled_layer_y_ranges item',
            )
        )
        for values in sampled_layer_y_ranges_data
    )
    quality = document['layer_quality']
    if not isinstance(quality, list) or len(quality) != len(z_layers):
        raise ValueError('layer_quality must match z_layers')
    validated_quality = []
    for index, item in enumerate(quality):
        if not isinstance(item, dict):
            raise ValueError(f'layer_quality[{index}] must be a mapping')
        rmse = float(item['fit_rmse_m'])
        condition = float(item['condition_number'])
        sample_count = item.get(
            'fitted_sample_count',
            item.get('fitted_count'),
        )
        if (
            not isinstance(sample_count, int)
            or isinstance(sample_count, bool)
            or sample_count < 0
        ):
            raise ValueError(
                f'layer_quality[{index}] fitted sample count is invalid'
            )
        if not np.isfinite(rmse) or rmse < 0.0 or rmse > max_fit_rmse:
            raise ValueError(f'layer_quality[{index}] fit RMSE is untrusted')
        if not np.isfinite(condition) or condition <= 0.0 or condition > max_condition_number:
            raise ValueError(f'layer_quality[{index}] condition number is untrusted')
        if sample_count < MIN_LAYER_SAMPLES:
            raise ValueError(f'layer_quality[{index}] has too few samples')
        for count_key in (
                'collected_count', 'fitted_count',
                'fitted_sample_count', 'rejected_count'):
            if count_key not in item:
                continue
            count_value = item[count_key]
            if (
                not isinstance(count_value, int)
                or isinstance(count_value, bool)
                or count_value < 0
            ):
                raise ValueError(
                    f'layer_quality[{index}].{count_key} is invalid'
                )
        if 'z_m' in item:
            z_m = float(item['z_m'])
            if not np.isfinite(z_m):
                raise ValueError(f'layer_quality[{index}].z_m is invalid')
        validated_quality.append(deepcopy(item))
    metadata = document.get('metadata', {})
    if not isinstance(metadata, dict):
        raise ValueError('metadata must be a mapping')
    fit_statistics = {}
    for key in (
            'collected_sample_count', 'fitted_sample_count',
            'fit_outlier_count', 'sample_count', 'skipped_count'):
        if key not in document:
            continue
        value = document[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f'{key} must be a non-negative integer')
        fit_statistics[key] = value
    for key in ('fit_outliers', 'skipped_poses'):
        if key not in document:
            continue
        value = document[key]
        if not isinstance(value, list):
            raise ValueError(f'{key} must be a list')
        fit_statistics[key] = deepcopy(value)
    return HandeyeCalibration(
        z_layers=z_layers,
        layer_matrices=layer_matrices,
        layer_offsets=layer_offsets,
        image_center=image_center,
        tool_to_camera_xy=tool_offset,
        workspace_min=workspace_min,
        workspace_max=workspace_max,
        layer_quality=tuple(validated_quality),
        metadata=dict(metadata),
        verified_pixel_to_base_affine=verified_affine,
        pixel_to_base_delta=pixel_to_base_delta,
        base_xy_offset=base_xy_offset,
        known_target_xy=known_target_xy,
        output_anchor_xy=output_anchor_xy,
        output_correction_matrix=output_correction_matrix,
        output_offset_xy=output_offset_xy,
        calibration_workspace=calibration_workspace,
        sampled_layer_y_ranges=sampled_layer_y_ranges,
        fit_statistics=fit_statistics,
    )


def load_calibration(path, **validation_options):
    target = Path(path).expanduser()
    with target.open('r', encoding='utf-8') as source:
        document = yaml.safe_load(source)
    return calibration_from_document(document, **validation_options)


def reload_calibration(path, current=None, **validation_options):
    try:
        return load_calibration(path, **validation_options), None
    except (KeyError, OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        return current, str(exc)


def save_calibration_atomic(path, calibration):
    validated = calibration_from_document(calibration.to_document())
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor = None
    temp_path = None
    try:
        file_descriptor, temp_name = tempfile.mkstemp(
            dir=str(target.parent),
            prefix=f'.{target.name}.',
            suffix='.tmp',
        )
        temp_path = Path(temp_name)
        with os.fdopen(file_descriptor, 'w', encoding='utf-8') as output:
            file_descriptor = None
            yaml.safe_dump(
                validated.to_document(),
                output,
                allow_unicode=True,
                sort_keys=False,
            )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp_path, target)
        temp_path = None
        directory_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
