from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from control.handeye_calibration_io import (
    calibration_from_document,
    fit_layer_robust,
    load_calibration,
    reload_calibration,
    save_calibration_atomic,
)
import numpy as np
import yaml


CONFIG_PATH = Path(__file__).parents[1] / 'config' / 'handeye_xy.yaml'
WS712_ROOT_FIELDS = {
    'calibration_type': 'eye_in_hand_layered_xy',
    'verified_pixel_to_base_affine': [
        [-0.000330057522, 0.0000168016813, 0.276025283],
        [-0.0000326029527, -0.000226905888, 0.460851696],
    ],
    'pixel_to_base_delta': [
        [0.00019789440480462867, 2.1425139146954032e-05],
        [-9.579321558853683e-06, 0.00010392996284165016],
    ],
    'z_layers': [0.3],
    'layer_matrices': [[
        [0.00019789440480462867, 2.1425139146954032e-05],
        [-9.579321558853683e-06, 0.00010392996284165016],
    ]],
    'base_xy_offset': [0.027893982209114, 0.09695701268516971],
    'layer_offsets': [[0.027893982209114, 0.09695701268516971]],
    'known_target_xy': [0.08, 0.33],
    'output_anchor_xy': [0.08, 0.33],
    'output_correction_matrix': [
        [2.22035638, -0.55975371],
        [4.71126038, -3.28855304],
    ],
    'output_offset_xy': [-0.0112, 0.0783],
    'image_center': [540.0, 360.0],
    'tool_to_camera_xy': [0.0, 0.0],
    'workspace': {
        'x': [0.06, 0.1],
        'y': [0.235, 0.25],
        'z': [0.3, 0.3],
    },
    'sampled_layer_y_ranges': [[0.235, 0.25]],
    'layer_quality': [{
        'z_m': 0.3,
        'fit_rmse_m': 0.0018173612703451718,
        'condition_number': 3.7645810056246516,
        'collected_count': 9,
        'fitted_count': 9,
        'rejected_count': 0,
    }],
    'collected_sample_count': 9,
    'fitted_sample_count': 9,
    'fit_outlier_count': 0,
    'fit_outliers': [],
    'sample_count': 9,
    'skipped_count': 0,
    'skipped_poses': [],
}


def calibration_document():
    return {
        'schema_version': 1,
        'calibration_type': 'eye_in_hand_layered_xy',
        'z_layers': [0.1, 0.2],
        'layer_matrices': [
            [[0.001, 0.0], [0.0, 0.001]],
            [[0.002, 0.0], [0.0, 0.002]],
        ],
        'layer_offsets': [[0.0, 0.0], [0.01, 0.02]],
        'image_center': [540.0, 360.0],
        'tool_to_camera_xy': [0.0, 0.0],
        'workspace': {'min': [-0.32, -0.32, 0.05], 'max': [0.38, 0.38, 0.30]},
        'layer_quality': [
            {'fit_rmse_m': 0.001, 'condition_number': 5.0, 'fitted_sample_count': 9},
            {'fit_rmse_m': 0.002, 'condition_number': 6.0, 'fitted_sample_count': 8},
        ],
        'metadata': {'urdf_sha256': 'test'},
    }


class HandeyeCalibrationIoTest(unittest.TestCase):
    def test_ws712_root_parameters_are_all_copied_exactly(self):
        document = yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8'))

        copied_fields = {
            key: document[key]
            for key in WS712_ROOT_FIELDS
        }

        self.assertEqual(copied_fields, WS712_ROOT_FIELDS)
        self.assertEqual(document['schema_version'], 1)
        self.assertEqual(document['runtime_workspace'], {
            'min': [-0.32, -0.32, 0.05],
            'max': [0.38, 0.38, 0.30],
        })
        self.assertEqual(
            document['metadata']['source_sha256'],
            '252119e2a30c5f65b1778aae66ddc13a7294606db4fbd751a69d0eeb18634990',
        )

    def test_ws712_verified_affine_has_runtime_priority(self):
        calibration = load_calibration(CONFIG_PATH)
        pixel = np.array([540.0, 360.0])

        point = calibration.pixel_to_base(pixel, None)

        expected = np.asarray(
            WS712_ROOT_FIELDS['verified_pixel_to_base_affine'],
        ) @ np.array([540.0, 360.0, 1.0])
        np.testing.assert_allclose(point, expected, rtol=0.0, atol=1e-15)

    def test_ws712_sample_and_runtime_workspaces_stay_distinct(self):
        calibration = load_calibration(CONFIG_PATH)

        self.assertEqual(
            calibration.calibration_workspace,
            WS712_ROOT_FIELDS['workspace'],
        )
        np.testing.assert_array_equal(
            calibration.workspace_min,
            [-0.32, -0.32, 0.05],
        )
        np.testing.assert_array_equal(
            calibration.workspace_max,
            [0.38, 0.38, 0.30],
        )

    def test_ws712_full_document_survives_atomic_round_trip(self):
        calibration = load_calibration(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'handeye.yaml'
            save_calibration_atomic(path, calibration)
            round_trip = yaml.safe_load(path.read_text(encoding='utf-8'))

        copied_fields = {
            key: round_trip[key]
            for key in WS712_ROOT_FIELDS
        }
        self.assertEqual(copied_fields, WS712_ROOT_FIELDS)

    def test_layer_interpolation_and_transform(self):
        calibration = calibration_from_document(calibration_document())

        matrix, offset = calibration.mapping_at_z(0.15)
        point = calibration.pixel_to_base([550.0, 370.0], [0.1, 0.2, 0.15])

        np.testing.assert_allclose(matrix, np.eye(2) * 0.0015)
        np.testing.assert_allclose(offset, [0.005, 0.01])
        np.testing.assert_allclose(point, [0.09, 0.195])

    def test_dynamic_mapping_applies_explicit_output_correction(self):
        document = calibration_document()
        document['output_anchor_xy'] = [0.08, 0.33]
        document['output_correction_matrix'] = [[2.0, 0.0], [0.0, 3.0]]
        document['output_offset_xy'] = [-0.01, 0.02]
        calibration = calibration_from_document(document)

        point = calibration.pixel_to_base([550.0, 370.0], [0.1, 0.2, 0.15])

        uncorrected = np.array([0.09, 0.195])
        expected = (
            np.array([0.08, 0.33])
            + np.diag([2.0, 3.0]) @ (uncorrected - [0.08, 0.33])
            + [-0.01, 0.02]
        )
        np.testing.assert_allclose(point, expected)

    def test_rejects_nan_shape_bad_layers_and_low_quality(self):
        mutations = (
            ('layer_matrices', [[[float('nan'), 0.0], [0.0, 1.0]]] * 2),
            ('z_layers', [0.2, 0.1]),
            (
                'layer_quality',
                [
                    {
                        'fit_rmse_m': 0.01,
                        'condition_number': 1.0,
                        'fitted_sample_count': 9,
                    }
                ] * 2,
            ),
        )
        for key, value in mutations:
            with self.subTest(key=key):
                document = calibration_document()
                document[key] = value
                with self.assertRaises(ValueError):
                    calibration_from_document(document)

    def test_robust_fit_rejects_one_outlier(self):
        pixels = np.array([
            [0.0, 0.0], [10.0, 0.0], [0.0, 10.0],
            [10.0, 10.0], [20.0, 0.0], [0.0, 20.0],
        ])
        ee_xy = pixels * 0.001
        ee_xy[-1] = [1.0, 1.0]

        fit = fit_layer_robust(ee_xy, pixels, max_rmse=0.001)

        self.assertIsNotNone(fit)
        assert fit is not None
        self.assertEqual(fit.rejected_indices, (5,))
        np.testing.assert_allclose(fit.matrix, np.eye(2) * 0.001, atol=1e-12)

    def test_atomic_round_trip_and_no_temp_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'handeye.yaml'
            calibration = calibration_from_document(calibration_document())

            save_calibration_atomic(path, calibration)
            loaded = load_calibration(path)

            np.testing.assert_allclose(loaded.layer_matrices, calibration.layer_matrices)
            self.assertEqual(list(Path(temp_dir).glob('*.tmp')), [])

    def test_bad_reload_preserves_last_valid_calibration(self):
        current = calibration_from_document(calibration_document())
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'handeye.yaml'
            path.write_text('schema_version: 1\nz_layers: [nan]\n', encoding='utf-8')

            loaded, error = reload_calibration(path, current=current)

        self.assertIs(loaded, current)
        self.assertIsNotNone(error)


if __name__ == '__main__':
    unittest.main()
