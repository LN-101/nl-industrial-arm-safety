"""Tests for ROS2 vision context snapshot helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

from camera.vision_context import (
    build_context_payload,
    finite_float_or_none,
    normalize_image_extension,
    write_rgb_snapshot,
)


class VisionContextPayloadTest(unittest.TestCase):
    """Verify the JSON service contract stays AI_ov compatible."""

    def test_payload_contains_required_context_fields(self) -> None:
        """Context payload includes the required image and safety fields."""
        message = build_context_payload(
            image_path=Path('/tmp/snapshot.jpg'),
            stamp='2026-07-05T10:00:00.000Z',
            frame_id='camera_frame',
            min_distance_m=0.42,
            human_closest_point=(0.1, 0.2, 0.7),
            arm_closest_point=(0.3, 0.2, 0.6),
            emergency_stop=False,
            fresh=True,
            age_ms=12.3456,
            source='min_dis',
            width=640,
            height=480,
            reason='person distance safe: 0.420m',
            threshold_m=0.2,
        )

        payload = json.loads(message)
        self.assertEqual(payload['image_path'], '/tmp/snapshot.jpg')
        self.assertEqual(payload['frame_id'], 'camera_frame')
        self.assertEqual(payload['min_distance_m'], 0.42)
        self.assertEqual(payload['human_closest_point']['z'], 0.7)
        self.assertEqual(payload['arm_closest_point']['x'], 0.3)
        self.assertFalse(payload['emergency_stop'])
        self.assertTrue(payload['fresh'])
        self.assertEqual(payload['age_ms'], 12.346)
        self.assertEqual(payload['source'], 'min_dis')
        self.assertEqual(payload['width'], 640)
        self.assertEqual(payload['height'], 480)
        self.assertEqual(payload['threshold_m'], 0.2)

    def test_non_finite_distance_becomes_null(self) -> None:
        """Non-finite numbers are serialized as JSON null, never NaN/inf."""
        message = build_context_payload(
            image_path=Path('/tmp/snapshot.jpg'),
            stamp='2026-07-05T10:00:00.000Z',
            frame_id='camera_frame',
            min_distance_m=math.inf,
            human_closest_point=None,
            arm_closest_point=None,
            emergency_stop=False,
            fresh=True,
            age_ms=1.0,
            source='min_dis',
            width=640,
            height=480,
        )

        payload = json.loads(message)
        self.assertIsNone(payload['min_distance_m'])
        self.assertIsNone(payload['human_closest_point'])
        self.assertIsNone(payload['arm_closest_point'])

    def test_normalize_image_extension_rejects_bad_values(self) -> None:
        """Only the supported image extensions are accepted."""
        self.assertEqual(normalize_image_extension('.JPG'), 'jpg')
        self.assertEqual(normalize_image_extension('png'), 'png')
        with self.assertRaises(ValueError):
            normalize_image_extension('gif')

    def test_finite_float_or_none_rejects_nan_and_inf(self) -> None:
        """Accept finite values and reject JSON-invalid numbers."""
        self.assertEqual(finite_float_or_none('0.5'), 0.5)
        self.assertIsNone(finite_float_or_none(float('nan')))
        self.assertIsNone(finite_float_or_none(float('inf')))

    def test_write_rgb_snapshot_creates_supported_image_file(self) -> None:
        """The image writer reports success only after a file exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            image = [
                [[255, 0, 0], [255, 0, 0], [255, 0, 0]],
                [[255, 0, 0], [255, 0, 0], [255, 0, 0]],
            ]

            output_path = write_rgb_snapshot(
                image,
                Path(temp_dir) / 'snapshot',
                image_extension='jpg',
                jpeg_quality=90,
            )

            self.assertEqual(output_path.suffix, '.jpg')
            self.assertTrue(output_path.is_file())


if __name__ == '__main__':
    unittest.main()
