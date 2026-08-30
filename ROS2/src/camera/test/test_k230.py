"""Tests for the K230 serial goal bridge helpers."""

from __future__ import annotations

import unittest

from camera.k230 import (
    GOAL_TYPES,
    RobotJointPublisher,
    parse_k230_frame,
    pixel_goal_to_point,
)


class K230GoalBridgeTest(unittest.TestCase):
    """Verify K230 frame parsing and goal coordinate conversion."""

    def test_parse_typed_frame(self) -> None:
        """A typed K230 frame yields a goal type and pixel coordinates."""
        parsed = parse_k230_frame(b'EA,320,240P')

        self.assertEqual(parsed, ('A', [320.0, 240.0]))

    def test_parse_rejects_malformed_frame(self) -> None:
        """Malformed or non-finite frames are rejected."""
        self.assertIsNone(parse_k230_frame(b'bad'))
        self.assertIsNone(parse_k230_frame(b'EA,nan,240P'))
        self.assertIsNone(parse_k230_frame(b'EA,320P'))

    def test_pixel_goal_to_point_uses_existing_calibration(self) -> None:
        """The image center maps to the existing arm-space calibration."""
        point = pixel_goal_to_point([320.0, 240.0])

        self.assertIsNotNone(point)
        assert point is not None
        self.assertAlmostEqual(point.x, 0.08)
        self.assertAlmostEqual(point.y, 0.3)
        self.assertAlmostEqual(point.z, 0.1)

    def test_pixel_goal_to_point_rejects_zero_goal(self) -> None:
        """A zero pixel goal means no valid target."""
        self.assertIsNone(pixel_goal_to_point([0.0, 0.0]))

    def test_handle_frame_updates_cache_without_auto_publish(self) -> None:
        """Serial frames only cache targets; /goal_type triggers publishing."""
        node = RobotJointPublisher.__new__(RobotJointPublisher)
        node.goal_pos = {goal_type: [0.0, 0.0] for goal_type in GOAL_TYPES}
        node.get_logger = lambda: _Logger()

        node.handle_frame(b'EA,320,240P')

        self.assertEqual(node.goal_pos['A'], [320.0, 240.0])


class _Logger:
    """Minimal logger for methods tested without constructing a ROS node."""

    def warn(self, _message: str) -> None:
        """Ignore warning text in unit tests."""
        return None


if __name__ == '__main__':
    unittest.main()
