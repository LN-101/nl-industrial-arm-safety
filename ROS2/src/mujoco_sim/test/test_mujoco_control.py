from __future__ import annotations

from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

from mujoco_sim.mujoco_sim import (
    DEFAULT_MAX_JOINT_SPEED,
    MujocoJointPublisher,
    rate_limit_joint_control,
    SimulationStepScheduler,
)
import numpy as np


class MujocoControlTest(unittest.TestCase):
    def test_calibration_owner_selects_dedicated_control_topic(self):
        node = MujocoJointPublisher.__new__(MujocoJointPublisher)
        node.viewer_active = False
        node.viewer = None
        node.handeye_active = True
        node.target_ctrl = np.zeros(6)
        node.joint_names = [f'j{index}_joint' for index in range(1, 7)]
        node.get_logger = lambda: type(
            'Logger',
            (),
            {'warn': lambda self, message: None},
        )()
        message = type('JointState', (), {
            'name': [],
            'position': [0.1] * 6,
        })()

        node.joint_control_callback(message)
        np.testing.assert_array_equal(node.target_ctrl, np.zeros(6))
        node.handeye_control_callback(message)
        np.testing.assert_allclose(node.target_ctrl, [0.1] * 6)

    def test_rate_limit_caps_each_joint_per_tick(self):
        commanded = np.zeros(6, dtype=np.float64)
        target = np.array([1.0, -1.0, 0.005, -0.005, 0.5, -0.5])

        result = rate_limit_joint_control(
            commanded,
            target,
            DEFAULT_MAX_JOINT_SPEED,
            0.01,
        )

        np.testing.assert_allclose(
            result,
            [0.01, -0.01, 0.005, -0.005, 0.01, -0.01],
        )

    def test_rate_limit_rejects_non_positive_speed(self):
        with self.assertRaises(ValueError):
            rate_limit_joint_control(np.zeros(6), np.ones(6), 0.0, 0.01)

    def test_step_scheduler_avoids_non_integer_timestep_drift(self):
        scheduler = SimulationStepScheduler(0.01, 0.003)

        step_counts = [scheduler.next_step_count() for _ in range(30)]

        self.assertEqual(sum(step_counts), 100)
        self.assertAlmostEqual(scheduler.remaining_time, 0.0, places=12)

    def test_joint_name_reordering_targets_canonical_order(self):
        joint_names = [f'j{index}_joint' for index in range(1, 7)]
        message_names = list(reversed(joint_names))
        message_positions = list(range(6))
        name_to_position = dict(zip(message_names, message_positions))

        ordered = [name_to_position[name] for name in joint_names]

        self.assertEqual(ordered, [5, 4, 3, 2, 1, 0])

    def test_mjcf_uses_authoritative_actuator_parameters(self):
        mjcf_path = (
            Path(__file__).parents[2]
            / 'arm_asset'
            / 'mjcf'
            / 'arm_mjcf.xml'
        )
        root = ET.parse(mjcf_path).getroot()
        joints = root.findall('.//joint')
        actuators = root.findall('./actuator/position')

        self.assertEqual(len(joints), 6)
        self.assertEqual(len(actuators), 6)
        self.assertTrue(all(
            joint.get('actuatorfrcrange') == '-300 300'
            for joint in joints
        ))
        for actuator in actuators:
            self.assertEqual(actuator.get('kp'), '800')
            self.assertEqual(actuator.get('kv'), '60')
            self.assertEqual(actuator.get('forcerange'), '-300 300')


if __name__ == '__main__':
    unittest.main()
