from __future__ import annotations

import json
import unittest
from unittest import mock

from control import ik_control
import numpy as np


class _Logger:
    def info(self, _message):
        return None

    def warn(self, _message):
        return None

    def error(self, _message):
        return None


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def make_controller():
    controller = ik_control.ArmIKController.__new__(ik_control.ArmIKController)
    controller.goal = np.zeros(3, dtype=np.float64)
    controller.current_joints = np.zeros(6, dtype=np.float64)
    controller.has_joint_state = False
    controller.stats = {'total': 0, 'success': 0, 'collision': 0, 'failure': 0}
    controller.workspace = {
        'x': (-0.22, 0.38),
        'y': (-0.22, 0.38),
        'z': (0.05, 0.33),
    }
    controller.j4_movement_history = []
    controller.last_j4 = None
    controller.joint_pub = _Publisher()
    controller.handeye_joint_pub = _Publisher()
    controller.ik_success_pub = _Publisher()
    controller.handeye_result_pub = _Publisher()
    controller.handeye_active = False
    controller.handeye_request_seq = None
    controller.last_handeye_seq = -1
    controller.get_logger = lambda: _Logger()
    return controller


class IkContractTest(unittest.TestCase):
    def test_calibration_owner_rejects_plain_goal(self):
        controller = make_controller()
        controller.handeye_active = True
        message = type('Point', (), {'x': 0.1, 'y': 0.1, 'z': 0.1})()

        with mock.patch.object(controller, 'ik_control') as solve:
            controller.goal_callback(message)

        solve.assert_not_called()

    def test_handeye_result_preserves_request_sequence(self):
        controller = make_controller()
        controller.handeye_active = True
        request = type('String', (), {
            'data': json.dumps({'seq': 7, 'x': 0.1, 'y': 0.1, 'z': 0.1}),
        })()

        def publish_success():
            controller.publish_ik_result(True)

        with mock.patch.object(controller, 'ik_control', side_effect=publish_success):
            controller.handeye_goal_callback(request)

        result = json.loads(controller.handeye_result_pub.messages[0].data)
        self.assertEqual(result, {'reason': 'ok', 'seq': 7, 'success': True})

    def test_handeye_control_uses_dedicated_topic(self):
        controller = make_controller()
        controller.handeye_request_seq = 3
        candidate = np.zeros(6, dtype=np.float64)
        with mock.patch.object(
                ik_control,
                'multi_start_ik',
                return_value=(candidate, 0.0)), \
                mock.patch.object(ik_control, 'position_error', return_value=0.009), \
                mock.patch.object(
                    ik_control,
                    'fk_pose',
                    return_value=(np.zeros(3), np.array([0.0, 0.0, 1.0])),
                ), \
                mock.patch.object(
                    ik_control,
                    'check_self_collision',
                    return_value=(False, ''),
                ):
            controller.get_clock = lambda: type('Clock', (), {
                'now': lambda self: type('Now', (), {'to_msg': lambda self: None})(),
            })()
            controller.ik_control()

        self.assertEqual(controller.joint_pub.messages, [])
        self.assertEqual(len(controller.handeye_joint_pub.messages), 1)

    def test_authoritative_tolerances_and_workspace(self):
        controller = make_controller()

        self.assertEqual(ik_control.IK_POSITION_TOL, 0.01)
        self.assertAlmostEqual(np.degrees(ik_control.IK_ORIENTATION_TOL), 10.0)
        self.assertEqual(ik_control.PINK_POSITION_COST, 50.0)
        self.assertEqual(ik_control.PINK_ORIENTATION_COST, 5.0)
        self.assertEqual(ik_control.PINK_LM_DAMPING, 1e-4)
        self.assertEqual(ik_control.PINK_POSTURE_COST, 0.05)
        self.assertEqual(ik_control.COMBINED_POSITION_WEIGHT, 1.0)
        self.assertEqual(ik_control.COMBINED_ORIENTATION_WEIGHT, 0.5)
        self.assertEqual(ik_control.SOLUTION_J4_ABSOLUTE_WEIGHT, 0.5)
        self.assertEqual(ik_control.J4_PENALTY_WEIGHT, 0.2)
        self.assertEqual(ik_control.IK_RESCUE_POSTURE_WEIGHT, 0.02)
        self.assertEqual(ik_control.IK_RESCUE_NEUTRAL_WEIGHT, 0.01)
        self.assertEqual(ik_control.IK_RESCUE_J4_WEIGHT, 0.05)
        self.assertEqual(ik_control.IK_RESCUE_LIMIT_WEIGHT, 0.03)
        self.assertEqual(controller.workspace['x'], (-0.22, 0.38))
        self.assertEqual(controller.workspace['y'], (-0.22, 0.38))
        self.assertEqual(controller.workspace['z'], (0.05, 0.33))

    def test_feasible_solution_sorts_before_infeasible_solution(self):
        reference = np.zeros(6, dtype=np.float64)
        feasible = np.ones(6, dtype=np.float64)
        infeasible = np.zeros(6, dtype=np.float64)

        def position_error(candidate, _target, _data=None):
            return 0.009 if candidate is feasible else 0.011

        with mock.patch.object(ik_control, 'position_error', side_effect=position_error), \
                mock.patch.object(ik_control, 'orientation_error', return_value=0.0):
            feasible_key = ik_control.solution_sort_key(feasible, np.zeros(3), reference, 0.008)
            infeasible_key = ik_control.solution_sort_key(
                infeasible,
                np.zeros(3),
                reference,
                0.008,
            )

        self.assertLess(feasible_key, infeasible_key)

    def test_invalid_pink_solution_uses_orientation_rescue(self):
        rescued = np.full(6, 0.1, dtype=np.float64)
        with mock.patch.object(ik_control, 'PINK_SOLVER', {'qp_solver': 'test'}), \
                mock.patch.object(ik_control, 'make_seed_configs', return_value=[np.zeros(6)]), \
                mock.patch.object(ik_control, 'solve_ik_pink', return_value=(np.zeros(6), 0.02)), \
                mock.patch.object(ik_control, 'orientation_error', return_value=0.0), \
                mock.patch.object(
                    ik_control,
                    'solve_ik_orientation_rescue',
                    return_value=(rescued, 0.005),
                ) as rescue:
            result, error = ik_control.multi_start_ik(np.array([0.1, 0.1, 0.1]))

        np.testing.assert_array_equal(result, rescued)
        self.assertEqual(error, 0.005)
        rescue.assert_called_once()

    def test_out_of_workspace_goal_publishes_failure(self):
        controller = make_controller()
        message = type('Point', (), {'x': 0.39, 'y': 0.0, 'z': 0.1})()

        controller.goal_callback(message)

        self.assertEqual([result.data for result in controller.ik_success_pub.messages], [False])
        self.assertEqual(controller.joint_pub.messages, [])

    def test_position_and_orientation_must_both_pass(self):
        controller = make_controller()
        candidate = np.zeros(6, dtype=np.float64)
        with mock.patch.object(ik_control, 'multi_start_ik', return_value=(candidate, 0.0)), \
                mock.patch.object(ik_control, 'position_error', return_value=0.009), \
                mock.patch.object(
                    ik_control,
                    'fk_pose',
                    return_value=(
                        np.zeros(3),
                        np.array([
                            0.0,
                            np.sin(np.deg2rad(11.0)),
                            np.cos(np.deg2rad(11.0)),
                        ]),
                    ),
                ):
            controller.ik_control()

        self.assertEqual([result.data for result in controller.ik_success_pub.messages], [False])
        self.assertEqual(controller.joint_pub.messages, [])

    def test_collision_publishes_failure(self):
        controller = make_controller()
        candidate = np.zeros(6, dtype=np.float64)
        with mock.patch.object(
                ik_control,
                'multi_start_ik',
                return_value=(candidate, 0.0)), \
                mock.patch.object(ik_control, 'position_error', return_value=0.009), \
                mock.patch.object(
                    ik_control,
                    'fk_pose',
                    return_value=(np.zeros(3), np.array([0.0, 0.0, 1.0])),
                ), \
                mock.patch.object(
                    ik_control,
                    'check_self_collision',
                    return_value=(True, 'collision'),
                ):
            controller.ik_control()

        self.assertEqual([result.data for result in controller.ik_success_pub.messages], [False])
        self.assertEqual(controller.joint_pub.messages, [])


if __name__ == '__main__':
    unittest.main()
