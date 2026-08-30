from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from main.arm_state import (
    ARM_FRAME_HEADER,
    ARM_FRAME_TAIL,
    DEFAULT_ARM_RULES,
    default_handeye_calibration_path,
    DEFAULT_PERSONNEL_DISTANCE_M,
    DEFAULT_SAFETY_RULES_VERSION,
    GOAL_HEIGHT,
    GOAL_TYPES,
    K230_AFFINE_X,
    K230_AFFINE_Y,
    K230_FRAME_HEADER,
    K230_FRAME_TAIL,
    k230_pixel_goal_to_arm_point,
    LOOK_JOINT_POSITIONS,
    MIDDLE_JOINT_POSITIONS,
    parse_arm_joint_frame,
    parse_k230_goal_frame,
    PICK_COMMAND,
    PLACE_COMMAND,
    RECOVER_COMMAND,
    reset_personnel_distance_rule,
    RobotJointPublisher,
    SECOND_MIDDLE_JOINT_POSITIONS,
    STOP_COMMAND,
    UNIFIED_FIELD_NOOP,
    UNIFIED_WATCHDOG_OK,
    WORKSPACE_BOUNDS,
)
import numpy as np
import serial


class _Logger:
    def info(self, _message: str) -> None:
        return None

    def warn(self, _message: str) -> None:
        return None

    def error(self, _message: str) -> None:
        return None


class _FakePublisher:
    def __init__(self) -> None:
        self.messages = []

    def publish(self, message) -> None:
        self.messages.append(message)


class _FakeSerial:
    def __init__(self, fail: bool = False) -> None:
        self.is_open = True
        self.fail = fail
        self.writes = []

    def write(self, payload: bytes) -> None:
        if self.fail:
            raise serial.SerialException('serial write failure')
        self.writes.append(payload.decode('utf-8'))

    def flush(self) -> None:
        return None


class _BoolMsg:
    def __init__(self, data: bool) -> None:
        self.data = data


class _PointMsg:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


class _JointMsg:
    def __init__(self, position) -> None:
        self.position = position


class _RecordingEvent:
    # 伪 shutdown_event：wait 时记录当前 pending 关节，立即返回（不中断）

    def __init__(self, node) -> None:
        self.node = node
        self.snapshots = []
        self.waits = []

    def wait(self, timeout=None) -> bool:
        self.waits.append(timeout)
        pending = self.node.pending_joint_positions
        self.snapshots.append(None if pending is None else list(pending))
        return False

    def is_set(self) -> bool:
        return False


class _IkSuccessSleepStub:
    # 伪 sleep_until_shutdown：记录时长，并模拟等待期收到 /ik_success=True

    def __init__(self, node) -> None:
        self.node = node
        self.calls = []

    def __call__(self, seconds: float, restart_after_pause=True) -> bool:
        self.calls.append(seconds)
        self.node.ik_success_flag = True
        return False


def make_protocol_node(fail_serial: bool = False) -> RobotJointPublisher:
    node = RobotJointPublisher.__new__(RobotJointPublisher)
    node.get_logger = lambda: _Logger()
    node.ser = _FakeSerial(fail=fail_serial)
    node.serial_lock = threading.Lock()
    node.shutdown_event = threading.Event()
    node.stop_flag = False
    node.recover_pending = False
    node.capture_resume_condition = threading.Condition()
    node.capture_resume_snapshot = None
    node.capture_resume_generation = 0
    node.capture_pause_generation = 0
    node.capture_pause_started_at = None
    node.capture_pause_total_seconds = 0.0
    node.capture_in_progress = False
    node.ARM_FRAME_HEADER = ARM_FRAME_HEADER
    node.ARM_FRAME_TAIL = ARM_FRAME_TAIL
    node.K230_FRAME_HEADER = K230_FRAME_HEADER
    node.K230_FRAME_TAIL = K230_FRAME_TAIL
    node.pending_joint_positions = None
    node.pending_stop = UNIFIED_FIELD_NOOP
    node.pending_pump = UNIFIED_FIELD_NOOP
    node.pending_speed = UNIFIED_FIELD_NOOP
    node.pending_ok = UNIFIED_WATCHDOG_OK
    node.last_sent_command = None
    return node


def make_json_node(fail_serial: bool = False) -> RobotJointPublisher:
    node = make_protocol_node(fail_serial=fail_serial)
    node.config_path = None
    node.safety_distance_pub = _FakePublisher()
    node.arm_capture = False
    node.arm_capture_goal = 'A'
    node.arm_stop = False
    node.arm_recover = False
    node.arm_decelerate_percent = 0
    node.arm_safety_distance = 0.2
    node.previous_rule_commands = {
        'arm_capture': False,
        'arm_stop': False,
        'arm_recover': False,
    }
    node.previous_decelerate_percent = None
    return node


def make_sequence_node() -> RobotJointPublisher:
    node = make_protocol_node()
    node.arm_capture_goal = 'A'
    node.goal_height = GOAL_HEIGHT
    node.workspace_bounds = dict(WORKSPACE_BOUNDS)
    node.handeye_mapping_enabled = False
    node.handeye_calibration = None
    node.current_ee_position = None
    node.look_joint_positions = list(LOOK_JOINT_POSITIONS)
    node.joint_pos = np.zeros(6, dtype=np.float64)
    node.goal = np.zeros(3, dtype=np.float64)
    node.goal_pos = {goal_type: [0.0, 0.0] for goal_type in GOAL_TYPES}
    node.ik_success_flag = False
    node.capture_lock = threading.Lock()
    node.capture_thread = None
    node.capture_in_progress = False
    node.handeye_calibration_active = False
    node.capture_success = False
    node.capture_goal_control_lock = threading.Lock()
    node.capture_goal_control_active = False
    node.capture_goal_control_pending = False
    node.capture_goal_control_running = False
    node.capture_goal_control_thread = None
    node.k230_goal_event = threading.Event()
    node.goal_event = threading.Event()
    node.goal_pub = _FakePublisher()
    node.goal_type_pub = _FakePublisher()
    node.arm_capture_pub = _FakePublisher()
    node.capture_status_pub = _FakePublisher()
    node.handeye_d_pub = _FakePublisher()
    node.get_clock = lambda: type('Clock', (), {
        'now': lambda self: type('Now', (), {'to_msg': lambda self: None})(),
    })()
    return node


def status_states(node) -> list:
    return [json.loads(message.data)['state'] for message in node.capture_status_pub.messages]


def status_reasons(node) -> list:
    return [json.loads(message.data).get('reason') for message in node.capture_status_pub.messages]


def is_capture_values(node) -> list:
    return [message.data for message in node.arm_capture_pub.messages]


class ArmStateFrameParsingTest(unittest.TestCase):
    def test_handeye_default_path_uses_installed_control_config(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False), \
                mock.patch(
                    'main.arm_state.get_package_share_directory',
                    return_value='/tmp/control-share',
                ):
            os.environ.pop('AI_OV_HANDEYE_CALIBRATION_FILE', None)
            path = default_handeye_calibration_path()

        self.assertEqual(
            path,
            Path('/tmp/control-share/config/handeye_xy.yaml'),
        )

    def test_authoritative_capture_height_and_mapping_workspace(self) -> None:
        self.assertEqual(GOAL_HEIGHT, 0.105)
        self.assertEqual(WORKSPACE_BOUNDS, {
            'x_min': -0.32,
            'x_max': 0.38,
            'y_min': -0.32,
            'y_max': 0.38,
            'z_min': 0.05,
            'z_max': 0.30,
        })

    def test_parse_arm_joint_frame_accepts_space_or_comma_values(self) -> None:
        self.assertEqual(
            parse_arm_joint_frame(b'A1,2,3 4 5 6B'),
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )

    def test_parse_k230_goal_frame_maps_all_markers(self) -> None:
        parsed = parse_k230_goal_frame(b'E10,20,30,40,50,60,70,80P')

        self.assertEqual(
            parsed,
            {
                'A': [10.0, 20.0],
                'B': [30.0, 40.0],
                'C': [50.0, 60.0],
                'D': [70.0, 80.0],
            },
        )

    def test_parse_k230_goal_frame_rejects_invalid_values(self) -> None:
        self.assertIsNone(parse_k230_goal_frame(b'E10,20,30P'))
        self.assertIsNone(parse_k230_goal_frame(b'E10,nan,30,40,50,60,70,80P'))

    def test_k230_pixel_goal_to_arm_point_uses_published_affine_calibration(self) -> None:
        self.assertEqual(
            K230_AFFINE_X,
            (-0.000300211, 0.000011308, 0.256757),
        )
        self.assertEqual(
            K230_AFFINE_Y,
            (-0.000019886, -0.000271216, 0.460067),
        )
        self.assertAlmostEqual(K230_AFFINE_Y[2] - 0.450067, 0.01)
        point = k230_pixel_goal_to_arm_point([540.0, 360.0])

        self.assertIsNotNone(point)
        assert point is not None
        ax, ay, a0 = K230_AFFINE_X
        bx, by, b0 = K230_AFFINE_Y
        self.assertAlmostEqual(point[0], ax * 540.0 + ay * 360.0 + a0)
        self.assertAlmostEqual(point[1], bx * 540.0 + by * 360.0 + b0)
        self.assertAlmostEqual(point[2], GOAL_HEIGHT)

        custom = k230_pixel_goal_to_arm_point([540.0, 360.0], 0.2)
        assert custom is not None
        self.assertAlmostEqual(custom[2], 0.2)

    def test_k230_pixel_goal_to_arm_point_rejects_empty_goal(self) -> None:
        self.assertIsNone(k230_pixel_goal_to_arm_point([0.0, 0.0]))


class ArmStateRulesResetTest(unittest.TestCase):
    def test_reset_processed_bool_flags_consumes_string_capture_request(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'arm_rules.json'
            node = RobotJointPublisher.__new__(RobotJointPublisher)
            node.config_path = path
            node.get_logger = lambda: _Logger()
            config = {'arm_capture': 'True', 'arm_capture_goal': 'A'}

            node.reset_processed_bool_flags(config, ['arm_capture'])

            self.assertEqual(config['arm_capture'], 'False')
            self.assertEqual(json.loads(path.read_text())['arm_capture'], 'False')

    def test_reset_arm_rules_to_defaults_clears_stale_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'arm_rules.json'
            path.write_text(
                json.dumps(
                    {
                        'arm_capture': 'True',
                        'arm_capture_goal': 'B',
                        'arm_decelerate': '0.4',
                        'arm_stop': 'False',
                        'arm_recover': 'True',
                        'arm_safety_distance': '0.3',
                        'arm_capture_object': '加工件',
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )
            node = RobotJointPublisher.__new__(RobotJointPublisher)
            node.config_path = path
            node.get_logger = lambda: _Logger()

            node.reset_arm_rules_to_defaults()

            self.assertEqual(
                json.loads(path.read_text(encoding='utf-8')),
                DEFAULT_ARM_RULES,
            )

    def test_reset_arm_rules_to_defaults_repairs_corrupted_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'arm_rules.json'
            path.write_text('{"arm', encoding='utf-8')
            node = RobotJointPublisher.__new__(RobotJointPublisher)
            node.config_path = path
            node.get_logger = lambda: _Logger()

            node.reset_arm_rules_to_defaults()

            self.assertEqual(
                json.loads(path.read_text(encoding='utf-8')),
                DEFAULT_ARM_RULES,
            )

    def test_reset_arm_rules_to_defaults_replaces_non_dict_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'arm_rules.json'
            path.write_text('[]', encoding='utf-8')
            node = RobotJointPublisher.__new__(RobotJointPublisher)
            node.config_path = path
            node.get_logger = lambda: _Logger()

            node.reset_arm_rules_to_defaults()

            self.assertEqual(
                json.loads(path.read_text(encoding='utf-8')),
                DEFAULT_ARM_RULES,
            )

    def test_reset_arm_rules_to_defaults_skips_none_config_path(self) -> None:
        node = RobotJointPublisher.__new__(RobotJointPublisher)
        node.config_path = None
        node.get_logger = lambda: _Logger()

        node.reset_arm_rules_to_defaults()

    def test_reset_arm_rules_to_defaults_raises_when_write_fails(self) -> None:
        node = RobotJointPublisher.__new__(RobotJointPublisher)
        node.config_path = Path('/nonexistent-dir/arm_rules.json')
        node.get_logger = lambda: _Logger()

        with self.assertRaises(RuntimeError):
            node.reset_arm_rules_to_defaults()

    def test_reset_arm_rules_to_defaults_syncs_ai_personnel_distance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            arm_path = root / 'arm_rules.json'
            safety_path = root / 'safety_rules.example.json'
            arm_path.write_text('{"arm_capture": "True"}', encoding='utf-8')
            safety_document = _safety_rule_document(distance=0.2, version=6)
            safety_path.write_text(json.dumps(safety_document), encoding='utf-8')
            node = RobotJointPublisher.__new__(RobotJointPublisher)
            node.config_path = arm_path
            node.safety_rules_path = safety_path
            node.get_logger = lambda: _Logger()

            node.reset_arm_rules_to_defaults()

            self.assertEqual(json.loads(arm_path.read_text()), DEFAULT_ARM_RULES)
            updated = json.loads(safety_path.read_text())
            self.assertEqual(updated['version'], DEFAULT_SAFETY_RULES_VERSION)
            self.assertEqual(
                updated['rules'][0]['conditions']['person_distance_m']['lt'],
                DEFAULT_PERSONNEL_DISTANCE_M,
            )
            self.assertEqual(updated['rules'][1], safety_document['rules'][1])

    def test_reset_arm_rules_to_defaults_preserves_ai_document_at_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            arm_path = root / 'arm_rules.json'
            safety_path = root / 'safety_rules.example.json'
            arm_path.write_text('{}', encoding='utf-8')
            original_text = json.dumps(_safety_rule_document(distance=0.4, version=1))
            safety_path.write_text(original_text, encoding='utf-8')
            node = RobotJointPublisher.__new__(RobotJointPublisher)
            node.config_path = arm_path
            node.safety_rules_path = safety_path
            node.get_logger = lambda: _Logger()

            node.reset_arm_rules_to_defaults()

            self.assertEqual(safety_path.read_text(encoding='utf-8'), original_text)

    def test_reset_arm_rules_to_defaults_resets_version_when_distance_is_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            arm_path = root / 'arm_rules.json'
            safety_path = root / 'safety_rules.example.json'
            arm_path.write_text('{}', encoding='utf-8')
            safety_path.write_text(
                json.dumps(_safety_rule_document(distance=0.4, version=6)),
                encoding='utf-8',
            )
            node = RobotJointPublisher.__new__(RobotJointPublisher)
            node.config_path = arm_path
            node.safety_rules_path = safety_path
            node.get_logger = lambda: _Logger()

            node.reset_arm_rules_to_defaults()

            updated = json.loads(safety_path.read_text(encoding='utf-8'))
            self.assertEqual(updated['version'], DEFAULT_SAFETY_RULES_VERSION)
            self.assertEqual(
                updated['rules'][0]['conditions']['person_distance_m']['lt'],
                DEFAULT_PERSONNEL_DISTANCE_M,
            )

    def test_invalid_ai_rules_fail_before_arm_rules_are_reset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            arm_path = root / 'arm_rules.json'
            safety_path = root / 'safety_rules.example.json'
            original_arm_text = '{"arm_capture": "True"}'
            arm_path.write_text(original_arm_text, encoding='utf-8')
            safety_path.write_text('{"version":', encoding='utf-8')
            node = RobotJointPublisher.__new__(RobotJointPublisher)
            node.config_path = arm_path
            node.safety_rules_path = safety_path
            node.get_logger = lambda: _Logger()

            with self.assertRaises(RuntimeError):
                node.reset_arm_rules_to_defaults()

            self.assertEqual(arm_path.read_text(encoding='utf-8'), original_arm_text)

    def test_reset_personnel_distance_rule_rejects_missing_target_rule(self) -> None:
        document = _safety_rule_document(distance=0.4)
        document['rules'] = document['rules'][1:]

        with self.assertRaises(ValueError):
            reset_personnel_distance_rule(document)


def _safety_rule_document(distance, version=1):
    return {
        'version': version,
        'rules': [
            {
                'id': 'stop_on_person_intrusion',
                'enabled': True,
                'conditions': {'person_distance_m': {'lt': distance}},
                'action': {'type': 'stop_motion'},
            },
            {
                'id': 'slow_near_unknown_object',
                'conditions': {'unknown_object_distance_m': {'lt': 0.25}},
            },
        ],
    }


class ArmStateUnifiedFrameProtocolTest(unittest.TestCase):
    def test_stop_callback_sends_estop_frame_synchronously_despite_dedup(self) -> None:
        node = make_protocol_node()
        expected = RobotJointPublisher.build_unified_frame(
            UNIFIED_WATCHDOG_OK, None, STOP_COMMAND, UNIFIED_FIELD_NOOP, UNIFIED_FIELD_NOOP)
        # 预置完全相同的上次发送帧，证明急停不会被任何去重逻辑吞掉
        node.last_sent_command = expected

        node.stop_callback(_BoolMsg(True))

        self.assertTrue(node.stop_flag)
        self.assertEqual(node.ser.writes, [expected])
        # 发送成功后按"发送后复位"规则清除 pending
        self.assertEqual(node.pending_stop, UNIFIED_FIELD_NOOP)

    def test_stop_callback_recover_sends_ef_frame(self) -> None:
        node = make_protocol_node()
        node.stop_flag = True

        node.stop_callback(_BoolMsg(False))

        self.assertFalse(node.stop_flag)
        self.assertEqual(len(node.ser.writes), 1)
        self.assertIn(f',{RECOVER_COMMAND},', node.ser.writes[0])

    def test_stop_callback_keeps_pending_stop_when_serial_write_fails(self) -> None:
        node = make_protocol_node(fail_serial=True)

        node.stop_callback(_BoolMsg(True))

        self.assertTrue(node.stop_flag)
        # 直发失败 -> 命令保留在 pending_stop，由 0.5s 发送拍重试
        self.assertEqual(node.pending_stop, STOP_COMMAND)

    def test_estop_clears_queued_motion_and_pump_commands(self) -> None:
        node = make_protocol_node()
        node.pending_joint_positions = [1, 2, 3, 4, 5, 6]
        node.pending_pump = PICK_COMMAND

        node.stop_callback(_BoolMsg(True))

        # 急停清空已排队的运动/泵命令，且急停帧本身不夹带它们
        self.assertIsNone(node.pending_joint_positions)
        self.assertEqual(node.pending_pump, UNIFIED_FIELD_NOOP)
        self.assertEqual(len(node.ser.writes), 1)
        self.assertIn(f',{STOP_COMMAND},', node.ser.writes[0])
        self.assertNotIn(PICK_COMMAND, node.ser.writes[0])

    def test_send_unified_command_drops_motion_while_estop_active(self) -> None:
        node = make_protocol_node()
        node.stop_flag = True
        node.pending_joint_positions = [1, 2, 3, 4, 5, 6]
        node.pending_pump = PICK_COMMAND

        node.send_unified_command()

        self.assertEqual(
            node.ser.writes,
            [RobotJointPublisher.build_unified_frame(
                UNIFIED_WATCHDOG_OK,
                None,
                UNIFIED_FIELD_NOOP,
                UNIFIED_FIELD_NOOP,
                UNIFIED_FIELD_NOOP,
            )],
        )
        self.assertIsNone(node.pending_joint_positions)
        self.assertEqual(node.pending_pump, UNIFIED_FIELD_NOOP)

    def test_recover_requeues_capture_joint_snapshot_once(self) -> None:
        node = make_protocol_node()
        node.capture_in_progress = True
        target = [1, 2, 3, 4, 5, 6]
        node.begin_capture_resume_stage(
            'middle', 7.0, joint_positions=target)

        node.stop_callback(_BoolMsg(True))
        self.assertIsNone(node.pending_joint_positions)

        node.stop_callback(_BoolMsg(False))

        self.assertEqual(node.pending_joint_positions, target)
        self.assertFalse(node.recover_pending)
        self.assertEqual(len(node.ser.writes), 2)
        node.stop_callback(_BoolMsg(False))
        self.assertEqual(len(node.ser.writes), 2)

    def test_each_estop_cycle_requeues_joint_target_once(self) -> None:
        node = make_protocol_node()
        node.capture_in_progress = True
        target = [1, 2, 3, 4, 5, 6]
        node.begin_capture_resume_stage(
            'middle', 7.0, joint_positions=target)
        node.send_unified_command()

        for _ in range(2):
            node.stop_callback(_BoolMsg(True))
            node.stop_callback(_BoolMsg(False))
            self.assertEqual(node.pending_joint_positions, target)
            node.send_unified_command()
            node.stop_callback(_BoolMsg(False))

        target_frames = [
            frame for frame in node.ser.writes
            if frame.startswith('U,OK,1,2,3,4,5,6,')
        ]
        self.assertEqual(len(target_frames), 3)

    def test_recover_does_not_repeat_sent_capture_pump(self) -> None:
        node = make_protocol_node()
        node.capture_in_progress = True
        node.begin_capture_resume_stage(
            'pick_pump', 2.0, pump_command=PICK_COMMAND)
        node.send_unified_command()

        node.stop_callback(_BoolMsg(True))
        node.stop_callback(_BoolMsg(False))

        self.assertEqual(node.pending_pump, UNIFIED_FIELD_NOOP)
        self.assertTrue(node.capture_resume_snapshot['command_sent'])

    def test_recover_requeues_unsent_capture_pump(self) -> None:
        node = make_protocol_node()
        node.capture_in_progress = True
        node.begin_capture_resume_stage(
            'pick_pump', 2.0, pump_command=PICK_COMMAND)

        node.stop_callback(_BoolMsg(True))
        node.stop_callback(_BoolMsg(False))

        self.assertEqual(node.pending_pump, PICK_COMMAND)

    def test_failed_recover_stays_blocked_until_periodic_retry(self) -> None:
        node = make_protocol_node()
        node.capture_in_progress = True
        target = [1, 2, 3, 4, 5, 6]
        node.begin_capture_resume_stage(
            'middle', 7.0, joint_positions=target)
        node.stop_callback(_BoolMsg(True))

        node.ser.fail = True
        node.stop_callback(_BoolMsg(False))
        self.assertTrue(node.recover_pending)
        self.assertEqual(node.pending_stop, RECOVER_COMMAND)
        self.assertIsNone(node.pending_joint_positions)

        node.ser.fail = False
        node.send_unified_command()

        self.assertFalse(node.recover_pending)
        self.assertEqual(node.pending_stop, UNIFIED_FIELD_NOOP)
        self.assertEqual(node.pending_joint_positions, target)

    def test_capture_wait_restarts_full_duration_after_pause_generation(self) -> None:
        node = make_protocol_node()
        node.capture_in_progress = True
        node.begin_capture_resume_stage(
            'middle', 1.0, joint_positions=[1, 2, 3, 4, 5, 6])
        clock = [0.0]

        class _AdvancingEvent:
            def __init__(self) -> None:
                self.calls = 0

            def is_set(self) -> bool:
                return False

            def wait(self, timeout=None) -> bool:
                self.calls += 1
                clock[0] += float(timeout or 0.0)
                if self.calls == 1:
                    # 模拟一次急停+解除在两次轮询之间完成。
                    node.capture_pause_generation += 1
                return False

        event = _AdvancingEvent()
        node.shutdown_event = event
        with mock.patch(
                'main.arm_state.time.monotonic', side_effect=lambda: clock[0]):
            self.assertFalse(node.sleep_until_shutdown(1.0))

        self.assertAlmostEqual(clock[0], 1.1)
        self.assertGreaterEqual(event.calls, 11)

    def test_capture_pump_wait_preserves_remaining_time_after_pause(self) -> None:
        node = make_protocol_node()
        node.capture_in_progress = True
        node.begin_capture_resume_stage(
            'pick_pump', 1.0, pump_command=PICK_COMMAND)
        clock = [0.0]

        class _AdvancingEvent:
            def __init__(self) -> None:
                self.calls = 0

            def is_set(self) -> bool:
                return False

            def wait(self, timeout=None) -> bool:
                self.calls += 1
                clock[0] += float(timeout or 0.0)
                if self.calls == 1:
                    # 0.1 秒有效流程时间后发生 0.4 秒急停，再完成解除。
                    node.capture_pause_generation += 1
                    node.capture_pause_total_seconds += 0.4
                    clock[0] += 0.4
                return False

        node.shutdown_event = _AdvancingEvent()
        with mock.patch(
                'main.arm_state.time.monotonic', side_effect=lambda: clock[0]):
            self.assertFalse(node.sleep_until_shutdown(
                1.0, restart_after_pause=False))

        # 1.0 秒有效等待 + 0.4 秒急停；没有像关节阶段那样重计完整 1 秒。
        self.assertAlmostEqual(clock[0], 1.4)

    def test_send_unified_command_sends_identical_idle_frames_every_tick(self) -> None:
        node = make_protocol_node()

        node.send_unified_command()
        node.send_unified_command()

        idle_frame = RobotJointPublisher.build_unified_frame(
            UNIFIED_WATCHDOG_OK, None, UNIFIED_FIELD_NOOP, UNIFIED_FIELD_NOOP, UNIFIED_FIELD_NOOP)
        # 喂狗帧连续两拍完全相同也必须都发送（不去重）
        self.assertEqual(node.ser.writes, [idle_frame, idle_frame])

    def test_send_unified_command_resets_one_shot_fields_after_send(self) -> None:
        node = make_protocol_node()
        node.pending_joint_positions = [1, 2, 3, 4, 5, 6]
        node.pending_stop = STOP_COMMAND
        node.pending_pump = PICK_COMMAND
        node.pending_speed = '55'

        node.send_unified_command()

        self.assertEqual(
            node.ser.writes,
            [f'U,OK,1,2,3,4,5,6,{STOP_COMMAND},{PICK_COMMAND},55*CHK\r\n'],
        )
        self.assertIsNone(node.pending_joint_positions)
        self.assertEqual(node.pending_stop, UNIFIED_FIELD_NOOP)
        self.assertEqual(node.pending_pump, UNIFIED_FIELD_NOOP)
        self.assertEqual(node.pending_speed, UNIFIED_FIELD_NOOP)

    def test_send_unified_command_preserves_pending_fields_on_serial_failure(self) -> None:
        node = make_protocol_node(fail_serial=True)
        node.pending_joint_positions = [1, 2, 3, 4, 5, 6]
        node.pending_stop = STOP_COMMAND
        node.pending_pump = PICK_COMMAND
        node.pending_speed = '55'

        node.send_unified_command()

        # 发送失败 -> 全部 pending 保留，下一拍重试
        self.assertEqual(node.pending_joint_positions, [1, 2, 3, 4, 5, 6])
        self.assertEqual(node.pending_stop, STOP_COMMAND)
        self.assertEqual(node.pending_pump, PICK_COMMAND)
        self.assertEqual(node.pending_speed, '55')


class ArmStateJsonProcessTest(unittest.TestCase):
    def test_json_process_does_not_overwrite_pending_stop_with_noop(self) -> None:
        node = make_json_node()
        node.pending_stop = STOP_COMMAND   # 已置位、尚未被发送拍运走
        node.pending_speed = '30'
        node.previous_decelerate_percent = 0

        node.json_process({})

        # 无边沿的 tick 不得把 pending 清回 'MM'（修 WS710 覆盖竞态）
        self.assertEqual(node.pending_stop, STOP_COMMAND)
        self.assertEqual(node.pending_speed, '30')

    def test_json_process_estop_edge_survives_next_tick_when_serial_down(self) -> None:
        node = make_json_node()
        node.ser = None   # 串口未连接：直发失败，命令落入 pending
        node.arm_stop = True

        node.json_process({})
        self.assertEqual(node.pending_stop, STOP_COMMAND)

        node.json_process({})   # 下一拍无边沿，pending 必须保持
        self.assertEqual(node.pending_stop, STOP_COMMAND)

    def test_json_process_estop_edge_sends_immediately(self) -> None:
        node = make_json_node()
        node.arm_stop = True

        node.json_process({})

        self.assertEqual(len(node.ser.writes), 1)
        self.assertIn(f',{STOP_COMMAND},', node.ser.writes[0])
        self.assertEqual(node.pending_stop, UNIFIED_FIELD_NOOP)

    def test_json_process_stop_wins_over_recover_in_same_tick(self) -> None:
        node = make_json_node()
        node.arm_stop = True
        node.arm_recover = True

        node.json_process({})

        self.assertEqual(len(node.ser.writes), 1)
        self.assertIn(f',{STOP_COMMAND},', node.ser.writes[0])
        self.assertNotIn(f',{RECOVER_COMMAND},', node.ser.writes[0])

    def test_json_recover_cannot_bypass_active_ros_estop(self) -> None:
        """Legacy JSON recovery cannot override an active ROS stop level."""
        node = make_json_node()
        node.stop_flag = True
        node.arm_recover = True

        node.json_process({'arm_recover': 'True'})

        self.assertEqual(node.ser.writes, [])
        self.assertEqual(node.pending_stop, UNIFIED_FIELD_NOOP)
        self.assertTrue(node.previous_rule_commands['arm_recover'])

    def test_json_recover_does_not_clear_active_capture_pending(self) -> None:
        node = make_json_node()
        target = [1, 2, 3, 4, 5, 6]
        node.capture_in_progress = True
        node.begin_capture_resume_stage(
            'middle', 7.0, joint_positions=target)
        node.arm_recover = True

        node.json_process({'arm_recover': 'True'})

        self.assertEqual(node.pending_joint_positions, target)
        self.assertFalse(node.recover_pending)
        self.assertIn(f',{RECOVER_COMMAND},', node.ser.writes[0])

    def test_json_process_queues_full_speed_once_at_startup(self) -> None:
        node = make_json_node()
        node.arm_decelerate_percent = 100
        node.previous_decelerate_percent = None

        node.json_process({'arm_decelerate': '1.0'})
        # 启动首拍显式下发一次全速（与 live DEC100 行为等价，R8）
        self.assertEqual(node.pending_speed, '100')

        node.pending_speed = UNIFIED_FIELD_NOOP   # 模拟发送拍已运走
        node.json_process({'arm_decelerate': '1.0'})
        # 无边沿不重发
        self.assertEqual(node.pending_speed, UNIFIED_FIELD_NOOP)


class ArmStateCaptureSequenceTest(unittest.TestCase):
    def test_enabled_mapping_fails_closed_without_calibration(self) -> None:
        node = make_sequence_node()
        node.handeye_mapping_enabled = True

        self.assertIsNone(node.pixel_goal_to_arm_point([540.0, 360.0]))
        self.assertFalse(node.is_goal_valid([540.0, 360.0]))

    def test_enabled_mapping_uses_trusted_calibration_only(self) -> None:
        node = make_sequence_node()
        node.handeye_mapping_enabled = True
        node.current_ee_position = np.array([0.1, 0.2, 0.1])

        class _Calibration:
            def pixel_to_base(self, pixel, end_effector):
                np.testing.assert_allclose(pixel, [540.0, 360.0])
                np.testing.assert_allclose(end_effector, [0.1, 0.2, 0.1])
                return np.array([0.12, 0.22])

            def contains(self, point):
                return tuple(point) == (0.12, 0.22, GOAL_HEIGHT)

        node.handeye_calibration = _Calibration()

        self.assertEqual(
            node.pixel_goal_to_arm_point([540.0, 360.0]),
            (0.12, 0.22, GOAL_HEIGHT),
        )

    def test_verified_affine_mapping_does_not_require_fk(self) -> None:
        node = make_sequence_node()
        node.handeye_mapping_enabled = True

        class _Calibration:
            verified_pixel_to_base_affine = np.ones((2, 3))

            def pixel_to_base(self, pixel, end_effector):
                self.pixel = pixel
                self.end_effector = end_effector
                return np.array([0.12, 0.22])

            def contains(self, point):
                return tuple(point) == (0.12, 0.22, GOAL_HEIGHT)

        calibration = _Calibration()
        node.handeye_calibration = calibration

        self.assertEqual(
            node.pixel_goal_to_arm_point([540.0, 360.0]),
            (0.12, 0.22, GOAL_HEIGHT),
        )
        self.assertIsNone(calibration.end_effector)

    def test_calibration_owner_rejects_capture_request(self) -> None:
        node = make_sequence_node()
        node.handeye_calibration_active = True

        self.assertTrue(node.handle_capture_request('A'))

        self.assertEqual(status_states(node), ['failed'])
        self.assertEqual(status_reasons(node), ['handeye_calibration_active'])
        self.assertFalse(node.capture_in_progress)

    def test_run_capture_sequence_rough_bound_reject_publishes_complete(self) -> None:
        node = make_sequence_node()
        # y=0.5 在 workspace_bounds 内但超出粗界限 0.38 -> 走粗界限拒绝路径
        node.run_capture_sequence('A', (0.1, 0.5, GOAL_HEIGHT), {})

        self.assertEqual(is_capture_values(node), [False])
        self.assertIn('failed', status_states(node))
        self.assertIn('target_out_of_workspace', status_reasons(node))
        self.assertEqual(node.pending_pump, UNIFIED_FIELD_NOOP)
        self.assertEqual(node.goal_pub.messages, [])

    def test_run_capture_sequence_workspace_reject_publishes_complete(self) -> None:
        node = make_sequence_node()
        # XY 在粗界限内，但 z=0.31 超出标定映射工作空间 z_max=0.30。
        node.run_capture_sequence('A', (0.1, 0.3, 0.31), {})

        self.assertEqual(is_capture_values(node), [False])
        self.assertIn('target_out_of_workspace', status_reasons(node))
        self.assertEqual(node.goal_pub.messages, [])

    def test_run_capture_sequence_ik_gate_failure_publishes_complete(self) -> None:
        node = make_sequence_node()
        node.sleep_until_shutdown = lambda seconds: False
        node.ik_success_flag = True   # 陈旧锁存值：序列必须先清零再等待本次结果

        point = k230_pixel_goal_to_arm_point([540.0, 360.0])
        node.run_capture_sequence('A', point, {})

        self.assertEqual(is_capture_values(node), [False])
        self.assertIn('ik_gate_failed', status_reasons(node))
        self.assertEqual(node.pending_pump, UNIFIED_FIELD_NOOP)
        self.assertEqual(len(node.goal_pub.messages), 1)
        self.assertFalse(node.capture_goal_control_active)

    def test_run_capture_sequence_interrupted_cancels_and_publishes_complete(self) -> None:
        node = make_sequence_node()
        node.shutdown_event.set()

        point = k230_pixel_goal_to_arm_point([540.0, 360.0])
        node.run_capture_sequence('A', point, {})

        self.assertEqual(is_capture_values(node), [False])
        self.assertIn('interrupted', status_states(node))
        self.assertFalse(node.capture_goal_control_active)

    def test_run_capture_sequence_happy_path_states_and_timings(self) -> None:
        node = make_sequence_node()
        sleep_stub = _IkSuccessSleepStub(node)
        node.sleep_until_shutdown = sleep_stub

        point = k230_pixel_goal_to_arm_point([540.0, 360.0])
        node.run_capture_sequence('A', point, {'target_source': 'merged_k230_frame'})

        self.assertEqual(
            status_states(node),
            [
                'pick_goal_published',
                'pick_command_sent',
                'lift_joint_command_queued',
                'middle_joint_command_queued',
                'second_middle_joint_command_queued',
                'place_command_sent',
                'look_joint_command_queued',
                'completed',
            ],
        )
        self.assertEqual(sleep_stub.calls, [29.0, 2.0, 5.0, 7.0, 7.0, 6.0, 8.0])

        payloads = [json.loads(message.data) for message in node.capture_status_pub.messages]
        self.assertEqual(payloads[3]['joint_positions'], list(MIDDLE_JOINT_POSITIONS))
        self.assertEqual(payloads[4]['joint_positions'], list(SECOND_MIDDLE_JOINT_POSITIONS))
        self.assertEqual(payloads[6]['joint_positions'], list(LOOK_JOINT_POSITIONS))
        self.assertEqual(is_capture_values(node), [False])
        self.assertEqual(node.pending_joint_positions, list(LOOK_JOINT_POSITIONS))
        self.assertEqual(node.pending_pump, PLACE_COMMAND)
        self.assertFalse(node.ik_success_flag)

        # 发布的 /goal 与唯一标定函数输出一致（发布路径无第二套系数）
        goal = node.goal_pub.messages[0]
        self.assertAlmostEqual(goal.x, point[0])
        self.assertAlmostEqual(goal.y, point[1])
        self.assertAlmostEqual(goal.z, point[2])

    def test_capture_goal_worker_missing_goal_publishes_complete(self) -> None:
        node = make_sequence_node()
        node.wait_for_capture_point = lambda target: (None, {})

        node.capture_goal_worker('A')

        self.assertEqual(is_capture_values(node), [False])
        self.assertIn('missing_k230_goal', status_reasons(node))
        self.assertFalse(node.capture_in_progress)

    def test_capture_goal_worker_exception_publishes_complete(self) -> None:
        node = make_sequence_node()

        def boom(_target):
            raise RuntimeError('boom')

        node.wait_for_capture_point = boom
        node.capture_goal_worker('A')

        self.assertEqual(is_capture_values(node), [False])
        self.assertTrue(
            any(reason and 'RuntimeError' in reason for reason in status_reasons(node)))
        self.assertFalse(node.capture_in_progress)

    def test_is_goal_valid_and_publish_path_share_single_calibration(self) -> None:
        node = make_sequence_node()
        pixel = [540.0, 360.0]
        point = k230_pixel_goal_to_arm_point(pixel, node.goal_height)

        # 校验路径与发布路径共用同一变换：同一像素输入产出同一坐标结论
        self.assertTrue(node.is_goal_valid(pixel))
        self.assertTrue(node.is_point_valid(point))

        # 超出工作空间的像素目标被同一函数拒绝
        self.assertFalse(node.is_goal_valid([2000.0, 2000.0]))
        # 零值像素目标（K230 无数据）被拒绝
        self.assertFalse(node.is_goal_valid([0.0, 0.0]))

    def test_wait_for_capture_point_accepts_merged_k230_frame(self) -> None:
        node = make_sequence_node()
        node.goal_pos['A'] = [540.0, 360.0]

        point, details = node.wait_for_capture_point('A')

        self.assertEqual(details.get('target_source'), 'merged_k230_frame')
        self.assertEqual(point, k230_pixel_goal_to_arm_point([540.0, 360.0], GOAL_HEIGHT))

    def test_wait_for_capture_point_accepts_goal_topic_source(self) -> None:
        node = make_sequence_node()
        node.goal = np.array([0.1, 0.3, 0.1])
        node.goal_event.set()

        point, details = node.wait_for_capture_point('A')

        self.assertEqual(details.get('target_source'), 'goal_topic')
        self.assertEqual(point, (0.1, 0.3, 0.1))

    def test_wait_for_capture_point_rejects_out_of_workspace_goal_topic(self) -> None:
        node = make_sequence_node()
        node.goal = np.array([5.0, 5.0, 5.0])
        node.goal_event.set()

        point, details = node.wait_for_capture_point('A')

        self.assertIsNone(point)
        self.assertEqual(details.get('reject_reason'), 'target_out_of_workspace')

    def test_goal_callback_sets_goal_event(self) -> None:
        node = make_sequence_node()

        node.goal_callback(_PointMsg(0.1, 0.2, 0.3))

        self.assertTrue(node.goal_event.is_set())
        self.assertEqual(list(node.goal), [0.1, 0.2, 0.3])

    def test_handle_capture_request_rejects_unsupported_goal_type(self) -> None:
        node = make_sequence_node()

        result = node.handle_capture_request('Z')

        self.assertTrue(result)
        self.assertIn('unsupported_goal_type', status_reasons(node))
        self.assertEqual(node.arm_capture_pub.messages, [])
        self.assertIsNone(node.capture_thread)

    def test_handle_capture_request_accepts_and_publishes_status(self) -> None:
        node = make_sequence_node()
        node.capture_goal_worker = lambda target: None

        result = node.handle_capture_request('b')

        self.assertTrue(result)
        node.capture_thread.join(timeout=1.0)
        self.assertEqual(is_capture_values(node), [True])
        self.assertEqual(status_states(node), ['accepted'])
        self.assertEqual(json.loads(node.capture_status_pub.messages[0].data)['goal_type'], 'B')


class ArmStateControlAndLookSequenceTest(unittest.TestCase):
    def test_calibration_owner_rejects_plain_control(self) -> None:
        node = make_protocol_node()
        node.handeye_calibration_active = True
        node.joint_pos = np.zeros(6, dtype=np.float64)

        node.control_callback(_JointMsg([0.1] * 6))

        np.testing.assert_array_equal(node.joint_pos, np.zeros(6))

    def test_send_look_position_sequence_queues_three_steps(self) -> None:
        node = make_protocol_node()
        node.look_joint_positions = list(LOOK_JOINT_POSITIONS)
        node.init_pos = True
        recorder = _RecordingEvent(node)
        node.shutdown_event = recorder

        node.send_look_position_sequence()

        j1, j2, j3, j4, j5, j6 = LOOK_JOINT_POSITIONS
        self.assertEqual(recorder.snapshots[0], [0, 0, j3, 0, 0, 0])
        self.assertEqual(recorder.snapshots[1], [0, j2, j3, j4, j5, 0])
        self.assertEqual(node.pending_joint_positions, list(LOOK_JOINT_POSITIONS))
        self.assertFalse(node.init_pos)

    def test_run_capture_goal_control_sequence_two_stage(self) -> None:
        node = make_protocol_node()
        node.look_joint_positions = list(LOOK_JOINT_POSITIONS)
        node.capture_goal_control_lock = threading.Lock()
        node.capture_goal_control_active = True
        node.capture_goal_control_pending = False
        node.capture_goal_control_running = True
        recorder = _RecordingEvent(node)
        node.shutdown_event = recorder

        target = [11, 22, 33, 44, 55, 66]
        node.run_capture_goal_control_sequence(target)

        # 第一段：j1,j4,j5,j6 取目标值，j2,j3 用观察位补齐
        self.assertEqual(
            recorder.snapshots[0],
            [11, LOOK_JOINT_POSITIONS[1], LOOK_JOINT_POSITIONS[2], 44, 55, 66],
        )
        # 第二段：全部关节
        self.assertEqual(recorder.snapshots[1], target)
        self.assertEqual(recorder.waits, [10.0, 0.3])
        self.assertFalse(node.capture_goal_control_running)
        self.assertFalse(node.capture_goal_control_active)

    def test_control_callback_respects_stop_flag(self) -> None:
        node = make_protocol_node()
        node.stop_flag = True
        node.capture_goal_control_lock = threading.Lock()
        node.capture_goal_control_running = False
        node.capture_goal_control_thread = None
        node.joint_pos = np.zeros(6, dtype=np.float64)

        node.control_callback(_JointMsg([0.1] * 6))

        self.assertFalse(node.capture_goal_control_running)
        self.assertIsNone(node.capture_goal_control_thread)

    def test_capture_control_during_estop_preserves_resume_target(self) -> None:
        node = make_protocol_node()
        node.stop_flag = True
        node.capture_in_progress = True
        node.capture_goal_control_lock = threading.Lock()
        node.capture_goal_control_active = True
        node.capture_goal_control_pending = True
        node.capture_goal_control_running = False
        node.capture_goal_control_thread = None
        node.joint_pos = np.zeros(6, dtype=np.float64)

        def record_target(target) -> None:
            node.begin_capture_resume_stage(
                'pick_goal_final', 0.3, joint_positions=target)

        node.run_capture_goal_control_sequence = record_target
        node.control_callback(_JointMsg([0.1] * 6))
        node.capture_goal_control_thread.join(timeout=1.0)

        expected = RobotJointPublisher.joint_radians_to_encoder_positions(
            [0.1] * 6)
        self.assertEqual(
            node.capture_resume_snapshot['joint_positions'], expected)
        self.assertIsNone(node.pending_joint_positions)

    def test_late_capture_control_cannot_overwrite_current_stage(self) -> None:
        node = make_protocol_node()
        node.capture_in_progress = True
        node.capture_goal_control_lock = threading.Lock()
        node.capture_goal_control_active = False
        node.capture_goal_control_pending = False
        node.capture_goal_control_running = False
        node.capture_goal_control_thread = None
        node.joint_pos = np.zeros(6, dtype=np.float64)
        current_target = [1, 2, 3, 4, 5, 6]
        node.begin_capture_resume_stage(
            'middle', 7.0, joint_positions=current_target)

        node.control_callback(_JointMsg([0.1] * 6))

        self.assertIsNone(node.capture_goal_control_thread)
        self.assertEqual(
            node.capture_resume_snapshot['joint_positions'], current_target)

    def test_control_callback_spawns_segmented_sequence_and_drops_while_running(self) -> None:
        node = make_protocol_node()
        node.capture_goal_control_lock = threading.Lock()
        node.capture_goal_control_active = False
        node.capture_goal_control_pending = False
        node.capture_goal_control_running = False
        node.capture_goal_control_thread = None
        started = []
        node.run_capture_goal_control_sequence = lambda target: started.append(list(target))

        node.control_callback(_JointMsg([0.1] * 6))
        node.capture_goal_control_thread.join(timeout=1.0)
        self.assertEqual(len(started), 1)
        self.assertEqual(
            started[0],
            RobotJointPublisher.joint_radians_to_encoder_positions([0.1] * 6),
        )

        # 分段发送运行中：丢弃新的 /control 消息
        node.control_callback(_JointMsg([0.2] * 6))
        self.assertEqual(len(started), 1)

    def test_joint_radians_to_encoder_positions_matches_legacy_convention(self) -> None:
        scale = 51200.0 / 6.28
        positions = RobotJointPublisher.joint_radians_to_encoder_positions([0.1] * 6)
        self.assertEqual(
            positions,
            [
                int(0.1 * scale) * 30,
                -int(0.1 * scale) * 30,
                int(0.1 * scale) * 30,
                int(0.1 * scale) * 10,
                int(0.1 * scale) * 10,
                int(0.1 * scale),
            ],
        )

        lift = RobotJointPublisher.joint_radians_to_encoder_positions(
            [0.1] * 6, j2_offset_radians=0.5)
        self.assertEqual(lift[1], -int((0.1 + 0.5) * scale) * 30)
        self.assertEqual(lift[0], positions[0])


class ArmStateSerialBufferTest(unittest.TestCase):
    def test_process_k230_frame_publishes_d_pixel(self) -> None:
        node = make_sequence_node()

        node.process_k230_frame(b'E10,20,30,40,50,60,70,80P')

        self.assertEqual(len(node.handeye_d_pub.messages), 1)
        message = node.handeye_d_pub.messages[0]
        self.assertEqual((message.point.x, message.point.y), (70.0, 80.0))

    def test_process_serial_buffer_merged_packet_half_k230_not_reprocessed(self) -> None:
        node = make_sequence_node()
        arm_frames = []
        node.robot_joint_sub = lambda observation: arm_frames.append(list(observation))

        buffer = bytearray(b'A1 2 3 4 5 6BE10,20')
        buffer = node.process_serial_buffer(buffer)

        # ARM 帧被消费一次，半包 K230 保留
        self.assertEqual(arm_frames, [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        self.assertEqual(bytes(buffer), b'E10,20')

        # 补齐 K230 剩余字节后：ARM 帧不得重复处理，K230 帧正常入库
        buffer += b',30,40,50,60,70,80P'
        buffer = node.process_serial_buffer(buffer)

        self.assertEqual(len(arm_frames), 1)
        self.assertEqual(node.goal_pos['A'], [10.0, 20.0])
        self.assertTrue(node.k230_goal_event.is_set())
        self.assertEqual(len(buffer), 0)

    def test_process_serial_buffer_merged_packet_complete(self) -> None:
        node = make_sequence_node()
        arm_frames = []
        node.robot_joint_sub = lambda observation: arm_frames.append(list(observation))

        buffer = bytearray(b'A1 2 3 4 5 6BE10,20,30,40,50,60,70,80P')
        buffer = node.process_serial_buffer(buffer)

        self.assertEqual(len(arm_frames), 1)
        self.assertEqual(node.goal_pos['D'], [70.0, 80.0])
        self.assertEqual(len(buffer), 0)

    def test_process_serial_buffer_clears_oversized_garbage(self) -> None:
        node = make_sequence_node()

        buffer = bytearray(b'A' + b'1' * 300)
        buffer = node.process_serial_buffer(buffer)

        self.assertEqual(len(buffer), 0)


if __name__ == '__main__':
    unittest.main()
