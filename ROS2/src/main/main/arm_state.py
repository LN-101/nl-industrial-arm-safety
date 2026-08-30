from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time

from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
from geometry_msgs.msg import Point, PointStamped
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
import serial
from std_msgs.msg import Bool, Float32, String


GOAL_TYPES = ('A', 'B', 'C', 'D')
ARM_FRAME_HEADER = ord('A')
ARM_FRAME_TAIL = ord('B')
K230_FRAME_HEADER = ord('E')
K230_FRAME_TAIL = ord('P')
CAPTURE_STATUS_TOPIC = '/arm_capture_status'
CAPTURE_TARGET_TIMEOUT_SECONDS = 2.0
SERIAL_BUFFER_LIMIT_BYTES = 256

# 统一串口帧协议（合并决策 Q1）：U,{OK},{p1..p6},{STOP},{PUMP},{SPD}*CHK\r\n
# 字段取 'MM' 表示本拍无操作；OK 为喂狗字段，必须按周期强发（不得被去重饿死）。
UNIFIED_FIELD_NOOP = 'MM'
UNIFIED_WATCHDOG_OK = 'OK'
UNIFIED_SEND_PERIOD_SECONDS = 0.5
PICK_COMMAND = 'CPQE'    # PUMP 字段：吸取
PLACE_COMMAND = 'PUT'    # PUMP 字段：放置（停止吸取）
STOP_COMMAND = 'CD'      # STOP 字段：急停
RECOVER_COMMAND = 'EF'   # STOP 字段：恢复运动

# K230 像素 -> 机械臂坐标备用仿射（mujoco_ws (5) 手眼参数）。
# 全文件唯一一套系数：is_goal_valid 校验与抓取发布路径共用 k230_pixel_goal_to_arm_point，
# 消除 WS710 校验/发布双系数 1cm 漂移。
K230_AFFINE_X = (-0.000300211, 0.000011308, 0.246757)
# 参考 Y 截距 0.450067，加 0.01 m 实机落点补偿。
K230_AFFINE_Y = (-0.000019886, -0.000271216, 0.470067)
GOAL_HEIGHT = 0.09

# 工作空间边界 + 粗界限第一道门（粗界限 0.38 与 ik_control workspace 上界对齐）
WORKSPACE_BOUNDS = {
    'x_min': -0.32,
    'x_max': 0.38,
    'y_min': -0.32,
    'y_max': 0.38,
    'z_min': 0.05,
    'z_max': 0.30,
}
ROUGH_XY_BOUND_MIN = -0.32
ROUGH_XY_BOUND_MAX = 0.38

# 抓放状态机时序（照抄 WS710 goal_pos_pub）
PICK_TRAVEL_SECONDS = 18.0
PICK_PUMP_SECONDS = 2.0
LIFT_TRAVEL_SECONDS = 5.0
MIDDLE_TRAVEL_SECONDS = 4.0
SECOND_MIDDLE_TRAVEL_SECONDS = 4.0
PLACE_PUMP_SECONDS = 6.0
LOOK_TRAVEL_SECONDS = 8.0
LIFT_J2_RADIANS = 0.5

# 关节空间路点（下位机原始编码值，队友标定）
LOOK_JOINT_POSITIONS = (383730, 515190, 635910, 970, 157790, 8489)
MIDDLE_JOINT_POSITIONS = (0, 452940, 330210, 10, 88010, 9986)
SECOND_MIDDLE_JOINT_POSITIONS = (0, 552940, 350210, 10, 68010, 9986)

# 开机观察位三步序列（取代旧 /control 首帧 init 三步，二者不得并存）
LOOK_SEQUENCE_DELAY_SECONDS = 8.0
LOOK_STEP1_WAIT_SECONDS = 5.0
LOOK_STEP2_WAIT_SECONDS = 2.0

# 与 AI 侧 local_safety_assistant/arm_rules.py 的 DEFAULT_ARM_RULE_DOCUMENT 保持一致；
# 该 JSON 通道由 07-09-migrate-json-commands-to-ros 任务负责最终下线。
DEFAULT_ARM_RULES = {
    'arm_capture': 'False',
    'arm_capture_goal': 'A',
    'arm_decelerate': '1.0',
    'arm_stop': 'False',
    'arm_recover': 'False',
    'arm_safety_distance': '0.4',
}
PERSONNEL_DISTANCE_RULE_ID = 'stop_on_person_intrusion'
DEFAULT_PERSONNEL_DISTANCE_M = float(DEFAULT_ARM_RULES['arm_safety_distance'])
DEFAULT_SAFETY_RULES_VERSION = 1
OWNER_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
)


def default_handeye_calibration_path():
    configured_path = os.environ.get('AI_OV_HANDEYE_CALIBRATION_FILE', '').strip()
    if configured_path:
        return Path(configured_path).expanduser()
    try:
        return (
            Path(get_package_share_directory('control'))
            / 'config'
            / 'handeye_xy.yaml'
        )
    except PackageNotFoundError:
        return (
            Path(__file__).resolve().parents[2]
            / 'control'
            / 'config'
            / 'handeye_xy.yaml'
        )


def reset_personnel_distance_rule(document, distance_m=DEFAULT_PERSONNEL_DISTANCE_M):
    if not isinstance(document, dict):
        raise ValueError('安全规则文档必须是 JSON 对象')

    version = document.get('version')
    if not isinstance(version, int) or isinstance(version, bool):
        raise ValueError("安全规则文档缺少整数 'version'")

    rules = document.get('rules')
    if not isinstance(rules, list):
        raise ValueError("安全规则文档缺少数组 'rules'")

    matching_indexes = []
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise ValueError(f'安全规则第 {index} 项必须是 JSON 对象')
        if rule.get('id') == PERSONNEL_DISTANCE_RULE_ID:
            matching_indexes.append(index)

    if len(matching_indexes) != 1:
        raise ValueError(
            f'安全规则必须且只能包含一条 {PERSONNEL_DISTANCE_RULE_ID!r} 规则')

    rule_index = matching_indexes[0]
    conditions = rules[rule_index].get('conditions')
    if not isinstance(conditions, dict):
        raise ValueError('人员安全距离规则缺少 conditions 对象')
    person_distance = conditions.get('person_distance_m')
    if not isinstance(person_distance, dict):
        raise ValueError('人员安全距离规则缺少 person_distance_m 对象')
    current_distance = person_distance.get('lt')
    if (
        not isinstance(current_distance, (int, float))
        or isinstance(current_distance, bool)
        or not math.isfinite(float(current_distance))
    ):
        raise ValueError('人员安全距离规则的 lt 必须是有限数值')

    if (
        float(current_distance) == float(distance_m)
        and version == DEFAULT_SAFETY_RULES_VERSION
    ):
        return document, False

    updated = deepcopy(document)
    updated['rules'][rule_index]['conditions']['person_distance_m']['lt'] = float(distance_m)
    updated['version'] = DEFAULT_SAFETY_RULES_VERSION
    return updated, True


def parse_numeric_frame(frame, header, tail, min_values):
    text = bytes(frame).decode('utf-8', errors='ignore').strip()
    if not text.startswith(header) or not text.endswith(tail):
        return None

    parts = text[1:-1].replace(',', ' ').split()
    if len(parts) < min_values:
        return None

    try:
        values = [float(part) for part in parts[:min_values]]
    except ValueError:
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    return values


def parse_arm_joint_frame(frame):
    return parse_numeric_frame(frame, 'A', 'B', 6)


def parse_k230_goal_frame(frame):
    values = parse_numeric_frame(frame, 'E', 'P', 8)
    if values is None:
        return None
    return {
        goal_type: [values[index * 2], values[index * 2 + 1]]
        for index, goal_type in enumerate(GOAL_TYPES)
    }


def k230_pixel_goal_to_arm_point(pixel_goal, goal_height=GOAL_HEIGHT):
    # 像素目标 -> 机械臂坐标的唯一标定入口（校验与发布共用，杜绝双系数漂移）
    if len(pixel_goal) < 2:
        return None

    pixel_x = float(pixel_goal[0])
    pixel_y = float(pixel_goal[1])
    if not math.isfinite(pixel_x) or not math.isfinite(pixel_y):
        return None
    if pixel_x == 0.0 and pixel_y == 0.0:
        return None

    ax, ay, a0 = K230_AFFINE_X
    bx, by, b0 = K230_AFFINE_Y
    return (
        ax * pixel_x + ay * pixel_y + a0,
        bx * pixel_x + by * pixel_y + b0,
        goal_height,
    )


class RobotJointPublisher(Node):

    def __init__(self):
        super().__init__('arm_state')
        self.ser = None
        self.serial_lock = threading.Lock()
        self.shutdown_event = threading.Event()
        self.receive_thread = None
        self.get_logger().info('node_init')

        self.joint_pos = np.zeros(6, dtype=np.float64)
        self.goal = np.zeros(3, dtype=np.float64)

        self.ARM_FRAME_HEADER = ARM_FRAME_HEADER
        self.ARM_FRAME_TAIL = ARM_FRAME_TAIL
        self.K230_FRAME_HEADER = K230_FRAME_HEADER
        self.K230_FRAME_TAIL = K230_FRAME_TAIL

        self.stop_flag = False
        self.run_count = 0
        # 开机观察位序列完成后置 False（由 look 序列驱动，不再由 /control 首帧驱动）
        self.init_pos = True

        self.arm_capture = False
        self.arm_capture_goal = 'A'
        self.arm_decelerate = False
        self.arm_decelerate_percent = 0
        self.arm_stop = False
        self.arm_recover = False
        self.arm_safety_distance = 0.2
        self.capture_lock = threading.Lock()
        self.capture_thread = None
        # 抓取恢复上下文独立于 pending 队列：pending 会在发送成功或急停时清空。
        self.capture_resume_condition = threading.Condition()
        self.capture_resume_snapshot = None
        self.capture_resume_generation = 0
        self.capture_pause_generation = 0
        self.capture_pause_started_at = None
        self.capture_pause_total_seconds = 0.0
        self.recover_pending = False
        self.k230_goal_event = threading.Event()
        self.goal_event = threading.Event()
        self.goal_pos = {goal_type: [0.0, 0.0] for goal_type in GOAL_TYPES}
        self.previous_rule_commands = {
            'arm_capture': False,
            'arm_stop': False,
            'arm_recover': False,
        }
        self.previous_decelerate_percent = None

        # 统一帧协议 pending 缓存：只置位，发送成功后由发送路径复位
        # （修 WS710 缺陷：json_process 不得在无边沿时把 pending 清回 'MM'）
        self.pending_joint_positions = None
        self.pending_stop = UNIFIED_FIELD_NOOP
        self.pending_pump = UNIFIED_FIELD_NOOP
        self.pending_speed = UNIFIED_FIELD_NOOP
        self.pending_ok = UNIFIED_WATCHDOG_OK
        # 仅观测/调试用；不做去重（喂狗与急停重发都必须按拍强发）
        self.last_sent_command = None

        # 观察位与 IK 分段控制（WS710）
        self.look_joint_positions = list(LOOK_JOINT_POSITIONS)
        self.look_thread = None
        self.capture_goal_control_active = False
        self.capture_goal_control_pending = False
        self.capture_goal_control_running = False
        self.capture_goal_control_lock = threading.Lock()
        self.capture_goal_control_thread = None

        # 抓取生命周期标志（WS710）
        self.capture_in_progress = False
        self.capture_success = False

        self.goal_height = GOAL_HEIGHT
        self.workspace_bounds = dict(WORKSPACE_BOUNDS)
        self.ik_success_flag = False
        self.handeye_calibration_active = False
        mapping_enabled_default = os.environ.get(
            'AI_OV_HANDEYE_MAPPING_ENABLED',
            'false',
        ).strip().lower() in {'1', 'true', 'yes', 'on'}
        self.declare_parameter(
            'handeye_mapping_enabled',
            mapping_enabled_default,
        )
        self.declare_parameter(
            'handeye_calibration_file',
            str(default_handeye_calibration_path()),
        )
        self.declare_parameter(
            'handeye_max_age_days',
            float(os.environ.get('AI_OV_HANDEYE_MAX_AGE_DAYS', '30.0')),
        )
        self.handeye_mapping_enabled = bool(
            self.get_parameter('handeye_mapping_enabled').value)
        self.handeye_calibration_path = Path(str(
            self.get_parameter('handeye_calibration_file').value)).expanduser()
        self.handeye_max_age_days = float(
            self.get_parameter('handeye_max_age_days').value)
        self.handeye_calibration = None
        self.handeye_fk_position = None
        self.current_ee_position = None
        self.load_runtime_handeye_calibration()

        self.config_path = self.resolve_config_path()
        self.safety_rules_path = self.resolve_safety_rules_path()
        self.reset_arm_rules_to_defaults()

        # 订阅者
        self.joint_sub = self.create_subscription(JointState, '/control', self.control_callback, 1)
        self.goal_sub = self.create_subscription(Point, '/goal', self.goal_callback, 1)
        self.stop_sub = self.create_subscription(Bool, '/emergency_stop', self.stop_callback, 1)
        self.ik_sub = self.create_subscription(Bool, '/ik_success', self.ik_success_callback, 1)
        self.handeye_active_sub = self.create_subscription(
            Bool,
            '/handeye/calibration_active',
            self.handeye_active_callback,
            OWNER_QOS,
        )
        self.handeye_control_sub = self.create_subscription(
            JointState,
            '/handeye/control',
            self.handeye_control_callback,
            1,
        )

        # 连接串口
        self.connect_ser()

        # 发布者
        self.joint_pub = self.create_publisher(JointState, '/robot_joint_state', 10)
        self.goal_pub = self.create_publisher(Point, '/goal', 1)
        self.safety_distance_pub = self.create_publisher(Float32, '/arm_safety_distance', 1)
        self.goal_type_pub = self.create_publisher(String, '/goal_type', 1)
        self.arm_capture_pub = self.create_publisher(Bool, '/is_capture', 1)
        self.capture_status_pub = self.create_publisher(String, CAPTURE_STATUS_TOPIC, 10)
        self.handeye_d_pub = self.create_publisher(
            PointStamped,
            '/handeye/d_pixel',
            10,
        )

        # 开启一个线程接受下位机数据
        if self.serial_is_open():
            self.receive_thread = threading.Thread(target=self.receive_data, daemon=True)
            self.receive_thread.start()
        else:
            self.get_logger().warn('串口未连接，跳过下位机数据接收线程')

        self.timer = self.create_timer(0.3, self.read_json)

        # 统一帧周期发送（0.5s 一拍，兼作喂狗）
        self.send_timer = self.create_timer(UNIFIED_SEND_PERIOD_SECONDS, self.send_unified_command)

        # 启动 8 秒后分三步进入观察位
        self.look_timer = self.create_timer(
            LOOK_SEQUENCE_DELAY_SECONDS, self.start_look_position_sequence)
        self.get_logger().info('将在8秒后开始分三步发送观察位置命令')

    def ik_success_callback(self, msg):
        self.ik_success_flag = msg.data

    def handeye_active_callback(self, msg):
        self.handeye_calibration_active = bool(msg.data)

    def handeye_control_callback(self, msg_data):
        if not self.handeye_calibration_active:
            return
        self.handle_control_message(msg_data)

    def load_runtime_handeye_calibration(self):
        if not self.handeye_mapping_enabled:
            return
        if self.handeye_max_age_days <= 0.0:
            self.get_logger().error('handeye_max_age_days 必须为正数；标定映射不可用')
            return
        try:
            from control.handeye_calibration_io import load_calibration

            calibration = load_calibration(self.handeye_calibration_path)
            created_at = calibration.metadata.get('created_at')
            if not isinstance(created_at, str):
                raise ValueError('标定 metadata.created_at 缺失')
            created_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            if created_time.tzinfo is None:
                raise ValueError('标定 created_at 必须带时区')
            age_seconds = (datetime.now(timezone.utc) - created_time).total_seconds()
            if age_seconds < 0.0 or age_seconds > self.handeye_max_age_days * 86400.0:
                raise ValueError('标定文件已过期或时间戳在未来')
            fk_callback = None
            if calibration.verified_pixel_to_base_affine is None:
                from control.ik_control import fk_position
                fk_callback = fk_position
            self.handeye_calibration = calibration
            self.handeye_fk_position = fk_callback
            self.get_logger().info(f'已启用可信手眼标定映射: {self.handeye_calibration_path}')
        except (ImportError, KeyError, OSError, TypeError, ValueError) as exc:
            self.handeye_calibration = None
            self.handeye_fk_position = None
            self.get_logger().error(
                f'手眼标定映射启用失败，将拒绝像素抓取目标: {exc}')

    def pixel_goal_to_arm_point(self, pixel_goal):
        if not self.handeye_mapping_enabled:
            return k230_pixel_goal_to_arm_point(pixel_goal, self.goal_height)
        if (
            self.handeye_calibration is None
            or len(pixel_goal) < 2
        ):
            return None
        verified_affine = getattr(
            self.handeye_calibration,
            'verified_pixel_to_base_affine',
            None,
        )
        if verified_affine is None and self.current_ee_position is None:
            return None
        pixel = np.asarray(pixel_goal[:2], dtype=np.float64)
        if not np.all(np.isfinite(pixel)) or np.allclose(pixel, 0.0):
            return None
        try:
            xy = self.handeye_calibration.pixel_to_base(
                pixel,
                self.current_ee_position,
            )
            point = (float(xy[0]), float(xy[1]), float(self.goal_height))
        except ValueError:
            return None
        if not self.handeye_calibration.contains(point):
            return None
        return point

    def start_look_position_sequence(self):
        # 启动观察位置的分步发送序列（开机 8 秒后执行一次）
        # 取消定时器，只执行一次
        self.look_timer.cancel()

        self.get_logger().info(f'开始分三步发送观察位置命令: {self.look_joint_positions}')
        self.look_thread = threading.Thread(target=self.send_look_position_sequence, daemon=True)
        self.look_thread.start()

    def send_look_position_sequence(self):
        # 分三步发送观察位（关节原始编码值，经统一帧运载）
        try:
            j1, j2, j3, j4, j5, j6 = self.look_joint_positions

            # 第一步：只发送第三轴（保持其他轴不动）
            self.queue_joint_command([0, 0, j3, 0, 0, 0])
            self.get_logger().info(f'第一步：发送第三轴数据 {j3}')
            if self.shutdown_event.wait(LOOK_STEP1_WAIT_SECONDS):
                return

            # 第二步：发送第二轴和第三轴（j4/j5 用观察位安全值）
            self.queue_joint_command([0, j2, j3, j4, j5, 0])
            self.get_logger().info(f'第二步：发送第二轴和第三轴数据 {j2}, {j3}')
            if self.shutdown_event.wait(LOOK_STEP2_WAIT_SECONDS):
                return

            # 第三步：发送所有轴
            self.queue_joint_command(self.look_joint_positions)
            self.get_logger().info(f'第三步：发送所有轴数据 {self.look_joint_positions}')

            self.init_pos = False
            self.get_logger().info('观察位置发送完成')
        except Exception as exc:
            self.get_logger().error(f'发送观察位置序列失败: {exc}')

    def resolve_config_path(self):
        env_path = os.environ.get('ARM_RULES_PATH')
        candidates = []
        if env_path:
            candidates.append(Path(env_path))
        candidates.extend([
            Path('/home/inteldk/AI_ov/Code/config/arm_rules.json'),
            Path('/home/robot/mujoco_ws/src/camera/arm_rules.json'),
        ])
        try:
            candidates.append(Path(get_package_share_directory('camera')) / 'arm_rules.json')
        except PackageNotFoundError:
            pass

        for path in candidates:
            if path.is_file():
                return path

        self.get_logger().warn(f'未找到 arm_rules.json，将使用默认配置。候选路径: {candidates}')
        return None

    def resolve_safety_rules_path(self):
        env_path = os.environ.get('SAFETY_RULES_PATH')
        candidates = []
        if env_path:
            candidates.append(Path(env_path))
        candidates.append(Path('/home/inteldk/AI_ov/Code/config/safety_rules.example.json'))

        for path in candidates:
            if path.is_file():
                return path

        self.get_logger().warn(f'未找到 AI safety rules，跳过启动重置。候选路径: {candidates}')
        return None

    def write_json_atomic(self, path, document):
        # 先写同目录临时文件再原子替换，避免读侧看到半截 JSON
        tmp_path = None
        try:
            fd, tmp_name = tempfile.mkstemp(
                dir=str(path.parent),
                prefix=path.name + '.',
                suffix='.tmp',
            )
            tmp_path = Path(tmp_name)
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(document, f, ensure_ascii=False, indent=2)
                f.write('\n')
                f.flush()
            os.replace(tmp_path, path)
            return True
        except OSError as exc:
            self.get_logger().error(f'写入配置失败 {path}: {exc}')
            if tmp_path is not None:
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            return False

    def write_config_atomic(self, config):
        return self.write_json_atomic(self.config_path, config)

    def prepare_safety_rules_reset(self):
        safety_rules_path = getattr(self, 'safety_rules_path', None)
        if safety_rules_path is None:
            return None, False
        try:
            with safety_rules_path.open('r', encoding='utf-8') as f:
                document = json.load(f)
            return reset_personnel_distance_rule(document)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(
                f'无法校验 AI safety rules，拒绝启动: {safety_rules_path}: {exc}') from exc

    def reset_arm_rules_to_defaults(self):
        # 启动时无条件覆盖为默认文档，清除节点停机期间遗留的脉冲命令（如 arm_capture）
        if self.config_path is None:
            self.get_logger().warn('未找到 arm_rules.json，跳过启动重置')
            return
        safety_document, safety_changed = self.prepare_safety_rules_reset()
        if not self.write_config_atomic(dict(DEFAULT_ARM_RULES)):
            # 清除失败意味着停机期间遗留的命令仍可能被当作新命令执行，拒绝带雷启动
            raise RuntimeError(f'无法重置 arm_rules.json，拒绝启动: {self.config_path}')
        if safety_changed and not self.write_json_atomic(self.safety_rules_path, safety_document):
            raise RuntimeError(f'无法重置 AI safety rules，拒绝启动: {self.safety_rules_path}')
        if safety_changed:
            self.get_logger().info(
                '启动时已将 AI 安全规则版本重置为 1、'
                f'人员安全距离重置为 {DEFAULT_PERSONNEL_DISTANCE_M:g} 米'
            )
        self.get_logger().info('启动时已将 arm_rules.json 重置为默认状态')

    @staticmethod
    def parse_bool(value, default=False):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in ('true', '1', 'yes', 'y', 'on'):
                return True
            if normalized in ('false', '0', 'no', 'n', 'off', ''):
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return default

    @staticmethod
    def parse_float(value, default):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def parse_percent(cls, value, default=0):
        if isinstance(value, bool):
            return 100 if value else 0

        if isinstance(value, str):
            normalized = value.strip()
            has_percent_suffix = normalized.endswith('%')
            if has_percent_suffix:
                normalized = normalized[:-1].strip()
            bool_value = cls.parse_bool(value, None)
            if bool_value is not None:
                return 100 if bool_value else 0
            value = normalized
        else:
            has_percent_suffix = False

        percent = cls.parse_float(value, default)
        if not has_percent_suffix and 0 < percent <= 1:
            percent *= 100
        return max(0, min(100, int(round(percent))))

    def serial_is_open(self):
        return self.ser is not None and self.ser.is_open

    def write_serial(self, command, log_message=None, sleep_time=0.0):
        if not self.serial_is_open():
            self.get_logger().warn('串口未连接，跳过发送')
            return False

        with self.serial_lock:
            try:
                self.ser.write(command.encode('utf-8'))
                self.ser.flush()
            except serial.SerialException as exc:
                self.get_logger().error(f'串口发送失败: {exc}')
                return False

        if log_message:
            self.get_logger().info(log_message)
        if sleep_time > 0:
            time.sleep(sleep_time)
        return True

    @staticmethod
    def build_unified_frame(ok, joint_positions, stop, pump, speed):
        # 构造统一帧：U,{OK},{p1..p6},{STOP},{PUMP},{SPD}*CHK\r\n
        if joint_positions is not None:
            p1, p2, p3, p4, p5, p6 = joint_positions
        else:
            p1 = p2 = p3 = p4 = p5 = p6 = UNIFIED_FIELD_NOOP
        return f'U,{ok},{p1},{p2},{p3},{p4},{p5},{p6},{stop},{pump},{speed}*CHK\r\n'

    def send_unified_command(self):
        # 按 0.5s 周期发送统一帧。
        # 与 WS710 原实现的两点刻意差异（合并内必修缺陷）：
        # - 不做同帧去重：OK 喂狗字段必须每拍强发，去重会饿死下位机看门狗，
        #   也可能吞掉急停/恢复重发（审查 R3/R6）。
        # - 发送失败时保留 pending_*，下一拍自动重试；复位只在确认写入后进行，
        #   且只复位与本帧一致的值，避免覆盖并发线程刚写入的新命令。
        if not self.serial_is_open():
            return

        with self.capture_resume_condition:
            motion_blocked = self.estop_motion_blocked()
            if motion_blocked:
                # 其他内部线程可能仍写 pending；急停期间一律丢弃，抓取续作依赖独立快照。
                self.pending_joint_positions = None
                self.pending_pump = UNIFIED_FIELD_NOOP

            joint_positions = None if motion_blocked else self.pending_joint_positions
            stop = self.pending_stop
            pump = UNIFIED_FIELD_NOOP if motion_blocked else self.pending_pump
            speed = self.pending_speed
            command = self.build_unified_frame(
                self.pending_ok, joint_positions, stop, pump, speed)

            if not self.write_serial(command):
                return
            self.last_sent_command = command

            # 一次性字段：发送成功后复位（绝不在别处清 'MM'）
            if self.pending_joint_positions == joint_positions:
                self.pending_joint_positions = None
            if stop != UNIFIED_FIELD_NOOP and self.pending_stop == stop:
                self.pending_stop = UNIFIED_FIELD_NOOP
            if pump != UNIFIED_FIELD_NOOP and self.pending_pump == pump:
                self.pending_pump = UNIFIED_FIELD_NOOP
            if speed != UNIFIED_FIELD_NOOP and self.pending_speed == speed:
                self.pending_speed = UNIFIED_FIELD_NOOP
            self.mark_capture_command_sent(joint_positions, pump)
            if stop == RECOVER_COMMAND:
                self.complete_estop_recovery()

    def send_stop_frame_now(self, stop_command, log_message):
        # 急停/恢复同步立即直发（不等 0.5s 发送拍）——live 立即性语义在统一帧上的重建。
        # 先置 pending_stop 再直发：直发失败（串口断开/异常）时命令保留在 pending_stop，
        # 由周期发送拍重试；成功后按"发送后复位"规则清除。
        # 只运载 STOP 字段，不夹带关节/泵/速度命令；急停同时清空已排队的运动与泵命令，
        # 避免统一帧队列把急停前的运动命令带到急停之后（急停语义不得劣化，R6）。
        with self.capture_resume_condition:
            self.pending_stop = stop_command
            if (
                stop_command == STOP_COMMAND
                or (stop_command == RECOVER_COMMAND and self.recover_pending)
            ):
                # CD 清掉急停前队列；有效 ROS 恢复的 EF 清掉急停期间旧队列。
                # 单独的 legacy JSON EF 不得误清正在执行的抓取命令。
                # 抓取阶段命令只允许由独立恢复快照重建。
                self.pending_joint_positions = None
                self.pending_pump = UNIFIED_FIELD_NOOP
            command = self.build_unified_frame(
                self.pending_ok,
                None,
                stop_command,
                UNIFIED_FIELD_NOOP,
                UNIFIED_FIELD_NOOP,
            )
            if self.write_serial(command, log_message, 0.1):
                self.last_sent_command = command
                if self.pending_stop == stop_command:
                    self.pending_stop = UNIFIED_FIELD_NOOP
                if stop_command == RECOVER_COMMAND:
                    self.complete_estop_recovery()
                return True
        self.get_logger().warn(f'急停/恢复直发未成功，保留 pending 待发送拍重试: {stop_command}')
        return False

    def reset_processed_bool_flags(self, config, processed_rule_names):
        if self.config_path is None or not processed_rule_names:
            return

        changed = False
        for rule_name in processed_rule_names:
            value = config.get(rule_name)
            if self.parse_bool(value, False):
                config[rule_name] = 'False' if isinstance(value, str) else False
                changed = True

        if not changed:
            return

        self.write_config_atomic(config)

    def read_json(self):
        # 读取配置文件
        if self.config_path is None:
            config = {}
        else:
            try:
                with self.config_path.open('r', encoding='utf-8') as f:
                    config = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                self.get_logger().error(f'读取配置失败: {exc}')
                return

        self.arm_capture = self.parse_bool(config.get('arm_capture'), False)  # 是否抓取
        self.arm_capture_goal = str(config.get('arm_capture_goal', 'A')).strip().upper()  # 抓取目标类型
        self.arm_decelerate = self.parse_bool(config.get('arm_decelerate'), False)  # 减速开关兼容
        self.arm_decelerate_percent = self.parse_percent(config.get('arm_decelerate'), 0)  # 减速百分比
        self.arm_stop = self.parse_bool(config.get('arm_stop'), False)  # 急停
        self.arm_recover = self.parse_bool(config.get('arm_recover'), False)  # 恢复运动
        # 人机安全距离
        self.arm_safety_distance = self.parse_float(config.get('arm_safety_distance'), 0.2)

        self.json_process(config)

    def json_process(self, config):
        processed_bool_rules = []

        # 急停/恢复：边沿触发，同步立即直发（保 live 立即性；失败时 pending 由发送拍重试）。
        # 急停优先于恢复（同拍双边沿只执行急停）。此处只置位/直发，绝不写 'MM'——
        # 发送成功后由发送路径复位，修复 WS710 每 0.3s 覆盖 pending_stop 的竞态。
        if self.arm_stop and not self.previous_rule_commands['arm_stop']:
            self.send_stop_frame_now(STOP_COMMAND, '急停！')
            processed_bool_rules.append('arm_stop')
        elif self.arm_recover and not self.previous_rule_commands['arm_recover']:
            if self.stop_flag:
                self.get_logger().warn(
                    '忽略 JSON 恢复请求：ROS /emergency_stop '
                    '仍为激活状态'
                )
            else:
                self.send_stop_frame_now(RECOVER_COMMAND, '恢复运动！')
            processed_bool_rules.append('arm_recover')
        self.previous_rule_commands['arm_stop'] = self.arm_stop
        self.previous_rule_commands['arm_recover'] = self.arm_recover

        # 抓取：上升沿触发
        if self.arm_capture and not self.previous_rule_commands['arm_capture']:
            if self.handle_capture_request(self.arm_capture_goal):
                processed_bool_rules.append('arm_capture')
        self.previous_rule_commands['arm_capture'] = self.arm_capture

        # 减速：契约不变（比例 0-1，"1"=全速 -> SPD=100），SPD 字段仅是运载方式（R8/R9）。
        # 启动时 previous_decelerate_percent=None，首个 tick 显式下发一次全速，与 live 等价。
        if self.arm_decelerate_percent > 0:
            if self.arm_decelerate_percent != self.previous_decelerate_percent:
                self.pending_speed = str(self.arm_decelerate_percent)
                self.get_logger().info(f'减速至 {self.arm_decelerate_percent}%！')
                if isinstance(config.get('arm_decelerate'), bool):
                    processed_bool_rules.append('arm_decelerate')
                self.previous_decelerate_percent = self.arm_decelerate_percent
        else:
            # 只复位边沿记忆，不清 pending_speed（已置位的速度命令由发送拍运走）
            self.previous_decelerate_percent = 0

        self.reset_processed_bool_flags(config, processed_bool_rules)

        # 人机安全距离
        arm_safety_distance = Float32()
        arm_safety_distance.data = self.arm_safety_distance
        self.safety_distance_pub.publish(arm_safety_distance)

    def robot_joint_sub(self, observation):
        self.joint_names = ['j1_joint', 'j2_joint', 'j3_joint', 'j4_joint', 'j5_joint', 'j6_joint']
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names

        # 位置
        msg.position = [0.0] * 6

        msg.position[0] = observation[0] * 6.28 / 51200.0 / 30.0
        msg.position[1] = observation[1] * 6.28 / 51200.0 / 30.0
        msg.position[2] = observation[2] * 6.28 / 51200.0 / 30.0
        msg.position[3] = observation[3] * 6.28 / 51200.0 / 10.0
        msg.position[4] = observation[4] * 6.28 / 51200.0 / 10.0
        msg.position[5] = observation[5] * 6.28 / 51200.0

        if self.handeye_mapping_enabled and self.handeye_fk_position is not None:
            try:
                self.current_ee_position = self.handeye_fk_position(
                    np.asarray(msg.position, dtype=np.float64))
            except (TypeError, ValueError):
                self.current_ee_position = None

        self.joint_pub.publish(msg)

    @staticmethod
    def joint_radians_to_encoder_positions(joint_pos, j2_offset_radians=0.0):
        # 关节弧度 -> 下位机编码值（j2 符号与各轴倍率沿用现有约定，勿改）
        return [
            int(joint_pos[0] * 51200.0 / 6.28) * 30,
            -int((joint_pos[1] + j2_offset_radians) * 51200.0 / 6.28) * 30,
            int(joint_pos[2] * 51200.0 / 6.28) * 30,
            int(joint_pos[3] * 51200.0 / 6.28) * 10,
            int(joint_pos[4] * 51200.0 / 6.28) * 10,
            int(joint_pos[5] * 51200.0 / 6.28),
        ]

    def publish_capture_status(self, state, goal_type=None, reason=None, **details):
        payload = {
            'state': state,
            'goal_type': goal_type or self.arm_capture_goal,
        }
        if reason:
            payload['reason'] = reason
        payload.update(details)

        message = String()
        message.data = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        self.capture_status_pub.publish(message)
        self.get_logger().info(f'抓取流程状态: {message.data}')

    def publish_goal_type(self, goal_type):
        message = String()
        message.data = goal_type
        self.goal_type_pub.publish(message)

    def publish_is_capture(self, active):
        message = Bool()
        message.data = bool(active)
        self.arm_capture_pub.publish(message)

    def publish_goal(self, x, y, z):
        goal = Point()
        goal.x = float(x)
        goal.y = float(y)
        goal.z = float(z)
        self.goal_pub.publish(goal)

    def publish_capture_complete(self, success, target_name=None, reason=None):
        # 抓取流程统一收尾：/is_capture 回 False + /arm_capture_status 事件。
        # 所有中止路径都必须经过这里（修 WS710 三个漏发路径导致 /is_capture 卡 True）；
        # reason='interrupted' 沿用 live 的 interrupted 状态词汇。
        self.clear_capture_resume_snapshot()
        self.publish_is_capture(False)
        if success:
            self.publish_capture_status('completed', target_name)
            self.get_logger().info('抓取完成')
            return
        if reason == 'interrupted':
            self.publish_capture_status('interrupted', target_name)
        elif reason:
            self.publish_capture_status('failed', target_name, reason=reason)
        self.get_logger().info('抓取取消')

    def estop_motion_blocked(self):
        return bool(self.stop_flag or self.recover_pending)

    def queue_joint_command(self, joint_positions):
        with self.capture_resume_condition:
            if self.estop_motion_blocked():
                return False
            self.pending_joint_positions = list(joint_positions)
            return True

    def capture_flow_time(self):
        now = time.monotonic()
        with self.capture_resume_condition:
            paused_seconds = self.capture_pause_total_seconds
            if self.capture_pause_started_at is not None:
                paused_seconds += max(0.0, now - self.capture_pause_started_at)
        return now - paused_seconds

    def wait_for_capture_resume(self):
        with self.capture_resume_condition:
            while (
                self.estop_motion_blocked()
                and not self.shutdown_event.is_set()
            ):
                self.capture_resume_condition.wait(timeout=0.1)
        return self.shutdown_event.is_set()

    def begin_capture_resume_stage(
            self, stage_name, duration, joint_positions=None, pump_command=None):
        if (joint_positions is None) == (pump_command is None):
            raise ValueError('抓取恢复阶段必须且只能包含关节目标或气泵命令')

        with self.capture_resume_condition:
            self.capture_resume_generation += 1
            stage_id = self.capture_resume_generation
            self.capture_resume_snapshot = {
                'stage_id': stage_id,
                'stage_name': str(stage_name),
                'duration': float(duration),
                'kind': 'joint' if joint_positions is not None else 'pump',
                'joint_positions': (
                    list(joint_positions) if joint_positions is not None else None
                ),
                'pump_command': pump_command,
                'command_sent': False,
            }
            if not self.estop_motion_blocked():
                if joint_positions is not None:
                    self.queue_joint_command(joint_positions)
                else:
                    self.pending_pump = pump_command
            return stage_id

    def finish_capture_resume_stage(self, stage_id):
        with self.capture_resume_condition:
            snapshot = self.capture_resume_snapshot
            if snapshot is not None and snapshot['stage_id'] == stage_id:
                self.capture_resume_snapshot = None

    def clear_capture_resume_snapshot(self):
        with self.capture_resume_condition:
            self.capture_resume_snapshot = None

    def mark_capture_command_sent(self, joint_positions, pump_command):
        with self.capture_resume_condition:
            snapshot = self.capture_resume_snapshot
            if snapshot is None:
                return
            if (
                snapshot['kind'] == 'joint'
                and joint_positions is not None
                and snapshot['joint_positions'] == list(joint_positions)
            ):
                snapshot['command_sent'] = True
            elif (
                snapshot['kind'] == 'pump'
                and pump_command != UNIFIED_FIELD_NOOP
                and snapshot['pump_command'] == pump_command
            ):
                snapshot['command_sent'] = True
            self.capture_resume_condition.notify_all()

    def complete_estop_recovery(self):
        with self.capture_resume_condition:
            if not self.recover_pending or self.stop_flag:
                return
            now = time.monotonic()
            if self.capture_pause_started_at is not None:
                self.capture_pause_total_seconds += max(
                    0.0, now - self.capture_pause_started_at)
                self.capture_pause_started_at = None
            self.recover_pending = False
            snapshot = self.capture_resume_snapshot
            if self.capture_in_progress and snapshot is not None:
                if snapshot['kind'] == 'joint':
                    snapshot['command_sent'] = False
                    self.pending_joint_positions = list(snapshot['joint_positions'])
                elif not snapshot['command_sent']:
                    self.pending_pump = snapshot['pump_command']
            self.capture_resume_condition.notify_all()

    def sleep_until_shutdown(self, seconds, restart_after_pause=True):
        duration = max(0.0, float(seconds))
        deadline = self.capture_flow_time() + duration
        pause_generation = self.capture_pause_generation

        while not self.shutdown_event.is_set():
            if self.wait_for_capture_resume():
                return True
            current_generation = self.capture_pause_generation
            flow_time = self.capture_flow_time()
            if current_generation != pause_generation:
                pause_generation = current_generation
                if restart_after_pause:
                    deadline = flow_time + duration

            remaining = deadline - flow_time
            if remaining <= 1e-9:
                return False
            if self.shutdown_event.wait(min(remaining, 0.1)):
                return True
        return True

    def handle_capture_request(self, goal_type):
        target_name = str(goal_type).strip().upper()
        if getattr(self, 'handeye_calibration_active', False):
            self.publish_capture_status(
                'failed',
                target_name,
                reason='handeye_calibration_active',
            )
            self.get_logger().warn('标定持有控制权，拒绝抓取请求')
            return True
        if target_name not in GOAL_TYPES:
            self.publish_capture_status(
                'failed',
                target_name,
                reason='unsupported_goal_type',
            )
            self.get_logger().warn(f'未知抓取目标类型: {goal_type}')
            return True

        with self.capture_lock:
            if self.capture_thread and self.capture_thread.is_alive():
                self.publish_capture_status(
                    'failed',
                    target_name,
                    reason='capture_in_progress',
                )
                self.get_logger().warn('抓取流程正在执行，忽略新的抓取请求')
                return True

            self.publish_is_capture(True)
            self.publish_capture_status('accepted', target_name)
            self.capture_thread = threading.Thread(
                target=self.capture_goal_worker,
                args=(target_name,),
                daemon=True,
            )
            self.capture_thread.start()
        return True

    def capture_goal_worker(self, target_name):
        self.capture_in_progress = True
        self.capture_success = False
        try:
            self.publish_capture_status('waiting_for_k230_goal', target_name)
            self.k230_goal_event.clear()
            self.goal_event.clear()
            self.publish_goal_type(target_name)

            pick_point, details = self.wait_for_capture_point(target_name)
            if pick_point is None:
                reason = details.get('reject_reason', 'missing_k230_goal')
                self.get_logger().warn(f'目标 {target_name} 无效、缺失或超出工作空间，取消抓取')
                self.publish_capture_complete(False, target_name, reason)
                return

            self.capture_success = True
            self.run_capture_sequence(target_name, pick_point, details)
        except Exception as exc:
            self.get_logger().error(f'抓取流程异常: {exc}')
            self.publish_capture_complete(False, target_name, f'{type(exc).__name__}: {exc}')
        finally:
            self.capture_in_progress = False

    def wait_for_capture_point(self, target_name):
        # 双源等待抓取目标：K230 合并帧 或 /goal 话题，带截止时间与点级校验
        reject_details = {}
        deadline = self.capture_flow_time() + CAPTURE_TARGET_TIMEOUT_SECONDS
        while (
            self.capture_flow_time() < deadline
            and not self.shutdown_event.is_set()
        ):
            if self.wait_for_capture_resume():
                break
            pixel_goal = list(self.goal_pos[target_name])
            point = self.pixel_goal_to_arm_point(pixel_goal)
            if point is not None:
                if self.is_point_valid(point):
                    if self.wait_for_capture_resume():
                        break
                    return point, {
                        'target_source': 'merged_k230_frame',
                        'pixel_x': float(pixel_goal[0]),
                        'pixel_y': float(pixel_goal[1]),
                    }
                reject_details = {'reject_reason': 'target_out_of_workspace'}

            if self.goal_event.wait(timeout=0.1):
                self.goal_event.clear()
                point = tuple(float(value) for value in self.goal)
                if any(value != 0.0 for value in point):
                    if self.is_point_valid(point):
                        if self.wait_for_capture_resume():
                            break
                        return point, {'target_source': 'goal_topic'}
                    reject_details = {'reject_reason': 'target_out_of_workspace'}

            self.k230_goal_event.wait(timeout=0.1)
            self.k230_goal_event.clear()

        return None, reject_details

    def is_point_valid(self, point):
        # 机械臂坐标点级工作空间校验（WS710 边界值，WS78 融合结构）
        bounds = self.workspace_bounds
        x, y, z = point[0], point[1], point[2]
        if not (bounds['x_min'] <= x <= bounds['x_max']):
            self.get_logger().warn(
                f'X坐标 {x:.3f} 超出工作空间范围 [{bounds["x_min"]:.2f}, {bounds["x_max"]:.2f}]')
            return False
        if not (bounds['y_min'] <= y <= bounds['y_max']):
            self.get_logger().warn(
                f'Y坐标 {y:.3f} 超出工作空间范围 [{bounds["y_min"]:.2f}, {bounds["y_max"]:.2f}]')
            return False
        if not (bounds['z_min'] <= z <= bounds['z_max']):
            self.get_logger().warn(
                f'Z坐标 {z:.3f} 超出工作空间范围 [{bounds["z_min"]:.2f}, {bounds["z_max"]:.2f}]')
            return False
        return True

    def is_goal_valid(self, target_pos):
        # 像素目标有效性：经唯一标定函数换算后做点级校验（与发布路径同一变换）
        point = self.pixel_goal_to_arm_point(target_pos)
        if point is None:
            return False
        return self.is_point_valid(point)

    def prepare_capture_goal_control(self):
        with self.capture_goal_control_lock:
            self.capture_goal_control_active = True
            self.capture_goal_control_pending = True
            self.capture_goal_control_running = False
            with self.capture_resume_condition:
                self.pending_joint_positions = None

    def cancel_capture_goal_control(self):
        with self.capture_goal_control_lock:
            self.capture_goal_control_active = False
            self.capture_goal_control_pending = False
        if self.capture_in_progress:
            self.clear_capture_resume_snapshot()

    def make_capture_partial_joint_positions(self, target_joint_positions):
        partial_joint_positions = list(self.look_joint_positions)
        for joint_index in (0, 3, 4, 5):
            partial_joint_positions[joint_index] = target_joint_positions[joint_index]
        return partial_joint_positions

    def make_capture_second_joint_positions(self, target_joint_positions):
        return list(target_joint_positions)

    def run_capture_goal_control_sequence(self, target_joint_positions):
        try:
            partial_joint_positions = self.make_capture_partial_joint_positions(
                target_joint_positions)
            if self.capture_in_progress:
                self.begin_capture_resume_stage(
                    'pick_goal_partial', 10.0,
                    joint_positions=partial_joint_positions,
                )
            else:
                self.queue_joint_command(partial_joint_positions)
            self.get_logger().info(
                '目标位置第一步：发送 j1,j4,j5,j6，j2,j3 用观察位补齐 '
                f'{partial_joint_positions}'
            )
            first_wait_interrupted = (
                self.sleep_until_shutdown(10.0)
                if self.capture_in_progress
                else self.shutdown_event.wait(10.0)
            )
            if first_wait_interrupted:
                return

            with self.capture_goal_control_lock:
                if self.capture_in_progress and not self.capture_goal_control_active:
                    return

            final_joint_positions = self.make_capture_second_joint_positions(
                target_joint_positions)
            if self.capture_in_progress:
                self.begin_capture_resume_stage(
                    'pick_goal_final', 0.3,
                    joint_positions=final_joint_positions,
                )
            else:
                self.queue_joint_command(final_joint_positions)
            self.get_logger().info(
                f'目标位置第二步：发送全部关节 {final_joint_positions}')
            if self.capture_in_progress:
                self.sleep_until_shutdown(0.3)
            else:
                self.shutdown_event.wait(0.3)
        finally:
            with self.capture_goal_control_lock:
                self.capture_goal_control_active = False
                self.capture_goal_control_running = False

    def run_capture_sequence(self, target_name, pick_point, details):
        # 抓放状态机（WS710 goal_pos_pub 时序 + live /arm_capture_status 事件流）。
        # 修 WS710 缺陷：粗界限拒绝、ik 门控失败、各步中断路径都必须发
        # publish_capture_complete，/is_capture 不得卡 True。
        pos_x = float(pick_point[0])
        pos_y = float(pick_point[1])

        # 粗界限第一道门（WS710 原有，0.38 与 ik_control workspace 对齐）
        if (not (ROUGH_XY_BOUND_MIN <= pos_x <= ROUGH_XY_BOUND_MAX)
                or not (ROUGH_XY_BOUND_MIN <= pos_y <= ROUGH_XY_BOUND_MAX)):
            self.get_logger().warn('目标超出工作空间，取消抓取')
            self.publish_capture_complete(False, target_name, 'target_out_of_workspace')
            return
        # 工作空间复检（与等待阶段共用同一校验/标定函数）
        if not self.is_point_valid(pick_point):
            self.get_logger().warn('目标超出工作空间，取消抓取')
            self.publish_capture_complete(False, target_name, 'target_out_of_workspace')
            return

        status_details = dict(details)
        status_details.update({
            'goal_x': pick_point[0],
            'goal_y': pick_point[1],
            'goal_z': pick_point[2],
        })
        self.publish_capture_status('pick_goal_published', target_name, **status_details)

        # /ik_success 门控握手：先清陈旧锁存值，只认可本次 /goal 发布后的求解结果
        self.ik_success_flag = False
        self.prepare_capture_goal_control()
        self.publish_goal(*pick_point)
        if self.sleep_until_shutdown(PICK_TRAVEL_SECONDS):
            self.cancel_capture_goal_control()
            self.publish_capture_complete(False, target_name, 'interrupted')
            return

        if not self.ik_success_flag:
            # 修 WS710：ik 门控失败也必须收尾（原实现裸 return，/is_capture 卡 True）
            self.cancel_capture_goal_control()
            self.publish_capture_complete(False, target_name, 'ik_gate_failed')
            return
        self.clear_capture_resume_snapshot()

        # 发送吸取命令（PUMP 字段经统一帧运载）
        stage_id = self.begin_capture_resume_stage(
            'pick_pump', PICK_PUMP_SECONDS, pump_command=PICK_COMMAND)
        self.publish_capture_status('pick_command_sent', target_name)
        self.get_logger().info('发送抓取命令(CPQE)')
        if self.sleep_until_shutdown(
                PICK_PUMP_SECONDS, restart_after_pause=False):
            self.publish_capture_complete(False, target_name, 'interrupted')
            return
        self.finish_capture_resume_stage(stage_id)

        # J2 抬升步：按当前关节角就地抬起吸住的物体（WS710 新步骤）
        lift_joint_positions = self.joint_radians_to_encoder_positions(
            self.joint_pos, j2_offset_radians=LIFT_J2_RADIANS)
        stage_id = self.begin_capture_resume_stage(
            'lift', LIFT_TRAVEL_SECONDS,
            joint_positions=lift_joint_positions,
        )
        self.publish_capture_status(
            'lift_joint_command_queued',
            target_name,
            joint_positions=list(lift_joint_positions),
        )
        if self.sleep_until_shutdown(LIFT_TRAVEL_SECONDS):
            self.publish_capture_complete(False, target_name, 'interrupted')
            return
        self.finish_capture_resume_stage(stage_id)

        # 回到中间位置
        stage_id = self.begin_capture_resume_stage(
            'middle', MIDDLE_TRAVEL_SECONDS,
            joint_positions=MIDDLE_JOINT_POSITIONS,
        )
        self.publish_capture_status(
            'middle_joint_command_queued',
            target_name,
            joint_positions=list(MIDDLE_JOINT_POSITIONS),
        )
        if self.sleep_until_shutdown(MIDDLE_TRAVEL_SECONDS):
            self.publish_capture_complete(False, target_name, 'interrupted')
            return
        self.finish_capture_resume_stage(stage_id)

        # 回到中间位置2
        stage_id = self.begin_capture_resume_stage(
            'second_middle', SECOND_MIDDLE_TRAVEL_SECONDS,
            joint_positions=SECOND_MIDDLE_JOINT_POSITIONS,
        )
        self.publish_capture_status(
            'second_middle_joint_command_queued',
            target_name,
            joint_positions=list(SECOND_MIDDLE_JOINT_POSITIONS),
        )
        if self.sleep_until_shutdown(SECOND_MIDDLE_TRAVEL_SECONDS):
            self.publish_capture_complete(False, target_name, 'interrupted')
            return
        self.finish_capture_resume_stage(stage_id)

        # 发送放置命令（停止吸取）
        stage_id = self.begin_capture_resume_stage(
            'place_pump', PLACE_PUMP_SECONDS, pump_command=PLACE_COMMAND)
        self.publish_capture_status('place_command_sent', target_name)
        self.get_logger().info('发送放置命令(PUT)')
        if self.sleep_until_shutdown(
                PLACE_PUMP_SECONDS, restart_after_pause=False):
            self.publish_capture_complete(False, target_name, 'interrupted')
            return
        self.finish_capture_resume_stage(stage_id)

        # 回观察位
        stage_id = self.begin_capture_resume_stage(
            'look', LOOK_TRAVEL_SECONDS,
            joint_positions=self.look_joint_positions,
        )
        self.publish_capture_status(
            'look_joint_command_queued',
            target_name,
            joint_positions=list(self.look_joint_positions),
        )
        if self.sleep_until_shutdown(LOOK_TRAVEL_SECONDS):
            self.publish_capture_complete(False, target_name, 'interrupted')
            return
        self.finish_capture_resume_stage(stage_id)

        self.publish_capture_complete(True, target_name)
        self.ik_success_flag = False

    def process_arm_frame(self, frame):
        joint_angles = parse_arm_joint_frame(frame)
        if joint_angles is None:
            self.get_logger().warn(f'忽略无效ARM帧: {bytes(frame)!r}')
            return
        self.robot_joint_sub(joint_angles)

    def process_k230_frame(self, frame):
        parsed = parse_k230_goal_frame(frame)
        if parsed is None:
            self.get_logger().warn(f'忽略无效K230帧: {bytes(frame)!r}')
            return
        self.goal_pos.update(parsed)
        d_pixel = PointStamped()
        d_pixel.header.stamp = self.get_clock().now().to_msg()
        d_pixel.header.frame_id = 'camera_optical_frame'
        d_pixel.point.x = float(parsed['D'][0])
        d_pixel.point.y = float(parsed['D'][1])
        d_pixel.point.z = 0.0
        self.handeye_d_pub.publish(d_pixel)
        self.k230_goal_event.set()

    def stop_callback(self, msg_data):
        requested_stop = bool(msg_data.data)
        with self.capture_resume_condition:
            if requested_stop == self.stop_flag:
                return
            self.stop_flag = requested_stop
            if requested_stop:
                self.capture_pause_generation += 1
                if self.capture_pause_started_at is None:
                    self.capture_pause_started_at = time.monotonic()
                self.recover_pending = False
                self.capture_resume_condition.notify_all()
            else:
                self.recover_pending = True

        if requested_stop:
            self.send_stop_frame_now(STOP_COMMAND, '急停！')
        else:
            self.send_stop_frame_now(RECOVER_COMMAND, '恢复运动！')

    def process_serial_buffer(self, buffer):
        # 就地消费串口缓冲中的完整帧；半包保留待补齐。返回处理后的缓冲。
        # 采 WS710 的 A..B 紧跟 E..P 合并包处理，并修复其缺陷：K230 半包时先消费
        # 已完整的 ARM 帧，避免下一轮重复处理；保留 live 的缓冲溢出清空保护。
        while buffer:
            if buffer[0] == self.ARM_FRAME_HEADER:
                tail_index = buffer.find(bytes([self.ARM_FRAME_TAIL]), 1)
                if tail_index < 0:
                    break

                next_byte_index = tail_index + 1
                if (next_byte_index < len(buffer)
                        and buffer[next_byte_index] == self.K230_FRAME_HEADER):
                    # 下位机 ARM 帧与 K230 帧粘连的合并包
                    arm_frame = bytes(buffer[:tail_index + 1])
                    k230_tail_index = buffer.find(bytes([self.K230_FRAME_TAIL]), tail_index + 2)
                    if k230_tail_index > tail_index:
                        k230_frame = bytes(buffer[tail_index + 1:k230_tail_index + 1])
                        del buffer[:k230_tail_index + 1]
                        self.process_arm_frame(arm_frame)
                        self.process_k230_frame(k230_frame)
                    else:
                        # K230 半包：先消费完整 ARM 帧再等待补齐（修 WS710 重复处理缺陷）
                        del buffer[:tail_index + 1]
                        self.process_arm_frame(arm_frame)
                        break
                else:
                    frame = bytes(buffer[:tail_index + 1])
                    del buffer[:tail_index + 1]
                    self.process_arm_frame(frame)
            elif buffer[0] == self.K230_FRAME_HEADER:
                tail_index = buffer.find(bytes([self.K230_FRAME_TAIL]), 1)
                if tail_index < 0:
                    break
                frame = bytes(buffer[:tail_index + 1])
                del buffer[:tail_index + 1]
                self.process_k230_frame(frame)
            else:
                del buffer[0]

        if len(buffer) > SERIAL_BUFFER_LIMIT_BYTES:
            self.get_logger().warn('串口缓存过长，已清空')
            buffer = bytearray([])
        return buffer

    # 处理下位机发送的数据
    def receive_data(self):
        # 频率
        rate = 1.0 / 10.0
        buffer = bytearray([])

        while rclpy.ok() and not self.shutdown_event.is_set():
            # 获取缓存区大小
            if not self.serial_is_open():
                time.sleep(rate)
                continue

            try:
                with self.serial_lock:
                    n = self.ser.in_waiting
                    if n > 0:
                        buffer += self.ser.read(n)
            except (serial.SerialException, TypeError, OSError) as exc:
                self.get_logger().error(f'串口读取失败: {exc}')
                break

            buffer = self.process_serial_buffer(buffer)

            time.sleep(rate)

    def control_callback(self, msg_data):
        if getattr(self, 'handeye_calibration_active', False):
            return
        self.handle_control_message(msg_data)

    def handle_control_message(self, msg_data):
        if len(msg_data.position) != 6:
            return
        if self.estop_motion_blocked() and not self.capture_in_progress:
            return
        self.joint_pos = msg_data.position

        # 计算所有关节的编码值
        target_joint_positions = self.joint_radians_to_encoder_positions(self.joint_pos)

        if not self.serial_is_open():
            self.get_logger().warn('串口未连接')
            return

        with self.capture_goal_control_lock:
            if self.capture_in_progress and not self.capture_goal_control_active:
                return
            if self.capture_goal_control_running:
                return

            self.capture_goal_control_active = True
            self.capture_goal_control_pending = False
            self.capture_goal_control_running = True
            self.get_logger().info(f'收到 IK 控制消息，启动分段发送 {target_joint_positions}')
            self.capture_goal_control_thread = threading.Thread(
                target=self.run_capture_goal_control_sequence,
                args=(target_joint_positions,),
                daemon=True,
            )
            self.capture_goal_control_thread.start()

    def goal_callback(self, msg_data):
        self.goal = np.array([msg_data.x, msg_data.y, msg_data.z])
        self.goal_event.set()

    # 连接串口
    def connect_ser(self):
        try:
            self.ser = serial.Serial(port='/dev/ttyUSB0', baudrate=115200, timeout=0.1)
            self.get_logger().info(f'serial open {self.ser.is_open}')
        except serial.SerialException as e:
            self.ser = None
            self.get_logger().warn(f'串口连接失败: {e}')

    def destroy_node(self):
        # 关闭串口
        self.shutdown_event.set()
        if self.look_timer is not None:
            self.look_timer.cancel()
        if self.receive_thread and self.receive_thread.is_alive():
            self.receive_thread.join(timeout=1.0)
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        if self.capture_goal_control_thread and self.capture_goal_control_thread.is_alive():
            self.capture_goal_control_thread.join(timeout=1.0)
        if self.look_thread and self.look_thread.is_alive():
            self.look_thread.join(timeout=1.0)
        if self.ser is not None:
            try:
                self.ser.cancel_read()
                self.ser.close()
            except serial.SerialException as exc:
                self.get_logger().warn(f'关闭串口失败: {exc}')
        super().destroy_node()


def main():
    node = None
    try:
        rclpy.init()
        node = RobotJointPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if node is not None:
            node.get_logger().error(f'arm_state 异常退出: {exc}')
        else:
            print(f'arm_state 初始化失败: {exc}')
        raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
