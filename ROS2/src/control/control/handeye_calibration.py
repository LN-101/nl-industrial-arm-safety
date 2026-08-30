from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from control.handeye_calibration_io import (
    fit_layer_robust,
    HandeyeCalibration,
    save_calibration_atomic,
    WS712_OUTPUT_CORRECTION_MATRIX,
    WS712_OUTPUT_OFFSET_XY,
)
from control.handeye_calibration_state import (
    CalibrationStateMachine,
    TERMINAL_STATES,
)
from control.ik_control import enforce_joint_limits, fk_position, JOINT_NAMES, urdf_path
from geometry_msgs.msg import PointStamped
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

IK_WORKSPACE = {
    'x': (-0.22, 0.38),
    'y': (-0.22, 0.38),
    'z': (0.05, 0.33),
}
MAPPING_WORKSPACE_MIN = np.array([-0.32, -0.32, 0.05])
MAPPING_WORKSPACE_MAX = np.array([0.38, 0.38, 0.30])
OWNER_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
)


class HandeyeCalibrationNode(Node):
    def __init__(self):
        super().__init__('handeye_calibration')
        self.declare_parameter('motion_enabled', False)
        self.declare_parameter('output_file', '~/.ros/ai_ov/handeye_xy.yaml')
        self.declare_parameter('x_values', [0.06, 0.08, 0.10])
        self.declare_parameter('y_values', [0.235, 0.24, 0.25])
        self.declare_parameter('calibration_z', 0.30)
        self.declare_parameter('known_target_xy', [0.08, 0.33])
        self.declare_parameter('output_anchor_xy', [0.08, 0.33])
        self.declare_parameter(
            'output_correction_matrix',
            WS712_OUTPUT_CORRECTION_MATRIX.reshape(-1).tolist(),
        )
        self.declare_parameter(
            'output_offset_xy',
            WS712_OUTPUT_OFFSET_XY.tolist(),
        )
        self.declare_parameter('image_center', [540.0, 360.0])
        self.declare_parameter('tool_to_camera_xy', [0.0, 0.0])
        self.declare_parameter('activation_delay', 0.5)
        self.declare_parameter('ik_timeout', 15.0)
        self.declare_parameter('arrival_timeout', 60.0)
        self.declare_parameter('return_timeout', 60.0)
        self.declare_parameter('settle_time', 6.0)
        self.declare_parameter('detection_timeout', 10.0)
        self.declare_parameter('position_tolerance', 0.012)
        self.declare_parameter('pixel_stability_std', 2.0)
        self.declare_parameter('samples_per_pose', 12)
        self.declare_parameter('max_fit_rmse', 0.003)

        self.motion_enabled = bool(self.get_parameter('motion_enabled').value)
        self.output_file = Path(str(self.get_parameter('output_file').value)).expanduser()
        self.x_values = np.asarray(self.get_parameter('x_values').value, dtype=np.float64)
        self.y_values = np.asarray(self.get_parameter('y_values').value, dtype=np.float64)
        self.calibration_z = float(self.get_parameter('calibration_z').value)
        self.known_target_xy = np.asarray(
            self.get_parameter('known_target_xy').value,
            dtype=np.float64,
        )
        self.output_anchor_xy = np.asarray(
            self.get_parameter('output_anchor_xy').value,
            dtype=np.float64,
        )
        self.output_correction_matrix = np.asarray(
            self.get_parameter('output_correction_matrix').value,
            dtype=np.float64,
        ).reshape(2, 2)
        self.output_offset_xy = np.asarray(
            self.get_parameter('output_offset_xy').value,
            dtype=np.float64,
        )
        self.image_center = np.asarray(
            self.get_parameter('image_center').value,
            dtype=np.float64,
        )
        self.tool_offset = np.asarray(
            self.get_parameter('tool_to_camera_xy').value,
            dtype=np.float64,
        )
        for name in (
                'activation_delay', 'ik_timeout', 'arrival_timeout',
                'return_timeout', 'settle_time', 'detection_timeout',
                'position_tolerance', 'pixel_stability_std',
                'max_fit_rmse'):
            setattr(self, name, float(self.get_parameter(name).value))
        self.samples_per_pose = int(self.get_parameter('samples_per_pose').value)
        self._validate_parameters()

        self.machine = CalibrationStateMachine()
        self.estop_active = False
        self.capture_active = False
        self.current_ee = None
        self.last_pixel = None
        self.last_pixel_time = None
        self.pixel_buffer = []
        self.start_position = None
        self.target_position = None
        self.poses = []
        self.pose_index = 0
        self.sample_ee = []
        self.sample_pixels = []
        self.next_seq = 1
        self.return_reason = None

        self.goal_pub = self.create_publisher(String, '/handeye/goal_request', 10)
        self.active_pub = self.create_publisher(
            Bool,
            '/handeye/calibration_active',
            OWNER_QOS,
        )
        self.status_pub = self.create_publisher(String, '/handeye/status', 10)
        self.create_subscription(
            String, '/handeye/ik_result', self.ik_result_callback, 10)
        self.create_subscription(
            JointState, '/robot_joint_state', self.joint_callback, 20)
        self.create_subscription(
            PointStamped, '/handeye/d_pixel', self.pixel_callback, 20)
        self.create_subscription(Bool, '/emergency_stop', self.estop_callback, 10)
        self.create_subscription(Bool, '/is_capture', self.capture_callback, 10)
        self.create_service(Trigger, '/handeye/start', self.start_callback)
        self.create_service(Trigger, '/handeye/cancel', self.cancel_callback)
        self.timer = self.create_timer(0.05, self.update)
        self.publish_active(False)
        self.publish_status('idle')

    def _validate_parameters(self):
        if not self.motion_enabled:
            return
        for values in (self.x_values, self.y_values):
            if values.ndim != 1 or len(values) < 3 or not np.all(np.diff(values) > 0.0):
                raise ValueError('x_values and y_values must be increasing vectors')
        if not all(np.all(np.isfinite(value)) for value in (
                self.x_values, self.y_values, self.known_target_xy,
                self.output_anchor_xy, self.output_correction_matrix,
                self.output_offset_xy, self.image_center, self.tool_offset)):
            raise ValueError('calibration parameters must be finite')
        for x_value in self.x_values:
            for y_value in self.y_values:
                if not self.in_ik_workspace((x_value, y_value, self.calibration_z)):
                    raise ValueError('calibration pose is outside IK workspace')

    @staticmethod
    def in_ik_workspace(position):
        return all(
            IK_WORKSPACE[axis][0] <= position[index] <= IK_WORKSPACE[axis][1]
            for index, axis in enumerate(('x', 'y', 'z'))
        )

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def publish_active(self, active):
        message = Bool()
        message.data = bool(active)
        self.active_pub.publish(message)

    def publish_status(self, state, reason=None, **details):
        message = String()
        payload = {'state': state, 'pose_index': self.pose_index, **details}
        if reason:
            payload['reason'] = reason
        message.data = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        self.status_pub.publish(message)

    def joint_callback(self, message):
        if len(message.position) < len(JOINT_NAMES):
            return
        if message.name:
            positions = dict(zip(message.name, message.position))
            if not all(name in positions for name in JOINT_NAMES):
                return
            joints = [positions[name] for name in JOINT_NAMES]
        else:
            joints = message.position[:len(JOINT_NAMES)]
        self.current_ee = fk_position(
            enforce_joint_limits(np.asarray(joints, dtype=np.float64)))

    def pixel_callback(self, message):
        pixel = np.asarray([message.point.x, message.point.y], dtype=np.float64)
        if not np.all(np.isfinite(pixel)) or np.allclose(pixel, 0.0):
            return
        self.last_pixel = pixel
        self.last_pixel_time = self.now()
        if self.machine.state == 'collecting':
            self.pixel_buffer.append(pixel)
            self.pixel_buffer = self.pixel_buffer[-self.samples_per_pose:]

    def estop_callback(self, message):
        self.estop_active = bool(message.data)
        if self.estop_active and self.machine.state not in {'idle', 'estopped'}:
            self.machine.estop()
            self.publish_active(False)
            self.publish_status('estopped', 'emergency_stop')

    def capture_callback(self, message):
        self.capture_active = bool(message.data)

    def start_callback(self, _request, response):
        if not self.motion_enabled:
            response.success = False
            response.message = 'motion_enabled=false；拒绝机械臂标定运动'
            return response
        if self.estop_active or self.capture_active:
            response.success = False
            response.message = '急停或抓取流程处于活动状态'
            return response
        if self.current_ee is None or not self.pixel_is_fresh():
            response.success = False
            response.message = '缺少实时关节状态或 D 像素'
            return response
        try:
            self.machine.start(self.now(), self.activation_delay)
        except RuntimeError as exc:
            response.success = False
            response.message = str(exc)
            return response
        self.start_position = self.current_ee.copy()
        self.poses = self.make_poses()
        self.pose_index = 0
        self.sample_ee = []
        self.sample_pixels = []
        self.return_reason = None
        self.publish_active(True)
        self.publish_status('activating')
        response.success = True
        response.message = f'开始 {len(self.poses)} 点标定'
        return response

    def cancel_callback(self, _request, response):
        if self.machine.state in TERMINAL_STATES:
            response.success = False
            response.message = '当前没有活动标定'
            return response
        if self.estop_active:
            self.machine.estop()
            self.publish_active(False)
            response.success = True
            response.message = '急停中止；未发布返回动作'
            return response
        self.return_reason = 'cancelled'
        self.request_position(self.start_position, returning=True)
        response.success = True
        response.message = '取消标定并请求返回起点'
        return response

    def make_poses(self):
        poses = []
        for row_index, y_value in enumerate(self.y_values):
            x_values = self.x_values if row_index % 2 == 0 else self.x_values[::-1]
            for x_value in x_values:
                poses.append(np.array([x_value, y_value, self.calibration_z]))
        return poses

    def request_position(self, position, returning=False):
        if position is None or self.estop_active or not self.in_ik_workspace(position):
            self.fail('unsafe_goal_request', allow_return=False)
            return
        seq = self.next_seq
        self.next_seq += 1
        self.target_position = np.asarray(position, dtype=np.float64)
        if returning:
            self.machine.begin_return(seq, self.now(), self.return_timeout)
        else:
            self.machine.request_ik(seq, self.now(), self.ik_timeout)
        message = String()
        message.data = json.dumps({
            'seq': seq,
            'x': float(position[0]),
            'y': float(position[1]),
            'z': float(position[2]),
        }, sort_keys=True)
        self.goal_pub.publish(message)
        self.publish_status(self.machine.state, seq=seq)

    def ik_result_callback(self, message):
        try:
            result = json.loads(message.data)
            seq = result['seq']
            success = result['success']
            if not isinstance(seq, int) or not isinstance(success, bool):
                raise ValueError
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return
        if self.machine.state == 'returning':
            if seq != self.machine.active_seq:
                return
            if not success:
                self.fail('return_ik_failed', allow_return=False)
            return
        accepted = self.machine.accept_ik_result(
            seq, success, self.now(), self.arrival_timeout)
        if accepted:
            self.publish_status(self.machine.state, seq=seq)
            if self.machine.state == 'failed':
                self.fail(self.machine.failure_reason or 'ik_failed')

    def pixel_is_fresh(self):
        return self.last_pixel_time is not None and self.now() - self.last_pixel_time <= 1.0

    def update(self):
        now = self.now()
        state = self.machine.state
        if state in TERMINAL_STATES:
            return
        if self.estop_active:
            self.estop_callback(type('Bool', (), {'data': True})())
            return
        if self.machine.timed_out(now):
            reason = 'return_timeout' if state == 'returning' else f'{state}_timeout'
            self.fail(reason, allow_return=False if state == 'returning' else True)
            return
        if self.machine.activation_ready(now):
            self.request_position(self.poses[self.pose_index])
            return
        if state in {'waiting_arrival', 'returning'} and self.current_ee is not None:
            if np.linalg.norm(self.current_ee - self.target_position) <= self.position_tolerance:
                if state == 'returning':
                    self.publish_active(False)
                    if self.return_reason == 'completed':
                        self.machine.complete()
                    else:
                        self.machine.fail(self.return_reason or 'returned_after_failure')
                    self.publish_status(self.machine.state, self.machine.failure_reason)
                else:
                    self.machine.arrived(now, self.settle_time)
                    self.publish_status('settling')
            return
        if state == 'settling' and self.machine.begin_sampling(now, self.detection_timeout):
            self.pixel_buffer = []
            self.publish_status('collecting')
            return
        if state == 'collecting':
            if not self.pixel_is_fresh() or len(self.pixel_buffer) < self.samples_per_pose:
                return
            pixels = np.asarray(self.pixel_buffer)
            if np.max(np.std(pixels, axis=0)) > self.pixel_stability_std:
                return
            self.sample_ee.append(self.current_ee.copy())
            self.sample_pixels.append(np.median(pixels, axis=0))
            self.machine.sample_complete()
            self.pose_index += 1
            if self.pose_index < len(self.poses):
                self.request_position(self.poses[self.pose_index])
            else:
                self.finish_fit()

    def finish_fit(self):
        fit = fit_layer_robust(
            np.asarray(self.sample_ee)[:, :2],
            np.asarray(self.sample_pixels),
            max_rmse=self.max_fit_rmse,
        )
        if fit is None or fit.rmse_m > self.max_fit_rmse or fit.condition_number > 1e4:
            self.fail('untrusted_fit')
            return
        inliers = np.asarray(fit.inlier_indices)
        ee_xy = np.asarray(self.sample_ee)[inliers, :2]
        pixels = np.asarray(self.sample_pixels)[inliers]
        unanchored = (
            ee_xy + self.tool_offset
            - (fit.matrix @ (pixels - self.image_center).T).T
        )
        offset = np.mean(self.known_target_xy - unanchored, axis=0)
        fit_outliers = [
            {
                'z_m': self.calibration_z,
                'actual_ee_xy': np.asarray(self.sample_ee)[index, :2].tolist(),
                'pixel': np.asarray(self.sample_pixels)[index].tolist(),
                'reason': 'robust fit rejected sample',
            }
            for index in fit.rejected_indices
        ]
        calibration = HandeyeCalibration(
            z_layers=np.array([self.calibration_z]),
            layer_matrices=np.array([fit.matrix]),
            layer_offsets=np.array([offset]),
            image_center=self.image_center,
            tool_to_camera_xy=self.tool_offset,
            workspace_min=MAPPING_WORKSPACE_MIN,
            workspace_max=MAPPING_WORKSPACE_MAX,
            layer_quality=({
                'z_m': self.calibration_z,
                'fit_rmse_m': fit.rmse_m,
                'condition_number': fit.condition_number,
                'collected_count': len(self.sample_pixels),
                'fitted_count': len(fit.inlier_indices),
                'rejected_count': len(fit.rejected_indices),
            },),
            metadata={
                'created_at': datetime.now(timezone.utc).isoformat(),
                'urdf_sha256': hashlib.sha256(Path(urdf_path).read_bytes()).hexdigest(),
                'collected_sample_count': len(self.sample_pixels),
                'rejected_sample_indices': list(fit.rejected_indices),
            },
            pixel_to_base_delta=fit.matrix,
            base_xy_offset=offset,
            known_target_xy=self.known_target_xy,
            output_anchor_xy=self.output_anchor_xy,
            output_correction_matrix=self.output_correction_matrix,
            output_offset_xy=self.output_offset_xy,
            calibration_workspace={
                'x': [float(self.x_values[0]), float(self.x_values[-1])],
                'y': [float(self.y_values[0]), float(self.y_values[-1])],
                'z': [self.calibration_z, self.calibration_z],
            },
            sampled_layer_y_ranges=((
                float(self.y_values[0]),
                float(self.y_values[-1]),
            ),),
            fit_statistics={
                'collected_sample_count': len(self.sample_pixels),
                'fitted_sample_count': len(fit.inlier_indices),
                'fit_outlier_count': len(fit.rejected_indices),
                'fit_outliers': fit_outliers,
                'sample_count': len(fit.inlier_indices),
                'skipped_count': 0,
                'skipped_poses': [],
            },
        )
        try:
            save_calibration_atomic(self.output_file, calibration)
        except (OSError, TypeError, ValueError) as exc:
            self.get_logger().error(f'保存标定产物失败: {exc}')
            self.fail('calibration_save_failed')
            return
        self.return_reason = 'completed'
        self.request_position(self.start_position, returning=True)

    def fail(self, reason, allow_return=True):
        self.publish_status('failed', reason)
        if allow_return and not self.estop_active and self.start_position is not None:
            self.return_reason = reason
            self.request_position(self.start_position, returning=True)
            return
        self.machine.fail(reason)
        self.publish_active(False)


def main(args=None):
    rclpy.init(args=args)
    node = HandeyeCalibrationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
