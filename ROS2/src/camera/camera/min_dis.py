#!/usr/bin/env python3.10
import json
import os
import threading
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from std_msgs.msg import Float32, String
from std_srvs.srv import Trigger

from camera.distance_estop import (
    build_distance_estop_payload,
    LatchedDistanceEstop,
    parse_reset_intent,
    resolve_distance_class_ids,
)
from camera.pose_distance import (
    compute_min_distance_between_sets,
    DEFAULT_DEPTH_RECOVERY_CONFIG,
    DepthRecoveryConfig,
    DetectionInstanceAssociator,
    PoseDepthResolver,
    PosePoint3D,
    select_pose_keypoints_by_class,
)
from camera.vision_context import (
    DEFAULT_VISION_FRAME_ID,
    DEFAULT_VISION_OUTPUT_DIR,
    DEFAULT_VISION_SERVICE_NAME,
    DEFAULT_VISION_SOURCE,
    build_context_payload,
    finite_float_or_none,
    normalize_image_extension,
    snapshot_timestamp,
    write_rgb_snapshot,
)

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
warnings.filterwarnings('ignore', message='CUDA initialization:.*', category=UserWarning)

from camera.orbbec_loader import load_orbbec

ob = load_orbbec()
from ultralytics import YOLO


DEFAULT_CPU_THREADS = 4
DETECTION_PERIOD_SECONDS = 0.1
DEFAULT_OPENVINO_MODEL_PATH = (
    '/home/inteldk/ROS2/src/camera/models/yolo26n_openvino_model'
)
DEFAULT_INFERENCE_DEVICE = 'intel:npu'
DEFAULT_KEYPOINT_CONFIDENCE_THRESHOLD = 0.30
OPENVINO_MODEL_TASK = 'pose'
OPENVINO_WARMUP_IMAGE_SIZE = 640


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        # 声明参数
        self.declare_parameter('model_path', DEFAULT_OPENVINO_MODEL_PATH)
        self.declare_parameter('inference_device', DEFAULT_INFERENCE_DEVICE)
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('depth_max_mm', 10000)
        self.declare_parameter(
            'depth_recovery_absolute_tolerance_mm',
            DEFAULT_DEPTH_RECOVERY_CONFIG.absolute_tolerance_mm,
        )
        self.declare_parameter(
            'depth_recovery_relative_tolerance',
            DEFAULT_DEPTH_RECOVERY_CONFIG.relative_tolerance,
        )
        self.declare_parameter(
            'depth_recovery_min_candidates',
            DEFAULT_DEPTH_RECOVERY_CONFIG.min_candidates,
        )
        self.declare_parameter(
            'depth_recovery_max_mad_mm',
            DEFAULT_DEPTH_RECOVERY_CONFIG.max_mad_mm,
        )
        self.declare_parameter(
            'direct_depth_uncertainty_mm',
            DEFAULT_DEPTH_RECOVERY_CONFIG.direct_uncertainty_mm,
        )
        self.declare_parameter(
            'recovered_depth_uncertainty_mm',
            DEFAULT_DEPTH_RECOVERY_CONFIG.recovered_uncertainty_mm,
        )
        self.declare_parameter(
            'depth_history_max_age_ms',
            DEFAULT_DEPTH_RECOVERY_CONFIG.history_max_age_seconds * 1000.0,
        )
        self.declare_parameter(
            'depth_history_pixel_gate_px',
            DEFAULT_DEPTH_RECOVERY_CONFIG.history_pixel_gate_px,
        )
        self.declare_parameter(
            'detection_association_max_center_distance_ratio',
            (
                DEFAULT_DEPTH_RECOVERY_CONFIG
                .association_max_center_distance_ratio
            ),
        )
        self.declare_parameter(
            'detection_association_min_center_distance_px',
            DEFAULT_DEPTH_RECOVERY_CONFIG.association_min_center_distance_px,
        )
        self.declare_parameter('show_window', True)
        self.declare_parameter('cpu_threads', DEFAULT_CPU_THREADS)
        self.declare_parameter(
            'keypoint_conf_threshold',
            DEFAULT_KEYPOINT_CONFIDENCE_THRESHOLD,
        )
        self.declare_parameter(
            'detection_image_path',
            '/home/inteldk/ROS2/Log/latest_detection.jpg',
        )  # 检测画面实时保存路径（覆盖写，只保留最新一张）
        self.declare_parameter('estop_request_topic', '/safety/estop/request')
        self.declare_parameter('estop_source', 'min_distance_camera')
        self.declare_parameter(
            'estop_release_margin_m',
            0.05,
        )
        self.declare_parameter(
            'estop_release_safe_frames',
            3,
        )
        self.declare_parameter(
            'estop_release_max_age_seconds',
            0.5,
        )
        self.declare_parameter(
            'estop_no_distance_release_seconds',
            5.0,
        )
        self.declare_parameter('vision_service_name', DEFAULT_VISION_SERVICE_NAME)
        self.declare_parameter('vision_output_dir', str(DEFAULT_VISION_OUTPUT_DIR))
        self.declare_parameter('vision_max_age_ms', 1000.0)
        self.declare_parameter('vision_image_extension', 'jpg')
        self.declare_parameter('vision_jpeg_quality', 92)
        self.declare_parameter('vision_frame_id', DEFAULT_VISION_FRAME_ID)
        self.declare_parameter('vision_source', DEFAULT_VISION_SOURCE)

        # 获取参数
        model_path = Path(str(self.get_parameter('model_path').value)).expanduser()
        self.inference_device = str(
            self.get_parameter('inference_device').value
        ).strip().lower()
        if not model_path.is_dir():
            raise FileNotFoundError(f'OpenVINO 模型目录不存在: {model_path}')
        if not any(model_path.glob('*.xml')) or not any(model_path.glob('*.bin')):
            raise FileNotFoundError(
                f'OpenVINO 模型目录缺少 .xml 或 .bin 文件: {model_path}'
            )
        if self.inference_device.lower() != DEFAULT_INFERENCE_DEVICE:
            raise ValueError(
                'inference_device 必须为 intel:npu，'
                f'当前值: {self.inference_device!r}'
            )
        self.get_logger().info(f'使用 OpenVINO 模型: {model_path}')
        self.get_logger().info(f'YOLO 推理设备: {self.inference_device}')
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.depth_max_mm = self.get_parameter('depth_max_mm').value
        self.show_window = self.get_parameter('show_window').value
        self.cpu_threads = int(self.get_parameter('cpu_threads').value)
        if self.cpu_threads <= 0:
            raise ValueError('cpu_threads must be greater than 0')
        import torch

        torch.set_num_threads(self.cpu_threads)
        torch.set_num_interop_threads(1)
        self.get_logger().info(
            f'YOLO 前后处理 CPU 线程限制: intra-op={torch.get_num_threads()}, '
            f'inter-op={torch.get_num_interop_threads()}'
        )
        self.keypoint_conf_threshold = float(
            self.get_parameter('keypoint_conf_threshold').value
        )
        if not 0.0 <= self.keypoint_conf_threshold <= 1.0:
            raise ValueError('keypoint_conf_threshold must be between 0 and 1')
        self.depth_recovery_config = DepthRecoveryConfig(
            depth_max_mm=float(self.depth_max_mm),
            absolute_tolerance_mm=float(
                self.get_parameter(
                    'depth_recovery_absolute_tolerance_mm'
                ).value
            ),
            relative_tolerance=float(
                self.get_parameter('depth_recovery_relative_tolerance').value
            ),
            min_candidates=int(
                self.get_parameter('depth_recovery_min_candidates').value
            ),
            max_mad_mm=float(
                self.get_parameter('depth_recovery_max_mad_mm').value
            ),
            direct_uncertainty_mm=float(
                self.get_parameter('direct_depth_uncertainty_mm').value
            ),
            recovered_uncertainty_mm=float(
                self.get_parameter('recovered_depth_uncertainty_mm').value
            ),
            history_max_age_seconds=float(
                self.get_parameter('depth_history_max_age_ms').value
            ) / 1000.0,
            history_pixel_gate_px=float(
                self.get_parameter('depth_history_pixel_gate_px').value
            ),
            association_max_center_distance_ratio=float(
                self.get_parameter(
                    'detection_association_max_center_distance_ratio'
                ).value
            ),
            association_min_center_distance_px=float(
                self.get_parameter(
                    'detection_association_min_center_distance_px'
                ).value
            ),
        )
        self.detection_associator = DetectionInstanceAssociator(
            max_age_seconds=(
                self.depth_recovery_config.history_max_age_seconds
            ),
            max_center_distance_ratio=(
                self.depth_recovery_config
                .association_max_center_distance_ratio
            ),
            min_center_distance_px=(
                self.depth_recovery_config.association_min_center_distance_px
            ),
        )
        self.depth_resolver = PoseDepthResolver(self.depth_recovery_config)
        self.get_logger().info(
            '深度恢复参数: '
            f'历史={self.depth_recovery_config.history_max_age_seconds * 1000:.0f}ms, '
            f'参考门限=max({self.depth_recovery_config.absolute_tolerance_mm:.0f}mm, '
            f'{self.depth_recovery_config.relative_tolerance:.1%}), '
            f'MAD<={self.depth_recovery_config.max_mad_mm:.0f}mm'
        )
        self.detection_image_path = str(self.get_parameter('detection_image_path').value)
        try:
            os.makedirs(os.path.dirname(self.detection_image_path) or '.', exist_ok=True)
        except Exception as e:
            self.get_logger().warn(f"创建检测图片保存目录失败（实时保存将不可用）: {e}")
        self.get_logger().info(f"图片将实时保存到: {self.detection_image_path}")
        self.estop_request_topic = self.get_parameter('estop_request_topic').value
        self.estop_source = self.get_parameter('estop_source').value
        self.estop_release_margin_m = float(
            self.get_parameter('estop_release_margin_m').value
        )
        self.estop_release_safe_frames = int(
            self.get_parameter('estop_release_safe_frames').value
        )
        self.estop_release_max_age_seconds = float(
            self.get_parameter('estop_release_max_age_seconds').value
        )
        self.estop_no_distance_release_seconds = float(
            self.get_parameter('estop_no_distance_release_seconds').value
        )
        self.vision_service_name = str(self.get_parameter('vision_service_name').value)
        self.vision_output_dir = Path(
            str(self.get_parameter('vision_output_dir').value)
        ).expanduser()
        self.vision_max_age_ms = float(self.get_parameter('vision_max_age_ms').value)
        self.vision_image_extension = normalize_image_extension(
            self.get_parameter('vision_image_extension').value
        )
        self.vision_jpeg_quality = int(self.get_parameter('vision_jpeg_quality').value)
        self.vision_frame_id = str(self.get_parameter('vision_frame_id').value)
        self.vision_source = str(self.get_parameter('vision_source').value)
        if self.vision_max_age_ms <= 0:
            raise ValueError('vision_max_age_ms must be greater than 0')
        self.vision_jpeg_quality = max(1, min(self.vision_jpeg_quality, 100))

        # 安全距离阈值（单位：米）以及相机急停锁存状态。
        self.stop_distance_threshold = 0.2
        self.last_logged_stop_distance_threshold = self.stop_distance_threshold
        self.distance_estop = LatchedDistanceEstop(
            release_margin_m=self.estop_release_margin_m,
            required_safe_frames=self.estop_release_safe_frames,
            max_evidence_age_seconds=self.estop_release_max_age_seconds,
            no_distance_release_seconds=(
                self.estop_no_distance_release_seconds
            ),
        )

        # 发布者
        self.estop_request_pub = self.create_publisher(String, self.estop_request_topic, 10)
        self.min_distance_pub = self.create_publisher(Float32, '/min_distance', 10)
        self.arm_point_pub = self.create_publisher(PointStamped, '/arm_closest_point', 10)
        self.human_point_pub = self.create_publisher(PointStamped, '/human_closest_point', 10)
        self.get_logger().info(f"急停请求将发布到: {self.estop_request_topic}")
        self._vision_lock = threading.Lock()
        self._latest_rgb_frame = None
        self._latest_context = None
        self._latest_context_monotonic = None
        self._latest_context_wall_time = None

        self.vision_context_service = self.create_service(
            Trigger,
            self.vision_service_name,
            self.capture_vision_context_callback,
        )
        self.get_logger().info(
            f"视觉上下文快照服务: {self.vision_service_name} -> {self.vision_output_dir}"
        )

        # 订阅者
        self.arm_safety_distance_sub = self.create_subscription(
            Float32,
            '/arm_safety_distance',
            self.arm_safety_distance_callback,
            1,
        )
        self.estop_request_sub = self.create_subscription(
            String,
            self.estop_request_topic,
            self.estop_request_callback,
            10,
        )

        # 初始化Orbbec相机
        self.get_logger().info("初始化Orbbec相机...")
        self.ob_context = None
        self.ob_device = None
        self.ob_device_info = None
        self.ob_cam = None
        self.initialize_camera()

        # 加载 OpenVINO YOLO 模型并严格要求使用 NPU。
        self.get_logger().info(f"正在加载模型: {model_path}")
        try:
            import openvino as ov

            available_devices = ov.Core().available_devices
            if 'NPU' not in available_devices:
                raise RuntimeError(
                    'OpenVINO NPU 不可用，当前可用设备: '
                    f'{available_devices}'
                )

            self.yolo_model = YOLO(str(model_path), task=OPENVINO_MODEL_TASK)
            self.yolo_model.overrides['conf'] = self.conf_threshold
            self.yolo_model.overrides['device'] = self.inference_device
            self.yolo_model.overrides['verbose'] = False

            warmup_image = np.zeros(
                (OPENVINO_WARMUP_IMAGE_SIZE, OPENVINO_WARMUP_IMAGE_SIZE, 3),
                dtype=np.uint8,
            )
            self.yolo_model(
                warmup_image,
                conf=self.conf_threshold,
                device=self.inference_device,
                verbose=False,
            )

            compiled_model = self.yolo_model.predictor.model.ov_compiled_model
            execution_devices = compiled_model.get_property('EXECUTION_DEVICES')
            if isinstance(execution_devices, str):
                execution_device_names = [execution_devices]
            else:
                execution_device_names = [str(device) for device in execution_devices]
            if not any('NPU' in device.upper() for device in execution_device_names):
                raise RuntimeError(
                    'OpenVINO 未在 NPU 上执行，实际执行设备: '
                    f'{execution_device_names}'
                )

            self.person_class, self.arm_classes = resolve_distance_class_ids(
                self.yolo_model.names
            )
            keypoint_names = getattr(
                self.yolo_model.predictor.model,
                'kpt_names',
                {},
            )
            self.keypoint_counts_by_class = {
                int(class_id): len(names)
                for class_id, names in keypoint_names.items()
            }
            required_keypoint_classes = {
                self.person_class,
                *self.arm_classes,
            }
            missing_keypoint_classes = sorted(
                required_keypoint_classes - self.keypoint_counts_by_class.keys()
            )
            if missing_keypoint_classes:
                raise RuntimeError(
                    'YOLO 模型元数据缺少类别关键点名称: '
                    f'{missing_keypoint_classes}'
                )
            self.get_logger().info(
                '✅ OpenVINO YOLO 模型加载完成 '
                f'(NPU模式: {execution_device_names})'
            )
            self.get_logger().info(
                'YOLO 安全类别: '
                f'person={self.person_class}, '
                f'arm={sorted(self.arm_classes)}'
            )
            self.get_logger().info(
                'YOLO 关键点数量: '
                f'{self.keypoint_counts_by_class}, '
                f'置信度阈值={self.keypoint_conf_threshold:.2f}'
            )
        except Exception as e:
            self.get_logger().error(f"❌ YOLO 模型加载失败: {e}")
            raise

        # 创建定时器
        self.timer = self.create_timer(DETECTION_PERIOD_SECONDS, self.timer_callback)

        self.frame_count = 0
        self.fps_start_time = time.time()

        self.get_logger().info("✅ CameraNode 初始化完成")
        self.get_logger().info(
            "📸 检测画面保存已启用（与 WS710 一致：仅在未检出人/机械臂且 show_window 时覆盖写最新一帧）")

    def arm_safety_distance_callback(self, msg):
        previous_threshold = self.stop_distance_threshold
        self.stop_distance_threshold = msg.data
        if (
            abs(
                self.stop_distance_threshold
                - self.last_logged_stop_distance_threshold
            )
            > 1e-6
        ):
            self.get_logger().info(
                f'收到安全距离: {self.stop_distance_threshold:.3f}m'
            )
            self.last_logged_stop_distance_threshold = self.stop_distance_threshold
        if abs(self.stop_distance_threshold - previous_threshold) > 1e-6:
            self.distance_estop.invalidate_safety_evidence(
                clear_distance_history=True
            )

    def estop_request_callback(self, msg):
        """Evaluate an explicit reset intent addressed to the camera source."""
        intent, error = parse_reset_intent(
            str(getattr(msg, 'data', '')),
            str(self.estop_source or 'min_distance_camera'),
        )
        if error is not None:
            self.get_logger().warn(
                f'忽略无效相机急停复位请求: {error}'
            )
            return
        if intent is None:
            return

        decision = self.distance_estop.request_reset(self.stop_distance_threshold)
        requester = intent.requester_source
        if decision.cleared:
            reason = f'camera emergency stop reset accepted from {requester}'
            self.publish_estop_request(
                False,
                reason,
                min_distance=self.distance_estop.latest_distance_m,
            )
            self.get_logger().info(
                f'✅ 相机急停已解除：{decision.reason}，请求来源={requester}'
            )
            return

        if decision.accepted:
            self.get_logger().info(
                f'相机急停已处于解除状态，请求来源={requester}'
            )
            return

        reason = f'camera emergency stop reset rejected: {decision.reason}'
        self.publish_estop_request(
            True,
            reason,
            min_distance=self.distance_estop.latest_distance_m,
        )
        self.get_logger().warn(
            f'⛔ 拒绝解除相机急停：{decision.reason}，请求来源={requester}'
        )

    def save_image(self, image):
        """实时保存图片（覆盖保存，只保留最新的一张）；写失败只告警，不抛异常"""
        try:
            if image is None:
                return False

            # 使用固定的文件名，每次覆盖
            save_path = self.detection_image_path

            # 保存图片
            success = cv2.imwrite(save_path, image)
            if not success:
                self.get_logger().warn(
                    f"保存图片失败: {save_path}",
                    throttle_duration_sec=5.0,
                )
            return bool(success)
        except Exception as e:
            self.get_logger().warn(
                f"保存图片时出错: {e}",
                throttle_duration_sec=5.0,
            )
            return False


    def publish_estop_request(self, active, reason, min_distance=None):
        if not rclpy.ok():
            return

        payload = build_distance_estop_payload(
            source=str(self.estop_source or 'min_distance_camera'),
            active=bool(active),
            reason=reason,
            threshold_m=float(self.stop_distance_threshold),
            distance_m=min_distance,
            trigger_distance_m=self.distance_estop.trigger_distance_m,
            release_distance_m=self.distance_estop.release_gate_m(
                self.stop_distance_threshold
            ),
        )

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.estop_request_pub.publish(msg)


    def cache_vision_context(
        self,
        color_img,
        *,
        min_distance,
        human_closest_point,
        arm_closest_point,
        emergency_stop,
        reason,
    ):
        """Cache the latest completed frame analysis for on-demand snapshots."""
        rgb_frame = np.asarray(color_img)
        if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
            self.get_logger().warn(
                f"跳过视觉上下文缓存，RGB帧形状无效: {rgb_frame.shape}"
            )
            return

        context = {
            'min_distance_m': finite_float_or_none(min_distance),
            'human_closest_point': human_closest_point,
            'arm_closest_point': arm_closest_point,
            'emergency_stop': bool(emergency_stop),
            'threshold_m': finite_float_or_none(self.stop_distance_threshold),
            'reason': str(reason or ''),
            'width': int(rgb_frame.shape[1]),
            'height': int(rgb_frame.shape[0]),
        }
        with self._vision_lock:
            self._latest_rgb_frame = rgb_frame.copy()
            self._latest_context = context
            self._latest_context_monotonic = time.monotonic()
            self._latest_context_wall_time = time.time()


    def capture_vision_context_callback(self, _request, response):
        with self._vision_lock:
            if self._latest_rgb_frame is None:
                rgb_frame = None
            else:
                rgb_frame = self._latest_rgb_frame.copy()
            context = None
            if self._latest_context is not None:
                context = dict(self._latest_context)
            captured_monotonic = self._latest_context_monotonic
            captured_wall_time = self._latest_context_wall_time

        if rgb_frame is None or context is None or captured_monotonic is None:
            response.success = False
            response.message = 'No fresh vision context has been computed yet.'
            return response

        age_ms = (time.monotonic() - captured_monotonic) * 1000.0
        if age_ms > self.vision_max_age_ms:
            response.success = False
            response.message = (
                f'Latest vision context is stale: {age_ms:.1f}ms old '
                f'(limit {self.vision_max_age_ms:.1f}ms).'
            )
            return response

        filename = (
            f'vision_context_{int(time.time() * 1000)}.'
            f'{self.vision_image_extension}'
        )
        image_path = self.vision_output_dir / filename
        try:
            image_path = write_rgb_snapshot(
                rgb_frame,
                image_path,
                image_extension=self.vision_image_extension,
                jpeg_quality=self.vision_jpeg_quality,
            ).resolve()
            response.message = build_context_payload(
                image_path=image_path,
                stamp=snapshot_timestamp(captured_wall_time),
                frame_id=self.vision_frame_id,
                min_distance_m=context.get('min_distance_m'),
                human_closest_point=context.get('human_closest_point'),
                arm_closest_point=context.get('arm_closest_point'),
                emergency_stop=bool(context.get('emergency_stop')),
                fresh=True,
                age_ms=age_ms,
                source=self.vision_source,
                width=int(context.get('width') or rgb_frame.shape[1]),
                height=int(context.get('height') or rgb_frame.shape[0]),
                reason=str(context.get('reason') or ''),
                threshold_m=context.get('threshold_m'),
            )
        except Exception as error:
            response.success = False
            response.message = (
                f'Failed to capture vision context: {type(error).__name__}: '
                f'{error}'
            )
            return response

        response.success = True
        return response


    def initialize_camera(self):
        """初始化Orbbec相机"""
        try:
            success, self.ob_context, self.ob_device, self.ob_device_info = ob.ob_init()
            if not success:
                raise RuntimeError("Orbbec SDK初始化失败")

            sn = self.ob_device_info.get_sn()
            fw_version = self.ob_device_info.get_firmware_version()
            self.get_logger().info(f"✅ 设备初始化成功 - SN: {sn}, 固件: {fw_version}")

            self.ob_cam = ob.OB()
            if not self.ob_cam.startPipeline(self.ob_device):
                raise RuntimeError("启动流水线失败")

            time.sleep(2)

        except Exception as e:
            self.get_logger().error(f"❌ 相机初始化失败: {e}")
            raise



    def get_frames(self):
        """获取彩色帧和深度帧"""
        try:
            if self.ob_cam is None:
                return None, None

            depth_frame = self.ob_cam.getDepthFrame()
            color_frame = self.ob_cam.getColorFrame()

            return color_frame, depth_frame
        except Exception as e:
            self.get_logger().error(f"获取帧失败: {e}")
            return None, None

    def get_points_3d(
        self,
        depth_img,
        keypoints,
        timestamp_seconds,
        color_image_shape,
    ):
        """Convert identified keypoints to quality-tagged 3D points.

        Args:
            depth_img: 深度图像
            keypoints: PoseKeypoint列表
            timestamp_seconds: 当前帧的单调时钟时间
        Returns:
            valid_points: PosePoint3D列表
        """
        valid_points = []

        resolved_depths = self.depth_resolver.resolve(
            depth_img,
            keypoints,
            timestamp_seconds,
            source_image_shape=color_image_shape,
        )
        for keypoint, measurement in resolved_depths:
            if not measurement.valid:
                continue

            x, y = keypoint.pixel

            try:
                point3d = self.ob_cam.get3DPoint(
                    x,
                    y,
                    float(measurement.depth_mm),
                )
                if point3d and len(point3d) == 3:
                    X, Y, Z = point3d
                    X_m, Y_m, Z_m = X/1000.0, Y/1000.0, Z/1000.0
                    if abs(X_m) < 3 and abs(Y_m) < 3 and 0.1 < Z_m < 3:
                        valid_points.append(
                            PosePoint3D(
                                keypoint=keypoint,
                                xyz_m=(X_m, Y_m, Z_m),
                                measurement=measurement,
                            )
                        )
            except Exception as e:
                self.get_logger().debug(f"get3DPoint失败: {e}")
                continue

        return valid_points

    def timer_callback(self):
        """定时器回调 - 检测人机最短距离并发布急停请求，按需保存检测画面"""
        try:
            color_img, depth_img = self.get_frames()

            if color_img is None or depth_img is None:
                self.distance_estop.invalidate_safety_evidence()
                if self.distance_estop.latched:
                    self.publish_estop_request(
                        True,
                        'camera frame unavailable; emergency stop remains latched',
                    )
                return
            frame_geometry = (color_img.shape[:2], depth_img.shape[:2])
            if frame_geometry != getattr(self, '_last_frame_geometry', None):
                self._last_frame_geometry = frame_geometry
                self.get_logger().info(
                    '彩色关键点到深度图坐标映射: '
                    f'color={color_img.shape[1]}x{color_img.shape[0]} -> '
                    f'depth={depth_img.shape[1]}x{depth_img.shape[0]}'
                )

            self.frame_count += 1

            # 计算FPS
            if self.frame_count % 30 == 0:
                elapsed = time.time() - self.fps_start_time
                fps = 30 / elapsed if elapsed > 0 else 0
                self.get_logger().debug(f"FPS: {fps:.1f}")
                self.fps_start_time = time.time()

            # YOLO 检测
            frame_timestamp = time.monotonic()
            color_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
            results = self.yolo_model(color_bgr,
                                     conf=self.conf_threshold,
                                     device=self.inference_device,
                                     verbose=False)

            # 按检测类别提取模型实际输出的 pose 关键点。
            person_detected = False
            arm_detected = False
            human_keypoints_2d = []
            arm_keypoints_2d = []
            if results and results[0].boxes is not None:
                class_ids = results[0].boxes.cls.cpu().numpy().astype(np.int64)
                person_detected = bool(np.any(class_ids == self.person_class))
                arm_detected = any(
                    np.any(class_ids == arm_class)
                    for arm_class in self.arm_classes
                )
                if results[0].keypoints is not None:
                    detection_instance_ids = self.detection_associator.assign(
                        class_ids,
                        results[0].boxes.xyxy.cpu().numpy(),
                        frame_timestamp,
                    )
                    keypoint_xy = results[0].keypoints.xy.cpu().numpy()
                    keypoint_confidence = results[0].keypoints.conf
                    if keypoint_confidence is not None:
                        keypoint_confidence = keypoint_confidence.cpu().numpy()
                    human_keypoints_2d, arm_keypoints_2d = (
                        select_pose_keypoints_by_class(
                            class_ids=class_ids,
                            keypoint_xy=keypoint_xy,
                            keypoint_confidence=keypoint_confidence,
                            keypoint_counts_by_class=(
                                self.keypoint_counts_by_class
                            ),
                            person_class_id=self.person_class,
                            arm_class_ids=self.arm_classes,
                            confidence_threshold=(
                                self.keypoint_conf_threshold
                            ),
                            detection_instance_ids=detection_instance_ids,
                            image_shape=results[0].orig_shape,
                        )
                    )

            # 目标丢失不再自动解除已经触发的相机急停。
            if not person_detected or not arm_detected:
                missing_reason = (
                    'No person detected'
                    if not person_detected
                    else 'No arm detected'
                )
                emergency_stop = self.distance_estop.observe_distance(
                    None,
                    self.stop_distance_threshold,
                )
                reason = missing_reason
                if emergency_stop:
                    reason = (
                        f'{missing_reason}; camera emergency stop remains latched'
                    )
                self.publish_estop_request(emergency_stop, reason)
                self.cache_vision_context(
                    color_img,
                    min_distance=None,
                    human_closest_point=None,
                    arm_closest_point=None,
                    emergency_stop=emergency_stop,
                    reason=reason,
                )

                if self.show_window:
                    annotated_img = (
                        results[0].plot()
                        if results and results[0].boxes is not None
                        else color_bgr
                    )
                    status_text = (
                        'STOP LATCHED' if emergency_stop else 'Waiting...'
                    )
                    if not emergency_stop and not person_detected:
                        status_text = "No person detected"
                    elif not emergency_stop and not arm_detected:
                        status_text = "No arm detected"
                    status_color = (
                        (0, 0, 255)
                        if emergency_stop
                        else (255, 255, 255)
                    )
                    cv2.putText(annotated_img, status_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                    cv2.imshow('Safety Detection', annotated_img)
                    # cv2.imshow('Depth', depth_display)

                    # 实时保存最新一帧（未检出目标时；与 WS710 行为一致）
                    self.save_image(annotated_img)

                    cv2.waitKey(1)
                return

            # 将两类有效关键点转换到三维并计算全部点对中的最短距离。
            human_points_3d = self.get_points_3d(
                depth_img,
                human_keypoints_2d,
                frame_timestamp,
                color_img.shape,
            )
            arm_points_3d = self.get_points_3d(
                depth_img,
                arm_keypoints_2d,
                frame_timestamp,
                color_img.shape,
            )
            min_distance, closest_human, closest_arm = (
                compute_min_distance_between_sets(
                    human_points_3d,
                    arm_points_3d,
                )
            )
            closest_human_point = None
            closest_arm_point = None
            closest_human_2d = None
            closest_arm_2d = None
            if closest_human is not None:
                closest_human_2d = closest_human.pixel
                closest_human_point = closest_human.xyz_m
            if closest_arm is not None:
                closest_arm_2d = closest_arm.pixel
                closest_arm_point = closest_arm.xyz_m

            # 发布最短距离
            dist_msg = Float32()
            dist_msg.data = min_distance
            self.min_distance_pub.publish(dist_msg)

            # 发布最近的点
            if closest_human_point is not None:
                person_msg = PointStamped()
                person_msg.header.stamp = self.get_clock().now().to_msg()
                person_msg.header.frame_id = "camera_frame"
                person_msg.point.x = closest_human_point[0]
                person_msg.point.y = closest_human_point[1]
                person_msg.point.z = closest_human_point[2]
                self.human_point_pub.publish(person_msg)

            if closest_arm_point is not None:
                arm_msg = PointStamped()
                arm_msg.header.stamp = self.get_clock().now().to_msg()
                arm_msg.header.frame_id = "camera_frame"
                arm_msg.point.x = closest_arm_point[0]
                arm_msg.point.y = closest_arm_point[1]
                arm_msg.point.z = closest_arm_point[2]
                self.arm_point_pub.publish(arm_msg)

            # 安全策略：触发后保持，距离历史只增加解除资格，不自动清除。
            observed_distance = (
                min_distance if np.isfinite(min_distance) else None
            )
            emergency_stop = self.distance_estop.observe_distance(
                observed_distance,
                self.stop_distance_threshold,
            )
            if emergency_stop:
                if observed_distance is None:
                    reason = (
                        'distance unavailable; '
                        'camera emergency stop remains latched'
                    )
                elif min_distance < self.stop_distance_threshold:
                    reason = f"person distance below threshold: {min_distance:.3f}m"
                else:
                    release_gate = self.distance_estop.release_gate_m(
                        self.stop_distance_threshold
                    )
                    if release_gate is None:
                        reason = (
                            'camera emergency stop latched; release gate '
                            'unavailable; explicit reset required'
                        )
                    else:
                        reason = (
                            'camera emergency stop latched; retained distances '
                            f'{len(self.distance_estop.distance_history_m)}/'
                            f'{self.distance_estop.distance_history_size}, '
                            f'latest {self.estop_release_safe_frames} must be '
                            f'>= {release_gate:.3f}m; explicit reset required'
                        )
                self.publish_estop_request(
                    True,
                    reason,
                    min_distance=observed_distance,
                )
                self.cache_vision_context(
                    color_img,
                    min_distance=observed_distance,
                    human_closest_point=closest_human_point,
                    arm_closest_point=closest_arm_point,
                    emergency_stop=True,
                    reason=reason,
                )
                if (
                    observed_distance is not None
                    and min_distance < self.stop_distance_threshold
                ):
                    self.get_logger().warn(
                        f"🚨 紧急停止！机械臂距离人: {min_distance:.3f}m",
                        throttle_duration_sec=0.5
                    )
                else:
                    self.get_logger().warn(
                        f"🔒 急停保持，等待确认解除：{reason}",
                        throttle_duration_sec=1.0,
                    )
            elif observed_distance is not None:
                reason = f"person distance safe: {min_distance:.3f}m"
                self.publish_estop_request(
                    False,
                    reason,
                    min_distance=min_distance,
                )
                self.cache_vision_context(
                    color_img,
                    min_distance=min_distance,
                    human_closest_point=closest_human_point,
                    arm_closest_point=closest_arm_point,
                    emergency_stop=False,
                    reason=reason,
                )
                if min_distance < 0.5:
                    self.get_logger().info(
                        f"⚠️ 警告：机械臂距离人 {min_distance:.3f}m",
                        throttle_duration_sec=1.0
                    )
                else:
                    self.get_logger().debug(
                        f"✅ 安全，距离: {min_distance:.3f}m",
                        throttle_duration_sec=1.0
                    )
            else:
                reason = 'distance unavailable'
                self.publish_estop_request(False, reason)
                self.cache_vision_context(
                    color_img,
                    min_distance=None,
                    human_closest_point=closest_human_point,
                    arm_closest_point=closest_arm_point,
                    emergency_stop=False,
                    reason=reason,
                )

            # 显示图像
            if self.show_window:
                annotated_img = (
                    results[0].plot(kpt_radius=0, kpt_line=False)
                    if results and results[0].boxes is not None
                    else color_bgr
                )

                # 只突出经过类别数量、置信度和坐标过滤的真实 pose 关键点。
                for point in human_keypoints_2d:
                    cv2.circle(
                        annotated_img,
                        point.pixel,
                        4,
                        (255, 128, 0),
                        -1,
                    )
                for point in arm_keypoints_2d:
                    cv2.circle(
                        annotated_img,
                        point.pixel,
                        4,
                        (0, 255, 255),
                        -1,
                    )

                if min_distance < float('inf'):
                    # 显示距离
                    if emergency_stop:
                        color = (0, 0, 255)  # 红色-停止
                        status = "STOP LATCHED"
                    elif min_distance < 0.5:
                        color = (0, 255, 255)  # 黄色-警告
                        status = "WARNING"
                    else:
                        color = (0, 255, 0)  # 绿色-安全
                        status = "SAFE"

                    distance_text = (
                        f'Min Distance: {min_distance:.3f}m '
                        f'(H:{len(human_points_3d)} A:{len(arm_points_3d)} kpts)'
                    )
                    cv2.putText(
                        annotated_img,
                        distance_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                    )
                    cv2.putText(annotated_img, status, (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    # 在最近的人点和机械臂点之间画线
                    if closest_human_2d and closest_arm_2d:
                        cv2.line(annotated_img, closest_human_2d,
                                closest_arm_2d, color, 2)

                        # 在最近点位置画大圆（突出显示）
                        cv2.circle(annotated_img, closest_human_2d, 12, (0, 0, 255), -1)
                        cv2.circle(annotated_img, closest_arm_2d, 10, (0, 0, 255), -1)

                        # 显示距离值
                        mid_point = ((closest_human_2d[0] + closest_arm_2d[0]) // 2,
                                    (closest_human_2d[1] + closest_arm_2d[1]) // 2)
                        cv2.putText(annotated_img, f"{min_distance:.2f}m", mid_point,
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                cv2.imshow('Safety Detection', annotated_img)

                # 与 WS710 一致：检测主路径不写盘、不显示深度图（20Hz 下逐帧 jpg 写盘
                # 开销大；按需取当前画面请走 vision_context 快照服务）
                # depth_display = self.depth_to_colormap(depth_img)
                # cv2.imshow('Depth', depth_display)
                # self.save_image(annotated_img)

                cv2.waitKey(1)

        except Exception as e:
            self.distance_estop.invalidate_safety_evidence()
            self.get_logger().error(f"回调函数错误: {e}")
            import traceback
            traceback.print_exc()

    def depth_to_colormap(self, depth_image):
        """深度图转伪彩色图"""
        if depth_image is None:
            return None
        depth_clipped = np.clip(depth_image, 0, self.depth_max_mm)
        normalized = ((depth_clipped / self.depth_max_mm) * 255).clip(0, 255).astype(np.uint8)
        return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

    def destroy_node(self):
        """清理资源"""
        try:
            if getattr(self, 'timer', None) is not None:
                self.timer.cancel()
                self.destroy_timer(self.timer)
                self.timer = None
        except Exception as e:
            if rclpy.ok():
                self.get_logger().error(f"停止定时器时出错: {e}")
            else:
                print(f"停止定时器时出错: {e}")

        try:
            if self.ob_cam:
                self.ob_cam.stopPipeline()
                self.ob_cam = None
                if rclpy.ok():
                    self.get_logger().info("相机流水线已停止")
                else:
                    print("相机流水线已停止")
        except Exception as e:
            if rclpy.ok():
                self.get_logger().error(f"停止相机时出错: {e}")
            else:
                print(f"停止相机时出错: {e}")

        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CameraNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
