#!/usr/bin/env python3.10
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import os
import json
import time

from camera.orbbec_loader import load_orbbec

ob = load_orbbec()
print("成功导入 Orbbec 模块")

from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        # 获取包路径
        camera_pkg_path = get_package_share_directory('camera')

        # 模型路径
        model_path = os.path.join(camera_pkg_path, 'models', 'yolo26n.pt')

        self.get_logger().info(f"使用模型: {model_path}")

        # 声明参数
        self.declare_parameter('model_path', model_path)
        self.declare_parameter('conf_threshold', 0.6)
        self.declare_parameter('depth_max_mm', 10000)
        self.declare_parameter('show_window', True)

        # 获取参数
        model_path = self.get_parameter('model_path').value
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.depth_max_mm = self.get_parameter('depth_max_mm').value
        self.show_window = self.get_parameter('show_window').value

        # 初始化 CV 桥接
        self.bridge = CvBridge()

        # 发布者
        self.goal_pub = self.create_publisher(Point, '/goal', 10)
        self.stop_pub = self.create_publisher(Bool, '/emergency_stop', 10)

        # 初始化Orbbec相机 - 使用不同的变量名避免与ROS2内置属性冲突
        self.get_logger().info("初始化Orbbec相机...")
        self.ob_context = None
        self.ob_device = None
        self.ob_device_info = None
        self.ob_cam = None
        self.initialize_camera()

        # 加载 YOLO 模型
        self.get_logger().info(f"正在加载模型: {model_path}")
        try:
            self.yolo_model = YOLO(model_path)
            self.yolo_model.overrides['conf'] = self.conf_threshold
            self.yolo_model.overrides['device'] = 'cpu'
            self.yolo_model.overrides['verbose'] = False
            self.get_logger().info("✅ YOLO 模型加载完成")
        except Exception as e:
            self.get_logger().error(f"❌ YOLO 模型加载失败: {e}")
            raise

        # 加载安全配置
        self.safety_rules_path = os.path.join(camera_pkg_path, 'safety_rules.json')
        self.stop_color = None
        self.load_safety_config()

        # 创建定时器
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10Hz
        self.config_timer = self.create_timer(1.0, self.load_safety_config)

        self.frame_count = 0
        self.fps_start_time = time.time()


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

            self.get_logger().info("✅ 等待相机稳定...")
            time.sleep(2)
            self.get_logger().info("✅ 流水线已启动")

        except Exception as e:
            self.get_logger().error(f"❌ 相机初始化失败: {e}")
            raise

    def load_safety_config(self):
        """加载安全规则配置文件"""
        try:
            if os.path.exists(self.safety_rules_path):
                with open(self.safety_rules_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for rule in data.get('rules', []):
                    if rule.get('id') == 'visual_color_estop':
                        color = rule['trigger']['value']
                        if color != self.stop_color:
                            self.stop_color = color
                            self.get_logger().info(f"更新急停颜色: {color}")
                        return
        except Exception as e:
            self.get_logger().warn(f"加载安全配置失败: {e}")

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

    def timer_callback(self):
        """定时器回调，处理相机帧"""
        try:
            color_img, depth_img = self.get_frames()

            if color_img is None:
                return

            self.frame_count += 1

            # 计算FPS
            if self.frame_count % 30 == 0:
                elapsed = time.time() - self.fps_start_time
                fps = 30 / elapsed if elapsed > 0 else 0
                self.get_logger().info(f"FPS: {fps:.1f}")
                self.fps_start_time = time.time()

            # 颜色检测
            detected_color, center, area = self.detect_color_blocks(color_img)

            # 紧急停止检查
            if self.stop_color and detected_color == self.stop_color:
                stop_msg = Bool()
                stop_msg.data = True
                self.stop_pub.publish(stop_msg)
                self.get_logger().error(f"⚠️ 检测到{detected_color}色块，急停！",
                                      throttle_duration_sec=1.0)
                return
            else:
                stop_msg = Bool()
                stop_msg.data = False
                self.stop_pub.publish(stop_msg)

            # YOLO 人物检测
            if color_img is not None and depth_img is not None:
                try:
                    # 转换为BGR用于YOLO
                    color_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

                    results = self.yolo_model(color_bgr,
                                             classes=[0],
                                             conf=self.conf_threshold,
                                             device='cpu',
                                             verbose=False)

                    if results and results[0].boxes is not None:
                        annotated_img = results[0].plot() if hasattr(results[0], 'plot') else color_bgr
                        boxes = results[0].boxes
                        xyxy = boxes.xyxy.cpu().numpy()

                        for i, box in enumerate(xyxy):
                            x1, y1, x2, y2 = map(int, box)
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                            if 0 <= cx < 640 and 0 <= cy < 480:
                                depth_mm = depth_img[cy, cx]
                                if 0 < depth_mm < self.depth_max_mm:
                                    try:
                                        point3d = self.ob_cam.get3DPoint(cx, cy, depth_mm)
                                        if point3d and len(point3d) == 3:
                                            X, Y, Z = point3d

                                            if max(abs(X), abs(Y), abs(Z)) <= 3000:
                                                point_msg = Point()
                                                point_msg.x = X / 1000.0
                                                point_msg.y = -Y / 1000.0
                                                point_msg.z = Z / 1000.0
                                                self.goal_pub.publish(point_msg)

                                                self.get_logger().info(
                                                    f"目标: ({X/1000:.2f}, {Y/1000:.2f}, {Z/1000:.2f})m",
                                                    throttle_duration_sec=1.0
                                                )
                                    except Exception as e:
                                        self.get_logger().warn(f"获取3D坐标失败: {e}")

                        if self.show_window:
                            cv2.imshow('YOLO Detection', annotated_img)

                            if depth_img is not None:
                                depth_display = self.depth_to_colormap(depth_img)
                                cv2.imshow('Depth', depth_display)

                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('q'):
                                self.get_logger().info("用户按Q退出")
                                rclpy.shutdown()
                    else:
                        if self.show_window:
                            cv2.imshow('YOLO Detection', color_bgr)
                            cv2.waitKey(1)

                except Exception as e:
                    self.get_logger().error(f"YOLO检测错误: {e}")

        except Exception as e:
            self.get_logger().error(f"回调函数错误: {e}")

    def detect_color_blocks(self, frame):
        """检测颜色色块"""
        if frame is None:
            return None, None, 0

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        color_ranges = {
            'yellow': [([20, 50, 50], [30, 255, 255])],
            'purple': [([125, 50, 50], [155, 255, 255])]
        }

        min_area = 500
        kernel = np.ones((5, 5), np.uint8)

        best_color = None
        best_area = 0

        for color_name, ranges in color_ranges.items():
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)

            for lower, upper in ranges:
                mask_part = cv2.inRange(hsv, np.array(lower), np.array(upper))
                mask = cv2.bitwise_or(mask, mask_part)

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area and area > best_area:
                    best_area = area
                    best_color = color_name

        return best_color, None, best_area

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
            if self.ob_cam:
                self.ob_cam.stopPipeline()
                self.ob_cam = None
                self.get_logger().info("相机流水线已停止")
        except Exception as e:
            self.get_logger().error(f"停止相机时出错: {e}")

        cv2.destroyAllWindows()
        super().destroy_node()

    def pos_init(self):
        """初始化位置"""
        self.get_logger().info("初始化位置...")
        pass



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
