"""ROS2 bridge from K230 serial target pixels to arm /goal points."""

import math
import threading
import time

from geometry_msgs.msg import Point

import rclpy
from rclpy.node import Node

import serial

from std_msgs.msg import String


SERIAL_PORT = '/dev/ttyUSB1'
SERIAL_BAUDRATE = 115200
FRAME_HEADER = 'E'
FRAME_TAIL = 'P'
GOAL_TYPES = ('A', 'B', 'C', 'D')


def parse_k230_frame(frame_data):
    """Parse one K230 frame into an optional goal type and pixel goal."""
    text = bytes(frame_data).decode('utf-8', errors='ignore').strip()
    if not text.startswith(FRAME_HEADER) or not text.endswith(FRAME_TAIL):
        return None

    body = text[1:-1]
    goal_type = None
    if len(body) >= 2 and body[0].upper() in GOAL_TYPES and body[1] == ',':
        goal_type = body[0].upper()
        body = body[2:]

    parts = [part.strip() for part in body.split(',')]
    if len(parts) < 2:
        return None

    try:
        pixel_goal = [float(parts[0]), float(parts[1])]
    except ValueError:
        return None
    if not all(math.isfinite(value) for value in pixel_goal):
        return None
    return goal_type, pixel_goal


def pixel_goal_to_point(pixel_goal):
    """Convert a K230 image-space pixel goal into the ROS /goal Point."""
    if len(pixel_goal) < 2:
        return None
    pixel_x = float(pixel_goal[0])
    pixel_y = float(pixel_goal[1])
    if not math.isfinite(pixel_x) or not math.isfinite(pixel_y):
        return None
    if pixel_x == 0.0 and pixel_y == 0.0:
        return None

    point = Point()
    point.x = (13.0 - pixel_x * 26.0 / 640.0) / 100.0 + 0.08
    point.y = (7.5 - pixel_y * 15.0 / 480.0) / 100.0 + 0.3
    point.z = 0.1
    return point


class RobotJointPublisher(Node):
    """Publish K230 serial target coordinates as ROS /goal messages."""

    def __init__(self):
        """Initialize publishers, subscriptions, and serial reception."""
        super().__init__('k230_node')
        self.ser = None
        self.receive_thread = None
        self.shutdown_event = threading.Event()

        self.goal_pub = self.create_publisher(Point, '/goal', 1)
        self.goal_type_sub = self.create_subscription(
            String,
            '/goal_type',
            self.goal_type_callback,
            1,
        )

        self.goal_pos = {goal_type: [0.0, 0.0] for goal_type in GOAL_TYPES}

        self.connect_ser()
        if self.serial_is_open():
            self.receive_thread = threading.Thread(
                target=self.receive_data,
                daemon=True,
            )
            self.receive_thread.start()
        else:
            self.get_logger().warn('K230串口未连接，跳过目标接收线程')

    def goal_type_callback(self, msg):
        """Publish the latest stored pixel goal for a target type."""
        goal_type = str(msg.data or '').strip().upper()
        pixel_goal = self.goal_pos.get(goal_type)
        if pixel_goal is None:
            self.get_logger().warn(f'未知抓取目标类型: {msg.data}')
            return
        if not self.publish_goal_from_pixel(
            pixel_goal,
            f'goal_type {goal_type}',
        ):
            self.get_logger().debug(f'目标 {goal_type} 暂无有效K230坐标')

    def receive_data(self):
        """Read K230 serial data and dispatch complete frames."""
        rate = 1.0 / 10.0
        buffer = bytearray()

        while rclpy.ok() and not self.shutdown_event.is_set():
            if not self.serial_is_open():
                time.sleep(rate)
                continue

            waiting = self.ser.in_waiting
            if waiting > 0:
                buffer += self.ser.read(waiting)

            while ord(FRAME_TAIL) in buffer:
                tail_index = buffer.index(ord(FRAME_TAIL))
                frame = buffer[:tail_index + 1]
                buffer = buffer[tail_index + 1:]
                self.handle_frame(frame)

            if len(buffer) > 64:
                self.get_logger().warn('K230串口缓存过长，已清空')
                buffer = bytearray()
            time.sleep(rate)

    def handle_frame(self, frame):
        """Handle one parsed K230 frame and store typed target coordinates."""
        parsed = parse_k230_frame(frame)
        if parsed is None:
            self.get_logger().warn(f'忽略无效K230帧: {bytes(frame)!r}')
            return

        goal_type, pixel_goal = parsed
        if goal_type in self.goal_pos:
            self.goal_pos[goal_type] = pixel_goal

    def look_pos(self):
        """Return the arm camera view to the fixed observation position."""
        goal = Point()
        goal.x = 0.08
        goal.y = 0.25
        goal.z = 0.2
        self.goal_pub.publish(goal)
        time.sleep(8.0)

    def publish_goal_from_pixel(self, pixel_goal, source):
        """Publish one K230 pixel goal after converting it to arm space."""
        goal = pixel_goal_to_point(pixel_goal)
        if goal is None:
            return False

        self.get_logger().info(
            f'发布K230目标({source}): x={goal.x:.3f} y={goal.y:.3f}'
        )
        self.goal_pub.publish(goal)
        return True

    def connect_ser(self):
        """Open the K230 serial device if it is available."""
        try:
            self.ser = serial.Serial(
                port=SERIAL_PORT,
                baudrate=SERIAL_BAUDRATE,
                timeout=0.1,
            )
            self.get_logger().info(f'K230串口已打开: {SERIAL_PORT}')
        except serial.SerialException as error:
            self.ser = None
            self.get_logger().error(f'K230串口打开失败 {SERIAL_PORT}: {error}')

    def serial_is_open(self):
        """Return whether the K230 serial connection is currently open."""
        return (
            self.ser is not None
            and bool(getattr(self.ser, 'is_open', False))
        )

    def destroy_node(self):
        """Stop the receive thread and close the K230 serial connection."""
        self.shutdown_event.set()
        if self.serial_is_open():
            self.ser.cancel_read()
            self.ser.close()
        if self.receive_thread is not None:
            self.receive_thread.join(timeout=1.0)
        super().destroy_node()


def main(args=None):
    """Run the K230 ROS2 node."""
    node = None
    try:
        rclpy.init(args=args)
        node = RobotJointPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\n用户中断')
    except Exception as error:
        print(f'Error: {error}')
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
