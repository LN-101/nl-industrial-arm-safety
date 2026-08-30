import sys
from threading import Thread

from geometry_msgs.msg import Point
from PyQt5 import QtWidgets
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32

from .main_window import MainWindow


JOINT_NAMES = [
    'j1_joint',
    'j2_joint',
    'j3_joint',
    'j4_joint',
    'j5_joint',
    'j6_joint',
]


class GuiNode(Node):
    """ROS 2 bridge for the local arm control GUI."""

    def __init__(self):
        super().__init__('arm_gui_node')

        self.goal_pub = self.create_publisher(Point, '/goal', 10)
        self.control_pub = self.create_publisher(JointState, '/control', 10)
        self.stop_pub = self.create_publisher(Bool, '/emergency_stop', 10)

        self.create_subscription(
            JointState,
            '/robot_joint_state',
            self._robot_joint_callback,
            10,
        )
        self.create_subscription(
            JointState,
            '/mujoco_joint_state',
            self._mujoco_joint_callback,
            10,
        )
        self.create_subscription(
            Float32,
            '/min_distance',
            self._min_distance_callback,
            10,
        )

        self.robot_joint_callback = None
        self.mujoco_joint_callback = None
        self.min_distance_callback = None
        self.get_logger().info('Arm GUI node started')

    def publish_goal(self, x, y, z):
        msg = Point()
        msg.x = float(x)
        msg.y = float(y)
        msg.z = float(z)
        self.goal_pub.publish(msg)
        self.get_logger().info(
            f'Published /goal: x={msg.x:.3f}, y={msg.y:.3f}, z={msg.z:.3f}'
        )

    def publish_joint_command(self, positions):
        if len(positions) != len(JOINT_NAMES):
            raise ValueError('joint command must contain exactly 6 positions')

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = [float(position) for position in positions]
        self.control_pub.publish(msg)
        self.get_logger().info('Published manual /control command')

    def publish_emergency_stop(self, enabled):
        msg = Bool()
        msg.data = bool(enabled)
        self.stop_pub.publish(msg)
        state = 'enabled' if msg.data else 'released'
        self.get_logger().warn(f'Emergency stop {state}')

    def _robot_joint_callback(self, msg):
        if self.robot_joint_callback is not None:
            self.robot_joint_callback(msg)

    def _mujoco_joint_callback(self, msg):
        if self.mujoco_joint_callback is not None:
            self.mujoco_joint_callback(msg)

    def _min_distance_callback(self, msg):
        if self.min_distance_callback is not None:
            self.min_distance_callback(msg.data)


def main(args=None):
    rclpy.init(args=args)
    ros_node = GuiNode()

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName('Arm GUI')

    window = MainWindow(ros_node)
    window.show()

    executor = MultiThreadedExecutor()
    executor.add_node(ros_node)
    spin_thread = Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        exit_code = app.exec_()
    finally:
        executor.shutdown()
        spin_thread.join(timeout=1.0)
        ros_node.destroy_node()
        rclpy.shutdown()

    return exit_code


if __name__ == '__main__':
    sys.exit(main())
