import sys
import os
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
import signal
import atexit
import numpy as np
import mujoco
import mujoco.viewer as viewer
from ament_index_python.packages import get_package_share_directory


JOINT_NAMES = ['j1_joint', 'j2_joint', 'j3_joint', 'j4_joint', 'j5_joint', 'j6_joint']
CONTROL_PERIOD = 0.01
DEFAULT_MAX_JOINT_SPEED = 1.0
OWNER_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
)


def rate_limit_joint_control(commanded, target, max_joint_speed, period):
    if max_joint_speed <= 0.0:
        raise ValueError('max_joint_speed must be positive')
    max_delta = max_joint_speed * period
    return commanded + np.clip(target - commanded, -max_delta, max_delta)


class SimulationStepScheduler:
    def __init__(self, control_period, simulation_timestep):
        if control_period <= 0.0 or simulation_timestep <= 0.0:
            raise ValueError('control period and simulation timestep must be positive')
        self.control_period = control_period
        self.simulation_timestep = simulation_timestep
        self.remaining_time = 0.0

    def next_step_count(self):
        self.remaining_time += self.control_period
        timing_tolerance = self.simulation_timestep * 1e-12
        step_count = int(
            (self.remaining_time + timing_tolerance) / self.simulation_timestep
        )
        if step_count == 0:
            return 1
        self.remaining_time -= step_count * self.simulation_timestep
        return step_count


class MujocoJointPublisher(Node):
    def __init__(self):
        super().__init__('mujoco_sim')
        self.viewer_active = False
        self.publisher_ = self.create_publisher(JointState, 'mujoco_joint_state', 10)
        self.goal_sub = self.create_subscription(Point, '/goal', self.goal_callback, 10)
        self.joint_control_sub = self.create_subscription(JointState, '/control', self.joint_control_callback, 10)
        self.handeye_control_sub = self.create_subscription(
            JointState,
            '/handeye/control',
            self.handeye_control_callback,
            10,
        )
        self.handeye_active_sub = self.create_subscription(
            Bool,
            '/handeye/calibration_active',
            self.handeye_active_callback,
            OWNER_QOS,
        )
        model_pkg_path = get_package_share_directory('arm_asset')

        # 模型路径
        xml_path = os.path.join(model_pkg_path, 'mjcf', 'arm_mjcf.xml')

        # 加载MuJoCo模型
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.action_prev = np.zeros(6, dtype=np.float32)
        # 启动 mujoco viewer

        self.viewer = None
        self.target_ctrl = np.zeros(6, dtype=np.float64)
        self.commanded_ctrl = np.zeros(6, dtype=np.float64)
        self.joint_names = JOINT_NAMES
        self.handeye_active = False
        self.declare_parameter('max_joint_speed', DEFAULT_MAX_JOINT_SPEED)
        self.max_joint_speed = float(self.get_parameter('max_joint_speed').value)
        if self.max_joint_speed <= 0.0:
            raise ValueError('max_joint_speed must be positive')
        try:
            if viewer is not None:
                if hasattr(viewer, 'launch_passive'):
                    self.viewer = viewer.launch_passive(model=self.model, data=self.data)
                elif hasattr(viewer, 'launch'):
                    self.viewer = viewer.launch(model=self.model, data=self.data)
                else:
                    self.viewer = mujoco.viewer.launch_passive(model=self.model, data=self.data)

                self.viewer_active = True


        except Exception as e:
            self.get_logger().warn(f"⚠️ Could not start viewer: {e}")
            self.viewer_active = False
            self.viewer = None

        # 创建定时器 (100Hz)
        self.timer = self.create_timer(CONTROL_PERIOD, self.timer_callback)
        self.step_scheduler = SimulationStepScheduler(
            CONTROL_PERIOD,
            self.model.opt.timestep,
        )

        # 初始目标
        self.goal = np.array([0.25, 0.25, 0.25], dtype=np.float32)

        # 获取末端执行器body ID

        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'ee_center_body')


        self.initial_ee_pos = np.zeros(3, dtype=np.float32)

        # 注册退出处理函数
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        # 控制台输出控制
        self.last_print_time = 0
        self.timer_ticks = 0

    def joint_control_callback(self, msg):
        if self.handeye_active:
            return
        self.update_target_control(msg)

    def handeye_active_callback(self, msg):
        self.handeye_active = bool(msg.data)

    def handeye_control_callback(self, msg):
        if not self.handeye_active:
            return
        self.update_target_control(msg)

    def update_target_control(self, msg):
        if len(msg.position) < 6:
            self.get_logger().warn(f"忽略 /control: position 长度不足 6 ({len(msg.position)})")
            return

        if msg.name:
            name_to_position = dict(zip(msg.name, msg.position))
            try:
                self.target_ctrl = np.array(
                    [name_to_position[name] for name in self.joint_names],
                    dtype=np.float64,
                )
                return
            except KeyError as exc:
                self.get_logger().warn(f"忽略 /control: 缺少关节 {exc}")
                return

        self.target_ctrl = np.array(msg.position[:6], dtype=np.float64)

    def goal_callback(self, msg):
        """ROS话题回调，接收目标位置"""
        new_goal = np.array([msg.x, msg.y, msg.z], dtype=np.float32)
        self.goal = new_goal
        # self.get_logger().info(f"📡 收到新目标: X={msg.x:.3f}mm, Y={msg.y:.3f}mm, Z={msg.z:.3f}mm")

    def signal_handler(self, signum, frame):
        """处理系统信号"""
        self.get_logger().info(f"📡 Received signal {signum}, cleaning up...")
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        """安全清理资源"""
        if self.viewer_active and self.viewer is not None:
            try:
                self.viewer.close()
            except:
                pass
            self.viewer_active = False


    def timer_callback(self):
        """定时器回调函数 - 主控制循环"""

        self.timer_ticks += 1
        self.commanded_ctrl = rate_limit_joint_control(
            self.commanded_ctrl,
            self.target_ctrl,
            self.max_joint_speed,
            CONTROL_PERIOD,
        )
        self.data.ctrl[:6] = self.commanded_ctrl

        for _ in range(self.step_scheduler.next_step_count()):
            mujoco.mj_step(self.model, self.data)
        if self.timer_ticks % 5 == 0:
            self.publish_joint_state()
        if self.end_effector_id >= 0:  # 确保body ID有效
            ee_pos = self.data.xpos[self.end_effector_id]  # xpos存储body的位置
            # if self.timer_ticks % 10 == 0:
                # print(f"末端位置: {ee_pos}")
        else:
            self.get_logger().error("❌ 找不到末端执行器 body")
        # 检查查看器
        if self.viewer_active and self.viewer is not None:
            if hasattr(self.viewer, 'is_running') and not self.viewer.is_running():
                self.get_logger().warn("⚠️ MuJoCo viewer stopped unexpectedly")
                self.viewer_active = False


        # 同步查看器
        if self.viewer_active and self.viewer is not None:
            try:
                self.viewer.sync()
            except Exception as e:
                self.get_logger().error(f"❌ Error syncing viewer: {e}")
                self.viewer_active = False

    def publish_joint_state(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.data.qpos[:6].copy().tolist()
        msg.velocity = self.data.qvel[:6].copy().tolist()
        self.publisher_.publish(msg)

    def __del__(self):
        """析构函数"""
        self.cleanup()

def main(args=None):
    rclpy.init(args=args)
    node = MujocoJointPublisher()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("🛑 Keyboard interrupt received")
    except Exception as e:
        node.get_logger().error(f"❌ Unexpected error: {e}")
    finally:
        node.cleanup()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
