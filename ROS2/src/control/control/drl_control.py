import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from rclpy.clock import Clock
from sensor_msgs.msg import JointState
import numpy as np
from stable_baselines3 import PPO
import os
from ament_index_python.packages import get_package_share_directory

class RobotJointPublisher(Node):
    def __init__(self):
        super().__init__('drl_control')
                # 获取包路径
        policy_pkg_path = get_package_share_directory('control')

        # 模型路径
        model_path = os.path.join(policy_pkg_path, 'models', 'policy.zip')
        # 声明参数
        self.declare_parameter('control_rate', 10.0)
        self.declare_parameter('action_scale', 0.1)

        # 获取参数

        self.action_scale = self.get_parameter('action_scale').value

        # 订阅者
        self.goal_sub = self.create_subscription(Point, '/goal', self.goal_callback, 10)
        self.joint_sub = self.create_subscription(JointState, '/mujoco_joint_state', self.obs_callback, 10)

        # 发布者
        self.joint_pub = self.create_publisher(JointState, '/control', 10)

        # 状态变量
        self.rl_joint_pos = np.zeros(6, dtype=np.float64)
        self.goal = np.zeros(3, dtype=np.float64)

        self.joint_pos = np.zeros(6, dtype=np.float64)

        # 关节限位（弧度）
        self.joint_limits_low = np.array([-3.14, -3.14, -3.14, -3.14, -2.2, -3.14])
        self.joint_limits_high = np.array([3.14, 0.0, 3.14, 3.14, 2.2, 3.14])

        # 加载 RL 模型
        self.get_logger().info(f"加载模型: {model_path}")
        try:
            self.rl_model = PPO.load(model_path, device='cpu')
            self.get_logger().info("✅ RL模型加载成功")
        except Exception as e:
            self.get_logger().error(f"❌ RL模型加载失败: {e}")
            raise

        self.get_logger().info("✅ DRL控制节点初始化完成")

    def obs_callback(self, msg_data):
        """接收机械臂关节状态"""
        self.joint_pos = np.array(msg_data.position, dtype=np.float64)

    def get_observation(self) -> np.ndarray:
        """构建观测空间"""
        return np.concatenate([self.joint_pos, self.goal])

    def goal_callback(self, msg_data):
        """接收目标位置（来自相机）"""
        self.goal[0] = msg_data.x
        self.goal[1] = msg_data.y
        self.goal[2] = msg_data.z

        self.control_callback()
        # self.get_logger().info(f"📡 收到目标: ({self.goal[0]:.2f}, {self.goal[1]:.2f}, {self.goal[2]:.2f})m")

    def control_callback(self):
        """定时控制回调"""

        # 获取当前观测
        obs = self.get_observation()

        # 运行多次迭代以达到目标
        for _ in range(100):

            action, _ = self.rl_model.predict(obs, deterministic=True)
            # 更新关节位置
            self.rl_joint_pos = np.zeros(6, dtype=np.float32)
            for i in range(6):
                self.rl_joint_pos[i] = self.joint_limits_low[i] + (action[i] + 1) * 0.5 * (self.joint_limits_high[i]- self.joint_limits_low[i])

            # 更新观测
            obs = np.concatenate([self.rl_joint_pos, self.goal])

        # 应用输出缩放（映射到MuJoCo控制范围）
        scaled_pos = self.rl_joint_pos

        # 创建控制消息
        control_msg = JointState()
        control_msg.header.stamp = Clock().now().to_msg()
        control_msg.name = ['j1_joint', 'j2_joint', 'j3_joint', 'j4_joint', 'j5_joint', 'j6_joint']
        control_msg.position = scaled_pos.tolist()

        # 发布控制指令
        self.joint_pub.publish(control_msg)

        # 定期打印（每10次）
        if hasattr(self, '_print_counter'):
            self._print_counter += 1
        else:
            self._print_counter = 0

    def __del__(self):
        """析构函数"""
        pass

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RobotJointPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n🛑 用户中断")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
if __name__ == '__main__':
    main()