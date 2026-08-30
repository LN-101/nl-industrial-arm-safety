# `ros2/` ROS 2 工作空间

**English:** [README.en.md](README.en.md)

这是 `nl-industrial-arm-safety` 的权威 ROS 2 工作空间，负责把自然语言安全助手连接到相机、机械臂控制器、仿真器和操作员 GUI。构建产物 `build/`、`install/`、`log/` 和 `.runtime/` 只在本机生成，不提交到 Git。

## 包结构

| 包 | 主要文件/节点 | 职责 |
| --- | --- | --- |
| `camera` | `camera/min_dis.py`、`distance_estop.py`、`pose_distance.py` | Orbbec RGB-D、YOLO/OpenVINO 人体/机械臂距离、相机急停和视觉上下文 |
| `control` | `control/ik_control.py`、`drl_control.py` | 逆运动学、DRL 策略、关节控制和手眼参数读取 |
| `main` | `launch/*.launch.py`、`main/arm_state.py` | 启动图、串口协议、工作空间边界、抓放状态机和安全规则同步 |
| `main` | `main/estop_aggregator.py` | 聚合语音、相机、反馈和手动急停来源 |
| `arm_asset` | `urdf/`、`mjcf/`、`meshes/` | 机械臂模型、碰撞/视觉网格和 MuJoCo 资源 |
| `mujoco_sim` | `mujoco_sim/mujoco_sim.py` | `/goal`、`/control`、手眼控制和关节状态的 MuJoCo 仿真 |
| `arm_gui` | `arm_gui/gui_node.py`、`main_window.py` | 状态显示、目标/关节命令和手动急停界面 |

## 关键文件

```text
ros2/
├── colcon_defaults.yaml       colcon 默认构建选项
├── src/
│   ├── arm_asset/             URDF、MJCF、STL 网格和展示图片
│   ├── camera/                相机、距离安全和视觉快照
│   ├── control/               IK、DRL、手眼和策略资源
│   ├── main/                  真实机械臂启动图、状态机和急停仲裁
│   ├── mujoco_sim/            仿真节点和仿真测试
│   └── arm_gui/               PyQt5 ROS 2 GUI
└── readme.md                  本中文说明
```

相机包中随源码保留的二进制 SDK/模型资源可能有独立许可证，重新分发前必须核查来源和条款。

## 构建

在仓库根目录执行：

```bash
source /opt/ros/jazzy/setup.bash
cd ros2
colcon build
source install/setup.bash
```

如果使用外部 ROS 2 工作空间，根目录启动脚本支持：

```bash
AI_OV_ROS2_WORKSPACE=/path/to/ros2 ./scripts/start_web_with_ros.sh
```

默认启动脚本会优先使用仓库内的 `./ros2`，并在真实启动前拒绝存在未提交的 ROS 2 源码/配置变更。仅在明确的本地实验中使用 `AI_OV_ALLOW_DIRTY_ROS=1`。

## 启动图

### 默认真实机械臂路径

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch main arm.launch.py
```

根目录的完整入口会先构建/加载工作空间，再启动 Web UI：

```bash
cd ..
./scripts/start_web_with_ros.sh
```

`main/launch/arm.launch.py` 默认连接：

```text
control/ik_control
        |
main/arm_state -------- main/estop_aggregator
        |
camera/min_dis -------- /emergency_stop
```

默认图不会自动启动 MuJoCo 或 K230 节点。真实硬件启动前，应先审查 launch 参数、串口设备、模型路径、相机状态和急停链路。

### 其他 launch 文件

| 文件 | 用途 |
| --- | --- |
| `arm.launch.py` | 默认控制、状态、相机和急停图 |
| `arm_ik_real.launch.py` | 更直接的真实机械臂 IK 路径 |
| `arm_ik_sim.launch.py` | IK + MuJoCo 仿真路径 |
| `arm_drl_sim.launch.py` | DRL 控制 + MuJoCo 仿真路径 |
| `handeye_calibration.launch.py` | 手眼标定维护图，默认不启用运动 |

仿真/标定都不是生产真实机械臂默认路径。请先阅读对应 launch 文件和节点参数，再在目标设备执行。

## 主要 ROS 2 接口

| 接口 | 类型 | 发布/订阅方 | 作用 |
| --- | --- | --- | --- |
| `/goal` | `geometry_msgs/Point` | AI 桥接、GUI、`arm_state`/仿真 | 目标笛卡尔坐标 |
| `/control` | `sensor_msgs/JointState` | 控制器、GUI、仿真 | 六关节控制目标 |
| `/robot_joint_state` | `sensor_msgs/JointState` | `arm_state` | 真实机械臂反馈 |
| `/mujoco_joint_state` | `sensor_msgs/JointState` | `mujoco_sim`、GUI | 仿真关节反馈 |
| `/min_distance` | `std_msgs/Float32` | `camera/min_dis`、GUI | 人/臂最小距离 |
| `/safety/estop/request` | `std_msgs/String` | 语音/相机、`estop_aggregator` | 带来源和原因的急停请求 JSON |
| `/emergency_stop` | `std_msgs/Bool` | 急停聚合器、控制器、GUI | 有效急停状态 |
| `/voice/transcript` | `std_msgs/String` | AI 桥接 | 语音转写文本 |
| `/voice/assistant_response` | `std_msgs/String` | AI 桥接 | 助手回答 |
| `/vision/capture_snapshot` | `std_srvs/srv/Trigger` | `camera/min_dis` | 获取带时间戳的视觉快照上下文 |

视觉快照调用示例：

```bash
ros2 service call /vision/capture_snapshot std_srvs/srv/Trigger '{}'
```

服务成功时，`message` 为 JSON，包含图片路径、时间戳、帧 ID、最小距离、最近点、急停状态、帧新鲜度和来源。

## 相机安全节点

`camera/min_dis.py` 负责：

- 读取 Orbbec RGB-D 数据；
- 使用 YOLO/OpenVINO 姿态模型识别人和机械臂关键点；
- 在深度无效时使用受限的历史/候选恢复策略；
- 发布 `/min_distance` 和视觉上下文；
- 通过 `/safety/estop/request` 请求锁存相机急停。

默认启动图会把视觉推理限制到指定 CPU、线程数和 `nice`/CPUWeight 范围，以避免语音播放被视觉推理饿死。相机故障、帧过期或距离未知不能被解释为“工作空间安全”。

## 控制与急停

`control/ik_control.py` 从 URDF 读取运动链，进行目标姿态/工作空间/关节限制检查，并将关节目标交给控制接口。`main/arm_state.py` 负责串口反馈、抓放状态机、工作空间边界和控制器通信；它还会拒绝过期、越界或急停状态下的命令。

`main/estop_aggregator.py` 将多个来源合并为一个有效状态：

```json
{"source":"voice_assistant","active":true,"latch":true,"reason":"用户请求急停"}
```

只允许对应来源按规则释放锁存；清除某一个来源不能覆盖其他仍然激活的来源。物理急停和下位机安全链路仍然是最终保护措施。

## 仿真与 GUI

MuJoCo 节点加载 `arm_asset/mjcf/arm_mjcf.xml`，订阅 `/goal` 和 `/control`，发布 `/mujoco_joint_state`，并对关节目标做速度限制。GUI 节点提供目标点、六关节控制、状态订阅和手动 `/emergency_stop`。

这些路径需要桌面显示、MuJoCo/PyQt5 和完整 ROS 2 Python 依赖：

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch main arm_ik_sim.launch.py
```

## 测试

```bash
source /opt/ros/jazzy/setup.bash
cd ros2
colcon test --event-handlers console_direct+
```

当前工作空间可以成功 `colcon build`。`colcon test` 中仍有既存的风格/文档检查失败和少量机械臂状态标定/时序断言差异；这些结果不代表真实硬件安全验收。
