# 项目脚本与启动工具 (`scripts/`)

本目录集中存放系统的运行启动脚本、进程与资源调度辅助库，以及独立的路由/规则测试工具。

---

## 脚本概览

| 脚本文件 | 类型 | 职责说明 |
| --- | --- | --- |
| [`start_web_no_ros.sh`](start_web_no_ros.sh) | Shell 脚本 | **Web / 语音 dry-run 启动脚本**。启动移动端 Web UI，默认开启 `--dry-run-ros2`（不发布真实机械臂硬件命令），支持可选拉起 Orbbec 相机视觉快照服务。 |
| [`start_web_with_ros.sh`](start_web_with_ros.sh) | Shell 脚本 | **Web + ROS 2 完整图启动脚本**。自动构建并加载 ROS 2 工作空间，启动 `main/launch/arm.launch.py`（含 IK 控制、人体距离检测、多来源急停仲裁）以及 Web 交互界面。 |
| [`launcher_nice.sh`](launcher_nice.sh) | Shell 库 | **进程调度与资源分配辅助库**。由启动脚本 `source` 引入，负责校验 nice 优先级、cgroup v2 `CPUWeight`、CPU 核心亲和性绑定（taskset）及 systemd 用户会话支持。 |
| [`test_rule_query_routing.py`](test_rule_query_routing.py) | Python 脚本 | **规则查询与意图路由测试脚本**。独立验证自然语言规则查询识别（`should_read_rules`）、路由器 Prompt 关键指令构建以及意图分发覆盖率。 |

---

## 脚本详细说明

### 1. `start_web_no_ros.sh`（Dry-Run 启动）

适用于算法调试、Web UI 界面联调、ASR 语音识别、LLM 对话逻辑及 MeloTTS/Piper 语音合成测试，无需连接真实机械臂硬件。

#### 运行方式

从项目根目录执行：

```bash
./scripts/start_web_no_ros.sh
```

#### 常用参数

- `--with-vision-service`：构建并拉起 ROS 2 相机快照服务（用于视觉大模型图像上下文抓取）；
- `--skip-vision-build`：跳过 ROS 2 `camera` 包的构建直接启动视觉服务；
- `--vision-service-name <name>`：指定 ROS 2 Trigger 视觉快照服务名（默认 `/vision/capture_snapshot`）；
- `--vision-output-dir <dir>`：指定快照保存目录（默认 `$ROS2_WORKSPACE/.runtime/vision_snapshots`）；
- `--vision-show-window`：显示 `min_dis` OpenCV 图像预览窗口；
- 其他参数透传给 [`Code/web_ui.py`](../Code/web_ui.py)。

---

### 2. `start_web_with_ros.sh`（完整 ROS 2 图启动）

适用于连接真实机械臂或完整 ROS 2 节点图运行。该脚本会执行工作空间干净度检查，启动底层控制节点，并拉起具备真实 ROS 2 命令发布能力的 Web UI。

#### 运行方式

从项目根目录执行：

```bash
./scripts/start_web_with_ros.sh
```

#### 常用参数

- `--with-ros-launch`：启动默认机械臂 ROS 2 Launch 图（默认行为）；
- `--no-ros-launch`：仅启动 Web UI 并连接到已由其他终端运行的 ROS 2 图（避免重复拉起节点）；
- `--skip-ros-build`：跳过 `colcon build` 直接启动；
- `--no-min-dis-window`：后台运行 `min_dis` 距离检测但隐藏 OpenCV 预览窗口；
- `--ros-launch-package <pkg>`：指定 Launch 包（默认 `main`）；
- `--ros-launch-file <file>`：指定 Launch 文件（默认 `arm.launch.py`）。

> [!IMPORTANT]
> 为确保操作安全，该脚本在启动前会检查 ROS 2 工作空间是否存在未提交的代码变更。若在本地调试中确需临时运行未提交代码，可设置环境变量 `AI_OV_ALLOW_DIRTY_ROS=1`。

---

### 3. `launcher_nice.sh`（资源与调度校验库）

由 `start_web_no_ros.sh` 与 `start_web_with_ros.sh` 自动引入，主要功能包括：

- **优先级约束**：校验 `AI_OV_VOICE_NICE` 与 `AI_OV_MIN_DIS_NICE`（范围 0~19），确保非特权用户无法非法提权；
- **cgroup v2 CPU 权重分配**：通过 `systemd-run --user --scope` 隔离语音进程组（默认权重 500）与视觉推理进程（默认权重 25），避免 YOLO 视觉推理抢占 P-Core 算力导致 TTS 语音卡顿；
- **核心亲和性校验**：校验 `taskset -c` 格式（如 `4-6` 将视觉绑定至 E-Core）；
- **环境预检**：检查 systemd 用户会话是否已激活 `cpu` 控制器。

---

### 4. `test_rule_query_routing.py`（规则路由验证）

用于快速检验自然语言意图分类器对“安全规则查询”句式的识别准确性。

#### 运行方式

从项目根目录执行：

```bash
python scripts/test_rule_query_routing.py
```

#### 校验内容

1. **Python 函数识别**：验证 `should_read_rules()` 对 20 余种常见及边缘安全规则问法（如“当前安全规则是什么”、“有哪些规则”、“防护门规则是什么”等）的判定；
2. **Prompt 构建检查**：验证 `build_agent_router_prompt()` 是否正确注入 `load_rules` 指令与示例；
3. **集成链路检查**：验证规则查询端到端路由配置一致性。

---

## 常用环境变量

| 环境变量 | 默认值 | 作用说明 |
| --- | --- | --- |
| `AI_OV_PYTHON` | `./qwen35_env/bin/python` | 主 Python 解释器路径 |
| `AI_OV_WEB_HOST` | `0.0.0.0` | Web 服务监听地址 |
| `AI_OV_WEB_PORT` | `8787` | Web 服务端口 |
| `AI_OV_WEB_ADMIN_USERNAME` | `admin` | Web 登录用户名 |
| `AI_OV_WEB_ADMIN_PASSWORD` | `12345` | Web 登录密码（生产环境务必修改） |
| `AI_OV_TTS_ENGINE` | `auto` | TTS 引擎选择（`auto` / `melo` / `piper`） |
| `AI_OV_ROS2_WORKSPACE` | `./ros2` | ROS 2 工作空间根目录 |
| `AI_OV_ROS2_SETUP` | `/opt/ros/jazzy/setup.bash` | ROS 2 系统环境变量脚本 |
| `AI_OV_VOICE_CPU_WEIGHT` | `500` | Web / 语音进程的 cgroup CPUWeight |
| `AI_OV_MIN_DIS_CPU_WEIGHT` | `25` | 视觉距离检测进程的 cgroup CPUWeight |
| `AI_OV_MIN_DIS_CPUS` | `4-6` | 视觉检测绑定的 CPU 核心编号 |
