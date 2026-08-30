# 可自然语言交互的工业机械臂安全协作系统

**English:** [Natural-Language Interactive Safe HRC for Industrial Arms](README.en.md)

`LN-101/nl-industrial-arm-safety` 是一个面向工业机械臂人机协作的研究与原型系统。它将本地 OpenVINO 语音/大语言模型/视觉模型栈、移动端 Web UI、ROS 2 控制节点、人体与机械臂距离监测，以及多来源急停处理组合在一起。

> **安全声明**
>
> 本仓库是研究与原型系统，不是经过安全认证的机器人控制器。AI、视觉、ROS 2、串口反馈和软件急停逻辑都不能作为唯一的保护措施。任何真实机械臂操作前，必须由具备资质的工程师在目标设备上完整验证机器人、控制器、接线、工作空间和急停链路，并先使用仿真或 dry-run 模式。
>
> Web 服务默认使用 `admin` / `12345`，仅适合本地开发。开放网络访问前，请设置强密码 `AI_OV_WEB_ADMIN_PASSWORD`，并绑定到受信任的网络接口。

## 系统能力

系统允许操作员使用文字或语音查询、确认并操作机械臂工作空间；所有硬件命令都必须经过确定性校验和 ROS 2 边界。

| 能力 | 当前实现 | 运行条件 |
| --- | --- | --- |
| 多模态交互 | Qwen3.5 OpenVINO 模型、文字对话、图像/视觉上下文 | 本地模型和 OpenVINO |
| 自然语音交互 | Whisper ASR、中文文本规范化、流式 MeloTTS、可选 Piper | 音频设备及 ASR/TTS 资源 |
| 远程 Web UI | 带认证的移动优先界面，支持文字、语音、状态、确认和播放 | Python 运行环境，本地生成 HTTPS 证书 |
| 工作空间查询 | 安全规则、对象映射、机械臂状态和视觉上下文工具 | 示例配置；实时状态需要 ROS 2 |
| 受控配置修改 | 带确认流程的 JSON 规则和对象映射替换 | 可写运行时目录，模型不得直接写 JSON |
| 目标物协作 | 自然语言对象/标签映射和抓取命令路由 | ROS 2 arm/control 包及已配置的硬件 |
| 主动安全监测 | 人体/机械臂距离估计、视觉快照和相机停机锁存 | Orbbec/Gemini、YOLO/OpenVINO、ROS 2 |
| 多来源急停 | 语音、相机、反馈和手动急停统一进入 ROS 2 仲裁 | ROS 2 图和硬件安全链路 |
| 仿真与 GUI | MuJoCo 启动路径和 ROS 2 机械臂 GUI | MuJoCo、PyQt5 和桌面显示 |

机械臂底层采用基于 STM32F407 的专用主控板（见 [`STM32/`](STM32/)），包含定制 PCB 原理图、立创 EDA 工程以及多轴 CAN 伺服驱动固件，提供硬件级看门狗、上电零点校验与急停仲裁。

## 系统架构

```text
操作员语音/文字
        |
        v
移动端 Web UI / CLI
        |
        v
ASR -> 意图规范化 -> 工具校验 -> LLM 响应 -> TTS
                         |
                         v
                 ROS 2 消息桥接
                         |
       +-----------------+-----------------+
       |                 |                 |
  arm/control       camera/min_dis     main/estop
       |                 |                 |
       +--------- ROS 2 安全状态 ---------+
                         |
                         v
            USART1 (115200 DMA 协议)
                         |
                         v
            STM32F407 底层控制器 (STM32/)
       +-----------------+-----------------+
       |                 |                 |
  CAN2 (500k)        UART3 DMA         UART4 DMA
  6轴SMD电机驱动     K230目标视觉      吸盘气泵控制
```

`ros2/` 是本公开仓库中的 ROS 2 权威源码目录。历史开发快照、内部审计材料和原始工作区保留在本地 Git 中，不属于公开 GitHub 发布内容。

## 目录结构

```text
Code/                       兼容 CLI 入口和测试
local_safety_assistant/     语音、模型、规则、Web 和 ROS 2 桥接逻辑
ros2/                       集成后的权威 ROS 2 工作空间
  src/camera/               相机、距离、视觉快照和相机停机节点
  src/control/              IK、DRL、手眼标定和机械臂控制辅助逻辑
  src/main/                 启动文件、机械臂状态和急停聚合器
  src/arm_asset/            URDF/MJCF 资源和网格
  src/mujoco_sim/           MuJoCo 仿真节点
  src/arm_gui/              ROS 2 状态/停机桌面 GUI
STM32/                      STM32 底层控制器（原理图、PCB 工程与固件源码）
  Control_Code/             STM32F407 嵌入式固件工程（Keil MDK / CubeMX）
  jlc.epro2                 立创 EDA 专业版 PCB 工程文件
  SCH_2026-06-28.png        主控板电路原理图
  3D_PCB1_2026-06-28.png    主控板 3D 渲染图
scripts/                    系统启动、调度配置与独立测试脚本
  start_web_no_ros.sh       Web/语音 dry-run 启动脚本
  start_web_with_ros.sh     Web + ROS 2 实际图启动脚本
  launcher_nice.sh          进程优先级与 cgroup CPU 调度辅助脚本
  test_rule_query_routing.py 规则查询路由验证测试脚本
```

模型、虚拟环境、缓存、构建产物、运行时状态和机器专属证书由 `.gitignore` 排除。

## 代码导览

### `Code/`：命令入口与测试

`Code/` 是面向开发者的兼容入口层。这里的脚本通常只负责解析命令行并调用
`local_safety_assistant/` 中的实现，便于从仓库根目录执行和测试。

| 路径 | 职责 |
| --- | --- |
| `Code/test.py` | OpenVINO 设备、模型、生成和 ASR 冒烟测试入口 |
| `Code/voice_stack.py` | ASR → 规则/意图处理 → LLM → TTS 的语音栈入口 |
| `Code/web_ui.py` | 移动端 Web UI 的兼容启动入口 |
| `Code/download.py` | 从 Hugging Face 下载被 `.gitignore` 忽略的模型目录 |
| `Code/piper_tts.py` | Piper TTS 评测/调用适配器 |
| `Code/vision_snapshot_node.py` | 非 ROS 2 进程侧的视觉快照兼容入口 |
| `Code/config/` | 安全规则、机械臂规则和对象映射示例 |
| `Code/runtime/` | 运行时配置示例，不存放模型权重或凭据 |
| `Code/tests/` | 规则、确认流程、语音栈、Web UI 和启动脚本测试 |

详细命令和参数见 [`Code/README.md`](Code/README.md)。

### `local_safety_assistant/`：本地安全助手核心

这是 Python 核心包，负责把模型输出限制在可校验的安全工具和 ROS 2 消息边界内。

| 路径 | 职责 |
| --- | --- |
| `app.py` | `status` 入口，检查 OpenVINO 设备、模型、TTS 和规则文件 |
| `config.py` | 项目路径、模型别名和默认模型配置 |
| `rules.py` | JSON 规则加载、业务校验、预览和原子写入；模型不能直接写文件 |
| `arm_rules.py` | 机械臂急停、恢复、减速等状态规则和持久化边界 |
| `object_mapping.py` | A/B/C/D 对象映射的校验、查询和更新 |
| `workspace_snapshot.py` | 汇总规则、对象映射、机械臂状态和待确认操作 |
| `stack/` | ASR、OpenVINO LLM、TTS、设备选择、视觉快照和 ROS 2 桥接 |
| `web/` | HTTP/HTTPS 服务、登录会话、确认流程、语音流和急停 API |

核心请求路径是：

```text
输入文字/语音
  -> stack/cli.py 或 web/server.py
  -> stack/pipeline.py
  -> rules.py / object_mapping.py / workspace_snapshot.py
  -> stack/ros2_bridge.py
  -> ROS 2 话题或 dry-run 计划
```

需要直接查看 Python 核心实现时，从 [`local_safety_assistant/README.md`](local_safety_assistant/README.md) 开始。

### `ros2/`：ROS 2 工作空间

`ros2/` 是与相机、机械臂控制器、仿真器和 GUI 对接的权威工作空间。构建产物不会提交，源码集中在 `ros2/src/`。

| 包 | 关键文件/节点 | 职责 |
| --- | --- | --- |
| `camera` | `min_dis.py`、`distance_estop.py`、`pose_distance.py`、`vision_context.py` | Orbbec RGB-D、YOLO/OpenVINO 人体/机械臂距离、相机急停和视觉快照服务 |
| `control` | `ik_control.py`、`drl_control.py`、`handeye_calibration*.py` | IK、DRL、手眼标定和关节控制辅助逻辑 |
| `main` | `launch/arm.launch.py`、`arm_state.py`、`estop_aggregator.py` | 默认启动图、串口机械臂状态机和多来源急停仲裁 |
| `arm_asset` | `urdf/`、`mjcf/`、`meshes/` | 机械臂 URDF/MJCF 描述和网格资源 |
| `mujoco_sim` | `mujoco_sim.py` | `/goal`、`/control` 和关节状态的 MuJoCo 仿真 |
| `arm_gui` | `gui_node.py`、`main_window.py` | ROS 2 状态显示、目标/关节命令和手动急停 GUI |

默认启动关系如下：

```text
scripts/start_web_with_ros.sh
  -> ros2 launch main arm.launch.py
     -> control/ik_control
     -> main/arm_state + main/estop_aggregator
     -> camera/min_dis
```

仿真和 GUI 不会被默认真实机械臂启动图自动拉起，需要按包和 launch 文件单独选择。ROS 2 话题、服务和构建说明见 [`ros2/readme.md`](ros2/readme.md)。

### `STM32/`：底层控制器与硬件设计

`STM32/` 包含机械臂的定制硬件 PCB 设计与嵌入式实时控制固件，负责执行 ROS 2 下发的轨迹指令并保障电气与物理安全。

| 路径 | 职责 |
| --- | --- |
| `STM32/Control_Code/` | STM32F407 固件源码（CAN2 500k SMD 电机驱动、统一串口协议、看门狗、OLED 状态显示） |
| `STM32/jlc.epro2` | 立创 EDA 专业版 PCB 工程文件（原理图与双层板 PCB 走线） |
| `STM32/SCH_2026-06-28.png` | 主控板电路原理图 |
| `STM32/3D_PCB1_2026-06-28.png` | 主控板 3D 渲染与器件布局图 |

详细硬件设计、通信协议规范与固件开发说明见 [`STM32/README.md`](STM32/README.md)。

### 配置文件怎么分工

- `Code/config/safety_rules.example.json`：AI/规则层的安全规则示例；
- `Code/config/arm_rules.json`：AI 层机械臂急停、恢复和减速状态；
- `Code/config/object_mapping.example.json`：对象标号 A/B/C/D 示例映射；
- `ros2/src/camera/{arm_rules,safety_rules}.json`：ROS 2 相机包运行时配置；
- `ros2/src/control/config/handeye_xy.yaml`：显式启用手眼映射时的参数基线。

这些文件的修改应经过校验和确认流程；不要把模型生成的原始文本直接写入 JSON 或 YAML。

## 按目标选择入口

| 目标 | 入口 |
| --- | --- |
| 只检查设备/模型 | `Code/test.py --list-devices`、`inventory` |
| 只测试文字对话 | `Code/voice_stack.py text-turn --skip-tts --dry-run-ros2` |
| 调试手机 Web UI，不启动 ROS 2 | `./scripts/start_web_no_ros.sh` |
| 连接完整 ROS 2 图 | `./scripts/start_web_with_ros.sh` |
| 直接测试 ROS 2 视觉快照 | `ros2 service call /vision/capture_snapshot std_srvs/srv/Trigger {}` |

项目目录的默认中文说明：[`Code/README.md`](Code/README.md)、[`local_safety_assistant/README.md`](local_safety_assistant/README.md)、[`ros2/readme.md`](ros2/readme.md)、[`STM32/README.md`](STM32/README.md)、[`scripts/README.md`](scripts/README.md)。

## 环境要求

当前开发路径面向 Linux 主机：

- Python 3.12，以及支持 OpenVINO 的 Intel CPU/GPU/NPU；
- ROS 2 Jazzy、`colcon` 和 `ros2/src/` 下各包声明的依赖；
- 可选：Orbbec/Gemini 相机及厂商 SDK、udev 规则；
- 可选：串口机械臂控制器，当前 ROS 实现默认设备为 `/dev/ttyUSB0`；
- 可选：MuJoCo、PyQt5 和桌面显示环境。

语音/LLM 模型权重、Python 环境、MeloTTS 运行时、厂商 SDK 和其他外部资源不随仓库提供。ROS 2 工作空间中保留了相机与控制包当前使用的小型 YOLO/OpenVINO 和控制器资源，重新分发前请核查来源和许可证。

## 快速开始：模型与 CLI

从仓库根目录执行。现有脚本默认使用名为 `qwen35_env` 的 Python 环境，也可以将命令中的解释器路径替换为自己的环境：

```bash
./qwen35_env/bin/python Code/test.py --list-devices
./qwen35_env/bin/python Code/test.py inventory
./qwen35_env/bin/python Code/voice_stack.py plan
```

需要时，将模型下载到被忽略的 `models/` 目录：

```bash
./qwen35_env/bin/python Code/download.py qwen35-2b
./qwen35_env/bin/python Code/download.py asr-whisper-large-v3-turbo
```

下载器使用 Hugging Face 仓库；模型主机需要认证时可设置 `HF_TOKEN`。重新分发权重前请阅读对应模型卡和许可证。

不合成音频且不发布真实 ROS 2 命令的文字测试：

```bash
./qwen35_env/bin/python Code/voice_stack.py text-turn \
  --text "请说明机械臂急停规则" \
  --skip-tts \
  --dry-run-ros2
```

更多音频文件和麦克风命令见 [`Code/README.md`](Code/README.md)。

## 快速开始：不启动 ROS 2 的 Web UI

该模式适合本地界面和语音栈调试，不会发布真实机械臂命令：

```bash
AI_OV_WEB_HOST=127.0.0.1 \
AI_OV_WEB_ADMIN_PASSWORD='replace-this-password' \
./scripts/start_web_no_ros.sh
```

启动脚本会在被忽略的 `.runtime/web_ui/ssl/` 下生成本地自签名证书。该证书在浏览器中出现警告属于预期行为；可使用 `--help` 查看视觉服务和 Web UI 覆盖参数。

## 快速开始：ROS 2 集成

仓库启动脚本检测到 `./ros2` 时会自动使用它，也可以通过 `AI_OV_ROS2_WORKSPACE` 指向外部 ROS 2 工作空间。

```bash
source /opt/ros/jazzy/setup.bash
cd ros2
colcon build
source install/setup.bash
cd ..
```

然后启动 Web + ROS 2：

```bash
AI_OV_WEB_HOST=127.0.0.1 \
AI_OV_WEB_ADMIN_PASSWORD='replace-this-password' \
./scripts/start_web_with_ros.sh
```

默认启动 `ros2 launch main arm.launch.py`，这是实际机械臂路径，不会启动 MuJoCo 或 K230 相机节点。只有在其他终端已经负责目标 ROS 2 图时才使用 `./scripts/start_web_with_ros.sh --no-ros-launch`。启动脚本默认拒绝存在未提交 ROS 2 源码/配置变更的工作区；明确进行本地实验时才设置 `AI_OV_ALLOW_DIRTY_ROS=1`。

视觉快照服务提供 Trigger 服务：

```bash
ros2 service call /vision/capture_snapshot std_srvs/srv/Trigger {}
```

ROS 2 话题、服务和标定说明见 [`ros2/readme.md`](ros2/readme.md)。

## 安全与安全边界

- 自然语言输出不是直接的硬件命令通道。语音/LLM 意图必须规范化、校验、必要时确认，再转换为确定性的 ROS 2 消息；
- 规则和映射修改必须使用校验后的持久化辅助函数，模型响应不得直接编辑运行时 JSON；
- 相机距离故障和帧过期属于安全状态，不能证明工作空间安全；必须在目标设备上完整验证相机、ROS 2、控制器和硬件停机行为；
- Web UI 只适合开发期间的受信任局域网。生产部署前应更改默认凭据、限制绑定接口并增加网络边界；
- 源码包含硬件路径、标定值、模型资源和串口假设，适配其他机械臂或相机前必须重新检查。

公开源码不能替代目标设备调试、风险评估、调试验收或正式的机械安全规程。

## 测试与检查

不启动模型和硬件时，可以执行确定性测试：

```bash
./qwen35_env/bin/python -m unittest discover -s Code/tests
./qwen35_env/bin/python -m py_compile \
  Code/voice_stack.py Code/web_ui.py \
  local_safety_assistant/stack/*.py \
  local_safety_assistant/web/*.py \
  scripts/*.py
bash -n scripts/start_web_no_ros.sh scripts/start_web_with_ros.sh
```

ROS 2 包测试：

```bash
source /opt/ros/jazzy/setup.bash
cd ros2
colcon test --event-handlers console_direct+
```

部分检查需要 OpenVINO 模型、相机、MuJoCo、桌面显示或串口控制器。在当前快照中，ROS 2 工作空间构建成功；`colcon test` 仍有既存风格/文档检查失败和 3 个机械臂状态标定/时序断言差异，本 README 不宣称所有 ROS 2 测试全绿。

## 👥 贡献者与团队分工 / Contributors

本项目由团队协同研发完成，各成员分工如下：

<table>
  <tr>
    <td align="center" width="33.3%">
      <a href="https://github.com/LN-101">
        <img src="https://github.com/LN-101.png?size=100" width="100px;" alt="LN-101"/>
        <br />
        <sub><b>LN-101</b></sub>
      </a>
      <br />
      <sub><b>AI 算法与系统编排</b></sub>
      <br />
      <sub>负责 <code>local_safety_assistant/</code> 与 <code>Code/</code>（OpenVINO 大模型、ASR 语音识别、TTS 语音合成、Web UI、安全规则引擎及 ROS 2 桥接层）</sub>
    </td>
    <td align="center" width="33.3%">
      <a href="https://github.com/nanshanbot">
        <img src="https://github.com/nanshanbot.png?size=100" width="100px;" alt="nanshanbot"/>
        <br />
        <sub><b>nanshanbot</b></sub>
      </a>
      <br />
      <sub><b>ROS 2 机器人系统</b></sub>
      <br />
      <sub>负责 <code>ros2/</code> 工作空间（机械臂运动学控制、相机 RGB-D 距离安全监测、多来源急停仲裁、MuJoCo 仿真与 PyQt5 桌面 GUI）</sub>
    </td>
    <td align="center" width="33.3%">
      <a href="https://github.com/RM-wD55">
        <img src="https://github.com/RM-wD55.png?size=100" width="100px;" alt="MY-GIRL (RM-wD55)"/>
        <br />
        <sub><b>MY-GIRL (RM-wD55)</b></sub>
      </a>
      <br />
      <sub><b>STM32 硬件与嵌入式固件</b></sub>
      <br />
      <sub>负责 <code>STM32/</code> 子系统（立创 EDA 主控板 PCB 原理图设计、STM32F407 固件、CAN 6轴电机驱动、串口协议与硬件看门狗）</sub>
    </td>
  </tr>
</table>

## 贡献指南

贡献应保持自然语言编排、规则校验、ROS 2 命令路由、相机安全状态和硬件控制器边界之间的分层。修改路由、规则校验、急停行为和启动契约时，请同时补充聚焦测试。不要提交模型权重、虚拟环境、运行时状态、凭据或未经审查的协作者快照。

## 许可证

项目代码和公开文档采用 [Apache License 2.0](LICENSE)。第三方模型、厂商 SDK、二进制资源和外部运行时可能适用独立许可证，不自动包含在根目录许可证中；重新分发前请逐项核查。
