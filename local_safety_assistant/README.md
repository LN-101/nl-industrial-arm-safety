# `local_safety_assistant/` 本地安全助手核心

这是项目的 Python 核心包。它把本地 ASR、LLM、TTS、视觉上下文和 Web UI 连接起来，同时把模型输出限制在可校验的规则、确认和 ROS 2 消息边界内。

## 处理链路

```text
文字/音频输入
    |
    +--> stack/asr.py -------- Whisper ASR
    |
    +--> stack/pipeline.py --- 意图归一化、规则工具、LLM 回答
    |          |
    |          +-------------- rules.py / arm_rules.py
    |          +-------------- object_mapping.py
    |          +-------------- workspace_snapshot.py
    |
    +--> stack/tts.py -------- Melo/MOSS/Piper TTS
    |
    +--> stack/ros2_bridge.py 确定性 ROS 2 计划或发布
    |
    +--> web/service.py ------ Web 会话、确认、急停和流式音频
```

## 文件和模块

| 路径 | 职责 |
| --- | --- |
| `app.py` | `status` 命令，检查 OpenVINO 设备、模型、TTS 和规则文件 |
| `config.py` | 项目根目录、模型别名和默认模型路径 |
| `rules.py` | 安全规则 JSON 的加载、业务校验、预览和原子写入 |
| `arm_rules.py` | 机械臂急停、恢复、减速请求及运行时规则同步 |
| `object_mapping.py` | A/B/C/D 对象映射的读取、校验和更新 |
| `confirmation.py` | 高风险动作的确认状态和生命周期 |
| `workspace_snapshot.py` | 将规则、对象映射、机械臂状态和待确认操作汇总为快照 |
| `model_testbed.py` | OpenVINO 模型发现、设备检查和生成测试 |
| `stack/asr.py` | Whisper OpenVINO ASR 和识别结果标准化 |
| `stack/llm.py` | Qwen/OpenVINO 文本/VLM 推理适配 |
| `stack/pipeline.py` | 单轮输入、工具调用、回答和动作意图编排 |
| `stack/tts.py` | Melo、MOSS、Piper 的统一 TTS 接口 |
| `stack/devices.py` | CPU/GPU/NPU 设备探测和阶段级设备选择 |
| `stack/microphone.py` | 麦克风采集、端点检测和 WAV 输入 |
| `stack/vision.py` | ROS 2 Trigger 视觉快照的解析、路径校验和缓存 |
| `stack/vision_node.py` | 可独立启动的视觉快照 Trigger 节点 |
| `stack/ros2_bridge.py` | 将已确认的语音结果转换为 ROS 2 消息计划 |
| `stack/safety_batch.py` | 批量安全语音回归执行 |
| `web/server.py` | HTTP/HTTPS 服务、认证、路由和静态响应 |
| `web/service.py` | Web 业务状态、确认、急停、聊天和音频流 |
| `web/ui.py` | 移动端控制页面 HTML |
| `web/assets/` | 急停提示音及其文本资源 |

## 入口命令

从仓库根目录运行：

```bash
PY=./qwen35_env/bin/python

# 检查模型和设备计划
$PY -m local_safety_assistant.app status
$PY Code/voice_stack.py plan

# 文字回合，不合成语音，不发布真实 ROS 2 命令
$PY Code/voice_stack.py text-turn \
  --text "请说明机械臂急停规则" \
  --skip-tts \
  --dry-run-ros2

# 启动 Web 服务（开发时优先使用根目录启动脚本）
$PY Code/web_ui.py --help
./scripts/start_web_no_ros.sh
```

`Code/voice_stack.py` 和 `Code/web_ui.py` 是兼容入口，实际解析器分别位于 `stack/cli.py` 和 `web/server.py`。

## 安全边界

### 规则和映射

模型只能提出工具调用或规则补丁，不能直接写入 JSON。`rules.py` 会检查：

- 文档版本、规则数组和唯一 ID；
- `enabled`、条件和动作的数据类型；
- 不允许修改的身份字段和动作类型；
- 人员距离阈值的数值范围；
- 临时文件写入、`fsync` 和原子替换。

对象映射同样需要结构校验。所有需要改变安全状态或配置的操作都必须进入 `confirmation.py` 的确认流程。

### ROS 2 消息计划

`stack/ros2_bridge.py` 先生成可检查的 `Ros2MessagePlan`，默认话题包括：

| 话题 | 类型 | 说明 |
| --- | --- | --- |
| `/voice/transcript` | `std_msgs/String` | 输入文本 |
| `/voice/assistant_response` | `std_msgs/String` | 助手回答 |
| `/safety/estop/request` | `std_msgs/String` | 带来源、锁存和原因的急停请求 |
| `/emergency_stop` | `std_msgs/Bool` | 直接急停状态 |
| `/goal` | `geometry_msgs/Point` | 坐标目标 |

文字解释、规则编辑、对象映射和抓取意图不会绕过工具层直接变成 `/goal`。开发和测试时使用 `--dry-run-ros2` 查看计划。

### Web 会话

`web/service.py` 管理登录会话、待确认操作、聊天回合取消、外部急停和音频/图像输出。默认 Web 凭据只适合本地开发；生产部署必须更改密码、限制网络接口并启用合适的 TLS 与访问边界。

## 配置来源

默认配置在 `stack/config.py`：

- ASR：`models/asr/whisper-large-v3-turbo-int4-ov`；
- LLM：`models/Qwen3.5-2B-int4-ov`；
- 规则：`Code/config/safety_rules.example.json`；
- 对象映射：`Code/config/object_mapping.example.json`；
- 机械臂规则：`Code/config/arm_rules.json`；
- 视觉快照服务：`/vision/capture_snapshot`。

模型权重、TTS 运行时和 Python 环境不随仓库提供。缺失模型时，状态检查会报告缺失，不应通过提交大文件来绕过环境配置。

## 测试

```bash
./qwen35_env/bin/python -m unittest discover -s Code/tests
./qwen35_env/bin/python -m py_compile \
  local_safety_assistant/stack/*.py \
  local_safety_assistant/web/*.py
```

测试覆盖规则校验、确认状态、ASR/TTS 适配、视觉快照解析、Web API、语音栈和 ROS 2 消息计划。通过单元测试不等于目标机械臂安全验收。
