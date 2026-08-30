# `Code/` 命令入口与测试

**English:** [README.en.md](README.en.md)

`Code/` 是仓库根目录下的开发者入口层。这里的 Python 文件大多是兼容性包装器、模型测试命令或批处理工具；核心安全逻辑位于 [`local_safety_assistant/`](../local_safety_assistant/)。

## 文件职责

| 路径 | 作用 |
| --- | --- |
| `test.py` | 设备、模型、文本生成和 ASR 冒烟测试入口，实际调用 `local_safety_assistant.model_testbed` |
| `voice_stack.py` | 语音栈 CLI 入口，实际调用 `local_safety_assistant.stack.cli` |
| `web_ui.py` | 移动端 Web UI 入口，实际调用 `local_safety_assistant.web.server` |
| `download.py` | 从 Hugging Face 下载模型到被忽略的 `models/` |
| `piper_tts.py` | Piper TTS 的外部运行器适配 |
| `qwen35_benchmark.py` | Qwen/OpenVINO 生成吞吐和设备遥测基准 |
| `asr_benchmark.py` | Whisper ASR 性能测试辅助 |
| `batch_voice_safety.py` | 批量语音安全回归任务 |
| `assistant.py` | 早期本地助手兼容入口 |
| `vision_snapshot_node.py` | 视觉快照兼容入口；ROS 2 集成实现位于 `ros2/src/camera/` |
| `config/` | 安全规则、机械臂规则和对象标号映射示例 |
| `runtime/` | 不含凭据的运行时配置样例 |
| `tests/` | Python 单元测试和启动脚本契约测试 |

不要直接从模型输出写入 `Code/config/`。规则和对象映射必须通过核心包中的校验与确认流程更新。

## 环境

从仓库根目录执行命令。项目默认使用 `qwen35_env`，也可以把下面命令中的解释器替换成自己的 Python 3.12 环境。

```bash
PY=./qwen35_env/bin/python
$PY -m unittest discover -s Code/tests
$PY -m py_compile Code/*.py local_safety_assistant/stack/*.py local_safety_assistant/web/*.py
```

模型、虚拟环境、缓存和运行时输出不在 Git 中。模型下载前请阅读模型卡和许可证。

## 设备与模型检查

```bash
$PY Code/test.py --list-devices
$PY Code/test.py inventory
$PY Code/voice_stack.py plan
$PY Code/voice_stack.py audio-devices
```

`plan` 输出 ASR、LLM、视觉和 TTS 的 OpenVINO 设备选择；没有对应模型或硬件时，检查命令会明确报告缺失状态。

下载模型到被忽略的 `models/` 目录：

```bash
$PY Code/download.py qwen35-2b
$PY Code/download.py asr-whisper-large-v3-turbo
```

可用别名以 `Code/download.py --help` 和源码中的 `MODEL_REPOS` 为准。需要 Hugging Face 权限时，在本机设置 `HF_TOKEN`，不要把令牌写入命令历史或仓库。

## 文字、音频和语音栈

### 文字回合

先使用 dry-run，确认不会发布真实 ROS 2 命令：

```bash
$PY Code/voice_stack.py text-turn \
  --text "请说明机械臂急停规则" \
  --skip-tts \
  --dry-run-ros2
```

常用参数包括 `--llm-model`、`--asr-model`、`--rules`、`--object-mapping`、`--arm-rules` 和 `--max-new-tokens`。

### 音频文件

输入必须是可读取的 PCM WAV，推荐单声道、16 kHz：

```bash
$PY Code/voice_stack.py audio-file \
  --audio /path/to/sample_16k.wav \
  --language zh \
  --skip-tts \
  --dry-run-ros2
```

### 麦克风

```bash
$PY Code/voice_stack.py listen \
  --language zh \
  --max-turns 1 \
  --no-ros2
```

调试麦克风时先使用 `--no-ros2`；`--speech-threshold`、`--trailing-silence-seconds`、`--mic-device` 和 `--max-utterance-seconds` 用于调整端点检测。

### TTS

```bash
$PY Code/voice_stack.py tts \
  --text "机械臂已进入安全状态" \
  --tts-engine melo
```

支持 `moss`、`melo` 和 `piper` 三种适配器。运行时、模型和厂商/第三方依赖需要在本机单独安装；本仓库不提供一键安装脚本。

## ROS 2 桥接

语音栈默认只生成计划；启用 ROS 2 桥接后，完成的回合会按确定性规则发布：

```bash
$PY Code/voice_stack.py ros2-text-turn \
  --text "请立即急停机械臂" \
  --skip-tts \
  --dry-run-ros2
```

去掉 `--dry-run-ros2` 才会尝试连接 ROS 2。桥接使用的默认话题为：

| 话题 | 类型 | 用途 |
| --- | --- | --- |
| `/voice/transcript` | `std_msgs/String` | 规范化后的输入文本 |
| `/voice/assistant_response` | `std_msgs/String` | 助手回答 |
| `/safety/estop/request` | `std_msgs/String` | 带来源和原因的急停请求 |
| `/emergency_stop` | `std_msgs/Bool` | 直接急停状态 |
| `/goal` | `geometry_msgs/Point` | 经过解析的目标坐标 |

规则查询、规则编辑、对象映射和抓取意图不会被当成普通坐标命令直接发布；它们必须经过对应的确认/解析流程。

## Web UI

从仓库根目录使用启动脚本：

```bash
./scripts/start_web_no_ros.sh
```

这个模式默认使用 `--dry-run-ros2`，适合界面、ASR、LLM 和 TTS 联调。完整 ROS 2 集成使用：

```bash
./scripts/start_web_with_ros.sh
```

详细环境变量包括 `AI_OV_PYTHON`、`AI_OV_WEB_HOST`、`AI_OV_WEB_PORT`、`AI_OV_WEB_ADMIN_PASSWORD`、`AI_OV_TTS_ENGINE` 和 `AI_OV_ROS2_WORKSPACE`。Web 默认凭据只适合本地开发，部署前必须更改。

## 配置文件

- `config/safety_rules.example.json`：规则文档示例，包含版本、规则条件和动作；
- `config/arm_rules.json`：机械臂急停、恢复、减速等运行时规则；
- `config/object_mapping.example.json`：A/B/C/D 对象标号示例；
- `runtime/voice_stack.example.json`：不含秘密的语音栈参数样例。

修改规则时，核心包会检查 JSON 类型、重复 ID、已知业务字段、距离范围和不可修改字段，并使用临时文件加原子替换写回。

## 测试范围

```bash
$PY -m unittest discover -s Code/tests
$PY -m py_compile Code/*.py local_safety_assistant/stack/*.py local_safety_assistant/web/*.py scripts/*.py
bash -n scripts/start_web_no_ros.sh scripts/start_web_with_ros.sh
```

测试不代表真实机械臂安全认证。涉及 ROS 2、相机、串口、模型和 TTS 的集成行为必须在目标设备上另外验收。
