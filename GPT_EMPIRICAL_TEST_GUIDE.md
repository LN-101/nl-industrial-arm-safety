# GPT 实测补全测试指南

适用对象：后续接手本项目测试和报告补数的 GPT/Agent。

当前结论先说清楚：报告现在缺的不是“完全没有任何测试文件”，而是缺一轮可直接写进最终报告的、带日期、设备、模型、命令和原始产物路径的完整实测闭环。`benchmarks/` 里已有 ASR、Qwen、语音安全批测等历史结果，可以作为参考和复核对象，但最终报告不要直接凭旧文件补数，除非重新确认运行日期、设备、模型版本和命令条件。

## 0. 硬规则

1. 不编造任何数值。没有跑出来就写 `TODO: 待实测`。
2. 每个可进报告的指标必须同时有：测试日期、命令、设备、模型、输入数据、原始输出路径。
3. 不覆盖旧结果。每次新测试都创建新的 run id 目录。
4. smoke/debug 结果不能冒充正式 benchmark。正式结果必须使用本指南中的“正式测试”命令或等价命令。
5. K230、真实机械臂、STM32/CAN、相机、人机距离、急停延迟等硬件结论必须有照片、视频、串口日志、ROS2 日志、示波器/逻辑分析仪记录或用户明确提供的事实。软件仓库存在代码不等于实机已测。

## 1. 建立本轮证据目录

```bash
cd /home/inteldk/AI_ov
RUN_ID="$(date +%Y%m%d_%H%M%S)_report_evidence"
EVIDENCE_DIR="benchmarks/report_evidence/${RUN_ID}"
mkdir -p "${EVIDENCE_DIR}/logs"
```

后续所有命令尽量用 `tee` 保存日志，例如：

```bash
<command> 2>&1 | tee "${EVIDENCE_DIR}/logs/<name>.log"
```

最后必须生成：

```bash
find "${EVIDENCE_DIR}" -type f | sort > "${EVIDENCE_DIR}/ARTIFACTS.txt"
```

## 2. 环境与资产清单

这些结果用于报告的“测试环境、模型版本、平台资源利用”部分。

```bash
date -Is | tee "${EVIDENCE_DIR}/logs/00_date.txt"
git rev-parse HEAD | tee "${EVIDENCE_DIR}/logs/00_git_head.txt"
git status --short | tee "${EVIDENCE_DIR}/logs/00_git_status.txt"
uname -a | tee "${EVIDENCE_DIR}/logs/00_uname.txt"
lscpu | tee "${EVIDENCE_DIR}/logs/00_lscpu.txt"
free -h | tee "${EVIDENCE_DIR}/logs/00_memory.txt"
lspci -nn | tee "${EVIDENCE_DIR}/logs/00_lspci.txt"
./qwen35_env/bin/python Code/test.py devices --json | tee "${EVIDENCE_DIR}/openvino_devices.json"
./qwen35_env/bin/python Code/test.py inventory --json | tee "${EVIDENCE_DIR}/model_inventory.json"
./qwen35_env/bin/python Code/voice_stack.py plan --json | tee "${EVIDENCE_DIR}/voice_device_plan.json"
./qwen35_env/bin/python Code/voice_stack.py audio-devices --json | tee "${EVIDENCE_DIR}/audio_devices.json"
```

检查重点：

- `openvino_devices.json` 里是否有 CPU/GPU/NPU。
- `model_inventory.json` 里哪些模型实际存在。当前常见别名包括 `qwen35-2b`、`qwen35-4b`、`qwen35-9b`、`whisper-large-v3-turbo`、`deepseek-1.5b`。
- 如果 `whisper-large-v3-int4-ov` 不存在，不要跑默认 ASR 双模型 benchmark；只跑 turbo 模型并在报告写明。

## 3. LLM/VLM 实测

### 3.1 smoke：确认模型能加载和生成

```bash
./qwen35_env/bin/python Code/test.py generate --model qwen35-2b --device GPU --prompt "人员进入机械臂安全区时系统应该怎么处理？" --max-new-tokens 80 2>&1 | tee "${EVIDENCE_DIR}/logs/llm_smoke_qwen35_2b_gpu.log"
./qwen35_env/bin/python Code/test.py generate --model qwen35-4b --device GPU --prompt "防护门打开时机械臂还能继续运行吗？" --max-new-tokens 80 2>&1 | tee "${EVIDENCE_DIR}/logs/llm_smoke_qwen35_4b_gpu.log"
./qwen35_env/bin/python Code/test.py generate --model qwen35-9b --device GPU --prompt "请简要说明机械臂急停和限速规则。" --max-new-tokens 80 2>&1 | tee "${EVIDENCE_DIR}/logs/llm_smoke_qwen35_9b_gpu.log"
```

如果 GPU/NPU 失败，可以改 CPU 复核“能否运行”，但 CPU 结果只能标为 debug，不能和 GPU 正式 benchmark 混在同一性能表中。

### 3.2 正式：Qwen3.5 benchmark

```bash
./qwen35_env/bin/python Code/qwen35_benchmark.py \
  --device GPU \
  --run-id "${RUN_ID}_qwen35_2b_4b_gpu" \
  --model qwen35-2b \
  --model qwen35-4b \
  --prompt-timeout-seconds 180 \
  --max-new-tokens 180 \
  2>&1 | tee "${EVIDENCE_DIR}/logs/qwen35_2b_4b_gpu_benchmark.log"
```

可选：如果时间和显存允许，再单独跑 9B，不要让 9B 失败影响 2B/4B 正式结果。

```bash
./qwen35_env/bin/python Code/qwen35_benchmark.py \
  --device GPU \
  --run-id "${RUN_ID}_qwen35_9b_gpu_optional" \
  --model qwen35-9b \
  --prompt-timeout-seconds 240 \
  --max-new-tokens 180 \
  2>&1 | tee "${EVIDENCE_DIR}/logs/qwen35_9b_gpu_optional.log"
```

需要回填的关键产物：

- `benchmarks/qwen35_2b_vs_qwopus35_4b/${RUN_ID}_qwen35_2b_4b_gpu/report.md`
- `benchmarks/qwen35_2b_vs_qwopus35_4b/${RUN_ID}_qwen35_2b_4b_gpu/results.json`
- `raw/` 下每个 prompt 的原始输出

报告可引用字段：

- 平均 official tokens/s
- TTFT
- TPOT
- full response seconds
- quality mean
- tool contract pass rate
- failure count

## 4. ASR 实测

当前本地通常只有 `models/asr/whisper-large-v3-turbo-int4-ov`。正式测试不要调用缺失模型。

```bash
./qwen35_env/bin/python Code/asr_benchmark.py \
  --device CPU \
  --sample-count 8 \
  --model turbo-int4=models/asr/whisper-large-v3-turbo-int4-ov \
  --results-dir "${EVIDENCE_DIR}/asr_cpu_8" \
  2>&1 | tee "${EVIDENCE_DIR}/logs/asr_cpu_8.log"

./qwen35_env/bin/python Code/asr_benchmark.py \
  --device GPU \
  --sample-count 8 \
  --model turbo-int4=models/asr/whisper-large-v3-turbo-int4-ov \
  --results-dir "${EVIDENCE_DIR}/asr_gpu_8" \
  2>&1 | tee "${EVIDENCE_DIR}/logs/asr_gpu_8.log"
```

可进报告字段：

- RTF
- mixed CER
- Han CER
- total audio seconds
- total inference seconds
- sample 输出中的 reference/hypothesis 对照

如果要做更大样本，增加 `--sample-count`，并记录耗时。不要把 8 条样本的结果写成完整数据集结论。

## 5. TTS 实测

先跑引擎可用性。失败也要保留日志，不能静默跳过。

```bash
./qwen35_env/bin/python Code/voice_stack.py tts \
  --tts-engine melo \
  --text "人员进入机械臂安全区，请立即停止机械臂并确认安全后复位。" \
  --output-name "${RUN_ID}_melo_smoke" \
  2>&1 | tee "${EVIDENCE_DIR}/logs/tts_melo_smoke.log"

./qwen35_env/bin/python Code/voice_stack.py tts \
  --tts-engine moss \
  --text "防护门打开时，机械臂应停止运行。" \
  --output-name "${RUN_ID}_moss_smoke" \
  2>&1 | tee "${EVIDENCE_DIR}/logs/tts_moss_smoke.log"

./qwen35_env/bin/python Code/voice_stack.py tts \
  --tts-engine piper \
  --text "安全光栅被遮挡，需要停机检查。" \
  --output-name "${RUN_ID}_piper_smoke" \
  2>&1 | tee "${EVIDENCE_DIR}/logs/tts_piper_smoke.log"
```

记录：

- 是否生成 WAV。
- 输出文件路径。
- 命令日志里的耗时。
- 主观试听是否存在爆音、截断、异常停顿。主观项必须标注“人工试听”，不要写成客观指标。

## 6. ASR -> LLM -> TTS/ROS2 语音链路

### 6.1 文本轮次

```bash
./qwen35_env/bin/python Code/voice_stack.py text-turn \
  --text "当前安全规则是什么？" \
  --skip-tts \
  --max-new-tokens 180 \
  2>&1 | tee "${EVIDENCE_DIR}/logs/voice_text_turn_load_rules.log"

./qwen35_env/bin/python Code/voice_stack.py ros2-text-turn \
  --dry-run-ros2 \
  --text "有人进入机械臂安全区，请触发急停。" \
  --skip-tts \
  --max-new-tokens 180 \
  2>&1 | tee "${EVIDENCE_DIR}/logs/voice_ros2_text_turn_estop_dry_run.log"
```

### 6.2 正式批量语音安全测试

完整批测会调用 TTS 生成输入语音，再跑 ASR 和 LLM。首次运行较慢。

```bash
./qwen35_env/bin/python Code/batch_voice_safety.py \
  --count 32 \
  --output-dir "${EVIDENCE_DIR}/voice_safety_batch_32" \
  --max-new-tokens 180 \
  2>&1 | tee "${EVIDENCE_DIR}/logs/voice_safety_batch_32.log"
```

如果只做预检查，可以先跑：

```bash
./qwen35_env/bin/python Code/batch_voice_safety.py \
  --count 3 \
  --output-dir "${EVIDENCE_DIR}/voice_safety_batch_smoke_3" \
  --max-new-tokens 180 \
  2>&1 | tee "${EVIDENCE_DIR}/logs/voice_safety_batch_smoke_3.log"
```

正式报告优先引用完整 `count=32` 结果。产物包括：

- `results.json`
- `results.csv`
- `report.md`
- `audio_raw_44k/`
- `audio_input_16k/`
- 如启用 `--include-output-tts`，还有 `audio_output/`

报告可引用字段：

- 通过率
- 首轮 ASR/LLM 加载耗时
- 热态单轮平均耗时
- ASR 推理平均耗时
- LLM 推理平均耗时
- 逐条失败项和缺失关键词组

## 7. Web UI 与交互证据

启动 Web UI：

```bash
AI_OV_WEB_PORT=8787 AI_OV_TTS_ENGINE=melo ./start_web_no_ros.sh 2>&1 | tee "${EVIDENCE_DIR}/logs/web_ui_start.log"
```

浏览器访问：

- 本机：`https://127.0.0.1:8787/`
- 默认登录：`admin / 12345`

需要截图或录像：

- 登录页或主界面。
- 文本输入“当前安全规则是什么？”后的回答。
- 语音/音频上传或麦克风交互。
- 规则查询、规则修改拦截或草案流程。
- 如果接入视觉服务，截图当前画面分析结果。

如果浏览器、摄像头或麦克风不可用，不要写“已完成 Web UI 实测”；只能写“CLI 链路已测，Web UI 待实测”。

## 8. ROS2 与真实硬件证据

这些测试通常不完全在 `/home/inteldk/AI_ov` 内完成，但最终报告必须补齐。没有硬件时只能保留 TODO。

### 8.1 ROS2 基线

```bash
cd /home/inteldk/ROS2
source /opt/ros/jazzy/setup.bash
source install/setup.bash
colcon test --packages-select camera control main mujoco_sim --event-handlers console_direct+ 2>&1 | tee "/home/inteldk/AI_ov/${EVIDENCE_DIR}/logs/ros2_colcon_test.log"
```

记录：

- build/test 是否通过。
- 失败包名和错误。
- 不能把 build 通过写成实机通过。

### 8.2 全局相机与人机距离急停

必须记录：

- 相机型号、连接方式、`lsusb` 或 ROS2 节点日志。
- 人员检测画面截图或视频。
- 距离阈值、触发距离、恢复距离。
- `/safety/estop/request`、`/emergency_stop`、状态管理日志。
- 从人员进入阈值到急停触发的延迟统计。

建议产物：

- `benchmarks/report_evidence/<RUN_ID>/vision_distance_estop/`
- 原始视频。
- 关键帧截图。
- ROS2 topic echo 日志。
- 延迟统计 CSV。

没有这些产物时，报告只能写“机制已代码实现/待实测”，不能写具体延迟或成功率。

### 8.3 K230 标号识别与按标号取物

必须记录：

- K230 识别画面或串口日志。
- 标号 -> 物体映射变更前后记录。
- DK2500 收到串口结果的日志。
- ROS2 发出抓取目标的日志。
- 机械臂完成取放的视频。
- 成功/失败次数统计。

报告字段建议：

| 项目 | 需要证据 |
|---|---|
| 标号识别准确率 | 固定样本数、正确次数、原始识别日志 |
| 端到端成功率 | 指令次数、成功取放次数、失败原因 |
| 响应时间 | 语音结束、识别结果、机械臂启动、放置完成的时间戳 |

没有实机视频/日志时，不要补成功率。

### 8.4 STM32/CAN 底层兜底

仅凭 ROS2 串口节点不能证明 STM32F4/CAN 1 ms 兜底。必须有：

- STM32 固件版本或提交号。
- 串口/CAN 协议文档。
- 心跳丢失、过流、限位、急停输入的测试记录。
- 示波器、逻辑分析仪或固件日志证明周期。

没有硬件记录时，报告应写 `TODO: STM32F4/CAN 底层 1 ms 复核待实测`。

## 9. 最终汇总文件

完成测试后，在本轮证据目录写一个 `SUMMARY.md`，格式如下：

```markdown
# Report Evidence Summary

Run ID: <RUN_ID>
Date: <date -Is>
Tester: GPT/Agent + 人工复核人

## Can Be Used In Report

| 模块 | 指标 | 数值 | 条件 | 原始产物 |
|---|---:|---:|---|---|
| ASR | Han CER | ... | turbo-int4, CPU/GPU, 8 samples | ... |
| LLM | official tok/s | ... | qwen35-2b, GPU | ... |
| 语音链路 | 通过率 | ... | 32 cases | ... |

## Debug Only / Not For Report

| 模块 | 原因 | 产物 |
|---|---|---|
| ... | CPU debug only / sample too small / command failed | ... |

## TODO Evidence Gaps

- TODO: K230 标号识别准确率。
- TODO: 人机距离急停阈值和延迟。
- TODO: STM32/CAN 1 ms 兜底。
- TODO: 真实机械臂取放成功率。
```

## 10. 回填报告时的写法

可以写：

- “在 DK2500 本地环境下，使用 `qwen35_env` 执行 OpenVINO GenAI 推理测试，测试命令和原始结果见 `<path>`。”
- “Whisper large-v3-turbo INT4 在 FLEURS 中文样本上的 RTF/CER 如表所示，样本数为 8，结果仅代表本轮样本。”
- “语音安全批测共 32 条，覆盖人员进入、防护门、安全光栅、急停、限速等场景，通过率为 X，失败项见原始报告。”

不能写：

- “系统识别率达到 99%”但没有样本数和原始记录。
- “急停延迟 1 ms”但只有 ROS2 代码，没有硬件测量。
- “K230 识别稳定可靠”但没有串口日志、视频和统计。
- “全部功能已实测通过”但 Web UI、相机、机械臂或 STM32 还没跑。

## 11. 现有历史结果的位置

这些文件可用于了解已有测试形态，但正式报告前应优先重跑：

- `benchmarks/voice_safety_batch/report.md`
- `benchmarks/voice_safety_batch/results.json`
- `benchmarks/asr_results/fleurs_cmn_hans_cn_cpu_8.md`
- `benchmarks/asr_results/fleurs_cmn_hans_cn_gpu_8.md`
- `benchmarks/qwen35_2b_vs_qwopus35_4b/*/report.md`
- `benchmarks/qwen35_2b_vs_qwopus35_4b/*/results.json`

如果引用历史结果，必须在报告素材里注明“历史测试，日期/设备/命令见原始文件”，并说明是否已在本轮复核。
