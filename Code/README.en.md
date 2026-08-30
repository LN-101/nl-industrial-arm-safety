# Local ASR and Qwen3.5 Testbed

**中文（默认）:** [README.md](README.md)

Run all commands from the repository root with `qwen35_env`:

```bash
./qwen35_env/bin/python Code/test.py --list-devices
./qwen35_env/bin/python Code/test.py inventory
./qwen35_env/bin/python Code/voice_stack.py plan
```

## Text Generation Smoke Tests

Qwen3.5 2B and 9B are detected as VLM-style OpenVINO GenAI models, so the
testbed uses `VLMPipeline` automatically.

```bash
./qwen35_env/bin/python Code/test.py generate \
  --model qwen35-2b \
  --device CPU \
  --prompt "Answer in one short sentence: what can you do for a robot safety assistant?" \
  --max-new-tokens 12

./qwen35_env/bin/python Code/test.py generate \
  --model qwen35-9b \
  --device CPU \
  --prompt "Answer in one short sentence: what is your role?" \
  --max-new-tokens 8
```

Device fallback is enabled by default when a requested accelerator fails:

```bash
./qwen35_env/bin/python Code/test.py generate --model qwen35-2b --device NPU --max-new-tokens 6
./qwen35_env/bin/python Code/test.py generate --model qwen35-2b --device GPU --max-new-tokens 6
```

For the current Qwen3.5 2B GPU tuning notes, use the benchmark harness below
with the target OpenVINO device and record the resulting `PerfMetrics` values.

The Qwen3.5 benchmark harness records official generated-token throughput,
TPOT, TTFT, stream chunk cadence, and a hardware telemetry snapshot:

```bash
./qwen35_env/bin/python Code/qwen35_benchmark.py --model qwen35-2b --device GPU
```

Use `official_tokens_per_second` / `mean_official_tokens_per_second` as the
model throughput metric. `stream_chunks_per_second` is only a streaming
smoothness diagnostic.

## ASR Smoke Tests

The Whisper path accepts normalized 16 kHz audio. The built-in WAV loader is
dependency-light and supports PCM WAV files only.

```bash
./qwen35_env/bin/python Code/test.py asr --load-only --device CPU --model whisper-large-v3-turbo
./qwen35_env/bin/python Code/test.py asr --device CPU --audio-sample /path/to/sample_16k.wav --language en
```

If no audio sample is provided, the ASR command validates the model path and
prints the required sample format.

## Safety Assistant Scaffold

The initial project scaffold validates local model assets and a JSON safety-rule
file. The rule store is intentionally owned by code, not by direct model file
writes.

```bash
./qwen35_env/bin/python Code/assistant.py status
```

The modular voice stack entrypoint is:

```bash
./qwen35_env/bin/python Code/voice_stack.py plan
./qwen35_env/bin/python Code/voice_stack.py text-turn --text "请说明机械臂急停规则" --skip-tts
./qwen35_env/bin/python Code/voice_stack.py audio-file --audio /path/to/sample_16k.wav --language zh --skip-tts
./qwen35_env/bin/python Code/voice_stack.py tts --text "机械臂已停止，请确认安全区域。"
```

The default TTS engine remains MeloTTS. A faster optional Piper Huayan backend
is available for listening tests; its default pause setting is the tuned
`--piper-silence-scale 1.0` candidate:

```bash
./qwen35_env/bin/python Code/voice_stack.py tts \
  --tts-engine piper \
  --piper-model-dir .trellis/tasks/05-28-compare-natural-chinese-tts-models/piper_speaches_huayan_model \
  --piper-espeak-data-dir .trellis/tasks/05-28-compare-natural-chinese-tts-models/sherpa_kokoro_model/kokoro-int8-multi-lang-v1_1/espeak-ng-data \
  --text "机械臂已停止，请保持安全距离，确认防护门关闭后再复位。"
```

The latest blind listening pack is under
`.trellis/tasks/05-28-compare-natural-chinese-tts-models/listening_eval/tuned_piper10_vs_melo/`.
Keep Piper optional until that pack wins the listening check. The current
rhasspy model card lists the HuaYan dataset URL as missing and the license as
unknown, so production use also needs an explicit license-risk acceptance or a
replacement voice with clear licensing.

ASR transcripts are normalized before routing. Install OpenCC in `qwen35_env`
so Traditional Chinese Whisper output is converted to Simplified Chinese before
rule/tool intent detection:

```bash
./qwen35_env/bin/python -m pip install opencc-python-reimplemented
```

No-wake realtime microphone listening uses an optional `sounddevice` runtime
dependency. It captures 16 kHz mono microphone audio, segments one utterance at
a time with an energy threshold, then runs the same ASR -> LLM -> ROS2 path:

```bash
sudo apt install portaudio19-dev
./qwen35_env/bin/python -m pip install sounddevice
./qwen35_env/bin/python Code/voice_stack.py listen --no-wake --language zh --skip-tts --dry-run-ros2
```

Useful tuning options are `--speech-threshold`, `--trailing-silence-seconds`,
`--min-speech-seconds`, `--max-utterance-seconds`, `--mic-device`, and
`--max-turns`. Use `--no-ros2` when you only want local ASR/Qwen output during
microphone tuning.

The ROS2 bridge commands run the same voice turn and publish ROS2 topic
messages. Use `--dry-run-ros2` first to inspect the planned messages without
requiring `rclpy`:

```bash
./qwen35_env/bin/python Code/voice_stack.py ros2-text-turn --text "请立即急停机械臂" --skip-tts --dry-run-ros2
./qwen35_env/bin/python Code/voice_stack.py ros2-audio-file --audio /path/to/sample_16k.wav --language zh --skip-tts --dry-run-ros2
```

## Mobile Web UI

The hotspot web surface is a separate entrypoint:

```bash
./qwen35_env/bin/python Code/web_ui.py
```

For the HTTPS phone UI launcher:

```bash
./scripts/start_web_no_ros.sh
```

For real ROS2 Web debugging, build/source the ROS2 workspace, start the default
arm launch, and start the Web UI without `--dry-run-ros2` through:

```bash
./scripts/start_web_with_ros.sh
```

The default launch is `ros2 launch main arm.launch.py`; its default real-robot
path does not start MuJoCo simulation or the K230 camera node. Override the
target with `--ros-launch-package <pkg>` and `--ros-launch-file <file>` or the
matching `AI_OV_ROS2_LAUNCH_PACKAGE` / `AI_OV_ROS2_LAUNCH_FILE` environment
variables. Use `--skip-ros-build` when the ROS2 workspace is already built.
The default build runs plain `colcon build` inside the ROS2 workspace and lets
the ROS build/launch processes inherit that workspace's Python dependency
resolution; it does not force `--symlink-install` or prepend system
`dist-packages`.

Both Web launchers keep Web/ASR at effective nice `0` by default, and the MOSS
TTS child inherits that priority. Set `AI_OV_VOICE_NICE` to an unprivileged
value from `0` through `19` to override it. Vision inference runs at nice `10`
by default; set `AI_OV_MIN_DIS_NICE` in the same range to override it. Negative,
invalid, or unattainable values fail before Web or ROS processes start. The
real ROS launcher leaves arm control, arm state, and emergency-stop aggregation
at nice `0`; only `min_dis` is demoted.

MOSS streaming playback buffers `0.48` seconds of PCM before it starts. After
an underrun it waits for `0.64` seconds of PCM before resuming, giving playback
a larger fixed recovery reserve while retaining the low initial latency now
that YOLO no longer competes for the CPU. Set
`AI_OV_MOSS_PCM_BUFFER_SECONDS=2`, `3`, or `4` to opt back into a larger initial
reserve when comparing latency and continuity under contention.
The default Web stream sends `voice=Xiaoyu` without a prompt WAV, so the ONNX
runtime uses its built-in prompt codes and avoids encoding `zh_11.wav` for each
turn. An explicit `--moss-prompt-audio <wav>` or
`--moss-use-yangmi-prompt-audio` setting still selects reference-audio cloning.
Starting a new Web text or voice turn now stops local playback and waits for the
previous backend/MOSS stream to cancel before the new request begins. The ONNX
worker keeps exclusive ownership of the shared runtime until it exits and its
codec streaming state is reset. If cancellation logs `runtime restart
required` or `/health` reports `runtime_manager.runtime_healthy=false`, restart
the MOSS/Web launcher instead of retrying concurrent turns in the same server.
MOSS affinity is disabled by default; use
`AI_OV_MOSS_CPUS=<cpu-list>` only for explicit affinity experiments. These
settings do not change Web,
ASR, arm control, emergency-stop, or the existing `min_dis` E-core affinity.

If another terminal already owns the full ROS2 graph, attach only the Web UI
without starting duplicate arm/camera nodes:

```bash
./scripts/start_web_with_ros.sh --no-ros-launch
```

The real ROS launcher rejects `--dry-run-ros2`; use `scripts/start_web_no_ros.sh` for
safe UI-only testing.

Before starting the Orbbec/Gemini vision snapshot service as a normal user,
install the udev rules shipped with the ROS2 environment's `pyorbbecsdk`.
Gemini 336 enumerates as `2bc5:0803`; without this rule it can appear in
`lsusb` while the Orbbec SDK still fails with `Access denied` /
`usbEnumerator openUsbDevice failed`:

```bash
sudo sh "$AI_OV_ROS2_WORKSPACE/.venv/lib/python3.12/site-packages/pyorbbecsdk/shared/install_udev_rules.sh"
```

After installing the rule, unplug and replug the camera before starting the
vision service.

To also build and start the ROS2 RGB snapshot service used by visual-analysis
requests:

```bash
./scripts/start_web_no_ros.sh --with-vision-service
```

The launcher still passes `--dry-run-ros2` to the Web UI by default, so robot
command publishing remains disabled while `/vision/capture_snapshot` is
available for VLM snapshots. The service is provided by the ROS2 `camera`
package's `min_dis` node, which also owns the Orbbec RGB/depth pipeline and
minimum-distance context. Use `--vision-service-name <name>` or
`--vision-output-dir <dir>` to override the `min_dis` vision parameters. Use
`--vision-show-window` only when the OpenCV preview windows are wanted.

Do not use `--with-vision-service` if a full ROS2 launch such as
`ros2 launch main arm.launch.py` is already running `min_dis`; two
camera nodes can compete for the same Orbbec device.

It serves a mobile-first local UI with `admin / 12345` login by default,
text chat, browser voice upload, playback of generated TTS audio, and an
emergency-stop button. The first MVP intentionally excludes image upload and
full autonomous arm-agent control.

For real ROS2 publishing, source ROS2 and the workspace first. By default,
emergency-stop voice commands publish JSON requests to `/safety/estop/request`;
use `--direct-estop-topic` only when publishing `std_msgs/Bool` directly to
`/emergency_stop` is desired. The same emergency-stop and release commands also
write the configured arm runtime JSON (`--arm-rules`, default
`Code/config/arm_rules.json`) by setting `arm_stop` / `arm_recover`.

When the Web UI runs without `--dry-run-ros2`, it also subscribes to
`/safety/estop/request` and turns active external stop events (for example the
visual person-distance stop published by the `min_dis` node with
`source=min_distance_camera`) into a browser emergency alert: pending Web work
is canceled, an emergency popup shows the source/reason/distance/threshold, and
a pre-generated alert audio file is played directly with a browser `Audio`
element (never realtime TTS). Provide the audio file at
`local_safety_assistant/web/assets/emergency_alert.wav` with the fixed message
`警告！检测到不安全情况，已经急停`, or replace it with
`--emergency-alert-audio <path>`; it is served at `/emergency-alert-audio`.
The `min_dis` stop is latched after its first unsafe distance. Safe or missing
detections do not release it automatically. A confirmed release asks the camera
to reset, but the camera clears only after either three consecutive valid frames
at least `0.05m` above the configured stop distance or five seconds of valid
camera frames with no person detected. Person reappearance, camera frame loss,
or stale evidence cancels the no-person eligibility; otherwise the stop remains
active.
Synthetic events can be injected through the authenticated
`POST /api/estop/external` route using the same multi-source JSON contract
(`source`, `active`, `latch`, `reason`, plus optional `distance_m`,
`trigger_distance_m`, `threshold_m`, and `release_distance_m`). Camera alerts
show the first unsafe trigger distance, latest person-to-arm distance, and
release gate separately. Dismissing the popup never releases the real emergency stop;
release remains owned by the ROS2 stop-source logic.

The example rule file is:

```text
Code/config/safety_rules.example.json
```

Current-rule questions such as `当前安全规则是什么` are read through the
validated rule tool path, not from prompt memory. Each voice turn first asks
the 2B model for a bounded JSON route: final spoken reply, `load_rules`, or
`edit_rules`. The legacy keyword detector is only a fallback when the route is
invalid. The model may request rule reads with a structured JSON tool envelope
such as `{"type":"tool_call","name":"load_rules","arguments":{}}`; the legacy
`TOOL:load_rules` marker remains a fallback. The default example now
covers personnel intrusion, unknown objects, guard doors, light curtains, ROS
controller alarms, and teach-mode speed limits. After the JSON document is
loaded, Qwen is prompted to understand the rule fields and answer in natural
Chinese: broad questions should summarize every enabled rule briefly without
appending a follow-up suggestion tail, while specific questions should explain only the
matching JSON rule. `Code/voice_stack.py` and `Code/batch_voice_safety.py` both
default to this file and accept `--rules` to point at another validated JSON
rule document.

Rule-edit turns no longer construct a default 9B rule editor. The router should
select `edit_rules` for natural edit requests such as `调整人员安全距离到 1.2 米`;
the patch stage then produces a constrained patch against an existing rule, and
project code applies that patch through validated atomic writes. The checked-in
runtime default is `two-pass` while the MVP is being compared; use
`--rule-edit-strategy one-pass` or `--rule-edit-strategy two-pass` on
`Code/voice_stack.py text-turn --skip-tts` to compare both patch-generation
paths before choosing long-term policy.

Explicit vision requests such as `调用视觉，分析下当前工作环境` route through a
bounded `analyze_environment_vision` tool. The assistant calls a configurable
ROS2 `std_srvs/Trigger` snapshot service, defaulting to
`/vision/capture_snapshot`; on success the Trigger `message` must be JSON with
an `image_path` field and optional `stamp` / `frame_id` metadata:

```json
{"image_path": "/tmp/current_rgb_snapshot.jpg", "stamp": "2026-06-09T12:00:00", "frame_id": "camera_color"}
```

The first implementation is RGB-only. The assistant validates and copies the
snapshot into the authenticated Web runtime image directory, runs Qwen3.5 VLM
analysis, returns a concise Chinese safety summary, and renders the captured
image in the Web chat message. Depth metadata and colorized depth images are
reserved for a later iteration.

If the RGB camera is already published as `sensor_msgs/Image`, the included
snapshot provider node can expose that Trigger service:

```bash
source /opt/ros/<distro>/setup.bash
./qwen35_env/bin/python Code/vision_snapshot_node.py \
  --image-topic /camera/color/image_raw \
  --service-name /vision/capture_snapshot
```

The node accepts common RGB image encodings (`rgb8`, `bgr8`, `rgba8`, `bgra8`,
and `mono8`) and writes JPEG snapshots under `.runtime/vision_snapshots/`.

## Optional Downloads

The testbed is offline once model directories exist. Use the downloader only
to fetch or refresh model assets:

```bash
./qwen35_env/bin/python Code/download.py qwen35-2b
./qwen35_env/bin/python Code/download.py qwen35-9b
./qwen35_env/bin/python Code/download.py asr-whisper-large-v3
./qwen35_env/bin/python Code/download.py asr-whisper-large-v3-turbo
```
