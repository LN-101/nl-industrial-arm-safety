# `local_safety_assistant/` Local Safety Assistant Core

**中文（默认）:** [README.md](README.md)

This is the project's core Python package. It integrates local ASR, LLM/VLM, TTS, vision context, and the Web UI, while strictly bounding model outputs within verifiable rules, confirmation lifecycles, and deterministic ROS 2 message boundaries.

> **Module Maintainer**: [@LN-101](https://github.com/LN-101) (Responsible for AI core architecture, OpenVINO inference, Whisper/MeloTTS speech pipeline, safety rule engine, Web UI, and ROS 2 message bridge)

## Processing Pipeline

```text
Text / Audio Input
    |
    +--> stack/asr.py -------- Whisper ASR
    |
    +--> stack/pipeline.py --- Intent normalization, rule tools, LLM response
    |          |
    |          +-------------- rules.py / arm_rules.py
    |          +-------------- object_mapping.py
    |          +-------------- workspace_snapshot.py
    |
    +--> stack/tts.py -------- Melo / MOSS / Piper TTS
    |
    +--> stack/ros2_bridge.py  Deterministic ROS 2 message plan or publish
    |
    +--> web/service.py ------ Web session, confirmations, emergency stop, streaming audio
```

## Files and Modules

| Path | Responsibility |
| --- | --- |
| `app.py` | `status` command; checks OpenVINO devices, models, TTS runtimes, and rule files |
| `config.py` | Project root discovery, model aliases, and default asset paths |
| `rules.py` | Loading, schema & business validation, preview generation, and atomic writing of safety rule JSON |
| `arm_rules.py` | Robot arm E-stop, reset, slowdown requests, and runtime rule synchronization |
| `object_mapping.py` | Reading, validating, and updating A/B/C/D object-to-label mappings |
| `confirmation.py` | Confirmation state machine and lifecycle management for high-risk operations |
| `workspace_snapshot.py` | Aggregating rules, object mappings, arm status, and pending confirmations into workspace snapshots |
| `model_testbed.py` | OpenVINO model discovery, device inspection, and text/ASR generation smoke tests |
| `stack/asr.py` | Whisper OpenVINO ASR integration and transcription normalization |
| `stack/llm.py` | Qwen/OpenVINO text/VLM inference adapter |
| `stack/pipeline.py` | Single-turn interaction, tool routing, response synthesis, and action intent orchestration |
| `stack/tts.py` | Unified TTS interface supporting MeloTTS, MOSS, and Piper |
| `stack/devices.py` | CPU/GPU/NPU hardware probing and stage-wise device selection |
| `stack/microphone.py` | Microphone audio capture, voice activity/endpoint detection, and WAV input |
| `stack/vision.py` | ROS 2 Trigger vision snapshot parsing, path verification, and caching |
| `stack/vision_node.py` | Standalone ROS 2 camera snapshot Trigger node |
| `stack/ros2_bridge.py` | Deterministic translation of confirmed voice/text intents into ROS 2 message plans |
| `stack/safety_batch.py` | Batch voice safety regression test runner |
| `web/server.py` | HTTP/HTTPS server, authentication, routing, and static asset serving |
| `web/service.py` | Web service business state, confirmation flows, E-stop, chat session, and audio streaming |
| `web/ui.py` | Mobile-first control dashboard HTML/JS generation |
| `web/assets/` | E-stop alert sounds and localized text assets |

## Entry Commands

Run all commands from the repository root:

```bash
PY=./qwen35_env/bin/python

# Check model inventory and device allocation plan
$PY -m local_safety_assistant.app status
$PY Code/voice_stack.py plan

# Text turn without speech synthesis or live ROS 2 publishing
$PY Code/voice_stack.py text-turn \
  --text "Please explain the robotic arm emergency stop rules" \
  --skip-tts \
  --dry-run-ros2

# Launch Web server (prefer root launcher scripts during development)
$PY Code/web_ui.py --help
./scripts/start_web_no_ros.sh
```

`Code/voice_stack.py` and `Code/web_ui.py` are compatibility wrappers; the underlying CLI implementations reside in `stack/cli.py` and `web/server.py`.

## Safety Boundaries

### Rules and Mappings

The model is strictly restricted to proposing tool calls or rule patches and cannot write directly to JSON files. `rules.py` enforces:

- Document versioning, rule arrays, and unique IDs;
- Strict data types for `enabled`, conditions, and actions;
- Immutable identity fields and action types;
- Numerical range limits on human distance thresholds;
- Safe writing via temporary files, `fsync`, and atomic replacement (`os.replace`).

Object mappings undergo equivalent structural validation. All operations that alter safety rules or configurations must pass through the `confirmation.py` confirmation workflow.

### ROS 2 Message Planning

`stack/ros2_bridge.py` first generates an inspectable `Ros2MessagePlan`. Default topics include:

| Topic | Type | Description |
| --- | --- | --- |
| `/voice/transcript` | `std_msgs/String` | Recognized input text |
| `/voice/assistant_response` | `std_msgs/String` | Assistant text response |
| `/safety/estop/request` | `std_msgs/String` | E-stop request with source, latch flag, and reason |
| `/emergency_stop` | `std_msgs/Bool` | Direct emergency stop state |
| `/goal` | `geometry_msgs/Point` | Coordinate goal point |

Explanations, rule edits, object mappings, and grasp intents never bypass the tool layer to publish directly to `/goal`. Use `--dry-run-ros2` during development and testing to inspect message plans without real hardware interaction.

### Web Sessions

`web/service.py` manages authenticated sessions, pending confirmations, chat turn cancellation, external E-stops, and audio/image streaming. Default credentials (`admin` / `12345`) are intended for local development only; production deployments require updating passwords, restricting network interfaces, and configuring TLS.

## Configuration Sources

Default configuration is resolved via `stack/config.py`:

- ASR: `models/asr/whisper-large-v3-turbo-int4-ov`
- LLM: `models/Qwen3.5-2B-int4-ov`
- Rules: `Code/config/safety_rules.example.json`
- Object Mapping: `Code/config/object_mapping.example.json`
- Arm Rules: `Code/config/arm_rules.json`
- Vision Snapshot Service: `/vision/capture_snapshot`

Model weights, TTS runtimes, and virtual environments are not tracked in Git. When models are absent, status checks will explicitly report missing assets; do not commit large binary model files.

## Testing

```bash
./qwen35_env/bin/python -m unittest discover -s Code/tests
./qwen35_env/bin/python -m py_compile \
  local_safety_assistant/stack/*.py \
  local_safety_assistant/web/*.py
```

The test suite covers rule validation, confirmation lifecycles, ASR/TTS adapters, vision snapshot parsing, Web APIs, voice stack, and ROS 2 message planning. Passing unit tests does not substitute for hardware-in-the-loop safety acceptance on the physical robot arm.
