# Natural-Language Interactive Human-Robot Safety Collaboration System for Industrial Robotic Arms

**中文（默认）:** [README.md](README.md)

`LN-101/nl-industrial-arm-safety` is an integrated research and prototype
system for natural-language collaboration with an industrial robotic arm. It
combines a local OpenVINO voice/LLM/VLM stack, a mobile Web UI, ROS 2 control
nodes, active human-arm distance monitoring, and layered emergency-stop
handling.

> **Safety notice**
>
> This repository is a research/prototype system, not a safety-certified robot
> controller. AI, vision, ROS 2, serial feedback, and software emergency-stop
> logic must never be treated as the only protective measure. Validate the
> complete robot, controller, wiring, workspace, and emergency-stop chain with
> qualified engineers before any real-machine operation. Start with simulation
> or dry-run mode.
>
> The current Web runtime defaults to `admin` / `12345` for local development.
> Set a strong `AI_OV_WEB_ADMIN_PASSWORD` and bind the server to a trusted
> interface before enabling network access.

## What It Does

The system lets an operator use text or speech to inspect and act on a robot
workspace while keeping hardware commands behind deterministic validation and
ROS 2 boundaries.

| Capability | Current implementation | Runtime requirements |
| --- | --- | --- |
| Multimodal interaction | Qwen3.5 OpenVINO model path, text turns, image/vision context integration | Local model assets and OpenVINO |
| Free voice interaction | Whisper ASR, Chinese text normalization, streaming MeloTTS, optional Piper backend | Audio device, ASR/TTS assets |
| Remote Web UI | Authenticated mobile-first UI for text, voice, status, confirmations, and playback | Python runtime; HTTPS certificate generated locally |
| Workspace queries | Safety-rule, label/object mapping, arm-state, and vision-context tools | Checked-in example/config files; ROS 2 for live state |
| Controlled editing | Validated JSON rule and object-mapping replacement with confirmation flow | Writable runtime directory; never edit JSON directly from model text |
| Object collaboration | Natural-language object/label mapping and grasp command routing | ROS 2 arm/control packages and configured hardware |
| Active safety monitoring | Person/arm distance estimation, vision snapshots, and camera-source stop latching | Orbbec/Gemini camera, YOLO/OpenVINO assets, ROS 2 |
| Multi-source emergency stop | Voice, camera, feedback, and manual sources are routed through ROS 2 arbitration | ROS 2 graph and hardware safety chain |
| Simulation and GUI | MuJoCo launch path and ROS 2 arm GUI package | MuJoCo/PyQt5 and desktop display |

The independent STM32/CAN control board described in the project presentation is
an external hardware safety boundary. Its firmware and electrical safety design
are not contained in this repository.

## Architecture

```text
Operator voice/text
        |
        v
Mobile Web UI / CLI
        |
        v
ASR -> normalized intent -> validated tools -> LLM response -> TTS
                                      |
                                      v
                              ROS 2 message bridge
                                      |
            +-------------------------+-------------------------+
            |                         |                         |
       arm/control               camera/min_dis             main/estop
            |                         |                         |
            +------------ ROS 2 safety state ------------------+
                                      |
                                      v
                         serial feedback / arm controller
```

The active ROS 2 source of truth is the `ros2/` directory. The older
`FromBroToRos/` material is archival context and is not a runtime workspace.

## Repository Layout

```text
Code/                         Compatibility CLI entrypoints and tests
local_safety_assistant/       Voice, model, rule, Web, and ROS 2 bridge logic
ros2/                         Authoritative ROS 2 workspace imported for this monorepo
  src/camera/                 Camera, distance, vision snapshot, and camera stop node
  src/control/                IK, DRL, hand-eye, and arm control helpers
  src/main/                   Launch files, arm state, and emergency-stop aggregator
  src/arm_asset/              URDF/MJCF assets and meshes
  src/mujoco_sim/             MuJoCo simulation node
  src/arm_gui/                Desktop ROS 2 status/stop GUI
start_web_no_ros.sh           Web/voice dry-run launcher
start_web_with_ros.sh         Web + ROS 2 real-graph launcher
```

Historical project material, internal audit reports, and the original
development snapshots are retained in the local Git history but are not part
of the public GitHub release.

Generated directories, local models, virtual environments, caches, build
outputs, and machine-specific certificates are intentionally excluded by
`.gitignore`.

## Prerequisites

The current development path targets a Linux host with:

* Python 3.12 and an OpenVINO-capable Intel device (CPU/GPU/NPU as available).
* ROS 2 Jazzy, `colcon`, and the dependencies declared by the packages under
  `ros2/src/`.
* Optional: an Orbbec/Gemini camera with its vendor SDK and udev rules.
* Optional: a serial-connected arm controller at the configured device path
  (the current ROS implementation defaults to `/dev/ttyUSB0`).
* Optional: MuJoCo, PyQt5, and a desktop display for simulation/GUI paths.

Voice/LLM model weights, Python environments, the MeloTTS runtime, vendor SDKs,
and other external assets are not bundled. The ROS 2 workspace does contain the
small YOLO/OpenVINO and controller artifacts currently required by its camera
and control packages; review their provenance and licenses before
redistribution. The repository currently does not promise a single lockfile
that reproduces every hardware-specific environment.

## Quick Start: Model and CLI Smoke Tests

Run commands from the repository root. The existing project scripts assume a
Python environment named `qwen35_env`; use another interpreter by replacing
the path in each command.

```bash
./qwen35_env/bin/python Code/test.py --list-devices
./qwen35_env/bin/python Code/test.py inventory
./qwen35_env/bin/python Code/voice_stack.py plan
```

Download supported OpenVINO model assets into the ignored `models/` directory
when needed:

```bash
./qwen35_env/bin/python Code/download.py qwen35-2b
./qwen35_env/bin/python Code/download.py asr-whisper-large-v3-turbo
```

The downloader uses Hugging Face repositories and accepts `HF_TOKEN` when a
model host requires authentication. Review the model card and license before
redistributing any downloaded weights.

Exercise a text turn without synthesizing audio or publishing real ROS 2
commands:

```bash
./qwen35_env/bin/python Code/voice_stack.py text-turn \
  --text "请说明机械臂急停规则" \
  --skip-tts \
  --dry-run-ros2
```

For audio-file and microphone commands, see
[`Code/README.md`](Code/README.md), which documents the current CLI options and
model-specific diagnostics.

## Quick Start: Web UI Without ROS 2

Use the dry-run launcher for local UI and voice-stack work. It does not publish
real arm commands:

```bash
AI_OV_WEB_HOST=127.0.0.1 \
AI_OV_WEB_ADMIN_PASSWORD='replace-this-password' \
./start_web_no_ros.sh
```

The launcher generates a local self-signed certificate under the ignored
`.runtime/web_ui/ssl/` directory. A browser warning is expected for that local
certificate. Use `--help` to inspect vision-service and Web UI overrides.

## Quick Start: ROS 2 Integration

The monorepo launchers automatically use `./ros2` when it exists. You can
override the workspace with `AI_OV_ROS2_WORKSPACE` for an external checkout.

Build the workspace first:

```bash
source /opt/ros/jazzy/setup.bash
cd ros2
colcon build
source install/setup.bash
cd ..
```

Then start the integrated Web + ROS 2 path:

```bash
AI_OV_WEB_HOST=127.0.0.1 \
AI_OV_WEB_ADMIN_PASSWORD='replace-this-password' \
./start_web_with_ros.sh
```

By default this starts `ros2 launch main arm.launch.py`, which is the real-robot
launch path and does not start MuJoCo or the K230 camera node. Use
`./start_web_with_ros.sh --no-ros-launch` only when another terminal already
owns the intended ROS 2 graph. The launcher refuses dirty ROS 2 source/config
changes by default; use `AI_OV_ALLOW_DIRTY_ROS=1` only for a deliberate local
experiment.

For the vision snapshot service, the ROS 2 package exposes a Trigger service:

```bash
ros2 service call /vision/capture_snapshot std_srvs/srv/Trigger {}
```

The full ROS 2 package notes and topic/service examples are in
[`ros2/readme.md`](ros2/readme.md).

## Safety and Security Boundaries

* Natural-language output is not a direct hardware command channel. Voice/LLM
  intent is normalized, validated, confirmed where required, and translated to
  deterministic ROS 2 messages.
* Rule and mapping changes must use the validated persistence helpers. Do not
  edit runtime JSON files from a model response or bypass confirmation logic.
* Camera distance failures and stale frames are safety states, not proof that a
  workspace is clear. Validate the complete camera, ROS 2, controller, and
  hardware stop behavior on the target machine.
* The Web UI is intended for a trusted LAN during development. Change the
  default credentials, bind to an appropriate interface, and place production
  deployments behind a suitable network boundary.
* The source tree contains hardware-specific paths, calibration values, model
  assets, and serial assumptions. Review them before adapting the system to a
  different arm or camera.

Before operating real hardware, perform a target-machine safety review of the
camera, ROS 2, controller, and emergency-stop paths. The public source is not
a substitute for commissioning, risk assessment, or an approved machine-safety
procedure.

## Tests and Checks

Without model or hardware startup, the deterministic test suites can be run as:

```bash
./qwen35_env/bin/python -m unittest discover -s Code/tests
./qwen35_env/bin/python -m py_compile \
  Code/voice_stack.py Code/web_ui.py \
  local_safety_assistant/stack/*.py \
  local_safety_assistant/web/*.py
bash -n start_web_no_ros.sh start_web_with_ros.sh
```

For ROS 2 package checks, source the ROS distribution and run the package test
workflow from `ros2/`:

```bash
source /opt/ros/jazzy/setup.bash
cd ros2
colcon test --event-handlers console_direct+
```

Some checks require OpenVINO model files, a camera, MuJoCo, a display, or a
serial controller and will not be meaningful on a machine without those
dependencies.

At this snapshot, the ROS 2 workspace builds successfully, while `colcon test`
still reports pre-existing style/docstring failures and three arm-state
calibration/timing assertion mismatches. This README does not claim that every
ROS 2 test is green.

## Project Documentation

* [`Code/README.md`](Code/README.md) — detailed local model, voice-stack, and Web commands.
* [`ros2/readme.md`](ros2/readme.md) — ROS 2 vision snapshot and workspace notes.

## Contributing

Contributions should preserve the separation between natural-language
orchestration, validated safety rules, ROS 2 command routing, camera safety
state, and the hardware controller boundary. Include focused tests for changes
to routing, rule validation, emergency-stop behavior, and launch contracts.
Do not submit model weights, virtual environments, generated runtime state,
credentials, or unreviewed teammate snapshots.

## License

Project code and documentation are released under the
[Apache License 2.0](LICENSE). Third-party models, vendor SDKs, binary assets,
and external runtime trees may have separate licenses and are not covered by
this root license. Review their individual terms before redistribution.
