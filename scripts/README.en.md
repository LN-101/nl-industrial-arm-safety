# Project Scripts and Launch Utilities (`scripts/`)

This directory contains system launch scripts, process/cgroup resource scheduling helper libraries, and standalone routing/safety test tools.

> **Maintainer**: [@LN-101](https://github.com/LN-101) (Responsible for system-level launchers, cgroup resource scheduling, and integration test scripts)

---

## Scripts Catalog

| Script | Type | Description |
| --- | --- | --- |
| [`start_web_no_ros.sh`](start_web_no_ros.sh) | Shell script | **Web / Voice dry-run launcher**. Starts the mobile Web UI with `--dry-run-ros2` by default (no actual robot hardware commands published), with optional Orbbec camera snapshot service support. |
| [`start_web_with_ros.sh`](start_web_with_ros.sh) | Shell script | **Web + ROS 2 integrated graph launcher**. Automatically builds and sources the ROS 2 workspace, launches `main/launch/arm.launch.py` (IK control, human-arm distance monitoring, multi-source estop arbitration), and starts the Web UI. |
| [`launcher_nice.sh`](launcher_nice.sh) | Shell library | **Process scheduling and cgroup resource helper**. Sourced by launch scripts to validate nice priorities, cgroup v2 `CPUWeight`, CPU core affinity (`taskset`), and systemd user session support. |
| [`test_rule_query_routing.py`](test_rule_query_routing.py) | Python script | **Rule query and router prompt test tool**. Standalone verification for natural-language rule query detection (`should_read_rules`), router prompt construction, and routing coverage. |

---

## Detailed Usage

### 1. `start_web_no_ros.sh` (Dry-Run Launch)

Suitable for algorithm debugging, Web UI frontend development, ASR speech recognition, LLM dialog logic, and MeloTTS/Piper speech synthesis testing without real robot hardware.

#### How to Run

From the repository root:

```bash
./scripts/start_web_no_ros.sh
```

#### Common Options

- `--with-vision-service`: Build and start the ROS 2 camera snapshot service (for VLM visual context extraction);
- `--skip-vision-build`: Skip building the `camera` package before starting the vision service;
- `--vision-service-name <name>`: ROS 2 Trigger snapshot service name (default: `/vision/capture_snapshot`);
- `--vision-output-dir <dir>`: Directory to save captured snapshots (default: `$ROS2_WORKSPACE/.runtime/vision_snapshots`);
- `--vision-show-window`: Show the OpenCV preview windows for `min_dis`;
- All other arguments are forwarded to [`Code/web_ui.py`](../Code/web_ui.py).

---

### 2. `start_web_with_ros.sh` (Full ROS 2 Graph Launch)

Suitable for hardware execution or complete ROS 2 graph integration. Performs workspace cleanliness checks, starts low-level arm control nodes, and launches the Web UI with live ROS 2 command publishing.

#### How to Run

From the repository root:

```bash
./scripts/start_web_with_ros.sh
```

#### Common Options

- `--with-ros-launch`: Start default arm ROS 2 launch (default);
- `--no-ros-launch`: Attach Web UI without starting duplicate nodes (when another terminal already owns the ROS 2 graph);
- `--skip-ros-build`: Skip workspace `colcon build`;
- `--no-min-dis-window`: Disable OpenCV preview windows while keeping detection active;
- `--ros-launch-package <pkg>`: ROS 2 launch package (default: `main`);
- `--ros-launch-file <file>`: ROS 2 launch file (default: `arm.launch.py`).

> [!IMPORTANT]
> The launcher blocks execution if there are uncommitted changes in the ROS 2 workspace to ensure safety. Set `AI_OV_ALLOW_DIRTY_ROS=1` if you need to run local uncommitted code during experiments.

---

### 3. `launcher_nice.sh` (Resource Scheduling Helper)

Sourced automatically by `start_web_no_ros.sh` and `start_web_with_ros.sh`:

- **Priority Validation**: Validates `AI_OV_VOICE_NICE` and `AI_OV_MIN_DIS_NICE` (range `0..19`) and prevents unprivileged escalation;
- **cgroup v2 CPUWeight Isolation**: Uses `systemd-run --user --scope` to isolate voice/web processes (weight `500`) from vision inference (weight `25`), preventing YOLO inference from starving P-cores during streaming TTS;
- **CPU Affinity Validation**: Validates `taskset -c` format (e.g. `4-6` for E-cores);
- **Session Check**: Verifies active user systemd session and `cpu` controller delegation.

---

### 4. `test_rule_query_routing.py` (Rule Query Routing Test)

Verifies intent classification for safety rule query phrasing.

#### How to Run

From the repository root:

```bash
python scripts/test_rule_query_routing.py
```

---

## Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `AI_OV_PYTHON` | `./qwen35_env/bin/python` | Main Python interpreter path |
| `AI_OV_WEB_HOST` | `0.0.0.0` | Web server bind host |
| `AI_OV_WEB_PORT` | `8787` | Web server port |
| `AI_OV_WEB_ADMIN_USERNAME` | `admin` | Web login username |
| `AI_OV_WEB_ADMIN_PASSWORD` | `12345` | Web login password |
| `AI_OV_TTS_ENGINE` | `auto` | TTS engine selection (`auto` / `melo` / `piper`) |
| `AI_OV_ROS2_WORKSPACE` | `./ros2` | ROS 2 workspace root path |
| `AI_OV_ROS2_SETUP` | `/opt/ros/jazzy/setup.bash` | System ROS 2 environment script |
| `AI_OV_VOICE_CPU_WEIGHT` | `500` | cgroup CPUWeight for Web/Voice processes |
| `AI_OV_MIN_DIS_CPU_WEIGHT` | `25` | cgroup CPUWeight for vision distance node |
| `AI_OV_MIN_DIS_CPUS` | `4-6` | CPU core affinity for vision distance node |
