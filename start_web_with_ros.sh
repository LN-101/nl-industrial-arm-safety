#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT_DIR/launcher_nice.sh"
PYTHON_BIN="${AI_OV_PYTHON:-$ROOT_DIR/qwen35_env/bin/python}"

HOST="${AI_OV_WEB_HOST:-0.0.0.0}"
PORT="${AI_OV_WEB_PORT:-8787}"
ADMIN_USERNAME="${AI_OV_WEB_ADMIN_USERNAME:-admin}"
ADMIN_PASSWORD="${AI_OV_WEB_ADMIN_PASSWORD:-12345}"
RUNTIME_DIR="${AI_OV_WEB_RUNTIME_DIR:-$ROOT_DIR/.runtime/web_ui}"
TTS_ENGINE="${AI_OV_TTS_ENGINE:-auto}"
MOSS_PCM_BUFFER_SECONDS="${AI_OV_MOSS_PCM_BUFFER_SECONDS:-0.48}"
MOSS_CPUS="${AI_OV_MOSS_CPUS-}"
SSL_DIR="${AI_OV_WEB_SSL_DIR:-$RUNTIME_DIR/ssl}"
SSL_CERTFILE="${AI_OV_WEB_SSL_CERTFILE:-$SSL_DIR/webui.crt}"
SSL_KEYFILE="${AI_OV_WEB_SSL_KEYFILE:-$SSL_DIR/webui.key}"
SSL_SANFILE="${AI_OV_WEB_SSL_SANFILE:-$SSL_DIR/webui.san}"
SSL_DAYS="${AI_OV_WEB_SSL_DAYS:-3650}"
VOICE_NICE="${AI_OV_VOICE_NICE:-0}"
MIN_DIS_NICE="${AI_OV_MIN_DIS_NICE:-10}"
VOICE_CPU_WEIGHT="${AI_OV_VOICE_CPU_WEIGHT:-500}"
MIN_DIS_CPU_WEIGHT="${AI_OV_MIN_DIS_CPU_WEIGHT:-25}"
MIN_DIS_CPUS="${AI_OV_MIN_DIS_CPUS:-4-6}"
MIN_DIS_CPU_THREADS="${AI_OV_MIN_DIS_CPU_THREADS:-3}"

DEFAULT_ROS2_WORKSPACE="$ROOT_DIR/ros2"
if [[ ! -d "$DEFAULT_ROS2_WORKSPACE" ]]; then
  DEFAULT_ROS2_WORKSPACE="/home/inteldk/ROS2"
fi
ROS2_WORKSPACE="${AI_OV_ROS2_WORKSPACE:-$DEFAULT_ROS2_WORKSPACE}"
ROS2_SETUP="${AI_OV_ROS2_SETUP:-/opt/ros/jazzy/setup.bash}"
ROS2_WORKSPACE_SETUP="${AI_OV_ROS2_WORKSPACE_SETUP:-$ROS2_WORKSPACE/install/setup.bash}"
ROS2_LAUNCH_PACKAGE="${AI_OV_ROS2_LAUNCH_PACKAGE:-main}"
ROS2_LAUNCH_FILE="${AI_OV_ROS2_LAUNCH_FILE:-arm.launch.py}"

WEB_PID=""
ROS_LAUNCH_PID=""
START_ROS_LAUNCH=1
BUILD_ROS_WORKSPACE=1
SHOW_MIN_DIS_WINDOW=1
WEB_ARGS=()
LAUNCHER_NICE=""
VOICE_NICE_ADJUSTMENT=""
ROS_NICE_ADJUSTMENT=""

usage() {
  cat <<'EOF'
Usage: ./start_web_with_ros.sh [launcher options] [web_ui.py options]

Launcher options:
  --with-ros-launch        Start the arm ROS launch (default; kept for compatibility).
  --no-ros-launch          Attach to an already running ROS2 graph without starting launch.
  --skip-ros-build         Start the ROS2 launch without rebuilding the workspace.
  --no-min-dis-window      Disable the min_dis OpenCV window; detection stays active.
  --ros-launch-package PKG ROS2 launch package. Default: main.
  --ros-launch-file FILE   ROS2 launch file. Default: arm.launch.py.
  -h, --help               Show this help.

All other options are forwarded to Code/web_ui.py. By default, this launcher
builds/sources the ROS2 workspace, starts the default arm launch, and starts the
Web UI with real ROS2 publishing enabled, so it does not pass --dry-run-ros2.
The workspace build uses plain colcon build and inherits the ROS environment.
The default arm.launch.py path does not start MuJoCo simulation or K230.
Web/ASR uses AI_OV_VOICE_NICE (default 0); MOSS TTS inherits that priority.
The arm launch keeps control nodes at nice 0 and gives only min_dis the
AI_OV_MIN_DIS_NICE value (default 10).
CPU contention is arbitrated with cgroup v2 CPUWeight scopes because nice is
neutralized across setsid sessions by kernel autogrouping: Web/ASR/MOSS runs
at AI_OV_VOICE_CPU_WEIGHT (default 500) and only min_dis runs at
AI_OV_MIN_DIS_CPU_WEIGHT (default 25); control/arm/estop stay at the default
weight 100. min_dis is additionally pinned to AI_OV_MIN_DIS_CPUS (default 4-6,
the E-cores) so vision inference cannot throttle the P-cores' turbo headroom
that streaming TTS depends on. Its PyTorch/BLAS thread pool uses
AI_OV_MIN_DIS_CPU_THREADS (default 3). Requires an active user systemd session.
Use ./start_web_no_ros.sh for dry-run Web testing.

Pass --no-ros-launch only when another terminal already owns the ROS2 graph and
you want this script to attach Web publishing without starting a duplicate of:
  ros2 launch main arm.launch.py

The launcher refuses to build/launch when the ROS2 workspace has uncommitted
source or configuration changes, so the arm only runs reviewed, committed code.
Local model assets under src/camera/models are excluded from this check. Set
AI_OV_ALLOW_DIRTY_ROS=1 to bypass the remaining check deliberately.
EOF
}

parse_args() {
  while (($#)); do
    case "$1" in
      --with-ros-launch)
        START_ROS_LAUNCH=1
        shift
        ;;
      --no-ros-launch)
        START_ROS_LAUNCH=0
        shift
        ;;
      --skip-ros-build)
        BUILD_ROS_WORKSPACE=0
        shift
        ;;
      --no-min-dis-window)
        SHOW_MIN_DIS_WINDOW=0
        shift
        ;;
      --ros-launch-package)
        if (($# < 2)); then
          echo "[ai-ov] --ros-launch-package requires a value." >&2
          exit 2
        fi
        ROS2_LAUNCH_PACKAGE="$2"
        shift 2
        ;;
      --ros-launch-file)
        if (($# < 2)); then
          echo "[ai-ov] --ros-launch-file requires a value." >&2
          exit 2
        fi
        ROS2_LAUNCH_FILE="$2"
        shift 2
        ;;
      --dry-run-ros2)
        echo "[ai-ov] start_web_with_ros.sh does not accept --dry-run-ros2." >&2
        echo "[ai-ov] Use ./start_web_no_ros.sh for dry-run Web testing." >&2
        exit 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      --)
        shift
        WEB_ARGS+=("$@")
        break
        ;;
      *)
        WEB_ARGS+=("$1")
        shift
        ;;
    esac
  done
}

resolve_lan_ipv4() {
  local candidate
  for candidate in $(hostname -I 2>/dev/null || true); do
    if [[ "$candidate" == *.* ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 0
}

ensure_https_certificate() {
  local lan_ip
  local san_entries
  lan_ip="$(resolve_lan_ipv4)"
  san_entries="DNS:localhost,IP:127.0.0.1"
  if [[ -n "$lan_ip" ]]; then
    san_entries="$san_entries,IP:$lan_ip"
  fi

  if [[ -f "$SSL_CERTFILE" && -f "$SSL_KEYFILE" && -f "$SSL_SANFILE" ]] && [[ "$(cat "$SSL_SANFILE")" == "$san_entries" ]]; then
    return
  fi
  if ! command -v openssl >/dev/null 2>&1; then
    echo "[ai-ov] openssl is required to generate the default HTTPS certificate." >&2
    echo "[ai-ov] Or set AI_OV_WEB_SSL_CERTFILE and AI_OV_WEB_SSL_KEYFILE to existing files." >&2
    exit 2
  fi
  mkdir -p "$SSL_DIR"
  chmod 700 "$SSL_DIR"

  echo "[ai-ov] generating self-signed HTTPS certificate: $SSL_CERTFILE"
  openssl req \
    -x509 \
    -newkey rsa:2048 \
    -sha256 \
    -nodes \
    -days "$SSL_DAYS" \
    -keyout "$SSL_KEYFILE" \
    -out "$SSL_CERTFILE" \
    -subj "/CN=ai-ov-webui" \
    -addext "subjectAltName = $san_entries" \
    >/dev/null 2>&1
  printf '%s\n' "$san_entries" > "$SSL_SANFILE"
  chmod 600 "$SSL_KEYFILE"
  chmod 644 "$SSL_CERTFILE"
  chmod 644 "$SSL_SANFILE"
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  set +e
  request_stop_process_group "$WEB_PID" "web/ROS runtime"
  request_stop_process_group "$ROS_LAUNCH_PID" "ROS2 launch"
  wait_for_process_group_stop "$WEB_PID" "web/ROS runtime"
  wait_for_process_group_stop "$ROS_LAUNCH_PID" "ROS2 launch"
  exit "$status"
}

request_stop_process_group() {
  local pid="$1"
  local label="$2"
  if ! process_group_is_running "$pid"; then
    return
  fi
  echo "[ai-ov] stopping $label..."
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
}

wait_for_process_group_stop() {
  local pid="$1"
  local label="$2"
  if ! process_group_is_running "$pid"; then
    return
  fi
  for _ in {1..40}; do
    if ! process_group_is_running "$pid"; then
      return
    fi
    sleep 0.25
  done
  echo "[ai-ov] force stopping $label..."
  kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
}

process_group_is_running() {
  local pgid="${1:-}"
  [[ -n "$pgid" ]] || return 1
  kill -0 -- "-$pgid" 2>/dev/null
}

source_ros_setup() {
  local setup_file="$1"
  set +u
  source "$setup_file"
  set -u
}

validate_no_dry_run_forwarding() {
  local arg
  for arg in "${WEB_ARGS[@]}"; do
    if [[ "$arg" == "--dry-run-ros2" ]]; then
      echo "[ai-ov] start_web_with_ros.sh does not accept --dry-run-ros2." >&2
      echo "[ai-ov] Use ./start_web_no_ros.sh for dry-run Web testing." >&2
      exit 2
    fi
  done
}

validate_ros_paths() {
  if [[ ! -d "$ROS2_WORKSPACE" ]]; then
    echo "[ai-ov] ROS2 workspace not found: $ROS2_WORKSPACE" >&2
    echo "[ai-ov] Set AI_OV_ROS2_WORKSPACE=/path/to/ROS2 to override." >&2
    exit 2
  fi
  if [[ ! -f "$ROS2_SETUP" ]]; then
    echo "[ai-ov] ROS2 setup file not found: $ROS2_SETUP" >&2
    echo "[ai-ov] Set AI_OV_ROS2_SETUP=/opt/ros/<distro>/setup.bash to override." >&2
    exit 2
  fi
  if [[ ! -f "$ROS2_WORKSPACE_SETUP" ]]; then
    echo "[ai-ov] ROS2 workspace setup file not found: $ROS2_WORKSPACE_SETUP" >&2
    echo "[ai-ov] Build the workspace first, or set AI_OV_ROS2_WORKSPACE_SETUP=/path/to/setup.bash." >&2
    exit 2
  fi
}

validate_ros_build_inputs() {
  if [[ ! -d "$ROS2_WORKSPACE" ]]; then
    echo "[ai-ov] ROS2 workspace not found: $ROS2_WORKSPACE" >&2
    echo "[ai-ov] Set AI_OV_ROS2_WORKSPACE=/path/to/ROS2 to override." >&2
    exit 2
  fi
  if [[ ! -f "$ROS2_SETUP" ]]; then
    echo "[ai-ov] ROS2 setup file not found: $ROS2_SETUP" >&2
    echo "[ai-ov] Set AI_OV_ROS2_SETUP=/opt/ros/<distro>/setup.bash to override." >&2
    exit 2
  fi
}

source_ros_environment() {
  validate_ros_paths
  source_ros_setup "$ROS2_SETUP"
  source_ros_setup "$ROS2_WORKSPACE_SETUP"
  if ! command -v ros2 >/dev/null 2>&1; then
    echo "[ai-ov] ros2 command is unavailable after sourcing ROS2 setup files." >&2
    exit 2
  fi
}

ensure_python_can_import_ros2() {
  if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import rclpy
PY
  then
    echo "[ai-ov] Python runtime cannot import rclpy after sourcing ROS2." >&2
    echo "[ai-ov] Python: $PYTHON_BIN" >&2
    echo "[ai-ov] ROS2 setup: $ROS2_SETUP" >&2
    echo "[ai-ov] Workspace setup: $ROS2_WORKSPACE_SETUP" >&2
    exit 2
  fi
}

ensure_ros_workspace_clean() {
  local dirty_status
  if [[ "${AI_OV_ALLOW_DIRTY_ROS:-0}" == "1" ]]; then
    echo "[ai-ov] AI_OV_ALLOW_DIRTY_ROS=1 set; skipping ROS2 workspace cleanliness check." >&2
    return 0
  fi
  if ! command -v git >/dev/null 2>&1 \
    || ! git -C "$ROS2_WORKSPACE" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "[ai-ov] ROS2 workspace is not a git checkout; skipping cleanliness check." >&2
    return 0
  fi
  dirty_status="$(
    git -C "$ROS2_WORKSPACE" status --porcelain --untracked-files=all -- \
      . \
      ':(exclude)src/camera/models/**'
  )"
  if [[ -n "$dirty_status" ]]; then
    echo "[ai-ov] ROS2 workspace has uncommitted changes; refusing to run them on the arm:" >&2
    printf '%s\n' "$dirty_status" >&2
    echo "[ai-ov] Commit or stash the changes first (dirty ROS2 workspaces are blocked by the safety policy)," >&2
    echo "[ai-ov] or set AI_OV_ALLOW_DIRTY_ROS=1 to launch anyway." >&2
    exit 2
  fi
}

build_ros_workspace() {
  validate_ros_build_inputs
  echo "[ai-ov] building ROS2 workspace before launch..."
  (
    cd "$ROS2_WORKSPACE"
    source_ros_setup "$ROS2_SETUP"
    if ! command -v colcon >/dev/null 2>&1; then
      echo "[ai-ov] colcon command is unavailable after sourcing ROS2." >&2
      exit 2
    fi
    colcon build
  )
}

start_ros_launch() {
  ensure_ros_workspace_clean
  if ((BUILD_ROS_WORKSPACE)); then
    build_ros_workspace
    ROS2_WORKSPACE_SETUP="${AI_OV_ROS2_WORKSPACE_SETUP:-$ROS2_WORKSPACE/install/setup.bash}"
  fi
  source_ros_environment
  local ros_launch_cmd=(ros2 launch "$ROS2_LAUNCH_PACKAGE" "$ROS2_LAUNCH_FILE")
  if [[ "$ROS2_LAUNCH_PACKAGE" == "main" && "$ROS2_LAUNCH_FILE" == "arm.launch.py" ]]; then
    ros_launch_cmd+=("min_dis_nice:=$MIN_DIS_NICE" "min_dis_cpu_weight:=$MIN_DIS_CPU_WEIGHT" "min_dis_cpus:=$MIN_DIS_CPUS" "min_dis_cpu_threads:=$MIN_DIS_CPU_THREADS")
  fi
  if ((!SHOW_MIN_DIS_WINDOW)); then
    ros_launch_cmd+=(show_window:=false)
  fi
  echo "[ai-ov] starting ROS2 launch: ${ros_launch_cmd[*]}"
  setsid nice -n "$ROS_NICE_ADJUSTMENT" "${ros_launch_cmd[@]}" &
  ROS_LAUNCH_PID=$!
  sleep 2
  if ! process_group_is_running "$ROS_LAUNCH_PID"; then
    echo "[ai-ov] ROS2 launch exited during startup." >&2
    exit 1
  fi
}

wait_for_children() {
  if [[ -n "$ROS_LAUNCH_PID" ]]; then
    wait -n "$WEB_PID" "$ROS_LAUNCH_PID"
  else
    wait "$WEB_PID"
  fi
}

parse_args "$@"
validate_no_dry_run_forwarding

validate_nice_value "$VOICE_NICE" AI_OV_VOICE_NICE
validate_nice_value "$MIN_DIS_NICE" AI_OV_MIN_DIS_NICE
validate_cpu_weight "$VOICE_CPU_WEIGHT" AI_OV_VOICE_CPU_WEIGHT
validate_cpu_weight "$MIN_DIS_CPU_WEIGHT" AI_OV_MIN_DIS_CPU_WEIGHT
validate_cpu_list "$MIN_DIS_CPUS" AI_OV_MIN_DIS_CPUS
validate_positive_integer "$MIN_DIS_CPU_THREADS" AI_OV_MIN_DIS_CPU_THREADS
ensure_user_cpu_weight_support
LAUNCHER_NICE="$(read_effective_nice)"
VOICE_NICE_ADJUSTMENT="$(nice_adjustment "$VOICE_NICE" "$LAUNCHER_NICE" AI_OV_VOICE_NICE)"
if ((START_ROS_LAUNCH)); then
  ROS_NICE_ADJUSTMENT="$(nice_adjustment 0 "$LAUNCHER_NICE" "ROS control nice")"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ai-ov] Python runtime not found or not executable: $PYTHON_BIN" >&2
  echo "[ai-ov] Set AI_OV_PYTHON=/path/to/python if you want to use another environment." >&2
  exit 2
fi

if ! command -v setsid >/dev/null 2>&1; then
  echo "[ai-ov] setsid is required so the launcher can clean process groups." >&2
  exit 2
fi

trap cleanup EXIT INT TERM

if ((START_ROS_LAUNCH)); then
  start_ros_launch
else
  source_ros_environment
fi
ensure_python_can_import_ros2
ensure_https_certificate

LAN_IP="$(resolve_lan_ipv4)"

echo "[ai-ov] starting Web UI with real ROS2 publishing enabled..."
echo "[ai-ov] Local URL: https://127.0.0.1:$PORT/"
if [[ -n "$LAN_IP" ]]; then
  echo "[ai-ov] Phone URL: https://$LAN_IP:$PORT/"
fi
echo "[ai-ov] Bind address: $HOST:$PORT"
echo "[ai-ov] Login: $ADMIN_USERNAME / $ADMIN_PASSWORD"
echo "[ai-ov] Runtime dir: $RUNTIME_DIR"
echo "[ai-ov] TTS engine: $TTS_ENGINE"
echo "[ai-ov] MOSS PCM target buffer: ${MOSS_PCM_BUFFER_SECONDS}s"
echo "[ai-ov] MOSS CPU affinity: ${MOSS_CPUS:-disabled}"
echo "[ai-ov] Web/ASR effective nice: $VOICE_NICE"
echo "[ai-ov] MOSS TTS effective nice: inherits Web ($VOICE_NICE)"
echo "[ai-ov] Web/ASR/MOSS CPU weight: $VOICE_CPU_WEIGHT (scope ai-ov-voice-$$)"
echo "[ai-ov] HTTPS cert: $SSL_CERTFILE"
echo "[ai-ov] ROS2 setup: $ROS2_SETUP"
echo "[ai-ov] ROS2 workspace setup: $ROS2_WORKSPACE_SETUP"
if ((START_ROS_LAUNCH)); then
  echo "[ai-ov] ROS2 launch: $ROS2_LAUNCH_PACKAGE $ROS2_LAUNCH_FILE"
  if [[ "$ROS2_LAUNCH_PACKAGE" == "main" && "$ROS2_LAUNCH_FILE" == "arm.launch.py" ]]; then
    echo "[ai-ov] ROS2 control/arm/estop effective nice: 0"
    echo "[ai-ov] ROS2 min_dis effective nice: $MIN_DIS_NICE"
    echo "[ai-ov] ROS2 control/arm/estop CPU weight: 100 (default)"
    echo "[ai-ov] ROS2 min_dis CPU weight: $MIN_DIS_CPU_WEIGHT"
    echo "[ai-ov] ROS2 min_dis CPU affinity: $MIN_DIS_CPUS"
    echo "[ai-ov] ROS2 min_dis CPU threads: $MIN_DIS_CPU_THREADS"
  else
    echo "[ai-ov] ROS2 launch inherited effective nice: 0"
  fi
else
  echo "[ai-ov] ROS2 launch: not started by this script"
fi

setsid systemd-run --user --scope --quiet --unit "ai-ov-voice-$$" \
  -p "CPUWeight=$VOICE_CPU_WEIGHT" \
  nice -n "$VOICE_NICE_ADJUSTMENT" "$PYTHON_BIN" "$ROOT_DIR/Code/web_ui.py" \
  --host "$HOST" \
  --port "$PORT" \
  --admin-username "$ADMIN_USERNAME" \
  --admin-password "$ADMIN_PASSWORD" \
  --runtime-dir "$RUNTIME_DIR" \
  --tts-engine "$TTS_ENGINE" \
  --moss-pcm-buffer-seconds "$MOSS_PCM_BUFFER_SECONDS" \
  --moss-cpus "$MOSS_CPUS" \
  --ssl-certfile "$SSL_CERTFILE" \
  --ssl-keyfile "$SSL_KEYFILE" \
  "${WEB_ARGS[@]}" &
WEB_PID=$!

wait_for_children
