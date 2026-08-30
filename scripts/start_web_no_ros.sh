#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/launcher_nice.sh"
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
DEFAULT_ROS2_PYTHON="$ROS2_WORKSPACE/.venv/bin/python"
if [[ -n "${AI_OV_ROS2_PYTHON:-}" ]]; then
  ROS2_PYTHON="$AI_OV_ROS2_PYTHON"
elif [[ -x "$DEFAULT_ROS2_PYTHON" ]]; then
  ROS2_PYTHON="$DEFAULT_ROS2_PYTHON"
else
  ROS2_PYTHON="$(command -v python3 || true)"
fi
if [[ -n "${AI_OV_ROS2_PYTHONPATH_PREPEND+x}" ]]; then
  ROS2_PYTHONPATH_PREPEND="$AI_OV_ROS2_PYTHONPATH_PREPEND"
elif [[ "$ROS2_PYTHON" == "/usr/bin/python3" && -d /usr/lib/python3/dist-packages ]]; then
  ROS2_PYTHONPATH_PREPEND="/usr/lib/python3/dist-packages"
else
  ROS2_PYTHONPATH_PREPEND=""
fi
VISION_SERVICE_NAME="${AI_OV_VISION_SERVICE_NAME:-/vision/capture_snapshot}"
VISION_OUTPUT_DIR="${AI_OV_VISION_OUTPUT_DIR:-$ROS2_WORKSPACE/.runtime/vision_snapshots}"
VISION_SHOW_WINDOW="${AI_OV_VISION_SHOW_WINDOW:-false}"
ORBBEC_UDEV_RULES_SOURCE="${AI_OV_ORBBEC_UDEV_RULES_SOURCE:-}"
ORBBEC_UDEV_RULES_DEST="${AI_OV_ORBBEC_UDEV_RULES_DEST:-/etc/udev/rules.d/99-obsensor-libusb.rules}"
ORBBEC_GEMINI336_VENDOR_ID="2bc5"
ORBBEC_GEMINI336_PRODUCT_ID="0803"

WEB_PID=""
VISION_PID=""
START_VISION_SERVICE=0
BUILD_VISION_SERVICE=1
WEB_ARGS=()
LAUNCHER_NICE=""
VOICE_NICE_ADJUSTMENT=""
MIN_DIS_NICE_ADJUSTMENT=""

usage() {
  cat <<'EOF'
Usage: ./scripts/start_web_no_ros.sh [launcher options] [web_ui.py options]

Launcher options:
  --with-vision-service      Build and start the ROS2 camera snapshot service.
  --skip-vision-build        Start the vision service without rebuilding camera.
  --vision-service-name NAME ROS2 Trigger service name. Default: /vision/capture_snapshot.
  --vision-output-dir DIR    Snapshot output directory in the ROS2 workspace.
  --vision-show-window       Show the min_dis OpenCV preview windows.
  -h, --help                 Show this help.

All other options are forwarded to Code/web_ui.py. The Web UI still runs with
--dry-run-ros2 by default; this script only starts the vision snapshot service.
The vision service is provided by the ROS2 camera min_dis node.
Web/ASR uses AI_OV_VOICE_NICE (default 0); MOSS TTS inherits that priority.
The optional min_dis process uses AI_OV_MIN_DIS_NICE (default 10).
CPU contention is arbitrated with cgroup v2 CPUWeight scopes (nice is
neutralized across setsid sessions by autogrouping): Web/ASR/MOSS runs at
AI_OV_VOICE_CPU_WEIGHT (default 500) and the optional min_dis at
AI_OV_MIN_DIS_CPU_WEIGHT (default 25). min_dis is also pinned to
AI_OV_MIN_DIS_CPUS (default 4-6, the E-cores) so vision inference cannot
throttle the P-cores that streaming TTS needs. Its PyTorch/BLAS thread pool
uses AI_OV_MIN_DIS_CPU_THREADS (default 3). Requires an active user systemd session.
For Orbbec/Gemini cameras, install pyorbbecsdk's udev rules first if this
launcher reports a missing 2bc5:0803 rule.
EOF
}

parse_args() {
  while (($#)); do
    case "$1" in
      --with-vision-service)
        START_VISION_SERVICE=1
        shift
        ;;
      --skip-vision-build)
        BUILD_VISION_SERVICE=0
        shift
        ;;
      --vision-service-name)
        if (($# < 2)); then
          echo "[ai-ov] --vision-service-name requires a value." >&2
          exit 2
        fi
        VISION_SERVICE_NAME="$2"
        shift 2
        ;;
      --vision-output-dir)
        if (($# < 2)); then
          echo "[ai-ov] --vision-output-dir requires a value." >&2
          exit 2
        fi
        VISION_OUTPUT_DIR="$2"
        shift 2
        ;;
      --vision-show-window)
        VISION_SHOW_WINDOW=true
        shift
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
  set +e  # prevent errexit from aborting cleanup mid-way
  request_stop_process_group "$VISION_PID" "ROS2 vision snapshot service"
  request_stop_process_group "$WEB_PID" "web/no-ROS runtime"
  wait_for_process_group_stop "$VISION_PID" "ROS2 vision snapshot service"
  wait_for_process_group_stop "$WEB_PID" "web/no-ROS runtime"
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

with_ros2_pythonpath() {
  if [[ -n "$ROS2_PYTHONPATH_PREPEND" ]]; then
    PYTHONPATH="$ROS2_PYTHONPATH_PREPEND${PYTHONPATH:+:$PYTHONPATH}" "$@"
  else
    "$@"
  fi
}

find_orbbec_udev_rules_source() {
  if [[ -n "$ORBBEC_UDEV_RULES_SOURCE" ]]; then
    printf '%s\n' "$ORBBEC_UDEV_RULES_SOURCE"
    return 0
  fi
  with_ros2_pythonpath "$ROS2_PYTHON" - <<'PY'
from pathlib import Path
import importlib.util

spec = importlib.util.find_spec("pyorbbecsdk")
if spec is None or spec.origin is None:
    raise SystemExit(0)
rules = Path(spec.origin).resolve().parent / "shared" / "99-obsensor-libusb.rules"
if rules.is_file():
    print(rules)
PY
}

udev_rule_file_contains_gemini336_rule() {
  local rules_file="$1"
  [[ -r "$rules_file" ]] || return 1
  awk \
    -v vendor="$ORBBEC_GEMINI336_VENDOR_ID" \
    -v product="$ORBBEC_GEMINI336_PRODUCT_ID" '
      BEGIN {
        vendor = tolower(vendor)
        product = tolower(product)
      }
      {
        line = tolower($0)
        if (line ~ "idvendor[^0-9a-f]*" vendor && line ~ "idproduct[^0-9a-f]*" product) {
          found = 1
        }
      }
      END {
        exit found ? 0 : 1
      }
    ' "$rules_file"
}

print_orbbec_udev_install_hint() {
  local source_rules="$1"
  echo "[ai-ov] Orbbec Gemini 336 USB udev rule is missing: $ORBBEC_UDEV_RULES_DEST" >&2
  echo "[ai-ov] The camera can still appear in lsusb, but pyorbbecsdk may fail with:" >&2
  echo "[ai-ov]   Access denied (insufficient permissions) / usbEnumerator openUsbDevice failed" >&2
  if [[ -n "$source_rules" && -f "$source_rules" ]]; then
    local install_script
    install_script="$(dirname "$source_rules")/install_udev_rules.sh"
    if [[ -f "$install_script" ]]; then
      echo "[ai-ov] Install the official Orbbec SDK rules:" >&2
      echo "[ai-ov]   sudo sh \"$install_script\"" >&2
    else
      echo "[ai-ov] Install the official Orbbec SDK rules:" >&2
      echo "[ai-ov]   sudo install -m 0644 \"$source_rules\" \"$ORBBEC_UDEV_RULES_DEST\"" >&2
      echo "[ai-ov]   sudo udevadm control --reload-rules && sudo udevadm trigger" >&2
    fi
  else
    echo "[ai-ov] Could not locate pyorbbecsdk's 99-obsensor-libusb.rules via: $ROS2_PYTHON" >&2
    echo "[ai-ov] Set AI_OV_ORBBEC_UDEV_RULES_SOURCE=/path/to/99-obsensor-libusb.rules if it is elsewhere." >&2
  fi
  echo "[ai-ov] Then unplug/replug the Gemini 336 camera and rerun this launcher." >&2
}

ensure_orbbec_udev_rules() {
  if udev_rule_file_contains_gemini336_rule "$ORBBEC_UDEV_RULES_DEST"; then
    return 0
  fi

  local source_rules
  source_rules="$(find_orbbec_udev_rules_source || true)"
  print_orbbec_udev_install_hint "$source_rules"
  exit 2
}

start_vision_service() {
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
  if [[ ! -x "$ROS2_PYTHON" ]]; then
    echo "[ai-ov] ROS2 workspace Python not found or not executable: $ROS2_PYTHON" >&2
    echo "[ai-ov] Set AI_OV_ROS2_PYTHON=/path/to/python to override." >&2
    exit 2
  fi

  ensure_orbbec_udev_rules

  if ((BUILD_VISION_SERVICE)); then
    echo "[ai-ov] building ROS2 camera package for vision snapshot service..."
    (
      cd "$ROS2_WORKSPACE"
      source_ros_setup "$ROS2_SETUP"
      with_ros2_pythonpath "$ROS2_PYTHON" -m colcon build --symlink-install --packages-select camera
    )
  fi

  source_ros_setup "$ROS2_SETUP"
  source_ros_setup "$ROS2_WORKSPACE/install/setup.bash"

  echo "[ai-ov] starting ROS2 min_dis vision snapshot service: $VISION_SERVICE_NAME"
  local ros2_run_cmd=(
    ros2 run camera min_dis --ros-args
    -p "vision_service_name:=$VISION_SERVICE_NAME"
    -p "vision_output_dir:=$VISION_OUTPUT_DIR"
    -p "show_window:=$VISION_SHOW_WINDOW"
    -p "cpu_threads:=$MIN_DIS_CPU_THREADS"
  )
  if [[ -n "$ROS2_PYTHONPATH_PREPEND" ]]; then
    OMP_NUM_THREADS="$MIN_DIS_CPU_THREADS" MKL_NUM_THREADS="$MIN_DIS_CPU_THREADS" \
      OPENBLAS_NUM_THREADS="$MIN_DIS_CPU_THREADS" NUMEXPR_NUM_THREADS="$MIN_DIS_CPU_THREADS" \
      PYTHONPATH="$ROS2_PYTHONPATH_PREPEND${PYTHONPATH:+:$PYTHONPATH}" \
      setsid systemd-run --user --scope --quiet --unit "ai-ov-min-dis-$$" \
      -p "CPUWeight=$MIN_DIS_CPU_WEIGHT" \
      taskset -c "$MIN_DIS_CPUS" \
      nice -n "$MIN_DIS_NICE_ADJUSTMENT" "${ros2_run_cmd[@]}" &
  else
    OMP_NUM_THREADS="$MIN_DIS_CPU_THREADS" MKL_NUM_THREADS="$MIN_DIS_CPU_THREADS" \
      OPENBLAS_NUM_THREADS="$MIN_DIS_CPU_THREADS" NUMEXPR_NUM_THREADS="$MIN_DIS_CPU_THREADS" \
      setsid systemd-run --user --scope --quiet --unit "ai-ov-min-dis-$$" \
      -p "CPUWeight=$MIN_DIS_CPU_WEIGHT" \
      taskset -c "$MIN_DIS_CPUS" \
      nice -n "$MIN_DIS_NICE_ADJUSTMENT" "${ros2_run_cmd[@]}" &
  fi
  VISION_PID=$!
  sleep 1
  if ! process_group_is_running "$VISION_PID"; then
    echo "[ai-ov] ROS2 vision snapshot service exited during startup." >&2
    exit 1
  fi
}

parse_args "$@"

validate_nice_value "$VOICE_NICE" AI_OV_VOICE_NICE
validate_nice_value "$MIN_DIS_NICE" AI_OV_MIN_DIS_NICE
validate_cpu_weight "$VOICE_CPU_WEIGHT" AI_OV_VOICE_CPU_WEIGHT
validate_cpu_weight "$MIN_DIS_CPU_WEIGHT" AI_OV_MIN_DIS_CPU_WEIGHT
validate_cpu_list "$MIN_DIS_CPUS" AI_OV_MIN_DIS_CPUS
validate_positive_integer "$MIN_DIS_CPU_THREADS" AI_OV_MIN_DIS_CPU_THREADS
ensure_user_cpu_weight_support
LAUNCHER_NICE="$(read_effective_nice)"
VOICE_NICE_ADJUSTMENT="$(nice_adjustment "$VOICE_NICE" "$LAUNCHER_NICE" AI_OV_VOICE_NICE)"
MIN_DIS_NICE_ADJUSTMENT="$(nice_adjustment "$MIN_DIS_NICE" "$LAUNCHER_NICE" AI_OV_MIN_DIS_NICE)"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ai-ov] Python runtime not found or not executable: $PYTHON_BIN" >&2
  echo "[ai-ov] Set AI_OV_PYTHON=/path/to/python if you want to use another environment." >&2
  exit 2
fi

if ! command -v setsid >/dev/null 2>&1; then
  echo "[ai-ov] setsid is required so the launcher can clean the full Web/MOSS process group." >&2
  exit 2
fi

trap cleanup EXIT INT TERM

if ((START_VISION_SERVICE)); then
  start_vision_service
fi

ensure_https_certificate

LAN_IP="$(resolve_lan_ipv4)"

echo "[ai-ov] starting Web UI without AI_ov ROS2 command publishing..."
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
if ((START_VISION_SERVICE)); then
  echo "[ai-ov] Vision snapshot service: $VISION_SERVICE_NAME"
  echo "[ai-ov] Vision output dir: $VISION_OUTPUT_DIR"
  echo "[ai-ov] Vision min_dis preview windows: $VISION_SHOW_WINDOW"
  echo "[ai-ov] Vision min_dis effective nice: $MIN_DIS_NICE"
  echo "[ai-ov] Vision min_dis CPU weight: $MIN_DIS_CPU_WEIGHT"
  echo "[ai-ov] Vision min_dis CPU affinity: $MIN_DIS_CPUS"
  echo "[ai-ov] Vision min_dis CPU threads: $MIN_DIS_CPU_THREADS"
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
  --dry-run-ros2 \
  --ssl-certfile "$SSL_CERTFILE" \
  --ssl-keyfile "$SSL_KEYFILE" \
  "${WEB_ARGS[@]}" &
WEB_PID=$!

# wait in a loop: if interrupted by a signal, the trap fires and cleanup runs;
# if wait returns because the child exited, we pick up its exit code normally.
while kill -0 "$WEB_PID" 2>/dev/null; do
  wait "$WEB_PID" || true
done
