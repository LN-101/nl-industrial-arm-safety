#!/usr/bin/env bash

validate_nice_value() {
  local value="$1"
  local setting_name="$2"
  if [[ ! "$value" =~ ^([0-9]|1[0-9])$ ]]; then
    echo "[ai-ov] $setting_name must be an integer from 0 through 19; got: $value" >&2
    return 2
  fi
}

read_effective_nice() {
  local pid="${1:-$$}"
  local value
  value="$(ps -o ni= -p "$pid" 2>/dev/null)" || {
    echo "[ai-ov] could not read effective nice value for launcher PID $pid." >&2
    return 2
  }
  value="${value//[[:space:]]/}"
  if [[ ! "$value" =~ ^-?[0-9]+$ ]] || ((value < -20 || value > 19)); then
    echo "[ai-ov] launcher PID $pid reported an invalid effective nice value: $value" >&2
    return 2
  fi
  printf '%s\n' "$value"
}

nice_adjustment() {
  local requested="$1"
  local inherited="$2"
  local setting_name="$3"
  validate_nice_value "$requested" "$setting_name" || return
  if ((requested < inherited)); then
    echo "[ai-ov] $setting_name=$requested is unattainable: launcher inherited effective nice $inherited." >&2
    echo "[ai-ov] An unprivileged launcher cannot raise its priority; start from nice $requested or better." >&2
    return 2
  fi
  printf '%s\n' "$((requested - inherited))"
}

validate_cpu_weight() {
  local value="$1"
  local setting_name="$2"
  # Match the 1..10000 range purely in the regex: bash (( )) arithmetic reads
  # leading zeros as octal ("08" raises an arithmetic error that evaluates as
  # a passing condition) and huge digit strings overflow intmax.
  if [[ ! "$value" =~ ^([1-9][0-9]{0,3}|10000)$ ]]; then
    echo "[ai-ov] $setting_name must be an integer from 1 through 10000; got: $value" >&2
    return 2
  fi
}

# Validate a taskset -c CPU list such as "4-9", "0,2,4" or "4-7,10". Rejects
# empty strings and stray characters so a bad AI_OV_..._CPUS cannot be forwarded
# into the launch prefix, where taskset would fail only after the scope exists.
validate_cpu_list() {
  local value="$1"
  local setting_name="$2"
  if [[ ! "$value" =~ ^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$ ]]; then
    echo "[ai-ov] $setting_name must be a taskset CPU list like 4-9 or 0,2,4; got: $value" >&2
    return 2
  fi
}

validate_positive_integer() {
  local value="$1"
  local setting_name="$2"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ai-ov] $setting_name must be a positive integer; got: $value" >&2
    return 2
  fi
}

# Verify the current user session can apply cgroup v2 CPUWeight via
# `systemd-run --user --scope`. Requires an active user systemd manager with the
# cpu controller delegated to user@<uid>.service; otherwise CPUWeight is silently
# ignored, so fail loudly before starting any runtime process.
ensure_user_cpu_weight_support() {
  local uid controllers_file
  if ! command -v systemd-run >/dev/null 2>&1; then
    echo "[ai-ov] systemd-run is required to apply cgroup CPUWeight scopes." >&2
    return 2
  fi
  if ! systemctl --user is-active default.target >/dev/null 2>&1; then
    echo "[ai-ov] No active user systemd session; cgroup CPUWeight scopes are unavailable." >&2
    echo "[ai-ov] Log in through a normal desktop/SSH session so user@$(id -u).service is running." >&2
    return 2
  fi
  uid="$(id -u)"
  controllers_file="/sys/fs/cgroup/user.slice/user-$uid.slice/user@$uid.service/cgroup.controllers"
  if ! grep -qw cpu "$controllers_file" 2>/dev/null; then
    echo "[ai-ov] The cpu cgroup controller is not delegated to user@$uid.service; CPUWeight would be ignored." >&2
    echo "[ai-ov] Checked: $controllers_file" >&2
    return 2
  fi
}
