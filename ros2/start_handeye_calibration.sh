#!/usr/bin/env bash
set -euo pipefail

workspace="${AI_OV_ROS2_WORKSPACE:-/home/inteldk/ROS2}"
ros_setup="${AI_OV_ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
workspace_setup="${AI_OV_ROS2_SETUP:-${workspace}/install/setup.bash}"
output_file="${AI_OV_HANDEYE_OUTPUT:-${HOME}/.ros/ai_ov/handeye_xy.yaml}"

if [[ $# -ne 0 ]]; then
    echo "Usage: $0" >&2
    echo "This launcher intentionally has no --yes or non-interactive bypass." >&2
    exit 2
fi

for setup_file in "$ros_setup" "$workspace_setup"; do
    if [[ ! -f "$setup_file" ]]; then
        echo "Missing ROS setup: $setup_file" >&2
        exit 2
    fi
done

dirty_paths="$(git -C "$workspace" status --porcelain --untracked-files=all -- src)"
if [[ -n "$dirty_paths" ]]; then
    echo "ROS2 source tree is dirty; commit or isolate changes before calibration:" >&2
    echo "$dirty_paths" >&2
    exit 2
fi

source "$ros_setup"
source "$workspace_setup"

if [[ ! -e /dev/ttyUSB0 ]]; then
    echo "Serial device /dev/ttyUSB0 is unavailable." >&2
    exit 2
fi

topic_list="$(ros2 topic list)"
for required_topic in \
    /emergency_stop \
    /handeye/control \
    /handeye/d_pixel \
    /handeye/ik_result \
    /is_capture \
    /robot_joint_state; do
    if ! grep -Fxq "$required_topic" <<<"$topic_list"; then
        echo "Required topic is unavailable: $required_topic" >&2
        exit 2
    fi
done

estop_value="$(timeout 3 ros2 topic echo --once /emergency_stop std_msgs/msg/Bool 2>/dev/null || true)"
if ! grep -Fq 'data: false' <<<"$estop_value"; then
    echo "Emergency stop is active or unreadable; calibration refused." >&2
    exit 2
fi

capture_value="$(timeout 3 ros2 topic echo --once /is_capture std_msgs/msg/Bool 2>/dev/null || true)"
if ! grep -Fq 'data: false' <<<"$capture_value"; then
    echo "Capture flow is active or unreadable; calibration refused." >&2
    exit 2
fi

if ! timeout 3 ros2 topic echo --once /handeye/d_pixel geometry_msgs/msg/PointStamped >/dev/null; then
    echo "No fresh D pixel received within 3 seconds." >&2
    exit 2
fi

echo "Maintenance calibration will move the real arm through 9 points."
echo "Keep the emergency stop in hand and remove payloads/obstacles."
read -r -p "Type START-HANDEYE to continue: " confirmation
if [[ "$confirmation" != "START-HANDEYE" ]]; then
    echo "Calibration cancelled." >&2
    exit 2
fi

exec ros2 launch main handeye_calibration.launch.py \
    motion_enabled:=true \
    output_file:="$output_file"
