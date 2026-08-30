# ROS 2 Arm Workspace

**中文（默认）:** [readme.md](readme.md)

This directory is the authoritative ROS 2 workspace for the integrated
`nl-industrial-arm-safety` monorepo. It contains the packages that connect the
natural-language AI layer to the camera, arm controller, simulation, and
operator GUI.

> **Module Maintainer**: [@nanshanbot](https://github.com/nanshanbot) (Responsible for ROS 2 node architecture, arm kinematics control, camera RGB-D distance safety & vision snapshots, multi-source e-stop arbitration, MuJoCo simulation & PyQt5 GUI)

## Packages

| Package | Responsibility |
| --- | --- |
| `camera` | Orbbec RGB/depth input, person/arm distance estimation, camera emergency-stop source, and vision snapshot Trigger service |
| `control` | IK, DRL, hand-eye calibration, and control helpers |
| `main` | Launch files, arm state machine, serial feedback, and emergency-stop aggregation |
| `arm_asset` | URDF/MJCF robot description and meshes |
| `mujoco_sim` | MuJoCo simulation node |
| `arm_gui` | PyQt5 ROS 2 status and manual stop interface |

## Build

From the monorepo root:

```bash
source /opt/ros/jazzy/setup.bash
cd ros2
colcon build
source install/setup.bash
```

Generated `build/`, `install/`, `log/`, and `.runtime/` directories are local
state and are intentionally ignored.

## Launch

The integrated launcher normally starts the default arm path from the
repository root:

```bash
./scripts/start_web_with_ros.sh
```

To launch the ROS graph directly:

```bash
source /opt/ros/jazzy/setup.bash
source /path/to/nl-industrial-arm-safety/ros2/install/setup.bash
ros2 launch main arm.launch.py
```

The default real-robot launch does not start MuJoCo simulation or the K230
camera node. Inspect the launch files before selecting simulation or hardware
paths on a new machine.

```bash
ros2 topic pub /goal geometry_msgs/msg/Point "{x: 0.08, y: 0.25, z: 0.2}" --once
```

## Vision Context Snapshot

`camera.min_dis` exposes an AI_ov-compatible Trigger service while keeping the
existing minimum-distance topics unchanged.

Run the camera node:

```bash
source /opt/ros/<distro>/setup.bash
source /path/to/nl-industrial-arm-safety/ros2/install/setup.bash
ros2 run camera min_dis --ros-args \
  -p vision_service_name:=/vision/capture_snapshot \
  -p vision_output_dir:=/path/to/nl-industrial-arm-safety/ros2/.runtime/vision_snapshots
```

Call the service:

```bash
ros2 service call /vision/capture_snapshot std_srvs/srv/Trigger {}
```

On success, `message` is JSON with:

```json
{
  "image_path": "/path/to/nl-industrial-arm-safety/ros2/.runtime/vision_snapshots/vision_context_123.jpg",
  "stamp": "2026-07-05T10:00:00.000Z",
  "frame_id": "camera_frame",
  "min_distance_m": 0.42,
  "human_closest_point": {"x": 0.1, "y": 0.2, "z": 0.7},
  "arm_closest_point": {"x": 0.3, "y": 0.2, "z": 0.6},
  "emergency_stop": false,
  "fresh": true,
  "age_ms": 12.3,
  "source": "min_dis"
}
```

If no completed frame analysis is available, or the cached frame/context is
older than `vision_max_age_ms` (default `1000`), the service returns
`success=false` with a short human-readable error. The image extension parameter
accepts `jpg`, `jpeg`, or `png`.

Use Git branches or patches for team handoff. Do not copy a zip over this
workspace; stage teammate code elsewhere, review the diff, then merge selected
changes.
