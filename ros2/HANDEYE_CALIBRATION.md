# Hand-eye XY calibration maintenance procedure

This feature calibrates a planar eye-in-hand pixel-to-base XY mapping near the configured working height. It is not a full 6D hand-eye calibration and is never started by the production `arm.launch.py`.

## Safety model

- `handeye_calibration.launch.py` defaults to `motion_enabled:=false`.
- `start_handeye_calibration.sh` has no `--yes` bypass and requires the exact confirmation text `START-HANDEYE`.
- Calibration owns the IK command path while active. Ordinary `/goal`, ordinary `/control`, and new capture requests are rejected.
- Every calibration IK request carries a monotonically increasing `seq`; stale results are ignored.
- `/emergency_stop=true` aborts immediately, releases ownership, and never publishes an automatic return goal. Clearing the stop does not resume calibration.
- Normal cancellation/failure may request a return to the start pose, but return IK and arrival both have timeouts.
- The YAML artifact is written under `~/.ros/ai_ov/` with file and directory `fsync` plus atomic replace.

## Prerequisites

1. Review and commit/isolate the merge changes so `/home/inteldk/ROS2/src` is clean, then build and source `/home/inteldk/ROS2`.
2. Start the normal real-arm graph and confirm `/emergency_stop`, `/robot_joint_state`, `/is_capture`, and `/handeye/d_pixel` exist.
3. Confirm the source tree is clean, `/dev/ttyUSB0` is online, no capture is active, D is visible, the arm is unloaded, and the workspace is clear.
4. Keep the physical emergency stop in hand.

## Run

```bash
cd /home/inteldk/ROS2
./start_handeye_calibration.sh
```

The script performs the preflight checks, asks for `START-HANDEYE`, and starts the independent maintenance launch with motion explicitly enabled. Then start the state machine:

```bash
ros2 service call /handeye/start std_srvs/srv/Trigger '{}'
```

Monitor structured status:

```bash
ros2 topic echo /handeye/status std_msgs/msg/String
```

Cancel normally with:

```bash
ros2 service call /handeye/cancel std_srvs/srv/Trigger '{}'
```

## Review and deployment

The default newly fitted artifact is `~/.ros/ai_ov/handeye_xy.yaml`. Review `fit_rmse_m`, `condition_number`, sample counts, timestamp, URDF hash, workspace, and rejected samples before enabling runtime mapping.

Runtime mapping remains off by default. The installed `control/config/handeye_xy.yaml` is an exact parameter copy of the ws712 approved baseline and is selected when mapping is explicitly enabled without `AI_OV_HANDEYE_CALIBRATION_FILE`. Set `AI_OV_HANDEYE_CALIBRATION_FILE` to deploy a newly approved artifact instead. Enable mapping only after simulation, three-point low-speed real-arm validation, nine-point validation, and measured landing error approval. Set `AI_OV_HANDEYE_MAPPING_ENABLED=true` and optionally `AI_OV_HANDEYE_CALIBRATION_FILE` / `AI_OV_HANDEYE_MAX_AGE_DAYS` before starting the normal arm graph; equivalent `arm_state` ROS parameters are also supported. A missing, stale, malformed, low-quality, or out-of-workspace calibration fails closed; it does not silently fall back to fixed affine coefficients.

## Rollback

1. Set `handeye_mapping_enabled:=false` or remove the explicit override.
2. Stop the independent calibration launch.
3. Preserve the rejected artifact for diagnosis; do not overwrite the last approved artifact.
4. Rebuild/restart if source or package configuration changed.

Do not add hand-eye calibration to the default production `arm.launch.py`.
