import launch
import launch_ros

def generate_launch_description():

    control_node = launch_ros.actions.Node(
        package='control',
        executable='drl_control',
        output='screen',
        prefix=['/home/robot/miniconda3/envs/mujoco_to_ros2/bin/python3'],
        )


    control_mujoco_node = launch_ros.actions.Node(
        package='mujoco_sim',
        executable='mujoco_sim',
        output='screen',
    )


    launch_description = launch.LaunchDescription([
        control_node,
        control_mujoco_node,
    ])

    return launch_description
