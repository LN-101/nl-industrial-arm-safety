import launch
import launch_ros
from launch.substitutions import EnvironmentVariable

def generate_launch_description():
    cmeel_lib_path = '/home/robot/.local/lib/python3.10/site-packages/cmeel.prefix/lib'

    control_node = launch_ros.actions.Node(
        package='control',
        executable='ik_control',
        output='screen',
        additional_env={
            'LD_LIBRARY_PATH': [
                cmeel_lib_path,
                ':',
                EnvironmentVariable('LD_LIBRARY_PATH', default_value=''),
            ],
        },
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
