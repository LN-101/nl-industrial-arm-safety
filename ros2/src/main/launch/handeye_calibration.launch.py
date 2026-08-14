from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    motion_enabled = LaunchConfiguration('motion_enabled')
    output_file = LaunchConfiguration('output_file')
    return LaunchDescription([
        DeclareLaunchArgument(
            'motion_enabled',
            default_value='false',
            choices=['true', 'false'],
            description='Explicit maintenance-mode gate for arm motion.',
        ),
        DeclareLaunchArgument(
            'output_file',
            default_value='~/.ros/ai_ov/handeye_xy.yaml',
            description='Runtime calibration artifact; never a source-tree path.',
        ),
        Node(
            package='control',
            executable='handeye_calibration',
            name='handeye_calibration',
            output='screen',
            parameters=[{
                'motion_enabled': motion_enabled,
                'output_file': output_file,
            }],
        ),
    ])
