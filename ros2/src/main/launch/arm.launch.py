import os

import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
import launch_ros
from launch_ros.parameter_descriptions import ParameterValue


MIN_DIS_CPU_THREADS = '3'


def get_cmeel_library_path():
    candidates = []
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        candidates.append(os.path.join(conda_prefix, 'lib'))
        candidates.append(
            os.path.join(conda_prefix, 'lib', 'python3.10', 'site-packages', 'cmeel.prefix', 'lib')
        )
    candidates.append('/home/robot/.local/lib/python3.10/site-packages/cmeel.prefix/lib')

    for path in candidates:
        if os.path.isdir(path):
            return path

    return candidates[-1]


def generate_launch_description():
    cmeel_lib_path = get_cmeel_library_path()
    show_window = LaunchConfiguration('show_window')
    min_dis_nice = LaunchConfiguration('min_dis_nice')
    min_dis_cpu_weight = LaunchConfiguration('min_dis_cpu_weight')
    min_dis_cpus = LaunchConfiguration('min_dis_cpus')
    min_dis_cpu_threads = LaunchConfiguration('min_dis_cpu_threads')
    show_window_argument = DeclareLaunchArgument(
        'show_window',
        default_value='true',
        description='Show the min_dis OpenCV safety detection window.',
    )
    min_dis_nice_argument = DeclareLaunchArgument(
        'min_dis_nice',
        default_value='10',
        choices=[str(value) for value in range(20)],
        description='Effective nice value for min_dis vision inference.',
    )
    min_dis_cpu_weight_argument = DeclareLaunchArgument(
        'min_dis_cpu_weight',
        default_value='25',
        description=(
            'cgroup v2 CPUWeight (1..10000) for the min_dis vision scope. '
            'Lower than the voice weight so streaming TTS wins the CPU under '
            'contention; nice alone is neutralized by kernel autogrouping '
            'across separate launcher sessions.'
        ),
    )
    min_dis_cpus_argument = DeclareLaunchArgument(
        'min_dis_cpus',
        default_value='4-6',
        description=(
            'taskset CPU list confining min_dis vision inference. Defaults to '
            'E-cores 4-6, leaving P-cores 0-3 free for the voice stack. On this '
            '15W hybrid CPU, CPUWeight alone cannot stop vision from lighting up '
            'every core and throttling the whole package frequency (measured P-core '
            '2000->1600 MHz under E-core load), which starves streaming TTS; '
            'pinning vision off the P-cores preserves their turbo headroom.'
        ),
    )
    min_dis_cpu_threads_argument = DeclareLaunchArgument(
        'min_dis_cpu_threads',
        default_value=MIN_DIS_CPU_THREADS,
        description='Positive PyTorch/BLAS CPU thread count for min_dis inference.',
    )

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

    # control_mujoco_node = launch_ros.actions.Node(
    #     package='mujoco_sim',
    #     executable='mujoco_sim',
    #     output='screen',
    # )

    arm_state_node = launch_ros.actions.Node(
        package='main',
        executable='arm_state',
        output='screen',
    )
    estop_aggregator_node = launch_ros.actions.Node(
        package='main',
        executable='estop_aggregator',
        output='screen',
    )
    min_dis_node = launch_ros.actions.Node(
        package='camera',
        executable='min_dis',
        output='screen',
        prefix=[
            'systemd-run --user --scope --quiet -p CPUWeight=',
            min_dis_cpu_weight,
            ' taskset -c ',
            min_dis_cpus,
            ' nice -n ',
            min_dis_nice,
        ],
        parameters=[{
            'cpu_threads': ParameterValue(min_dis_cpu_threads, value_type=int),
            'show_window': ParameterValue(show_window, value_type=bool),
        }],
        additional_env={
            'OMP_NUM_THREADS': min_dis_cpu_threads,
            'MKL_NUM_THREADS': min_dis_cpu_threads,
            'OPENBLAS_NUM_THREADS': min_dis_cpu_threads,
            'NUMEXPR_NUM_THREADS': min_dis_cpu_threads,
        },
    )
    # k230_node = launch_ros.actions.Node(
    #     package='camera',
    #     executable='k230',
    #     output='screen',
    # )
    launch_description = launch.LaunchDescription([
        show_window_argument,
        min_dis_nice_argument,
        min_dis_cpu_weight_argument,
        min_dis_cpus_argument,
        min_dis_cpu_threads_argument,
        control_node,
        # control_mujoco_node,
        estop_aggregator_node,
        arm_state_node,
        # k230_node,
        min_dis_node,
    ])

    return launch_description
