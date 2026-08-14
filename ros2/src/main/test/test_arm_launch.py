import importlib.util
from pathlib import Path
import shlex

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.utilities import perform_substitutions
from launch_ros.actions import Node
import pytest


LAUNCH_FILE = Path(__file__).resolve().parents[1] / 'launch' / 'arm.launch.py'


def load_launch_module():
    spec = importlib.util.spec_from_file_location('arm_launch', LAUNCH_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_min_dis_has_nice_prefix():
    description = load_launch_module().generate_launch_description()
    arguments = {
        action.name: action
        for action in description.entities
        if isinstance(action, DeclareLaunchArgument)
    }
    nodes = [action for action in description.entities if isinstance(action, Node)]
    prefixed_nodes = [
        node
        for node in nodes
        if len(node.process_description._Executable__prefix) > 1
    ]

    assert arguments['min_dis_nice'].default_value[0].text == '10'
    assert arguments['min_dis_nice'].choices == [str(value) for value in range(20)]
    assert arguments['min_dis_cpu_weight'].default_value[0].text == '25'
    assert arguments['min_dis_cpus'].default_value[0].text == '4-6'
    assert arguments['min_dis_cpu_threads'].default_value[0].text == '3'
    assert len(prefixed_nodes) == 1
    assert prefixed_nodes[0]._Node__node_executable == 'min_dis'
    assert {
        node._Node__node_executable
        for node in nodes
        if len(node.process_description._Executable__prefix) == 1
    } == {'ik_control', 'arm_state', 'estop_aggregator'}
    prefix = prefixed_nodes[0].process_description._Executable__prefix
    assert prefix[0].text == 'systemd-run --user --scope --quiet -p CPUWeight='
    assert prefix[1].variable_name[0].text == 'min_dis_cpu_weight'
    assert prefix[2].text == ' taskset -c '
    assert prefix[3].variable_name[0].text == 'min_dis_cpus'
    assert prefix[4].text == ' nice -n '
    assert prefix[5].variable_name[0].text == 'min_dis_nice'

    context = LaunchContext()
    context.launch_configurations['min_dis_nice'] = '19'
    context.launch_configurations['min_dis_cpu_weight'] = '500'
    context.launch_configurations['min_dis_cpus'] = '0-3'
    assert perform_substitutions(context, [prefix[1]]) == '500'
    assert perform_substitutions(context, [prefix[3]]) == '0-3'
    assert perform_substitutions(context, [prefix[5]]) == '19'

    invalid_context = LaunchContext()
    invalid_context.launch_configurations['min_dis_nice'] = '20'
    with pytest.raises(RuntimeError, match='provided value "20" is not valid'):
        arguments['min_dis_nice'].execute(invalid_context)


def test_min_dis_nice_prefix_expands_to_separate_arguments():
    description = load_launch_module().generate_launch_description()
    context = LaunchContext()
    for action in description.entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    min_dis_node = next(
        action
        for action in description.entities
        if isinstance(action, Node) and action._Node__node_executable == 'min_dis'
    )

    prefix = min_dis_node.process_description._Executable__prefix
    expanded_prefix = perform_substitutions(context, prefix)

    assert shlex.split(expanded_prefix) == [
        'systemd-run', '--user', '--scope', '--quiet',
        '-p', 'CPUWeight=25',
        'taskset', '-c', '4-6',
        'nice', '-n', '10',
    ]
