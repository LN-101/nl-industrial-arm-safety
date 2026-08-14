from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node


LAUNCH_PATH = Path(__file__).parents[1] / 'launch' / 'handeye_calibration.launch.py'


class HandeyeLaunchTest(unittest.TestCase):
    def test_default_is_motion_disabled_and_separate_node(self):
        spec = importlib.util.spec_from_file_location('handeye_launch', LAUNCH_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        entities = module.generate_launch_description().entities
        arguments = {
            entity.name: entity
            for entity in entities
            if isinstance(entity, DeclareLaunchArgument)
        }
        nodes = [entity for entity in entities if isinstance(entity, Node)]

        self.assertEqual(arguments['motion_enabled'].default_value[0].text, 'false')
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].node_package, 'control')
        self.assertEqual(nodes[0].node_executable, 'handeye_calibration')


if __name__ == '__main__':
    unittest.main()
