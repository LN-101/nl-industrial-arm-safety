from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from local_safety_assistant.arm_rules import (
    ArmRulesValidationError,
    load_arm_rule_document,
    request_arm_deceleration,
    request_arm_estop,
    request_object_grasp,
    sync_personnel_distance_to_arm_rules,
    write_arm_rule_document,
)
from local_safety_assistant.rules import RuleValidationError


class ArmRulesTest(unittest.TestCase):
    def test_request_object_grasp_sets_capture_goal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "arm_rules.json"
            write_arm_rule_document(
                path,
                {
                    "arm_capture": False,
                    "arm_capture_goal": "b",
                    "arm_decelerate": 0.5,
                    "arm_stop": "False",
                    "arm_recover": "False",
                    "arm_safety_distance": 0.3,
                },
            )

            request_object_grasp(
                path,
                marker="a",
                object_name="red block",
                original_text="please grasp marker a",
            )
            document = load_arm_rule_document(path)

            self.assertEqual(document["arm_capture"], "True")
            self.assertEqual(document["arm_capture_goal"], "A")
            self.assertEqual(document["arm_capture_object"], "red block")
            self.assertEqual(
                document["arm_capture_original_text"],
                "please grasp marker a",
            )
            self.assertEqual(document["arm_decelerate"], "0.5")
            self.assertEqual(document["arm_safety_distance"], "0.3")

    def test_missing_file_request_creates_default_document(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "arm_rules.json"

            request_object_grasp(path, marker="D")
            document = load_arm_rule_document(path)

            self.assertEqual(document["arm_capture"], "True")
            self.assertEqual(document["arm_capture_goal"], "D")
            self.assertEqual(document["arm_stop"], "False")

    def test_invalid_goal_does_not_overwrite_existing_document(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "arm_rules.json"
            write_arm_rule_document(path, {"arm_capture_goal": "C"})

            with self.assertRaises(ArmRulesValidationError):
                request_object_grasp(path, marker="E")

            document = load_arm_rule_document(path)
            self.assertEqual(document["arm_capture"], "False")
            self.assertEqual(document["arm_capture_goal"], "C")

    def test_request_arm_deceleration_sets_fraction_and_preserves_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "arm_rules.json"
            write_arm_rule_document(
                path,
                {
                    "arm_capture": "True",
                    "arm_capture_goal": "B",
                    "arm_stop": "True",
                    "arm_recover": "False",
                    "arm_safety_distance": "0.4",
                    "arm_capture_object": "扳手",
                },
            )

            result = request_arm_deceleration(path, target_speed_percent=30)
            document = load_arm_rule_document(path)

            self.assertEqual(result.target_speed_percent, 30.0)
            self.assertEqual(result.arm_decelerate, "0.3")
            self.assertEqual(document["arm_decelerate"], "0.3")
            self.assertEqual(document["arm_capture"], "True")
            self.assertEqual(document["arm_capture_goal"], "B")
            self.assertEqual(document["arm_stop"], "True")
            self.assertEqual(document["arm_safety_distance"], "0.4")
            self.assertEqual(document["arm_capture_object"], "扳手")

    def test_request_arm_deceleration_rejects_out_of_range_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "arm_rules.json"
            write_arm_rule_document(path, {"arm_decelerate": "0.5"})

            with self.assertRaises(ArmRulesValidationError):
                request_arm_deceleration(path, target_speed_percent=130)

            document = load_arm_rule_document(path)
            self.assertEqual(document["arm_decelerate"], "0.5")

    def test_request_arm_estop_sets_stop_and_preserves_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "arm_rules.json"
            write_arm_rule_document(
                path,
                {
                    "arm_capture": "True",
                    "arm_capture_goal": "C",
                    "arm_decelerate": "0.5",
                    "arm_stop": "False",
                    "arm_recover": "True",
                    "arm_safety_distance": "0.4",
                    "arm_capture_object": "扳手",
                },
            )

            result = request_arm_estop(path, active=True)
            document = load_arm_rule_document(path)

            self.assertTrue(result.active)
            self.assertEqual(result.arm_stop, "True")
            self.assertEqual(result.arm_recover, "False")
            self.assertEqual(document["arm_stop"], "True")
            self.assertEqual(document["arm_recover"], "False")
            self.assertEqual(document["arm_capture"], "True")
            self.assertEqual(document["arm_capture_goal"], "C")
            self.assertEqual(document["arm_decelerate"], "0.5")
            self.assertEqual(document["arm_safety_distance"], "0.4")
            self.assertEqual(document["arm_capture_object"], "扳手")

    def test_request_arm_estop_release_sets_recover_and_preserves_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "arm_rules.json"
            write_arm_rule_document(
                path,
                {
                    "arm_capture": "True",
                    "arm_capture_goal": "B",
                    "arm_decelerate": "0.3",
                    "arm_stop": "True",
                    "arm_recover": "False",
                    "arm_safety_distance": "0.2",
                },
            )

            result = request_arm_estop(path, active=False)
            document = load_arm_rule_document(path)

            self.assertFalse(result.active)
            self.assertEqual(result.arm_stop, "False")
            self.assertEqual(result.arm_recover, "True")
            self.assertEqual(document["arm_stop"], "False")
            self.assertEqual(document["arm_recover"], "True")
            self.assertEqual(document["arm_capture"], "True")
            self.assertEqual(document["arm_capture_goal"], "B")
            self.assertEqual(document["arm_decelerate"], "0.3")

    def test_sync_personnel_distance_updates_only_safety_distance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "arm_rules.json"
            write_arm_rule_document(
                path,
                {
                    "arm_capture": "True",
                    "arm_capture_goal": "B",
                    "arm_decelerate": "0.5",
                    "arm_stop": "False",
                    "arm_recover": "True",
                    "arm_safety_distance": "0.2",
                },
            )

            result = sync_personnel_distance_to_arm_rules(
                _personnel_rule_document(distance=0.2),
                _personnel_rule_document(distance=0.4),
                path,
            )

            document = load_arm_rule_document(path)
            self.assertTrue(result.synced)
            self.assertEqual(result.distance_m, 0.4)
            self.assertEqual(document["arm_safety_distance"], "0.4")
            self.assertEqual(document["arm_capture"], "True")
            self.assertEqual(document["arm_capture_goal"], "B")
            self.assertEqual(document["arm_decelerate"], "0.5")
            self.assertEqual(document["arm_recover"], "True")

    def test_sync_personnel_distance_no_change_does_not_create_arm_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "arm_rules.json"

            result = sync_personnel_distance_to_arm_rules(
                _personnel_rule_document(distance=0.2),
                _personnel_rule_document(distance=0.2),
                path,
            )

            self.assertFalse(result.synced)
            self.assertFalse(path.exists())

    def test_sync_personnel_distance_disabled_rule_keeps_arm_file_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "arm_rules.json"
            write_arm_rule_document(path, {"arm_safety_distance": "0.2"})

            with self.assertRaises(RuleValidationError):
                sync_personnel_distance_to_arm_rules(
                    _personnel_rule_document(distance=0.2),
                    _personnel_rule_document(distance=0.4, enabled=False),
                    path,
                )

            document = load_arm_rule_document(path)
            self.assertEqual(document["arm_safety_distance"], "0.2")


def _personnel_rule_document(*, distance: float, enabled: bool = True) -> dict:
    return {
        "version": 1,
        "rules": [
            {
                "id": "stop_on_person_intrusion",
                "enabled": enabled,
                "conditions": {"person_distance_m": {"lt": distance}},
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
