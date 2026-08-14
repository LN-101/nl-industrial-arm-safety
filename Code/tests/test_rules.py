from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_safety_assistant.rules import (
    RuleValidationError,
    apply_rule_patch,
    load_rule_document,
    replace_rule,
    validate_rule_document,
    write_rule_document,
)


class SafetyRulesTest(unittest.TestCase):
    def test_default_example_rule_document_covers_current_safety_aspects(self) -> None:
        document = load_rule_document(PROJECT_ROOT / "Code" / "config" / "safety_rules.example.json")

        self.assertEqual(len(document["rules"]), 3)
        self.assertEqual(
            [rule["id"] for rule in document["rules"]],
            [
                "stop_on_person_intrusion",
                "slow_near_unknown_object",
                "hold_on_ros_controller_alarm",
            ],
        )
        names = " ".join(str(rule.get("name", "")) for rule in document["rules"])
        for keyword in ("人员", "未知物体", "控制器"):
            self.assertIn(keyword, names)
        self.assertTrue(all(rule.get("enabled") is True for rule in document["rules"]))

    def test_validate_accepts_rule_document(self) -> None:
        document = {
            "version": 1,
            "rules": [
                {
                    "id": "stop_on_person_intrusion",
                    "enabled": True,
                    "action": {"type": "stop_motion"},
                }
            ],
        }

        validated = validate_rule_document(document)

        self.assertEqual(validated["version"], 1)
        self.assertEqual(validated["rules"][0]["id"], "stop_on_person_intrusion")

    def test_validate_rejects_duplicate_rule_ids(self) -> None:
        document = {
            "version": 1,
            "rules": [
                {"id": "duplicate", "enabled": True},
                {"id": "duplicate", "enabled": False},
            ],
        }

        with self.assertRaises(RuleValidationError):
            validate_rule_document(document)

    def test_replace_rule_writes_incremented_document(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rules_path = Path(temp_dir) / "rules.json"
            write_rule_document(
                rules_path,
                {
                    "version": 1,
                    "rules": [
                        {
                            "id": "slow_near_unknown_object",
                            "enabled": True,
                            "action": {"type": "limit_speed", "max_speed_scale": 0.25},
                        }
                    ],
                },
            )

            replace_rule(
                rules_path,
                {
                    "id": "slow_near_unknown_object",
                    "enabled": False,
                    "action": {"type": "limit_speed", "max_speed_scale": 0.1},
                },
            )

            updated = load_rule_document(rules_path)
            self.assertEqual(updated["version"], 2)
            self.assertFalse(updated["rules"][0]["enabled"])
            self.assertEqual(updated["rules"][0]["action"]["max_speed_scale"], 0.1)

    def test_apply_rule_patch_updates_existing_scalar_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rules_path = Path(temp_dir) / "rules.json"
            write_rule_document(
                rules_path,
                {
                    "version": 1,
                    "rules": [
                        {
                            "id": "slow_near_unknown_object",
                            "enabled": True,
                            "conditions": {"unknown_object_distance_m": {"lt": 0.25}},
                            "action": {"type": "limit_speed", "max_speed_scale": 0.25},
                        }
                    ],
                },
            )

            apply_rule_patch(
                rules_path,
                {
                    "rule_id": "slow_near_unknown_object",
                    "changes": {
                        "enabled": False,
                        "conditions.unknown_object_distance_m.lt": 0.3,
                        "action.max_speed_scale": 0.1,
                    },
                },
            )

            updated = load_rule_document(rules_path)
            rule = updated["rules"][0]
            self.assertEqual(updated["version"], 2)
            self.assertFalse(rule["enabled"])
            self.assertEqual(rule["conditions"]["unknown_object_distance_m"]["lt"], 0.3)
            self.assertEqual(rule["action"]["max_speed_scale"], 0.1)
            self.assertEqual(rule["action"]["type"], "limit_speed")

    def test_apply_rule_patch_same_value_keeps_version_and_file_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rules_path = Path(temp_dir) / "rules.json"
            write_rule_document(
                rules_path,
                {
                    "version": 1,
                    "rules": [
                        {
                            "id": "slow_near_unknown_object",
                            "enabled": True,
                            "action": {"type": "limit_speed", "max_speed_scale": 0.25},
                        }
                    ],
                },
            )
            before_text = rules_path.read_text(encoding="utf-8")

            updated = apply_rule_patch(
                rules_path,
                {
                    "rule_id": "slow_near_unknown_object",
                    "changes": {
                        "enabled": True,
                        "action.max_speed_scale": 0.25,
                    },
                },
            )

            self.assertEqual(updated["version"], 1)
            self.assertEqual(rules_path.read_text(encoding="utf-8"), before_text)

    def test_apply_rule_patch_accepts_personnel_distance_and_temporary_disable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rules_path = Path(temp_dir) / "rules.json"
            write_rule_document(
                rules_path,
                {
                    "version": 1,
                    "rules": [
                        {
                            "id": "stop_on_person_intrusion",
                            "enabled": True,
                            "conditions": {"person_distance_m": {"lt": 1.0}},
                            "action": {"type": "stop_motion"},
                        }
                    ],
                },
            )

            apply_rule_patch(
                rules_path,
                {
                    "rule_id": "stop_on_person_intrusion",
                    "changes": {
                        "conditions.person_distance_m.lt": 1.2,
                        "enabled": False,
                    },
                },
            )

            updated = load_rule_document(rules_path)
            rule = updated["rules"][0]
            self.assertEqual(updated["version"], 2)
            self.assertEqual(rule["conditions"]["person_distance_m"]["lt"], 1.2)
            self.assertFalse(rule["enabled"])

    def test_apply_rule_patch_rejects_out_of_range_personnel_distance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rules_path = Path(temp_dir) / "rules.json"
            write_rule_document(
                rules_path,
                {
                    "version": 1,
                    "rules": [
                        {
                            "id": "stop_on_person_intrusion",
                            "enabled": True,
                            "conditions": {"person_distance_m": {"lt": 1.0}},
                            "action": {"type": "stop_motion"},
                        }
                    ],
                },
            )

            for value in (0.0, 0.05, 10000.0):
                with self.subTest(value=value):
                    with self.assertRaises(RuleValidationError):
                        apply_rule_patch(
                            rules_path,
                            {
                                "rule_id": "stop_on_person_intrusion",
                                "changes": {"conditions.person_distance_m.lt": value},
                            },
                        )

            updated = load_rule_document(rules_path)
            self.assertEqual(updated["version"], 1)
            self.assertEqual(updated["rules"][0]["conditions"]["person_distance_m"]["lt"], 1.0)

    def test_validate_rejects_invalid_personnel_distance_in_document(self) -> None:
        document = {
            "version": 1,
            "rules": [
                {
                    "id": "stop_on_person_intrusion",
                    "enabled": True,
                    "conditions": {"person_distance_m": {"lt": 10000.0}},
                }
            ],
        }

        with self.assertRaises(RuleValidationError):
            validate_rule_document(document)

    def test_apply_rule_patch_rejects_missing_rule_or_field(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rules_path = Path(temp_dir) / "rules.json"
            write_rule_document(
                rules_path,
                {
                    "version": 1,
                    "rules": [{"id": "slow_near_unknown_object", "enabled": True}],
                },
            )

            with self.assertRaises(RuleValidationError):
                apply_rule_patch(rules_path, {"rule_id": "missing", "changes": {"enabled": False}})
            with self.assertRaises(RuleValidationError):
                apply_rule_patch(
                    rules_path,
                    {"rule_id": "slow_near_unknown_object", "changes": {"action.max_speed_scale": 0.1}},
                )

            self.assertEqual(load_rule_document(rules_path)["version"], 1)

    def test_apply_rule_patch_rejects_forbidden_or_structural_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rules_path = Path(temp_dir) / "rules.json"
            write_rule_document(
                rules_path,
                {
                    "version": 1,
                    "rules": [
                        {
                            "id": "slow_near_unknown_object",
                            "enabled": True,
                            "name": "未知物体靠近限速规则",
                            "description": "降低速度。",
                            "action": {"type": "limit_speed", "max_speed_scale": 0.25},
                        }
                    ],
                },
            )
            rejected_changes = (
                {"id": "other"},
                {"name": "新名字"},
                {"description": "新描述"},
                {"action.type": "stop_motion"},
                {"action . type": "stop_motion"},
                {"action": {"max_speed_scale": 0.1}},
                {"action": {"type": "limit_speed", "max_speed_scale": 0.1}},
            )

            for changes in rejected_changes:
                with self.subTest(changes=changes):
                    with self.assertRaises(RuleValidationError):
                        apply_rule_patch(
                            rules_path,
                            {"rule_id": "slow_near_unknown_object", "changes": changes},
                        )

            updated = load_rule_document(rules_path)
            self.assertEqual(updated["version"], 1)
            self.assertEqual(updated["rules"][0]["action"]["max_speed_scale"], 0.25)

    def test_apply_rule_patch_rejects_type_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rules_path = Path(temp_dir) / "rules.json"
            write_rule_document(
                rules_path,
                {
                    "version": 1,
                    "rules": [
                        {
                            "id": "slow_near_unknown_object",
                            "enabled": True,
                            "action": {"type": "limit_speed", "max_speed_scale": 0.25},
                        }
                    ],
                },
            )

            for changes in (
                {"enabled": "false"},
                {"action.max_speed_scale": False},
                {"action.max_speed_scale": "0.1"},
            ):
                with self.subTest(changes=changes):
                    with self.assertRaises(RuleValidationError):
                        apply_rule_patch(
                            rules_path,
                            {"rule_id": "slow_near_unknown_object", "changes": changes},
                        )

            self.assertEqual(load_rule_document(rules_path)["version"], 1)


if __name__ == "__main__":
    unittest.main()
