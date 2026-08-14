from __future__ import annotations

import time
import unittest

from local_safety_assistant.confirmation import (
    ACTION_ESTOP_RELEASE,
    ACTION_ESTOP_TRIGGER,
    ACTION_OBJECT_GRASP_EXECUTION,
    ACTION_RULE_EDIT,
    ACTION_SPEED_CHANGE,
    build_estop_release_confirmation,
    build_object_grasp_execution_confirmation,
    build_object_mapping_update_confirmation,
    build_rule_edit_confirmation,
    build_speed_change_confirmation,
    interpret_spoken_confirmation,
    requires_confirmation,
)


class ConfirmationPolicyTest(unittest.TestCase):
    def test_policy_classifies_estop_trigger_as_immediate_but_release_as_confirmed(self) -> None:
        self.assertFalse(requires_confirmation(ACTION_ESTOP_TRIGGER))
        self.assertTrue(requires_confirmation(ACTION_ESTOP_RELEASE))

    def test_policy_classifies_future_grasp_and_speed_changes_as_confirmed(self) -> None:
        self.assertTrue(requires_confirmation(ACTION_OBJECT_GRASP_EXECUTION))
        self.assertTrue(requires_confirmation(ACTION_SPEED_CHANGE))
        grasp = build_object_grasp_execution_confirmation("抓取标号A", marker="A", object_name="红色方块")
        speed = build_speed_change_confirmation(
            "减速到30%",
            target_speed="30%",
            target_speed_percent=30,
            arm_decelerate="0.3",
            arm_rules_path="/tmp/arm_rules.json",
        )

        self.assertEqual(grasp.action_type, ACTION_OBJECT_GRASP_EXECUTION)
        self.assertEqual(speed.action_type, ACTION_SPEED_CHANGE)
        self.assertIn("标号A", grasp.summary)
        self.assertIn("红色方块", grasp.summary)
        self.assertIn("抓取", grasp.prompt)
        self.assertIn("标号A", grasp.prompt)
        self.assertIn("红色方块", grasp.prompt)
        self.assertIn("气泵", grasp.prompt)
        self.assertIn("标号A", "；".join(grasp.checklist))
        self.assertIn("红色方块", "；".join(grasp.checklist))
        self.assertIn("速度", speed.summary)
        self.assertIn("30%", speed.summary)
        self.assertEqual(speed.details["target_speed_percent"], 30)
        self.assertEqual(speed.details["arm_decelerate"], "0.3")
        self.assertEqual(speed.details["arm_rules_path"], "/tmp/arm_rules.json")

    def test_spoken_confirmation_rejects_generic_confirmation_for_pending_action(self) -> None:
        pending = build_estop_release_confirmation("解除急停")
        rule_pending = build_rule_edit_confirmation(
            "禁用人员侵入规则",
            rule_id="stop_on_person_intrusion",
            rule_name="人员侵入停机规则",
            patch={"rule_id": "stop_on_person_intrusion", "changes": {"enabled": False}},
            rules_path="/tmp/rules.json",
        )

        self.assertEqual(interpret_spoken_confirmation("确认", pending), "ambiguous")
        self.assertEqual(interpret_spoken_confirmation("确认解除急停", pending), "confirm")
        self.assertEqual(interpret_spoken_confirmation("取消", pending), "cancel")
        self.assertEqual(interpret_spoken_confirmation("确认修改", rule_pending), "ambiguous")
        self.assertEqual(interpret_spoken_confirmation("确认修改规则", rule_pending), "confirm")

    def test_spoken_confirmation_accepts_action_specific_mapping_update_confirmation(self) -> None:
        pending = build_object_mapping_update_confirmation(
            "把标号A的映射改成电池",
            marker="A",
            object_name="电池",
            object_mapping_path="/tmp/object_mapping.json",
        )

        self.assertEqual(interpret_spoken_confirmation("确认", pending), "ambiguous")
        self.assertEqual(interpret_spoken_confirmation("确认更新", pending), "confirm")
        self.assertEqual(interpret_spoken_confirmation("确认更新标号A映射", pending), "confirm")

    def test_spoken_confirmation_accepts_action_specific_speed_change_confirmation(self) -> None:
        pending = build_speed_change_confirmation("减速到30%", target_speed="30%", target_speed_percent=30)

        self.assertEqual(interpret_spoken_confirmation("确认", pending), "ambiguous")
        self.assertEqual(interpret_spoken_confirmation("确认减速", pending), "confirm")
        self.assertEqual(interpret_spoken_confirmation("确认限速到30%", pending), "confirm")

    def test_metadata_serializes_action_details_and_expiration(self) -> None:
        pending = build_rule_edit_confirmation(
            "禁用人员侵入规则",
            rule_id="stop_on_person_intrusion",
            rule_name="人员侵入停机规则",
            patch={"rule_id": "stop_on_person_intrusion", "changes": {"enabled": False}},
            rules_path="/tmp/rules.json",
        ).with_runtime_state(confirmation_id="abc", expires_at=time.monotonic() + 30.0)

        payload = pending.as_dict(now=time.monotonic())

        self.assertEqual(payload["id"], "abc")
        self.assertEqual(payload["action_type"], ACTION_RULE_EDIT)
        self.assertEqual(payload["details"]["rule_id"], "stop_on_person_intrusion")
        self.assertGreater(payload["expires_in_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
