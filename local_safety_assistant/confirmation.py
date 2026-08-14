"""Shared high-risk action confirmation contract."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any


ACTION_ESTOP_TRIGGER = "emergency_stop_trigger"
ACTION_ESTOP_RELEASE = "emergency_stop_release"
ACTION_RULE_EDIT = "rule_edit"
ACTION_OBJECT_MAPPING_UPDATE = "object_mapping_update"
ACTION_OBJECT_GRASP_EXECUTION = "object_grasp_execution"
ACTION_SPEED_CHANGE = "speed_change"
ACTION_GOAL_MOTION = "goal_motion"

DEFAULT_CONFIRMATION_TTL_SECONDS = 60.0

_CONFIRMATION_REQUIRED_ACTIONS = {
    ACTION_ESTOP_RELEASE,
    ACTION_RULE_EDIT,
    ACTION_OBJECT_MAPPING_UPDATE,
    ACTION_OBJECT_GRASP_EXECUTION,
    ACTION_SPEED_CHANGE,
    ACTION_GOAL_MOTION,
}

_AFFIRMATIVE_WORDS = (
    "确认执行",
    "确认解除",
    "确认复位",
    "确认清除",
    "确认修改",
    "确认更新",
    "确认抓取",
    "确认限速",
    "确认减速",
    "确认调速",
    "执行确认",
)
_NEGATIVE_WORDS = (
    "取消",
    "不要",
    "别执行",
    "不执行",
    "否",
    "不用",
    "停止确认",
    "先不要",
)
_ACTION_AFFIRMATIVE_WORDS = {
    ACTION_ESTOP_RELEASE: ("确认解除", "确认复位", "确认清除"),
    ACTION_RULE_EDIT: ("确认修改规则", "确认更新规则"),
    ACTION_OBJECT_MAPPING_UPDATE: ("确认更新", "确认更新映射", "确认修改映射"),
    ACTION_OBJECT_GRASP_EXECUTION: ("确认抓取",),
    ACTION_SPEED_CHANGE: ("确认限速", "确认减速", "确认调速"),
    ACTION_GOAL_MOTION: ("确认移动",),
}


@dataclass(frozen=True)
class PendingConfirmation:
    action_type: str
    original_text: str
    summary: str
    prompt: str
    details: dict[str, Any]
    checklist: tuple[str, ...] = ()
    confirmation_id: str | None = None
    expires_at: float | None = None

    def with_runtime_state(self, *, confirmation_id: str, expires_at: float) -> "PendingConfirmation":
        return replace(self, confirmation_id=confirmation_id, expires_at=expires_at)

    def as_dict(self, *, now: float | None = None) -> dict[str, Any]:
        expires_in_seconds: float | None = None
        if self.expires_at is not None and now is not None:
            expires_in_seconds = max(0.0, self.expires_at - now)
        return {
            "id": self.confirmation_id,
            "action_type": self.action_type,
            "original_text": self.original_text,
            "summary": self.summary,
            "prompt": self.prompt,
            "checklist": list(self.checklist),
            "details": dict(self.details),
            "expires_in_seconds": expires_in_seconds,
        }


def requires_confirmation(action_type: str) -> bool:
    return action_type in _CONFIRMATION_REQUIRED_ACTIONS


def build_estop_release_confirmation(original_text: str) -> PendingConfirmation:
    return PendingConfirmation(
        action_type=ACTION_ESTOP_RELEASE,
        original_text=original_text,
        summary="解除或复位急停",
        prompt="确认解除急停前，请确认危险源已排除、人员已离开机械臂工作区，并准备好重新复位设备。",
        checklist=(
            "现场人员已离开机械臂工作区。",
            "急停原因已经排查并解除。",
            "复位后将先低速观察机械臂状态。",
        ),
        details={"command": "estop_release"},
    )


def build_rule_edit_confirmation(
    original_text: str,
    *,
    rule_id: str,
    rule_name: str,
    patch: dict[str, Any],
    rules_path: str,
    change_summary: str | None = None,
) -> PendingConfirmation:
    target = rule_name or rule_id
    summary = f"修改安全规则：{target}"
    if change_summary:
        summary = f"{summary}，{change_summary}"
    return PendingConfirmation(
        action_type=ACTION_RULE_EDIT,
        original_text=original_text,
        summary=summary,
        prompt="确认修改安全规则前，请确认这不会降低当前作业的必要防护，并已记录现场风险。",
        checklist=(
            "理解本次修改会影响后续安全规则判断。",
            "若是禁用规则或放宽阈值，现场已经采取替代防护。",
            "修改后会复核规则版本和实际触发行为。",
        ),
        details={
            "rule_id": rule_id,
            "rule_name": rule_name,
            "patch": patch,
            "rules_path": rules_path,
            "change_summary": change_summary,
        },
    )


def build_object_mapping_update_confirmation(
    original_text: str,
    *,
    marker: str,
    object_name: str,
    object_mapping_path: str,
    previous_object: str | None = None,
) -> PendingConfirmation:
    summary = f"更新标号{marker} 的物体映射为{object_name}"
    if previous_object is not None:
        summary = f"更新标号{marker} 的物体映射：原值{previous_object}，改为{object_name}"
    return PendingConfirmation(
        action_type=ACTION_OBJECT_MAPPING_UPDATE,
        original_text=original_text,
        summary=summary,
        prompt=f"确认更新标号{marker} 前，请确认现场贴纸确实贴在{object_name} 上，避免后续抓取目标解析错误。",
        checklist=(
            "现场标号贴纸和物体名称已经人工核对。",
            "这个映射会影响后续抓取目标解析。",
            "如贴纸位置变化，应重新更新映射。",
        ),
        details={
            "marker": marker,
            "object_name": object_name,
            "object_mapping_path": object_mapping_path,
            "previous_object": previous_object,
        },
    )


def build_object_grasp_execution_confirmation(
    original_text: str,
    *,
    marker: str | None = None,
    object_name: str | None = None,
) -> PendingConfirmation:
    marker_label = f"标号{marker}" if marker else ""
    if marker_label and object_name:
        target = f"{marker_label}，实际物体：{object_name}"
    else:
        target = f"实际物体：{object_name}" if object_name else marker_label or "目标物体"
    return PendingConfirmation(
        action_type=ACTION_OBJECT_GRASP_EXECUTION,
        original_text=original_text,
        summary=f"执行抓取：{target}",
        prompt=f"确认抓取{target} 前，请确认气泵路径无人员、无障碍物，并且目标物体和标号匹配。",
        checklist=(
            "气泵和机械臂路径无人员进入。",
            f"目标物体和识别标号已核对：{target}。",
            "执行中可随时急停。",
        ),
        details={"marker": marker, "object_name": object_name},
    )


def build_speed_change_confirmation(
    original_text: str,
    *,
    target_speed: str | None = None,
    target_speed_percent: float | None = None,
    arm_decelerate: str | None = None,
    arm_rules_path: str | None = None,
) -> PendingConfirmation:
    target_text = target_speed or (
        f"{target_speed_percent:g}%" if target_speed_percent is not None else ""
    )
    summary = f"将机械臂目标速度调整为 {target_text}" if target_text else "修改速度或限速设置"
    prompt = "确认修改速度前，请确认当前工作区安全，且本次目标速度符合现场作业要求。"
    return PendingConfirmation(
        action_type=ACTION_SPEED_CHANGE,
        original_text=original_text,
        summary=summary,
        prompt=prompt,
        checklist=(
            "机械臂路径和人员安全距离已经确认。",
            "本次速度调整有明确作业原因。",
            "必要时先以低速点动验证。",
        ),
        details={
            "target_speed": target_text or target_speed,
            "target_speed_percent": target_speed_percent,
            "arm_decelerate": arm_decelerate,
            "arm_rules_path": arm_rules_path,
        },
    )


def build_goal_motion_confirmation(original_text: str, *, goal: dict[str, Any]) -> PendingConfirmation:
    return PendingConfirmation(
        action_type=ACTION_GOAL_MOTION,
        original_text=original_text,
        summary="移动机械臂到目标点",
        prompt="确认移动前，请确认目标点和机械臂路径安全，人员未进入工作区。",
        checklist=(
            "目标坐标已核对。",
            "机械臂路径无遮挡。",
            "人员已离开工作区。",
        ),
        details={"goal": dict(goal)},
    )


def interpret_spoken_confirmation(text: str, pending: PendingConfirmation) -> str:
    """Return ``confirm``, ``cancel``, or ``ambiguous`` for a spoken answer."""
    normalized = re.sub(r"\s+", "", text.strip())
    if not normalized:
        return "ambiguous"
    if any(word in normalized for word in _NEGATIVE_WORDS):
        return "cancel"
    if not any(word in normalized for word in _AFFIRMATIVE_WORDS):
        return "ambiguous"
    if pending.action_type == ACTION_ESTOP_RELEASE:
        return "confirm" if any(word in normalized for word in ("急停", "复位", "解除", "清除")) else "ambiguous"
    if pending.action_type == ACTION_RULE_EDIT:
        if _mentions_action_affirmative(normalized, pending.action_type):
            return "confirm"
        rule_name = str(pending.details.get("rule_name") or "")
        rule_id = str(pending.details.get("rule_id") or "")
        return "confirm" if _mentions_any(normalized, ("规则", "安全", rule_name, rule_id)) else "ambiguous"
    if pending.action_type == ACTION_OBJECT_MAPPING_UPDATE:
        if _mentions_action_affirmative(normalized, pending.action_type):
            return "confirm"
        marker = str(pending.details.get("marker") or "")
        object_name = str(pending.details.get("object_name") or "")
        return "confirm" if _mentions_any(normalized, (f"标号{marker}", marker, object_name, "映射")) else "ambiguous"
    if pending.action_type == ACTION_OBJECT_GRASP_EXECUTION:
        if _mentions_action_affirmative(normalized, pending.action_type):
            return "confirm"
        object_name = str(pending.details.get("object_name") or "")
        return "confirm" if _mentions_any(normalized, ("抓取", object_name)) else "ambiguous"
    if pending.action_type == ACTION_SPEED_CHANGE:
        if _mentions_action_affirmative(normalized, pending.action_type):
            return "confirm"
        return "confirm" if _mentions_any(normalized, ("速度", "限速", "调速")) else "ambiguous"
    if pending.action_type == ACTION_GOAL_MOTION:
        if _mentions_action_affirmative(normalized, pending.action_type):
            return "confirm"
        return "confirm" if _mentions_any(normalized, ("移动", "目标", "坐标")) else "ambiguous"
    return "ambiguous"


def _mentions_any(text: str, values: tuple[str, ...]) -> bool:
    return any(value and value in text for value in values)


def _mentions_action_affirmative(text: str, action_type: str) -> bool:
    return _mentions_any(text, _ACTION_AFFIRMATIVE_WORDS.get(action_type, ()))
