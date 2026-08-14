"""Validated JSON commands for the ROS2 arm runtime."""

from __future__ import annotations

import json
import math
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_safety_assistant.object_mapping import DEFAULT_ALLOWED_MARKERS, normalize_marker
from local_safety_assistant.rules import (
    PERSONNEL_DISTANCE_MAX_M,
    PERSONNEL_DISTANCE_MIN_M,
    PERSONNEL_DISTANCE_PATCH_PATH,
    PERSONNEL_DISTANCE_RULE_ID,
    RuleValidationError,
)


class ArmRulesValidationError(ValueError):
    """Raised when an arm runtime command document is not acceptable."""


ARM_RULE_FIELDS = (
    "arm_capture",
    "arm_capture_goal",
    "arm_decelerate",
    "arm_stop",
    "arm_recover",
    "arm_safety_distance",
)
ARM_RULE_TRACE_FIELDS = (
    "arm_capture_object",
    "arm_capture_original_text",
)
DEFAULT_ARM_RULE_DOCUMENT: dict[str, Any] = {
    "arm_capture": "False",
    "arm_capture_goal": "A",
    "arm_decelerate": "1.0",
    "arm_stop": "False",
    "arm_recover": "False",
    "arm_safety_distance": "0.2",
}
_MISSING = object()


@dataclass(frozen=True)
class SafetyDistanceArmRuleSyncResult:
    arm_rules_path: Path
    synced: bool
    distance_m: float | None = None


@dataclass(frozen=True)
class ArmDecelerationRequestResult:
    arm_rules_path: Path
    target_speed_percent: float
    arm_decelerate: str


@dataclass(frozen=True)
class ArmEstopRequestResult:
    arm_rules_path: Path
    active: bool
    arm_stop: str
    arm_recover: str


def load_arm_rule_document(path: Path) -> dict[str, Any]:
    with path.expanduser().open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    return validate_arm_rule_document(document)


def validate_arm_rule_document(
    document: Any,
    *,
    allowed_markers: tuple[str, ...] = DEFAULT_ALLOWED_MARKERS,
) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ArmRulesValidationError("Arm rules document must be a JSON object.")

    validated = deepcopy(DEFAULT_ARM_RULE_DOCUMENT)
    for field in ARM_RULE_FIELDS:
        if field in document:
            validated[field] = document[field]

    capture_goal = normalize_marker(str(validated["arm_capture_goal"]))
    if capture_goal not in set(allowed_markers):
        allowed_text = "、".join(allowed_markers)
        raise ArmRulesValidationError(
            f"Current arm runtime only supports capture goals {allowed_text}."
        )
    validated["arm_capture_goal"] = capture_goal

    for field in ("arm_capture", "arm_stop", "arm_recover"):
        validated[field] = _normalize_bool_token(validated[field], field)

    validated["arm_decelerate"] = _normalize_float_token(
        validated["arm_decelerate"],
        "arm_decelerate",
        min_value=0.0,
        max_value=1.0,
    )
    validated["arm_safety_distance"] = _normalize_float_token(
        validated["arm_safety_distance"],
        "arm_safety_distance",
        min_value=0.0,
        max_value=5.0,
    )
    for field in ARM_RULE_TRACE_FIELDS:
        value = document.get(field)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ArmRulesValidationError(f"Field {field!r} must be a string.")
        validated[field] = value.strip()
    return validated


def write_arm_rule_document(path: Path, document: dict[str, Any]) -> None:
    validated = validate_arm_rule_document(document)
    target = path.expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_name(f".{target.name}.tmp")

    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(validated, handle, indent=2, ensure_ascii=True)
        handle.write("\n")

    os.replace(temp_path, target)


def request_object_grasp(
    path: Path,
    *,
    marker: str,
    object_name: str | None = None,
    original_text: str | None = None,
    allowed_markers: tuple[str, ...] = DEFAULT_ALLOWED_MARKERS,
) -> dict[str, Any]:
    normalized_marker = normalize_marker(marker)
    if normalized_marker not in set(allowed_markers):
        allowed_text = "、".join(allowed_markers)
        raise ArmRulesValidationError(
            f"Current arm runtime only supports capture goals {allowed_text}."
        )

    try:
        document = load_arm_rule_document(path)
    except FileNotFoundError:
        document = deepcopy(DEFAULT_ARM_RULE_DOCUMENT)

    updated = {
        **document,
        "arm_capture": "True",
        "arm_capture_goal": normalized_marker,
    }
    if object_name:
        updated["arm_capture_object"] = str(object_name).strip()
    if original_text:
        updated["arm_capture_original_text"] = str(original_text).strip()
    write_arm_rule_document(path, updated)
    return validate_arm_rule_document(updated, allowed_markers=allowed_markers)


def request_arm_deceleration(
    path: Path,
    *,
    target_speed_percent: float,
) -> ArmDecelerationRequestResult:
    """Write a confirmed target speed percentage into the arm runtime JSON."""
    arm_decelerate = _arm_decelerate_from_percent(target_speed_percent)
    try:
        document = load_arm_rule_document(path)
    except FileNotFoundError:
        document = deepcopy(DEFAULT_ARM_RULE_DOCUMENT)

    updated = {**document, "arm_decelerate": arm_decelerate}
    write_arm_rule_document(path, updated)
    validated = load_arm_rule_document(path)
    return ArmDecelerationRequestResult(
        arm_rules_path=path,
        target_speed_percent=float(target_speed_percent),
        arm_decelerate=str(validated["arm_decelerate"]),
    )


def request_arm_estop(
    path: Path,
    *,
    active: bool,
) -> ArmEstopRequestResult:
    """Write an emergency-stop or recovery request into the arm runtime JSON."""
    if not isinstance(active, bool):
        raise ArmRulesValidationError("Emergency stop active state must be boolean.")
    try:
        document = load_arm_rule_document(path)
    except FileNotFoundError:
        document = deepcopy(DEFAULT_ARM_RULE_DOCUMENT)

    updated = {
        **document,
        "arm_stop": "True" if active else "False",
        "arm_recover": "False" if active else "True",
    }
    write_arm_rule_document(path, updated)
    validated = load_arm_rule_document(path)
    return ArmEstopRequestResult(
        arm_rules_path=path,
        active=active,
        arm_stop=str(validated["arm_stop"]),
        arm_recover=str(validated["arm_recover"]),
    )


def sync_personnel_distance_to_arm_rules(
    previous_rule_document: dict[str, Any],
    updated_rule_document: dict[str, Any],
    arm_rules_path: Path,
) -> SafetyDistanceArmRuleSyncResult:
    """Sync changed personnel distance from safety rules into arm runtime JSON."""
    previous_distance = _personnel_distance_value(previous_rule_document)
    updated_distance = _personnel_distance_value(updated_rule_document)
    if _same_distance_value(previous_distance, updated_distance):
        return SafetyDistanceArmRuleSyncResult(arm_rules_path=arm_rules_path, synced=False)

    distance_m = _require_enabled_personnel_distance(updated_rule_document)
    try:
        arm_document = load_arm_rule_document(arm_rules_path)
    except FileNotFoundError:
        arm_document = deepcopy(DEFAULT_ARM_RULE_DOCUMENT)

    write_arm_rule_document(arm_rules_path, {**arm_document, "arm_safety_distance": distance_m})
    return SafetyDistanceArmRuleSyncResult(
        arm_rules_path=arm_rules_path,
        synced=True,
        distance_m=distance_m,
    )


def _normalize_bool_token(value: Any, field: str) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return "True"
        if normalized in {"false", "0", "no", "n", "off", ""}:
            return "False"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "True" if bool(value) else "False"
    raise ArmRulesValidationError(f"Field {field!r} must be boolean-like.")


def _normalize_float_token(
    value: Any,
    field: str,
    *,
    min_value: float,
    max_value: float,
) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ArmRulesValidationError(f"Field {field!r} must be numeric.") from exc
    if not min_value <= parsed <= max_value:
        raise ArmRulesValidationError(
            f"Field {field!r} must be between {min_value:g} and {max_value:g}."
        )
    return f"{parsed:g}"


def _arm_decelerate_from_percent(target_speed_percent: float) -> str:
    if isinstance(target_speed_percent, bool):
        raise ArmRulesValidationError("Target speed percent must be numeric.")
    try:
        parsed = float(target_speed_percent)
    except (TypeError, ValueError) as exc:
        raise ArmRulesValidationError("Target speed percent must be numeric.") from exc
    if not math.isfinite(parsed):
        raise ArmRulesValidationError("Target speed percent must be finite.")
    if not 0.0 <= parsed <= 100.0:
        raise ArmRulesValidationError("Target speed percent must be between 0 and 100.")
    return _normalize_float_token(parsed / 100.0, "arm_decelerate", min_value=0.0, max_value=1.0)


def _personnel_distance_rule(document: dict[str, Any]) -> dict[str, Any] | object:
    rules = document.get("rules")
    if not isinstance(rules, list):
        return _MISSING
    for rule in rules:
        if isinstance(rule, dict) and rule.get("id") == PERSONNEL_DISTANCE_RULE_ID:
            return rule
    return _MISSING


def _personnel_distance_value(document: dict[str, Any]) -> Any:
    rule = _personnel_distance_rule(document)
    if rule is _MISSING:
        return _MISSING
    assert isinstance(rule, dict)
    current: Any = rule
    for part in PERSONNEL_DISTANCE_PATCH_PATH.split("."):
        if not isinstance(current, dict) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _require_enabled_personnel_distance(document: dict[str, Any]) -> float:
    rule = _personnel_distance_rule(document)
    if rule is _MISSING:
        raise RuleValidationError("人员安全距离规则不存在，机械臂运行时距离未同步。")
    assert isinstance(rule, dict)
    if rule.get("enabled", True) is not True:
        raise RuleValidationError("人员安全距离规则当前未启用，机械臂运行时距离未同步。")
    value = _personnel_distance_value(document)
    if value is _MISSING:
        raise RuleValidationError("人员安全距离阈值缺失，机械臂运行时距离未同步。")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuleValidationError("人员安全距离阈值必须是米为单位的数字，机械臂运行时距离未同步。")
    distance_m = float(value)
    if not math.isfinite(distance_m):
        raise RuleValidationError("人员安全距离阈值必须是有限数字，机械臂运行时距离未同步。")
    if not PERSONNEL_DISTANCE_MIN_M <= distance_m <= PERSONNEL_DISTANCE_MAX_M:
        raise RuleValidationError(
            "人员安全距离阈值超出允许范围，机械臂运行时距离未同步。"
        )
    return distance_m


def _same_distance_value(left: Any, right: Any) -> bool:
    if left is _MISSING or right is _MISSING:
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if isinstance(left, bool) or isinstance(right, bool):
            return left == right
        return float(left) == float(right)
    return left == right
