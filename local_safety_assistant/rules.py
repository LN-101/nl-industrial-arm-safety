"""Validated JSON safety-rule storage.

The LLM path should propose rule documents or patches, but this module owns
validation and atomic writes. That keeps raw model text away from rule files.
"""

from __future__ import annotations

import json
import math
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class RuleValidationError(ValueError):
    """Raised when a safety-rule document is not acceptable."""


_FORBIDDEN_PATCH_PATHS = {"id", "name", "description", "action.type"}
_SCALAR_TYPES = (str, int, float, bool)
PERSONNEL_DISTANCE_RULE_ID = "stop_on_person_intrusion"
PERSONNEL_DISTANCE_PATCH_PATH = "conditions.person_distance_m.lt"
PERSONNEL_DISTANCE_MIN_M = 0.1
PERSONNEL_DISTANCE_MAX_M = 5.0
_MISSING = object()


@dataclass(frozen=True)
class RulePatchChange:
    path: str
    previous_value: Any
    new_value: Any

    @property
    def changed(self) -> bool:
        return self.previous_value != self.new_value


@dataclass(frozen=True)
class RulePatchPreview:
    document: dict[str, Any]
    rule_id: str
    changes: tuple[RulePatchChange, ...]

    @property
    def changed(self) -> bool:
        return any(change.changed for change in self.changes)


def load_rule_document(path: Path) -> dict[str, Any]:
    with path.expanduser().open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    return validate_rule_document(document)


def validate_rule_document(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise RuleValidationError("Rule document must be a JSON object.")

    version = document.get("version")
    if not isinstance(version, int):
        raise RuleValidationError("Rule document must contain integer field 'version'.")

    rules = document.get("rules")
    if not isinstance(rules, list):
        raise RuleValidationError("Rule document must contain list field 'rules'.")

    seen_ids: set[str] = set()
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise RuleValidationError(f"Rule at index {index} must be a JSON object.")

        rule_id = rule.get("id")
        if not isinstance(rule_id, str) or not rule_id.strip():
            raise RuleValidationError(f"Rule at index {index} must contain non-empty string field 'id'.")
        if rule_id in seen_ids:
            raise RuleValidationError(f"Duplicate rule id: {rule_id}")
        seen_ids.add(rule_id)

        enabled = rule.get("enabled", True)
        if not isinstance(enabled, bool):
            raise RuleValidationError(f"Rule {rule_id!r} field 'enabled' must be boolean when present.")

        action = rule.get("action")
        if action is not None and not isinstance(action, (dict, str)):
            raise RuleValidationError(f"Rule {rule_id!r} field 'action' must be object or string.")
        _validate_known_rule_business_fields(rule)

    return deepcopy(document)


def write_rule_document(path: Path, document: dict[str, Any]) -> None:
    validated = validate_rule_document(document)
    target = path.expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_name(f".{target.name}.tmp")

    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(validated, handle, indent=2, ensure_ascii=True)
        handle.write("\n")

    os.replace(temp_path, target)


def replace_rule(path: Path, replacement: dict[str, Any]) -> dict[str, Any]:
    document = load_rule_document(path)
    rule_id = replacement.get("id")
    if not isinstance(rule_id, str) or not rule_id.strip():
        raise RuleValidationError("Replacement rule must contain non-empty string field 'id'.")

    updated = False
    rules = []
    for rule in document["rules"]:
        if rule["id"] == rule_id:
            rules.append(deepcopy(replacement))
            updated = True
        else:
            rules.append(rule)

    if not updated:
        rules.append(deepcopy(replacement))

    new_document = {**document, "version": document["version"] + 1, "rules": rules}
    validate_rule_document(new_document)
    write_rule_document(path, new_document)
    return new_document


def apply_rule_patch(path: Path, patch: dict[str, Any]) -> dict[str, Any]:
    """Apply a strict patch to existing scalar fields in one rule."""
    document = load_rule_document(path)
    preview = preview_rule_patch_changes(document, patch)
    if not preview.changed:
        return validate_rule_document(document)
    write_rule_document(path, preview.document)
    return preview.document


def preview_rule_patch(document: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Return the validated result of a patch without writing it."""
    return preview_rule_patch_changes(document, patch).document


def preview_rule_patch_changes(document: dict[str, Any], patch: dict[str, Any]) -> RulePatchPreview:
    """Return the validated patch result and scalar fields that would change."""
    rule_id, changes = _parse_rule_patch(patch)
    validated_document = validate_rule_document(document)
    rules = []
    updated = False
    patch_changes: list[RulePatchChange] = []

    for rule in validated_document["rules"]:
        if rule["id"] != rule_id:
            rules.append(rule)
            continue
        patched_rule, changed = _apply_changes_to_rule(rule, changes)
        patch_changes.extend(changed)
        rules.append(patched_rule)
        updated = True

    if not updated:
        raise RuleValidationError(f"Patch target rule does not exist: {rule_id}")

    version = validated_document["version"] + 1 if any(change.changed for change in patch_changes) else validated_document["version"]
    new_document = {**validated_document, "version": version, "rules": rules}
    return RulePatchPreview(
        document=validate_rule_document(new_document),
        rule_id=rule_id,
        changes=tuple(patch_changes),
    )


def _parse_rule_patch(patch: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(patch, dict):
        raise RuleValidationError("Rule patch must be a JSON object.")

    rule_id = patch.get("rule_id") or patch.get("id")
    if not isinstance(rule_id, str) or not rule_id.strip():
        raise RuleValidationError("Rule patch must contain non-empty string field 'rule_id'.")

    changes = patch.get("changes")
    if not isinstance(changes, dict) or not changes:
        raise RuleValidationError("Rule patch must contain non-empty object field 'changes'.")

    return rule_id.strip(), _flatten_patch_changes(deepcopy(changes))


def _flatten_patch_changes(changes: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in changes.items():
        if not isinstance(key, str) or not key.strip():
            raise RuleValidationError("Rule patch change paths must be non-empty strings.")
        field_path = f"{prefix}.{key.strip()}" if prefix else key.strip()
        if isinstance(value, dict):
            raise RuleValidationError(f"Rule patch changes must use dotted scalar paths, not objects: {field_path}")
        flattened[field_path] = value
    return flattened


def _apply_changes_to_rule(
    rule: dict[str, Any],
    changes: dict[str, Any],
) -> tuple[dict[str, Any], list[RulePatchChange]]:
    patched = deepcopy(rule)
    patch_changes: list[RulePatchChange] = []
    for path, value in changes.items():
        parts = _parse_patch_path(path)
        previous_value = _validate_patch_value(patched, parts, value)
        patch_change = RulePatchChange(
            path=".".join(parts),
            previous_value=deepcopy(previous_value),
            new_value=deepcopy(value),
        )
        patch_changes.append(patch_change)
        if previous_value == value:
            continue
        target = patched
        for part in parts[:-1]:
            child = target[part]
            if not isinstance(child, dict):
                raise RuleValidationError(f"Patch path crosses non-object field: {path}")
            target = child
        target[parts[-1]] = deepcopy(value)
    return patched, patch_changes


def _parse_patch_path(path: Any) -> tuple[str, ...]:
    if not isinstance(path, str) or not path.strip():
        raise RuleValidationError("Rule patch change paths must be non-empty strings.")
    normalized = path.strip()
    parts = tuple(part.strip() for part in normalized.split("."))
    if any(not part for part in parts):
        raise RuleValidationError(f"Rule patch path is invalid: {normalized}")
    canonical = ".".join(parts)
    if canonical in _FORBIDDEN_PATCH_PATHS or parts[0] in {"id", "name", "description"}:
        raise RuleValidationError(f"Rule patch path is not allowed: {normalized}")
    return parts


def _validate_patch_value(rule: dict[str, Any], parts: tuple[str, ...], value: Any) -> Any:
    current: Any = rule
    path = ".".join(parts)
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            raise RuleValidationError(f"Rule patch path does not exist: {path}")
        current = current[part]

    if isinstance(current, dict) or isinstance(current, list):
        raise RuleValidationError(f"Rule patch cannot replace object or list field: {path}")
    if not _is_scalar(value):
        raise RuleValidationError(f"Rule patch value must be scalar: {path}")
    if not _is_compatible_scalar_type(current, value):
        raise RuleValidationError(f"Rule patch value type does not match existing field: {path}")
    _validate_patch_business_value(rule, parts, value)
    return current


def _validate_known_rule_business_fields(rule: dict[str, Any]) -> None:
    if rule.get("id") != PERSONNEL_DISTANCE_RULE_ID:
        return
    value = _nested_value(rule, tuple(PERSONNEL_DISTANCE_PATCH_PATH.split(".")))
    if value is _MISSING:
        return
    _validate_personnel_distance_value(value)


def _validate_patch_business_value(rule: dict[str, Any], parts: tuple[str, ...], value: Any) -> None:
    if rule.get("id") != PERSONNEL_DISTANCE_RULE_ID:
        return
    if ".".join(parts) != PERSONNEL_DISTANCE_PATCH_PATH:
        return
    _validate_personnel_distance_value(value)


def _nested_value(value: dict[str, Any], parts: tuple[str, ...]) -> Any:
    current: Any = value
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _validate_personnel_distance_value(value: Any) -> None:
    path = PERSONNEL_DISTANCE_PATCH_PATH
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuleValidationError(f"Personnel safety distance threshold must be a number in meters: {path}")
    distance_m = float(value)
    if not math.isfinite(distance_m):
        raise RuleValidationError(f"Personnel safety distance threshold must be a finite number in meters: {path}")
    if not PERSONNEL_DISTANCE_MIN_M <= distance_m <= PERSONNEL_DISTANCE_MAX_M:
        min_text = f"{PERSONNEL_DISTANCE_MIN_M:g}"
        max_text = f"{PERSONNEL_DISTANCE_MAX_M:g}"
        raise RuleValidationError(
            f"Personnel safety distance threshold must be between {min_text} and {max_text} meters: {path}"
        )


def _is_scalar(value: Any) -> bool:
    return isinstance(value, _SCALAR_TYPES)


def _is_compatible_scalar_type(current: Any, value: Any) -> bool:
    if isinstance(current, bool):
        return isinstance(value, bool)
    if isinstance(current, (int, float)) and not isinstance(current, bool):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if isinstance(current, str):
        return isinstance(value, str)
    return type(value) is type(current)
