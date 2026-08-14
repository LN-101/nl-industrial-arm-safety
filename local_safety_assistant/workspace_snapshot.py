"""Bounded read-only workspace facts for routing and guidance."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_safety_assistant.arm_rules import load_arm_rule_document
from local_safety_assistant.confirmation import PendingConfirmation
from local_safety_assistant.object_mapping import DEFAULT_ALLOWED_MARKERS, load_object_mapping_document
from local_safety_assistant.rules import load_rule_document


@dataclass(frozen=True)
class WorkspaceSnapshotError:
    source: str
    path: Path | None
    message: str


@dataclass(frozen=True)
class WorkspaceRuleSummary:
    path: Path
    version: int
    total_rules: int
    enabled_rules: int
    disabled_rules: int
    enabled_rule_names: tuple[str, ...]


@dataclass(frozen=True)
class WorkspaceObjectMappingEntry:
    marker: str
    object_name: str
    enabled: bool


@dataclass(frozen=True)
class WorkspaceObjectMappingSummary:
    path: Path
    version: int
    entries: tuple[WorkspaceObjectMappingEntry, ...]


@dataclass(frozen=True)
class WorkspaceArmRuntimeSummary:
    path: Path
    capture_requested: bool
    capture_goal: str
    capture_object: str | None
    stop_requested: bool
    recover_requested: bool
    decelerate: str
    safety_distance: str


@dataclass(frozen=True)
class WorkspacePendingConfirmationSummary:
    action_type: str
    summary: str
    original_text: str
    expires_in_seconds: float | None


@dataclass(frozen=True)
class WorkspaceSnapshot:
    rules: WorkspaceRuleSummary | None = None
    object_mapping: WorkspaceObjectMappingSummary | None = None
    arm_runtime: WorkspaceArmRuntimeSummary | None = None
    pending_confirmation: WorkspacePendingConfirmationSummary | None = None
    errors: tuple[WorkspaceSnapshotError, ...] = ()

    def error_for(self, source: str) -> WorkspaceSnapshotError | None:
        for error in self.errors:
            if error.source == source:
                return error
        return None


def load_workspace_snapshot(
    *,
    rules_path: Path | None = None,
    object_mapping_path: Path | None = None,
    arm_rules_path: Path | None = None,
    pending_confirmation: PendingConfirmation | None = None,
    now: float | None = None,
) -> WorkspaceSnapshot:
    """Load cheap, bounded workspace facts without calling models or vision."""
    errors: list[WorkspaceSnapshotError] = []
    rules = _load_rule_summary(rules_path, errors)
    object_mapping = _load_object_mapping_summary(object_mapping_path, errors)
    arm_runtime = _load_arm_runtime_summary(arm_rules_path, errors)
    pending = _pending_confirmation_summary(pending_confirmation, now=now)
    return WorkspaceSnapshot(
        rules=rules,
        object_mapping=object_mapping,
        arm_runtime=arm_runtime,
        pending_confirmation=pending,
        errors=tuple(errors),
    )


def _load_rule_summary(
    path: Path | None,
    errors: list[WorkspaceSnapshotError],
) -> WorkspaceRuleSummary | None:
    if path is None:
        return None
    try:
        document = load_rule_document(path)
    except (OSError, ValueError) as error:
        errors.append(_snapshot_error("rules", path, error))
        return None

    rules = document.get("rules")
    rule_items = [rule for rule in rules if isinstance(rule, dict)] if isinstance(rules, list) else []
    enabled = [rule for rule in rule_items if bool(rule.get("enabled", True))]
    names = tuple(_rule_name(rule, index) for index, rule in enumerate(enabled[:6], start=1))
    version = document.get("version")
    return WorkspaceRuleSummary(
        path=path,
        version=version if isinstance(version, int) else -1,
        total_rules=len(rule_items),
        enabled_rules=len(enabled),
        disabled_rules=max(0, len(rule_items) - len(enabled)),
        enabled_rule_names=names,
    )


def _load_object_mapping_summary(
    path: Path | None,
    errors: list[WorkspaceSnapshotError],
) -> WorkspaceObjectMappingSummary | None:
    if path is None:
        return None
    try:
        document = load_object_mapping_document(path)
    except (OSError, ValueError) as error:
        errors.append(_snapshot_error("object_mapping", path, error))
        return None

    markers = document.get("markers")
    entries: list[WorkspaceObjectMappingEntry] = []
    if isinstance(markers, dict):
        for marker in DEFAULT_ALLOWED_MARKERS:
            entry = markers.get(marker)
            if not isinstance(entry, dict):
                continue
            object_name = str(entry.get("object", "")).strip()
            if not object_name:
                continue
            enabled = entry.get("enabled", True)
            entries.append(
                WorkspaceObjectMappingEntry(
                    marker=marker,
                    object_name=object_name,
                    enabled=enabled if isinstance(enabled, bool) else True,
                )
            )
    version = document.get("version")
    return WorkspaceObjectMappingSummary(
        path=path,
        version=version if isinstance(version, int) else -1,
        entries=tuple(entries),
    )


def _load_arm_runtime_summary(
    path: Path | None,
    errors: list[WorkspaceSnapshotError],
) -> WorkspaceArmRuntimeSummary | None:
    if path is None:
        return None
    try:
        document = load_arm_rule_document(path)
    except (OSError, ValueError) as error:
        errors.append(_snapshot_error("arm_runtime", path, error))
        return None

    return WorkspaceArmRuntimeSummary(
        path=path,
        capture_requested=_bool_token(document.get("arm_capture")),
        capture_goal=str(document.get("arm_capture_goal") or "").strip(),
        capture_object=_optional_text(document.get("arm_capture_object")),
        stop_requested=_bool_token(document.get("arm_stop")),
        recover_requested=_bool_token(document.get("arm_recover")),
        decelerate=str(document.get("arm_decelerate") or "").strip(),
        safety_distance=str(document.get("arm_safety_distance") or "").strip(),
    )


def _pending_confirmation_summary(
    pending: PendingConfirmation | None,
    *,
    now: float | None,
) -> WorkspacePendingConfirmationSummary | None:
    if pending is None:
        return None
    expires_in_seconds: float | None = None
    if pending.expires_at is not None:
        observed_now = time.monotonic() if now is None else now
        expires_in_seconds = max(0.0, pending.expires_at - observed_now)
    return WorkspacePendingConfirmationSummary(
        action_type=pending.action_type,
        summary=pending.summary,
        original_text=pending.original_text,
        expires_in_seconds=expires_in_seconds,
    )


def _rule_name(rule: dict[str, Any], index: int) -> str:
    for key in ("name", "title", "description", "id"):
        value = rule.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"第 {index} 条规则"


def _bool_token(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "on"}
    return bool(value)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _snapshot_error(source: str, path: Path, error: Exception) -> WorkspaceSnapshotError:
    return WorkspaceSnapshotError(source=source, path=path, message=str(error))
