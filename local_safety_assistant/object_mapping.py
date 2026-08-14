"""Validated JSON object-marker mapping storage."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ObjectMappingValidationError(ValueError):
    """Raised when an object-marker mapping document is not acceptable."""


DEFAULT_ALLOWED_MARKERS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class ObjectMappingUpdatePreview:
    document: dict[str, Any]
    marker: str
    previous_object: str
    new_object: str

    @property
    def changed(self) -> bool:
        return normalize_object_name(self.previous_object) != self.new_object


def load_object_mapping_document(
    path: Path,
    *,
    allowed_markers: tuple[str, ...] = DEFAULT_ALLOWED_MARKERS,
) -> dict[str, Any]:
    with path.expanduser().open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    return validate_object_mapping_document(document, allowed_markers=allowed_markers)


def validate_object_mapping_document(
    document: Any,
    *,
    allowed_markers: tuple[str, ...] = DEFAULT_ALLOWED_MARKERS,
) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ObjectMappingValidationError("Object mapping document must be a JSON object.")

    version = document.get("version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise ObjectMappingValidationError("Object mapping document must contain integer field 'version'.")

    markers = document.get("markers")
    if not isinstance(markers, dict):
        raise ObjectMappingValidationError("Object mapping document must contain object field 'markers'.")

    allowed = set(allowed_markers)
    for marker, entry in markers.items():
        if marker not in allowed:
            allowed_text = ", ".join(allowed_markers)
            raise ObjectMappingValidationError(f"Unsupported marker {marker!r}; allowed markers: {allowed_text}.")
        if not isinstance(entry, dict):
            raise ObjectMappingValidationError(f"Marker {marker!r} entry must be a JSON object.")

        object_name = entry.get("object")
        if not isinstance(object_name, str) or not object_name.strip():
            raise ObjectMappingValidationError(f"Marker {marker!r} must contain non-empty string field 'object'.")

        enabled = entry.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ObjectMappingValidationError(f"Marker {marker!r} field 'enabled' must be boolean when present.")

    return deepcopy(document)


def write_object_mapping_document(
    path: Path,
    document: dict[str, Any],
    *,
    allowed_markers: tuple[str, ...] = DEFAULT_ALLOWED_MARKERS,
) -> None:
    validated = validate_object_mapping_document(document, allowed_markers=allowed_markers)
    target = path.expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_name(f".{target.name}.tmp")

    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(validated, handle, indent=2, ensure_ascii=True)
        handle.write("\n")

    os.replace(temp_path, target)


def update_object_mapping(
    path: Path,
    *,
    marker: str,
    object_name: str,
    allowed_markers: tuple[str, ...] = DEFAULT_ALLOWED_MARKERS,
) -> dict[str, Any]:
    document = load_object_mapping_document(path, allowed_markers=allowed_markers)
    preview = preview_object_mapping_update_changes(
        document,
        marker=marker,
        object_name=object_name,
        allowed_markers=allowed_markers,
    )
    if preview.changed:
        write_object_mapping_document(path, preview.document, allowed_markers=allowed_markers)
    return preview.document


def preview_object_mapping_update(
    document: dict[str, Any],
    *,
    marker: str,
    object_name: str,
    allowed_markers: tuple[str, ...] = DEFAULT_ALLOWED_MARKERS,
) -> dict[str, Any]:
    """Return the validated mapping update result without writing it."""
    return preview_object_mapping_update_changes(
        document,
        marker=marker,
        object_name=object_name,
        allowed_markers=allowed_markers,
    ).document


def preview_object_mapping_update_changes(
    document: dict[str, Any],
    *,
    marker: str,
    object_name: str,
    allowed_markers: tuple[str, ...] = DEFAULT_ALLOWED_MARKERS,
) -> ObjectMappingUpdatePreview:
    """Return the validated mapping update result and value change metadata."""
    normalized_marker = normalize_marker(marker)
    normalized_object = normalize_object_name(object_name)
    if normalized_marker not in set(allowed_markers):
        allowed_text = "、".join(allowed_markers)
        raise ObjectMappingValidationError(f"Current object mapping only supports markers {allowed_text}.")
    if not normalized_object:
        raise ObjectMappingValidationError("Object name must be a non-empty string.")

    document = validate_object_mapping_document(document, allowed_markers=allowed_markers)
    markers = document.get("markers")
    if not isinstance(markers, dict) or normalized_marker not in markers:
        allowed_text = "、".join(allowed_markers)
        raise ObjectMappingValidationError(f"Current object mapping only supports markers {allowed_text}.")

    entry = deepcopy(markers[normalized_marker])
    if not isinstance(entry, dict):
        raise ObjectMappingValidationError(f"Marker {normalized_marker!r} entry must be a JSON object.")
    previous_object = str(entry.get("object", ""))
    entry["object"] = normalized_object

    updated_markers = deepcopy(markers)
    updated_markers[normalized_marker] = entry
    version = (
        document["version"] + 1
        if normalize_object_name(previous_object) != normalized_object
        else document["version"]
    )
    updated_document = {**document, "version": version, "markers": updated_markers}
    return ObjectMappingUpdatePreview(
        document=validate_object_mapping_document(updated_document, allowed_markers=allowed_markers),
        marker=normalized_marker,
        previous_object=previous_object,
        new_object=normalized_object,
    )


def get_object_mapping(
    path: Path,
    *,
    marker: str,
    allowed_markers: tuple[str, ...] = DEFAULT_ALLOWED_MARKERS,
) -> dict[str, Any]:
    normalized_marker = normalize_marker(marker)
    if normalized_marker not in set(allowed_markers):
        allowed_text = "、".join(allowed_markers)
        raise ObjectMappingValidationError(f"Current object mapping only supports markers {allowed_text}.")

    document = load_object_mapping_document(path, allowed_markers=allowed_markers)
    markers = document.get("markers")
    if not isinstance(markers, dict) or normalized_marker not in markers:
        allowed_text = "、".join(allowed_markers)
        raise ObjectMappingValidationError(f"Current object mapping only supports markers {allowed_text}.")

    entry = deepcopy(markers[normalized_marker])
    if not isinstance(entry, dict):
        raise ObjectMappingValidationError(f"Marker {normalized_marker!r} entry must be a JSON object.")
    object_name = entry.get("object")
    enabled = entry.get("enabled", True)
    return {
        "version": document["version"],
        "marker": normalized_marker,
        "object": object_name,
        "enabled": enabled,
    }


def resolve_object_grasp_target(
    path: Path,
    *,
    marker: str | None = None,
    object_name: str | None = None,
    allowed_markers: tuple[str, ...] = DEFAULT_ALLOWED_MARKERS,
) -> dict[str, Any]:
    normalized_marker = normalize_marker(marker) if marker is not None else None
    normalized_object = normalize_object_name(object_name or "")
    if bool(normalized_marker) == bool(normalized_object):
        raise ObjectMappingValidationError("抓取目标必须只指定标号或物体名称之一。")

    document = load_object_mapping_document(path, allowed_markers=allowed_markers)
    markers = document.get("markers")
    if not isinstance(markers, dict):
        raise ObjectMappingValidationError("Object mapping document must contain object field 'markers'.")

    if normalized_marker:
        return _resolve_grasp_marker(document, markers, normalized_marker, allowed_markers)
    return _resolve_grasp_object_name(document, markers, normalized_object)


def _resolve_grasp_marker(
    document: dict[str, Any],
    markers: dict[str, Any],
    marker: str,
    allowed_markers: tuple[str, ...],
) -> dict[str, Any]:
    if marker not in set(allowed_markers):
        allowed_text = "、".join(allowed_markers)
        raise ObjectMappingValidationError(f"当前只支持标号{allowed_text}。")
    if marker not in markers:
        allowed_text = "、".join(allowed_markers)
        raise ObjectMappingValidationError(f"当前只支持标号{allowed_text}。")

    entry = markers[marker]
    if not isinstance(entry, dict):
        raise ObjectMappingValidationError(f"Marker {marker!r} entry must be a JSON object.")
    object_name = normalize_object_name(str(entry.get("object", "")))
    if not bool(entry.get("enabled", True)):
        raise ObjectMappingValidationError(f"标号{marker} 对应{object_name}，但该标号当前未启用。")
    return {
        "version": document["version"],
        "marker": marker,
        "object": object_name,
        "enabled": True,
        "target_source": "marker",
    }


def _resolve_grasp_object_name(
    document: dict[str, Any],
    markers: dict[str, Any],
    object_name: str,
) -> dict[str, Any]:
    enabled_matches: list[tuple[str, str]] = []
    disabled_matches: list[tuple[str, str]] = []
    for marker, entry in markers.items():
        if not isinstance(entry, dict):
            continue
        mapped_object = normalize_object_name(str(entry.get("object", "")))
        if mapped_object != object_name:
            continue
        if bool(entry.get("enabled", True)):
            enabled_matches.append((marker, mapped_object))
        else:
            disabled_matches.append((marker, mapped_object))

    if len(enabled_matches) == 1:
        marker, mapped_object = enabled_matches[0]
        return {
            "version": document["version"],
            "marker": marker,
            "object": mapped_object,
            "enabled": True,
            "target_source": "object_name",
        }
    if len(enabled_matches) > 1:
        labels = "、".join(f"标号{marker}" for marker, _ in enabled_matches)
        raise ObjectMappingValidationError(
            f"物体名称“{object_name}”同时对应{labels}，请改用具体标号。"
        )
    if disabled_matches:
        labels = "、".join(f"标号{marker}" for marker, _ in disabled_matches)
        raise ObjectMappingValidationError(f"物体名称“{object_name}”只对应未启用的{labels}。")
    raise ObjectMappingValidationError(f"没有找到已启用的物体名称“{object_name}”。")


def normalize_marker(value: str) -> str:
    return value.strip().upper()


def normalize_object_name(value: str) -> str:
    return " ".join(value.strip().split())
