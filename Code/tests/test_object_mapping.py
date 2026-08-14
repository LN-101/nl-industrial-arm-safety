from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from local_safety_assistant.object_mapping import (
    ObjectMappingValidationError,
    get_object_mapping,
    load_object_mapping_document,
    resolve_object_grasp_target,
    update_object_mapping,
    validate_object_mapping_document,
    write_object_mapping_document,
)


def sample_mapping_document() -> dict:
    return {
        "version": 1,
        "markers": {
            "A": {"object": "红色方块", "enabled": True},
            "B": {"object": "蓝色圆柱", "enabled": True},
            "C": {"object": "扳手", "enabled": True},
            "D": {"object": "空位", "enabled": True},
        },
    }


class ObjectMappingTest(unittest.TestCase):
    def test_update_object_mapping_increments_version_and_preserves_entry_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "object_mapping.json"
            write_object_mapping_document(mapping_path, sample_mapping_document())

            updated = update_object_mapping(mapping_path, marker="a", object_name="  夹具  ")

            self.assertEqual(updated["version"], 2)
            self.assertEqual(updated["markers"]["A"]["object"], "夹具")
            self.assertTrue(updated["markers"]["A"]["enabled"])
            loaded = load_object_mapping_document(mapping_path)
            self.assertEqual(loaded, updated)

    def test_update_object_mapping_same_value_keeps_version_and_file_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "object_mapping.json"
            write_object_mapping_document(mapping_path, sample_mapping_document())
            before_text = mapping_path.read_text(encoding="utf-8")

            updated = update_object_mapping(mapping_path, marker="a", object_name=" 红色方块 ")

            self.assertEqual(updated["version"], 1)
            self.assertEqual(updated["markers"]["A"]["object"], "红色方块")
            self.assertEqual(mapping_path.read_text(encoding="utf-8"), before_text)

    def test_get_object_mapping_returns_marker_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "object_mapping.json"
            write_object_mapping_document(mapping_path, sample_mapping_document())

            mapping = get_object_mapping(mapping_path, marker="b")

            self.assertEqual(mapping["version"], 1)
            self.assertEqual(mapping["marker"], "B")
            self.assertEqual(mapping["object"], "蓝色圆柱")
            loaded = load_object_mapping_document(mapping_path)
            self.assertEqual(loaded["version"], 1)

    def test_resolve_object_grasp_target_accepts_marker_and_object_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "object_mapping.json"
            write_object_mapping_document(mapping_path, sample_mapping_document())

            marker_target = resolve_object_grasp_target(mapping_path, marker="a")
            object_target = resolve_object_grasp_target(mapping_path, object_name=" 扳手 ")

            self.assertEqual(marker_target["marker"], "A")
            self.assertEqual(marker_target["object"], "红色方块")
            self.assertEqual(marker_target["target_source"], "marker")
            self.assertEqual(object_target["marker"], "C")
            self.assertEqual(object_target["object"], "扳手")
            self.assertEqual(object_target["target_source"], "object_name")
            self.assertEqual(load_object_mapping_document(mapping_path)["version"], 1)

    def test_resolve_object_grasp_target_rejects_ambiguous_or_disabled_object(self) -> None:
        document = sample_mapping_document()
        document["markers"]["B"]["object"] = "红色方块"
        document["markers"]["C"]["enabled"] = False
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "object_mapping.json"
            write_object_mapping_document(mapping_path, document)

            with self.assertRaises(ObjectMappingValidationError):
                resolve_object_grasp_target(mapping_path, object_name="红色方块")
            with self.assertRaises(ObjectMappingValidationError):
                resolve_object_grasp_target(mapping_path, marker="C")
            with self.assertRaises(ObjectMappingValidationError):
                resolve_object_grasp_target(mapping_path, object_name="扳手")

    def test_update_object_mapping_rejects_unknown_marker_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "object_mapping.json"
            write_object_mapping_document(mapping_path, sample_mapping_document())

            with self.assertRaises(ObjectMappingValidationError):
                update_object_mapping(mapping_path, marker="E", object_name="托盘")

            loaded = load_object_mapping_document(mapping_path)
            self.assertEqual(loaded["version"], 1)
            self.assertNotIn("E", loaded["markers"])

    def test_update_object_mapping_rejects_empty_object_name_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "object_mapping.json"
            write_object_mapping_document(mapping_path, sample_mapping_document())

            with self.assertRaises(ObjectMappingValidationError):
                update_object_mapping(mapping_path, marker="A", object_name="   ")

            loaded = load_object_mapping_document(mapping_path)
            self.assertEqual(loaded["version"], 1)
            self.assertEqual(loaded["markers"]["A"]["object"], "红色方块")

    def test_validate_object_mapping_rejects_invalid_schema(self) -> None:
        invalid_documents = [
            [],
            {"version": "1", "markers": {}},
            {"version": True, "markers": {}},
            {"version": 1, "markers": []},
            {"version": 1, "markers": {"E": {"object": "托盘"}}},
            {"version": 1, "markers": {"A": {"object": ""}}},
            {"version": 1, "markers": {"A": {"object": "托盘", "enabled": "yes"}}},
        ]

        for document in invalid_documents:
            with self.subTest(document=json.dumps(document, ensure_ascii=False)):
                with self.assertRaises(ObjectMappingValidationError):
                    validate_object_mapping_document(document)


if __name__ == "__main__":
    unittest.main()
