from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Code.qwen35_benchmark import (
    DEFAULT_CASES,
    BenchmarkCase,
    collect_gpu_frequency_snapshot,
    exact_json_object,
    generation_settings_summary,
    perf_metrics_snapshot,
    primary_tokens_per_second,
    quality_score,
    reasoning_leak_flags,
    score_from_tool_contract,
    strip_reasoning_for_tool_scoring,
    structured_output_regex_for_case,
    summarize_model_results,
    processed_tool_contract_score,
    tool_contract_score,
)
from local_safety_assistant.config import MODEL_ALIASES


class Qwen35BenchmarkConfigTest(unittest.TestCase):
    def test_qwen35_4b_alias_exists(self) -> None:
        self.assertIn("qwen35-4b", MODEL_ALIASES)
        self.assertEqual(MODEL_ALIASES["qwen35-4b"].path.name, "Qwen3.5-4B-int4-ov")

    def test_prompt_suite_covers_required_groups(self) -> None:
        groups = {case.group for case in DEFAULT_CASES}

        self.assertIn("short_safety_qa", groups)
        self.assertIn("long_form_safety_rule_explanation", groups)
        self.assertIn("strict_instruction_following", groups)
        self.assertIn("json_only_structured_output", groups)
        self.assertIn("marker_tool_load_rules", groups)
        self.assertIn("marker_tool_edit_rules", groups)
        self.assertIn("planned_structured_tool_envelope", groups)
        self.assertIn("refusal_safety_boundary", groups)
        self.assertIn("vision_image", groups)
        self.assertIn("vision_video", groups)


class Qwen35BenchmarkScoringTest(unittest.TestCase):
    def test_exact_json_rejects_extra_prose(self) -> None:
        self.assertIsNone(exact_json_object('{"risk":"door"} trailing'))
        self.assertEqual(exact_json_object('{"risk":"door"}'), {"risk": "door"})

    def test_tool_marker_scoring_requires_exact_tool_line(self) -> None:
        case = BenchmarkCase("T", "marker", "prompt", 16, "tool_marker_load_rules")

        score = tool_contract_score(case, "TOOL:load_rules")

        self.assertIsNotNone(score)
        assert score is not None
        self.assertTrue(score["valid_tool_name"])
        self.assertTrue(score["valid_json_or_marker"])
        self.assertTrue(score["no_extra_prose"])
        self.assertEqual(quality_score(case, "TOOL:load_rules", None), 2)

    def test_structured_tool_json_scoring(self) -> None:
        case = BenchmarkCase("T", "json_tool", "prompt", 32, "tool_json_load_rules")
        text = '{"type":"tool_call","name":"load_rules","arguments":{}}'

        score = tool_contract_score(case, text)

        self.assertIsNotNone(score)
        assert score is not None
        self.assertTrue(all(score.values()))
        self.assertEqual(quality_score(case, text, None), 2)

    def test_structured_tool_json_requires_tool_call_type(self) -> None:
        case = BenchmarkCase("T", "json_tool", "prompt", 32, "tool_json_load_rules")

        score = tool_contract_score(case, '{"name":"load_rules","arguments":{}}')

        self.assertIsNotNone(score)
        assert score is not None
        self.assertFalse(score["valid_json_or_marker"])
        self.assertEqual(quality_score(case, '{"name":"load_rules","arguments":{}}', None), 1)

    def test_refusal_scores_direct_write_claim_as_failure(self) -> None:
        case = BenchmarkCase("T", "refusal", "prompt", 32, "refusal")

        self.assertEqual(quality_score(case, "规则文件已修改并保存。", None), 0)
        self.assertEqual(quality_score(case, "不能直接写入，需要通过 edit_rules 验证。", None), 2)


class Qwen35ReasoningMitigationTest(unittest.TestCase):
    def test_strip_reasoning_keeps_tool_after_closed_think(self) -> None:
        raw = "The user wants a tool call.\n</think>\n\nTOOL:edit_rules\n"

        self.assertEqual(strip_reasoning_for_tool_scoring(raw), "TOOL:edit_rules")

    def test_strip_reasoning_keeps_tool_after_unclosed_think(self) -> None:
        raw = "<think>\nThe user wants a tool call.\nTOOL:load_rules\n"

        self.assertEqual(strip_reasoning_for_tool_scoring(raw), "TOOL:load_rules")

    def test_strip_reasoning_extracts_json_tool_from_prose(self) -> None:
        raw = 'Thinking Process: choose the rule reader.\n{"type":"tool_call","name":"load_rules","arguments":{}}'

        self.assertEqual(
            strip_reasoning_for_tool_scoring(raw),
            '{"type":"tool_call","name":"load_rules","arguments":{}}',
        )

    def test_reasoning_leak_flags_cover_observed_prefixes(self) -> None:
        flags = reasoning_leak_flags("Thinking Process: inspect the request")

        self.assertTrue(flags["reasoning_prefix"])
        self.assertFalse(flags["think_tag"])

    def test_processed_tool_score_uses_raw_direct_write_claim(self) -> None:
        case = BenchmarkCase("T", "marker", "prompt", 16, "tool_marker_edit_rules")
        raw = "规则文件已修改并保存。\nTOOL:edit_rules"
        processed = "TOOL:edit_rules"

        score = processed_tool_contract_score(case, processed_text=processed, raw_text=raw)

        self.assertIsNotNone(score)
        assert score is not None
        self.assertFalse(score["no_direct_rule_file_write_claim"])
        self.assertEqual(score_from_tool_contract(score, None), 1)

    def test_structured_tool_regexes_match_expected_contracts(self) -> None:
        marker = BenchmarkCase("T", "marker", "prompt", 16, "tool_marker_load_rules")
        tool_json = BenchmarkCase("T", "json_tool", "prompt", 32, "tool_json_load_rules")

        self.assertEqual(structured_output_regex_for_case(marker), "TOOL:load_rules")
        self.assertIn('"tool_call"', structured_output_regex_for_case(tool_json))
        self.assertIn("structured_output_config", generation_settings_summary(tool_json, "x"))


class Qwen35BenchmarkMetricsTest(unittest.TestCase):
    def test_perf_metrics_snapshot_extracts_official_generated_token_metrics(self) -> None:
        class Pair:
            def __init__(self, mean: float, std: float = 0.0) -> None:
                self.mean = mean
                self.std = std

        class Metrics:
            def get_throughput(self) -> Pair:
                return Pair(27.5, 0.2)

            def get_tpot(self) -> Pair:
                return Pair(36.4, 0.1)

            def get_ttft(self) -> Pair:
                return Pair(280.0, 4.0)

            def get_generate_duration(self) -> Pair:
                return Pair(3200.0, 5.0)

            def get_tokenization_duration(self) -> Pair:
                return Pair(8.0)

            def get_detokenization_duration(self) -> Pair:
                return Pair(5.0)

            def get_inference_duration(self) -> Pair:
                return Pair(3100.0)

            def get_ipot(self) -> Pair:
                return Pair(35.0)

            def get_chat_template_duration(self) -> Pair:
                return Pair(-1.0)

            def get_num_generated_tokens(self) -> int:
                return 88

            def get_num_input_tokens(self) -> int:
                return 142

            def get_load_time(self) -> float:
                return 0.0

        class Result:
            perf_metrics = Metrics()

        snapshot = perf_metrics_snapshot(Result())

        self.assertTrue(snapshot["available"])
        self.assertEqual(snapshot["num_generated_tokens"], 88)
        self.assertEqual(snapshot["num_input_tokens"], 142)
        self.assertEqual(snapshot["throughput_tokens_per_second"], 27.5)
        self.assertEqual(snapshot["tpot_ms"], 36.4)
        self.assertIsNone(snapshot["metrics"]["chat_template_duration"]["mean"])

    def test_primary_tokens_per_second_prefers_official_metrics_over_stream_chunks(self) -> None:
        value, source = primary_tokens_per_second(27.5, 19.0)

        self.assertEqual(value, 27.5)
        self.assertEqual(source, "openvino_perf_metrics")

        fallback_value, fallback_source = primary_tokens_per_second(None, 19.0)

        self.assertEqual(fallback_value, 19.0)
        self.assertEqual(fallback_source, "stream_chunks")

    def test_summary_separates_official_tokens_and_stream_chunks(self) -> None:
        result = {
            "failure": None,
            "quality_score": 2,
            "tool_score": None,
            "bug_flags": {},
            "tokens_per_second": 27.5,
            "official_tokens_per_second": 27.5,
            "official_tpot_ms": 36.4,
            "stream_chunks_per_second": 19.0,
            "ttft_seconds": 0.28,
            "generation_seconds": 3.2,
            "total_seconds": 3.5,
        }

        summary = summarize_model_results([result])

        self.assertEqual(summary["mean_tokens_per_second"], 27.5)
        self.assertEqual(summary["mean_official_tokens_per_second"], 27.5)
        self.assertEqual(summary["mean_tpot_ms"], 36.4)
        self.assertEqual(summary["mean_stream_chunks_per_second"], 19.0)

    def test_gpu_frequency_snapshot_reads_sysfs_style_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            card_dir = Path(tmp) / "sys/devices/pci0000:00/0000:00:02.0/drm/card1"
            card_dir.mkdir(parents=True)
            (card_dir / "gt_cur_freq_mhz").write_text("1200\n", encoding="utf-8")
            (card_dir / "gt_act_freq_mhz").write_text("1188\n", encoding="utf-8")
            (card_dir / "gt_RP0_freq_mhz").write_text("2000\n", encoding="utf-8")

            snapshot = collect_gpu_frequency_snapshot(
                (str(Path(tmp) / "sys/devices/pci*/**/drm/card*/gt_cur_freq_mhz"),)
            )

        self.assertTrue(snapshot["available"])
        frequencies = snapshot["devices"][0]["frequencies_mhz"]
        self.assertEqual(frequencies["gt_cur_freq_mhz"], 1200)
        self.assertEqual(frequencies["gt_act_freq_mhz"], 1188)
        self.assertEqual(frequencies["gt_RP0_freq_mhz"], 2000)


if __name__ == "__main__":
    unittest.main()
