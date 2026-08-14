from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_safety_assistant.stack.safety_batch import (
    DEFAULT_CASES,
    KeywordGroup,
    TARGET_SAMPLE_RATE,
    build_parser,
    convert_to_mono_16k,
    evaluate_response,
)


class SafetyBatchEvaluationTest(unittest.TestCase):
    def test_default_cases_include_current_rule_query_and_rules_path(self) -> None:
        args = build_parser().parse_args([])

        self.assertEqual(args.count, len(DEFAULT_CASES))
        self.assertTrue(any(case.case_id == "S31" and "当前安全规则" in case.utterance for case in DEFAULT_CASES))
        self.assertTrue(any(case.case_id == "S32" and "未知物体靠近限速规则" in case.utterance for case in DEFAULT_CASES))
        self.assertGreaterEqual(args.max_new_tokens, 160)
        self.assertEqual(args.rules.name, "safety_rules.example.json")

    def test_response_passes_when_each_keyword_group_matches(self) -> None:
        result = evaluate_response(
            "立即停止机械臂，并确认安全区域后再复位。",
            (
                KeywordGroup("停机", ("停止", "急停")),
                KeywordGroup("确认", ("确认", "检查")),
            ),
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.matched["停机"], "停止")
        self.assertEqual(result.matched["确认"], "确认")

    def test_response_reports_missing_keyword_groups(self) -> None:
        result = evaluate_response(
            "请联系管理员。",
            (
                KeywordGroup("停机", ("停止", "急停")),
                KeywordGroup("限速", ("限速", "减速")),
            ),
        )

        self.assertFalse(result.passed)
        self.assertIn("停机", result.missing)
        self.assertIn("限速", result.missing)


class SafetyBatchAudioTest(unittest.TestCase):
    def test_convert_to_mono_16k_resamples_stereo_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.wav"
            target = Path(temp_dir) / "target.wav"
            sample_rate = 44100
            samples = np.zeros((sample_rate, 2), dtype=np.float32)
            sf.write(source, samples, sample_rate, subtype="PCM_16")

            audio_seconds, _, source_sample_rate = convert_to_mono_16k(source, target)
            converted, converted_rate = sf.read(target, dtype="float32", always_2d=False)

            self.assertEqual(source_sample_rate, sample_rate)
            self.assertEqual(converted_rate, TARGET_SAMPLE_RATE)
            self.assertAlmostEqual(audio_seconds, 1.0, places=2)
            self.assertEqual(converted.ndim, 1)


if __name__ == "__main__":
    unittest.main()
