from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_safety_assistant.confirmation import (
    ACTION_OBJECT_GRASP_EXECUTION,
    ACTION_OBJECT_MAPPING_UPDATE,
    ACTION_RULE_EDIT,
    ACTION_SPEED_CHANGE,
    build_object_grasp_execution_confirmation,
)
from local_safety_assistant.stack.asr import AsrResult
from local_safety_assistant.stack.cli import (
    build_endpointing_config,
    build_microphone_config,
    build_moss_tts_config,
    build_pipeline,
    build_parser,
    build_tts_engine,
    publish_or_print_ros2_plan,
    should_play_tts,
    parse_microphone_device,
)
from local_safety_assistant.stack.config import (
    DEFAULT_MOSS_YANGMI_PROMPT_AUDIO,
    DEFAULT_SYSTEM_PROMPT,
    MeloTtsConfig,
    MossTtsConfig,
    PiperTtsConfig,
)
from local_safety_assistant.stack.devices import build_device_plan, normalize_devices
from local_safety_assistant.stack.llm import LlmResult, build_prompt, strip_thinking_text
from local_safety_assistant.stack.microphone import (
    CapturedUtterance,
    EndpointingConfig,
    EnergyEndpointDetector,
    segment_frames,
    write_utterance_wav,
)
from local_safety_assistant.stack.pipeline import (
    CAPABILITY_RESPONSE,
    RULE_EDIT_STRATEGY_ONE_PASS,
    RULE_EDIT_STRATEGY_TWO_PASS,
    VISION_ANALYSIS_FALLBACK_RESPONSE,
    VISION_ANALYSIS_SYSTEM_PROMPT,
    VoicePipeline,
    build_vision_analysis_prompt,
    build_agent_router_prompt,
    convert_traditional_to_simplified,
    extract_object_mapping_query,
    extract_object_mapping_update,
    extract_arm_deceleration_target_percent,
    extract_object_grasp_target,
    extract_rule_patch_payload,
    extract_rule_replacement,
    parse_agent_route_decision,
    parse_agent_tool_request,
    parse_agent_final_response,
    sanitize_spoken_response,
    is_unusable_vision_analysis_response,
    normalize_asr_text,
    routes_to_rule_editor,
    should_edit_rules,
    should_grasp_object,
    should_analyze_vision,
    should_guide_workspace_snapshot,
    should_ignore_asr_noise,
    should_query_object_mapping,
    should_query_object_mapping_table,
    should_query_arm_runtime,
    should_request_arm_deceleration,
    should_update_object_mapping,
    should_reject_rule_authoring,
    should_read_rules,
    strip_route_marker,
)
from local_safety_assistant.stack.ros2_bridge import (
    DEFAULT_ESTOP_REQUEST_TOPIC,
    DEFAULT_GOAL_TOPIC,
    DEFAULT_RESPONSE_TOPIC,
    DEFAULT_TRANSCRIPT_TOPIC,
    ROS2_BOOL,
    ROS2_POINT,
    ROS2_STRING,
    Ros2BridgeConfig,
    Ros2VoiceBridge,
    build_voice_ros2_plan,
    detect_estop_command,
    detect_goal_command,
    sync_estop_plans_to_arm_rules,
)
from local_safety_assistant.stack.tts import (
    MeloTtsBridge,
    MossTtsBridge,
    PiperTtsBridge,
    TtsResult,
    prepare_melotts_text,
)
from local_safety_assistant.stack.vision import (
    Ros2TriggerVisionSnapshotProvider,
    VisionImageArtifact,
    format_vision_snapshot_error,
    snapshot_from_trigger_message,
)
from local_safety_assistant.stack.vision_node import image_message_to_rgb_array, snapshot_payload
from local_safety_assistant.object_mapping import load_object_mapping_document, write_object_mapping_document
from local_safety_assistant.arm_rules import load_arm_rule_document, write_arm_rule_document
from local_safety_assistant.rules import load_rule_document, write_rule_document


class DevicePlanTest(unittest.TestCase):
    def test_normalize_devices_deduplicates_and_filters(self) -> None:
        self.assertEqual(normalize_devices(["gpu", "CPU", "GPU", "unknown", "npu"]), ("GPU", "CPU", "NPU"))

    def test_device_plan_spreads_work_across_available_devices(self) -> None:
        plan = build_device_plan(["CPU", "GPU", "NPU"])

        self.assertEqual(plan.asr.run_order, ("GPU", "CPU", "NPU"))
        self.assertEqual(plan.llm_small.run_order, ("GPU", "CPU", "NPU"))
        self.assertEqual(plan.llm_large.run_order, ("CPU", "GPU", "NPU"))
        self.assertEqual(plan.tts.run_order, ("CPU", "GPU"))
        self.assertEqual(plan.tts_bert.run_order, ("CPU", "GPU", "NPU"))

    def test_asr_device_plan_uses_cpu_first_when_gpu_is_unavailable(self) -> None:
        plan = build_device_plan(["CPU", "NPU"])

        self.assertEqual(plan.asr.run_order, ("CPU", "NPU"))

    def test_device_plan_falls_back_to_cpu_when_no_known_devices(self) -> None:
        plan = build_device_plan([])

        for stage in plan.stages():
            self.assertEqual(stage.selected, "CPU")


class MeloTtsBridgeTest(unittest.TestCase):
    def test_build_command_contains_expected_devices_and_flags(self) -> None:
        config = MeloTtsConfig(
            binary=Path("/tmp/meloTTS_ov"),
            model_dir=Path("/tmp/ov_models"),
            output_dir=Path("/tmp/out"),
            language="ZH",
            speed=0.9,
            quantize=True,
            disable_bert=False,
            disable_nf=True,
        )
        bridge = MeloTtsBridge(config, tts_device="GPU", bert_device="NPU", denoise_device="CPU")

        command = bridge.build_command(Path("/tmp/input.txt"), Path("/tmp/out/voice"))

        self.assertIn("--tts_device", command)
        self.assertEqual(command[command.index("--tts_device") + 1], "GPU")
        self.assertEqual(command[command.index("--bert_device") + 1], "NPU")
        self.assertEqual(command[command.index("--disable_nf") + 1], "true")
        self.assertEqual(command[command.index("--language") + 1], "ZH")

    def test_build_command_uses_slower_default_speed(self) -> None:
        config = MeloTtsConfig(
            binary=Path("/tmp/meloTTS_ov"),
            model_dir=Path("/tmp/ov_models"),
            output_dir=Path("/tmp/out"),
        )
        bridge = MeloTtsBridge(config)

        command = bridge.build_command(Path("/tmp/input.txt"), Path("/tmp/out/voice"))

        self.assertEqual(command[command.index("--speed") + 1], "0.8")

    def test_synthesize_requires_binary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = MeloTtsConfig(
                binary=Path(temp_dir) / "missing_melo",
                model_dir=Path(temp_dir),
                output_dir=Path(temp_dir) / "out",
            )
            bridge = MeloTtsBridge(config)

            with self.assertRaises(FileNotFoundError):
                bridge.synthesize("hello")

    def test_prepare_melotts_text_expands_acronyms_and_pauses(self) -> None:
        text = "ROS 控制器报警时进入安全保持并通知 CPU/GPU 侧"

        prepared = prepare_melotts_text(text)

        self.assertEqual(prepared, "R O S 控制器报警时，进入安全保持，并通知 C P U 和 G P U 侧。")

    def test_prepare_melotts_text_handles_adjacent_acronyms(self) -> None:
        prepared = prepare_melotts_text("ROS控制器报警后请确认CPU/GPU侧状态")

        self.assertEqual(prepared, "R O S控制器报警后，请确认C P U 和 G P U侧状态。")

    def test_prepare_melotts_text_preserves_urgent_final_tone(self) -> None:
        prepared = prepare_melotts_text("机械臂急停")

        self.assertEqual(prepared, "机械臂急停！")

    def test_prepare_melotts_text_handles_punctuation_only_text(self) -> None:
        prepared = prepare_melotts_text(",,,")

        self.assertEqual(prepared, "")


class PiperTtsBridgeTest(unittest.TestCase):
    def test_default_silence_scale_uses_tuned_candidate(self) -> None:
        self.assertEqual(PiperTtsConfig().silence_scale, 1.0)

    def test_build_command_contains_expected_runner_and_model_paths(self) -> None:
        config = PiperTtsConfig(
            python=Path("/tmp/tts_eval_env/bin/python"),
            runner=Path("/tmp/piper_tts.py"),
            model_dir=Path("/tmp/piper_model"),
            espeak_data_dir=Path("/tmp/espeak-ng-data"),
            output_dir=Path("/tmp/out"),
            speed=1.1,
            silence_scale=0.6,
            threads=2,
        )
        bridge = PiperTtsBridge(config)

        command = bridge.build_command(Path("/tmp/input.txt"), Path("/tmp/out/voice.wav"))

        self.assertEqual(command[0], "/tmp/tts_eval_env/bin/python")
        self.assertEqual(command[1], "/tmp/piper_tts.py")
        self.assertEqual(command[command.index("--model-dir") + 1], "/tmp/piper_model")
        self.assertEqual(command[command.index("--espeak-data-dir") + 1], "/tmp/espeak-ng-data")
        self.assertEqual(command[command.index("--speed") + 1], "1.1")
        self.assertEqual(command[command.index("--silence-scale") + 1], "0.6")
        self.assertEqual(command[command.index("--threads") + 1], "2")

    def test_synthesize_invokes_runner_with_prepared_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            python = root / "python"
            runner = root / "piper_tts.py"
            model_dir = root / "model"
            espeak_data_dir = root / "espeak-ng-data"
            output_dir = root / "out"
            python.touch()
            runner.touch()
            model_dir.mkdir()
            espeak_data_dir.mkdir()
            config = PiperTtsConfig(
                python=python,
                runner=runner,
                model_dir=model_dir,
                espeak_data_dir=espeak_data_dir,
                output_dir=output_dir,
            )
            bridge = PiperTtsBridge(config)
            observed: dict[str, str] = {}

            def fake_run(command, cwd, check, capture_output, text, timeout):
                input_path = Path(command[command.index("--text-file") + 1])
                output_path = Path(command[command.index("--output-file") + 1])
                observed["input_text"] = input_path.read_text(encoding="utf-8")
                observed["silence_scale"] = command[command.index("--silence-scale") + 1]
                output_path.write_bytes(b"RIFF")
                return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

            with patch("local_safety_assistant.stack.tts.subprocess.run", side_effect=fake_run):
                result = bridge.synthesize("ROS控制器报警后请确认CPU/GPU侧状态", output_name="piper_test")

        self.assertEqual(result.text, "R O S控制器报警后，请确认C P U 和 G P U侧状态。")
        self.assertEqual(observed["input_text"], "R O S控制器报警后，请确认C P U 和 G P U侧状态。\n")
        self.assertEqual(observed["silence_scale"], "1.0")
        self.assertEqual(result.audio_paths, (output_dir / "piper_test.wav",))


class MossTtsBridgeTest(unittest.TestCase):
    def test_default_config_uses_builtin_xiaoyu_voice_and_streaming_decode(self) -> None:
        config = MossTtsConfig(
            executable=Path("/tmp/moss-tts-nano"),
            source_dir=Path("/tmp/MOSS-TTS-Nano"),
            model_dir=Path("/tmp/models/tts"),
            output_dir=Path("/tmp/out"),
        )
        bridge = MossTtsBridge(config)

        command = bridge.build_command(Path("/tmp/input.txt"), Path("/tmp/out/voice.wav"))

        self.assertEqual(command[:4], ("/tmp/moss-tts-nano", "generate", "--backend", "onnx"))
        self.assertNotIn("--cpu-list", command)
        self.assertEqual(command[command.index("--voice") + 1], "Xiaoyu")
        self.assertNotIn("--prompt-speech", command)
        self.assertEqual(command[command.index("--sample-mode") + 1], "fixed")
        self.assertEqual(command[command.index("--realtime-streaming-decode") + 1], "1")
        self.assertEqual(command[command.index("--max-new-frames") + 1], "375")
        self.assertEqual(command[command.index("--audio-repetition-penalty") + 1], "1.2")

    def test_explicit_yangmi_prompt_audio_mode_passes_prompt_speech(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["tts", "--text", "hello", "--moss-use-yangmi-prompt-audio"])
        config = build_moss_tts_config(args)
        bridge = MossTtsBridge(config)

        command = bridge.build_command(Path("/tmp/input.txt"), Path("/tmp/out/voice.wav"))

        self.assertEqual(config.prompt_audio, DEFAULT_MOSS_YANGMI_PROMPT_AUDIO)
        self.assertEqual(command[command.index("--voice") + 1], "Xiaoyu")
        self.assertEqual(command[command.index("--prompt-speech") + 1], str(DEFAULT_MOSS_YANGMI_PROMPT_AUDIO))

    def test_synthesize_invokes_moss_cli_with_prepared_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executable = root / "moss-tts-nano"
            source_dir = root / "MOSS-TTS-Nano"
            model_dir = root / "models" / "tts"
            prompt_audio = source_dir / "assets" / "audio" / "zh_11.wav"
            output_dir = root / "out"
            executable.parent.mkdir(parents=True, exist_ok=True)
            executable.touch()
            prompt_audio.parent.mkdir(parents=True, exist_ok=True)
            prompt_audio.touch()
            model_dir.mkdir(parents=True)
            config = MossTtsConfig(
                executable=executable,
                source_dir=source_dir,
                model_dir=model_dir,
                output_dir=output_dir,
                prompt_audio=prompt_audio,
            )
            bridge = MossTtsBridge(config)
            observed: dict[str, str] = {}

            def fake_run(command, cwd, check, capture_output, text, timeout):
                input_path = Path(command[command.index("--text-file") + 1])
                output_path = Path(command[command.index("--output-audio-path") + 1])
                observed["input_text"] = input_path.read_text(encoding="utf-8")
                observed["cwd"] = str(cwd)
                observed["streaming"] = command[command.index("--realtime-streaming-decode") + 1]
                output_path.write_bytes(b"RIFF")
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with patch("local_safety_assistant.stack.tts.subprocess.run", side_effect=fake_run):
                result = bridge.synthesize("ROS控制器报警后请确认CPU/GPU侧状态", output_name="moss_test")

        self.assertEqual(observed["input_text"], "R O S控制器报警后，请确认C P U 和 G P U侧状态。\n")
        self.assertEqual(observed["cwd"], str(source_dir))
        self.assertEqual(observed["streaming"], "1")
        self.assertEqual(result.audio_paths, (output_dir / "moss_test.wav",))

    def test_synthesize_resolves_relative_output_dir_against_cwd_independently(self) -> None:
        with tempfile.TemporaryDirectory(dir=".runtime") as temp_dir:
            root = Path(temp_dir)
            executable = root / "moss-tts-nano"
            source_dir = root / "MOSS-TTS-Nano"
            model_dir = root / "models" / "tts"
            prompt_audio = source_dir / "assets" / "audio" / "zh_11.wav"
            current_dir = Path.cwd()
            relative_output_dir = root.relative_to(current_dir) / "relative_moss_out"
            executable.parent.mkdir(parents=True, exist_ok=True)
            executable.touch()
            prompt_audio.parent.mkdir(parents=True, exist_ok=True)
            prompt_audio.touch()
            model_dir.mkdir(parents=True)
            config = MossTtsConfig(
                executable=executable,
                source_dir=source_dir,
                model_dir=model_dir,
                output_dir=relative_output_dir,
                prompt_audio=prompt_audio,
            )
            bridge = MossTtsBridge(config)

            def fake_run(command, cwd, check, capture_output, text, timeout):
                output_path = Path(command[command.index("--output-audio-path") + 1])
                output_path.write_bytes(b"RIFF")
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with patch("local_safety_assistant.stack.tts.subprocess.run", side_effect=fake_run):
                result = bridge.synthesize("你好", output_name="moss_relative")

        self.assertTrue(result.audio_paths[0].is_absolute())
        self.assertEqual(result.audio_paths[0], (current_dir / relative_output_dir / "moss_relative.wav").resolve())

    def test_build_tts_engine_defaults_to_moss_and_keeps_melo_switch(self) -> None:
        parser = build_parser()

        default_args = parser.parse_args(["tts", "--text", "hello"])
        melo_args = parser.parse_args(["tts", "--tts-engine", "melo", "--text", "hello"])

        self.assertIsInstance(build_tts_engine(default_args), MossTtsBridge)
        self.assertIsInstance(build_tts_engine(melo_args, plan=build_device_plan(["CPU"])), MeloTtsBridge)


class FakeAsr:
    def __init__(self, text: str | None = None) -> None:
        self.text = text

    def transcribe_wav(self, audio_path: Path) -> AsrResult:
        return AsrResult(
            text=self.text if self.text is not None else f"transcribed {audio_path.name}",
            model="fake-asr",
            device="CPU",
            audio_seconds=1.0,
            load_seconds=0.0,
            inference_seconds=0.01,
        )


class FakeLlm:
    def __init__(
        self,
        response: str | list[str] | tuple[str, ...] | None = None,
        model: str = "fake-llm",
        vision_response: str = "画面中未发现明显人员侵入，工位整体可见。请操作员继续确认机械臂周边无障碍物。",
    ) -> None:
        self.response = response
        self.responses = list(response) if isinstance(response, (list, tuple)) else None
        self.model = model
        self.calls: list[str] = []
        self.image_calls: list[tuple[str, Path, str | None]] = []
        self.vision_response = vision_response

    def generate(self, user_text: str) -> LlmResult:
        self.calls.append(user_text)
        if self.responses is not None and self.responses:
            text = self.responses.pop(0)
        else:
            text = self.response if isinstance(self.response, str) else None
        text = text if text is not None else f"reply to {user_text}"
        return LlmResult(
            prompt=user_text,
            text=text,
            model=self.model,
            device="NPU",
            load_seconds=0.0,
            inference_seconds=0.02,
        )

    def generate_with_image(
        self,
        user_text: str,
        image_path: Path,
        *,
        max_new_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> LlmResult:
        self.image_calls.append((user_text, image_path, system_prompt))
        return LlmResult(
            prompt=user_text,
            text=self.vision_response,
            model=self.model,
            device="GPU",
            load_seconds=0.0,
            inference_seconds=0.03,
        )


class FakeTts:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def synthesize(self, text: str, *, output_name: str | None = None) -> TtsResult:
        self.calls.append(text)
        return TtsResult(
            text=text,
            audio_paths=(Path("/tmp/fake.wav"),),
            command=("fake-tts",),
            elapsed_seconds=0.03,
            stdout="",
            stderr="",
        )


class FakeVisionSnapshotProvider:
    def __init__(self, artifact: VisionImageArtifact | None = None, error: Exception | None = None) -> None:
        self.artifact = artifact
        self.error = error
        self.calls = 0

    def capture_snapshot(self) -> VisionImageArtifact:
        self.calls += 1
        if self.error is not None:
            raise self.error
        if self.artifact is None:
            raise RuntimeError("missing fake artifact")
        return self.artifact


def complex_rule_document() -> dict:
    return {
        "version": 1,
        "rules": [
            {
                "id": "stop_on_person_intrusion",
                "name": "人员侵入停机规则",
                "enabled": True,
                "description": "人员进入保护距离内时立即停止机械臂运动。",
                "conditions": {"person_distance_m": {"lt": 0.8}},
                "action": {"type": "stop_motion", "severity": "critical", "requires_reset": True},
            },
            {
                "id": "slow_near_unknown_object",
                "name": "未知物体靠近限速规则",
                "enabled": True,
                "description": "未知物体靠近抓取路径时降低速度。",
                "conditions": {"unknown_object_distance_m": {"lt": 0.25}},
                "action": {"type": "limit_speed", "max_speed_scale": 0.25},
            },
            {
                "id": "stop_on_guard_door_open",
                "name": "防护门打开停机规则",
                "enabled": True,
                "description": "防护门打开时停止机械臂并要求复位确认。",
                "conditions": {"guard_door_open": {"eq": True}},
                "action": {"type": "stop_motion", "severity": "critical", "requires_reset": True},
            },
            {
                "id": "stop_on_light_curtain_blocked",
                "name": "安全光栅遮挡停机规则",
                "enabled": True,
                "description": "安全光栅被遮挡时停止机械臂。",
                "conditions": {"light_curtain_blocked": {"eq": True}},
                "action": {"type": "stop_motion", "severity": "critical", "requires_reset": True},
            },
            {
                "id": "hold_on_ros_controller_alarm",
                "name": "ROS 控制器报警安全保持规则",
                "enabled": True,
                "description": "ROS 控制器报警时进入安全保持并通知操作员。",
                "conditions": {"ros_controller_alarm": {"eq": True}},
                "action": {"type": "safety_hold", "notify_operator": True, "requires_manual_check": True},
            },
            {
                "id": "limit_speed_in_teach_mode",
                "name": "示教模式限速规则",
                "enabled": True,
                "description": "示教模式下最大速度限制到百分之十。",
                "conditions": {"teach_mode": {"eq": True}},
                "action": {"type": "limit_speed", "max_speed_scale": 0.1},
            },
        ],
    }


def object_mapping_document() -> dict:
    return {
        "version": 1,
        "markers": {
            "A": {"object": "红色方块", "enabled": True},
            "B": {"object": "蓝色圆柱", "enabled": True},
            "C": {"object": "扳手", "enabled": True},
            "D": {"object": "空位", "enabled": True},
        },
    }


class FakeTokenizer:
    def __init__(self) -> None:
        self.extra_context = None

    def apply_chat_template(self, messages, add_generation_prompt, chat_template="", tools=None, extra_context=None):
        self.extra_context = extra_context
        return f"templated:{messages[-1]['content']}"


class VoicePipelineTest(unittest.TestCase):
    def test_exact_estop_commands_bypass_llm(self) -> None:
        expected_responses = {
            "急停": "机械臂急停！",
            "解除急停": "已收到解除急停请求，请确认现场安全后再执行。",
        }

        for command, expected_response in expected_responses.items():
            with self.subTest(command=command):
                llm = FakeLlm()
                result = VoicePipeline(llm=llm).run_text_turn(command, synthesize=False)

                self.assertEqual(result.response_text, expected_response)
                self.assertEqual(llm.calls, [])

    def test_non_exact_estop_command_keeps_llm_route(self) -> None:
        llm = FakeLlm()

        VoicePipeline(llm=llm).run_text_turn("请解除急停", synthesize=False)

        self.assertIn("结构化工具路由器", llm.calls[0])

    def test_text_turn_can_skip_tts(self) -> None:
        llm = FakeLlm()
        pipeline = VoicePipeline(llm=llm, tts=FakeTts())

        result = pipeline.run_text_turn("hello", synthesize=False)

        self.assertEqual(result.input_text, "hello")
        self.assertEqual(result.response_text, "reply to hello")
        self.assertIsNone(result.tts)
        self.assertIn("结构化工具路由器", llm.calls[0])
        self.assertEqual(llm.calls[-1], "hello")

    def test_audio_turn_runs_asr_llm_tts(self) -> None:
        pipeline = VoicePipeline(asr=FakeAsr("测试语音"), llm=FakeLlm(), tts=FakeTts())

        result = pipeline.run_audio_file(Path("sample.wav"), synthesize=True)

        self.assertEqual(result.input_text, "测试语音")
        self.assertEqual(result.response_text, "reply to 测试语音")
        self.assertIsNotNone(result.asr)
        self.assertIsNotNone(result.tts)
        self.assertEqual(result.tts.audio_paths, (Path("/tmp/fake.wav"),))

    def test_audio_turn_corrects_common_asr_misrecognitions_before_llm(self) -> None:
        llm = FakeLlm()
        pipeline = VoicePipeline(asr=FakeAsr("机器臂气崩需要线速吗"), llm=llm)

        result = pipeline.run_audio_file(Path("sample.wav"), synthesize=False)

        self.assertEqual(result.input_text, "机械臂气泵需要限速吗")
        self.assertIn("机械臂气泵需要限速吗", llm.calls[0])
        self.assertEqual(llm.calls[-1], "机械臂气泵需要限速吗")
        self.assertEqual(result.response_text, "reply to 机械臂气泵需要限速吗")

    def test_vision_request_routes_to_snapshot_and_vlm(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "snapshot.jpg"
            image_path.write_bytes(b"fake image")
            artifact = VisionImageArtifact(
                image_path=image_path,
                source="/vision/capture_snapshot",
                metadata={"stamp": "2026-06-09T12:00:00", "frame_id": "camera_color"},
            )
            provider = FakeVisionSnapshotProvider(artifact)
            llm = FakeLlm(
                '{"type":"tool_call","name":"analyze_environment_vision","arguments":{}}',
                vision_response="画面显示工位可见，未发现明显人员进入机械臂工作区。请继续确认气泵附近无遮挡。",
            )
            pipeline = VoicePipeline(llm=llm, vision_snapshot_provider=provider)

            result = pipeline.run_text_turn("调用视觉，分析下当前工作环境", synthesize=False)

        self.assertEqual(provider.calls, 1)
        self.assertEqual(len(llm.image_calls), 1)
        vision_prompt, observed_image_path, system_prompt = llm.image_calls[0]
        self.assertEqual(observed_image_path, image_path)
        self.assertEqual(system_prompt, VISION_ANALYSIS_SYSTEM_PROMPT)
        self.assertIn("<image>", vision_prompt)
        self.assertIn("4 到 6 句简洁中文", vision_prompt)
        self.assertIn("第一句简述画面中的整体场景", vision_prompt)
        self.assertIn("第二句描述主要可见对象及其大致位置关系", vision_prompt)
        self.assertNotEqual(system_prompt, DEFAULT_SYSTEM_PROMPT)
        for forbidden in ("load_rules", "edit_rules", "物体映射", "结构化工具路由器"):
            self.assertNotIn(forbidden, system_prompt or "")
        self.assertIn("调用视觉分析下当前工作环境", vision_prompt)
        self.assertIn("工位可见", result.response_text)
        self.assertEqual(result.vision_artifacts, (artifact,))

    def test_vision_acknowledgement_response_uses_fallback_and_keeps_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "snapshot.jpg"
            image_path.write_bytes(b"fake image")
            artifact = VisionImageArtifact(image_path=image_path)
            provider = FakeVisionSnapshotProvider(artifact)
            llm = FakeLlm(
                '{"type":"tool_call","name":"analyze_environment_vision","arguments":{}}',
                vision_response="收到。",
            )
            pipeline = VoicePipeline(llm=llm, vision_snapshot_provider=provider)

            result = pipeline.run_text_turn("调用视觉，分析下当前工作环境", synthesize=False)

        self.assertEqual(provider.calls, 1)
        self.assertEqual(result.response_text, VISION_ANALYSIS_FALLBACK_RESPONSE)
        self.assertEqual(result.vision_artifacts, (artifact,))
        self.assertEqual(len(llm.image_calls), 1)

    def test_explicit_vision_request_uses_backstop_when_router_misses(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "snapshot.jpg"
            image_path.write_bytes(b"fake image")
            provider = FakeVisionSnapshotProvider(VisionImageArtifact(image_path=image_path))
            llm = FakeLlm('{"type":"final","content":"我现在看不到画面。"}')
            pipeline = VoicePipeline(llm=llm, vision_snapshot_provider=provider)

            result = pipeline.run_text_turn("请视觉分析当前画面", synthesize=False)

        self.assertEqual(provider.calls, 1)
        self.assertEqual(len(llm.image_calls), 1)
        self.assertNotIn("看不到", result.response_text)
        self.assertTrue(result.vision_artifacts)

    def test_vision_request_without_provider_returns_unavailable(self) -> None:
        llm = FakeLlm('{"type":"tool_call","name":"analyze_environment_vision","arguments":{}}')
        pipeline = VoicePipeline(llm=llm)

        result = pipeline.run_text_turn("调用视觉，分析下当前工作环境", synthesize=False)

        self.assertEqual(result.response_text, "当前未配置视觉快照服务，无法获取相机图像。")
        self.assertEqual(result.vision_artifacts, ())
        self.assertEqual(llm.image_calls, [])

    def test_broad_workspace_question_returns_guidance_without_llm_or_vision(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rules_path = root / "rules.json"
            mapping_path = root / "object_mapping.json"
            arm_rules_path = root / "arm_rules.json"
            image_path = root / "snapshot.jpg"
            image_path.write_bytes(b"fake image")
            write_rule_document(rules_path, complex_rule_document())
            write_object_mapping_document(mapping_path, object_mapping_document())
            write_arm_rule_document(arm_rules_path, {"arm_capture": "True", "arm_capture_goal": "C"})
            provider = FakeVisionSnapshotProvider(VisionImageArtifact(image_path=image_path))
            llm = FakeLlm('{"type":"tool_call","name":"analyze_environment_vision","arguments":{}}')
            pipeline = VoicePipeline(
                llm=llm,
                rules_path=rules_path,
                object_mapping_path=mapping_path,
                arm_rules_path=arm_rules_path,
                vision_snapshot_provider=provider,
            )

            result = pipeline.run_text_turn("当前工作区什么情况", synthesize=False)

        self.assertEqual(llm.calls, [])
        self.assertEqual(provider.calls, 0)
        self.assertEqual(llm.image_calls, [])
        self.assertEqual(result.vision_artifacts, ())
        self.assertIn("不会一次性展开所有状态", result.response_text)
        self.assertIn("当前安全规则是什么", result.response_text)
        self.assertIn("当前物体映射表", result.response_text)
        self.assertIn("当前机械臂抓取请求", result.response_text)
        self.assertIn("调用视觉分析当前画面", result.response_text)

    def test_workspace_guidance_mentions_pending_confirmation_when_available(self) -> None:
        pending = build_object_grasp_execution_confirmation(
            "帮我抓取标号A",
            marker="A",
            object_name="红色方块",
        ).with_runtime_state(confirmation_id="confirm-1", expires_at=time.monotonic() + 30.0)
        pipeline = VoicePipeline(
            llm=FakeLlm(),
            pending_confirmation_provider=lambda: pending,
        )

        result = pipeline.run_text_turn("当前工作环境什么情况", synthesize=False)

        self.assertIn("当前有待确认操作", result.response_text)
        self.assertIn("执行抓取", result.response_text)
        self.assertIn("调用视觉分析当前画面", result.response_text)

    def test_vision_snapshot_failure_returns_safe_message_without_artifact(self) -> None:
        provider = FakeVisionSnapshotProvider(error=RuntimeError("camera offline"))
        llm = FakeLlm('{"type":"tool_call","name":"analyze_environment_vision","arguments":{}}')
        pipeline = VoicePipeline(llm=llm, vision_snapshot_provider=provider)

        result = pipeline.run_text_turn("调用视觉，分析下当前工作环境", synthesize=False)

        self.assertIn("视觉服务当前不可用", result.response_text)
        self.assertIn("camera offline", result.response_text)
        self.assertEqual(result.vision_artifacts, ())
        self.assertEqual(llm.image_calls, [])

    def test_vision_snapshot_failure_reports_orbbec_usb_permission_hint(self) -> None:
        provider = FakeVisionSnapshotProvider(
            error=RuntimeError("RuntimeError: usbEnumerator openUsbDevice failed!")
        )
        llm = FakeLlm('{"type":"tool_call","name":"analyze_environment_vision","arguments":{}}')
        pipeline = VoicePipeline(llm=llm, vision_snapshot_provider=provider)

        result = pipeline.run_text_turn("调用视觉，分析下当前工作环境", synthesize=False)

        self.assertIn("视觉服务当前不可用", result.response_text)
        self.assertIn("Orbbec/Gemini 摄像头 USB 打开失败", result.response_text)
        self.assertIn("99-obsensor-libusb.rules", result.response_text)
        self.assertIn("usbEnumerator openUsbDevice failed", result.response_text)
        self.assertEqual(result.vision_artifacts, ())
        self.assertEqual(llm.image_calls, [])

    def test_format_vision_snapshot_error_reports_orbbec_udev_hint(self) -> None:
        message = format_vision_snapshot_error(
            "Open device failed: Access denied (insufficient permissions)"
        )

        self.assertIn("Orbbec/Gemini 摄像头 USB 打开失败", message)
        self.assertIn("/etc/udev/rules.d/99-obsensor-libusb.rules", message)
        self.assertIn("原始错误", message)

    def test_format_vision_snapshot_error_is_idempotent(self) -> None:
        message = format_vision_snapshot_error("usbEnumerator openUsbDevice failed!")

        self.assertEqual(format_vision_snapshot_error(message), message)
        self.assertEqual(
            format_vision_snapshot_error(f"Vision snapshot service failed: {message}"),
            f"Vision snapshot service failed: {message}",
        )

    def test_reported_assistant_address_correction_is_internal_only(self) -> None:
        llm = FakeLlm(
            "收到，正在修正语音识别错误：\n"
            "1. “机械皮出手” → 机械臂\n"
            "2. “出手” → 急停\n\n"
            "**安全指令：**\n"
            "机械臂急停！"
        )
        pipeline = VoicePipeline(asr=FakeAsr("你好机械皮出手"), llm=llm, tts=FakeTts())

        result = pipeline.run_audio_file(Path("sample.wav"), synthesize=True)

        self.assertEqual(result.input_text, "你好机械臂助手")
        self.assertIn("你好机械臂助手", llm.calls[0])
        self.assertEqual(llm.calls[-1], "你好机械臂助手")
        self.assertEqual(result.response_text, "你好，我是机械臂安全助手。")
        self.assertIsNotNone(result.tts)
        assert result.tts is not None
        self.assertEqual(result.tts.text, "你好，我是机械臂安全助手。")
        for forbidden in ("修正", "纠正", "误识别", "→", "**", "1."):
            self.assertNotIn(forbidden, result.response_text)

    def test_capability_prompt_returns_complete_spoken_answer(self) -> None:
        llm = FakeLlm(
            "收到。我将立即启动安全语音助手，准备用简短中文播报以下安全操作内容：\n\n"
            "1. **急停/停机**：立即停止机械臂运动。\n"
            "2. **复位**：执行机械臂复位操作。\n"
            "6. **光栅**："
        )
        pipeline = VoicePipeline(llm=llm)

        result = pipeline.run_text_turn("你能做什么", synthesize=False)

        self.assertEqual(result.response_text, CAPABILITY_RESPONSE)
        self.assertIn("急停", result.response_text)
        self.assertIn("气泵", result.response_text)
        self.assertIn("安全规则", result.response_text)
        self.assertNotIn("防护门", result.response_text)
        self.assertNotIn("光栅", result.response_text)
        self.assertNotIn("1.", result.response_text)
        self.assertNotIn("**", result.response_text)
        self.assertNotIn("播报以下", result.response_text)

    def test_acknowledgement_does_not_speak_model_route_chatter(self) -> None:
        llm = FakeLlm(
            "谢谢，我会继续保持简洁。\n\n"
            "ROUTE:rule_editor_9b\n"
            "将立即处理您的请求。"
        )
        pipeline = VoicePipeline(llm=llm)

        result = pipeline.run_text_turn("做得好", synthesize=False)

        self.assertEqual(result.response_text, "谢谢，我会继续保持简洁。")
        self.assertNotIn("ROUTE", result.response_text)
        self.assertNotIn("规则编辑器", result.response_text)

    def test_acknowledgement_with_only_internal_route_chatter_falls_back(self) -> None:
        llm = FakeLlm(
            "收到，已确认安全规则已就绪。\n\n"
            "ROUTE:rule_editor_9b\n"
            "将立即处理您的请求。"
        )
        pipeline = VoicePipeline(llm=llm)

        result = pipeline.run_text_turn("做得好", synthesize=False)

        self.assertEqual(result.response_text, "收到。")

    def test_empty_array_model_output_for_thanks_falls_back_to_spoken_ack(self) -> None:
        llm = FakeLlm("[]")
        pipeline = VoicePipeline(llm=llm)

        result = pipeline.run_text_turn("谢谢你", synthesize=False)

        self.assertEqual(result.response_text, "收到。")
        self.assertEqual(len(llm.calls), 2)
        self.assertIn("结构化工具路由器", llm.calls[0])
        self.assertEqual(llm.calls[1], "谢谢你")

    def test_rule_query_loads_current_rules_through_agent_tool(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rules_path = Path(temp_dir) / "rules.json"
            write_rule_document(
                rules_path,
                {
                    "version": 1,
                    "rules": [
                        {
                            "id": "stop_on_person_intrusion",
                            "name": "人员侵入停机规则",
                            "enabled": True,
                            "conditions": {"person_distance_m": {"lt": 0.8}},
                            "action": {"type": "stop_motion"},
                        },
                        {
                            "id": "slow_near_unknown_object",
                            "name": "未知物体靠近限速规则",
                            "enabled": True,
                            "conditions": {"unknown_object_distance_m": {"lt": 0.25}},
                            "action": {"type": "limit_speed", "max_speed_scale": 0.25},
                        },
                    ],
                },
            )
            llm = FakeLlm(
                [
                    "TOOL:load_rules",
                    "当前启用两条规则：人员进入 0.8 米保护距离内立即停止机械臂；"
                    "未知物体距离抓取路径 0.25 米内时限速到百分之二十五。"
                    "后续可以问人员阈值或未知物体限速动作。",
                ]
            )
            pipeline = VoicePipeline(llm=llm, rules_path=rules_path)

            result = pipeline.run_text_turn("当前有哪些规则", synthesize=False)

            self.assertEqual(len(llm.calls), 2)
            self.assertIn("结构化工具路由器", llm.calls[0])
            self.assertIn("当前有哪些规则", llm.calls[0])
            self.assertIn("当前规则文档", llm.calls[1])
            self.assertIn("stop_on_person_intrusion", llm.calls[1])
            self.assertIn("slow_near_unknown_object", llm.calls[1])
            self.assertIn("不要照抄英文 snake_case", llm.calls[1])
            self.assertIn("禁止输出后续建议", llm.calls[1])
            self.assertNotIn("后续具体提问建议", llm.calls[1])
            self.assertNotIn("建议必须点名具体规则和字段", llm.calls[1])
            self.assertIn("人员进入", result.response_text)
            self.assertIn("未知物体", result.response_text)
            self.assertIn("0.8", result.response_text)
            self.assertIn("0.25", result.response_text)
            self.assertNotIn("后续可以问", result.response_text)
            self.assertNotIn("人员阈值", result.response_text)
            self.assertNotIn("stop_on_person_intrusion", result.response_text)
            self.assertNotIn("slow_near_unknown_object", result.response_text)

    def test_broad_rule_query_strips_followups_and_cleans_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rules_path = Path(temp_dir) / "rules.json"
            write_rule_document(
                rules_path,
                {
                    "version": 1,
                    "rules": [
                        {
                            "id": "stop_on_person_intrusion",
                            "name": "人员侵入停机规则",
                            "enabled": True,
                            "conditions": {"person_distance_m": {"lt": 0.8}},
                            "action": {"type": "stop_motion"},
                        },
                        {
                            "id": "slow_near_unknown_object",
                            "name": "未知物体靠近限速规则",
                            "enabled": True,
                            "conditions": {"unknown_object_distance_m": {"lt": 0.25}},
                            "action": {"type": "limit_speed", "max_speed_scale": 0.25},
                        },
                        {
                            "id": "hold_on_ros_controller_alarm",
                            "name": "ROS 控制器报警安全保持规则",
                            "enabled": True,
                            "conditions": {"ros_controller_alarm": {"eq": True}},
                            "action": {
                                "type": "safety_hold",
                                "notify_operator": True,
                                "requires_manual_check": True,
                            },
                        },
                    ],
                },
            )
            llm = FakeLlm(
                [
                    "TOOL:load_rules",
                    "规则总览：；人员侵入停机：检测到人员进入保护距离时，立即停止机械臂。"
                    "未知物体靠近限速：发现未知物体靠近抓取路径时，降低速度等待确认。"
                    "ROS 控制器报警：机器人收到报警时，进入安全保持状态并通知操作员。"
                    "建议后续提问：想问防护门打开规则的复位条件吗？"
                    "想问示教模式限速阈值或控制器报警动作吗？"
                    "想问光栅障碍物检测规则、物体标号映射规则或抓取目标解析规则吗？",
                ]
            )
            pipeline = VoicePipeline(llm=llm, rules_path=rules_path)

            result = pipeline.run_text_turn("当前安全规则是什么", synthesize=False)

            self.assertIn("禁止输出后续建议", llm.calls[1])
            self.assertNotIn("建议格式示例", llm.calls[1])
            self.assertNotIn("防护门打开规则的复位条件", llm.calls[1])
            self.assertNotIn("示教模式限速阈值", llm.calls[1])
            self.assertNotIn("：；", result.response_text)
            self.assertIn("人员侵入停机", result.response_text)
            self.assertIn("未知物体靠近限速", result.response_text)
            self.assertIn("ROS 控制器报警", result.response_text)
            self.assertNotIn("建议后续提问", result.response_text)
            self.assertNotIn("想问", result.response_text)
            self.assertNotIn("防护门", result.response_text)
            self.assertNotIn("示教模式", result.response_text)
            self.assertNotIn("光栅", result.response_text)
            self.assertNotIn("物体标号", result.response_text)
            self.assertNotIn("抓取目标", result.response_text)
            self.assertNotIn("防护门打开规则的复位条件", result.response_text)
            self.assertNotIn("示教模式限速阈值", result.response_text)

    def test_rule_query_loads_rules_when_2b_only_says_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rules_path = Path(temp_dir) / "rules.json"
            write_rule_document(
                rules_path,
                {
                    "version": 1,
                    "rules": [
                        {
                            "id": "stop_on_person_intrusion",
                            "name": "人员侵入停机规则",
                            "enabled": True,
                            "conditions": {"person_distance_m": {"lt": 0.8}},
                            "action": {"type": "stop_motion"},
                        },
                        {
                            "id": "slow_near_unknown_object",
                            "name": "未知物体靠近限速规则",
                            "enabled": True,
                            "conditions": {"unknown_object_distance_m": {"lt": 0.25}},
                            "action": {"type": "limit_speed", "max_speed_scale": 0.25},
                        },
                    ],
                },
            )
            llm = FakeLlm(["TOOL:load_rules", "当前安全规则已加载。"])
            pipeline = VoicePipeline(llm=llm, rules_path=rules_path)

            result = pipeline.run_text_turn("当前安全规则是什么", synthesize=False)

            self.assertEqual(len(llm.calls), 2)
            self.assertIn("当前安全规则是什么", llm.calls[0])
            self.assertIn("当前规则文档", llm.calls[1])
            self.assertIn("模型没有生成可用的中文总结", result.response_text)
            self.assertIn("当前共有 2 条规则", result.response_text)
            self.assertIn("人员侵入停机规则", result.response_text)
            self.assertIn("未知物体靠近限速规则", result.response_text)
            self.assertIn("触发条件", result.response_text)
            self.assertNotIn("已加载", result.response_text)
            for english in (
                "stop_on_person_intrusion",
                "slow_near_unknown_object",
                "person_distance_m",
                "unknown_object_distance_m",
                "stop_motion",
                "limit_speed",
                "max_speed_scale",
            ):
                self.assertNotIn(english, result.response_text)

    def test_rule_query_loads_current_rules_through_structured_tool_call(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rules_path = Path(temp_dir) / "rules.json"
            write_rule_document(
                rules_path,
                {
                    "version": 1,
                    "rules": [
                        {
                            "id": "stop_on_guard_door_open",
                            "name": "防护门打开停机规则",
                            "enabled": True,
                            "conditions": {"guard_door_open": {"eq": True}},
                            "action": {"type": "stop_motion", "requires_reset": True},
                        }
                    ],
                },
            )
            llm = FakeLlm(
                [
                    '{"type":"tool_call","name":"load_rules","arguments":{}}',
                    "防护门打开停机规则当前启用，防护门打开时会停止机械臂并要求复位确认。",
                ]
            )
            pipeline = VoicePipeline(llm=llm, rules_path=rules_path)

            result = pipeline.run_text_turn("当前安全规则是什么", synthesize=False)

            self.assertEqual(len(llm.calls), 2)
            self.assertIn("当前规则文档", llm.calls[1])
            self.assertIn("stop_on_guard_door_open", llm.calls[1])
            self.assertIn("防护门打开", result.response_text)
            self.assertNotIn("stop_on_guard_door_open", result.response_text)

    def test_rule_query_without_rules_path_does_not_hallucinate(self) -> None:
        llm = FakeLlm("TOOL:load_rules")
        pipeline = VoicePipeline(llm=llm)

        result = pipeline.run_text_turn("当前有哪些规则", synthesize=False)

        self.assertEqual(len(llm.calls), 1)
        self.assertIn("当前有哪些规则", llm.calls[0])
        self.assertEqual(result.response_text, "当前未配置安全规则文件，无法读取当前规则。")

    def test_rule_query_without_tool_marker_still_requires_rules_path(self) -> None:
        llm = FakeLlm("当前安全规则已加载。")
        pipeline = VoicePipeline(llm=llm)

        result = pipeline.run_text_turn("当前安全规则是什么", synthesize=False)

        self.assertEqual(len(llm.calls), 2)
        self.assertIn("当前安全规则是什么", llm.calls[0])
        self.assertEqual(llm.calls[1], "当前安全规则是什么")
        self.assertEqual(result.response_text, "当前未配置安全规则文件，无法读取当前规则。")

    def test_rule_query_logs_route_final_intermediate_result(self) -> None:
        llm = FakeLlm('{"type":"final","content":"当前安全规则已加载。"}')
        pipeline = VoicePipeline(llm=llm)

        with self.assertLogs("local_safety_assistant.stack.pipeline", level="INFO") as captured:
            result = pipeline.run_text_turn("当前安全规则是什么", synthesize=False)

        log_text = "\n".join(captured.output)
        self.assertIn("agent_route", log_text)
        self.assertIn('"decision_type": "final"', log_text)
        self.assertIn("当前安全规则已加载", log_text)
        self.assertEqual(result.response_text, "收到。")

    def test_rule_query_logs_rule_read_intermediate_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rules_path = Path(temp_dir) / "rules.json"
            write_rule_document(
                rules_path,
                {
                    "version": 1,
                    "rules": [
                        {
                            "id": "stop_on_person_intrusion",
                            "name": "人员侵入停机规则",
                            "enabled": True,
                            "conditions": {"person_distance_m": {"lt": 0.8}},
                            "action": {"type": "stop_motion"},
                        }
                    ],
                },
            )
            llm = FakeLlm(["TOOL:load_rules", "人员小于 0.8 米时立即停机。"])
            pipeline = VoicePipeline(llm=llm, rules_path=rules_path)

            with self.assertLogs("local_safety_assistant.stack.pipeline", level="INFO") as captured:
                result = pipeline.run_text_turn("当前安全规则是什么", synthesize=False)

        log_text = "\n".join(captured.output)
        self.assertIn("agent_route", log_text)
        self.assertIn('"decision_type": "tool_call"', log_text)
        self.assertIn('"tool_name": "load_rules"', log_text)
        self.assertIn("rule_read", log_text)
        self.assertIn('"used_fallback": false', log_text)
        self.assertIn("人员小于 0.8 米时立即停机", log_text)
        self.assertEqual(result.response_text, "人员小于 0.8 米时立即停机。")

    def test_rule_query_uses_minimal_fallback_if_model_repeats_tool_request(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rules_path = Path(temp_dir) / "rules.json"
            write_rule_document(
                rules_path,
                {
                    "version": 1,
                    "rules": [
                        {
                            "id": "stop_on_person_intrusion",
                            "name": "人员侵入停机规则",
                            "enabled": True,
                            "conditions": {"person_distance_m": {"lt": 0.8}},
                            "action": {"type": "stop_motion", "severity": "critical"},
                        }
                    ],
                },
            )
            llm = FakeLlm(["TOOL:load_rules", "TOOL:load_rules"])
            pipeline = VoicePipeline(llm=llm, rules_path=rules_path)

            result = pipeline.run_text_turn("当前安全规则是什么", synthesize=False)

            self.assertEqual(len(llm.calls), 2)
            self.assertIn("当前规则文档", llm.calls[1])
            self.assertIn("模型没有生成可用的中文总结", result.response_text)
            self.assertIn("当前共有 1 条规则", result.response_text)
            self.assertIn("人员侵入停机规则", result.response_text)
            self.assertNotIn("stop_on_person_intrusion", result.response_text)
            self.assertNotIn("person_distance_m", result.response_text)
            self.assertNotIn("stop_motion", result.response_text)
            self.assertNotIn("severity", result.response_text)
            self.assertNotIn("TOOL", result.response_text)

    def test_rule_query_uses_minimal_fallback_if_model_only_acknowledges_load(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rules_path = Path(temp_dir) / "rules.json"
            write_rule_document(
                rules_path,
                {
                    "version": 1,
                    "rules": [
                        {
                            "id": "slow_near_unknown_object",
                            "name": "未知物体靠近限速规则",
                            "enabled": True,
                            "conditions": {"unknown_object_distance_m": {"lt": 0.25}},
                            "action": {"type": "limit_speed", "max_speed_scale": 0.25},
                        }
                    ],
                },
            )
            llm = FakeLlm(["TOOL:load_rules", "当前安全规则已加载。"])
            pipeline = VoicePipeline(llm=llm, rules_path=rules_path)

            result = pipeline.run_text_turn("当前安全规则是什么", synthesize=False)

            self.assertEqual(len(llm.calls), 2)
            self.assertIn("当前规则文档", llm.calls[1])
            self.assertIn("模型没有生成可用的中文总结", result.response_text)
            self.assertIn("当前共有 1 条规则", result.response_text)
            self.assertIn("未知物体靠近限速规则", result.response_text)
            self.assertNotIn("slow_near_unknown_object", result.response_text)
            self.assertNotIn("unknown_object_distance_m", result.response_text)
            self.assertNotIn("max_speed_scale", result.response_text)
            self.assertNotIn("已加载", result.response_text)

    def test_rule_query_accepts_natural_final_answer_without_rule_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rules_path = Path(temp_dir) / "rules.json"
            write_rule_document(
                rules_path,
                {
                    "version": 1,
                    "rules": [
                        {
                            "id": "stop_on_person_intrusion",
                            "name": "人员侵入停机规则",
                            "enabled": True,
                            "conditions": {"person_distance_m": {"lt": 0.8}},
                            "action": {"type": "stop_motion"},
                        },
                        {
                            "id": "slow_near_unknown_object",
                            "name": "未知物体靠近限速规则",
                            "enabled": True,
                            "conditions": {"unknown_object_distance_m": {"lt": 0.25}},
                            "action": {"type": "limit_speed", "max_speed_scale": 0.25},
                        },
                    ],
                },
            )
            llm = FakeLlm(
                [
                    "TOOL:load_rules",
                    "当前安全规则共两条：人员距离小于 0.8 米时停止；未知物体距离小于 0.25 米时限速。",
                ]
            )
            pipeline = VoicePipeline(llm=llm, rules_path=rules_path)

            result = pipeline.run_text_turn("当前安全规则是什么", synthesize=False)

            self.assertEqual(
                result.response_text,
                "当前安全规则共两条：人员距离小于 0.8 米时停止；未知物体距离小于 0.25 米时限速。",
            )
            self.assertNotIn("stop_on_person_intrusion", result.response_text)
            self.assertNotIn("slow_near_unknown_object", result.response_text)
            self.assertNotIn("stop_motion", result.response_text)
            self.assertNotIn("limit_speed", result.response_text)

    def test_traditional_chinese_limit_speed_rule_query_reads_rules(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rules_path = Path(temp_dir) / "rules.json"
            write_rule_document(rules_path, complex_rule_document())
            llm = FakeLlm(
                [
                    "TOOL:load_rules",
                    "当前限速相关规则有两条：未知物体靠近限速规则（id: slow_near_unknown_object）"
                    "在抓取路径 0.25 米内会限速到百分之二十五；"
                    "示教模式限速规则（id: limit_speed_in_teach_mode）"
                    "会把速度限制到正常速度的百分之十。",
                ]
            )
            pipeline = VoicePipeline(llm=llm, rules_path=rules_path)

            result = pipeline.run_text_turn("詳細說一說當前的限速規則", synthesize=False)

            self.assertIn("详细说一说当前的限速规则", llm.calls[0])
            self.assertEqual(len(llm.calls), 2)
            self.assertIn("当前规则文档", llm.calls[1])
            self.assertIn("limit_speed_in_teach_mode", llm.calls[1])
            self.assertIn("候选规则提示：本问题只应回答这些候选规则", llm.calls[1])
            self.assertIn("未知物体靠近限速规则", llm.calls[1])
            self.assertIn("示教模式限速规则", llm.calls[1])
            self.assertIn("候选规则超过一条，必须逐条覆盖全部候选规则", llm.calls[1])
            self.assertIn("严禁输出 id", llm.calls[1])
            self.assertIn("未知物体", result.response_text)
            self.assertIn("示教模式", result.response_text)
            self.assertNotIn("id:", result.response_text)
            self.assertNotIn("slow_near_unknown_object", result.response_text)
            self.assertNotIn("limit_speed_in_teach_mode", result.response_text)
            self.assertNotEqual(result.response_text, "机械臂已限速。")

    def test_rule_query_accepts_useful_answer_that_mentions_loaded_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rules_path = Path(temp_dir) / "rules.json"
            write_rule_document(rules_path, complex_rule_document())
            llm = FakeLlm(
                [
                    "TOOL:load_rules",
                    "已读取当前规则，共六条：人员进入 0.8 米内停机，未知物体 0.25 米内限速，"
                    "防护门打开和光栅遮挡会停机，控制器报警进入安全保持，示教模式限速到百分之十。"
                    "后续可以问防护门规则或示教限速阈值。",
                ]
            )
            pipeline = VoicePipeline(llm=llm, rules_path=rules_path)

            result = pipeline.run_text_turn("当前安全规则是什么", synthesize=False)

            self.assertIn("已读取当前规则", result.response_text)
            self.assertIn("防护门", result.response_text)
            self.assertIn("示教模式", result.response_text)
            self.assertNotIn("后续可以问", result.response_text)
            self.assertNotIn("模型没有生成可用的中文总结", result.response_text)

    def test_broad_rule_query_summarizes_complex_rules_without_followups(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rules_path = Path(temp_dir) / "rules.json"
            write_rule_document(rules_path, complex_rule_document())
            llm = FakeLlm(
                [
                    "TOOL:load_rules",
                    "当前有六类启用安全规则：人员进入 0.8 米保护距离内会立即停机；"
                    "未知物体靠近抓取路径 0.25 米内会限速到百分之二十五；"
                    "防护门打开和安全光栅遮挡都会触发停机并要求确认复位；"
                    "ROS 控制器报警时进入安全保持并通知操作员；"
                    "示教模式下速度限制到百分之十。"
                    "后续可以具体问防护门打开规则、安全光栅规则或示教限速阈值。",
                ]
            )
            pipeline = VoicePipeline(llm=llm, rules_path=rules_path)

            result = pipeline.run_text_turn("请总结当前全部安全规则", synthesize=False)

            self.assertEqual(len(llm.calls), 2)
            for text in (
                "stop_on_guard_door_open",
                "stop_on_light_curtain_blocked",
                "hold_on_ros_controller_alarm",
                "limit_speed_in_teach_mode",
            ):
                self.assertIn(text, llm.calls[1])
            self.assertIn("问题类型提示：这是规则总览问题", llm.calls[1])
            for text in ("人员", "未知物体", "防护门", "光栅", "控制器", "示教"):
                self.assertIn(text, result.response_text)
            self.assertNotIn("后续可以具体问", result.response_text)
            self.assertNotIn("示教限速阈值", result.response_text)
            for english in (
                "stop_on_guard_door_open",
                "light_curtain_blocked",
                "hold_on_ros_controller_alarm",
                "max_speed_scale",
            ):
                self.assertNotIn(english, result.response_text)

    def test_specific_rule_query_explains_matching_json_rule_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rules_path = Path(temp_dir) / "rules.json"
            write_rule_document(rules_path, complex_rule_document())
            llm = FakeLlm(
                [
                    "TOOL:load_rules",
                    "防护门打开停机规则当前启用。只要防护门被打开，就停止机械臂运动；"
                    "这条规则属于严重级别，需要现场确认后复位，不涉及限速阈值。",
                ]
            )
            pipeline = VoicePipeline(llm=llm, rules_path=rules_path)

            result = pipeline.run_text_turn("请详细说明防护门打开规则", synthesize=False)

            self.assertEqual(len(llm.calls), 2)
            self.assertIn("如果问题指向某一条或某一类规则", llm.calls[1])
            self.assertIn("问题类型提示：这是具体规则问题", llm.calls[1])
            self.assertIn("禁止输出后续建议", llm.calls[1])
            self.assertIn("不要补充通用安全建议", llm.calls[1])
            self.assertIn("候选规则提示：本问题只应回答这些候选规则：防护门打开停机规则", llm.calls[1])
            self.assertIn("防护门", result.response_text)
            self.assertIn("启用", result.response_text)
            self.assertIn("停止", result.response_text)
            self.assertIn("复位", result.response_text)
            for unrelated in ("人员进入 0.8", "未知物体", "安全光栅", "示教模式", "ROS 控制器"):
                self.assertNotIn(unrelated, result.response_text)

    def test_structured_router_adjusts_person_distance_without_keyword_intent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rules_path = Path(temp_dir) / "rules.json"
            write_rule_document(rules_path, complex_rule_document())
            llm = FakeLlm(
                [
                    '{"type":"tool_call","name":"edit_rules","arguments":{}}',
                    '{"type":"tool_call","name":"edit_rules","arguments":{'
                    '"rule_id":"stop_on_person_intrusion",'
                    '"changes":{"conditions.person_distance_m.lt":1.2}}}',
                ],
                model="qwen35-2b",
            )
            pipeline = VoicePipeline(llm=llm, rules_path=rules_path)

            result = pipeline.run_text_turn("把人员安全距离调整为1.2米", synthesize=False)

            updated = load_rule_document(rules_path)
            self.assertEqual(updated["version"], 2)
            self.assertEqual(updated["rules"][0]["conditions"]["person_distance_m"]["lt"], 1.2)
            self.assertIsNotNone(result.rule_update)
            assert result.rule_update is not None
            self.assertEqual(result.rule_update.rule_id, "stop_on_person_intrusion")
            self.assertIn("人员侵入停机规则", result.response_text)
            self.assertIn("人员保护距离阈值", result.response_text)
            self.assertIn("1.2 米", result.response_text)
            self.assertIn("当前版本 2", result.response_text)
            self.assertEqual(len(llm.calls), 2)
            self.assertIn("结构化工具路由器", llm.calls[0])
            self.assertIn("把人员安全距离调整为1.2米", llm.calls[0])
            self.assertIn("当前规则文档", llm.calls[1])

    def test_structured_router_rejects_vague_rule_edit_without_patch_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rules_path = Path(temp_dir) / "rules.json"
            write_rule_document(rules_path, complex_rule_document())
            llm = FakeLlm(
                [
                    '{"type":"tool_call","name":"edit_rules","arguments":{}}',
                    '{"type":"tool_call","name":"edit_rules","arguments":{'
                    '"rule_id":"stop_on_person_intrusion",'
                    '"changes":{"conditions.person_distance_m.lt":0.1}}}',
                ],
                model="qwen35-2b",
            )
            pipeline = VoicePipeline(
                llm=llm,
                rules_path=rules_path,
                require_confirmation_for_side_effects=True,
            )

            result = pipeline.run_text_turn("可以更改吗", synthesize=False)

            self.assertEqual(len(llm.calls), 1)
            self.assertEqual(load_rule_document(rules_path)["version"], 1)
            self.assertIsNone(result.rule_update)
            self.assertIsNone(result.pending_confirmation)
            self.assertIn("请明确要修改哪条已有安全规则", result.response_text)

    def test_rule_edit_rejects_model_value_not_grounded_in_user_request(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rules_path = Path(temp_dir) / "rules.json"
            write_rule_document(rules_path, complex_rule_document())
            llm = FakeLlm(
                [
                    '{"type":"tool_call","name":"edit_rules","arguments":{}}',
                    '{"type":"tool_call","name":"edit_rules","arguments":{'
                    '"rule_id":"stop_on_person_intrusion",'
                    '"changes":{"conditions.person_distance_m.lt":0.1}}}',
                ],
                model="qwen35-2b",
            )
            pipeline = VoicePipeline(
                llm=llm,
                rules_path=rules_path,
                require_confirmation_for_side_effects=True,
            )

            result = pipeline.run_text_turn("把人员安全距离调整为0.4米", synthesize=False)

            document = load_rule_document(rules_path)
            self.assertEqual(document["version"], 1)
            self.assertEqual(document["rules"][0]["conditions"]["person_distance_m"]["lt"], 0.8)
            self.assertEqual(len(llm.calls), 2)
            self.assertIsNone(result.rule_update)
            self.assertIsNone(result.pending_confirmation)
            self.assertIn("请明确要修改哪条已有安全规则", result.response_text)

    def test_rule_edit_rejects_model_boolean_not_grounded_in_user_request(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rules_path = Path(temp_dir) / "rules.json"
            write_rule_document(rules_path, complex_rule_document())
            llm = FakeLlm(
                [
                    '{"type":"tool_call","name":"edit_rules","arguments":{}}',
                    '{"type":"tool_call","name":"edit_rules","arguments":{'
                    '"rule_id":"stop_on_person_intrusion","changes":{"enabled":true}}}',
                ],
                model="qwen35-2b",
            )
            pipeline = VoicePipeline(
                llm=llm,
                rules_path=rules_path,
                require_confirmation_for_side_effects=True,
            )

            result = pipeline.run_text_turn("暂时禁用人员安全距离规则", synthesize=False)

            self.assertEqual(load_rule_document(rules_path)["version"], 1)
            self.assertIsNone(result.rule_update)
            self.assertIsNone(result.pending_confirmation)
            self.assertIn("请明确要修改哪条已有安全规则", result.response_text)

    def test_rule_distance_change_syncs_arm_runtime_distance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rules_path = root / "rules.json"
            arm_rules_path = root / "arm_rules.json"
            write_rule_document(rules_path, complex_rule_document())
            write_arm_rule_document(
                arm_rules_path,
                {
                    "arm_capture": "True",
                    "arm_capture_goal": "C",
                    "arm_decelerate": "0.5",
                    "arm_stop": "False",
                    "arm_recover": "True",
                    "arm_safety_distance": "0.2",
                },
            )
            llm = FakeLlm(
                [
                    '{"type":"tool_call","name":"edit_rules","arguments":{}}',
                    '{"type":"tool_call","name":"edit_rules","arguments":{'
                    '"rule_id":"stop_on_person_intrusion",'
                    '"changes":{"conditions.person_distance_m.lt":0.4}}}',
                ],
                model="qwen35-2b",
            )
            pipeline = VoicePipeline(llm=llm, rules_path=rules_path, arm_rules_path=arm_rules_path)

            result = pipeline.run_text_turn("把人员安全距离调整为0.4米", synthesize=False)

            arm_rules = load_arm_rule_document(arm_rules_path)
            self.assertEqual(arm_rules["arm_safety_distance"], "0.4")
            self.assertEqual(arm_rules["arm_capture"], "True")
            self.assertEqual(arm_rules["arm_capture_goal"], "C")
            self.assertEqual(arm_rules["arm_decelerate"], "0.5")
            self.assertEqual(arm_rules["arm_recover"], "True")
            self.assertIn("机械臂运行时安全距离已同步为 0.4 米", result.response_text)

    def test_rule_non_distance_change_leaves_arm_runtime_distance_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rules_path = root / "rules.json"
            arm_rules_path = root / "arm_rules.json"
            write_rule_document(rules_path, complex_rule_document())
            write_arm_rule_document(arm_rules_path, {"arm_safety_distance": "0.2"})
            llm = FakeLlm(
                [
                    '{"type":"tool_call","name":"edit_rules","arguments":{}}',
                    '{"type":"tool_call","name":"edit_rules","arguments":{'
                    '"rule_id":"stop_on_person_intrusion","changes":{"enabled":false}}}',
                ],
                model="qwen35-2b",
            )
            pipeline = VoicePipeline(llm=llm, rules_path=rules_path, arm_rules_path=arm_rules_path)

            result = pipeline.run_text_turn("暂时禁用人员安全距离规则", synthesize=False)

            self.assertIsNotNone(result.rule_update)
            self.assertEqual(load_arm_rule_document(arm_rules_path)["arm_safety_distance"], "0.2")
            self.assertNotIn("运行时安全距离已同步", result.response_text)

    def test_rule_change_rejects_out_of_range_person_distance_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rules_path = root / "rules.json"
            arm_rules_path = root / "arm_rules.json"
            write_rule_document(rules_path, complex_rule_document())
            write_arm_rule_document(arm_rules_path, {"arm_safety_distance": "0.2"})
            llm = FakeLlm(
                [
                    '{"type":"tool_call","name":"edit_rules","arguments":{}}',
                    '{"type":"tool_call","name":"edit_rules","arguments":{'
                    '"rule_id":"stop_on_person_intrusion",'
                    '"changes":{"conditions.person_distance_m.lt":10000.0}}}',
                ],
                model="qwen35-2b",
            )
            pipeline = VoicePipeline(llm=llm, rules_path=rules_path, arm_rules_path=arm_rules_path)

            result = pipeline.run_text_turn("把人员安全距离调整为10000米", synthesize=False)

            updated = load_rule_document(rules_path)
            self.assertEqual(updated["version"], 1)
            self.assertEqual(updated["rules"][0]["conditions"]["person_distance_m"]["lt"], 0.8)
            self.assertEqual(load_arm_rule_document(arm_rules_path)["arm_safety_distance"], "0.2")
            self.assertIsNone(result.rule_update)
            self.assertIn("规则修改请求未通过安全验证", result.response_text)
            self.assertIn("未写入规则文件", result.response_text)
            self.assertIn("0.1 到 5 米", result.response_text)

    def test_rule_change_can_temporarily_disable_and_restore_person_distance_rule(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rules_path = Path(temp_dir) / "rules.json"
            write_rule_document(rules_path, complex_rule_document())
            disable_llm = FakeLlm(
                [
                    '{"type":"tool_call","name":"edit_rules","arguments":{}}',
                    '{"type":"tool_call","name":"edit_rules","arguments":{'
                    '"rule_id":"stop_on_person_intrusion","changes":{"enabled":false}}}',
                ],
                model="qwen35-2b",
            )
            disable_pipeline = VoicePipeline(llm=disable_llm, rules_path=rules_path)

            disabled = disable_pipeline.run_text_turn("暂时禁用人员安全距离规则", synthesize=False)

            after_disable = load_rule_document(rules_path)
            self.assertEqual(after_disable["version"], 2)
            self.assertFalse(after_disable["rules"][0]["enabled"])
            self.assertIsNotNone(disabled.rule_update)
            self.assertIn("启用状态改为禁用", disabled.response_text)

            restore_llm = FakeLlm(
                [
                    '{"type":"tool_call","name":"edit_rules","arguments":{}}',
                    '{"type":"tool_call","name":"edit_rules","arguments":{'
                    '"rule_id":"stop_on_person_intrusion","changes":{"enabled":true}}}',
                ],
                model="qwen35-2b",
            )
            restore_pipeline = VoicePipeline(llm=restore_llm, rules_path=rules_path)

            restored = restore_pipeline.run_text_turn("恢复人员安全距离规则", synthesize=False)

            after_restore = load_rule_document(rules_path)
            self.assertEqual(after_restore["version"], 3)
            self.assertTrue(after_restore["rules"][0]["enabled"])
            self.assertIsNotNone(restored.rule_update)
            self.assertIn("启用状态改为启用", restored.response_text)

    def test_rule_change_two_pass_uses_same_2b_patch_and_writes_rule(self) -> None:
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
            llm = FakeLlm(
                [
                    '{"type":"tool_call","name":"edit_rules","arguments":{}}',
                    '{"type":"tool_call","name":"edit_rules","arguments":{'
                    '"rule_id":"slow_near_unknown_object",'
                    '"changes":{"action.max_speed_scale":0.1}}}',
                ],
                model="qwen35-2b",
            )
            pipeline = VoicePipeline(
                llm=llm,
                rules_path=rules_path,
                rule_edit_strategy=RULE_EDIT_STRATEGY_TWO_PASS,
            )

            result = pipeline.run_text_turn("请把未知物体附近的限速规则改成 0.1", synthesize=False)

            self.assertEqual(len(llm.calls), 2)
            self.assertIn("结构化工具路由器", llm.calls[0])
            self.assertIn("请把未知物体附近的限速规则改成 0.1", llm.calls[0])
            self.assertIn("当前规则文档", llm.calls[1])
            self.assertIn("edit_rules", llm.calls[1])
            updated = load_rule_document(rules_path)
            self.assertEqual(updated["version"], 2)
            self.assertEqual(updated["rules"][0]["action"]["max_speed_scale"], 0.1)
            self.assertIsNotNone(result.rule_update)
            self.assertEqual(result.rule_update.rule_id, "slow_near_unknown_object")
            self.assertEqual(result.rule_update.strategy, RULE_EDIT_STRATEGY_TWO_PASS)
            self.assertEqual(result.rule_update.patch_llm.model, "qwen35-2b")
            self.assertNotIn("slow_near_unknown_object", result.response_text)

    def test_rule_change_one_pass_uses_patch_prompt_without_initial_answer(self) -> None:
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
            llm = FakeLlm(
                [
                    '{"type":"tool_call","name":"edit_rules","arguments":{}}',
                    '{"type":"tool_call","name":"edit_rules","arguments":{'
                    '"rule_id":"slow_near_unknown_object",'
                    '"changes":{"action.max_speed_scale":0.1}}}',
                ]
            )
            pipeline = VoicePipeline(
                llm=llm,
                rules_path=rules_path,
                rule_edit_strategy=RULE_EDIT_STRATEGY_ONE_PASS,
            )

            result = pipeline.run_text_turn("请把未知物体附近的限速规则改成 0.1", synthesize=False)

            self.assertEqual(len(llm.calls), 2)
            self.assertIn("结构化工具路由器", llm.calls[0])
            self.assertIn("当前规则文档", llm.calls[1])
            self.assertIn("策略：one-pass", llm.calls[1])
            self.assertEqual(load_rule_document(rules_path)["rules"][0]["action"]["max_speed_scale"], 0.1)
            self.assertIsNotNone(result.rule_update)
            self.assertEqual(result.rule_update.strategy, RULE_EDIT_STRATEGY_ONE_PASS)

    def test_rule_add_or_delete_request_is_rejected_before_llm(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rules_path = Path(temp_dir) / "rules.json"
            write_rule_document(
                rules_path,
                {
                    "version": 1,
                    "rules": [{"id": "slow_near_unknown_object", "enabled": True}],
                },
            )
            llm = FakeLlm()
            pipeline = VoicePipeline(llm=llm, rules_path=rules_path)

            result = pipeline.run_text_turn("请新增一条安全规则", synthesize=False)

            self.assertEqual(llm.calls, [])
            self.assertEqual(result.llm.model, "deterministic-router")
            self.assertIsNone(result.rule_update)
            self.assertEqual(load_rule_document(rules_path)["version"], 1)
            self.assertIn("不支持新增或删除规则", result.response_text)

    def test_structured_edit_rules_with_patch_arguments_applies_without_9b(self) -> None:
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
            llm = FakeLlm(
                '{"type":"tool_call","name":"edit_rules","arguments":{'
                '"rule_id":"slow_near_unknown_object",'
                '"changes":{"action.max_speed_scale":0.1}}}',
                model="qwen35-2b",
            )
            pipeline = VoicePipeline(llm=llm, rules_path=rules_path)

            result = pipeline.run_text_turn("请把未知物体附近的限速规则改成0.1", synthesize=False)

            self.assertEqual(len(llm.calls), 1)
            self.assertEqual(load_rule_document(rules_path)["rules"][0]["action"]["max_speed_scale"], 0.1)
            self.assertIsNotNone(result.rule_update)
            self.assertEqual(result.rule_update.patch_llm.model, "qwen35-2b")

    def test_rule_change_confirmation_mode_returns_pending_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rules_path = Path(temp_dir) / "rules.json"
            write_rule_document(rules_path, complex_rule_document())
            llm = FakeLlm(
                [
                    '{"type":"tool_call","name":"edit_rules","arguments":{}}',
                    '{"type":"tool_call","name":"edit_rules","arguments":{'
                    '"rule_id":"stop_on_person_intrusion","changes":{"enabled":false}}}',
                ],
                model="qwen35-2b",
            )
            pipeline = VoicePipeline(
                llm=llm,
                rules_path=rules_path,
                require_confirmation_for_side_effects=True,
            )

            result = pipeline.run_text_turn("暂时禁用人员安全距离规则", synthesize=False)

            self.assertEqual(load_rule_document(rules_path)["version"], 1)
            self.assertIsNone(result.rule_update)
            self.assertIsNotNone(result.pending_confirmation)
            assert result.pending_confirmation is not None
            self.assertEqual(result.pending_confirmation.action_type, ACTION_RULE_EDIT)
            self.assertEqual(result.pending_confirmation.details["rule_id"], "stop_on_person_intrusion")
            self.assertIn("需要确认", result.response_text)

    def test_rule_change_repairs_legacy_rules_array_for_person_distance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rules_path = Path(temp_dir) / "rules.json"
            write_rule_document(rules_path, complex_rule_document())
            llm = FakeLlm(
                '收到，已为您修改规则。 json; {"type":"tool_call","name":"edit_rules",'
                '"arguments":{"rules":[{"id":"safety_distance","name":"安全距离",'
                '"conditions":{"distance_m":1.0},"action":"stop","enabled":true}]}}',
                model="qwen35-2b",
            )
            pipeline = VoicePipeline(llm=llm, rules_path=rules_path)

            result = pipeline.run_text_turn("把人员安全距离修改为1m", synthesize=False)

            updated = load_rule_document(rules_path)
            self.assertEqual(updated["version"], 2)
            self.assertEqual(updated["rules"][0]["conditions"]["person_distance_m"]["lt"], 1.0)
            self.assertIsNotNone(result.rule_update)
            assert result.rule_update is not None
            self.assertEqual(result.rule_update.rule_id, "stop_on_person_intrusion")
            self.assertEqual(
                result.rule_update.patch,
                {
                    "rule_id": "stop_on_person_intrusion",
                    "changes": {"conditions.person_distance_m.lt": 1.0},
                },
            )
            self.assertNotIn("json", result.response_text.lower())

    def test_rule_change_repairs_wrong_patch_path_for_person_distance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rules_path = Path(temp_dir) / "rules.json"
            write_rule_document(rules_path, complex_rule_document())
            llm = FakeLlm(
                [
                    '{"type":"tool_call","name":"edit_rules","arguments":{}}',
                    '{"type":"tool_call","name":"edit_rules","arguments":{'
                    '"rule_id":"safety_distance",'
                    '"changes":{"conditions.distance_m":0.5}}}',
                ],
                model="qwen35-2b",
            )
            pipeline = VoicePipeline(llm=llm, rules_path=rules_path)

            result = pipeline.run_text_turn("把人员安全距离修改为0.5m", synthesize=False)

            updated = load_rule_document(rules_path)
            self.assertEqual(updated["version"], 2)
            self.assertEqual(updated["rules"][0]["conditions"]["person_distance_m"]["lt"], 0.5)
            self.assertIsNotNone(result.rule_update)
            assert result.rule_update is not None
            self.assertEqual(result.rule_update.rule_id, "stop_on_person_intrusion")

    def test_route_marker_no_longer_triggers_9b_editor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rules_path = Path(temp_dir) / "rules.json"
            write_rule_document(
                rules_path,
                {
                    "version": 1,
                    "rules": [{"id": "stop_on_person_intrusion", "enabled": True}],
                },
            )
            primary = FakeLlm("route:rule_editor_9b\n交给 9B。", model="qwen35-2b")
            pipeline = VoicePipeline(llm=primary, rules_path=rules_path)

            result = pipeline.run_text_turn("按我的新要求处理", synthesize=False)

            self.assertEqual(load_rule_document(rules_path)["rules"][0]["enabled"], True)
            self.assertIsNone(result.rule_update)
            self.assertEqual(result.response_text, "收到。")

    def test_rule_change_rejection_explains_restriction_and_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rules_path = Path(temp_dir) / "rules.json"
            write_rule_document(rules_path, complex_rule_document())
            llm = FakeLlm(
                [
                    '{"type":"tool_call","name":"edit_rules","arguments":{}}',
                    '{"type":"tool_call","name":"edit_rules","arguments":{'
                    '"rule_id":"stop_on_person_intrusion",'
                    '"changes":{"action.type":"limit_speed"}}}',
                ],
                model="qwen35-2b",
            )
            pipeline = VoicePipeline(llm=llm, rules_path=rules_path)

            result = pipeline.run_text_turn("把人员侵入规则改成限速", synthesize=False)

            updated = load_rule_document(rules_path)
            self.assertEqual(updated["version"], 1)
            self.assertEqual(updated["rules"][0]["action"]["type"], "stop_motion")
            self.assertIsNone(result.rule_update)
            self.assertIn("规则修改请求未通过安全验证", result.response_text)
            self.assertIn("未写入规则文件", result.response_text)
            self.assertIn("动作类型属于受保护字段", result.response_text)
            self.assertIn("把对应已有规则的启用状态暂时设为禁用", result.response_text)

    def test_object_mapping_update_routes_and_writes_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "object_mapping.json"
            write_object_mapping_document(mapping_path, object_mapping_document())
            llm = FakeLlm('{"type":"tool_call","name":"update_object_mapping","arguments":{}}')
            pipeline = VoicePipeline(llm=llm, object_mapping_path=mapping_path)

            result = pipeline.run_text_turn("把标号A的映射改成红色方块夹具", synthesize=False)

            updated = load_object_mapping_document(mapping_path)
            self.assertEqual(updated["version"], 2)
            self.assertEqual(updated["markers"]["A"]["object"], "红色方块夹具")
            self.assertIsNotNone(result.object_mapping_update)
            assert result.object_mapping_update is not None
            self.assertEqual(result.object_mapping_update.marker, "A")
            self.assertEqual(result.object_mapping_update.object_name, "红色方块夹具")
            self.assertIn("已更新映射：A 现在对应红色方块夹具", result.response_text)
            self.assertIn("请确认现场 A 标贴", result.response_text)

    def test_object_mapping_update_confirmation_mode_returns_pending_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "object_mapping.json"
            write_object_mapping_document(mapping_path, object_mapping_document())
            llm = FakeLlm('{"type":"tool_call","name":"update_object_mapping","arguments":{}}')
            pipeline = VoicePipeline(
                llm=llm,
                object_mapping_path=mapping_path,
                require_confirmation_for_side_effects=True,
            )

            result = pipeline.run_text_turn("把标号A的映射改成红色方块夹具", synthesize=False)

            loaded = load_object_mapping_document(mapping_path)
            self.assertEqual(loaded["version"], 1)
            self.assertEqual(loaded["markers"]["A"]["object"], "红色方块")
            self.assertIsNone(result.object_mapping_update)
            self.assertIsNotNone(result.pending_confirmation)
            assert result.pending_confirmation is not None
            self.assertEqual(result.pending_confirmation.action_type, ACTION_OBJECT_MAPPING_UPDATE)
            self.assertEqual(result.pending_confirmation.details["marker"], "A")
            self.assertEqual(result.pending_confirmation.details["object_name"], "红色方块夹具")
            self.assertIn("标号A", result.pending_confirmation.summary)
            self.assertIn("红色方块夹具", result.pending_confirmation.summary)
            self.assertIn("标号A", result.pending_confirmation.prompt)
            self.assertIn("红色方块夹具", result.pending_confirmation.prompt)
            self.assertIn("需要确认", result.response_text)

    def test_object_mapping_update_rejects_ambiguous_marker_rename_with_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "object_mapping.json"
            write_object_mapping_document(mapping_path, object_mapping_document())
            llm = FakeLlm('{"type":"tool_call","name":"edit_object_mapping","arguments":{}}')
            pipeline = VoicePipeline(
                llm=llm,
                object_mapping_path=mapping_path,
                require_confirmation_for_side_effects=True,
            )

            result = pipeline.run_text_turn("把标号D改为工具箱", synthesize=False)
            loaded = load_object_mapping_document(mapping_path)

            self.assertEqual(loaded["version"], 1)
            self.assertEqual(loaded["markers"]["D"]["object"], "空位")
            self.assertIsNone(result.object_mapping_update)
            self.assertIsNone(result.pending_confirmation)
            self.assertIn("这次没有执行修改", result.response_text)
            self.assertIn("把标号D的映射改为工具箱", result.response_text)
            self.assertIn("把标号D的物体改为工具箱", result.response_text)

    def test_object_mapping_update_accepts_tool_alias_for_explicit_mapping_request(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "object_mapping.json"
            write_object_mapping_document(mapping_path, object_mapping_document())
            llm = FakeLlm('{"type":"tool_call","name":"edit_object_mapping","arguments":{}}')
            pipeline = VoicePipeline(
                llm=llm,
                object_mapping_path=mapping_path,
                require_confirmation_for_side_effects=True,
            )

            result = pipeline.run_text_turn("把标号D的映射改为工具箱", synthesize=False)
            loaded = load_object_mapping_document(mapping_path)

            self.assertEqual(loaded["version"], 1)
            self.assertIsNone(result.object_mapping_update)
            self.assertIsNotNone(result.pending_confirmation)
            assert result.pending_confirmation is not None
            self.assertEqual(result.pending_confirmation.action_type, ACTION_OBJECT_MAPPING_UPDATE)
            self.assertEqual(result.pending_confirmation.details["marker"], "D")
            self.assertEqual(result.pending_confirmation.details["object_name"], "工具箱")

    def test_object_mapping_update_accepts_structured_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "object_mapping.json"
            write_object_mapping_document(mapping_path, object_mapping_document())
            llm = FakeLlm(
                '{"type":"tool_call","name":"update_object_mapping",'
                '"arguments":{"marker":"B","object_name":"夹具"}}'
            )
            pipeline = VoicePipeline(llm=llm, object_mapping_path=mapping_path)

            result = pipeline.run_text_turn("把标号B的物体改成夹具", synthesize=False)

            updated = load_object_mapping_document(mapping_path)
            self.assertEqual(updated["version"], 2)
            self.assertEqual(updated["markers"]["B"]["object"], "夹具")
            self.assertIsNotNone(result.object_mapping_update)

    def test_object_mapping_query_reads_mapping_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "object_mapping.json"
            write_object_mapping_document(mapping_path, object_mapping_document())
            llm = FakeLlm('{"type":"tool_call","name":"get_object_mapping","arguments":{}}')
            pipeline = VoicePipeline(llm=llm, object_mapping_path=mapping_path)

            result = pipeline.run_text_turn("查询标号B", synthesize=False)

            loaded = load_object_mapping_document(mapping_path)
            self.assertEqual(loaded["version"], 1)
            self.assertIsNotNone(result.object_mapping_query)
            assert result.object_mapping_query is not None
            self.assertEqual(result.object_mapping_query.marker, "B")
            self.assertEqual(result.object_mapping_query.object_name, "蓝色圆柱")
            self.assertIn("当前映射：标号B 对应蓝色圆柱", result.response_text)

    def test_object_mapping_table_query_reads_all_mappings_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "object_mapping.json"
            document = object_mapping_document()
            document["markers"]["C"]["enabled"] = False
            write_object_mapping_document(mapping_path, document)
            llm = FakeLlm('{"type":"tool_call","name":"get_object_mapping","arguments":{}}')
            pipeline = VoicePipeline(llm=llm, object_mapping_path=mapping_path)

            result = pipeline.run_text_turn("说一说当前的物体映射表", synthesize=False)

            loaded = load_object_mapping_document(mapping_path)
            self.assertEqual(loaded["version"], 1)
            self.assertIsNone(result.object_mapping_query)
            self.assertIsNotNone(result.object_mapping_table_query)
            assert result.object_mapping_table_query is not None
            self.assertEqual(result.object_mapping_table_query.version, 1)
            table_entries = [
                (entry.marker, entry.object_name, entry.enabled)
                for entry in result.object_mapping_table_query.mappings
            ]
            self.assertEqual(
                table_entries,
                [
                    ("A", "红色方块", True),
                    ("B", "蓝色圆柱", True),
                    ("C", "扳手", False),
                    ("D", "空位", True),
                ],
            )
            self.assertIn("当前物体映射表", result.response_text)
            self.assertIn("标号A 对应红色方块", result.response_text)
            self.assertIn("标号C 对应扳手（当前未启用）", result.response_text)

    def test_object_mapping_table_query_fallback_detects_all_mapping_phrase(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "object_mapping.json"
            write_object_mapping_document(mapping_path, object_mapping_document())
            llm = FakeLlm("普通回复")
            pipeline = VoicePipeline(llm=llm, object_mapping_path=mapping_path)

            result = pipeline.run_text_turn("查看全部物体映射", synthesize=False)

            self.assertIsNotNone(result.object_mapping_table_query)
            self.assertIn("标号D 对应空位", result.response_text)

    def test_object_mapping_table_query_overrides_wrong_update_tool_route(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "object_mapping.json"
            write_object_mapping_document(mapping_path, object_mapping_document())
            llm = FakeLlm('{"type":"tool_call","name":"update_object_mapping","arguments":{}}')
            pipeline = VoicePipeline(llm=llm, object_mapping_path=mapping_path)

            result = pipeline.run_text_turn("查看当前物体映射表", synthesize=False)

            loaded = load_object_mapping_document(mapping_path)
            self.assertEqual(loaded["version"], 1)
            self.assertIsNone(result.object_mapping_update)
            self.assertIsNotNone(result.object_mapping_table_query)

    def test_object_mapping_update_missing_marker_does_not_write(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "object_mapping.json"
            write_object_mapping_document(mapping_path, object_mapping_document())
            llm = FakeLlm('{"type":"tool_call","name":"update_object_mapping","arguments":{}}')
            pipeline = VoicePipeline(llm=llm, object_mapping_path=mapping_path)

            result = pipeline.run_text_turn("把这个改成红色方块", synthesize=False)

            updated = load_object_mapping_document(mapping_path)
            self.assertEqual(updated["version"], 1)
            self.assertIsNone(result.object_mapping_update)
            self.assertIn("请使用“标号A、标号B、标号C 或 标号D”", result.response_text)

    def test_object_mapping_update_missing_object_does_not_write(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "object_mapping.json"
            write_object_mapping_document(mapping_path, object_mapping_document())
            llm = FakeLlm('{"type":"tool_call","name":"update_object_mapping","arguments":{}}')
            pipeline = VoicePipeline(llm=llm, object_mapping_path=mapping_path)

            result = pipeline.run_text_turn("把标号A的映射改成", synthesize=False)

            updated = load_object_mapping_document(mapping_path)
            self.assertEqual(updated["version"], 1)
            self.assertIsNone(result.object_mapping_update)
            self.assertIn("请说明这个字母现在对应哪个物体", result.response_text)

    def test_object_mapping_tool_arguments_cannot_supply_missing_object_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "object_mapping.json"
            write_object_mapping_document(mapping_path, object_mapping_document())
            llm = FakeLlm(
                '{"type":"tool_call","name":"update_object_mapping",'
                '"arguments":{"marker":"A","object_name":"夹具"}}'
            )
            pipeline = VoicePipeline(llm=llm, object_mapping_path=mapping_path)

            result = pipeline.run_text_turn("把标号A的映射改成", synthesize=False)

            updated = load_object_mapping_document(mapping_path)
            self.assertEqual(updated["version"], 1)
            self.assertEqual(updated["markers"]["A"]["object"], "红色方块")
            self.assertIsNone(result.object_mapping_update)
            self.assertIn("请说明这个字母现在对应哪个物体", result.response_text)

    def test_object_mapping_update_unknown_marker_does_not_write(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "object_mapping.json"
            write_object_mapping_document(mapping_path, object_mapping_document())
            llm = FakeLlm('{"type":"tool_call","name":"update_object_mapping","arguments":{}}')
            pipeline = VoicePipeline(llm=llm, object_mapping_path=mapping_path)

            result = pipeline.run_text_turn("把标号E的映射改成托盘", synthesize=False)

            updated = load_object_mapping_document(mapping_path)
            self.assertEqual(updated["version"], 1)
            self.assertIsNone(result.object_mapping_update)
            self.assertIn("当前只支持标号A、标号B、标号C、标号D", result.response_text)

    def test_object_mapping_tool_arguments_cannot_bypass_labeled_marker_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "object_mapping.json"
            write_object_mapping_document(mapping_path, object_mapping_document())
            llm = FakeLlm(
                '{"type":"tool_call","name":"update_object_mapping",'
                '"arguments":{"marker":"A","object_name":"夹具"}}'
            )
            pipeline = VoicePipeline(llm=llm, object_mapping_path=mapping_path)

            result = pipeline.run_text_turn("把 A 改成夹具", synthesize=False)

            updated = load_object_mapping_document(mapping_path)
            self.assertEqual(updated["version"], 1)
            self.assertEqual(updated["markers"]["A"]["object"], "红色方块")
            self.assertIsNone(result.object_mapping_update)
            self.assertIn("请使用“标号A、标号B、标号C 或 标号D”", result.response_text)

    def test_object_mapping_tool_arguments_must_match_labeled_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "object_mapping.json"
            write_object_mapping_document(mapping_path, object_mapping_document())
            llm = FakeLlm(
                '{"type":"tool_call","name":"update_object_mapping",'
                '"arguments":{"marker":"B","object_name":"夹具"}}'
            )
            pipeline = VoicePipeline(llm=llm, object_mapping_path=mapping_path)

            result = pipeline.run_text_turn("把标号A的映射改成夹具", synthesize=False)

            updated = load_object_mapping_document(mapping_path)
            self.assertEqual(updated["version"], 1)
            self.assertIsNone(result.object_mapping_update)
            self.assertIn("用户原文中的标号与工具参数不一致", result.response_text)

    def test_object_mapping_query_intent_overrides_wrong_update_tool_route(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "object_mapping.json"
            write_object_mapping_document(mapping_path, object_mapping_document())
            llm = FakeLlm(
                '{"type":"tool_call","name":"update_object_mapping",'
                '"arguments":{"marker":"A","object_name":"夹具"}}'
            )
            pipeline = VoicePipeline(llm=llm, object_mapping_path=mapping_path)

            result = pipeline.run_text_turn("查询标号A", synthesize=False)

            loaded = load_object_mapping_document(mapping_path)
            self.assertEqual(loaded["version"], 1)
            self.assertEqual(loaded["markers"]["A"]["object"], "红色方块")
            self.assertIsNotNone(result.object_mapping_query)
            self.assertIsNone(result.object_mapping_update)

    def test_object_mapping_update_intent_overrides_wrong_query_tool_route(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "object_mapping.json"
            write_object_mapping_document(mapping_path, object_mapping_document())
            llm = FakeLlm(
                '{"type":"tool_call","name":"get_object_mapping",'
                '"arguments":{"marker":"A","object_name":"夹具"}}'
            )
            pipeline = VoicePipeline(llm=llm, object_mapping_path=mapping_path)

            result = pipeline.run_text_turn("把标号A的映射改成夹具", synthesize=False)

            updated = load_object_mapping_document(mapping_path)
            self.assertEqual(updated["version"], 2)
            self.assertEqual(updated["markers"]["A"]["object"], "夹具")
            self.assertIsNone(result.object_mapping_query)
            self.assertIsNotNone(result.object_mapping_update)

    def test_object_grasp_marker_target_resolves_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "object_mapping.json"
            write_object_mapping_document(mapping_path, object_mapping_document())
            llm = FakeLlm('{"type":"final","content":"收到。"}')
            pipeline = VoicePipeline(llm=llm, object_mapping_path=mapping_path)

            result = pipeline.run_text_turn("帮我抓取标号A", synthesize=False)

            self.assertIsNotNone(result.object_grasp_intent)
            assert result.object_grasp_intent is not None
            self.assertEqual(result.object_grasp_intent.marker, "A")
            self.assertEqual(result.object_grasp_intent.object_name, "红色方块")
            self.assertEqual(result.object_grasp_intent.target_source, "marker")
            self.assertIn("已识别抓取目标：标号A，对应红色方块", result.response_text)
            loaded = load_object_mapping_document(mapping_path)
            self.assertEqual(loaded["version"], 1)

    def test_object_grasp_object_name_target_resolves_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "object_mapping.json"
            write_object_mapping_document(mapping_path, object_mapping_document())
            llm = FakeLlm('{"type":"final","content":"收到。"}')
            pipeline = VoicePipeline(llm=llm, object_mapping_path=mapping_path)

            result = pipeline.run_text_turn("我需要扳手", synthesize=False)

            self.assertIsNotNone(result.object_grasp_intent)
            assert result.object_grasp_intent is not None
            self.assertEqual(result.object_grasp_intent.marker, "C")
            self.assertEqual(result.object_grasp_intent.object_name, "扳手")
            self.assertEqual(result.object_grasp_intent.target_source, "object_name")
            self.assertIn("已识别抓取目标：扳手，对应标号C", result.response_text)

    def test_object_grasp_confirmation_mode_returns_pending_without_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "object_mapping.json"
            write_object_mapping_document(mapping_path, object_mapping_document())
            llm = FakeLlm('{"type":"final","content":"收到。"}')
            pipeline = VoicePipeline(
                llm=llm,
                object_mapping_path=mapping_path,
                require_confirmation_for_side_effects=True,
            )

            result = pipeline.run_text_turn("帮我抓取标号A", synthesize=False)
            plans = build_voice_ros2_plan(result)

            self.assertIsNone(result.object_grasp_intent)
            self.assertIsNotNone(result.pending_confirmation)
            assert result.pending_confirmation is not None
            self.assertEqual(result.pending_confirmation.action_type, ACTION_OBJECT_GRASP_EXECUTION)
            self.assertEqual(result.pending_confirmation.details["marker"], "A")
            self.assertEqual(result.pending_confirmation.details["object_name"], "红色方块")
            self.assertIn("这个操作需要确认", result.response_text)
            self.assertIn("标号A", result.response_text)
            self.assertIn("红色方块", result.response_text)
            self.assertEqual([plan.topic for plan in plans], [DEFAULT_TRANSCRIPT_TOPIC, DEFAULT_RESPONSE_TOPIC])

    def test_object_grasp_unknown_object_does_not_create_intent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "object_mapping.json"
            write_object_mapping_document(mapping_path, object_mapping_document())
            llm = FakeLlm('{"type":"final","content":"收到。"}')
            pipeline = VoicePipeline(llm=llm, object_mapping_path=mapping_path)

            result = pipeline.run_text_turn("给我绿色球", synthesize=False)

            self.assertIsNone(result.object_grasp_intent)
            self.assertIn("抓取目标未确认", result.response_text)
            self.assertIn("没有找到已启用的物体名称", result.response_text)

    def test_object_grasp_ambiguous_object_name_does_not_create_intent(self) -> None:
        document = object_mapping_document()
        document["markers"]["B"]["object"] = "红色方块"
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "object_mapping.json"
            write_object_mapping_document(mapping_path, document)
            llm = FakeLlm('{"type":"final","content":"收到。"}')
            pipeline = VoicePipeline(llm=llm, object_mapping_path=mapping_path)

            result = pipeline.run_text_turn("给我红色方块", synthesize=False)

            self.assertIsNone(result.object_grasp_intent)
            self.assertIn("请改用具体标号", result.response_text)
            self.assertIn("标号A", result.response_text)
            self.assertIn("标号B", result.response_text)

    def test_object_grasp_disabled_mapping_entry_does_not_create_intent(self) -> None:
        document = object_mapping_document()
        document["markers"]["C"]["enabled"] = False
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "object_mapping.json"
            write_object_mapping_document(mapping_path, document)
            llm = FakeLlm('{"type":"final","content":"收到。"}')
            pipeline = VoicePipeline(llm=llm, object_mapping_path=mapping_path)

            marker_result = pipeline.run_text_turn("抓取标号C", synthesize=False)
            name_result = pipeline.run_text_turn("我需要扳手", synthesize=False)

            self.assertIsNone(marker_result.object_grasp_intent)
            self.assertIn("标号C 对应扳手，但该标号当前未启用", marker_result.response_text)
            self.assertIsNone(name_result.object_grasp_intent)
            self.assertIn("物体名称“扳手”只对应未启用的标号C", name_result.response_text)

    def test_object_grasp_bare_letter_does_not_create_intent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "object_mapping.json"
            write_object_mapping_document(mapping_path, object_mapping_document())
            llm = FakeLlm('{"type":"final","content":"收到。"}')
            pipeline = VoicePipeline(llm=llm, object_mapping_path=mapping_path)

            result = pipeline.run_text_turn("抓取 A", synthesize=False)

            self.assertIsNone(result.object_grasp_intent)
            self.assertIn("不能只说 A、B、C 或 D", result.response_text)

    def test_object_grasp_question_like_text_does_not_route(self) -> None:
        llm = FakeLlm('{"type":"final","content":"我可以帮你确认映射。"}')
        pipeline = VoicePipeline(llm=llm)

        question_result = pipeline.run_text_turn("给我看看红色方块", synthesize=False)
        explanation_result = pipeline.run_text_turn("我需要说明扳手怎么抓取", synthesize=False)

        self.assertIsNone(question_result.object_grasp_intent)
        self.assertIsNone(explanation_result.object_grasp_intent)
        self.assertFalse(should_grasp_object("给我看看红色方块"))
        self.assertFalse(should_grasp_object("我需要说明扳手怎么抓取"))

    def test_arm_runtime_query_reads_json_request_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            arm_rules_path = Path(temp_dir) / "arm_rules.json"
            write_arm_rule_document(
                arm_rules_path,
                {
                    "arm_capture": "True",
                    "arm_capture_goal": "C",
                    "arm_capture_object": "扳手",
                    "arm_decelerate": "0.5",
                    "arm_safety_distance": "0.4",
                },
            )
            llm = FakeLlm('{"type":"final","content":"收到。"}')
            pipeline = VoicePipeline(llm=llm, arm_rules_path=arm_rules_path)

            result = pipeline.run_text_turn("当前机械臂抓取请求是什么", synthesize=False)

            loaded = load_arm_rule_document(arm_rules_path)
            self.assertEqual(loaded["arm_capture"], "True")
            self.assertEqual(loaded["arm_capture_goal"], "C")
            self.assertIsNotNone(result.arm_runtime_query)
            assert result.arm_runtime_query is not None
            self.assertTrue(result.arm_runtime_query.capture_requested)
            self.assertEqual(result.arm_runtime_query.capture_goal, "C")
            self.assertEqual(result.arm_runtime_query.capture_object, "扳手")
            self.assertIn("有待执行抓取请求", result.response_text)
            self.assertIn("标号C（扳手）", result.response_text)
            self.assertEqual(llm.calls, [])

    def test_arm_deceleration_parser_accepts_percent_and_plain_number(self) -> None:
        percent_target, percent_error = extract_arm_deceleration_target_percent("限速到30%")
        plain_target, plain_error = extract_arm_deceleration_target_percent("把机械臂速度降到 30")

        self.assertIsNone(percent_error)
        self.assertEqual(percent_target, 30.0)
        self.assertIsNone(plain_error)
        self.assertEqual(plain_target, 30.0)
        self.assertTrue(should_request_arm_deceleration("速度限制到30%"))
        self.assertFalse(should_request_arm_deceleration("当前限速规则是什么"))

    def test_arm_deceleration_confirmation_mode_returns_pending_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            arm_rules_path = Path(temp_dir) / "arm_rules.json"
            write_arm_rule_document(
                arm_rules_path,
                {
                    "arm_capture": "True",
                    "arm_capture_goal": "B",
                    "arm_decelerate": "0.5",
                    "arm_safety_distance": "0.4",
                },
            )
            llm = FakeLlm('{"type":"final","content":"收到。"}')
            pipeline = VoicePipeline(
                llm=llm,
                arm_rules_path=arm_rules_path,
                require_confirmation_for_side_effects=True,
            )

            result = pipeline.run_text_turn("减速到30%", synthesize=False)
            plans = build_voice_ros2_plan(result)
            loaded = load_arm_rule_document(arm_rules_path)

            self.assertEqual(loaded["arm_decelerate"], "0.5")
            self.assertIsNone(result.arm_deceleration_request)
            self.assertIsNotNone(result.pending_confirmation)
            assert result.pending_confirmation is not None
            self.assertEqual(result.pending_confirmation.action_type, ACTION_SPEED_CHANGE)
            self.assertEqual(result.pending_confirmation.details["target_speed"], "30%")
            self.assertEqual(result.pending_confirmation.details["target_speed_percent"], 30.0)
            self.assertEqual(result.pending_confirmation.details["arm_decelerate"], "0.3")
            self.assertIn("30%", result.pending_confirmation.summary)
            self.assertIn("这个操作需要确认", result.response_text)
            self.assertEqual([plan.topic for plan in plans], [DEFAULT_TRANSCRIPT_TOPIC, DEFAULT_RESPONSE_TOPIC])
            self.assertEqual(llm.calls, [])

    def test_arm_deceleration_without_percent_rejects_without_pending_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            arm_rules_path = Path(temp_dir) / "arm_rules.json"
            write_arm_rule_document(arm_rules_path, {"arm_decelerate": "0.5"})
            pipeline = VoicePipeline(
                llm=FakeLlm('{"type":"final","content":"收到。"}'),
                arm_rules_path=arm_rules_path,
                require_confirmation_for_side_effects=True,
            )

            result = pipeline.run_text_turn("减速", synthesize=False)
            loaded = load_arm_rule_document(arm_rules_path)

            self.assertIsNone(result.pending_confirmation)
            self.assertIsNone(result.arm_deceleration_request)
            self.assertEqual(loaded["arm_decelerate"], "0.5")
            self.assertIn("请说清楚目标速度百分比", result.response_text)

    def test_arm_deceleration_no_confirmation_mode_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            arm_rules_path = Path(temp_dir) / "arm_rules.json"
            write_arm_rule_document(
                arm_rules_path,
                {
                    "arm_capture": "True",
                    "arm_capture_goal": "C",
                    "arm_decelerate": "0.5",
                    "arm_safety_distance": "0.4",
                },
            )
            pipeline = VoicePipeline(llm=FakeLlm(), arm_rules_path=arm_rules_path)

            result = pipeline.run_text_turn("速度限制到30", synthesize=False)
            plans = build_voice_ros2_plan(result)
            loaded = load_arm_rule_document(arm_rules_path)

            self.assertIsNotNone(result.arm_deceleration_request)
            assert result.arm_deceleration_request is not None
            self.assertEqual(result.arm_deceleration_request.arm_decelerate, "0.3")
            self.assertEqual(loaded["arm_decelerate"], "0.3")
            self.assertEqual(loaded["arm_capture"], "True")
            self.assertEqual(loaded["arm_capture_goal"], "C")
            self.assertEqual(loaded["arm_safety_distance"], "0.4")
            self.assertIn("目标速度为 30%", result.response_text)
            self.assertEqual([plan.topic for plan in plans], [DEFAULT_TRANSCRIPT_TOPIC, DEFAULT_RESPONSE_TOPIC])


class LlmPromptTest(unittest.TestCase):
    def test_build_prompt_forces_non_thinking_chat_prefix(self) -> None:
        prompt = build_prompt("system", "user")

        self.assertIn("enable_thinking=false", prompt)
        self.assertIn("<|im_start|>assistant\n<think>\n\n</think>\n\n", prompt)

    def test_build_prompt_passes_enable_thinking_false_to_tokenizer(self) -> None:
        tokenizer = FakeTokenizer()

        prompt = build_prompt("system", "user", tokenizer=tokenizer)

        self.assertEqual(prompt, "templated:user")
        self.assertEqual(tokenizer.extra_context, {"enable_thinking": False})

    def test_strip_thinking_text_removes_generated_think_block(self) -> None:
        self.assertEqual(strip_thinking_text("<think>hidden</think>\n公开回答"), "公开回答")

    def test_strip_thinking_text_removes_unclosed_think_block(self) -> None:
        self.assertEqual(strip_thinking_text("公开回答\n<think>hidden reasoning"), "公开回答")


class SafetyRoutingTest(unittest.TestCase):
    def test_normalize_asr_text_corrects_robot_safety_terms(self) -> None:
        self.assertEqual(normalize_asr_text("机器臂 机械皮 气崩 线速 光山"), "机械臂 机械臂 气泵 限速 光栅")
        self.assertEqual(normalize_asr_text("机停"), "急停")
        self.assertEqual(normalize_asr_text("你好机械皮出手"), "你好机械臂助手")
        self.assertEqual(normalize_asr_text("詳細說一說當前的限速規則"), "详细说一说当前的限速规则")
        self.assertEqual(normalize_asr_text("当前人员安全，距离是多少？"), "当前人员安全距离是多少")
        self.assertEqual(normalize_asr_text("把人员安全距离改成 1.2 米。"), "把人员安全距离改成 1.2 米")
        self.assertEqual(
            convert_traditional_to_simplified("請說明臺灣機器人規劃軌跡"),
            "请说明台湾机器人规划轨迹",
        )

    def test_asr_noise_returns_no_answer_without_llm_or_tts(self) -> None:
        llm = FakeLlm("不应调用")
        tts = FakeTts()
        pipeline = VoicePipeline(asr=FakeAsr("thank you for watching"), llm=llm, tts=tts)

        with self.assertLogs("local_safety_assistant.stack.pipeline", level="INFO") as logs:
            result = pipeline.run_audio_file(Path("/tmp/noise.wav"), synthesize=True)

        self.assertEqual(result.response_text, "")
        self.assertEqual(result.metadata, {"no_answer": True, "reason": "asr_noise"})
        self.assertEqual(llm.calls, [])
        self.assertEqual(tts.calls, [])
        self.assertTrue(any("已忽略疑似 ASR 噪声" in line for line in logs.output))

    def test_asr_noise_ignores_marker_repetition_and_japanese_end_phrase(self) -> None:
        self.assertTrue(should_ignore_asr_noise("C 石石石石"))
        self.assertTrue(should_ignore_asr_noise("おわり"))
        self.assertFalse(should_ignore_asr_noise("查询标号C"))
        self.assertFalse(should_ignore_asr_noise("确认更新"))

    def test_sanitize_spoken_response_removes_internal_correction_lines(self) -> None:
        response = sanitize_spoken_response(
            "收到，正在修正语音识别错误：\n"
            "1. “机械皮出手” → 机械臂\n"
            "2. “出手” → 急停\n\n"
            "**安全指令：**\n"
            "机械臂急停！"
        )

        self.assertEqual(response, "机械臂急停！")

    def test_sanitize_spoken_response_removes_route_chatter_lines(self) -> None:
        response = sanitize_spoken_response(
            "收到。\n"
            "ROUTE:rule_editor_9b\n"
            "将立即处理您的请求。"
        )

        self.assertEqual(response, "收到。")

    def test_sanitize_spoken_response_falls_back_for_raw_json(self) -> None:
        response = sanitize_spoken_response(
            '{"id":"stop_on_person_intrusion","conditions":{"person_distance_m":{"lt":0.8}},"action":{"type":"stop_motion"}}'
        )

        self.assertEqual(response, "收到。")

    def test_rule_read_raw_json_response_uses_chinese_fallback_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rules_path = Path(temp_dir) / "rules.json"
            write_rule_document(rules_path, complex_rule_document())
            llm = FakeLlm(
                [
                    '{"type":"tool_call","name":"load_rules","arguments":{}}',
                    json.dumps(complex_rule_document()["rules"][0], ensure_ascii=False),
                ]
            )
            pipeline = VoicePipeline(llm=llm, rules_path=rules_path)

            result = pipeline.run_text_turn("当前安全规则是什么", synthesize=False)

        self.assertIn("规则文件已读取", result.response_text)
        self.assertIn("人员侵入停机规则", result.response_text)
        self.assertNotIn("conditions", result.response_text)
        self.assertNotIn("action", result.response_text)

    def test_sanitize_spoken_response_falls_back_when_only_route_chatter_remains(self) -> None:
        response = sanitize_spoken_response(
            "收到，已确认安全规则已就绪。\n"
            "ROUTE:rule_editor_9b\n"
            "将立即处理您的请求。"
        )

        self.assertEqual(response, "收到。")

    def test_sanitize_spoken_response_removes_stray_semicolon_artifacts_between_sentences(self) -> None:
        response = sanitize_spoken_response(
            "当前防护门状态正常。\n"
            "机械臂可以继续保持待命。"
        )

        self.assertEqual(response, "当前防护门状态正常。机械臂可以继续保持待命。")
        self.assertNotIn("。；", response)

    def test_sanitize_spoken_response_preserves_legitimate_chinese_semicolons(self) -> None:
        response = sanitize_spoken_response(
            "人员进入安全区；机械臂会立即停止。\n"
            "防护门打开；需要确认复位。"
        )

        self.assertEqual(response, "人员进入安全区；机械臂会立即停止。防护门打开；需要确认复位。")
        self.assertIn("安全区；机械臂", response)
        self.assertIn("打开；需要", response)

    def test_should_edit_rules_detects_rule_change_intent(self) -> None:
        self.assertTrue(should_edit_rules("请把限速规则改成 0.1"))
        self.assertTrue(should_edit_rules("把人员安全距离调整为0.4米"))
        self.assertTrue(should_edit_rules("暂时禁用人员安全距离规则"))
        self.assertTrue(should_edit_rules("恢复人员安全距离规则"))
        self.assertFalse(should_edit_rules("可以更改吗"))
        self.assertFalse(should_edit_rules("这个能调整吗"))
        self.assertFalse(should_edit_rules("能修改规则吗"))
        self.assertFalse(should_edit_rules("请说明限速规则"))
        self.assertFalse(should_edit_rules("请说明防护门打开规则"))
        self.assertTrue(should_reject_rule_authoring("请新增一条安全规则"))
        self.assertTrue(should_reject_rule_authoring("请删除防护门打开规则"))
        self.assertFalse(should_reject_rule_authoring("请把限速规则改成 0.1"))

    def test_should_update_object_mapping_detects_marker_reassignment(self) -> None:
        self.assertTrue(should_update_object_mapping("把标号A的映射改成红色方块"))
        self.assertTrue(should_update_object_mapping("把标号A的物体改成红色方块"))
        self.assertTrue(should_update_object_mapping("标号B贴到新的物体上，叫夹具"))
        self.assertFalse(should_update_object_mapping("把标号D改为工具箱"))
        self.assertFalse(should_update_object_mapping("这个标号D以后叫托盘"))
        self.assertFalse(should_update_object_mapping("把 A 改成红色方块"))
        self.assertFalse(should_update_object_mapping("标号A 是什么"))
        self.assertFalse(should_update_object_mapping("把限速规则改成 0.1"))
        self.assertTrue(should_query_object_mapping("查询标号A"))
        self.assertTrue(should_query_object_mapping("标号B对应什么"))
        self.assertFalse(should_query_object_mapping("A 是什么"))
        self.assertTrue(should_query_object_mapping_table("说一说当前的物体映射表"))
        self.assertTrue(should_query_object_mapping_table("查看全部物体映射"))
        self.assertFalse(should_query_object_mapping_table("查询标号A"))
        self.assertFalse(should_query_object_mapping_table("修改当前物体映射表"))
        self.assertTrue(should_guide_workspace_snapshot("当前工作区什么情况"))
        self.assertTrue(should_guide_workspace_snapshot("当前工作环境什么情况"))
        self.assertFalse(should_guide_workspace_snapshot("当前物体映射表"))
        self.assertFalse(should_guide_workspace_snapshot("当前安全规则是什么"))
        self.assertTrue(should_query_arm_runtime("当前机械臂抓取请求是什么"))
        self.assertFalse(should_query_arm_runtime("帮我抓取标号A"))

        marker, object_name, error = extract_object_mapping_update("标号B贴到新的物体上，叫夹具")
        self.assertEqual((marker, object_name, error), ("B", "夹具", None))

        marker, object_name, error = extract_object_mapping_update("标号A的物体对应的是水杯")
        self.assertEqual((marker, object_name, error), ("A", "水杯", None))

        marker, error = extract_object_mapping_query("标号C是什么")
        self.assertEqual((marker, error), ("C", None))

    def test_should_grasp_object_detects_targets_without_mapping_side_effects(self) -> None:
        self.assertTrue(should_grasp_object("抓取标号A"))
        self.assertTrue(should_grasp_object("帮我抓取红色方块"))
        self.assertTrue(should_grasp_object("给我蓝色圆柱"))
        self.assertTrue(should_grasp_object("我需要扳手"))
        self.assertFalse(should_grasp_object("查询标号A"))
        self.assertFalse(should_grasp_object("标号A对应什么"))
        self.assertFalse(should_grasp_object("把标号A改成夹具"))
        self.assertFalse(should_grasp_object("给我看看红色方块"))
        self.assertFalse(should_grasp_object("我需要说明扳手怎么抓取"))

        marker, object_name, target_source, error = extract_object_grasp_target("抓取标号B")
        self.assertEqual((marker, object_name, target_source, error), ("B", None, "marker", None))

        marker, object_name, target_source, error = extract_object_grasp_target("给我蓝色圆柱")
        self.assertEqual((marker, object_name, target_source, error), (None, "蓝色圆柱", "object_name", None))

        marker, object_name, target_source, error = extract_object_grasp_target("抓取 A")
        self.assertEqual((marker, object_name, target_source), (None, None, None))
        self.assertIsNotNone(error)

    def test_should_read_rules_detects_current_rule_queries_only(self) -> None:
        self.assertTrue(should_read_rules("当前安全规则是什么"))
        self.assertTrue(should_read_rules("已有规则列表"))
        self.assertTrue(should_read_rules("启用规则详情"))
        self.assertTrue(should_read_rules("请说明机械臂急停规则"))
        self.assertTrue(should_read_rules("请说明防护门打开规则"))
        self.assertTrue(should_read_rules("詳細說一說當前的限速規則"))
        self.assertFalse(should_read_rules("请禁用人员附近限速规则"))
        self.assertFalse(should_read_rules("现在急停吗"))

    def test_agent_router_prompt_and_parser_accept_tool_and_final_routes(self) -> None:
        prompt = build_agent_router_prompt("把人员安全距离调整为1.2米")
        edit_decision = parse_agent_route_decision('{"type":"tool_call","name":"edit_rules","arguments":{}}')
        read_decision = parse_agent_route_decision('{"route":"load_rules","arguments":{}}')
        vision_decision = parse_agent_route_decision(
            '{"type":"tool_call","name":"analyze_environment_vision","arguments":{}}'
        )
        mapping_decision = parse_agent_route_decision(
            '{"type":"tool_call","name":"update_object_mapping","arguments":{}}'
        )
        mapping_query_decision = parse_agent_route_decision(
            '{"type":"tool_call","name":"get_object_mapping","arguments":{}}'
        )
        final_decision = parse_agent_route_decision('{"type":"final","content":"不客气。"}')

        self.assertIn("结构化工具路由器", prompt)
        self.assertIn("调整", prompt)
        self.assertIn("analyze_environment_vision", prompt)
        self.assertIn("get_object_mapping", prompt)
        self.assertIn("update_object_mapping", prompt)
        self.assertIn("物体映射表", prompt)
        self.assertNotIn("防护门", prompt)
        self.assertNotIn("光栅", prompt)
        self.assertIsNotNone(edit_decision)
        assert edit_decision is not None
        self.assertIsNotNone(edit_decision.tool_request)
        assert edit_decision.tool_request is not None
        self.assertEqual(edit_decision.tool_request.name, "edit_rules")
        self.assertIsNotNone(read_decision)
        assert read_decision is not None
        self.assertIsNotNone(read_decision.tool_request)
        assert read_decision.tool_request is not None
        self.assertEqual(read_decision.tool_request.name, "load_rules")
        self.assertIsNotNone(vision_decision)
        assert vision_decision is not None
        self.assertIsNotNone(vision_decision.tool_request)
        assert vision_decision.tool_request is not None
        self.assertEqual(vision_decision.tool_request.name, "analyze_environment_vision")
        self.assertIsNotNone(mapping_decision)
        assert mapping_decision is not None
        self.assertIsNotNone(mapping_decision.tool_request)
        assert mapping_decision.tool_request is not None
        self.assertEqual(mapping_decision.tool_request.name, "update_object_mapping")
        self.assertIsNotNone(mapping_query_decision)
        assert mapping_query_decision is not None
        self.assertIsNotNone(mapping_query_decision.tool_request)
        assert mapping_query_decision.tool_request is not None
        self.assertEqual(mapping_query_decision.tool_request.name, "get_object_mapping")
        self.assertIsNotNone(final_decision)
        assert final_decision is not None
        self.assertEqual(final_decision.final_content, "不客气。")

    def test_vision_trigger_detection_and_prompt(self) -> None:
        artifact = VisionImageArtifact(
            image_path=Path("/tmp/snapshot.jpg"),
            metadata={"stamp": "2026-06-09T12:00:00", "frame_id": "camera_color"},
        )
        prompt = build_vision_analysis_prompt("调用视觉，分析下当前工作环境", artifact)

        self.assertTrue(should_analyze_vision("调用视觉，分析下当前工作环境"))
        self.assertTrue(should_analyze_vision("请视觉分析当前画面"))
        self.assertFalse(should_analyze_vision("当前工作环境什么情况"))
        self.assertFalse(should_analyze_vision("当前工作区什么情况"))
        self.assertFalse(should_analyze_vision("当前安全规则是什么"))
        self.assertIn("<image>", prompt)
        self.assertIn("4 到 6 句简洁中文", prompt)
        self.assertIn("第一句简述画面中的整体场景", prompt)
        self.assertIn("第二句描述主要可见对象及其大致位置关系", prompt)
        self.assertIn("不要编造", prompt)
        self.assertIn("camera_color", prompt)

    def test_vision_response_validator_rejects_generic_acknowledgements(self) -> None:
        for text in ("", "收到", "收到。", "好的", "OK", "ok。"):
            with self.subTest(text=text):
                self.assertTrue(is_unusable_vision_analysis_response(text))

        self.assertFalse(
            is_unusable_vision_analysis_response(
                "画面显示工位可见，未发现明显人员进入机械臂工作区。请继续确认气泵附近无遮挡。"
            )
        )

    def test_snapshot_trigger_message_requires_valid_image_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "snapshot.jpg"
            image_path.write_bytes(b"jpeg")
            message = json.dumps(
                {
                    "image_path": str(image_path),
                    "stamp": "2026-06-09T12:00:00",
                    "frame_id": "camera_color",
                },
                ensure_ascii=False,
            )

            artifact = snapshot_from_trigger_message(message, source="/vision/capture_snapshot")

        self.assertEqual(artifact.image_path, image_path.resolve())
        self.assertEqual(artifact.source, "/vision/capture_snapshot")
        self.assertEqual(artifact.metadata["frame_id"], "camera_color")

    def test_vision_snapshot_node_converts_bgr8_to_rgb_and_builds_payload(self) -> None:
        message = SimpleNamespace(
            height=1,
            width=2,
            step=6,
            encoding="bgr8",
            data=bytes([10, 20, 30, 40, 50, 60]),
            header=SimpleNamespace(
                frame_id="camera_color",
                stamp=SimpleNamespace(sec=10, nanosec=25),
            ),
        )

        array = image_message_to_rgb_array(message)
        payload = json.loads(snapshot_payload(Path("/tmp/snapshot.jpg"), message, topic="/camera/color/image_raw"))

        self.assertEqual(array.tolist(), [[[30, 20, 10], [60, 50, 40]]])
        self.assertEqual(payload["image_path"], "/tmp/snapshot.jpg")
        self.assertEqual(payload["stamp"], "10.000000025")
        self.assertEqual(payload["frame_id"], "camera_color")
        self.assertEqual(payload["topic"], "/camera/color/image_raw")

    def test_route_marker_detection_is_case_insensitive_and_stripped(self) -> None:
        self.assertTrue(routes_to_rule_editor("route:rule_editor_9b\n处理规则"))
        self.assertEqual(strip_route_marker("route:rule_editor_9b\n处理规则"), "处理规则")

    def test_agent_tool_request_detection_accepts_text_and_json(self) -> None:
        text_request = parse_agent_tool_request("TOOL:load_rules")
        punctuated_request = parse_agent_tool_request("TOOL：load_rules。")
        json_request = parse_agent_tool_request('{"tool":"edit_rules","argument":"disable"}')
        structured_request = parse_agent_tool_request(
            '{"type":"tool_call","name":"edit_rules","arguments":{'
            '"rule_id":"slow_near_unknown_object","changes":{"enabled":false}}}'
        )
        mapping_request = parse_agent_tool_request(
            '{"type":"tool_call","name":"update_object_mapping","arguments":{"marker":"A","object":"夹具"}}'
        )
        mapping_alias_request = parse_agent_tool_request(
            '{"type":"tool_call","name":"edit_object_mapping","arguments":{"marker":"A","object":"夹具"}}'
        )
        mapping_query_request = parse_agent_tool_request(
            '{"type":"tool_call","name":"get_object_mapping","arguments":{"marker":"A"}}'
        )
        unknown_request = parse_agent_tool_request('{"type":"tool_call","name":"unknown_tool","arguments":{}}')
        prefixed_structured_request = parse_agent_tool_request(
            '收到，已为您修改规则。 json; {"type":"tool_call","name":"edit_rules",'
            '"arguments":{"rules":[{"id":"safety_distance","conditions":{"distance_m":1.0}}]}}'
        )
        final_content = parse_agent_final_response('{"type":"final","content":"当前状态正常。"}')

        self.assertIsNotNone(text_request)
        assert text_request is not None
        self.assertEqual(text_request.name, "load_rules")
        self.assertIsNotNone(punctuated_request)
        assert punctuated_request is not None
        self.assertEqual(punctuated_request.name, "load_rules")
        self.assertIsNotNone(json_request)
        assert json_request is not None
        self.assertEqual(json_request.name, "edit_rules")
        self.assertEqual(json_request.argument, "disable")
        self.assertIsNotNone(structured_request)
        assert structured_request is not None
        self.assertEqual(structured_request.name, "edit_rules")
        self.assertEqual(
            structured_request.arguments,
            {"rule_id": "slow_near_unknown_object", "changes": {"enabled": False}},
        )
        self.assertIsNotNone(mapping_request)
        assert mapping_request is not None
        self.assertEqual(mapping_request.name, "update_object_mapping")
        self.assertEqual(mapping_request.arguments, {"marker": "A", "object": "夹具"})
        self.assertIsNotNone(mapping_alias_request)
        assert mapping_alias_request is not None
        self.assertEqual(mapping_alias_request.name, "update_object_mapping")
        self.assertEqual(mapping_alias_request.arguments, {"marker": "A", "object": "夹具"})
        self.assertIsNotNone(mapping_query_request)
        assert mapping_query_request is not None
        self.assertEqual(mapping_query_request.name, "get_object_mapping")
        self.assertEqual(mapping_query_request.arguments, {"marker": "A"})
        self.assertIsNotNone(unknown_request)
        assert unknown_request is not None
        self.assertEqual(unknown_request.name, "unknown_tool")
        self.assertIsNotNone(prefixed_structured_request)
        assert prefixed_structured_request is not None
        self.assertEqual(prefixed_structured_request.name, "edit_rules")
        self.assertEqual(
            prefixed_structured_request.arguments,
            {"rules": [{"id": "safety_distance", "conditions": {"distance_m": 1.0}}]},
        )
        self.assertEqual(final_content, "当前状态正常。")

    def test_extract_rule_patch_payload_accepts_tool_envelope(self) -> None:
        patch = extract_rule_patch_payload(
            '```json\n{"type":"tool_call","name":"edit_rules","arguments":{'
            '"rule_id":"slow_near_unknown_object","changes":{"enabled":false}}}\n```'
        )

        self.assertEqual(patch["rule_id"], "slow_near_unknown_object")
        self.assertEqual(patch["changes"], {"enabled": False})

    def test_extract_rule_replacement_reads_json_from_model_text(self) -> None:
        replacement = extract_rule_replacement(
            '```json\n{"id":"stop_on_person_intrusion","enabled":false}\n```'
        )

        self.assertEqual(replacement["id"], "stop_on_person_intrusion")
        self.assertFalse(replacement["enabled"])


class Ros2VisionSnapshotProviderTest(unittest.TestCase):
    def _build_ros2_modules(
        self,
        *,
        executor_calls: list[str],
        future_done: bool,
        response: object,
    ) -> dict[str, ModuleType]:
        test_case = self

        class FakeFuture:
            def done(self) -> bool:
                return future_done

            def result(self) -> object:
                return response

        class FakeClient:
            def wait_for_service(self, timeout_sec: float | None = None) -> bool:
                return True

            def call_async(self, request: object) -> FakeFuture:
                return FakeFuture()

        class FakeNode:
            context = object()

            def create_client(self, srv_type: object, service_name: str) -> FakeClient:
                return FakeClient()

            def destroy_node(self) -> None:
                executor_calls.append("destroy_node")

        class FakeExecutor:
            def __init__(self, *, context: object) -> None:
                executor_calls.append("create_executor")

            def add_node(self, node: FakeNode) -> bool:
                executor_calls.append("add_node")
                return True

            def spin_until_future_complete(
                self, future: FakeFuture, timeout_sec: float | None = None
            ) -> None:
                executor_calls.append("spin_until_future_complete")

            def remove_node(self, node: FakeNode) -> None:
                executor_calls.append("remove_node")

            def shutdown(self, timeout_sec: float | None = None) -> bool:
                executor_calls.append("shutdown")
                return True

        class FakeTriggerRequest:
            pass

        class FakeTrigger:
            Request = FakeTriggerRequest

        rclpy_module = ModuleType("rclpy")
        rclpy_module.ok = lambda: True
        rclpy_module.create_node = lambda name: FakeNode()
        rclpy_module.spin_until_future_complete = lambda *args, **kwargs: test_case.fail(
            "global rclpy executor must not be used for vision snapshot"
        )
        executors_module = ModuleType("rclpy.executors")
        executors_module.SingleThreadedExecutor = FakeExecutor
        rclpy_module.executors = executors_module
        std_srvs_module = ModuleType("std_srvs")
        std_srvs_srv_module = ModuleType("std_srvs.srv")
        std_srvs_srv_module.Trigger = FakeTrigger
        std_srvs_module.srv = std_srvs_srv_module

        return {
            "rclpy": rclpy_module,
            "rclpy.executors": executors_module,
            "std_srvs": std_srvs_module,
            "std_srvs.srv": std_srvs_srv_module,
        }

    def test_capture_snapshot_uses_private_executor_not_global(self) -> None:
        executor_calls: list[str] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "snapshot.jpg"
            image_path.write_bytes(b"jpeg")
            response = SimpleNamespace(
                success=True,
                message=json.dumps({"image_path": str(image_path)}, ensure_ascii=False),
            )
            modules = self._build_ros2_modules(
                executor_calls=executor_calls,
                future_done=True,
                response=response,
            )
            provider = Ros2TriggerVisionSnapshotProvider(timeout_seconds=0.1)

            with patch.dict(sys.modules, modules):
                artifact = provider.capture_snapshot()

        self.assertEqual(artifact.image_path, image_path.resolve())
        self.assertIn("create_executor", executor_calls)
        self.assertIn("spin_until_future_complete", executor_calls)
        self.assertEqual(executor_calls[-3:], ["remove_node", "shutdown", "destroy_node"])

    def test_capture_snapshot_timeout_raises_runtime_error(self) -> None:
        executor_calls: list[str] = []
        modules = self._build_ros2_modules(
            executor_calls=executor_calls,
            future_done=False,
            response=None,
        )
        provider = Ros2TriggerVisionSnapshotProvider(timeout_seconds=0.1)

        with patch.dict(sys.modules, modules):
            with self.assertRaises(RuntimeError) as ctx:
                provider.capture_snapshot()

        self.assertIn("timed out", str(ctx.exception))
        self.assertEqual(executor_calls[-3:], ["remove_node", "shutdown", "destroy_node"])


class Ros2BridgeTest(unittest.TestCase):
    def test_estop_trigger_publishes_safety_request_by_default(self) -> None:
        result = VoicePipeline(llm=FakeLlm()).run_text_turn("请立即急停机械臂", synthesize=False)

        plans = build_voice_ros2_plan(result)

        self.assertEqual([plan.topic for plan in plans], [DEFAULT_TRANSCRIPT_TOPIC, DEFAULT_RESPONSE_TOPIC, DEFAULT_ESTOP_REQUEST_TOPIC])
        self.assertEqual(plans[-1].message_type, ROS2_STRING)
        self.assertIn('"source": "voice_assistant"', plans[-1].payload)
        self.assertIn('"active": true', plans[-1].payload)
        self.assertNotIn('"reset_sources"', plans[-1].payload)

    def test_estop_trigger_plan_syncs_arm_runtime_stop_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            arm_rules_path = Path(temp_dir) / "arm_rules.json"
            write_arm_rule_document(
                arm_rules_path,
                {
                    "arm_capture": "True",
                    "arm_capture_goal": "C",
                    "arm_decelerate": "0.5",
                    "arm_recover": "True",
                    "arm_safety_distance": "0.4",
                },
            )
            result = VoicePipeline(llm=FakeLlm()).run_text_turn("请立即急停机械臂", synthesize=False)
            config = Ros2BridgeConfig()
            plans = build_voice_ros2_plan(result, config)

            sync_result = sync_estop_plans_to_arm_rules(plans, config, arm_rules_path)
            document = load_arm_rule_document(arm_rules_path)

            self.assertIsNotNone(sync_result)
            self.assertEqual(document["arm_stop"], "True")
            self.assertEqual(document["arm_recover"], "False")
            self.assertEqual(document["arm_capture"], "True")
            self.assertEqual(document["arm_capture_goal"], "C")
            self.assertEqual(document["arm_decelerate"], "0.5")
            self.assertEqual(document["arm_safety_distance"], "0.4")

    def test_ros2_cli_dry_run_syncs_estop_plan_to_arm_runtime_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            arm_rules_path = Path(temp_dir) / "arm_rules.json"
            args = build_parser().parse_args(
                [
                    "ros2-text-turn",
                    "--text",
                    "请立即急停机械臂",
                    "--skip-tts",
                    "--dry-run-ros2",
                    "--arm-rules",
                    str(arm_rules_path),
                ]
            )
            result = VoicePipeline(llm=FakeLlm()).run_text_turn(args.text, synthesize=False)

            with patch("sys.stdout", new=io.StringIO()):
                publish_or_print_ros2_plan(result, args)
            document = load_arm_rule_document(arm_rules_path)

            self.assertEqual(document["arm_stop"], "True")
            self.assertEqual(document["arm_recover"], "False")

    def test_estop_release_publishes_inactive_safety_request(self) -> None:
        result = VoicePipeline(llm=FakeLlm()).run_text_turn("解除急停", synthesize=False)

        plans = build_voice_ros2_plan(result)

        self.assertEqual(result.response_text, "已收到解除急停请求，请确认现场安全后再执行。")
        self.assertEqual(plans[-1].topic, DEFAULT_ESTOP_REQUEST_TOPIC)
        self.assertIn('"active": false', plans[-1].payload)
        self.assertIn('"latch": false', plans[-1].payload)
        self.assertIn('"reset_sources": ["min_distance_camera"]', plans[-1].payload)

    def test_direct_estop_release_plan_syncs_arm_runtime_recover_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            arm_rules_path = Path(temp_dir) / "arm_rules.json"
            write_arm_rule_document(
                arm_rules_path,
                {
                    "arm_capture": "True",
                    "arm_capture_goal": "B",
                    "arm_decelerate": "0.3",
                    "arm_stop": "True",
                    "arm_recover": "False",
                    "arm_safety_distance": "0.2",
                },
            )
            result = VoicePipeline(llm=FakeLlm()).run_text_turn("解除急停", synthesize=False)
            config = Ros2BridgeConfig(use_estop_request=False)
            plans = build_voice_ros2_plan(result, config)

            sync_result = sync_estop_plans_to_arm_rules(plans, config, arm_rules_path)
            document = load_arm_rule_document(arm_rules_path)

            self.assertIsNotNone(sync_result)
            self.assertEqual(document["arm_stop"], "False")
            self.assertEqual(document["arm_recover"], "True")
            self.assertEqual(document["arm_capture"], "True")
            self.assertEqual(document["arm_capture_goal"], "B")
            self.assertEqual(document["arm_decelerate"], "0.3")

    def test_direct_estop_mode_publishes_bool_topic(self) -> None:
        result = VoicePipeline(llm=FakeLlm()).run_text_turn("马上停止机械臂", synthesize=False)

        plans = build_voice_ros2_plan(result, Ros2BridgeConfig(use_estop_request=False))

        self.assertEqual(plans[-1].topic, "/emergency_stop")
        self.assertEqual(plans[-1].message_type, ROS2_BOOL)
        self.assertIs(plans[-1].payload, True)

    def test_goal_command_publishes_point(self) -> None:
        result = VoicePipeline(llm=FakeLlm()).run_text_turn("移动到目标 0.2 -0.1 0.35", synthesize=False)

        plans = build_voice_ros2_plan(result)

        self.assertEqual(plans[-1].topic, DEFAULT_GOAL_TOPIC)
        self.assertEqual(plans[-1].message_type, ROS2_POINT)
        self.assertEqual(plans[-1].payload, {"x": 0.2, "y": -0.1, "z": 0.35})

    def test_explanatory_safety_text_does_not_publish_command_topics(self) -> None:
        result = VoicePipeline(llm=FakeLlm()).run_text_turn("请说明机械臂急停规则", synthesize=False)

        plans = build_voice_ros2_plan(result)

        self.assertEqual([plan.topic for plan in plans], [DEFAULT_TRANSCRIPT_TOPIC, DEFAULT_RESPONSE_TOPIC])

    def test_question_with_force_word_does_not_publish_command_topics(self) -> None:
        result = VoicePipeline(llm=FakeLlm()).run_text_turn("机械臂能不能立即急停", synthesize=False)

        plans = build_voice_ros2_plan(result)

        self.assertEqual([plan.topic for plan in plans], [DEFAULT_TRANSCRIPT_TOPIC, DEFAULT_RESPONSE_TOPIC])

    def test_can_disable_observability_topics_for_command_only_bridge(self) -> None:
        result = VoicePipeline(llm=FakeLlm()).run_text_turn("立即急停", synthesize=False)
        config = Ros2BridgeConfig(publish_transcript=False, publish_response=False)

        plans = build_voice_ros2_plan(result, config)

        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].topic, DEFAULT_ESTOP_REQUEST_TOPIC)

    def test_rule_update_turn_does_not_publish_robot_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rules_path = Path(temp_dir) / "rules.json"
            write_rule_document(
                rules_path,
                {
                    "version": 1,
                    "rules": [{"id": "limit_speed_near_person", "enabled": True}],
                },
            )
            llm = FakeLlm(
                [
                    '{"type":"tool_call","name":"edit_rules","arguments":{}}',
                    '{"type":"tool_call","name":"edit_rules","arguments":{'
                    '"rule_id":"limit_speed_near_person","changes":{"enabled":false}}}',
                ]
            )
            pipeline = VoicePipeline(
                llm=llm,
                rules_path=rules_path,
                rule_edit_strategy=RULE_EDIT_STRATEGY_TWO_PASS,
            )

            result = pipeline.run_text_turn("请禁用人员附近限速规则", synthesize=False)
            plans = build_voice_ros2_plan(result)

            self.assertIsNotNone(result.rule_update)
            self.assertEqual([plan.topic for plan in plans], [DEFAULT_TRANSCRIPT_TOPIC, DEFAULT_RESPONSE_TOPIC])

    def test_object_mapping_update_turn_does_not_publish_robot_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "object_mapping.json"
            write_object_mapping_document(mapping_path, object_mapping_document())
            llm = FakeLlm('{"type":"tool_call","name":"update_object_mapping","arguments":{}}')
            pipeline = VoicePipeline(llm=llm, object_mapping_path=mapping_path)

            result = pipeline.run_text_turn("把标号A的映射改成急停按钮", synthesize=False)
            plans = build_voice_ros2_plan(result)

            self.assertIsNotNone(result.object_mapping_update)
            self.assertEqual([plan.topic for plan in plans], [DEFAULT_TRANSCRIPT_TOPIC, DEFAULT_RESPONSE_TOPIC])

    def test_object_mapping_query_turn_does_not_publish_robot_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "object_mapping.json"
            write_object_mapping_document(mapping_path, object_mapping_document())
            llm = FakeLlm('{"type":"tool_call","name":"get_object_mapping","arguments":{}}')
            pipeline = VoicePipeline(llm=llm, object_mapping_path=mapping_path)

            result = pipeline.run_text_turn("查询标号A", synthesize=False)
            plans = build_voice_ros2_plan(result)

            self.assertIsNotNone(result.object_mapping_query)
            self.assertEqual([plan.topic for plan in plans], [DEFAULT_TRANSCRIPT_TOPIC, DEFAULT_RESPONSE_TOPIC])

    def test_object_mapping_table_query_turn_does_not_publish_robot_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "object_mapping.json"
            write_object_mapping_document(mapping_path, object_mapping_document())
            llm = FakeLlm('{"type":"tool_call","name":"get_object_mapping","arguments":{}}')
            pipeline = VoicePipeline(llm=llm, object_mapping_path=mapping_path)

            result = pipeline.run_text_turn("说一说当前的物体映射表", synthesize=False)
            plans = build_voice_ros2_plan(result)

            self.assertIsNotNone(result.object_mapping_table_query)
            self.assertEqual([plan.topic for plan in plans], [DEFAULT_TRANSCRIPT_TOPIC, DEFAULT_RESPONSE_TOPIC])

    def test_object_grasp_turn_does_not_publish_robot_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "object_mapping.json"
            write_object_mapping_document(mapping_path, object_mapping_document())
            llm = FakeLlm('{"type":"final","content":"收到。"}')
            pipeline = VoicePipeline(llm=llm, object_mapping_path=mapping_path)

            result = pipeline.run_text_turn("给我红色方块", synthesize=False)
            plans = build_voice_ros2_plan(result)

            self.assertIsNotNone(result.object_grasp_intent)
            self.assertEqual([plan.topic for plan in plans], [DEFAULT_TRANSCRIPT_TOPIC, DEFAULT_RESPONSE_TOPIC])

    def test_intent_helpers_expose_detected_command_details(self) -> None:
        estop = detect_estop_command("立刻停止机械臂")
        goal = detect_goal_command("去到坐标 1 2 3")

        self.assertIsNotNone(estop)
        self.assertTrue(estop.active)
        self.assertIsNotNone(goal)
        self.assertEqual((goal.x, goal.y, goal.z), (1.0, 2.0, 3.0))

    def test_empty_publish_plan_does_not_require_ros2_runtime(self) -> None:
        Ros2VoiceBridge().publish_plans([])

    def test_publish_uses_private_executor_when_global_executor_is_busy(self) -> None:
        published_payloads: list[str] = []
        executor_calls: list[str] = []

        class FakeString:
            def __init__(self) -> None:
                self.data = ""

        class FakeBool:
            def __init__(self) -> None:
                self.data = False

        class FakePoint:
            pass

        class FakePublisher:
            def publish(self, message: object) -> None:
                published_payloads.append(str(getattr(message, "data", "")))

        class FakeNode:
            context = object()

            def create_publisher(self, message_class: object, topic: str, qos_depth: int) -> FakePublisher:
                return FakePublisher()

            def destroy_node(self) -> None:
                executor_calls.append("destroy_node")

        class FakeExecutor:
            def __init__(self, *, context: object) -> None:
                executor_calls.append("create_executor")

            def add_node(self, node: FakeNode) -> bool:
                executor_calls.append("add_node")
                return True

            def spin_once(self, timeout_sec: float | None = None) -> None:
                executor_calls.append("spin_once")

            def remove_node(self, node: FakeNode) -> None:
                executor_calls.append("remove_node")

            def shutdown(self, timeout_sec: float | None = None) -> bool:
                executor_calls.append("shutdown")
                return True

        rclpy_module = ModuleType("rclpy")
        rclpy_module.ok = lambda: True
        rclpy_module.create_node = lambda name: FakeNode()
        rclpy_module.spin_once = lambda *args, **kwargs: self.fail(
            "global rclpy executor must not be used for publishing"
        )
        executors_module = ModuleType("rclpy.executors")
        executors_module.SingleThreadedExecutor = FakeExecutor
        geometry_module = ModuleType("geometry_msgs")
        geometry_msg_module = ModuleType("geometry_msgs.msg")
        geometry_msg_module.Point = FakePoint
        geometry_module.msg = geometry_msg_module
        std_module = ModuleType("std_msgs")
        std_msg_module = ModuleType("std_msgs.msg")
        std_msg_module.Bool = FakeBool
        std_msg_module.String = FakeString
        std_module.msg = std_msg_module
        rclpy_module.executors = executors_module

        modules = {
            "rclpy": rclpy_module,
            "rclpy.executors": executors_module,
            "geometry_msgs": geometry_module,
            "geometry_msgs.msg": geometry_msg_module,
            "std_msgs": std_module,
            "std_msgs.msg": std_msg_module,
        }
        plan = build_voice_ros2_plan(
            VoicePipeline(llm=FakeLlm()).run_text_turn("解除急停", synthesize=False),
            Ros2BridgeConfig(
                publish_transcript=False,
                publish_response=False,
                wait_for_subscribers_seconds=0.0,
            ),
        )

        with patch.dict(sys.modules, modules):
            Ros2VoiceBridge(
                Ros2BridgeConfig(wait_for_subscribers_seconds=0.0)
            ).publish_plans(plan)

        self.assertEqual(len(published_payloads), 1)
        self.assertIn('"active": false', published_payloads[0])
        self.assertIn("create_executor", executor_calls)
        self.assertIn("spin_once", executor_calls)
        self.assertEqual(executor_calls[-3:], ["remove_node", "shutdown", "destroy_node"])


class MicrophoneEndpointingTest(unittest.TestCase):
    def test_short_noise_below_minimum_speech_duration_is_ignored(self) -> None:
        config = EndpointingConfig(
            sample_rate=10,
            speech_threshold=0.5,
            min_speech_seconds=0.25,
            trailing_silence_seconds=0.2,
            pre_roll_seconds=0.1,
            max_utterance_seconds=2.0,
        )
        frames = [
            np.zeros(1, dtype=np.float32),
            np.ones(1, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
        ]

        self.assertEqual(segment_frames(frames, config, flush=False), [])

    def test_trailing_silence_emits_completed_utterance(self) -> None:
        config = EndpointingConfig(
            sample_rate=10,
            speech_threshold=0.5,
            min_speech_seconds=0.2,
            trailing_silence_seconds=0.2,
            pre_roll_seconds=0.1,
            max_utterance_seconds=2.0,
        )
        frames = [
            np.zeros(1, dtype=np.float32),
            np.ones(1, dtype=np.float32),
            np.ones(1, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
        ]

        utterances = segment_frames(frames, config, flush=False)

        self.assertEqual(len(utterances), 1)
        self.assertEqual(utterances[0].reason, "trailing_silence")
        self.assertAlmostEqual(utterances[0].speech_seconds, 0.2)
        self.assertGreaterEqual(utterances[0].audio_seconds, utterances[0].speech_seconds)

    def test_max_utterance_seconds_caps_long_speech(self) -> None:
        detector = EnergyEndpointDetector(
            EndpointingConfig(
                sample_rate=10,
                speech_threshold=0.5,
                min_speech_seconds=0.1,
                trailing_silence_seconds=2.0,
                pre_roll_seconds=0.0,
                max_utterance_seconds=0.3,
            )
        )

        utterance = None
        for _ in range(5):
            utterance = detector.accept(np.ones(1, dtype=np.float32))
            if utterance is not None:
                break

        self.assertIsNotNone(utterance)
        assert utterance is not None
        self.assertEqual(utterance.reason, "max_utterance_seconds")
        self.assertGreaterEqual(utterance.audio_seconds, 0.3)

    def test_write_utterance_wav_creates_pcm16_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            utterance = CapturedUtterance(
                audio=np.array([-1.0, 0.0, 1.0], dtype=np.float32),
                sample_rate=16000,
                audio_seconds=3 / 16000,
                speech_seconds=3 / 16000,
                peak=1.0,
                rms=0.5,
                reason="test",
            )

            path = write_utterance_wav(utterance, Path(temp_dir), prefix="test_")

            self.assertTrue(path.exists())
            self.assertEqual(path.suffix, ".wav")


class ListenCliTest(unittest.TestCase):
    def test_build_pipeline_does_not_construct_9b_rule_editor_by_default(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["text-turn", "--text", "hello", "--skip-tts"])

        with (
            patch("local_safety_assistant.stack.cli.available_openvino_devices", return_value=["CPU"]),
            patch("local_safety_assistant.stack.cli.QwenLlmEngine") as llm_class,
        ):
            build_pipeline(args, include_asr=False, include_tts=False)

        self.assertEqual(llm_class.call_count, 1)
        self.assertEqual(llm_class.call_args.kwargs["model"], "qwen35-2b")

    def test_listen_no_wake_parser_accepts_realtime_options(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "listen",
                "--no-wake",
                "--skip-tts",
                "--dry-run-ros2",
                "--no-play-tts",
                "--max-turns",
                "1",
                "--play-device",
                "USB Speaker",
                "--mic-device",
                "3",
                "--mic-block-ms",
                "40",
                "--speech-threshold",
                "0.02",
                "--trailing-silence-seconds",
                "0.4",
            ]
        )
        mic_config = build_microphone_config(args)
        endpoint_config = build_endpointing_config(args, mic_config.sample_rate)

        self.assertEqual(args.command, "listen")
        self.assertTrue(args.no_wake)
        self.assertTrue(args.skip_tts)
        self.assertTrue(args.dry_run_ros2)
        self.assertTrue(args.no_play_tts)
        self.assertEqual(args.play_device, "USB Speaker")
        self.assertEqual(args.max_turns, 1)
        self.assertEqual(mic_config.device, 3)
        self.assertAlmostEqual(mic_config.block_seconds, 0.04)
        self.assertAlmostEqual(endpoint_config.speech_threshold, 0.02)
        self.assertAlmostEqual(endpoint_config.trailing_silence_seconds, 0.4)

    def test_listen_no_wake_parser_uses_tuned_endpointing_defaults(self) -> None:
        parser = build_parser()

        args = parser.parse_args(["listen", "--no-wake"])
        mic_config = build_microphone_config(args)
        endpoint_config = build_endpointing_config(args, mic_config.sample_rate)

        self.assertAlmostEqual(endpoint_config.speech_threshold, 0.06)
        self.assertAlmostEqual(endpoint_config.trailing_silence_seconds, 1.6)

    def test_audio_devices_parser_accepts_json_output(self) -> None:
        args = build_parser().parse_args(["audio-devices", "--json"])

        self.assertEqual(args.command, "audio-devices")
        self.assertTrue(args.json)

    def test_tts_parser_accepts_piper_engine_options(self) -> None:
        args = build_parser().parse_args(
            [
                "tts",
                "--tts-engine",
                "piper",
                "--piper-model-dir",
                "/tmp/piper_model",
                "--piper-espeak-data-dir",
                "/tmp/espeak-ng-data",
                "--piper-threads",
                "2",
                "--piper-silence-scale",
                "0.6",
                "--text",
                "机械臂已停止",
            ]
        )

        self.assertEqual(args.command, "tts")
        self.assertEqual(args.tts_engine, "piper")
        self.assertEqual(args.piper_model_dir, Path("/tmp/piper_model"))
        self.assertEqual(args.piper_espeak_data_dir, Path("/tmp/espeak-ng-data"))
        self.assertEqual(args.piper_threads, 2)
        self.assertEqual(args.piper_silence_scale, 0.6)

    def test_tts_parser_uses_tuned_piper_silence_default(self) -> None:
        args = build_parser().parse_args(["tts", "--tts-engine", "piper", "--text", "机械臂已停止"])

        self.assertEqual(args.piper_silence_scale, 1.0)

    def test_listen_turn_plays_tts_unless_disabled_or_skipped(self) -> None:
        result = VoicePipeline(llm=FakeLlm(), tts=FakeTts()).run_text_turn("hello", synthesize=True)
        parser = build_parser()

        play_args = parser.parse_args(["listen", "--no-wake"])
        no_play_args = parser.parse_args(["listen", "--no-wake", "--no-play-tts"])
        skip_args = parser.parse_args(["listen", "--no-wake", "--skip-tts"])

        self.assertTrue(should_play_tts(play_args, result))
        self.assertFalse(should_play_tts(no_play_args, result))
        self.assertFalse(should_play_tts(skip_args, result))

    def test_parse_microphone_device_keeps_named_devices(self) -> None:
        self.assertEqual(parse_microphone_device("2"), 2)
        self.assertEqual(parse_microphone_device("USB Microphone"), "USB Microphone")
        self.assertIsNone(parse_microphone_device(None))


if __name__ == "__main__":
    unittest.main()
