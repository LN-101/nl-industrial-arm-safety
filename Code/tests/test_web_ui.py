from __future__ import annotations

import array
import json
import math
import threading
import tempfile
import time
import unittest
import wave
from http.client import HTTPConnection
from pathlib import Path
from unittest.mock import patch

from local_safety_assistant.stack.asr import AsrResult
from local_safety_assistant.stack.llm import LlmResult
from local_safety_assistant.stack.cli import build_moss_tts_config
from local_safety_assistant.stack.config import MossTtsConfig, VoiceStackConfig
from local_safety_assistant.stack.pipeline import VISION_ANALYSIS_FALLBACK_RESPONSE, VoicePipeline
from local_safety_assistant.stack.ros2_bridge import DEFAULT_ESTOP_REQUEST_TOPIC, ROS2_STRING, Ros2BridgeConfig, Ros2VoiceBridge
from local_safety_assistant.stack.tts import TtsResult
from local_safety_assistant.stack.vision import VisionImageArtifact
from local_safety_assistant.arm_rules import load_arm_rule_document, write_arm_rule_document
from local_safety_assistant.object_mapping import load_object_mapping_document, write_object_mapping_document
from local_safety_assistant.rules import load_rule_document, write_rule_document
from local_safety_assistant.web.server import WebUiHTTPServer, build_parser
from local_safety_assistant.web.service import (
    DEFAULT_EMERGENCY_ALERT_AUDIO_URL,
    DEFAULT_EMERGENCY_ALERT_AUDIO_PATH,
    MOSS_STREAM_CLOSE_REQUEST_TIMEOUT_SECONDS,
    MossStreamingServer,
    WEB_VOICE_UPLOAD_MAX_BYTES,
    SessionStore,
    WebUiConfig,
    WebUiService,
    _build_runtime_args,
    _moss_available,
    _resolve_tts_engine,
    _split_moss_tts_text_segments,
)
from local_safety_assistant.web.ui import render_index_html


class FakeAsr:
    def __init__(self, text: str = "请立即急停机械臂") -> None:
        self.text = text

    def transcribe_wav(self, audio_path: Path) -> AsrResult:
        return AsrResult(
            text=self.text,
            model="fake-asr",
            device="CPU",
            audio_seconds=1.0,
            load_seconds=0.0,
            inference_seconds=0.0,
        )


class FakeLlm:
    def __init__(
        self,
        response: str | list[str] | tuple[str, ...] = "已收到。",
        vision_response: str = "画面中工位可见，未发现明显人员进入机械臂工作区。",
    ) -> None:
        self.response = response
        self.responses = list(response) if isinstance(response, (list, tuple)) else None
        self.vision_response = vision_response
        self.image_calls: list[tuple[str, Path, str | None]] = []

    def generate(self, user_text: str) -> LlmResult:
        if self.responses is not None and self.responses:
            text = self.responses.pop(0)
        else:
            text = self.response if isinstance(self.response, str) else "已收到。"
        return LlmResult(
            prompt=user_text,
            text=text,
            model="fake-llm",
            device="CPU",
            load_seconds=0.0,
            inference_seconds=0.0,
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
            model="fake-llm",
            device="GPU",
            load_seconds=0.0,
            inference_seconds=0.0,
        )


class BlockingFakeLlm(FakeLlm):
    def __init__(self, response: str = "已收到。") -> None:
        super().__init__(response)
        self.started = threading.Event()
        self.release = threading.Event()

    def generate(self, user_text: str) -> LlmResult:
        self.started.set()
        if not self.release.wait(timeout=2.0):
            raise RuntimeError("fake LLM generation was not released")
        return super().generate(user_text)


class FakeTts:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def synthesize(self, text: str, *, output_name: str | None = None) -> TtsResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        audio_path = self.output_dir / f"{output_name or 'voice'}.wav"
        audio_path.write_bytes(b"RIFF")
        return TtsResult(
            text=text,
            audio_paths=(audio_path,),
            command=("fake-tts",),
            elapsed_seconds=0.01,
            stdout="",
            stderr="",
        )


class FakeVisionSnapshotProvider:
    def __init__(self, artifact: VisionImageArtifact) -> None:
        self.artifact = artifact
        self.calls = 0

    def capture_snapshot(self) -> VisionImageArtifact:
        self.calls += 1
        return self.artifact


class FakeMossStreamer:
    def __init__(
        self,
        *,
        chunks: tuple[bytes, ...] = (b"\x00\x00\x01\x00", b"\x02\x00\x03\x00"),
        sample_rate: int = 48000,
        channels: int = 2,
        block_start: bool = False,
    ) -> None:
        self.chunks = chunks
        self.sample_rate = sample_rate
        self.channels = channels
        self.started = threading.Event()
        self.release = threading.Event()
        if not block_start:
            self.release.set()
        self.started_texts: list[str] = []
        self.status_requests: list[str] = []
        self.audio_requests: list[str] = []
        self.closed_audio_urls: list[str] = []

    def status(self) -> dict[str, object]:
        return {"running": True, "base_url": "fake://moss", "demo_id": "demo-test"}

    def start_stream(self, text: str) -> dict[str, object]:
        self.started_texts.append(text)
        self.started.set()
        if not self.release.wait(timeout=5.0):
            raise RuntimeError("fake MOSS stream start was not released")
        return {
            "stream_id": "stream-test",
            "audio_url": "moss://audio",
            "status_url": "moss://status",
            "result_url": "moss://result",
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "ready": False,
            "failed": False,
        }

    def stream_status(self, status_url: str) -> dict[str, object]:
        self.status_requests.append(status_url)
        return {
            "stream_id": "stream-test",
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "ready": True,
            "failed": False,
            "run_status": "complete",
        }

    def stream_audio(self, audio_url: str):
        self.audio_requests.append(audio_url)
        yield from self.chunks

    def close_stream(self, audio_url: str) -> None:
        self.closed_audio_urls.append(audio_url)


class LookaheadFakeMossStreamer(FakeMossStreamer):
    def __init__(self) -> None:
        super().__init__(chunks=())
        self.first_stream_open = threading.Event()
        self.finish_first_stream = threading.Event()
        self.second_started = threading.Event()
        self.second_started_while_first_stream_open = False
        self._start_lock = threading.Lock()

    def start_stream(self, text: str) -> dict[str, object]:
        with self._start_lock:
            index = len(self.started_texts)
            self.started_texts.append(text)
            if index == 0:
                self.started.set()
            if index == 1:
                self.second_started_while_first_stream_open = (
                    self.first_stream_open.is_set() and not self.finish_first_stream.is_set()
                )
                self.second_started.set()
        return {
            "stream_id": f"stream-test-{index}",
            "audio_url": f"moss://audio/{index}",
            "status_url": f"moss://status/{index}",
            "result_url": f"moss://result/{index}",
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "ready": False,
            "failed": False,
        }

    def stream_audio(self, audio_url: str):
        self.audio_requests.append(audio_url)
        if audio_url == "moss://audio/0":
            self.first_stream_open.set()
            yield _pcm16_square_chunk(4000, 0.02)
            if not self.finish_first_stream.wait(timeout=2.0):
                raise RuntimeError("fake first MOSS stream was not released")
            yield _pcm16_square_chunk(5000, 0.02)
            return
        yield _pcm16_square_chunk(6000, 0.02)


class SegmentJumpFakeMossStreamer(FakeMossStreamer):
    def __init__(self) -> None:
        super().__init__(chunks=())

    def start_stream(self, text: str) -> dict[str, object]:
        index = len(self.started_texts)
        self.started_texts.append(text)
        self.started.set()
        return {
            "stream_id": f"stream-test-{index}",
            "audio_url": f"moss://audio/{index}",
            "status_url": f"moss://status/{index}",
            "result_url": f"moss://result/{index}",
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "ready": False,
            "failed": False,
        }

    def stream_audio(self, audio_url: str):
        self.audio_requests.append(audio_url)
        index = int(audio_url.rsplit("/", 1)[-1])
        amplitude = 18000 if index == 0 else -18000
        yield _pcm16_constant_chunk(amplitude, 0.12)


class WarmableFakeAsr(FakeAsr):
    def __init__(self) -> None:
        self.warm_calls = 0

    def warm_load(self) -> dict[str, object]:
        self.warm_calls += 1
        return {"status": "ready", "model": "fake-asr", "device": "CPU"}


class WarmableFakeLlm(FakeLlm):
    def __init__(self, response: str = "已收到。", model: str = "fake-llm") -> None:
        super().__init__(response)
        self.model = model
        self.warm_calls = 0

    def generate(self, user_text: str) -> LlmResult:
        result = super().generate(user_text)
        return LlmResult(
            prompt=result.prompt,
            text=result.text,
            model=self.model,
            device=result.device,
            load_seconds=result.load_seconds,
            inference_seconds=result.inference_seconds,
        )

    def warm_load(self) -> dict[str, object]:
        self.warm_calls += 1
        return {"status": "ready", "model": self.model, "device": "CPU"}


class WarmableFakeMossStreamer(FakeMossStreamer):
    def __init__(self) -> None:
        super().__init__()
        self.warm_calls = 0
        self.shutdown_calls = 0

    def warm_start(self) -> dict[str, object]:
        self.warm_calls += 1
        return {"status": "ready", "cpu_threads": 4}

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class FakeMossProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_timeouts: list[float] = []

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.returncode = -15

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        if timeout is not None:
            self.wait_timeouts.append(timeout)
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def personnel_rule_document(*, distance: float) -> dict[str, object]:
    return {
        "version": 1,
        "rules": [
            {
                "id": "stop_on_person_intrusion",
                "name": "人员侵入停机规则",
                "enabled": True,
                "conditions": {"person_distance_m": {"lt": distance}},
                "action": {"type": "stop_motion"},
            }
        ],
    }


def _pcm16_square_chunk(amplitude: int, seconds: float, *, sample_rate: int = 48000, channels: int = 2) -> bytes:
    frames = int(sample_rate * seconds)
    samples = array.array("h")
    for frame_index in range(frames):
        value = amplitude if frame_index % 2 == 0 else -amplitude
        for _ in range(channels):
            samples.append(value)
    return samples.tobytes()


def _pcm16_constant_chunk(amplitude: int, seconds: float, *, sample_rate: int = 48000, channels: int = 2) -> bytes:
    frames = int(sample_rate * seconds)
    samples = array.array("h", [amplitude] * frames * channels)
    return samples.tobytes()


def _pcm16_silence(seconds: float, *, sample_rate: int = 48000, channels: int = 2) -> bytes:
    frames = int(sample_rate * seconds)
    return b"\x00\x00" * frames * channels


def _pcm16_samples(pcm: bytes) -> array.array[int]:
    samples = array.array("h")
    samples.frombytes(pcm)
    return samples


def _pcm16_duration_seconds(pcm: bytes, *, sample_rate: int = 48000, channels: int = 2) -> float:
    return len(pcm) / float(sample_rate * channels * 2)


def _pcm16_rms(pcm: bytes) -> float:
    samples = array.array("h")
    samples.frombytes(pcm)
    if not samples:
        return 0.0
    total = 0.0
    for sample in samples:
        total += float(sample) * float(sample)
    return math.sqrt(total / len(samples)) / 32768.0


def _pcm16_peak(pcm: bytes) -> int:
    samples = array.array("h")
    samples.frombytes(pcm)
    if not samples:
        return 0
    return max(abs(sample) for sample in samples)


class FailingChatService:
    def __init__(self) -> None:
        self.config = WebUiConfig(port=0)
        self.sessions = SessionStore(self.config.session_ttl_seconds)

    def login(self, username: str, password: str) -> str | None:
        if username == "admin" and password == "12345":
            return self.sessions.create()
        return None

    def logout(self, token: str | None) -> None:
        self.sessions.revoke(token)

    def is_authenticated(self, token: str | None) -> bool:
        return self.sessions.validate(token)

    def status(self, token: str | None = None) -> dict[str, object]:
        return {"authenticated": self.is_authenticated(token), "status_line": "ok"}

    def handle_chat(self, text: str) -> object:
        raise RuntimeError("missing tts")

    def handle_voice(self, audio_bytes: bytes, content_type: str | None = None) -> object:
        raise RuntimeError("missing asr")

    def handle_voice_progressive(self, audio_bytes: bytes, content_type: str | None = None) -> object:
        raise RuntimeError("missing asr")

    def cancel_active_web_turns(self) -> object:
        return type(
            "CancelResponse",
            (),
            {
                "as_dict": lambda self: {
                    "ok": True,
                    "status": "canceled",
                    "message": "已取消当前 Web 语音任务。",
                    "canceled_confirmations": False,
                    "canceled_turns": 0,
                    "status_line": "已取消当前 Web 语音任务。",
                }
            },
        )()

    def handle_estop(self) -> object:
        raise RuntimeError("missing ros2")

    def serve_audio_path(self, filename: str) -> Path | None:
        return None


class WebUiServiceTest(unittest.TestCase):
    def build_service(
        self,
        temp_dir: str,
        *,
        dry_run: bool = True,
        llm: FakeLlm | None = None,
        tts: object | None = None,
        moss_streamer: object | None = None,
        vision_snapshot_provider: object | None = None,
        rules_path: Path | None = None,
        object_mapping_path: Path | None = None,
        arm_rules_path: Path | None = None,
        require_confirmation: bool = False,
        asr_text: str = "请立即急停机械臂",
    ) -> WebUiService:
        config = WebUiConfig(runtime_dir=Path(temp_dir), ros2_dry_run=dry_run)
        streamer = moss_streamer or FakeMossStreamer()
        stack_config = VoiceStackConfig(arm_rules_path=arm_rules_path or Path(temp_dir) / "arm_rules.json")
        pipeline = VoicePipeline(
            asr=FakeAsr(asr_text),
            llm=llm or FakeLlm(),
            tts=tts or FakeTts(config.audio_dir),
            rules_path=rules_path,
            object_mapping_path=object_mapping_path,
            arm_rules_path=stack_config.arm_rules_path,
            vision_snapshot_provider=vision_snapshot_provider,
            require_confirmation_for_side_effects=require_confirmation,
        )
        bridge = Ros2VoiceBridge(Ros2BridgeConfig(source="web_ui"))
        return WebUiService(
            config=config,
            pipeline=pipeline,
            ros2_bridge=bridge,
            stack_config=stack_config,
            resolved_tts_engine="moss",
            available_tts_engines=("moss", "fake"),
            moss_streamer=streamer,
        )

    def test_login_accepts_admin_and_rejects_wrong_password(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self.build_service(temp_dir)

            self.assertIsNone(service.login("admin", "wrong"))
            token = service.login("admin", "12345")

            self.assertIsNotNone(token)
            self.assertTrue(service.is_authenticated(token))

    def test_session_store_expires_tokens(self) -> None:
        store = SessionStore(ttl_seconds=0)
        token = store.create()

        time.sleep(0.001)

        self.assertFalse(store.validate(token))

    def test_handle_chat_returns_audio_url_and_ros2_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self.build_service(temp_dir)

            response = service.handle_chat("你好")

            self.assertEqual(response.mode, "text")
            self.assertEqual(response.input_text, "你好")
            self.assertEqual(response.response_text, "已收到。")
            self.assertEqual(response.audio_urls, ("/audio/voice.wav",))
            self.assertEqual(response.ros2_error, None)

    def test_handle_chat_progressive_returns_vision_image_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_image = root / "camera_snapshot.jpg"
            source_image.write_bytes(b"jpeg")
            provider = FakeVisionSnapshotProvider(
                VisionImageArtifact(
                    image_path=source_image,
                    source="/vision/capture_snapshot",
                    metadata={"stamp": "2026-06-09T12:00:00", "frame_id": "camera_color"},
                )
            )
            llm = FakeLlm(
                response='{"type":"tool_call","name":"analyze_environment_vision","arguments":{}}',
                vision_response="画面中工位可见，未发现明显人员进入机械臂工作区。",
            )
            service = self.build_service(
                temp_dir,
                llm=llm,
                vision_snapshot_provider=provider,
            )

            payload = service.handle_chat_progressive("调用视觉，分析下当前工作环境")

            self.assertEqual(provider.calls, 1)
            self.assertEqual(payload["response_text"], "画面中工位可见，未发现明显人员进入机械臂工作区。")
            self.assertEqual(len(payload["image_artifacts"]), 1)
            artifact = payload["image_artifacts"][0]
            self.assertTrue(artifact["url"].startswith("/images/vision_"))
            self.assertEqual(payload["image_urls"], [artifact["url"]])
            self.assertEqual(artifact["metadata"]["frame_id"], "camera_color")
            served = service.serve_image_path(Path(artifact["url"]).name)
            self.assertIsNotNone(served)
            assert served is not None
            self.assertEqual(served.read_bytes(), b"jpeg")

    def test_handle_chat_progressive_replaces_invalid_vision_text_and_keeps_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_image = root / "camera_snapshot.jpg"
            source_image.write_bytes(b"jpeg")
            provider = FakeVisionSnapshotProvider(
                VisionImageArtifact(
                    image_path=source_image,
                    source="/vision/capture_snapshot",
                    metadata={"frame_id": "camera_color"},
                )
            )
            llm = FakeLlm(
                response='{"type":"tool_call","name":"analyze_environment_vision","arguments":{}}',
                vision_response="收到。",
            )
            service = self.build_service(
                temp_dir,
                llm=llm,
                vision_snapshot_provider=provider,
            )

            payload = service.handle_chat_progressive("调用视觉，分析下当前工作环境")

            self.assertEqual(payload["response_text"], VISION_ANALYSIS_FALLBACK_RESPONSE)
            self.assertEqual(len(payload["image_artifacts"]), 1)
            self.assertEqual(payload["image_urls"], [payload["image_artifacts"][0]["url"]])
            self.assertEqual(provider.calls, 1)

    def test_handle_chat_progressive_workspace_guidance_does_not_call_vision(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_image = root / "camera_snapshot.jpg"
            source_image.write_bytes(b"jpeg")
            provider = FakeVisionSnapshotProvider(
                VisionImageArtifact(
                    image_path=source_image,
                    source="/vision/capture_snapshot",
                    metadata={"frame_id": "camera_color"},
                )
            )
            service = self.build_service(
                temp_dir,
                llm=FakeLlm(response='{"type":"tool_call","name":"analyze_environment_vision","arguments":{}}'),
                vision_snapshot_provider=provider,
            )

            payload = service.handle_chat_progressive("当前工作区什么情况")

            self.assertEqual(provider.calls, 0)
            self.assertEqual(payload["image_artifacts"], [])
            self.assertEqual(payload["image_urls"], [])
            self.assertIn("不会一次性展开所有状态", payload["response_text"])
            self.assertIn("当前物体映射表", payload["response_text"])
            self.assertIn("调用视觉分析当前画面", payload["response_text"])

    def test_handle_chat_progressive_returns_text_before_moss_stream_start_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            streamer = FakeMossStreamer(block_start=True)
            service = self.build_service(temp_dir, moss_streamer=streamer)
            try:
                started_at = time.monotonic()
                payload = service.handle_chat_progressive("你好")
                elapsed = time.monotonic() - started_at

                self.assertLess(elapsed, 0.5)
                self.assertEqual(payload["response_text"], "已收到。")
                self.assertEqual(payload["audio_url"], f"/api/chat-stream/{payload['turn_id']}/audio")
                self.assertEqual(payload["status"], "starting_stream")
                self.assertFalse(payload["done"])
                self.assertTrue(streamer.started.wait(timeout=1.0))
                status = service.progressive_turn_status(str(payload["turn_id"]))
                self.assertIsNotNone(status)
                assert status is not None
                self.assertEqual(status["status"], "starting_stream")
                self.assertNotIn("chunks", status)
            finally:
                streamer.release.set()

            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                status = service.progressive_turn_status(str(payload["turn_id"]))
                if status and status["done"]:
                    break
                time.sleep(0.01)

            self.assertIsNotNone(status)
            assert status is not None
            self.assertEqual(status["status"], "complete")
            self.assertEqual(status["moss_stream_id"], "stream-test")

    def test_handle_chat_progressive_keeps_rule_summary_in_single_moss_stream(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            response_text = (
                "当前共有 5 条启用规则：；人员侵入停机：检测到人员进入保护距离（小于 1.0 米）时，"
                "立即停止机械臂。未知物体靠近限速：发现未知物体靠近抓取路径（小于 0.25 米）时，"
                "降低速度并通知。防护门打开停机：防护门打开时，停止机械臂并等待确认复位。"
                "安全光栅遮挡停机：光栅被遮挡时，立即停止运动。ROS 控制器报警保持："
                "检测到报警时，进入安全保持状态并通知。"
            )
            streamer = FakeMossStreamer()
            service = self.build_service(temp_dir, llm=FakeLlm(response_text), moss_streamer=streamer)

            payload = service.handle_chat_progressive("当前安全状态")
            deadline = time.monotonic() + 2.0
            status = payload
            while time.monotonic() < deadline:
                latest = service.progressive_turn_status(str(payload["turn_id"]))
                if latest is not None:
                    status = latest
                if status["done"]:
                    break
                time.sleep(0.01)
            response = service.progressive_turn_audio(str(payload["turn_id"]))
            self.assertIsNotNone(response)
            assert response is not None
            b"".join(response.iterator)

            self.assertEqual(status["status"], "complete")
            self.assertEqual(len(streamer.started_texts), 1)
            self.assertNotIn("：；", streamer.started_texts[0])
            self.assertIn("ROS 控制器报警保持", streamer.started_texts[0])
            self.assertNotIn("R O S", streamer.started_texts[0])
            self.assertEqual(payload["tts_segment_count"], 1)
            self.assertNotIn("chunks", status)

    def test_progressive_turn_audio_starts_fallback_segments_sequentially(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            response_text = "防护门打开时立即停止机械臂并保持制动状态。" * 20
            streamer = LookaheadFakeMossStreamer()
            service = self.build_service(temp_dir, llm=FakeLlm(response_text), moss_streamer=streamer)

            payload = service.handle_chat_progressive("当前安全状态")
            response = service.progressive_turn_audio(str(payload["turn_id"]))

            self.assertIsNotNone(response)
            assert response is not None
            iterator = iter(response.iterator)
            first_chunk = next(iterator)
            self.assertTrue(first_chunk)
            self.assertFalse(streamer.second_started.wait(timeout=0.10))

            streamer.finish_first_stream.set()
            remaining = b"".join(iterator)

        self.assertTrue(remaining)
        self.assertTrue(streamer.second_started.wait(timeout=1.0))
        self.assertFalse(streamer.second_started_while_first_stream_open)
        self.assertGreaterEqual(len(streamer.started_texts), 2)
        self.assertEqual("".join(streamer.started_texts), response_text)
        self.assertEqual(streamer.audio_requests, [f"moss://audio/{index}" for index in range(len(streamer.started_texts))])

    def test_progressive_turn_audio_proxies_moss_pcm_stream(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            streamer = FakeMossStreamer(chunks=(b"\x00\x00\x01\x00", b"\x02\x00\x03\x00"))
            service = self.build_service(temp_dir, moss_streamer=streamer)

            payload = service.handle_chat_progressive("你好")
            response = service.progressive_turn_audio(str(payload["turn_id"]))

            self.assertIsNotNone(response)
            assert response is not None
            self.assertEqual(response.sample_rate, 48000)
            self.assertEqual(response.channels, 2)
            self.assertEqual(response.stream_id, "stream-test")
            self.assertEqual(b"".join(response.iterator), b"\x00\x00\x01\x00\x02\x00\x03\x00")
            self.assertEqual(streamer.audio_requests, ["moss://audio"])
            self.assertEqual(streamer.closed_audio_urls, ["moss://audio"])

    def test_progressive_turn_audio_closes_upstream_stream_when_client_disconnects(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            active = _pcm16_square_chunk(4000, 0.20)
            streamer = FakeMossStreamer(chunks=(active, active, active))
            service = self.build_service(temp_dir, moss_streamer=streamer)

            payload = service.handle_chat_progressive("你好")
            response = service.progressive_turn_audio(str(payload["turn_id"]))

            self.assertIsNotNone(response)
            assert response is not None
            iterator = response.iterator
            self.assertTrue(next(iterator))
            iterator.close()

            self.assertEqual(streamer.closed_audio_urls, ["moss://audio"])

    def test_progressive_turn_audio_compresses_long_low_energy_gap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            active = _pcm16_square_chunk(4000, 0.20)
            long_silence = _pcm16_silence(2.00)
            streamer = FakeMossStreamer(chunks=(active, long_silence, active))
            service = self.build_service(temp_dir, moss_streamer=streamer)

            payload = service.handle_chat_progressive("请说明测试语音")
            response = service.progressive_turn_audio(str(payload["turn_id"]))

            self.assertIsNotNone(response)
            assert response is not None
            output = b"".join(response.iterator)

        input_duration = _pcm16_duration_seconds(active + long_silence + active)
        output_duration = _pcm16_duration_seconds(output)
        self.assertAlmostEqual(input_duration, 2.4, places=2)
        self.assertLess(output_duration, 1.10)
        self.assertGreater(output_duration, 0.85)

    def test_progressive_turn_audio_does_not_boost_low_level_active_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lead_in = _pcm16_square_chunk(4000, 0.20)
            low_level = _pcm16_square_chunk(1200, 0.50)
            streamer = FakeMossStreamer(chunks=(lead_in, low_level))
            service = self.build_service(temp_dir, moss_streamer=streamer)

            payload = service.handle_chat_progressive("请说明测试语音")
            response = service.progressive_turn_audio(str(payload["turn_id"]))

            self.assertIsNotNone(response)
            assert response is not None
            output = b"".join(response.iterator)

        conditioned_low_level = output[len(lead_in) : len(lead_in) + len(low_level)]
        self.assertLessEqual(_pcm16_rms(conditioned_low_level), _pcm16_rms(low_level) + 0.001)

    def test_progressive_turn_audio_limits_near_full_scale_pcm(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            clipped = _pcm16_square_chunk(32000, 0.20)
            streamer = FakeMossStreamer(chunks=(clipped,))
            service = self.build_service(temp_dir, moss_streamer=streamer)

            payload = service.handle_chat_progressive("当前安全规则")
            response = service.progressive_turn_audio(str(payload["turn_id"]))

            self.assertIsNotNone(response)
            assert response is not None
            output = b"".join(response.iterator)

        self.assertLessEqual(_pcm16_peak(output), 26000)

    def test_progressive_turn_audio_declicks_large_active_boundary_jump(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            first = _pcm16_constant_chunk(18000, 0.12)
            second = _pcm16_constant_chunk(-18000, 0.12)
            streamer = FakeMossStreamer(chunks=(first, second))
            service = self.build_service(temp_dir, moss_streamer=streamer)

            payload = service.handle_chat_progressive("当前安全规则")
            response = service.progressive_turn_audio(str(payload["turn_id"]))

            self.assertIsNotNone(response)
            assert response is not None
            output = b"".join(response.iterator)

        boundary_sample_index = len(first) // 2
        samples = _pcm16_samples(output)
        self.assertGreater(samples[boundary_sample_index], 15000)
        self.assertLess(samples[boundary_sample_index + (48000 // 100 * 2)], 0)

    def test_progressive_turn_audio_declicks_extreme_length_segment_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            response_text = (
                "如果光栅仍被遮挡，请保持停机并通知操作员，确认安全区域无人后再复位，"
                "复位后先低速点动观察机械臂轨迹是否正常，并由现场负责人复核急停、限速和防护门联锁状态。"
                * 6
            )
            streamer = SegmentJumpFakeMossStreamer()
            service = self.build_service(temp_dir, llm=FakeLlm(response_text), moss_streamer=streamer)

            payload = service.handle_chat_progressive("请说明测试语音")
            response = service.progressive_turn_audio(str(payload["turn_id"]))

            self.assertIsNotNone(response)
            assert response is not None
            output = b"".join(response.iterator)

        first_segment_bytes = len(_pcm16_constant_chunk(18000, 0.12))
        boundary_sample_index = first_segment_bytes // 2
        samples = _pcm16_samples(output)
        self.assertGreaterEqual(len(streamer.started_texts), 2)
        self.assertGreater(samples[boundary_sample_index], 15000)
        self.assertLess(samples[boundary_sample_index + (48000 // 100 * 2)], 0)

    def test_progressive_turn_audio_fades_in_first_active_window_without_boosting(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            first = _pcm16_square_chunk(20000, 0.20)
            streamer = FakeMossStreamer(chunks=(first,))
            service = self.build_service(temp_dir, moss_streamer=streamer)

            payload = service.handle_chat_progressive("当前安全规则")
            response = service.progressive_turn_audio(str(payload["turn_id"]))

            self.assertIsNotNone(response)
            assert response is not None
            output = b"".join(response.iterator)

        first_10ms = output[: int(48000 * 2 * 2 * 0.01)]
        first_120ms = output[: int(48000 * 2 * 2 * 0.12)]
        self.assertLess(_pcm16_rms(first_10ms), _pcm16_rms(first_120ms) * 0.35)
        self.assertLessEqual(_pcm16_rms(output), _pcm16_rms(first) + 0.001)

    def test_split_moss_tts_text_segments_keeps_compact_multi_sentence_reply_single_stream(self) -> None:
        text = "防护门打开时立即停止机械臂。复位前确认人员已经离开安全区。"

        segments = _split_moss_tts_text_segments(text)

        self.assertEqual([segment.text for segment in segments], [text])
        self.assertEqual(segments[0].pause_after_seconds, 0.0)

    def test_split_moss_tts_text_segments_keeps_normal_rule_summary_single_stream(self) -> None:
        text = (
            "当前共有 5 条启用规则：；人员侵入停机：检测到人员进入保护距离（小于 1.0 米）时，"
            "立即停止机械臂。未知物体靠近限速：发现未知物体靠近抓取路径（小于 0.25 米）时，"
            "降低速度并通知。防护门打开停机：防护门打开时，停止机械臂并等待确认复位。"
            "安全光栅遮挡停机：光栅被遮挡时，立即停止运动。ROS 控制器报警保持："
            "检测到报警时，进入安全保持状态并通知。"
        )

        segments = _split_moss_tts_text_segments(text)

        self.assertEqual(len(segments), 1)
        self.assertNotIn("：；", segments[0].text)
        self.assertIn("ROS 控制器报警保持", segments[0].text)
        self.assertEqual(segments[0].pause_after_seconds, 0.0)

    def test_split_moss_tts_text_segments_splits_only_extreme_long_reply(self) -> None:
        text = (
            "如果光栅仍被遮挡，请保持停机并通知操作员，确认安全区域无人后再复位，"
            "复位后先低速点动观察机械臂轨迹是否正常，并由现场负责人复核急停、限速和防护门联锁状态。"
            * 6
        )

        segments = _split_moss_tts_text_segments(text)

        self.assertGreater(len(segments), 1)
        self.assertEqual("".join(segment.text for segment in segments), text)
        self.assertTrue(all(len(segment.text) <= 260 for segment in segments))
        self.assertTrue(all(segment.pause_after_seconds == 0.0 for segment in segments))

    def test_split_moss_tts_text_segments_keeps_short_reply_single_stream(self) -> None:
        segments = _split_moss_tts_text_segments("机械臂已停止，请确认安全区域。")

        self.assertEqual([segment.text for segment in segments], ["机械臂已停止，请确认安全区域。"])
        self.assertEqual(segments[0].pause_after_seconds, 0.0)

    def test_handle_voice_progressive_uses_moss_stream_after_asr(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            streamer = FakeMossStreamer()
            service = self.build_service(temp_dir, moss_streamer=streamer)

            def fake_convert(input_path: Path, output_path: Path) -> None:
                self.assertEqual(input_path.suffix, ".webm")
                output_path.write_bytes(b"RIFF")

            with patch("local_safety_assistant.web.service._ffmpeg_convert_to_wav", side_effect=fake_convert):
                payload = service.handle_voice_progressive(b"browser audio", content_type="audio/webm")

            self.assertEqual(payload["mode"], "voice")
            self.assertEqual(payload["input_text"], "请立即急停机械臂")
            self.assertEqual(payload["response_text"], "机械臂急停！")
            self.assertEqual(payload["audio_url"], f"/api/chat-stream/{payload['turn_id']}/audio")

            deadline = time.monotonic() + 2.0
            status = payload
            while time.monotonic() < deadline:
                latest = service.progressive_turn_status(str(payload["turn_id"]))
                if latest is not None:
                    status = latest
                if status["done"]:
                    break
                time.sleep(0.01)

            self.assertEqual(status["status"], "complete")
            self.assertEqual(streamer.started_texts, ["机械臂急停！"])

    def test_handle_voice_progressive_converts_wav_upload_with_distinct_temp_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self.build_service(temp_dir)

            def fake_convert(input_path: Path, output_path: Path) -> None:
                self.assertEqual(input_path.name, "upload.wav")
                self.assertEqual(output_path.name, "recording.wav")
                self.assertNotEqual(input_path, output_path)
                output_path.write_bytes(b"RIFF")

            with patch("local_safety_assistant.web.service._ffmpeg_convert_to_wav", side_effect=fake_convert):
                payload = service.handle_voice_progressive(b"RIFF browser wav", content_type="audio/wav")

            self.assertEqual(payload["mode"], "voice")
            self.assertEqual(payload["input_text"], "请立即急停机械臂")

    def test_handle_voice_progressive_no_answer_skips_moss_stream(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            streamer = FakeMossStreamer()
            service = self.build_service(temp_dir, moss_streamer=streamer, asr_text="thank you for watching")

            def fake_convert(input_path: Path, output_path: Path) -> None:
                output_path.write_bytes(b"RIFF")

            with patch("local_safety_assistant.web.service._ffmpeg_convert_to_wav", side_effect=fake_convert):
                payload = service.handle_voice_progressive(b"browser audio", content_type="audio/webm")

            self.assertEqual(payload["mode"], "voice")
            self.assertEqual(payload["response_text"], "")
            self.assertEqual(payload["metadata"], {"no_answer": True, "reason": "asr_noise"})
            self.assertEqual(payload["audio_url"], None)
            self.assertEqual(payload["status"], "complete")
            self.assertTrue(payload["done"])
            self.assertEqual(payload["tts_segment_count"], 0)
            self.assertEqual(streamer.started_texts, [])

    def test_cancel_active_web_turns_marks_progressive_job_canceled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            streamer = FakeMossStreamer(block_start=True)
            service = self.build_service(temp_dir, moss_streamer=streamer)

            payload = service.handle_chat_progressive("你好")
            self.assertTrue(streamer.started.wait(timeout=1.0))

            canceled = service.cancel_active_web_turns()
            streamer.release.set()
            status = service.progressive_turn_status(str(payload["turn_id"]))

            self.assertTrue(canceled.ok)
            self.assertEqual(canceled.canceled_turns, 1)
            self.assertIsNotNone(status)
            assert status is not None
            self.assertEqual(status["status"], "canceled")
            self.assertTrue(status["done"])

    def test_cancel_active_web_turns_reports_upstream_close_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            streamer = FakeMossStreamer()
            service = self.build_service(temp_dir, moss_streamer=streamer)
            payload = service.handle_chat_progressive("你好")
            deadline = time.monotonic() + 1.0
            turn_status = None
            while time.monotonic() < deadline:
                with service._progressive_turns_lock:
                    turn_job = service._progressive_turns.get(str(payload["turn_id"]))
                turn_status = turn_job.as_dict() if turn_job is not None else None
                if turn_status and turn_status.get("moss_stream_id"):
                    break
                time.sleep(0.01)
            self.assertIsNotNone(turn_status)
            assert turn_status is not None
            self.assertEqual(turn_status["moss_stream_id"], "stream-test")
            with patch.object(streamer, "close_stream", side_effect=RuntimeError("close timeout")):
                canceled = service.cancel_active_web_turns()

            self.assertEqual(canceled.canceled_turns, 1)
            self.assertEqual(service.status(None)["last_error"], "MOSS stream cancellation failed: close timeout")

    def test_cancel_active_web_turns_prevents_inflight_turn_from_starting_moss(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            llm = BlockingFakeLlm("已收到。")
            streamer = FakeMossStreamer()
            service = self.build_service(temp_dir, llm=llm, moss_streamer=streamer)
            result: dict[str, object] = {}
            errors: list[BaseException] = []

            def run_turn() -> None:
                try:
                    result["payload"] = service.handle_chat_progressive("你好")
                except BaseException as error:
                    errors.append(error)

            thread = threading.Thread(target=run_turn)
            thread.start()
            self.assertTrue(llm.started.wait(timeout=1.0))

            canceled = service.cancel_active_web_turns()
            llm.release.set()
            thread.join(timeout=2.0)

            self.assertFalse(errors)
            payload = result["payload"]
            assert isinstance(payload, dict)
            self.assertTrue(canceled.ok)
            self.assertEqual(payload["status"], "canceled")
            self.assertIsNone(payload["audio_url"])
            self.assertEqual(streamer.started_texts, [])

    def test_voice_handlers_reject_oversized_upload_before_ffmpeg(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self.build_service(temp_dir)
            too_large = b"x" * (WEB_VOICE_UPLOAD_MAX_BYTES + 1)

            for handler_name in ("handle_voice", "handle_voice_progressive"):
                with self.subTest(handler=handler_name):
                    with patch("local_safety_assistant.web.service._ffmpeg_convert_to_wav") as convert:
                        with self.assertRaisesRegex(ValueError, "Voice upload too large"):
                            getattr(service, handler_name)(too_large, content_type="audio/webm")

                    self.assertFalse(convert.called)

    def test_voice_handlers_reject_empty_upload_before_ffmpeg(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self.build_service(temp_dir)

            for handler_name in ("handle_voice", "handle_voice_progressive"):
                with self.subTest(handler=handler_name):
                    with patch("local_safety_assistant.web.service._ffmpeg_convert_to_wav") as convert:
                        with self.assertRaisesRegex(ValueError, "audio is required"):
                            getattr(service, handler_name)(b"", content_type="audio/webm")

                    self.assertFalse(convert.called)

    def test_voice_handlers_reject_invalid_webm_upload_as_bad_request(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self.build_service(temp_dir)

            completed = type(
                "Completed",
                (),
                {
                    "returncode": 1,
                    "stderr": "[matroska,webm] EBML header parsing failed\nInvalid data found when processing input",
                    "stdout": "",
                },
            )()

            for handler_name in ("handle_voice", "handle_voice_progressive"):
                with self.subTest(handler=handler_name):
                    with patch("local_safety_assistant.web.service.subprocess.run", return_value=completed):
                        with self.assertRaisesRegex(ValueError, "Invalid uploaded audio; please retry recording\\."):
                            getattr(service, handler_name)(b"not a valid webm", content_type="audio/webm")

    def test_handle_chat_plans_estop_from_user_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self.build_service(temp_dir)

            response = service.handle_chat("请立即急停机械臂")
            arm_rules = load_arm_rule_document(service.stack_config.arm_rules_path)

            self.assertEqual(response.ros2_plan[-1]["topic"], DEFAULT_ESTOP_REQUEST_TOPIC)
            self.assertEqual(response.ros2_plan[-1]["message_type"], ROS2_STRING)
            self.assertIn("web_ui", response.ros2_plan[-1]["payload"])
            self.assertEqual(arm_rules["arm_stop"], "True")
            self.assertEqual(arm_rules["arm_recover"], "False")

    def test_handle_estop_uses_direct_web_reason(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self.build_service(temp_dir)

            response = service.handle_estop()
            arm_rules = load_arm_rule_document(service.stack_config.arm_rules_path)

            self.assertTrue(response.ok)
            self.assertEqual(response.ros2_plan[0]["topic"], DEFAULT_ESTOP_REQUEST_TOPIC)
            self.assertIn("web-ui emergency stop", response.ros2_plan[0]["payload"])
            self.assertEqual(arm_rules["arm_stop"], "True")
            self.assertEqual(arm_rules["arm_recover"], "False")

    def test_estop_release_requires_confirmation_before_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self.build_service(temp_dir)

            response = service.handle_chat("解除急停")

            self.assertEqual(response.ros2_plan, ())
            self.assertIsNotNone(response.confirmation)
            self.assertFalse(service.stack_config.arm_rules_path.exists())
            assert response.confirmation is not None
            self.assertEqual(response.confirmation.action_type, "emergency_stop_release")

            confirmed = service.confirm_pending(str(response.confirmation.confirmation_id))
            arm_rules = load_arm_rule_document(service.stack_config.arm_rules_path)

            self.assertTrue(confirmed.ok)
            self.assertEqual(confirmed.status, "confirmed")
            self.assertEqual(confirmed.ros2_plan[-1]["topic"], DEFAULT_ESTOP_REQUEST_TOPIC)
            self.assertIn('"active": false', confirmed.ros2_plan[-1]["payload"])
            self.assertIn(
                '"reset_sources": ["min_distance_camera"]',
                confirmed.ros2_plan[-1]["payload"],
            )
            self.assertIn("连续三帧距离安全", confirmed.response_text)
            self.assertIn("连续五秒无人", confirmed.response_text)
            self.assertEqual(arm_rules["arm_stop"], "False")
            self.assertEqual(arm_rules["arm_recover"], "True")

    def test_cancel_confirmation_leaves_estop_release_unpublished(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self.build_service(temp_dir)

            response = service.handle_chat("解除急停")
            assert response.confirmation is not None
            canceled = service.cancel_pending(str(response.confirmation.confirmation_id))
            confirmed = service.confirm_pending(str(response.confirmation.confirmation_id))

            self.assertEqual(canceled.status, "canceled")
            self.assertEqual(canceled.ros2_plan, ())
            self.assertFalse(confirmed.ok)
            self.assertEqual(confirmed.status, "missing")
            self.assertFalse(service.stack_config.arm_rules_path.exists())

    def test_cancel_confirmation_is_idempotent_when_id_is_already_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self.build_service(temp_dir)

            response = service.handle_chat("解除急停")
            assert response.confirmation is not None
            confirmation_id = str(response.confirmation.confirmation_id)

            first = service.cancel_pending(confirmation_id)
            repeated = service.cancel_pending(confirmation_id)

            self.assertEqual(first.status, "canceled")
            self.assertEqual(first.response_text, "已取消，未执行任何操作。")
            self.assertTrue(repeated.ok)
            self.assertEqual(repeated.status, "canceled")
            self.assertNotIn("没有可取消", repeated.response_text)
            self.assertIn("本次没有执行任何新操作", repeated.response_text)
            self.assertFalse(service.stack_config.arm_rules_path.exists())

    def test_canceling_stale_id_does_not_remove_newer_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self.build_service(temp_dir)

            first = service.handle_chat("解除急停")
            second = service.handle_chat("解除急停")
            assert first.confirmation is not None
            assert second.confirmation is not None

            canceled = service.cancel_pending(str(first.confirmation.confirmation_id))
            current = service.confirmations.current()

            self.assertTrue(canceled.ok)
            self.assertEqual(canceled.status, "canceled")
            self.assertIn("本次没有执行任何新操作", canceled.response_text)
            self.assertIsNotNone(current)
            assert current is not None
            self.assertEqual(current.confirmation_id, second.confirmation.confirmation_id)
            self.assertFalse(service.stack_config.arm_rules_path.exists())

    def test_expired_confirmation_id_does_not_execute(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self.build_service(temp_dir)
            service.confirmations.ttl_seconds = -0.01

            response = service.handle_chat("解除急停")
            assert response.confirmation is not None
            confirmed = service.confirm_pending(str(response.confirmation.confirmation_id))

            self.assertFalse(confirmed.ok)
            self.assertEqual(confirmed.status, "missing")
            self.assertEqual(confirmed.ros2_plan, ())
            self.assertFalse(service.stack_config.arm_rules_path.exists())
            self.assertEqual(confirmed.as_dict()["error"], "确认请求已过期或已被新的指令替代，未执行任何操作。")

    def test_rule_edit_confirmation_writes_only_after_confirm(self) -> None:
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
                        }
                    ],
                },
            )
            service = self.build_service(
                temp_dir,
                llm=FakeLlm(
                    [
                        '{"type":"tool_call","name":"edit_rules","arguments":{}}',
                        '{"type":"tool_call","name":"edit_rules","arguments":{'
                        '"rule_id":"stop_on_person_intrusion","changes":{"enabled":false}}}',
                    ]
                ),
                rules_path=rules_path,
                require_confirmation=True,
            )

            response = service.handle_chat("暂时禁用人员侵入规则")

            self.assertEqual(load_rule_document(rules_path)["version"], 1)
            self.assertIsNotNone(response.confirmation)
            assert response.confirmation is not None
            self.assertEqual(response.confirmation.action_type, "rule_edit")
            self.assertIn("启用状态改为禁用（原值启用）", response.confirmation.summary)
            self.assertIn("启用状态改为禁用（原值启用）", response.response_text)

            confirmed = service.confirm_pending(str(response.confirmation.confirmation_id))

            updated = load_rule_document(rules_path)
            self.assertTrue(confirmed.ok)
            self.assertEqual(updated["version"], 2)
            self.assertFalse(updated["rules"][0]["enabled"])
            self.assertIn("安全规则已更新", confirmed.response_text)
            self.assertIn("启用状态改为禁用（原值启用）", confirmed.response_text)

    def test_rule_edit_same_value_skips_confirmation_and_write(self) -> None:
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
                        }
                    ],
                },
            )
            service = self.build_service(
                temp_dir,
                llm=FakeLlm(
                    [
                        '{"type":"tool_call","name":"edit_rules","arguments":{}}',
                        '{"type":"tool_call","name":"edit_rules","arguments":{'
                        '"rule_id":"stop_on_person_intrusion","changes":{"enabled":true}}}',
                    ]
                ),
                rules_path=rules_path,
                require_confirmation=True,
            )

            response = service.handle_chat("保持人员侵入规则启用")

            self.assertIsNone(response.confirmation)
            self.assertEqual(load_rule_document(rules_path)["version"], 1)
            self.assertIn("安全规则无需修改", response.response_text)
            self.assertIn("启用状态当前已是启用", response.response_text)
            self.assertIn("未进入确认阶段", response.response_text)

    def test_rule_edit_confirmation_syncs_arm_distance_after_confirm(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rules_path = root / "rules.json"
            arm_rules_path = root / "arm_rules.json"
            write_rule_document(rules_path, personnel_rule_document(distance=0.2))
            write_arm_rule_document(
                arm_rules_path,
                {
                    "arm_capture": "True",
                    "arm_capture_goal": "B",
                    "arm_decelerate": "0.5",
                    "arm_stop": "False",
                    "arm_recover": "True",
                    "arm_safety_distance": "0.2",
                },
            )
            service = self.build_service(
                temp_dir,
                llm=FakeLlm(
                    [
                        '{"type":"tool_call","name":"edit_rules","arguments":{}}',
                        '{"type":"tool_call","name":"edit_rules","arguments":{'
                        '"rule_id":"stop_on_person_intrusion",'
                        '"changes":{"conditions.person_distance_m.lt":0.4}}}',
                    ]
                ),
                rules_path=rules_path,
                arm_rules_path=arm_rules_path,
                require_confirmation=True,
            )

            response = service.handle_chat("把人员安全距离调整为0.4米")

            self.assertEqual(
                load_rule_document(rules_path)["rules"][0]["conditions"]["person_distance_m"]["lt"],
                0.2,
            )
            self.assertEqual(load_arm_rule_document(arm_rules_path)["arm_safety_distance"], "0.2")
            self.assertIsNotNone(response.confirmation)
            assert response.confirmation is not None

            confirmed = service.confirm_pending(str(response.confirmation.confirmation_id))

            updated_rules = load_rule_document(rules_path)
            updated_arm_rules = load_arm_rule_document(arm_rules_path)
            self.assertTrue(confirmed.ok)
            self.assertEqual(
                updated_rules["rules"][0]["conditions"]["person_distance_m"]["lt"],
                0.4,
            )
            self.assertEqual(updated_arm_rules["arm_safety_distance"], "0.4")
            self.assertEqual(updated_arm_rules["arm_capture"], "True")
            self.assertEqual(updated_arm_rules["arm_capture_goal"], "B")
            self.assertEqual(updated_arm_rules["arm_decelerate"], "0.5")
            self.assertEqual(updated_arm_rules["arm_recover"], "True")
            self.assertIn("机械臂运行时安全距离已同步为 0.4 米", confirmed.response_text)

    def test_rule_edit_confirmation_cancel_leaves_rules_and_arm_distance_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rules_path = root / "rules.json"
            arm_rules_path = root / "arm_rules.json"
            write_rule_document(rules_path, personnel_rule_document(distance=0.2))
            write_arm_rule_document(arm_rules_path, {"arm_safety_distance": "0.2"})
            service = self.build_service(
                temp_dir,
                llm=FakeLlm(
                    [
                        '{"type":"tool_call","name":"edit_rules","arguments":{}}',
                        '{"type":"tool_call","name":"edit_rules","arguments":{'
                        '"rule_id":"stop_on_person_intrusion",'
                        '"changes":{"conditions.person_distance_m.lt":0.4}}}',
                    ]
                ),
                rules_path=rules_path,
                arm_rules_path=arm_rules_path,
                require_confirmation=True,
            )

            response = service.handle_chat("把人员安全距离调整为0.4米")
            assert response.confirmation is not None
            canceled = service.cancel_pending(str(response.confirmation.confirmation_id))

            self.assertEqual(canceled.status, "canceled")
            self.assertEqual(
                load_rule_document(rules_path)["rules"][0]["conditions"]["person_distance_m"]["lt"],
                0.2,
            )
            self.assertEqual(load_arm_rule_document(arm_rules_path)["arm_safety_distance"], "0.2")

    def test_rule_edit_confirmation_expiry_leaves_rules_and_arm_distance_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rules_path = root / "rules.json"
            arm_rules_path = root / "arm_rules.json"
            write_rule_document(rules_path, personnel_rule_document(distance=0.2))
            write_arm_rule_document(arm_rules_path, {"arm_safety_distance": "0.2"})
            service = self.build_service(
                temp_dir,
                llm=FakeLlm(
                    [
                        '{"type":"tool_call","name":"edit_rules","arguments":{}}',
                        '{"type":"tool_call","name":"edit_rules","arguments":{'
                        '"rule_id":"stop_on_person_intrusion",'
                        '"changes":{"conditions.person_distance_m.lt":0.4}}}',
                    ]
                ),
                rules_path=rules_path,
                arm_rules_path=arm_rules_path,
                require_confirmation=True,
            )
            service.confirmations.ttl_seconds = -0.01

            response = service.handle_chat("把人员安全距离调整为0.4米")
            assert response.confirmation is not None
            confirmed = service.confirm_pending(str(response.confirmation.confirmation_id))

            self.assertFalse(confirmed.ok)
            self.assertEqual(confirmed.status, "missing")
            self.assertEqual(
                load_rule_document(rules_path)["rules"][0]["conditions"]["person_distance_m"]["lt"],
                0.2,
            )
            self.assertEqual(load_arm_rule_document(arm_rules_path)["arm_safety_distance"], "0.2")

    def test_object_mapping_confirmation_writes_only_after_confirm(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            streamer = FakeMossStreamer()
            mapping_path = Path(temp_dir) / "object_mapping.json"
            write_object_mapping_document(
                mapping_path,
                {
                    "version": 1,
                    "markers": {
                        "A": {"object": "红色方块", "enabled": True},
                        "B": {"object": "蓝色圆柱", "enabled": True},
                        "C": {"object": "扳手", "enabled": True},
                        "D": {"object": "空位", "enabled": True},
                    },
                },
            )
            service = self.build_service(
                temp_dir,
                llm=FakeLlm('{"type":"tool_call","name":"update_object_mapping","arguments":{}}'),
                object_mapping_path=mapping_path,
                require_confirmation=True,
                moss_streamer=streamer,
            )

            response = service.handle_chat("把标号A的映射改成夹具")

            self.assertEqual(load_object_mapping_document(mapping_path)["markers"]["A"]["object"], "红色方块")
            self.assertIsNotNone(response.confirmation)
            assert response.confirmation is not None
            self.assertEqual(response.confirmation.action_type, "object_mapping_update")
            self.assertIn("原值红色方块，改为夹具", response.confirmation.summary)
            self.assertIn("原值红色方块，改为夹具", response.response_text)

            confirmed = service.confirm_pending(str(response.confirmation.confirmation_id))

            self.assertTrue(confirmed.ok)
            self.assertEqual(load_object_mapping_document(mapping_path)["markers"]["A"]["object"], "夹具")
            self.assertIn("已更新映射", confirmed.response_text)
            self.assertIn("现在对应夹具，原来对应红色方块", confirmed.response_text)
            confirmed_payload = confirmed.as_dict()
            self.assertTrue(str(confirmed_payload["audio_url"]).startswith("/api/chat-stream/"))
            self.assertEqual(confirmed_payload["mode"], "confirmation")
            self.assertTrue(streamer.started.wait(timeout=1.0))
            self.assertEqual(streamer.started_texts, [confirmed.response_text])

    def test_object_mapping_same_value_skips_confirmation_and_write(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "object_mapping.json"
            write_object_mapping_document(
                mapping_path,
                {
                    "version": 1,
                    "markers": {
                        "A": {"object": "红色方块", "enabled": True},
                        "B": {"object": "蓝色圆柱", "enabled": True},
                    },
                },
            )
            service = self.build_service(
                temp_dir,
                llm=FakeLlm('{"type":"tool_call","name":"update_object_mapping","arguments":{}}'),
                object_mapping_path=mapping_path,
                require_confirmation=True,
            )

            response = service.handle_chat("把标号A的映射改成红色方块")

            self.assertIsNone(response.confirmation)
            self.assertEqual(load_object_mapping_document(mapping_path)["version"], 1)
            self.assertIn("物体映射无需修改", response.response_text)
            self.assertIn("标号A 当前已经对应红色方块", response.response_text)
            self.assertIn("未进入确认阶段", response.response_text)

    def test_cancel_confirmation_returns_assistant_text_and_playback_stream(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            streamer = FakeMossStreamer()
            service = self.build_service(temp_dir, moss_streamer=streamer)

            response = service.handle_chat("解除急停")
            assert response.confirmation is not None
            canceled = service.cancel_pending(str(response.confirmation.confirmation_id))
            canceled_payload = canceled.as_dict()

            self.assertEqual(canceled.status, "canceled")
            self.assertEqual(canceled.response_text, "已取消，未执行任何操作。")
            self.assertTrue(str(canceled_payload["audio_url"]).startswith("/api/chat-stream/"))
            self.assertEqual(canceled_payload["mode"], "confirmation")
            self.assertTrue(streamer.started.wait(timeout=1.0))
            self.assertEqual(streamer.started_texts, [canceled.response_text])

    def test_spoken_object_mapping_confirmation_resolves_dialog_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "object_mapping.json"
            write_object_mapping_document(
                mapping_path,
                {
                    "version": 1,
                    "markers": {
                        "A": {"object": "红色方块", "enabled": True},
                        "B": {"object": "蓝色圆柱", "enabled": True},
                    },
                },
            )
            service = self.build_service(
                temp_dir,
                llm=FakeLlm('{"type":"tool_call","name":"update_object_mapping","arguments":{}}'),
                object_mapping_path=mapping_path,
                require_confirmation=True,
                asr_text="确认更新",
            )

            response = service.handle_chat("把标号A的映射改成夹具")
            self.assertIsNotNone(response.confirmation)

            def fake_convert(input_path: Path, output_path: Path) -> None:
                output_path.write_bytes(b"RIFF")

            with patch("local_safety_assistant.web.service._ffmpeg_convert_to_wav", side_effect=fake_convert):
                payload = service.handle_voice_progressive(b"browser audio", content_type="audio/webm")

            self.assertIsNone(payload["confirmation"])
            self.assertEqual(
                payload["metadata"],
                {"confirmation_resolved": True, "confirmation_status": "confirmed"},
            )
            self.assertIn("已更新映射", payload["response_text"])
            self.assertEqual(load_object_mapping_document(mapping_path)["markers"]["A"]["object"], "夹具")

    def test_object_mapping_ambiguous_rename_explains_required_phrase(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "object_mapping.json"
            write_object_mapping_document(
                mapping_path,
                {
                    "version": 1,
                    "markers": {
                        "A": {"object": "红色方块", "enabled": True},
                        "B": {"object": "蓝色圆柱", "enabled": True},
                        "C": {"object": "扳手", "enabled": True},
                        "D": {"object": "空位", "enabled": True},
                    },
                },
            )
            service = self.build_service(
                temp_dir,
                llm=FakeLlm('{"type":"tool_call","name":"edit_object_mapping","arguments":{}}'),
                object_mapping_path=mapping_path,
                require_confirmation=True,
            )

            response = service.handle_chat("把标号D改为工具箱")

            self.assertIsNone(response.confirmation)
            self.assertEqual(load_object_mapping_document(mapping_path)["markers"]["D"]["object"], "空位")
            self.assertIn("这次没有执行修改", response.response_text)
            self.assertIn("把标号D的映射改为工具箱", response.response_text)
            self.assertIn("把标号D的物体改为工具箱", response.response_text)

    def test_object_grasp_requires_confirmation_before_execution_module(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "object_mapping.json"
            arm_rules_path = Path(temp_dir) / "arm_rules.json"
            write_object_mapping_document(
                mapping_path,
                {
                    "version": 1,
                    "markers": {
                        "A": {"object": "红色方块", "enabled": True},
                        "B": {"object": "蓝色圆柱", "enabled": True},
                        "C": {"object": "扳手", "enabled": True},
                        "D": {"object": "空位", "enabled": True},
                    },
                },
            )
            write_arm_rule_document(arm_rules_path, {"arm_capture_goal": "B"})
            service = self.build_service(
                temp_dir,
                object_mapping_path=mapping_path,
                arm_rules_path=arm_rules_path,
                require_confirmation=True,
            )

            response = service.handle_chat("帮我抓取标号A")

            self.assertEqual(response.ros2_plan, ())
            self.assertIsNotNone(response.confirmation)
            assert response.confirmation is not None
            self.assertEqual(response.confirmation.action_type, "object_grasp_execution")
            self.assertEqual(response.confirmation.details["marker"], "A")
            self.assertEqual(response.confirmation.details["object_name"], "红色方块")
            self.assertIn("标号A", response.confirmation.summary)
            self.assertIn("红色方块", response.confirmation.summary)

            confirmed = service.confirm_pending(str(response.confirmation.confirmation_id))

            self.assertTrue(confirmed.ok)
            self.assertEqual(confirmed.ros2_plan, ())
            self.assertIn("抓取已确认", confirmed.response_text)
            self.assertIn("已写入机械臂 JSON 执行请求", confirmed.response_text)
            arm_rules = load_arm_rule_document(arm_rules_path)
            self.assertEqual(arm_rules["arm_capture"], "True")
            self.assertEqual(arm_rules["arm_capture_goal"], "A")
            self.assertEqual(arm_rules["arm_capture_object"], "红色方块")
            self.assertEqual(arm_rules["arm_capture_original_text"], "帮我抓取标号A")

    def test_arm_deceleration_requires_confirmation_before_json_write(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            arm_rules_path = Path(temp_dir) / "arm_rules.json"
            write_arm_rule_document(
                arm_rules_path,
                {
                    "arm_capture": "True",
                    "arm_capture_goal": "B",
                    "arm_decelerate": "0.5",
                    "arm_stop": "False",
                    "arm_recover": "True",
                    "arm_safety_distance": "0.4",
                },
            )
            service = self.build_service(
                temp_dir,
                arm_rules_path=arm_rules_path,
                require_confirmation=True,
            )

            response = service.handle_chat("减速到30%")

            self.assertEqual(response.ros2_plan, ())
            self.assertIsNotNone(response.confirmation)
            assert response.confirmation is not None
            self.assertEqual(response.confirmation.action_type, "speed_change")
            self.assertEqual(response.confirmation.details["target_speed_percent"], 30.0)
            self.assertEqual(response.confirmation.details["arm_decelerate"], "0.3")
            self.assertEqual(load_arm_rule_document(arm_rules_path)["arm_decelerate"], "0.5")

            confirmed = service.confirm_pending(str(response.confirmation.confirmation_id))

            self.assertTrue(confirmed.ok)
            self.assertEqual(confirmed.ros2_plan, ())
            self.assertIn("目标速度为 30%", confirmed.response_text)
            self.assertIn("已写入机械臂 JSON 执行请求", confirmed.response_text)
            arm_rules = load_arm_rule_document(arm_rules_path)
            self.assertEqual(arm_rules["arm_decelerate"], "0.3")
            self.assertEqual(arm_rules["arm_capture"], "True")
            self.assertEqual(arm_rules["arm_capture_goal"], "B")
            self.assertEqual(arm_rules["arm_recover"], "True")
            self.assertEqual(arm_rules["arm_safety_distance"], "0.4")

    def test_later_confirmation_supersedes_stale_confirmation_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self.build_service(temp_dir)

            first = service.handle_chat("解除急停")
            assert first.confirmation is not None
            second = service.handle_chat("移动到目标 0.1 0.2 0.3")
            assert second.confirmation is not None
            stale = service.confirm_pending(str(first.confirmation.confirmation_id))

            self.assertFalse(stale.ok)
            self.assertEqual(stale.status, "missing")

    def test_serve_audio_path_rejects_missing_or_traversal_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self.build_service(temp_dir)
            audio = service.config.audio_dir / "reply.wav"
            audio.parent.mkdir(parents=True, exist_ok=True)
            audio.write_bytes(b"RIFF")

            self.assertEqual(service.serve_audio_path("reply.wav"), audio.resolve())
            self.assertIsNone(service.serve_audio_path("../secret.wav"))
            self.assertIsNone(service.serve_audio_path("missing.wav"))

    def test_serve_image_path_rejects_missing_or_traversal_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self.build_service(temp_dir)
            image = service.config.image_dir / "snapshot.jpg"
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"jpeg")

            self.assertEqual(service.serve_image_path("snapshot.jpg"), image.resolve())
            self.assertIsNone(service.serve_image_path("../secret.jpg"))
            self.assertIsNone(service.serve_image_path("missing.jpg"))

    def test_resolve_tts_engine_prefers_auto_order_and_allows_no_tts(self) -> None:
        self.assertEqual(_resolve_tts_engine("auto", ("melo",)), "melo")
        self.assertEqual(_resolve_tts_engine("auto", ()), "none")
        self.assertEqual(_resolve_tts_engine("piper", ("melo",)), "piper")

    def test_static_ui_reads_pcm_stream_with_web_audio(self) -> None:
        default_html = render_index_html("默认测试控制台")
        html = render_index_html("测试控制台", moss_pcm_buffer_seconds=3.5)

        self.assertEqual(WebUiConfig().moss_pcm_buffer_seconds, 0.48)
        self.assertIn("PCM_TARGET_BUFFER_SECONDS = 0.480", default_html)
        self.assertIn("response.body.getReader()", html)
        self.assertIn("schedulePcmChunk", html)
        self.assertIn("PCM_TARGET_BUFFER_SECONDS = 3.500", html)
        self.assertIn("PCM_REBUFFER_SECONDS = 0.64", html)
        self.assertIn("PCM_MIN_SCHEDULE_SECONDS = 0.16", html)
        self.assertIn("? PCM_TARGET_BUFFER_SECONDS", html)
        self.assertIn("underflowed ? PCM_REBUFFER_SECONDS", html)
        self.assertIn("pcmByteLengthForSeconds(requiredSeconds, sampleRate, channels)", html)
        self.assertIn("scheduleBufferedPcm", html)
        self.assertIn("PCM_STREAM_FADE_SECONDS", html)
        self.assertIn("PCM_FINAL_FADE_SECONDS", html)
        self.assertIn("playbackState.fadeStarted", html)
        self.assertIn("underflowed", html)
        self.assertIn("applyFinalPcmFade(audioContext, playbackState)", html)
        self.assertIn("if (endAt <= now)", html)
        self.assertIn("stopActivePlayback()", html)
        self.assertIn("resetPlaybackQueue()", html)
        self.assertIn("state.playback = Promise.resolve();", html)
        self.assertIn("new AbortController()", html)
        self.assertIn("signal: session.controller.signal", html)
        self.assertIn("state.activePlayback = session", html)
        self.assertIn("createGain()", html)
        self.assertIn("linearRampToValueAtTime", html)
        self.assertIn("playbackState.started = true", html)
        self.assertNotIn("audioBuffer.duration - fadeDuration", html)
        self.assertNotIn('id="recordBtn"', html)
        self.assertNotIn("recordBtn", html)
        self.assertNotIn("toggleRecording", html)
        self.assertIn('class="ghost-btn free-voice-toggle"', html)
        self.assertIn("activeRequestControllers: new Set()", html)
        self.assertIn("requestEpoch", html)
        self.assertIn("cancelActiveWebWork", html)
        self.assertIn("interruptActiveWebWorkLocally", html)
        self.assertIn('/api/turn/cancel', html)
        self.assertIn("isStaleRequest(session)", html)
        self.assertIn("window.isSecureContext", html)
        self.assertIn("image_artifacts", html)
        self.assertIn("message-image", html)
        self.assertIn("imageCaption", html)
        self.assertIn("confirmationModal", html)
        self.assertIn("/api/confirmation/confirm", html)
        self.assertIn("freeVoiceBtn", html)
        self.assertIn("voiceSphere", html)
        self.assertIn(".free-voice-panel", html)
        self.assertIn("position: fixed", html)
        self.assertIn("FREE_VOICE_SPEECH_THRESHOLD = 0.2", html)
        self.assertIn("FREE_VOICE_TRAILING_SILENCE_MS = 1300", html)
        self.assertIn("FREE_VOICE_PRE_ROLL_MS = 500", html)
        self.assertIn("FREE_VOICE_RECORDER_TIMESLICE_MS = 100", html)
        self.assertIn("preRollChunks", html)
        self.assertIn("preRollHeaderChunk", html)
        self.assertIn("trimFreeVoicePreRoll", html)
        self.assertIn("recorder.start(FREE_VOICE_RECORDER_TIMESLICE_MS)", html)
        self.assertIn("voice.chunks = buildFreeVoiceUtteranceChunks()", html)
        self.assertIn("chunks.unshift(voice.preRollHeaderChunk)", html)
        self.assertIn("getByteTimeDomainData", html)
        self.assertIn("confirmation_resolved", html)
        self.assertIn("showConfirmation(null)", html)
        self.assertIn("appendConfirmationAssistantMessage", html)
        self.assertIn('appendConfirmationAssistantMessage(payload, "已确认")', html)
        self.assertIn('appendConfirmationAssistantMessage(payload, "已取消")', html)
        self.assertIn("await followProgressiveTurn(payload, session)", html)
        self.assertIn('signal: session.controller.signal', html)
        self.assertNotIn("playAudioUrls", html)

    def test_static_ui_waits_for_backend_cancel_before_superseding_turns(self) -> None:
        html = render_index_html("测试控制台")
        cancel_source = html[
            html.index("async function requestBackendCancel") : html.index("function showConfirmation")
        ]
        chat_source = html[html.index("async function sendChat") : html.index("function wait")]
        voice_source = html[
            html.index("async function sendVoice") : html.index("function setFreeVoiceVisualState")
        ]

        self.assertIn("async function requestBackendCancel(reason)", cancel_source)
        self.assertIn("body: JSON.stringify({ reason })", cancel_source)
        self.assertIn("cancelBarrier: Promise.resolve()", html)
        self.assertIn("state.cancelBarrier.then(() => requestBackendCancel(reason))", cancel_source)
        self.assertIn("state.cancelBarrier = backendCancel.catch(() => {})", cancel_source)
        self.assertLess(cancel_source.index("stopActivePlayback()"), cancel_source.index("abortActiveRequests()"))
        self.assertLess(
            cancel_source.index("state.cancelBarrier.then(() => requestBackendCancel(reason))"),
            cancel_source.index("await backendCancel"),
        )
        self.assertLess(
            cancel_source.index("await backendCancel"),
            cancel_source.index("return cancellationEpoch"),
        )
        self.assertLess(
            chat_source.index('await cancelActiveWebWork("text_superseded")'),
            chat_source.index("const session = beginRequestSession()"),
        )
        self.assertLess(
            chat_source.index("const session = beginRequestSession()"),
            chat_source.index('fetchJson("/api/chat-stream"'),
        )
        self.assertLess(
            voice_source.index('await cancelActiveWebWork("voice_superseded")'),
            voice_source.index("const session = beginRequestSession()"),
        )
        self.assertLess(
            voice_source.index("const session = beginRequestSession()"),
            voice_source.index('fetch("/api/voice-stream"'),
        )
        self.assertIn("supersessionEpoch !== state.requestEpoch", chat_source)
        self.assertIn("supersessionEpoch !== state.requestEpoch", voice_source)

    def test_web_parser_validates_moss_buffer_and_cpu_affinity(self) -> None:
        parser = build_parser()

        args = parser.parse_args(["--moss-pcm-buffer-seconds", "4", "--moss-cpus", "0,2-3"])

        self.assertEqual(parser.parse_args([]).moss_pcm_buffer_seconds, 0.48)
        self.assertEqual(args.moss_pcm_buffer_seconds, 4.0)
        self.assertEqual(args.moss_cpus, "0,2-3")
        self.assertIsNone(parser.parse_args(["--moss-cpus", ""]).moss_cpus)
        for arguments in (
            ["--moss-pcm-buffer-seconds", "0"],
            ["--moss-pcm-buffer-seconds", "invalid"],
            ["--moss-cpus", "0 through 3"],
            ["--moss-cpus", "3-1"],
            ["--moss-cpus", "999"],
        ):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                parser.parse_args(arguments)

    def test_web_runtime_args_propagate_moss_cpu_affinity_to_tts_config(self) -> None:
        web_config = WebUiConfig(moss_cpu_affinity="0,2-3")
        stack_config = VoiceStackConfig(
            moss_tts=MossTtsConfig(cpu_affinity=web_config.moss_cpu_affinity),
        )

        runtime_args = _build_runtime_args(web_config, stack_config, "moss")
        moss_config = build_moss_tts_config(runtime_args)

        self.assertEqual(runtime_args.moss_cpus, "0,2-3")
        self.assertEqual(moss_config.cpu_affinity, "0,2-3")

    def _min_dis_estop_event(self, *, active: bool = True, distance: float = 0.42) -> dict[str, object]:
        return {
            "source": "min_distance_camera",
            "active": active,
            "latch": False,
            "reason": "person distance below threshold: 0.42m",
            "threshold_m": 0.5,
            "distance_m": distance,
            "trigger_distance_m": 0.42,
            "release_distance_m": 0.55,
        }

    def test_external_estop_event_cancels_progressive_turn_and_updates_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self.build_service(temp_dir)
            turn = service.handle_chat_progressive("你好")

            result = service.ingest_external_estop_event(
                json.dumps(self._min_dis_estop_event(), ensure_ascii=False)
            )
            status = service.status(None)
            job_status = service.progressive_turn_status(turn["turn_id"])

        self.assertTrue(result["ok"])
        self.assertFalse(result["ignored"])
        self.assertTrue(result["new_active"])
        self.assertGreaterEqual(result["canceled_turns"], 1)
        event = status["emergency_event"]
        self.assertIsNotNone(event)
        self.assertTrue(event["active"])
        self.assertEqual(event["source"], "min_distance_camera")
        self.assertEqual(event["distance_m"], 0.42)
        self.assertEqual(event["trigger_distance_m"], 0.42)
        self.assertEqual(event["threshold_m"], 0.5)
        self.assertEqual(event["release_distance_m"], 0.55)
        self.assertEqual(event["event_id"], 1)
        self.assertIn("person distance below threshold", event["reason"])
        self.assertIsNotNone(job_status)
        self.assertEqual(job_status["status"], "canceled")

    def test_external_estop_event_dedupes_repeated_active_messages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self.build_service(temp_dir)

            first = service.ingest_external_estop_event(self._min_dis_estop_event())
            second = service.ingest_external_estop_event(self._min_dis_estop_event(distance=0.4))
            status = service.status(None)

        self.assertTrue(first["new_active"])
        self.assertFalse(second["new_active"])
        self.assertEqual(second["event"]["event_id"], first["event"]["event_id"])
        self.assertEqual(status["emergency_event"]["event_id"], first["event"]["event_id"])
        self.assertEqual(status["emergency_event"]["trigger_distance_m"], 0.42)
        self.assertEqual(status["emergency_event"]["distance_m"], 0.4)

    def test_external_estop_clear_keeps_event_id_and_retrigger_creates_new_alert(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self.build_service(temp_dir)

            first = service.ingest_external_estop_event(self._min_dis_estop_event())
            cleared = service.ingest_external_estop_event(self._min_dis_estop_event(active=False))
            cleared_status = service.status(None)
            retriggered = service.ingest_external_estop_event(self._min_dis_estop_event())

        self.assertFalse(cleared["ignored"])
        self.assertFalse(cleared["event"]["active"])
        self.assertEqual(cleared["event"]["event_id"], first["event"]["event_id"])
        self.assertFalse(cleared_status["emergency_event"]["active"])
        self.assertTrue(retriggered["new_active"])
        self.assertEqual(retriggered["event"]["event_id"], first["event"]["event_id"] + 1)

    def test_external_estop_latched_clear_keeps_source_active(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self.build_service(temp_dir)

            first = service.ingest_external_estop_event(self._min_dis_estop_event())
            latched_clear = service.ingest_external_estop_event(
                {"source": "min_distance_camera", "active": False, "latch": True}
            )
            latched_status = service.status(None)
            reassert = service.ingest_external_estop_event(self._min_dis_estop_event())

        # active=false,latch=true must not clear the source per the aggregator contract.
        self.assertTrue(latched_clear["ignored"])
        self.assertEqual(latched_clear["reason"], "latched source remains active")
        self.assertTrue(latched_status["emergency_event"]["active"])
        self.assertEqual(latched_status["emergency_event"]["event_id"], first["event"]["event_id"])
        # Re-asserting the still-latched source must not replay the alert.
        self.assertFalse(reassert["new_active"])
        self.assertEqual(reassert["event"]["event_id"], first["event"]["event_id"])

    def test_external_estop_ignores_web_source_and_rejects_invalid_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self.build_service(temp_dir)

            result = service.ingest_external_estop_event({"source": "web_ui", "active": True, "latch": True})

            self.assertTrue(result["ignored"])
            self.assertIsNone(service.current_emergency_event())

            for payload in (
                "not json",
                "[1, 2]",
                {"active": True},
                {"source": "", "active": True},
                {"source": "min_distance_camera", "active": "yes"},
                {"source": "min_distance_camera", "active": True, "latch": "no"},
            ):
                with self.assertRaises(ValueError):
                    service.ingest_external_estop_event(payload)
            self.assertIsNone(service.current_emergency_event())

    def test_status_exposes_builtin_emergency_alert_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self.build_service(temp_dir)

            status = service.status(None)

        self.assertEqual(service.config.resolved_emergency_alert_audio, DEFAULT_EMERGENCY_ALERT_AUDIO_PATH)
        self.assertEqual(status["emergency_alert_audio_url"], DEFAULT_EMERGENCY_ALERT_AUDIO_URL)
        self.assertTrue(status["emergency_alert_audio_available"])
        self.assertEqual(status["emergency_alert_audio_path"], str(DEFAULT_EMERGENCY_ALERT_AUDIO_PATH))

    def test_builtin_emergency_alert_audio_is_valid_pcm_wav(self) -> None:
        with wave.open(str(DEFAULT_EMERGENCY_ALERT_AUDIO_PATH), "rb") as audio:
            self.assertEqual(audio.getnchannels(), 1)
            self.assertEqual(audio.getsampwidth(), 2)
            self.assertEqual(audio.getframerate(), 44100)
            self.assertGreater(audio.getnframes(), audio.getframerate())

    def test_serve_emergency_alert_audio_path_requires_configured_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_path = Path(temp_dir) / "assets" / "alert.wav"
            config = WebUiConfig(runtime_dir=Path(temp_dir), ros2_dry_run=True, emergency_alert_audio=custom_path)
            pipeline = VoicePipeline(asr=FakeAsr(), llm=FakeLlm(), tts=FakeTts(config.audio_dir), rules_path=None)
            service = WebUiService(
                config=config,
                pipeline=pipeline,
                ros2_bridge=Ros2VoiceBridge(Ros2BridgeConfig(source="web_ui")),
                stack_config=VoiceStackConfig(),
                resolved_tts_engine="moss",
                available_tts_engines=("moss",),
                moss_streamer=FakeMossStreamer(),
            )

            self.assertIsNone(service.serve_emergency_alert_audio_path())
            custom_path.parent.mkdir(parents=True, exist_ok=True)
            custom_path.write_bytes(b"RIFF")
            self.assertEqual(service.serve_emergency_alert_audio_path(), custom_path)

    def test_static_ui_includes_emergency_alert_hooks(self) -> None:
        html = render_index_html("测试控制台")

        self.assertIn('id="emergencyAlertModal"', html)
        self.assertIn('id="emergencyAlertTitle"', html)
        self.assertIn('id="emergencyAlertDetails"', html)
        self.assertIn('id="emergencyAlertAudioStatus"', html)
        self.assertIn('id="emergencyAlertDismissBtn"', html)
        self.assertIn("紧急停止告警", html)
        self.assertIn("EMERGENCY_ALERT_POLL_MS = 2000", html)
        self.assertIn("startEmergencyStatusPolling", html)
        self.assertIn("handleEmergencyStatusPayload(payload)", html)
        self.assertIn("emergency_alert_audio_url", html)
        self.assertIn("new Audio(audioUrl)", html)
        self.assertIn("event.event_id === state.emergency.seenEventId", html)
        self.assertIn('!emergencyAlertModal.classList.contains("hidden")', html)
        self.assertIn("updateEmergencyAlertDetails(event);", html)
        self.assertIn("playEmergencyAlertAudio", html)
        self.assertIn("emergencyEventDetailLines", html)
        self.assertIn("return value.toFixed(2)", html)
        self.assertIn("formatEmergencyReason(event.reason)", html)
        self.assertIn("触发距离：${formatEmergencyDistance(event.trigger_distance_m)} 米", html)
        self.assertIn("当前人机距离：${formatEmergencyDistance(event.distance_m)} 米", html)
        self.assertIn("当前人机距离：不可用", html)
        self.assertIn("解除门槛：${formatEmergencyDistance(event.release_distance_m)} 米", html)
        self.assertIn("未配置急停告警音频文件", html)
        self.assertIn("告警音频播放被浏览器拦截或失败", html)
        self.assertIn("不会解除急停", html)
        # A new active external event must interrupt local work and any dialog.
        self.assertIn("interruptActiveWebWorkLocally();\n      showConfirmation(null);", html)

    def test_warm_start_loads_default_local_stack_and_excludes_9b_rule_editor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = WebUiConfig(runtime_dir=Path(temp_dir), ros2_dry_run=True)
            asr = WarmableFakeAsr()
            llm = WarmableFakeLlm(model="qwen35-2b")
            streamer = WarmableFakeMossStreamer()
            pipeline = VoicePipeline(asr=asr, llm=llm, tts=None, rules_path=None)
            service = WebUiService(
                config=config,
                pipeline=pipeline,
                ros2_bridge=Ros2VoiceBridge(Ros2BridgeConfig(source="web_ui")),
                stack_config=VoiceStackConfig(),
                resolved_tts_engine="moss",
                available_tts_engines=("moss",),
                moss_streamer=streamer,
            )

            report = service.warm_start()
            status = service.status(None)
            service.shutdown()

        self.assertTrue(report["ready"])
        self.assertEqual(asr.warm_calls, 1)
        self.assertEqual(llm.warm_calls, 1)
        self.assertEqual(streamer.warm_calls, 1)
        self.assertEqual(streamer.shutdown_calls, 1)
        self.assertEqual(report["excluded_models"], ["qwen35-9b"])
        self.assertEqual(status["warmup"], report)


class MossStreamingServerTest(unittest.TestCase):
    def build_config(
        self,
        root: Path,
        *,
        cpu_threads: int = 4,
        timeout_seconds: float = 180.0,
        use_prompt_audio: bool = True,
    ) -> MossTtsConfig:
        executable = root / "moss-tts-nano"
        source_dir = root / "MOSS-TTS-Nano"
        model_dir = root / "models" / "tts"
        prompt_audio = source_dir / "assets" / "audio" / "zh_11.wav"
        demo_metadata = source_dir / "assets" / "demo.jsonl"
        executable.touch()
        source_dir.mkdir(parents=True, exist_ok=True)
        if use_prompt_audio:
            prompt_audio.parent.mkdir(parents=True, exist_ok=True)
            prompt_audio.write_bytes(b"RIFF")
            demo_metadata.write_text(
                json.dumps({"role": "assets/audio/zh_11.wav", "text": "demo"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        model_dir.mkdir(parents=True)
        return MossTtsConfig(
            executable=executable,
            source_dir=source_dir,
            model_dir=model_dir,
            output_dir=root / "out",
            prompt_audio=prompt_audio if use_prompt_audio else None,
            cpu_threads=cpu_threads,
            timeout_seconds=timeout_seconds,
        )

    def test_builtin_voice_mode_is_available_without_demo_prompt_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.build_config(Path(temp_dir), use_prompt_audio=False)

            available = _moss_available(config)

        self.assertTrue(available)

    def test_start_command_binds_localhost_and_uses_configured_cpu_threads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            server = MossStreamingServer(self.build_config(root, cpu_threads=4), audio_dir=root / "audio")
            server.port = 18083

            command = server._build_start_command()

        self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
        self.assertEqual(command[command.index("--cpu-threads") + 1], "4")
        self.assertEqual(command[:3], [str(root / "moss-tts-nano"), "serve", "--backend"])

    def test_status_includes_upstream_health_when_process_is_running(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            server = MossStreamingServer(self.build_config(root), audio_dir=root / "audio")
            server.base_url = "http://127.0.0.1:18083"
            server._process = FakeMossProcess()  # type: ignore[assignment]
            server._get_json = lambda url, timeout: {  # type: ignore[method-assign]
                "status": "ok",
                "runtime_manager": {
                    "runtime_count": 1,
                    "default_runtime_threads": 4,
                    "loaded_runtime_threads": [4],
                },
            }

            status = server.status()

        self.assertTrue(status["running"])
        self.assertEqual(status["upstream_health"]["runtime_manager"]["runtime_count"], 1)
        self.assertEqual(status["upstream_health"]["runtime_manager"]["default_runtime_threads"], 4)

    def test_get_json_uses_proxyless_local_upstream_opener(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            server = MossStreamingServer(self.build_config(root), audio_dir=root / "audio")

            class FakeResponse:
                def __enter__(self) -> "FakeResponse":
                    return self

                def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
                    return None

                def read(self) -> bytes:
                    return b'{"status":"ok"}'

            with patch("local_safety_assistant.web.service._LOCAL_UPSTREAM_OPENER.open", return_value=FakeResponse()) as open_call:
                payload = server._get_json("http://127.0.0.1:18083/health", timeout=0.5)

        self.assertEqual(payload, {"status": "ok"})
        open_call.assert_called_once_with("http://127.0.0.1:18083/health", timeout=0.5)

    def test_close_stream_posts_to_upstream_close_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            server = MossStreamingServer(self.build_config(root), audio_dir=root / "audio")

            class FakeResponse:
                def __enter__(self) -> "FakeResponse":
                    return self

                def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
                    return None

                def read(self) -> bytes:
                    return b"{}"

            with patch("local_safety_assistant.web.service._LOCAL_UPSTREAM_OPENER.open", return_value=FakeResponse()) as open_call:
                server.close_stream("http://127.0.0.1:18083/api/generate-stream/stream-1/audio")
                server.close_stream("http://127.0.0.1:18083/api/generate-stream/stream-1/result")

        open_call.assert_called_once()
        request = open_call.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:18083/api/generate-stream/stream-1/close")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(open_call.call_args.kwargs["timeout"], MOSS_STREAM_CLOSE_REQUEST_TIMEOUT_SECONDS)

    def test_stream_start_request_uses_startup_cpu_threads_not_request_variants(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            server = MossStreamingServer(self.build_config(root, cpu_threads=4), audio_dir=root / "audio")
            server.base_url = "http://127.0.0.1:18083"
            observed: dict[str, object] = {}

            def fake_post(path: str, form: dict[str, str], *, timeout: float) -> dict[str, object]:
                observed["path"] = path
                observed["form"] = form
                return {
                    "stream_id": "stream-test",
                    "audio_url": "/audio",
                    "status_url": "/status",
                    "result_url": "/result",
                    "sample_rate": 48000,
                    "channels": 2,
                }

            server.ensure_running = lambda: None  # type: ignore[method-assign]
            server._post_form = fake_post  # type: ignore[method-assign]

            payload = server.start_stream("ROS 控制器报警：；当前状态正常")

        form = observed["form"]
        self.assertIsInstance(form, dict)
        assert isinstance(form, dict)
        self.assertEqual(observed["path"], "/api/generate-stream/start")
        self.assertEqual(form["cpu_threads"], "4")
        self.assertNotEqual(form["cpu_threads"], "0")
        self.assertNotEqual(form["cpu_threads"], "1")
        self.assertEqual(form["enable_text_normalization"], "1")
        self.assertEqual(form["enable_normalize_tts_text"], "1")
        self.assertEqual(form["demo_id"], "demo-1")
        self.assertNotIn("voice", form)
        self.assertIn("ROS 控制器报警：当前状态正常", form["text"])
        self.assertNotIn("R O S", form["text"])
        self.assertEqual(payload["moss_audio_url"], "http://127.0.0.1:18083/audio")

    def test_stream_start_request_uses_builtin_voice_without_demo_id_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            server = MossStreamingServer(
                self.build_config(root, cpu_threads=3, use_prompt_audio=False),
                audio_dir=root / "audio",
            )
            server.base_url = "http://127.0.0.1:18083"
            observed: dict[str, object] = {}

            def fake_post(path: str, form: dict[str, str], *, timeout: float) -> dict[str, object]:
                observed["path"] = path
                observed["form"] = form
                return {
                    "stream_id": "stream-test",
                    "audio_url": "/audio",
                    "status_url": "/status",
                    "result_url": "/result",
                    "sample_rate": 48000,
                    "channels": 2,
                }

            server.ensure_running = lambda: None  # type: ignore[method-assign]
            server._post_form = fake_post  # type: ignore[method-assign]

            server.start_stream("当前状态正常")

        form = observed["form"]
        self.assertIsInstance(form, dict)
        assert isinstance(form, dict)
        self.assertEqual(observed["path"], "/api/generate-stream/start")
        self.assertEqual(form["voice"], "Xiaoyu")
        self.assertEqual(form["cpu_threads"], "3")
        self.assertNotIn("demo_id", form)

    def test_stream_start_rejects_non_positive_cpu_threads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            server = MossStreamingServer(self.build_config(root, cpu_threads=0), audio_dir=root / "audio")
            server.base_url = "http://127.0.0.1:18083"
            server.ensure_running = lambda: None  # type: ignore[method-assign]

            with self.assertRaisesRegex(ValueError, "cpu_threads must be greater than 0"):
                server.start_stream("当前状态正常")

    def test_health_wait_uses_configured_moss_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            server = MossStreamingServer(
                self.build_config(root, timeout_seconds=61.0),
                audio_dir=root / "audio",
            )
            server._process = FakeMossProcess()  # type: ignore[assignment]
            server._health_ok = lambda: False  # type: ignore[method-assign]

            with (
                patch("local_safety_assistant.web.service.time.monotonic", side_effect=[0.0, 29.0, 60.0, 62.0]),
                patch("local_safety_assistant.web.service.time.sleep") as sleep,
            ):
                with self.assertRaisesRegex(RuntimeError, "did not become ready within 61.0s"):
                    server._wait_until_healthy(["moss-tts-nano", "serve"])

        self.assertEqual(sleep.call_count, 2)

    def test_ensure_running_cleans_process_when_health_wait_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            server = MossStreamingServer(self.build_config(root), audio_dir=root / "audio")
            process = FakeMossProcess()

            def fail_health_wait(command: list[str]) -> None:
                raise RuntimeError("never ready")

            server._health_ok = lambda: False  # type: ignore[method-assign]
            server._wait_until_healthy = fail_health_wait  # type: ignore[method-assign]
            with (
                patch("local_safety_assistant.web.service._find_free_local_port", return_value=18084),
                patch("local_safety_assistant.web.service.subprocess.Popen", return_value=process),
            ):
                with self.assertRaisesRegex(RuntimeError, "never ready"):
                    server.ensure_running()

        self.assertEqual(process.terminate_calls, 1)
        self.assertEqual(process.kill_calls, 0)
        self.assertEqual(process.wait_timeouts, [5.0])
        self.assertIsNone(server._process)
        self.assertIsNone(server.base_url)
        self.assertIsNone(server.port)


class WebUiHTTPServerTest(unittest.TestCase):
    def test_parser_accepts_skip_warmup_option(self) -> None:
        args = build_parser().parse_args(["--skip-warmup"])

        self.assertTrue(args.skip_warmup)

    def test_parser_accepts_https_certificate_options(self) -> None:
        args = build_parser().parse_args(
            [
                "--ssl-certfile",
                "cert.pem",
                "--ssl-keyfile",
                "key.pem",
            ]
        )

        self.assertEqual(args.ssl_certfile, Path("cert.pem"))
        self.assertEqual(args.ssl_keyfile, Path("key.pem"))

    def test_voice_routes_reject_oversized_upload_before_service(self) -> None:
        for path in ("/api/voice", "/api/voice-stream"):
            with self.subTest(path=path):
                service = FailingChatService()
                server = WebUiHTTPServer(("127.0.0.1", 0), service)  # type: ignore[arg-type]
                try:
                    threading.Thread(target=server.serve_forever, daemon=True).start()
                    host, port = server.server_address
                    login = HTTPConnection(host, port)
                    login.request(
                        "POST",
                        "/api/login",
                        body='{"username":"admin","password":"12345"}',
                        headers={"Content-Type": "application/json"},
                    )
                    login_response = login.getresponse()
                    cookie = login_response.getheader("Set-Cookie")
                    login_response.read()

                    voice = HTTPConnection(host, port)
                    voice.putrequest("POST", path)
                    voice.putheader("Content-Type", "audio/webm")
                    voice.putheader("Cookie", cookie or "")
                    voice.putheader("Content-Length", str(WEB_VOICE_UPLOAD_MAX_BYTES + 1))
                    voice.endheaders()
                    response = voice.getresponse()
                    body = response.read().decode("utf-8")
                finally:
                    server.shutdown()
                    server.server_close()

                self.assertEqual(response.status, 400)
                self.assertIn("Voice upload too large", body)
                self.assertNotIn("missing asr", body)

    def test_chat_runtime_error_returns_service_unavailable_json(self) -> None:
        service = FailingChatService()
        server = WebUiHTTPServer(("127.0.0.1", 0), service)  # type: ignore[arg-type]
        try:
            __import__("threading").Thread(target=server.serve_forever, daemon=True).start()
            host, port = server.server_address
            login = HTTPConnection(host, port)
            login.request(
                "POST",
                "/api/login",
                body='{"username":"admin","password":"12345"}',
                headers={"Content-Type": "application/json"},
            )
            login_response = login.getresponse()
            cookie = login_response.getheader("Set-Cookie")
            login_response.read()

            chat = HTTPConnection(host, port)
            chat.request(
                "POST",
                "/api/chat",
                body='{"text":"hello"}'.encode("utf-8"),
                headers={"Content-Type": "application/json", "Cookie": cookie or ""},
            )
            response = chat.getresponse()
            body = response.read().decode("utf-8")
        finally:
            server.shutdown()
            server.server_close()

        self.assertEqual(response.status, 503)
        self.assertIn("missing tts", body)

    def test_turn_cancel_route_returns_json(self) -> None:
        service = FailingChatService()
        server = WebUiHTTPServer(("127.0.0.1", 0), service)  # type: ignore[arg-type]
        try:
            threading.Thread(target=server.serve_forever, daemon=True).start()
            host, port = server.server_address
            login = HTTPConnection(host, port)
            login.request(
                "POST",
                "/api/login",
                body='{"username":"admin","password":"12345"}',
                headers={"Content-Type": "application/json"},
            )
            login_response = login.getresponse()
            cookie = login_response.getheader("Set-Cookie")
            login_response.read()

            cancel = HTTPConnection(host, port)
            cancel.request(
                "POST",
                "/api/turn/cancel",
                body='{"reason":"test"}'.encode("utf-8"),
                headers={"Content-Type": "application/json", "Cookie": cookie or ""},
            )
            response = cancel.getresponse()
            body = response.read().decode("utf-8")
        finally:
            server.shutdown()
            server.server_close()

        self.assertEqual(response.status, 200)
        payload = json.loads(body)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["status"], "canceled")

    def test_external_estop_route_and_alert_audio_route(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = WebUiConfig(runtime_dir=Path(temp_dir), ros2_dry_run=True)
            pipeline = VoicePipeline(asr=FakeAsr(), llm=FakeLlm(), tts=FakeTts(config.audio_dir), rules_path=None)
            service = WebUiService(
                config=config,
                pipeline=pipeline,
                ros2_bridge=Ros2VoiceBridge(Ros2BridgeConfig(source="web_ui")),
                stack_config=VoiceStackConfig(),
                resolved_tts_engine="moss",
                available_tts_engines=("moss", "fake"),
                moss_streamer=FakeMossStreamer(),
            )
            event_body = json.dumps(
                {
                    "source": "min_distance_camera",
                    "active": True,
                    "latch": False,
                    "reason": "person distance below threshold: 0.42m",
                    "threshold_m": 0.5,
                    "distance_m": 0.42,
                },
                ensure_ascii=False,
            ).encode("utf-8")
            server = WebUiHTTPServer(("127.0.0.1", 0), service)
            try:
                threading.Thread(target=server.serve_forever, daemon=True).start()
                host, port = server.server_address

                unauth = HTTPConnection(host, port)
                unauth.request(
                    "POST",
                    "/api/estop/external",
                    body=event_body,
                    headers={"Content-Type": "application/json"},
                )
                unauth_response = unauth.getresponse()
                unauth_response.read()

                login = HTTPConnection(host, port)
                login.request(
                    "POST",
                    "/api/login",
                    body='{"username":"admin","password":"12345"}',
                    headers={"Content-Type": "application/json"},
                )
                login_response = login.getresponse()
                cookie = login_response.getheader("Set-Cookie")
                login_response.read()

                ingest = HTTPConnection(host, port)
                ingest.request(
                    "POST",
                    "/api/estop/external",
                    body=event_body,
                    headers={"Content-Type": "application/json", "Cookie": cookie or ""},
                )
                ingest_response = ingest.getresponse()
                ingest_body = json.loads(ingest_response.read().decode("utf-8"))

                invalid = HTTPConnection(host, port)
                invalid.request(
                    "POST",
                    "/api/estop/external",
                    body=b'{"source": "min_distance_camera"}',
                    headers={"Content-Type": "application/json", "Cookie": cookie or ""},
                )
                invalid_response = invalid.getresponse()
                invalid_response.read()

                status_request = HTTPConnection(host, port)
                status_request.request("GET", "/api/status", headers={"Cookie": cookie or ""})
                status_response = status_request.getresponse()
                status_payload = json.loads(status_response.read().decode("utf-8"))

                audio_request = HTTPConnection(host, port)
                audio_request.request("GET", "/emergency-alert-audio")
                audio_response = audio_request.getresponse()
                audio_body = audio_response.read()
            finally:
                server.shutdown()
                server.server_close()

        self.assertEqual(unauth_response.status, 401)
        self.assertEqual(ingest_response.status, 200)
        self.assertTrue(ingest_body["ok"])
        self.assertEqual(ingest_body["event"]["source"], "min_distance_camera")
        self.assertEqual(invalid_response.status, 400)
        self.assertEqual(status_payload["emergency_event"]["source"], "min_distance_camera")
        self.assertTrue(status_payload["emergency_event"]["active"])
        self.assertEqual(status_payload["emergency_alert_audio_url"], "/emergency-alert-audio")
        self.assertEqual(audio_response.status, 200)
        self.assertEqual(audio_body, DEFAULT_EMERGENCY_ALERT_AUDIO_PATH.read_bytes())

    def test_image_route_requires_auth_and_serves_runtime_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = WebUiConfig(runtime_dir=Path(temp_dir), ros2_dry_run=True)
            pipeline = VoicePipeline(asr=FakeAsr(), llm=FakeLlm(), tts=FakeTts(config.audio_dir), rules_path=None)
            service = WebUiService(
                config=config,
                pipeline=pipeline,
                ros2_bridge=Ros2VoiceBridge(Ros2BridgeConfig(source="web_ui")),
                stack_config=VoiceStackConfig(),
                resolved_tts_engine="moss",
                available_tts_engines=("moss", "fake"),
                moss_streamer=FakeMossStreamer(),
            )
            image = config.image_dir / "snapshot.jpg"
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"jpeg")
            server = WebUiHTTPServer(("127.0.0.1", 0), service)
            try:
                threading.Thread(target=server.serve_forever, daemon=True).start()
                host, port = server.server_address

                unauth = HTTPConnection(host, port)
                unauth.request("GET", "/images/snapshot.jpg")
                unauth_response = unauth.getresponse()
                unauth_body = unauth_response.read().decode("utf-8")

                login = HTTPConnection(host, port)
                login.request(
                    "POST",
                    "/api/login",
                    body='{"username":"admin","password":"12345"}',
                    headers={"Content-Type": "application/json"},
                )
                login_response = login.getresponse()
                cookie = login_response.getheader("Set-Cookie")
                login_response.read()

                image_request = HTTPConnection(host, port)
                image_request.request("GET", "/images/snapshot.jpg", headers={"Cookie": cookie or ""})
                image_response = image_request.getresponse()
                image_body = image_response.read()
            finally:
                server.shutdown()
                server.server_close()

        self.assertEqual(unauth_response.status, 401)
        self.assertIn("请先登录", unauth_body)
        self.assertEqual(image_response.status, 200)
        self.assertEqual(image_response.getheader("Content-Type"), "image/jpeg")
        self.assertEqual(image_body, b"jpeg")

    def test_chat_stream_audio_route_returns_progressive_pcm_stream(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = WebUiConfig(runtime_dir=Path(temp_dir), ros2_dry_run=True)
            pipeline = VoicePipeline(asr=FakeAsr(), llm=FakeLlm(), tts=FakeTts(config.audio_dir), rules_path=None)
            streamer = FakeMossStreamer(chunks=(b"\x10\x00\x11\x00", b"\x12\x00\x13\x00"))
            service = WebUiService(
                config=config,
                pipeline=pipeline,
                ros2_bridge=Ros2VoiceBridge(Ros2BridgeConfig(source="web_ui")),
                stack_config=VoiceStackConfig(),
                resolved_tts_engine="moss",
                available_tts_engines=("moss", "fake"),
                moss_streamer=streamer,
            )
            server = WebUiHTTPServer(("127.0.0.1", 0), service)
            try:
                threading.Thread(target=server.serve_forever, daemon=True).start()
                host, port = server.server_address
                login = HTTPConnection(host, port)
                login.request(
                    "POST",
                    "/api/login",
                    body='{"username":"admin","password":"12345"}',
                    headers={"Content-Type": "application/json"},
                )
                login_response = login.getresponse()
                cookie = login_response.getheader("Set-Cookie")
                login_response.read()

                chat = HTTPConnection(host, port)
                chat.request(
                    "POST",
                    "/api/chat-stream",
                    body='{"text":"hello"}'.encode("utf-8"),
                    headers={"Content-Type": "application/json", "Cookie": cookie or ""},
                )
                start_response = chat.getresponse()
                start_payload = json.loads(start_response.read().decode("utf-8"))

                status_payload = start_payload
                for _ in range(50):
                    status = HTTPConnection(host, port)
                    status.request(
                        "GET",
                        f"/api/chat-stream/{start_payload['turn_id']}",
                        headers={"Cookie": cookie or ""},
                    )
                    status_response = status.getresponse()
                    status_payload = json.loads(status_response.read().decode("utf-8"))
                    if status_payload.get("done"):
                        break
                    time.sleep(0.01)

                audio = HTTPConnection(host, port)
                audio.request(
                    "GET",
                    start_payload["audio_url"],
                    headers={"Cookie": cookie or ""},
                )
                audio_response = audio.getresponse()
                audio_body = audio_response.read()
            finally:
                server.shutdown()
                server.server_close()

        self.assertEqual(start_response.status, 200)
        self.assertEqual(start_payload["response_text"], "已收到。")
        self.assertEqual(start_payload["audio_url"], f"/api/chat-stream/{start_payload['turn_id']}/audio")
        self.assertEqual(status_response.status, 200)
        self.assertEqual(status_payload["status"], "complete")
        self.assertNotIn("chunks", status_payload)
        self.assertEqual(audio_response.status, 200)
        self.assertEqual(audio_response.getheader("Content-Type"), "application/octet-stream")
        self.assertEqual(audio_response.getheader("X-Audio-Codec"), "pcm_s16le")
        self.assertEqual(audio_response.getheader("X-Audio-Sample-Rate"), "48000")
        self.assertEqual(audio_response.getheader("X-Audio-Channels"), "2")
        self.assertEqual(audio_response.getheader("X-Stream-Id"), "stream-test")
        self.assertEqual(audio_body, b"\x10\x00\x11\x00\x12\x00\x13\x00")


if __name__ == "__main__":
    unittest.main()
