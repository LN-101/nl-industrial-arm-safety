"""Runtime service layer for the mobile web control surface."""

from __future__ import annotations

import array as array_module
import json
import math
import mimetypes
import re
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Protocol
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from local_safety_assistant.confirmation import (
    ACTION_ESTOP_RELEASE,
    ACTION_GOAL_MOTION,
    ACTION_OBJECT_GRASP_EXECUTION,
    ACTION_OBJECT_MAPPING_UPDATE,
    ACTION_RULE_EDIT,
    ACTION_SPEED_CHANGE,
    DEFAULT_CONFIRMATION_TTL_SECONDS,
    PendingConfirmation,
    build_estop_release_confirmation,
    build_goal_motion_confirmation,
    interpret_spoken_confirmation,
)
from local_safety_assistant.arm_rules import (
    request_arm_deceleration,
    request_object_grasp,
    sync_personnel_distance_to_arm_rules,
)
from local_safety_assistant.config import PROJECT_ROOT
from local_safety_assistant.object_mapping import update_object_mapping
from local_safety_assistant.rules import apply_rule_patch, load_rule_document
from local_safety_assistant.stack.cli import build_pipeline, build_ros2_bridge_config
from local_safety_assistant.stack.config import (
    MeloTtsConfig,
    MossTtsConfig,
    PiperTtsConfig,
    VoiceStackConfig,
)
from local_safety_assistant.stack.devices import available_openvino_devices
from local_safety_assistant.stack.pipeline import (
    VoicePipeline,
    build_arm_deceleration_success_response,
    build_object_mapping_update_success_response,
    build_rule_edit_success_response,
    normalize_asr_text,
)
from local_safety_assistant.stack.ros2_bridge import (
    DEFAULT_CAMERA_ESTOP_SOURCE,
    ROS2_BOOL,
    ROS2_STRING,
    Ros2MessagePlan,
    Ros2VoiceBridge,
    build_voice_ros2_plan,
    sync_estop_plans_to_arm_rules,
)
from local_safety_assistant.stack.vision import copy_vision_artifact_to_dir

_LOCAL_UPSTREAM_OPENER = urllib_request.build_opener(urllib_request.ProxyHandler({}))

DEFAULT_WEB_TITLE = "机械臂 Web 控制台"
DEFAULT_WEB_PORT = 8787
DEFAULT_WEB_ADMIN_USERNAME = "admin"
DEFAULT_WEB_ADMIN_PASSWORD = "12345"
WEB_TTS_JOB_TTL_SECONDS = 30 * 60
WEB_VOICE_UPLOAD_MAX_BYTES = 8 * 1024 * 1024
DEFAULT_MOSS_STREAM_HOST = "127.0.0.1"
DEFAULT_MOSS_STREAM_SAMPLE_RATE = 48000
DEFAULT_MOSS_STREAM_CHANNELS = 2
MOSS_STREAM_HEALTH_POLL_SECONDS = 0.25
MOSS_STREAM_HEALTH_REQUEST_TIMEOUT_SECONDS = 0.5
MOSS_STREAM_CLOSE_REQUEST_TIMEOUT_SECONDS = 15.0
MOSS_STREAM_PROCESS_STOP_TIMEOUT_SECONDS = 5.0
MOSS_STREAM_PCM_WINDOW_SECONDS = 0.10
MOSS_STREAM_PCM_SILENCE_RMS_THRESHOLD = 0.008
MOSS_STREAM_PCM_MAX_SILENCE_SECONDS = 0.55
MOSS_STREAM_PCM_HEADROOM = 26000.0
MOSS_STREAM_PCM_FADE_IN_SECONDS = 0.08
MOSS_STREAM_PCM_DECLICK_SECONDS = 0.006
MOSS_STREAM_PCM_DECLICK_DELTA = 9000
MOSS_STREAM_PCM_GAIN_RELEASE_PER_SECOND = 0.75
MOSS_TTS_SINGLE_STREAM_MAX_CHARS = 320
MOSS_TTS_SEGMENT_MAX_CHARS = 260
MOSS_TTS_SEGMENT_HARD_CHARS = 220
MOSS_TTS_MINOR_PAUSE_SECONDS = 0.0
MOSS_TTS_MAJOR_PAUSE_SECONDS = 0.0
MOSS_TTS_FALLBACK_PAUSE_SECONDS = 0.0
MOSS_TTS_SEGMENT_START_TIMEOUT_SECONDS = 120.0
DEFAULT_EMERGENCY_ALERT_AUDIO_URL = "/emergency-alert-audio"
DEFAULT_EMERGENCY_ALERT_AUDIO_FILENAME = "emergency_alert.wav"
DEFAULT_EMERGENCY_ALERT_AUDIO_PATH = Path(__file__).resolve().parent / "assets" / DEFAULT_EMERGENCY_ALERT_AUDIO_FILENAME

_AUDIO_SUFFIX_BY_MIME = {
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mp4": ".mp4",
    "audio/mpeg": ".mp3",
    "audio/aac": ".aac",
}


@dataclass(frozen=True)
class MossTtsTextSegment:
    text: str
    pause_after_seconds: float = 0.0


@dataclass(frozen=True)
class WebUiConfig:
    host: str = "0.0.0.0"
    port: int = DEFAULT_WEB_PORT
    title: str = DEFAULT_WEB_TITLE
    admin_username: str = DEFAULT_WEB_ADMIN_USERNAME
    admin_password: str = DEFAULT_WEB_ADMIN_PASSWORD
    session_ttl_seconds: int = 12 * 60 * 60
    runtime_dir: Path = PROJECT_ROOT / ".runtime" / "web_ui"
    tts_engine: str = "auto"
    ros2_dry_run: bool = False
    direct_estop_topic: bool = False
    warm_start: bool = True
    emergency_alert_audio: Path | None = None
    moss_pcm_buffer_seconds: float = 0.48
    moss_cpu_affinity: str | None = MossTtsConfig().cpu_affinity

    @property
    def audio_dir(self) -> Path:
        return self.runtime_dir / "audio"

    @property
    def resolved_emergency_alert_audio(self) -> Path:
        if self.emergency_alert_audio is not None:
            return self.emergency_alert_audio
        return DEFAULT_EMERGENCY_ALERT_AUDIO_PATH

    @property
    def image_dir(self) -> Path:
        return self.runtime_dir / "images"


@dataclass(frozen=True)
class WebEmergencyEvent:
    """Normalized external emergency-stop event from /safety/estop/request."""

    event_id: int
    source: str
    active: bool
    latch: bool
    reason: str
    distance_m: float | None
    trigger_distance_m: float | None
    threshold_m: float | None
    release_distance_m: float | None
    received_at: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "source": self.source,
            "active": self.active,
            "latch": self.latch,
            "reason": self.reason,
            "distance_m": self.distance_m,
            "trigger_distance_m": self.trigger_distance_m,
            "threshold_m": self.threshold_m,
            "release_distance_m": self.release_distance_m,
            "received_at": self.received_at,
        }


def _optional_event_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def parse_external_estop_event(payload: Any) -> dict[str, Any]:
    """Validate the multi-source estop JSON contract used on /safety/estop/request."""

    if isinstance(payload, bytes):
        payload = payload.decode("utf-8", errors="replace")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ValueError("emergency event payload must be valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("emergency event payload must be a JSON object")
    source = payload.get("source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("emergency event requires a non-empty string source")
    active = payload.get("active")
    if not isinstance(active, bool):
        raise ValueError("emergency event requires a boolean active field")
    latch = payload.get("latch", active)
    if not isinstance(latch, bool):
        raise ValueError("emergency event latch must be a boolean")
    reason = payload.get("reason")
    return {
        "source": source.strip(),
        "active": active,
        "latch": latch,
        "reason": str(reason) if reason is not None else "",
        "distance_m": _optional_event_number(payload.get("distance_m")),
        "trigger_distance_m": _optional_event_number(payload.get("trigger_distance_m")),
        "threshold_m": _optional_event_number(payload.get("threshold_m")),
        "release_distance_m": _optional_event_number(payload.get("release_distance_m")),
    }


@dataclass(frozen=True)
class WebTurnResponse:
    mode: str
    input_text: str
    response_text: str
    audio_urls: tuple[str, ...]
    ros2_plan: tuple[dict[str, Any], ...]
    ros2_error: str | None
    total_seconds: float
    image_artifacts: tuple[dict[str, Any], ...] = ()
    confirmation: PendingConfirmation | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "input_text": self.input_text,
            "response_text": self.response_text,
            "metadata": dict(self.metadata),
            "audio_urls": list(self.audio_urls),
            "image_artifacts": list(self.image_artifacts),
            "image_urls": [artifact["url"] for artifact in self.image_artifacts],
            "ros2_plan": list(self.ros2_plan),
            "ros2_error": self.ros2_error,
            "total_seconds": self.total_seconds,
            "confirmation": self.confirmation.as_dict(now=time.monotonic()) if self.confirmation else None,
            "status_line": self.status_line(),
        }

    def status_line(self) -> str:
        if self.metadata.get("canceled"):
            return "已取消"
        if self.metadata.get("no_answer"):
            return "已忽略无效语音"
        if self.confirmation is not None:
            return "等待确认"
        if self.ros2_error:
            return f"已完成，但 ROS2 发送失败：{self.ros2_error}"
        return "已完成"


@dataclass(frozen=True)
class WebEstopResponse:
    ok: bool
    message: str
    ros2_plan: tuple[dict[str, Any], ...]
    ros2_error: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "message": self.message,
            "ros2_plan": list(self.ros2_plan),
            "ros2_error": self.ros2_error,
            "status_line": self.status_line(),
        }

    def status_line(self) -> str:
        if self.ros2_error:
            return f"急停已生成，但 ROS2 发送失败：{self.ros2_error}"
        return "急停已发送"


@dataclass(frozen=True)
class WebConfirmationExecution:
    message: str
    response_text: str
    ros2_plan: tuple[dict[str, Any], ...] = ()
    ros2_error: str | None = None


@dataclass(frozen=True)
class WebConfirmationResponse:
    ok: bool
    status: str
    message: str
    confirmation: PendingConfirmation | None = None
    response_text: str = ""
    ros2_plan: tuple[dict[str, Any], ...] = ()
    ros2_error: str | None = None
    playback_turn: "WebProgressiveTurnJob | None" = None

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "ok": self.ok,
            "status": self.status,
            "message": self.message,
            "response_text": self.response_text,
            "confirmation": self.confirmation.as_dict(now=time.monotonic()) if self.confirmation else None,
            "ros2_plan": list(self.ros2_plan),
            "ros2_error": self.ros2_error,
            "status_line": self.status_line(),
        }
        if self.status in {"missing", "failed"}:
            payload["error"] = self.message
        if self.playback_turn is None:
            payload.update(
                {
                    "turn_id": None,
                    "mode": "confirmation",
                    "input_text": "",
                    "metadata": {},
                    "audio_url": None,
                    "sample_rate": DEFAULT_MOSS_STREAM_SAMPLE_RATE,
                    "channels": DEFAULT_MOSS_STREAM_CHANNELS,
                    "done": True,
                    "text_seconds": 0.0,
                    "tts_segment_count": 0,
                    "streamed_tts_segment_count": 0,
                    "stream_status": {},
                    "playback_status": "complete",
                    "playback_error": None,
                    "created_at": None,
                    "updated_at": None,
                    "completed_at": None,
                }
            )
            return payload

        playback = self.playback_turn.as_dict()
        payload.update(
            {
                "turn_id": playback["turn_id"],
                "mode": playback["mode"],
                "input_text": playback["input_text"],
                "metadata": playback["metadata"],
                "audio_url": playback["audio_url"],
                "sample_rate": playback["sample_rate"],
                "channels": playback["channels"],
                "done": playback["done"],
                "text_seconds": playback["text_seconds"],
                "tts_segment_count": playback["tts_segment_count"],
                "streamed_tts_segment_count": playback["streamed_tts_segment_count"],
                "stream_status": playback["stream_status"],
                "playback_status": playback["status"],
                "playback_error": playback["error"],
                "created_at": playback["created_at"],
                "updated_at": playback["updated_at"],
                "completed_at": playback["completed_at"],
            }
        )
        return payload

    def status_line(self) -> str:
        if self.ros2_error:
            return f"{self.message}，但 ROS2 发送失败：{self.ros2_error}"
        return self.message


@dataclass(frozen=True)
class WebCancelResponse:
    ok: bool
    status: str
    message: str
    canceled_confirmations: bool
    canceled_turns: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "status": self.status,
            "message": self.message,
            "canceled_confirmations": self.canceled_confirmations,
            "canceled_turns": self.canceled_turns,
            "status_line": self.message,
        }


@dataclass
class _StoredPendingConfirmation:
    confirmation: PendingConfirmation
    execute: Callable[[], WebConfirmationExecution]


class PendingConfirmationStore:
    def __init__(self, ttl_seconds: float = DEFAULT_CONFIRMATION_TTL_SECONDS) -> None:
        self.ttl_seconds = ttl_seconds
        self._pending: dict[str, _StoredPendingConfirmation] = {}
        self._lock = threading.Lock()

    def create(
        self,
        confirmation: PendingConfirmation,
        execute: Callable[[], WebConfirmationExecution],
    ) -> PendingConfirmation:
        confirmation_id = secrets.token_urlsafe(12)
        expires_at = time.monotonic() + self.ttl_seconds
        stored_confirmation = confirmation.with_runtime_state(
            confirmation_id=confirmation_id,
            expires_at=expires_at,
        )
        with self._lock:
            # A later high-risk request supersedes stale dialogs or spoken prompts.
            self._pending.clear()
            self._pending[confirmation_id] = _StoredPendingConfirmation(stored_confirmation, execute)
        return stored_confirmation

    def current(self) -> PendingConfirmation | None:
        now = time.monotonic()
        with self._lock:
            self._discard_expired_locked(now)
            if not self._pending:
                return None
            return next(iter(self._pending.values())).confirmation

    def pop(self, confirmation_id: str) -> _StoredPendingConfirmation | None:
        now = time.monotonic()
        with self._lock:
            self._discard_expired_locked(now)
            return self._pending.pop(confirmation_id, None)

    def cancel(self, confirmation_id: str | None = None) -> bool:
        with self._lock:
            if confirmation_id is None:
                had_pending = bool(self._pending)
                self._pending.clear()
                return had_pending
            return self._pending.pop(confirmation_id, None) is not None

    def _discard_expired_locked(self, now: float) -> None:
        expired = [
            confirmation_id
            for confirmation_id, stored in self._pending.items()
            if stored.confirmation.expires_at is not None and stored.confirmation.expires_at <= now
        ]
        for confirmation_id in expired:
            self._pending.pop(confirmation_id, None)


@dataclass
class WebProgressiveTurnJob:
    turn_id: str
    mode: str
    input_text: str
    response_text: str
    ros2_plan: tuple[dict[str, Any], ...]
    ros2_error: str | None
    text_seconds: float
    tts_segments: tuple[MossTtsTextSegment, ...] = ()
    image_artifacts: tuple[dict[str, Any], ...] = ()
    confirmation: PendingConfirmation | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "text_ready"
    moss_stream_id: str | None = None
    moss_audio_url: str | None = None
    audio_url: str | None = None
    sample_rate: int = DEFAULT_MOSS_STREAM_SAMPLE_RATE
    channels: int = DEFAULT_MOSS_STREAM_CHANNELS
    stream_status: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    streamed_tts_segment_count: int = 0
    created_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)
    completed_at: float | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def attach_stream(
        self,
        *,
        start_payload: dict[str, Any],
        audio_url: str,
        segment_index: int = 0,
    ) -> None:
        with self._lock:
            self.status = "streaming"
            self.moss_stream_id = str(start_payload.get("stream_id") or "")
            self.moss_audio_url = str(start_payload.get("moss_audio_url") or start_payload.get("audio_url") or "")
            self.audio_url = audio_url
            self.sample_rate = int(start_payload.get("sample_rate") or DEFAULT_MOSS_STREAM_SAMPLE_RATE)
            self.channels = int(start_payload.get("channels") or DEFAULT_MOSS_STREAM_CHANNELS)
            self.stream_status = dict(start_payload)
            self.stream_status["segment_index"] = segment_index
            self.stream_status["segment_count"] = len(self.tts_segments) or 1
            self.updated_at = time.monotonic()

    def set_proxy_audio_url(self, audio_url: str) -> None:
        with self._lock:
            self.audio_url = audio_url
            if self.status == "text_ready":
                self.status = "starting_stream"
            self.updated_at = time.monotonic()

    def update_stream_status(self, snapshot: dict[str, Any]) -> None:
        with self._lock:
            self.stream_status = dict(snapshot)
            if snapshot.get("failed"):
                self.status = "failed"
                self.error = str(snapshot.get("error") or snapshot.get("run_status") or "MOSS stream failed")
                self.completed_at = time.monotonic()
            elif snapshot.get("ready"):
                self.status = "complete"
                self.completed_at = time.monotonic()
            else:
                self.status = "streaming"
            self.sample_rate = int(snapshot.get("sample_rate") or self.sample_rate)
            self.channels = int(snapshot.get("channels") or self.channels)
            self.updated_at = time.monotonic()

    def mark_segment_streamed(self) -> None:
        with self._lock:
            self.streamed_tts_segment_count = min(
                len(self.tts_segments) or 1,
                self.streamed_tts_segment_count + 1,
            )
            self.updated_at = time.monotonic()

    def mark_complete(self) -> None:
        with self._lock:
            self.status = "complete"
            self.completed_at = time.monotonic()
            self.updated_at = self.completed_at

    def mark_failed(self, error: str) -> None:
        with self._lock:
            self.status = "failed"
            self.error = error
            self.completed_at = time.monotonic()
            self.updated_at = self.completed_at

    def mark_canceled(self) -> None:
        with self._lock:
            if self.status in {"complete", "failed", "canceled"}:
                return
            self.status = "canceled"
            self.error = None
            self.completed_at = time.monotonic()
            self.updated_at = self.completed_at

    def is_canceled(self) -> bool:
        with self._lock:
            return self.status == "canceled"

    def is_done(self) -> bool:
        with self._lock:
            return self.status in {"complete", "failed", "canceled"}

    def upstream_audio_url(self) -> str | None:
        with self._lock:
            return self.moss_audio_url

    def wait_for_moss_audio_url(self, *, timeout_seconds: float) -> str | None:
        deadline = time.monotonic() + timeout_seconds
        while True:
            with self._lock:
                if self.moss_audio_url:
                    return self.moss_audio_url
                if self.status == "failed":
                    raise RuntimeError(self.error or "MOSS realtime stream failed")
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.05)

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            status = self.status
            error = self.error
            completed_at = self.completed_at
            updated_at = self.updated_at
            stream_status = dict(self.stream_status)
            audio_url = self.audio_url
            sample_rate = self.sample_rate
            channels = self.channels
            moss_stream_id = self.moss_stream_id
            tts_segment_count = len(self.tts_segments)
            streamed_tts_segment_count = self.streamed_tts_segment_count
        return {
            "turn_id": self.turn_id,
            "mode": self.mode,
            "input_text": self.input_text,
            "response_text": self.response_text,
            "metadata": dict(self.metadata),
            "image_artifacts": list(self.image_artifacts),
            "image_urls": [artifact["url"] for artifact in self.image_artifacts],
            "ros2_plan": list(self.ros2_plan),
            "ros2_error": self.ros2_error,
            "confirmation": self.confirmation.as_dict(now=time.monotonic()) if self.confirmation else None,
            "text_seconds": self.text_seconds,
            "status": status,
            "done": status in {"complete", "failed", "canceled"},
            "audio_url": audio_url,
            "sample_rate": sample_rate,
            "channels": channels,
            "moss_stream_id": moss_stream_id,
            "tts_segment_count": tts_segment_count,
            "streamed_tts_segment_count": streamed_tts_segment_count,
            "stream_status": stream_status,
            "error": error,
            "created_at": self.created_at,
            "updated_at": updated_at,
            "completed_at": completed_at,
            "status_line": self.status_line(status, error),
        }

    def status_line(self, status: str, error: str | None) -> str:
        if self.metadata.get("canceled"):
            return "已取消"
        if self.metadata.get("no_answer"):
            return "已忽略无效语音"
        if self.confirmation is not None:
            return "等待确认"
        if status == "canceled":
            return "已取消"
        if status == "failed":
            return f"语音合成失败：{error}" if error else "语音合成失败"
        if status == "complete":
            if self.ros2_error:
                return f"已完成，但 ROS2 发送失败：{self.ros2_error}"
            return "已完成"
        if status == "starting_stream":
            return "MOSS 实时语音流启动中..."
        if status == "streaming":
            return "MOSS 实时语音流生成中..."
        return "文字已生成，等待 MOSS 实时语音流..."


@dataclass(frozen=True)
class WebAudioStreamResponse:
    iterator: Iterator[bytes]
    sample_rate: int
    channels: int
    stream_id: str


@dataclass
class _MossPcmConditionerState:
    silence_bytes_emitted: int = 0
    audible_bytes: int = 0
    gain: float = 1.0
    last_samples: tuple[int, ...] | None = None


class MossStreamProvider(Protocol):
    def status(self) -> dict[str, Any]:
        ...

    def start_stream(self, text: str) -> dict[str, Any]:
        ...

    def stream_status(self, status_url: str) -> dict[str, Any]:
        ...

    def stream_audio(self, audio_url: str) -> Iterator[bytes]:
        ...

    def close_stream(self, audio_url: str) -> None:
        ...


def _condition_moss_pcm16_stream(
    chunks: Iterator[bytes],
    *,
    sample_rate: int,
    channels: int,
) -> Iterator[bytes]:
    # Closing this generator must close the source generator too; CPython does
    # not propagate close() through plain iteration, so do it explicitly.
    try:
        yield from _condition_moss_pcm16_stream_unmanaged(chunks, sample_rate=sample_rate, channels=channels)
    finally:
        close = getattr(chunks, "close", None)
        if callable(close):
            close()


def _condition_moss_pcm16_stream_unmanaged(
    chunks: Iterator[bytes],
    *,
    sample_rate: int,
    channels: int,
) -> Iterator[bytes]:
    if sample_rate <= 0 or channels <= 0:
        yield from chunks
        return
    frame_bytes = channels * 2
    window_bytes = _aligned_pcm_byte_count(
        int(sample_rate * channels * 2 * MOSS_STREAM_PCM_WINDOW_SECONDS),
        frame_bytes,
    )
    max_silence_bytes = _aligned_pcm_byte_count(
        int(sample_rate * channels * 2 * MOSS_STREAM_PCM_MAX_SILENCE_SECONDS),
        frame_bytes,
    )
    if window_bytes <= 0 or max_silence_bytes <= 0:
        yield from chunks
        return

    state = _MossPcmConditionerState()
    remainder = b""
    for raw_chunk in chunks:
        if not raw_chunk:
            continue
        buffer = remainder + raw_chunk
        usable_length = len(buffer) - (len(buffer) % frame_bytes)
        if usable_length <= 0:
            remainder = buffer
            continue
        remainder = buffer[usable_length:]
        offset = 0
        while offset < usable_length:
            end = min(offset + window_bytes, usable_length)
            conditioned = _condition_moss_pcm16_window(
                buffer[offset:end],
                state=state,
                max_silence_bytes=max_silence_bytes,
                sample_rate=sample_rate,
                channels=channels,
            )
            if conditioned:
                yield conditioned
            offset = end
    if remainder:
        yield remainder


def _condition_moss_pcm16_window(
    pcm: bytes,
    *,
    state: _MossPcmConditionerState,
    max_silence_bytes: int,
    sample_rate: int,
    channels: int,
) -> bytes:
    samples = array_module.array("h")
    samples.frombytes(pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return b""

    rms = _pcm16_rms(samples)
    if rms < MOSS_STREAM_PCM_SILENCE_RMS_THRESHOLD:
        if state.silence_bytes_emitted >= max_silence_bytes:
            return b""
        allowed_bytes = max_silence_bytes - state.silence_bytes_emitted
        emitted = pcm[:allowed_bytes]
        state.silence_bytes_emitted += len(emitted)
        emitted_sample_count = len(emitted) // 2
        if emitted_sample_count > 0:
            _remember_last_moss_pcm16_frame(samples[:emitted_sample_count], state=state, channels=channels)
        state.gain = 1.0
        return emitted

    state.silence_bytes_emitted = 0
    peak = max(abs(sample) for sample in samples)
    target_gain = 1.0
    if peak > MOSS_STREAM_PCM_HEADROOM:
        target_gain = MOSS_STREAM_PCM_HEADROOM / peak
    state.gain = _moss_pcm_smoothed_gain(
        current_gain=state.gain,
        target_gain=target_gain,
        sample_count=len(samples),
        sample_rate=sample_rate,
        channels=channels,
    )

    fade_bytes = _pcm_byte_count_for_seconds(
        MOSS_STREAM_PCM_FADE_IN_SECONDS,
        sample_rate=sample_rate,
        channels=channels,
        frame_bytes=channels * 2,
    )
    bytes_per_sample = 2
    for index, sample in enumerate(samples):
        fade = _moss_pcm_fade_multiplier(
            state.audible_bytes + (index * bytes_per_sample),
            fade_bytes=fade_bytes,
        )
        scaled = int(round(sample * state.gain * fade))
        if scaled > 32767:
            scaled = 32767
        elif scaled < -32768:
            scaled = -32768
        samples[index] = scaled
    _declick_moss_pcm16_window(samples, state=state, sample_rate=sample_rate, channels=channels)
    state.audible_bytes += len(samples) * bytes_per_sample
    _remember_last_moss_pcm16_frame(samples, state=state, channels=channels)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples.tobytes()


def _moss_pcm_smoothed_gain(
    *,
    current_gain: float,
    target_gain: float,
    sample_count: int,
    sample_rate: int,
    channels: int,
) -> float:
    target_gain = max(0.0, min(1.0, target_gain))
    current_gain = max(0.0, min(1.0, current_gain))
    if target_gain <= current_gain:
        return target_gain
    if sample_rate <= 0 or channels <= 0:
        return target_gain
    window_seconds = sample_count / float(sample_rate * channels)
    release = MOSS_STREAM_PCM_GAIN_RELEASE_PER_SECOND * window_seconds
    return min(target_gain, current_gain + release)


def _declick_moss_pcm16_window(
    samples: array_module.array[int],
    *,
    state: _MossPcmConditionerState,
    sample_rate: int,
    channels: int,
) -> None:
    previous = state.last_samples
    if previous is None or sample_rate <= 0 or channels <= 0 or len(samples) < channels:
        return
    channel_count = min(channels, len(previous), len(samples))
    if channel_count <= 0:
        return
    first_delta = max(abs(samples[channel] - previous[channel]) for channel in range(channel_count))
    if first_delta < MOSS_STREAM_PCM_DECLICK_DELTA:
        return
    frame_count = len(samples) // channels
    ramp_frames = min(frame_count, max(1, int(sample_rate * MOSS_STREAM_PCM_DECLICK_SECONDS)))
    for frame_index in range(ramp_frames):
        mix = float(frame_index + 1) / float(ramp_frames)
        for channel in range(channel_count):
            sample_index = frame_index * channels + channel
            start = previous[channel]
            target = samples[sample_index]
            samples[sample_index] = int(round(start + ((target - start) * mix)))


def _remember_last_moss_pcm16_frame(
    samples: array_module.array[int],
    *,
    state: _MossPcmConditionerState,
    channels: int,
) -> None:
    if channels <= 0 or len(samples) < channels:
        return
    start = len(samples) - channels
    state.last_samples = tuple(int(sample) for sample in samples[start : start + channels])


def _moss_pcm_fade_multiplier(offset_bytes: int, *, fade_bytes: int) -> float:
    if fade_bytes <= 0 or offset_bytes >= fade_bytes:
        return 1.0
    return max(0.0, min(1.0, float(offset_bytes) / float(fade_bytes)))


def _pcm_byte_count_for_seconds(
    seconds: float,
    *,
    sample_rate: int | None,
    channels: int | None,
    frame_bytes: int,
) -> int:
    if seconds <= 0:
        return 0
    if sample_rate is None or channels is None:
        return int(seconds * DEFAULT_MOSS_STREAM_SAMPLE_RATE * DEFAULT_MOSS_STREAM_CHANNELS * 2)
    return _aligned_pcm_byte_count(int(seconds * sample_rate * frame_bytes), frame_bytes)


def _iter_pcm16_silence(seconds: float, *, sample_rate: int, channels: int) -> Iterator[bytes]:
    frame_bytes = channels * 2
    total_bytes = _aligned_pcm_byte_count(int(seconds * sample_rate * frame_bytes), frame_bytes)
    if total_bytes <= 0:
        return
    chunk = b"\x00" * min(total_bytes, 32 * 1024)
    remaining = total_bytes
    while remaining > 0:
        size = min(remaining, len(chunk))
        yield chunk[:size]
        remaining -= size


def _split_moss_tts_text_segments(text: str) -> tuple[MossTtsTextSegment, ...]:
    normalized = _prepare_moss_stream_tts_text(text)
    if not normalized:
        return ()

    if _moss_tts_text_len(normalized) <= MOSS_TTS_SINGLE_STREAM_MAX_CHARS:
        return (MossTtsTextSegment(normalized),)

    text_parts: list[str] = []
    for major_part in _split_text_after_delimiters(normalized, "。！？；;"):
        if _moss_tts_text_len(major_part) <= MOSS_TTS_SEGMENT_MAX_CHARS:
            text_parts.append(major_part)
            continue
        text_parts.extend(_split_long_moss_tts_part(major_part))

    compact_parts = _pack_moss_tts_text_parts(text_parts)
    if not compact_parts:
        return (MossTtsTextSegment(normalized),)

    segments: list[MossTtsTextSegment] = []
    for index, part in enumerate(compact_parts):
        pause = 0.0 if index == len(compact_parts) - 1 else _pause_after_moss_segment(part)
        segments.append(MossTtsTextSegment(part, pause_after_seconds=pause))
    return tuple(segments)


def _prepare_moss_stream_tts_text(text: str) -> str:
    """Keep Web MOSS streaming input close to the official ONNX demo path."""

    prepared = " ".join(text.strip().split())
    if not prepared:
        return prepared
    return _normalize_moss_stream_punctuation(prepared)


def _normalize_moss_stream_punctuation(text: str) -> str:
    normalized = text.strip()
    normalized = re.sub(r"([:：])\s*[;；]+", r"\1", normalized)
    normalized = re.sub(r"([。！？])\s*[;；]+", r"\1", normalized)
    normalized = re.sub(r"[;；]{2,}", "；", normalized)
    normalized = re.sub(r"，{2,}", "，", normalized)
    normalized = re.sub(r"、{2,}", "、", normalized)
    return normalized.strip("，、；; ")


def _pack_moss_tts_text_parts(parts: list[str]) -> tuple[str, ...]:
    packed: list[str] = []
    current = ""
    for part in parts:
        normalized_part = part.strip()
        if not normalized_part:
            continue
        if not current:
            current = normalized_part
            continue
        if _moss_tts_text_len(current + normalized_part) <= MOSS_TTS_SEGMENT_MAX_CHARS:
            current += normalized_part
            continue
        packed.append(current)
        current = normalized_part
    if current:
        packed.append(current)
    return tuple(packed)


def _split_long_moss_tts_part(text: str) -> list[str]:
    pieces = _split_text_after_delimiters(text, "，、,")
    segments: list[str] = []
    current = ""
    for piece in pieces:
        if not piece:
            continue
        if not current:
            current = piece
        elif _moss_tts_text_len(current + piece) <= MOSS_TTS_SEGMENT_HARD_CHARS:
            current += piece
        else:
            segments.extend(_split_overlong_moss_tts_part(current))
            current = piece
        if _moss_tts_text_len(current) >= MOSS_TTS_SEGMENT_HARD_CHARS:
            segments.extend(_split_overlong_moss_tts_part(current))
            current = ""
    if current:
        segments.extend(_split_overlong_moss_tts_part(current))
    return segments


def _split_overlong_moss_tts_part(text: str) -> list[str]:
    if _moss_tts_text_len(text) <= MOSS_TTS_SEGMENT_MAX_CHARS:
        return [text]
    parts: list[str] = []
    current = ""
    current_len = 0
    for char in text:
        char_len = 0 if char.isspace() else 1
        if current and current_len + char_len > MOSS_TTS_SEGMENT_HARD_CHARS:
            parts.append(current)
            current = char
            current_len = char_len
        else:
            current += char
            current_len += char_len
    if current:
        parts.append(current)
    return parts


def _split_text_after_delimiters(text: str, delimiters: str) -> list[str]:
    pattern = f"([{re.escape(delimiters)}])"
    pieces = re.split(pattern, text)
    result: list[str] = []
    current = ""
    for piece in pieces:
        if not piece:
            continue
        current += piece
        if piece in delimiters:
            result.append(current)
            current = ""
    if current:
        result.append(current)
    return result


def _pause_after_moss_segment(text: str) -> float:
    stripped = text.rstrip()
    if not stripped:
        return MOSS_TTS_FALLBACK_PAUSE_SECONDS
    if stripped[-1] in "。！？；;":
        return MOSS_TTS_MAJOR_PAUSE_SECONDS
    if stripped[-1] in "，、,":
        return MOSS_TTS_MINOR_PAUSE_SECONDS
    return MOSS_TTS_FALLBACK_PAUSE_SECONDS


def _moss_tts_text_len(text: str) -> int:
    return sum(0 if char.isspace() else 1 for char in text)


def _aligned_pcm_byte_count(byte_count: int, frame_bytes: int) -> int:
    if byte_count <= 0 or frame_bytes <= 0:
        return 0
    return byte_count - (byte_count % frame_bytes)


def _pcm16_rms(samples: array_module.array[int]) -> float:
    total = 0.0
    for sample in samples:
        total += float(sample) * float(sample)
    return math.sqrt(total / len(samples)) / 32768.0


class MossStreamingServer:
    """Manage a local MOSS ONNX streaming server and proxy its PCM stream."""

    def __init__(self, config: MossTtsConfig, *, audio_dir: Path) -> None:
        self.config = config
        self.audio_dir = audio_dir
        self.host = "127.0.0.1"
        self.port: int | None = None
        self.base_url: str | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        self._voice: str | None = None
        self._demo_id: str | None = None
        if config.prompt_audio is None:
            voice = str(config.voice or "").strip()
            if not voice:
                raise ValueError("MOSS voice must not be empty when prompt_audio is not configured.")
            self._voice = voice
        else:
            self._demo_id = _resolve_moss_demo_id(config.source_dir, config.prompt_audio)

    def status(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "base_url": self.base_url,
            "running": self._process is not None and self._process.poll() is None,
            "demo_id": self._demo_id,
            "voice": self._voice,
        }
        if payload["running"] and self.base_url is not None:
            try:
                payload["upstream_health"] = self._get_json(
                    f"{self.base_url}/health",
                    timeout=MOSS_STREAM_HEALTH_REQUEST_TIMEOUT_SECONDS,
                )
            except Exception as exc:
                payload["upstream_health_error"] = str(exc)
        return payload

    def start_stream(self, text: str) -> dict[str, Any]:
        self.ensure_running()
        prepared_text = _prepare_moss_stream_tts_text(text)
        form = {
            "text": prepared_text,
            "max_new_frames": str(self.config.max_new_frames),
            "voice_clone_max_text_tokens": str(self.config.voice_clone_max_text_tokens),
            "tts_max_batch_size": "0",
            "codec_max_batch_size": "0",
            "cpu_threads": str(self._configured_cpu_threads()),
            "attn_implementation": self.config.sample_mode,
            "do_sample": "0" if self.config.sample_mode == "greedy" else "1",
            "text_temperature": str(self.config.text_temperature),
            "text_top_p": str(self.config.text_top_p),
            "text_top_k": str(self.config.text_top_k),
            "audio_temperature": str(self.config.audio_temperature),
            "audio_top_p": str(self.config.audio_top_p),
            "audio_top_k": str(self.config.audio_top_k),
            "audio_repetition_penalty": str(self.config.audio_repetition_penalty),
            "enable_text_normalization": "1",
            "enable_normalize_tts_text": "1",
            "seed": "0",
        }
        if self._demo_id is not None:
            form["demo_id"] = self._demo_id
        elif self._voice is not None:
            form["voice"] = self._voice
        payload = self._post_form("/api/generate-stream/start", form, timeout=90.0)
        for key in ("audio_url", "status_url", "result_url"):
            if key in payload:
                payload[key] = self._absolute_url(str(payload[key]))
        payload["moss_audio_url"] = payload.get("audio_url")
        return payload

    def stream_status(self, status_url: str) -> dict[str, Any]:
        return self._get_json(status_url, timeout=5.0)

    def stream_audio(self, audio_url: str) -> Iterator[bytes]:
        with _LOCAL_UPSTREAM_OPENER.open(audio_url, timeout=120.0) as response:
            while True:
                chunk = response.read(32 * 1024)
                if not chunk:
                    break
                yield chunk

    def close_stream(self, audio_url: str) -> None:
        base_url, separator, suffix = audio_url.rpartition("/audio")
        if not separator or suffix:
            return
        request = urllib_request.Request(f"{base_url}/close", data=b"", method="POST")
        with _LOCAL_UPSTREAM_OPENER.open(
            request,
            timeout=MOSS_STREAM_CLOSE_REQUEST_TIMEOUT_SECONDS,
        ) as response:
            response.read()

    def ensure_running(self) -> None:
        with self._lock:
            if self._health_ok():
                return
            if self._process is not None and self._process.poll() is None:
                self._stop_process(self._process)
                self._process = None
            self.port = _find_free_local_port()
            self.base_url = f"http://{self.host}:{self.port}"
            self.audio_dir.mkdir(parents=True, exist_ok=True)
            command = self._build_start_command()
            process = subprocess.Popen(
                command,
                cwd=self.config.source_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._process = process
            try:
                self._wait_until_healthy(command)
            except Exception:
                self._stop_process(process)
                if self._process is process:
                    self._process = None
                self.base_url = None
                self.port = None
                raise

    def warm_start(self) -> dict[str, Any]:
        started = time.perf_counter()
        self.ensure_running()
        upstream = self._wait_until_upstream_warm()
        return {
            "component": "moss_tts",
            "status": "ready",
            "base_url": self.base_url,
            "demo_id": self._demo_id,
            "voice": self._voice,
            "cpu_threads": self._configured_cpu_threads(),
            "cpu_affinity": self.config.cpu_affinity,
            "upstream": upstream,
            "elapsed_seconds": time.perf_counter() - started,
        }

    def shutdown(self) -> None:
        with self._lock:
            process = self._process
            self._process = None
        if process is None or process.poll() is not None:
            return
        self._stop_process(process)

    def _configured_cpu_threads(self) -> int:
        threads = int(self.config.cpu_threads)
        if threads <= 0:
            raise ValueError("MOSS cpu_threads must be greater than 0.")
        return threads

    def _configured_startup_timeout_seconds(self) -> float:
        timeout_seconds = float(self.config.timeout_seconds)
        if timeout_seconds <= 0:
            raise ValueError("MOSS timeout_seconds must be greater than 0.")
        return timeout_seconds

    def _build_start_command(self) -> list[str]:
        if self.port is None:
            raise RuntimeError("MOSS streaming server port has not been allocated")
        command = [
            str(self.config.executable),
            "serve",
            "--backend",
            "onnx",
            "--onnx-model-dir",
            str(self.config.model_dir),
            "--output-dir",
            str(self.audio_dir),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--cpu-threads",
            str(self._configured_cpu_threads()),
            "--execution-provider",
            self.config.execution_provider,
            "--max-new-frames",
            str(self.config.max_new_frames),
        ]
        if self.config.cpu_affinity:
            command[:0] = ["taskset", "--cpu-list", self.config.cpu_affinity]
        return command

    def _wait_until_upstream_warm(self) -> dict[str, Any]:
        if self.base_url is None:
            raise RuntimeError("MOSS streaming server has not started")
        deadline = time.monotonic() + self._configured_startup_timeout_seconds()
        status_url = f"{self.base_url}/api/warmup-status"
        while time.monotonic() < deadline:
            snapshot = self._get_json(status_url, timeout=5.0)
            if snapshot.get("ready"):
                return snapshot
            if snapshot.get("failed"):
                message = str(snapshot.get("status_text") or snapshot.get("error") or "MOSS warmup failed")
                raise RuntimeError(message)
            time.sleep(0.25)
        raise RuntimeError("MOSS streaming server warmup did not complete before timeout")

    def _wait_until_healthy(self, command: list[str]) -> None:
        timeout_seconds = self._configured_startup_timeout_seconds()
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(f"MOSS streaming server exited early: {' '.join(command)}")
            if self._health_ok():
                return
            time.sleep(MOSS_STREAM_HEALTH_POLL_SECONDS)
        raise RuntimeError(
            f"MOSS streaming server did not become ready within {timeout_seconds:.1f}s: {' '.join(command)}"
        )

    def _health_ok(self) -> bool:
        if self.base_url is None:
            return False
        try:
            self._get_json(f"{self.base_url}/health", timeout=MOSS_STREAM_HEALTH_REQUEST_TIMEOUT_SECONDS)
        except Exception:
            return False
        return True

    def _stop_process(self, process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=MOSS_STREAM_PROCESS_STOP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=MOSS_STREAM_PROCESS_STOP_TIMEOUT_SECONDS)

    def _post_form(self, path: str, form: dict[str, str], *, timeout: float) -> dict[str, Any]:
        if self.base_url is None:
            raise RuntimeError("MOSS streaming server has not started")
        body = urllib_parse.urlencode(form).encode("utf-8")
        request = urllib_request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with _LOCAL_UPSTREAM_OPENER.open(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib_error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"MOSS stream start failed: HTTP {error.code} {detail}") from error

    def _get_json(self, url: str, *, timeout: float) -> dict[str, Any]:
        with _LOCAL_UPSTREAM_OPENER.open(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _absolute_url(self, value: str) -> str:
        if self.base_url is None:
            raise RuntimeError("MOSS streaming server has not started")
        if value.startswith("http://") or value.startswith("https://"):
            return value
        return f"{self.base_url}{value if value.startswith('/') else '/' + value}"


class SessionStore:
    def __init__(self, ttl_seconds: int) -> None:
        self.ttl_seconds = ttl_seconds
        self._sessions: dict[str, float] = {}
        self._lock = threading.Lock()

    def create(self) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._sessions[token] = time.monotonic() + self.ttl_seconds
        return token

    def validate(self, token: str | None) -> bool:
        if not token:
            return False
        with self._lock:
            expires_at = self._sessions.get(token)
            if expires_at is None:
                return False
            if expires_at < time.monotonic():
                self._sessions.pop(token, None)
                return False
            self._sessions[token] = time.monotonic() + self.ttl_seconds
            return True

    def revoke(self, token: str | None) -> None:
        if not token:
            return
        with self._lock:
            self._sessions.pop(token, None)


class WebUiService:
    def __init__(
        self,
        *,
        config: WebUiConfig,
        pipeline: VoicePipeline,
        ros2_bridge: Ros2VoiceBridge,
        stack_config: VoiceStackConfig,
        resolved_tts_engine: str,
        available_tts_engines: tuple[str, ...],
        moss_streamer: MossStreamProvider | None = None,
    ) -> None:
        self.config = config
        self.pipeline = pipeline
        self.ros2_bridge = ros2_bridge
        self.stack_config = stack_config
        self.resolved_tts_engine = resolved_tts_engine
        self.available_tts_engines = available_tts_engines
        self.moss_streamer = moss_streamer
        if self.moss_streamer is None and resolved_tts_engine == "moss":
            self.moss_streamer = MossStreamingServer(stack_config.moss_tts, audio_dir=config.audio_dir)
        self.sessions = SessionStore(config.session_ttl_seconds)
        self._turn_lock = threading.Lock()
        self._turn_cancel_epoch_lock = threading.Lock()
        self._turn_cancel_epoch = 0
        self._ros2_lock = threading.Lock()
        self._progressive_turns: dict[str, WebProgressiveTurnJob] = {}
        self._progressive_turns_lock = threading.Lock()
        self.confirmations = PendingConfirmationStore()
        self.pipeline.pending_confirmation_provider = self.confirmations.current
        self._last_error: str | None = None
        self._warmup_summary: dict[str, Any] | None = None
        self._emergency_lock = threading.Lock()
        self._emergency_event_seq = 0
        self._emergency_active_sources: dict[str, int] = {}
        self._emergency_event: WebEmergencyEvent | None = None
        self._estop_listener_thread: threading.Thread | None = None
        self.config.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.config.audio_dir.mkdir(parents=True, exist_ok=True)
        self.config.image_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def build_default(cls, config: WebUiConfig | None = None) -> "WebUiService":
        web_config = config or WebUiConfig()
        default_stack_config = VoiceStackConfig()
        stack_config = replace(
            default_stack_config,
            moss_tts=replace(default_stack_config.moss_tts, cpu_affinity=web_config.moss_cpu_affinity),
        )
        available_tts_engines = _available_tts_engines(stack_config)
        resolved_tts_engine = _resolve_tts_engine(web_config.tts_engine, available_tts_engines)
        args = _build_runtime_args(web_config, stack_config, resolved_tts_engine)
        include_tts = resolved_tts_engine != "none"
        pipeline = build_pipeline(args, include_asr=True, include_tts=include_tts)
        ros2_bridge = Ros2VoiceBridge(build_ros2_bridge_config(args))
        return cls(
            config=web_config,
            pipeline=pipeline,
            ros2_bridge=ros2_bridge,
            stack_config=stack_config,
            resolved_tts_engine=resolved_tts_engine,
            available_tts_engines=available_tts_engines,
        )

    def login(self, username: str, password: str) -> str | None:
        if not _constant_time_equal(username, self.config.admin_username):
            return None
        if not _constant_time_equal(password, self.config.admin_password):
            return None
        return self.sessions.create()

    def logout(self, token: str | None) -> None:
        self.sessions.revoke(token)

    def is_authenticated(self, token: str | None) -> bool:
        return self.sessions.validate(token)

    def status(self, token: str | None = None) -> dict[str, Any]:
        authenticated = self.is_authenticated(token)
        moss_tts = getattr(self.stack_config, "moss_tts", None)
        pending_confirmation = self.confirmations.current()
        alert_audio_path = self.serve_emergency_alert_audio_path()
        return {
            "title": self.config.title,
            "authenticated": authenticated,
            "login_required": True,
            "admin_username": self.config.admin_username,
            "resolved_tts_engine": self.resolved_tts_engine,
            "available_tts_engines": list(self.available_tts_engines),
            "requested_tts_engine": self.config.tts_engine,
            "moss_realtime_streaming_decode": getattr(moss_tts, "realtime_streaming_decode", None),
            "moss_streaming": self.moss_streamer.status() if self.moss_streamer is not None else None,
            "openvino_devices": list(available_openvino_devices()),
            "rules_path": str(self.stack_config.rules_path),
            "object_mapping_path": str(self.stack_config.object_mapping_path),
            "arm_rules_path": str(self.stack_config.arm_rules_path),
            "audio_dir": str(self.config.audio_dir),
            "image_dir": str(self.config.image_dir),
            "runtime_dir": str(self.config.runtime_dir),
            "vision_snapshot_service": self.stack_config.vision.snapshot_service,
            "ros2_dry_run": self.config.ros2_dry_run,
            "direct_estop_topic": self.config.direct_estop_topic,
            "warm_start": self.config.warm_start,
            "warmup": self._warmup_summary,
            "pending_confirmation": pending_confirmation.as_dict(now=time.monotonic())
            if pending_confirmation
            else None,
            "emergency_event": self.current_emergency_event(),
            "emergency_alert_audio_url": DEFAULT_EMERGENCY_ALERT_AUDIO_URL if alert_audio_path is not None else None,
            "emergency_alert_audio_available": alert_audio_path is not None,
            "emergency_alert_audio_path": str(self.config.resolved_emergency_alert_audio),
            "status_line": "已登录" if authenticated else "未登录",
            "last_error": self._last_error,
        }

    def warm_start(self) -> dict[str, Any]:
        started = time.perf_counter()
        steps: list[dict[str, Any]] = []
        steps.append(_warm_component("asr", getattr(self.pipeline, "asr", None)))
        steps.append(_warm_component("llm", getattr(self.pipeline, "llm", None)))
        if self.resolved_tts_engine == "moss":
            steps.append(_warm_component("moss_tts", self.moss_streamer))
        else:
            steps.append(
                {
                    "component": "moss_tts",
                    "status": "skipped",
                    "reason": f"resolved TTS engine is {self.resolved_tts_engine}",
                }
            )
        summary = {
            "ready": True,
            "elapsed_seconds": time.perf_counter() - started,
            "excluded_models": [self.stack_config.large_llm_model],
            "steps": steps,
        }
        self._warmup_summary = summary
        return summary

    def shutdown(self) -> None:
        shutdown = getattr(self.moss_streamer, "shutdown", None)
        if callable(shutdown):
            shutdown()

    def handle_chat(self, text: str) -> WebTurnResponse:
        if not text.strip():
            raise ValueError("text is required")
        turn_epoch = self._current_turn_cancel_epoch()
        turn_started = time.perf_counter()
        with self._turn_lock:
            if self._turn_was_interrupted(turn_epoch):
                return self._interrupted_turn_response("text", text, turn_started)
            result = self.pipeline.run_text_turn(text, synthesize=True)
            if self._turn_was_interrupted(turn_epoch):
                return self._interrupted_turn_response("text", text, turn_started)
            pending = self._create_confirmation_for_result(result)
            if pending is not None:
                ros2_plan: tuple[dict[str, Any], ...] = ()
                ros2_error = None
            else:
                ros2_plan, ros2_error, pending = self._publish_or_confirm_turn(result)
        return self._turn_response("text", result, ros2_plan, ros2_error, confirmation=pending)

    def handle_chat_progressive(self, text: str) -> dict[str, Any]:
        if not text.strip():
            raise ValueError("text is required")
        turn_epoch = self._current_turn_cancel_epoch()
        turn_started = time.perf_counter()
        with self._turn_lock:
            if self._turn_was_interrupted(turn_epoch):
                return self._interrupted_progressive_turn_dict("text", text, turn_started)
            result = self.pipeline.run_text_turn(text, synthesize=False)
            if self._turn_was_interrupted(turn_epoch):
                return self._interrupted_progressive_turn_dict("text", text, turn_started)
            pending = self._create_confirmation_for_result(result)
            if pending is not None:
                ros2_plan = ()
                ros2_error = None
            else:
                ros2_plan, ros2_error, pending = self._publish_or_confirm_turn(result)
            job = self._create_progressive_turn("text", result, ros2_plan, ros2_error, confirmation=pending)
        self._start_moss_stream(job)
        return job.as_dict()

    def handle_voice(self, audio_bytes: bytes, content_type: str | None = None) -> WebTurnResponse:
        _validate_voice_upload(audio_bytes)
        turn_epoch = self._current_turn_cancel_epoch()
        turn_started = time.perf_counter()
        with self._turn_lock:
            with tempfile.TemporaryDirectory(prefix="webui_upload_") as temp_dir:
                if self._turn_was_interrupted(turn_epoch):
                    return self._interrupted_turn_response("voice", "", turn_started)
                wav_path = _convert_uploaded_audio(Path(temp_dir), audio_bytes, content_type)
                if self._turn_was_interrupted(turn_epoch):
                    return self._interrupted_turn_response("voice", "", turn_started)
                pending_result = self._handle_spoken_confirmation_if_pending(wav_path, progressive=False)
                if pending_result is not None:
                    return pending_result
                result = self.pipeline.run_audio_file(wav_path, synthesize=True)
                if self._turn_was_interrupted(turn_epoch):
                    return self._interrupted_turn_response("voice", "", turn_started)
                pending = self._create_confirmation_for_result(result)
                if pending is not None:
                    ros2_plan = ()
                    ros2_error = None
                else:
                    ros2_plan, ros2_error, pending = self._publish_or_confirm_turn(result)
        return self._turn_response("voice", result, ros2_plan, ros2_error, confirmation=pending)

    def handle_voice_progressive(self, audio_bytes: bytes, content_type: str | None = None) -> dict[str, Any]:
        _validate_voice_upload(audio_bytes)
        turn_epoch = self._current_turn_cancel_epoch()
        turn_started = time.perf_counter()
        with self._turn_lock:
            with tempfile.TemporaryDirectory(prefix="webui_upload_") as temp_dir:
                if self._turn_was_interrupted(turn_epoch):
                    return self._interrupted_progressive_turn_dict("voice", "", turn_started)
                wav_path = _convert_uploaded_audio(Path(temp_dir), audio_bytes, content_type)
                if self._turn_was_interrupted(turn_epoch):
                    return self._interrupted_progressive_turn_dict("voice", "", turn_started)
                pending_result = self._handle_spoken_confirmation_if_pending(wav_path, progressive=True)
                if pending_result is not None:
                    return pending_result
                result = self.pipeline.run_audio_file(wav_path, synthesize=False)
                if self._turn_was_interrupted(turn_epoch):
                    return self._interrupted_progressive_turn_dict("voice", "", turn_started)
                pending = self._create_confirmation_for_result(result)
                if pending is not None:
                    ros2_plan = ()
                    ros2_error = None
                else:
                    ros2_plan, ros2_error, pending = self._publish_or_confirm_turn(result)
                job = self._create_progressive_turn("voice", result, ros2_plan, ros2_error, confirmation=pending)
        self._start_moss_stream(job)
        return job.as_dict()

    def handle_estop(self) -> WebEstopResponse:
        plan = self._build_estop_plan()
        arm_error = self._sync_estop_plans_to_arm_rules([plan])
        if self.config.ros2_dry_run:
            return WebEstopResponse(
                ok=arm_error is None,
                message="急停已发送" if arm_error is None else "急停已生成，但 JSON 写入失败",
                ros2_plan=(plan.as_dict(),),
                ros2_error=arm_error,
            )
        try:
            with self._ros2_lock:
                self.ros2_bridge.publish_plans([plan])
        except Exception as error:
            self._last_error = str(error)
            combined_error = _combine_runtime_errors(arm_error, str(error))
            return WebEstopResponse(
                ok=False,
                message="急停已生成，但发送失败",
                ros2_plan=(plan.as_dict(),),
                ros2_error=combined_error,
            )
        return WebEstopResponse(
            ok=arm_error is None,
            message="急停已发送" if arm_error is None else "急停已发送，但 JSON 写入失败",
            ros2_plan=(plan.as_dict(),),
            ros2_error=arm_error,
        )

    def serve_audio_path(self, filename: str) -> Path | None:
        path = (self.config.audio_dir / Path(filename).name).resolve()
        audio_root = self.config.audio_dir.resolve()
        if audio_root not in path.parents and path != audio_root:
            return None
        if not path.exists() or not path.is_file():
            return None
        return path

    def serve_image_path(self, filename: str) -> Path | None:
        path = (self.config.image_dir / Path(filename).name).resolve()
        image_root = self.config.image_dir.resolve()
        if image_root not in path.parents and path != image_root:
            return None
        if not path.exists() or not path.is_file():
            return None
        return path

    def progressive_turn_status(self, turn_id: str) -> dict[str, Any] | None:
        with self._progressive_turns_lock:
            job = self._progressive_turns.get(turn_id)
        if job is None:
            return None
        status_url = job.stream_status.get("status_url")
        if not job.is_done() and self.moss_streamer is not None and isinstance(status_url, str) and status_url:
            try:
                job.update_stream_status(self.moss_streamer.stream_status(status_url))
            except Exception as error:
                self._last_error = str(error)
                job.mark_failed(str(error))
        return job.as_dict()

    def progressive_turn_audio(self, turn_id: str) -> WebAudioStreamResponse | None:
        with self._progressive_turns_lock:
            job = self._progressive_turns.get(turn_id)
        if job is None or job.is_canceled() or not job.tts_segments or self.moss_streamer is None:
            return None
        payload = job.as_dict()
        stream_id = str(payload.get("moss_stream_id") or turn_id)
        sample_rate = int(payload.get("sample_rate") or DEFAULT_MOSS_STREAM_SAMPLE_RATE)
        channels = int(payload.get("channels") or DEFAULT_MOSS_STREAM_CHANNELS)
        raw_iterator = self._iter_moss_segment_audio(job, sample_rate=sample_rate, channels=channels)
        iterator = _condition_moss_pcm16_stream(raw_iterator, sample_rate=sample_rate, channels=channels)
        return WebAudioStreamResponse(
            iterator=iterator,
            sample_rate=sample_rate,
            channels=channels,
            stream_id=stream_id,
        )

    def cancel_active_web_turns(self) -> WebCancelResponse:
        self._mark_turns_interrupted()
        canceled_confirmations = self.confirmations.cancel(None)
        canceled_turns = self._cancel_progressive_turns()
        return WebCancelResponse(
            ok=True,
            status="canceled",
            message="已取消当前 Web 语音任务。",
            canceled_confirmations=canceled_confirmations,
            canceled_turns=canceled_turns,
        )

    def ingest_external_estop_event(self, payload: Any) -> dict[str, Any]:
        """Ingest one /safety/estop/request-style event from a non-Web stop source.

        A newly active external source first interrupts pending Web work through
        the same cancellation path used by manual cancel, then stores the event
        so /api/status pollers can render the alert. Repeated active messages
        for an already-active source keep the same event id so the browser does
        not stack dialogs or replay the alert audio.
        """

        fields = parse_external_estop_event(payload)
        if fields["source"] == self.ros2_bridge.config.source:
            # The Web UI's own emergency-stop button must not re-alert itself.
            return {"ok": True, "ignored": True, "reason": "web-ui own estop source", "event": None}
        with self._emergency_lock:
            is_new_active = fields["active"] and fields["source"] not in self._emergency_active_sources
        canceled_turns = 0
        if is_new_active:
            # Interrupt current Web work before the new event becomes visible.
            canceled_turns = self.cancel_active_web_turns().canceled_turns
        with self._emergency_lock:
            if fields["active"]:
                event_id = self._emergency_active_sources.get(fields["source"])
                if event_id is None:
                    self._emergency_event_seq += 1
                    event_id = self._emergency_event_seq
                    self._emergency_active_sources[fields["source"]] = event_id
            else:
                if fields["latch"]:
                    # Aggregator contract: a latched source stays active until it
                    # sends active=false,latch=false. Keep the mirror active so a
                    # later re-assert does not look like a brand-new stop event.
                    return {"ok": True, "ignored": True, "reason": "latched source remains active", "event": None}
                event_id = self._emergency_active_sources.pop(fields["source"], None)
                if event_id is None:
                    return {"ok": True, "ignored": True, "reason": "inactive source was not tracked", "event": None}
            event = WebEmergencyEvent(event_id=event_id, received_at=time.time(), **fields)
            self._emergency_event = event
        return {
            "ok": True,
            "ignored": False,
            "new_active": is_new_active,
            "canceled_turns": canceled_turns,
            "event": event.as_dict(),
        }

    def current_emergency_event(self) -> dict[str, Any] | None:
        with self._emergency_lock:
            return self._emergency_event.as_dict() if self._emergency_event is not None else None

    def serve_emergency_alert_audio_path(self) -> Path | None:
        path = self.config.resolved_emergency_alert_audio
        if not path.exists() or not path.is_file():
            return None
        return path

    def start_external_estop_listener(self, topic: str = "/safety/estop/request") -> None:
        """Subscribe to the multi-source estop bus and feed events into the alert state.

        ROS2 is an optional runtime dependency; rclpy is imported lazily and a
        missing environment raises RuntimeError instead of breaking the Web UI.
        """

        if self._estop_listener_thread is not None and self._estop_listener_thread.is_alive():
            return
        try:
            import rclpy  # noqa: F401
            from std_msgs.msg import String  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "External emergency-stop alert listener requires rclpy and std_msgs. "
                "Source the ROS2 environment, or run without the listener."
            ) from exc
        thread = threading.Thread(
            target=self._run_external_estop_listener,
            args=(topic,),
            name="web-ui-estop-alert-listener",
            daemon=True,
        )
        self._estop_listener_thread = thread
        thread.start()

    def _run_external_estop_listener(self, topic: str) -> None:
        import rclpy
        from std_msgs.msg import String

        if not rclpy.ok():
            rclpy.init(args=None)
        node = rclpy.create_node("web_ui_estop_alert_listener")

        def _on_message(message: Any) -> None:
            try:
                self.ingest_external_estop_event(str(message.data))
            except ValueError as error:
                self._last_error = f"忽略无效急停事件：{error}"

        node.create_subscription(String, topic, _on_message, 10)
        try:
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.2)
        finally:
            node.destroy_node()

    def _cancel_progressive_turns(self) -> int:
        canceled_turns = 0
        upstream_audio_urls: list[str] = []
        with self._progressive_turns_lock:
            jobs = list(self._progressive_turns.values())
        for job in jobs:
            if job.is_done():
                continue
            audio_url = job.upstream_audio_url()
            if isinstance(audio_url, str) and audio_url:
                upstream_audio_urls.append(audio_url)
            job.mark_canceled()
            canceled_turns += 1
        for audio_url in upstream_audio_urls:
            self._close_moss_stream_quietly(audio_url)
        return canceled_turns

    def _turn_response(
        self,
        mode: str,
        result: Any,
        ros2_plan: tuple[dict[str, Any], ...],
        ros2_error: str | None,
        *,
        confirmation: PendingConfirmation | None = None,
    ) -> WebTurnResponse:
        audio_urls = tuple(_audio_url_for_path(path) for path in getattr(result.tts, "audio_paths", ()) or ())
        image_artifacts = tuple(self._web_image_artifact(artifact) for artifact in getattr(result, "vision_artifacts", ()) or ())
        response = WebTurnResponse(
            mode=mode,
            input_text=result.input_text,
            response_text=result.response_text,
            metadata=dict(getattr(result, "metadata", {}) or {}),
            audio_urls=audio_urls,
            image_artifacts=image_artifacts,
            ros2_plan=ros2_plan,
            ros2_error=ros2_error,
            total_seconds=result.total_seconds,
            confirmation=confirmation,
        )
        if ros2_error:
            self._last_error = ros2_error
        return response

    def _current_turn_cancel_epoch(self) -> int:
        with self._turn_cancel_epoch_lock:
            return self._turn_cancel_epoch

    def _mark_turns_interrupted(self) -> None:
        with self._turn_cancel_epoch_lock:
            self._turn_cancel_epoch += 1

    def _turn_was_interrupted(self, turn_epoch: int) -> bool:
        with self._turn_cancel_epoch_lock:
            return turn_epoch != self._turn_cancel_epoch

    def _interrupted_turn_response(self, mode: str, input_text: str, started: float) -> WebTurnResponse:
        result = self._synthetic_turn_result(
            input_text=input_text,
            response_text="",
            total_seconds=time.perf_counter() - started,
            metadata={"canceled": True, "reason": "interrupted"},
        )
        return self._turn_response(mode, result, (), None)

    def _interrupted_progressive_turn_dict(self, mode: str, input_text: str, started: float) -> dict[str, Any]:
        now = time.monotonic()
        return {
            "turn_id": None,
            "mode": mode,
            "input_text": input_text,
            "response_text": "",
            "metadata": {"canceled": True, "reason": "interrupted"},
            "image_artifacts": [],
            "image_urls": [],
            "ros2_plan": [],
            "ros2_error": None,
            "confirmation": None,
            "text_seconds": time.perf_counter() - started,
            "status": "canceled",
            "done": True,
            "audio_url": None,
            "sample_rate": DEFAULT_MOSS_STREAM_SAMPLE_RATE,
            "channels": DEFAULT_MOSS_STREAM_CHANNELS,
            "moss_stream_id": None,
            "tts_segment_count": 0,
            "streamed_tts_segment_count": 0,
            "stream_status": {},
            "error": None,
            "created_at": now,
            "updated_at": now,
            "completed_at": now,
            "status_line": "已取消",
        }

    def _create_confirmation_for_result(self, result: Any) -> PendingConfirmation | None:
        confirmation = getattr(result, "pending_confirmation", None)
        if not isinstance(confirmation, PendingConfirmation):
            return None
        return self._store_pipeline_confirmation(confirmation)

    def _store_pipeline_confirmation(self, confirmation: PendingConfirmation) -> PendingConfirmation:
        if confirmation.action_type == ACTION_RULE_EDIT:
            return self.confirmations.create(confirmation, lambda: self._execute_rule_edit_confirmation(confirmation))
        if confirmation.action_type == ACTION_OBJECT_MAPPING_UPDATE:
            return self.confirmations.create(
                confirmation,
                lambda: self._execute_object_mapping_update_confirmation(confirmation),
            )
        if confirmation.action_type == ACTION_OBJECT_GRASP_EXECUTION:
            return self.confirmations.create(
                confirmation,
                lambda: self._execute_object_grasp_confirmation(confirmation),
            )
        if confirmation.action_type == ACTION_SPEED_CHANGE:
            return self.confirmations.create(
                confirmation,
                lambda: self._execute_speed_change_confirmation(confirmation),
            )
        return self.confirmations.create(
            confirmation,
            lambda: WebConfirmationExecution(
                message="确认已记录",
                response_text="确认已记录，但当前版本没有可执行动作。",
            ),
        )

    def _publish_or_confirm_turn(
        self,
        result: Any,
    ) -> tuple[tuple[dict[str, Any], ...], str | None, PendingConfirmation | None]:
        plans = build_voice_ros2_plan(result, self.ros2_bridge.config)
        confirmation = self._confirmation_for_ros2_plans(result, plans)
        if confirmation is not None:
            stored = self.confirmations.create(
                confirmation,
                lambda: self._execute_ros2_confirmation(plans),
            )
            return (), None, stored
        arm_error = self._sync_estop_plans_to_arm_rules(plans)
        ros2_plan, ros2_error = self._publish_plans(plans)
        return ros2_plan, _combine_runtime_errors(arm_error, ros2_error), None

    def _publish_turn(self, result: Any) -> tuple[tuple[dict[str, Any], ...], str | None]:
        plans = build_voice_ros2_plan(result, self.ros2_bridge.config)
        arm_error = self._sync_estop_plans_to_arm_rules(plans)
        ros2_plan, ros2_error = self._publish_plans(plans)
        return ros2_plan, _combine_runtime_errors(arm_error, ros2_error)

    def _publish_plans(self, plans: list[Ros2MessagePlan]) -> tuple[tuple[dict[str, Any], ...], str | None]:
        if self.config.ros2_dry_run:
            return tuple(plan.as_dict() for plan in plans), None
        try:
            with self._ros2_lock:
                published = self.ros2_bridge.publish_plans(plans) or plans
        except Exception as error:
            self._last_error = str(error)
            return tuple(plan.as_dict() for plan in plans), str(error)
        return tuple(plan.as_dict() for plan in published), None

    def _sync_estop_plans_to_arm_rules(self, plans: list[Ros2MessagePlan]) -> str | None:
        try:
            sync_estop_plans_to_arm_rules(
                plans,
                self.ros2_bridge.config,
                self.stack_config.arm_rules_path,
            )
        except (OSError, ValueError) as error:
            message = f"机械臂 JSON 急停状态未同步：{error}"
            self._last_error = message
            return message
        return None

    def _confirmation_for_ros2_plans(
        self,
        result: Any,
        plans: list[Ros2MessagePlan],
    ) -> PendingConfirmation | None:
        command_plans = [
            plan
            for plan in plans
            if plan.topic not in {self.ros2_bridge.config.transcript_topic, self.ros2_bridge.config.response_topic}
        ]
        for plan in command_plans:
            if plan.topic == self.ros2_bridge.config.estop_request_topic:
                try:
                    payload = json.loads(str(plan.payload))
                except (TypeError, ValueError):
                    payload = {}
                if payload.get("active") is False:
                    return build_estop_release_confirmation(str(getattr(result, "input_text", "") or plan.reason))
            if plan.topic == self.ros2_bridge.config.estop_bool_topic and plan.payload is False:
                return build_estop_release_confirmation(str(getattr(result, "input_text", "") or plan.reason))
            if plan.topic == self.ros2_bridge.config.goal_topic:
                goal = plan.payload if isinstance(plan.payload, dict) else {"payload": plan.payload}
                return build_goal_motion_confirmation(str(getattr(result, "input_text", "") or plan.reason), goal=goal)
        return None

    def _execute_rule_edit_confirmation(self, confirmation: PendingConfirmation) -> WebConfirmationExecution:
        details = confirmation.details
        rules_path = Path(str(details.get("rules_path") or ""))
        patch = details.get("patch")
        if not isinstance(patch, dict):
            raise ValueError("Pending rule edit is missing patch details.")
        previous = load_rule_document(rules_path)
        updated = apply_rule_patch(rules_path, patch)
        response_text = build_rule_edit_success_response(previous, updated, patch)
        sync_message = self._sync_personnel_distance_to_arm_rules(previous, updated)
        if sync_message:
            response_text = f"{response_text}{sync_message}"
        return WebConfirmationExecution(message="已确认并执行规则修改", response_text=response_text)

    def _execute_object_mapping_update_confirmation(self, confirmation: PendingConfirmation) -> WebConfirmationExecution:
        details = confirmation.details
        mapping_path = Path(str(details.get("object_mapping_path") or ""))
        marker = str(details.get("marker") or "")
        object_name = str(details.get("object_name") or "")
        previous_object = details.get("previous_object")
        update_object_mapping(mapping_path, marker=marker, object_name=object_name)
        response_text = build_object_mapping_update_success_response(
            marker,
            object_name,
            previous_object=str(previous_object) if previous_object is not None else None,
        )
        return WebConfirmationExecution(message="已确认并更新物体映射", response_text=response_text)

    def _execute_object_grasp_confirmation(self, confirmation: PendingConfirmation) -> WebConfirmationExecution:
        details = confirmation.details
        marker = str(details.get("marker") or "").strip()
        object_name = str(details.get("object_name") or "").strip()
        if not marker:
            raise ValueError("Pending object grasp is missing marker details.")
        request_object_grasp(
            self.stack_config.arm_rules_path,
            marker=marker,
            object_name=object_name,
            original_text=confirmation.original_text,
        )
        if marker and object_name:
            target = f"标号{marker}，对应{object_name}"
        else:
            target = object_name or (f"标号{marker}" if marker else "目标物体")
        response_text = f"抓取已确认：{target}。已写入机械臂 JSON 执行请求。"
        return WebConfirmationExecution(message="抓取确认已写入执行请求", response_text=response_text)

    def _execute_speed_change_confirmation(self, confirmation: PendingConfirmation) -> WebConfirmationExecution:
        details = confirmation.details
        raw_target_percent = details.get("target_speed_percent")
        if isinstance(raw_target_percent, bool):
            raise ValueError("Pending speed change is missing numeric target speed details.")
        try:
            target_percent = float(raw_target_percent)
        except (TypeError, ValueError) as exc:
            raise ValueError("Pending speed change is missing numeric target speed details.") from exc
        arm_rules_path = Path(str(details.get("arm_rules_path") or self.stack_config.arm_rules_path))
        result = request_arm_deceleration(
            arm_rules_path,
            target_speed_percent=target_percent,
        )
        response_text = build_arm_deceleration_success_response(
            result.target_speed_percent,
        )
        return WebConfirmationExecution(message="速度调整已写入执行请求", response_text=response_text)

    def _sync_personnel_distance_to_arm_rules(
        self,
        previous_document: dict[str, Any],
        updated_document: dict[str, Any],
    ) -> str:
        try:
            result = sync_personnel_distance_to_arm_rules(
                previous_document,
                updated_document,
                self.stack_config.arm_rules_path,
            )
        except (OSError, ValueError) as error:
            self._last_error = f"机械臂运行时安全距离未同步：{error}"
            return f" 但机械臂运行时安全距离未同步：{error}"
        if not result.synced or result.distance_m is None:
            return ""
        return f" 机械臂运行时安全距离已同步为 {result.distance_m:g} 米。"

    def _execute_ros2_confirmation(self, plans: list[Ros2MessagePlan]) -> WebConfirmationExecution:
        arm_error = self._sync_estop_plans_to_arm_rules(plans)
        ros2_plan, ros2_error = self._publish_plans(plans)
        response_text = "已确认并执行高风险操作。"
        for plan in plans:
            if plan.topic != self.ros2_bridge.config.estop_request_topic:
                continue
            try:
                payload = json.loads(str(plan.payload))
            except (TypeError, ValueError):
                continue
            if not isinstance(payload, dict):
                continue
            reset_sources = payload.get("reset_sources")
            if (
                payload.get("active") is False
                and isinstance(reset_sources, list)
                and DEFAULT_CAMERA_ESTOP_SOURCE in reset_sources
            ):
                response_text = (
                    "已确认并发送解除急停请求。相机仅在连续三帧距离安全，"
                    "或有效画面连续五秒无人后解除；否则继续保持急停。"
                )
                break
        return WebConfirmationExecution(
            message="已确认并执行",
            response_text=response_text,
            ros2_plan=ros2_plan,
            ros2_error=_combine_runtime_errors(arm_error, ros2_error),
        )

    def confirm_pending(self, confirmation_id: str, *, play_response: bool = True) -> WebConfirmationResponse:
        self._mark_turns_interrupted()
        self._cancel_progressive_turns()
        stored = self.confirmations.pop(confirmation_id)
        if stored is None:
            response = WebConfirmationResponse(
                ok=False,
                status="missing",
                message="确认请求已过期或已被新的指令替代，未执行任何操作。",
                response_text="确认请求已过期或已被新的指令替代，未执行任何操作。",
            )
            return response
        try:
            execution = stored.execute()
        except Exception as error:
            self._last_error = str(error)
            response = WebConfirmationResponse(
                ok=False,
                status="failed",
                message=f"确认执行失败：{error}",
                response_text=f"确认执行失败：{error}",
            )
            return response
        response = WebConfirmationResponse(
            ok=execution.ros2_error is None,
            status="confirmed",
            message=execution.message,
            confirmation=stored.confirmation,
            response_text=execution.response_text,
            ros2_plan=execution.ros2_plan,
            ros2_error=execution.ros2_error,
        )
        return self._attach_confirmation_playback(
            response,
            input_text=stored.confirmation.original_text,
        ) if play_response else response

    def cancel_pending(self, confirmation_id: str | None = None) -> WebConfirmationResponse:
        self._mark_turns_interrupted()
        self._cancel_progressive_turns()
        canceled = self.confirmations.cancel(confirmation_id)
        response_text = (
            "已取消，未执行任何操作。"
            if canceled
            else "取消请求已处理；该确认已不在待处理状态，本次没有执行任何新操作。"
        )
        return self._attach_confirmation_playback(
            WebConfirmationResponse(
                ok=True,
                status="canceled",
                message=response_text,
                response_text=response_text,
            ),
            input_text="取消确认",
        )

    def _attach_confirmation_playback(
        self,
        response: WebConfirmationResponse,
        *,
        input_text: str,
    ) -> WebConfirmationResponse:
        response_text = response.response_text or response.message
        if not response_text:
            return response
        result = self._synthetic_turn_result(
            input_text=input_text,
            response_text=response_text,
            total_seconds=0.0,
            metadata={
                "confirmation_resolved": True,
                "confirmation_status": response.status,
            },
        )
        job = self._create_progressive_turn(
            "confirmation",
            result,
            tuple(response.ros2_plan),
            response.ros2_error,
            confirmation=None,
        )
        self._start_moss_stream(job)
        return replace(response, playback_turn=job)

    def _handle_spoken_confirmation_if_pending(self, wav_path: Path, *, progressive: bool) -> Any | None:
        pending = self.confirmations.current()
        if pending is None:
            return None
        if self.pipeline.asr is None:
            raise RuntimeError("ASR engine is required for spoken confirmation.")
        started = time.perf_counter()
        asr_result = self.pipeline.asr.transcribe_wav(wav_path)
        spoken_text = normalize_asr_text(asr_result.text)
        decision = interpret_spoken_confirmation(spoken_text, pending)
        if decision == "confirm" and pending.confirmation_id is not None:
            confirmation_response = self.confirm_pending(pending.confirmation_id, play_response=False)
            result = self._synthetic_turn_result(
                input_text=spoken_text,
                response_text=confirmation_response.response_text or confirmation_response.message,
                total_seconds=time.perf_counter() - started,
                metadata={
                    "confirmation_resolved": True,
                    "confirmation_status": confirmation_response.status,
                },
            )
            if progressive:
                job = self._create_progressive_turn(
                    "voice",
                    result,
                    tuple(confirmation_response.ros2_plan),
                    confirmation_response.ros2_error,
                    confirmation=None,
                )
                self._start_moss_stream(job)
                return job.as_dict()
            return self._turn_response(
                "voice",
                result,
                tuple(confirmation_response.ros2_plan),
                confirmation_response.ros2_error,
            )
        if pending.confirmation_id is not None:
            self._mark_turns_interrupted()
            self._cancel_progressive_turns()
            self.confirmations.cancel(pending.confirmation_id)
        response_text = "已取消，未执行任何操作。" if decision == "cancel" else "语音确认不够明确，已取消，未执行任何操作。"
        result = self._synthetic_turn_result(
            input_text=spoken_text,
            response_text=response_text,
            total_seconds=time.perf_counter() - started,
            metadata={
                "confirmation_resolved": True,
                "confirmation_status": "canceled" if decision == "cancel" else "ambiguous",
            },
        )
        if progressive:
            job = self._create_progressive_turn("voice", result, (), None, confirmation=None)
            self._start_moss_stream(job)
            return job.as_dict()
        return self._turn_response("voice", result, (), None)

    def _synthetic_turn_result(
        self,
        *,
        input_text: str,
        response_text: str,
        total_seconds: float,
        metadata: dict[str, Any] | None = None,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            input_text=input_text,
            response_text=response_text,
            tts=None,
            total_seconds=total_seconds,
            vision_artifacts=(),
            metadata=dict(metadata or {}),
        )


    def _create_progressive_turn(
        self,
        mode: str,
        result: Any,
        ros2_plan: tuple[dict[str, Any], ...],
        ros2_error: str | None,
        *,
        confirmation: PendingConfirmation | None = None,
    ) -> WebProgressiveTurnJob:
        self._cleanup_progressive_turns()
        turn_id = secrets.token_urlsafe(12)
        job = WebProgressiveTurnJob(
            turn_id=turn_id,
            mode=mode,
            input_text=result.input_text,
            response_text=result.response_text,
            ros2_plan=ros2_plan,
            ros2_error=ros2_error,
            text_seconds=result.total_seconds,
            confirmation=confirmation,
            metadata=dict(getattr(result, "metadata", {}) or {}),
            image_artifacts=tuple(
                self._web_image_artifact(artifact) for artifact in getattr(result, "vision_artifacts", ()) or ()
            ),
            tts_segments=_split_moss_tts_text_segments(result.response_text),
        )
        with self._progressive_turns_lock:
            self._progressive_turns[turn_id] = job
        if not job.tts_segments:
            job.mark_complete()
        return job

    def _web_image_artifact(self, artifact: Any) -> dict[str, Any]:
        web_artifact = copy_vision_artifact_to_dir(artifact, self.config.image_dir)
        payload = web_artifact.as_dict()
        payload["url"] = _image_url_for_path(web_artifact.image_path)
        return payload

    def _cleanup_progressive_turns(self) -> None:
        cutoff = time.monotonic() - WEB_TTS_JOB_TTL_SECONDS
        with self._progressive_turns_lock:
            expired = [
                turn_id
                for turn_id, job in self._progressive_turns.items()
                if job.updated_at < cutoff and job.status in {"complete", "failed", "canceled"}
            ]
            for turn_id in expired:
                self._progressive_turns.pop(turn_id, None)

    def _start_moss_stream(self, job: WebProgressiveTurnJob) -> None:
        if not job.tts_segments:
            job.mark_complete()
            return
        proxy_audio_url = f"/api/chat-stream/{job.turn_id}/audio"
        job.set_proxy_audio_url(proxy_audio_url)
        if self.moss_streamer is None:
            job.mark_failed("MOSS realtime streaming server is unavailable.")
            return
        thread = threading.Thread(
            target=self._run_moss_stream_start,
            args=(job, proxy_audio_url),
            name=f"web-ui-moss-stream-{job.turn_id}",
            daemon=True,
        )
        thread.start()

    def _run_moss_stream_start(self, job: WebProgressiveTurnJob, proxy_audio_url: str) -> None:
        if job.is_canceled():
            return
        try:
            segment = job.tts_segments[0]
            start_payload = self.moss_streamer.start_stream(segment.text)
        except Exception as error:
            if job.is_canceled():
                return
            self._last_error = str(error)
            job.mark_failed(str(error))
            return
        if job.is_canceled():
            upstream_audio_url = str(start_payload.get("moss_audio_url") or start_payload.get("audio_url") or "")
            self._close_moss_stream_quietly(upstream_audio_url)
            return
        job.attach_stream(start_payload=start_payload, audio_url=proxy_audio_url)

    def _start_moss_segment_sync(
        self,
        job: WebProgressiveTurnJob,
        segment: MossTtsTextSegment,
        *,
        segment_index: int,
    ) -> str:
        if self.moss_streamer is None:
            raise RuntimeError("MOSS realtime streaming server is unavailable.")
        try:
            start_payload = self.moss_streamer.start_stream(segment.text)
        except Exception as error:
            self._last_error = str(error)
            job.mark_failed(str(error))
            raise
        job.attach_stream(
            start_payload=start_payload,
            audio_url=f"/api/chat-stream/{job.turn_id}/audio",
            segment_index=segment_index,
        )
        return str(start_payload.get("moss_audio_url") or start_payload.get("audio_url") or "")

    def _iter_moss_segment_audio(
        self,
        job: WebProgressiveTurnJob,
        *,
        sample_rate: int,
        channels: int,
    ) -> Iterator[bytes]:
        if self.moss_streamer is None:
            return
        segments = job.tts_segments
        for index, segment in enumerate(segments):
            if job.is_canceled():
                return
            if index == 0:
                upstream_audio_url = job.wait_for_moss_audio_url(timeout_seconds=MOSS_TTS_SEGMENT_START_TIMEOUT_SECONDS)
            else:
                upstream_audio_url = self._start_moss_segment_sync(
                    job,
                    segment,
                    segment_index=index,
                )
            if not upstream_audio_url:
                return
            try:
                for chunk in self.moss_streamer.stream_audio(upstream_audio_url):
                    if job.is_canceled():
                        return
                    yield chunk
            finally:
                # Always tell the upstream the stream is over. Without it an
                # abandoned stream keeps generating, fills its queue, and then
                # blocks the serve-wide execution lock for every later turn.
                self._close_moss_stream_quietly(upstream_audio_url)
            job.mark_segment_streamed()
            if segment.pause_after_seconds > 0.0:
                yield from _iter_pcm16_silence(segment.pause_after_seconds, sample_rate=sample_rate, channels=channels)

    def _close_moss_stream_quietly(self, audio_url: str) -> None:
        if self.moss_streamer is None or not audio_url:
            return
        try:
            self.moss_streamer.close_stream(audio_url)
        except Exception as error:
            self._last_error = f"MOSS stream cancellation failed: {error}"

    def _build_estop_plan(self) -> Ros2MessagePlan:
        config = self.ros2_bridge.config
        reason = "web-ui emergency stop"
        if not config.use_estop_request:
            return Ros2MessagePlan(
                topic=config.estop_bool_topic,
                message_type=ROS2_BOOL,
                payload=True,
                reason=reason,
            )
        payload = json.dumps(
            {
                "source": config.source,
                "active": True,
                "latch": True,
                "reason": reason,
            },
            ensure_ascii=False,
        )
        return Ros2MessagePlan(
            topic=config.estop_request_topic,
            message_type=ROS2_STRING,
            payload=payload,
            reason=reason,
        )


def _build_runtime_args(
    config: WebUiConfig,
    stack_config: VoiceStackConfig,
    tts_engine: str,
) -> SimpleNamespace:
    tts_output_dir = config.audio_dir
    moss_tts = getattr(stack_config, "moss_tts", None)
    return SimpleNamespace(
        llm_model=stack_config.llm_model,
        large_llm=False,
        max_new_tokens=stack_config.generation.max_new_tokens,
        asr_model=stack_config.asr_model,
        rules=stack_config.rules_path,
        object_mapping=stack_config.object_mapping_path,
        arm_rules=stack_config.arm_rules_path,
        cache_dir=stack_config.cache_dir,
        rule_edit_strategy=stack_config.rule_edit_strategy,
        vision_snapshot_service=stack_config.vision.snapshot_service,
        vision_snapshot_timeout=stack_config.vision.snapshot_timeout_seconds,
        tts_engine=tts_engine,
        tts_binary=stack_config.tts.binary,
        tts_model_dir=stack_config.tts.model_dir,
        tts_output_dir=tts_output_dir,
        tts_language=stack_config.tts.language,
        tts_speed=stack_config.tts.speed,
        tts_timeout=stack_config.tts.timeout_seconds,
        disable_tts_bert=stack_config.tts.disable_bert,
        disable_tts_denoise=stack_config.tts.disable_nf,
        moss_executable=getattr(moss_tts, "executable", PROJECT_ROOT / ".runtime" / "moss_tts_env" / "bin" / "moss-tts-nano"),
        moss_source_dir=getattr(moss_tts, "source_dir", PROJECT_ROOT / ".runtime" / "src" / "MOSS-TTS-Nano"),
        moss_model_dir=getattr(moss_tts, "model_dir", PROJECT_ROOT / "models" / "tts"),
        moss_voice=getattr(moss_tts, "voice", "Xiaoyu"),
        moss_prompt_audio=getattr(moss_tts, "prompt_audio", None),
        moss_cpu_threads=getattr(moss_tts, "cpu_threads", 4),
        moss_cpus=getattr(moss_tts, "cpu_affinity", None),
        moss_execution_provider=getattr(moss_tts, "execution_provider", "cpu"),
        moss_max_new_frames=getattr(moss_tts, "max_new_frames", 375),
        moss_voice_clone_max_text_tokens=getattr(moss_tts, "voice_clone_max_text_tokens", 75),
        moss_realtime_streaming_decode=getattr(moss_tts, "realtime_streaming_decode", 1),
        moss_timeout=getattr(moss_tts, "timeout_seconds", 180.0),
        piper_python=stack_config.piper_tts.python,
        piper_runner=stack_config.piper_tts.runner,
        piper_model_dir=stack_config.piper_tts.model_dir,
        piper_espeak_data_dir=stack_config.piper_tts.espeak_data_dir,
        piper_speed=stack_config.piper_tts.speed,
        piper_silence_scale=stack_config.piper_tts.silence_scale,
        piper_threads=stack_config.piper_tts.threads,
        piper_timeout=stack_config.piper_tts.timeout_seconds,
        dry_run_ros2=config.ros2_dry_run,
        direct_estop_topic=config.direct_estop_topic,
        ros2_node_name="web_ui_ros2_bridge",
        voice_source="web_ui",
        transcript_topic="/voice/transcript",
        response_topic="/voice/assistant_response",
        estop_request_topic="/safety/estop/request",
        estop_bool_topic="/emergency_stop",
        goal_topic="/goal",
        ros2_wait=0.2,
        no_transcript_topic=False,
        no_response_topic=False,
        no_command_topics=False,
        require_confirmation_for_side_effects=True,
    )


def _find_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _combine_runtime_errors(*errors: str | None) -> str | None:
    messages = [error for error in errors if error]
    if not messages:
        return None
    return "；".join(messages)


def _resolve_moss_demo_id(source_dir: Path, prompt_audio: Path | None) -> str:
    if prompt_audio is None:
        return "demo-1"
    source_root = source_dir.expanduser().resolve()
    target = prompt_audio.expanduser().resolve()
    metadata_path = source_root / "assets" / "demo.jsonl"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"MOSS demo metadata does not exist: {metadata_path}")

    demo_index = 0
    for raw_line in metadata_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        role = str(payload.get("role", "")).strip()
        text = str(payload.get("text", "")).strip()
        if not role or not text:
            continue
        prompt_path = (source_root / role).resolve()
        if not prompt_path.is_file():
            continue
        demo_index += 1
        if prompt_path == target:
            return f"demo-{demo_index}"
    raise FileNotFoundError(f"MOSS prompt audio is not listed in demo metadata: {target}")


def _warm_component(name: str, component: Any | None) -> dict[str, Any]:
    if component is None:
        return {"component": name, "status": "skipped", "reason": "not configured"}
    warm_start = getattr(component, "warm_start", None)
    if callable(warm_start):
        result = warm_start()
        if isinstance(result, dict):
            return {"component": name, **result}
        return {"component": name, "status": "ready", "result": result}
    warm_load = getattr(component, "warm_load", None)
    if callable(warm_load):
        result = warm_load()
        if isinstance(result, dict):
            return {"component": name, **result}
        return {"component": name, "status": "ready", "result": result}
    return {"component": name, "status": "skipped", "reason": "warm loading is not supported"}


def _available_tts_engines(stack_config: VoiceStackConfig) -> tuple[str, ...]:
    engines: list[str] = []
    moss_tts = getattr(stack_config, "moss_tts", None)
    if moss_tts is not None and _moss_available(moss_tts):
        engines.append("moss")
    if _melo_available(stack_config.tts):
        engines.append("melo")
    if _piper_available(stack_config.piper_tts):
        engines.append("piper")
    return tuple(engines)


def _resolve_tts_engine(requested: str, available: tuple[str, ...]) -> str:
    requested_normalized = requested.strip().lower()
    if requested_normalized == "auto":
        for engine in ("moss", "melo", "piper"):
            if engine in available:
                return engine
        return "none"
    return requested_normalized


def _moss_available(config: Any) -> bool:
    prompt_audio = config.prompt_audio
    prompt_ok = prompt_audio is None or prompt_audio.exists()
    return config.executable.exists() and config.source_dir.exists() and config.model_dir.exists() and prompt_ok


def _melo_available(config: MeloTtsConfig) -> bool:
    return config.binary.exists() and config.model_dir.exists()


def _piper_available(config: PiperTtsConfig) -> bool:
    if not config.python.exists() or not config.runner.exists():
        return False
    if config.model_dir is None or not config.model_dir.exists():
        return False
    if config.espeak_data_dir is not None and not config.espeak_data_dir.exists():
        return False
    return True


def _audio_url_for_path(path: Path) -> str:
    return f"/audio/{path.name}"


def _image_url_for_path(path: Path) -> str:
    return f"/images/{path.name}"


def _validate_voice_upload(audio_bytes: bytes) -> None:
    if not audio_bytes:
        raise ValueError("audio is required")
    if len(audio_bytes) > WEB_VOICE_UPLOAD_MAX_BYTES:
        raise ValueError(f"Voice upload too large; max {format_byte_limit(WEB_VOICE_UPLOAD_MAX_BYTES)}")


def _convert_uploaded_audio(temp_root: Path, audio_bytes: bytes, content_type: str | None) -> Path:
    uploaded = temp_root / f"upload{_suffix_for_content_type(content_type)}"
    wav_path = temp_root / "recording.wav"
    uploaded.write_bytes(audio_bytes)
    _ffmpeg_convert_to_wav(uploaded, wav_path)
    return wav_path


def _suffix_for_content_type(content_type: str | None) -> str:
    if not content_type:
        return ".webm"
    mime = content_type.split(";", 1)[0].strip().lower()
    suffix = _AUDIO_SUFFIX_BY_MIME.get(mime) or mimetypes.guess_extension(mime)
    return suffix or ".webm"


def format_byte_limit(value: int) -> str:
    if value % (1024 * 1024) == 0:
        return f"{value // (1024 * 1024)} MiB"
    if value % 1024 == 0:
        return f"{value // 1024} KiB"
    return f"{value} bytes"


def _ffmpeg_convert_to_wav(input_path: Path, output_path: Path) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise RuntimeError("ffmpeg is required for browser voice uploads.") from error
    if completed.returncode != 0:
        output = completed.stderr.strip() or completed.stdout.strip()
        if _is_invalid_uploaded_audio_error(output):
            raise ValueError("Invalid uploaded audio; please retry recording.")
        raise RuntimeError(
            "Failed to convert uploaded audio with ffmpeg: "
            f"{output}"
        )


def _is_invalid_uploaded_audio_error(output: str) -> bool:
    normalized = output.lower()
    return any(
        marker in normalized
        for marker in (
            "invalid data found when processing input",
            "ebml header parsing failed",
            "invalid ebml",
            "moov atom not found",
            "error opening input",
        )
    )


def _constant_time_equal(left: str, right: str) -> bool:
    return secrets.compare_digest(left.encode("utf-8"), right.encode("utf-8"))
