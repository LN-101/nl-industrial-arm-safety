"""OpenVINO GenAI Whisper ASR engine."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any


DEFAULT_ASR_HOTWORDS = (
    "机械臂",
    "安全",
    "规则",
    "急停",
    "距离",
    "人员",
    "限制",
    "气泵",
    "映射",
    "A",
    "B",
    "C",
    "D",
    "ABCD",
    "标号A",
    "标号B",
    "标号C",
    "标号D",
)


@dataclass(frozen=True)
class AsrResult:
    text: str
    model: str
    device: str
    audio_seconds: float
    load_seconds: float
    inference_seconds: float


class WhisperAsrEngine:
    """Lazy WhisperPipeline wrapper for 16 kHz PCM WAV transcription."""

    def __init__(
        self,
        *,
        model: str,
        device: str,
        cache_dir: Path,
        fallback: tuple[str, ...] = (),
        language: str | None = None,
        hotwords: tuple[str, ...] | str | None = DEFAULT_ASR_HOTWORDS,
    ) -> None:
        self.model = model
        self.device = device
        self.fallback = fallback
        self.cache_dir = cache_dir
        self.language = language
        self.hotwords = _format_hotwords(hotwords)
        self._pipe: Any | None = None
        self._loaded_device: str | None = None
        self._load_seconds = 0.0

    def transcribe_wav(self, audio_path: Path) -> AsrResult:
        from local_safety_assistant.model_testbed import load_wav_mono_16k

        audio = load_wav_mono_16k(audio_path.expanduser().resolve())
        return self.transcribe_audio(audio, audio_seconds=len(audio) / 16000.0)

    def transcribe_audio(self, audio: Any, *, audio_seconds: float) -> AsrResult:
        errors: list[str] = []
        for device in (self.device, *self.fallback):
            try:
                pipe, load_seconds = self._load(device)
                kwargs: dict[str, Any] = {}
                if self.language:
                    kwargs["language"] = self.language if self.language.startswith("<|") else f"<|{self.language}|>"
                if self.hotwords:
                    kwargs["hotwords"] = self.hotwords
                started = time.perf_counter()
                result = pipe.generate(audio.tolist(), **kwargs)
                elapsed = time.perf_counter() - started
                return AsrResult(
                    text=_extract_text(result).strip(),
                    model=self.model,
                    device=device,
                    audio_seconds=audio_seconds,
                    load_seconds=load_seconds,
                    inference_seconds=elapsed,
                )
            except Exception as error:
                errors.append(f"{device}: {type(error).__name__}: {error}")
                self._pipe = None
                self._loaded_device = None
        raise RuntimeError("All ASR devices failed: " + " | ".join(errors))

    def warm_load(self) -> dict[str, Any]:
        errors: list[str] = []
        for device in (self.device, *self.fallback):
            try:
                _, load_seconds = self._load(device)
                return {
                    "status": "ready",
                    "model": self.model,
                    "device": device,
                    "load_seconds": load_seconds,
                }
            except Exception as error:
                errors.append(f"{device}: {type(error).__name__}: {error}")
                self._pipe = None
                self._loaded_device = None
        raise RuntimeError("All ASR devices failed during warmup: " + " | ".join(errors))

    def _load(self, device: str) -> tuple[Any, float]:
        if self._pipe is not None and self._loaded_device == device:
            return self._pipe, 0.0

        import openvino as ov
        import openvino_genai as ov_genai
        from local_safety_assistant.model_testbed import detect_model, pipeline_properties, resolve_model_path

        model_path = resolve_model_path(self.model)
        info = detect_model(model_path)
        if not info.exists:
            raise FileNotFoundError(f"ASR model path does not exist: {info.path}")
        if info.kind != "asr":
            raise RuntimeError(f"Model is {info.kind!r}, not an ASR model: {info.path}")

        args = SimpleNamespace(
            max_prompt_len=128,
            min_response_len=4,
            max_new_tokens=64,
            npu_prefill_hint="STATIC",
            npu_generate_hint="FAST_COMPILE",
            npu_compiler_type=None,
        )
        props = pipeline_properties(ov.Core(), device, "asr", self.cache_dir, args)
        started = time.perf_counter()
        self._pipe = ov_genai.WhisperPipeline(info.path, device, **props)
        self._loaded_device = device
        self._load_seconds = time.perf_counter() - started
        return self._pipe, self._load_seconds


def _format_hotwords(hotwords: tuple[str, ...] | str | None) -> str | None:
    if isinstance(hotwords, str):
        return hotwords.strip() or None
    if hotwords is None:
        return None
    words = tuple(word.strip() for word in hotwords if word.strip())
    return "，".join(words) or None


def _extract_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    text = getattr(result, "text", None)
    if text:
        return str(text)
    texts = getattr(result, "texts", None)
    if texts:
        return "\n".join(str(item) for item in texts)
    return str(result)
