"""Realtime microphone capture and endpointing for voice turns."""

from __future__ import annotations

import math
import queue
import tempfile
import time
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


TARGET_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class MicrophoneConfig:
    sample_rate: int | None = None
    target_sample_rate: int = TARGET_SAMPLE_RATE
    block_seconds: float = 0.03
    channels: int = 1
    device: str | int | None = None
    queue_size: int = 128


@dataclass(frozen=True)
class EndpointingConfig:
    sample_rate: int = TARGET_SAMPLE_RATE
    speech_threshold: float = 0.06
    min_speech_seconds: float = 0.5
    trailing_silence_seconds: float = 1.6
    pre_roll_seconds: float = 0.4
    max_utterance_seconds: float = 12.0


@dataclass(frozen=True)
class CapturedUtterance:
    audio: np.ndarray
    sample_rate: int
    audio_seconds: float
    speech_seconds: float
    peak: float
    rms: float
    reason: str


@dataclass(frozen=True)
class AudioDeviceInfo:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: float


class EnergyEndpointDetector:
    """Dependency-light utterance endpointing based on frame RMS."""

    def __init__(self, config: EndpointingConfig | None = None) -> None:
        self.config = config or EndpointingConfig()
        self._pre_roll: deque[np.ndarray] = deque()
        self._speech_frames: list[np.ndarray] = []
        self._triggered = False
        self._speech_seconds = 0.0
        self._silence_seconds = 0.0
        self._total_seconds = 0.0
        self._pre_roll_seconds = 0.0

    def accept(self, frame: np.ndarray) -> CapturedUtterance | None:
        normalized = normalize_audio_frame(frame)
        if normalized.size == 0:
            return None

        frame_seconds = normalized.size / float(self.config.sample_rate)
        is_speech = frame_rms(normalized) >= self.config.speech_threshold

        if not self._triggered:
            if not is_speech:
                self._add_pre_roll(normalized, frame_seconds)
                return None
            self._triggered = True
            self._speech_frames = [*self._pre_roll, normalized]
            self._speech_seconds = frame_seconds
            self._silence_seconds = 0.0
            self._total_seconds = sum(chunk.size for chunk in self._speech_frames) / float(self.config.sample_rate)
            return None

        self._speech_frames.append(normalized)
        self._total_seconds += frame_seconds
        if is_speech:
            self._speech_seconds += frame_seconds
            self._silence_seconds = 0.0
        else:
            self._silence_seconds += frame_seconds

        if self._total_seconds >= self.config.max_utterance_seconds:
            return self._finish("max_utterance_seconds")
        if self._silence_seconds >= self.config.trailing_silence_seconds:
            return self._finish("trailing_silence")
        return None

    def flush(self) -> CapturedUtterance | None:
        if not self._triggered:
            return None
        return self._finish("flush")

    def _add_pre_roll(self, frame: np.ndarray, frame_seconds: float) -> None:
        self._pre_roll.append(frame)
        self._pre_roll_seconds += frame_seconds
        while self._pre_roll_seconds > self.config.pre_roll_seconds and self._pre_roll:
            removed = self._pre_roll.popleft()
            self._pre_roll_seconds -= removed.size / float(self.config.sample_rate)

    def _finish(self, reason: str) -> CapturedUtterance | None:
        audio = concatenate_audio(self._speech_frames)
        speech_seconds = self._speech_seconds
        self._reset_after_utterance()
        if speech_seconds < self.config.min_speech_seconds:
            return None
        return CapturedUtterance(
            audio=audio,
            sample_rate=self.config.sample_rate,
            audio_seconds=audio.size / float(self.config.sample_rate),
            speech_seconds=speech_seconds,
            peak=float(np.max(np.abs(audio))) if audio.size else 0.0,
            rms=frame_rms(audio),
            reason=reason,
        )

    def _reset_after_utterance(self) -> None:
        self._pre_roll.clear()
        self._speech_frames = []
        self._triggered = False
        self._speech_seconds = 0.0
        self._silence_seconds = 0.0
        self._total_seconds = 0.0
        self._pre_roll_seconds = 0.0


def iter_microphone_utterances(
    mic_config: MicrophoneConfig | None = None,
    endpoint_config: EndpointingConfig | None = None,
) -> Iterator[CapturedUtterance]:
    """Yield completed utterances from the default microphone until interrupted."""

    config = mic_config or MicrophoneConfig()
    capture_sample_rate = config.sample_rate or default_input_sample_rate(config.device)
    endpoint = EnergyEndpointDetector(endpoint_config or EndpointingConfig(sample_rate=config.target_sample_rate))
    frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=config.queue_size)
    sd = require_sounddevice()

    def callback(indata, frames_count, time_info, status) -> None:  # noqa: ANN001
        del frames_count, time_info
        if status:
            print(f"Microphone status: {status}")
        try:
            frames.put_nowait(np.array(indata, copy=True))
        except queue.Full:
            try:
                frames.get_nowait()
            except queue.Empty:
                pass
            frames.put_nowait(np.array(indata, copy=True))

    blocksize = max(1, int(round(capture_sample_rate * config.block_seconds)))
    try:
        with sd.InputStream(
            samplerate=capture_sample_rate,
            channels=config.channels,
            dtype="float32",
            blocksize=blocksize,
            device=config.device,
            callback=callback,
        ):
            while True:
                frame = resample_audio(frames.get(), capture_sample_rate, endpoint.config.sample_rate)
                utterance = endpoint.accept(frame)
                if utterance is not None:
                    yield utterance
                    drain_frame_queue(frames)
    except Exception as exc:
        raise RuntimeError(
            "Could not open microphone input stream. Run `Code/voice_stack.py audio-devices` "
            "to pick a valid --mic-device, or set --mic-sample-rate to a rate supported by that device."
        ) from exc


def require_sounddevice():
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Realtime microphone listening requires the optional 'sounddevice' Python package "
            "and the system PortAudio library. Install sounddevice in qwen35_env and install "
            "PortAudio for your OS, for example: ./qwen35_env/bin/python -m pip install sounddevice "
            "and sudo apt install portaudio19-dev"
        ) from exc
    return sd


def list_audio_devices() -> list[AudioDeviceInfo]:
    sd = require_sounddevice()
    devices = sd.query_devices()
    result: list[AudioDeviceInfo] = []
    for index, device in enumerate(devices):
        result.append(
            AudioDeviceInfo(
                index=index,
                name=str(device.get("name", "")),
                max_input_channels=int(device.get("max_input_channels", 0)),
                max_output_channels=int(device.get("max_output_channels", 0)),
                default_sample_rate=float(device.get("default_samplerate", 0.0)),
            )
        )
    return result


def default_input_sample_rate(device: str | int | None = None) -> int:
    sd = require_sounddevice()
    try:
        info = sd.query_devices(device, "input") if device is not None else sd.query_devices(kind="input")
        return int(round(float(info.get("default_samplerate") or TARGET_SAMPLE_RATE)))
    except Exception:
        return TARGET_SAMPLE_RATE


def drain_frame_queue(frames: queue.Queue[np.ndarray]) -> None:
    while True:
        try:
            frames.get_nowait()
        except queue.Empty:
            return


def play_wav_file(path: Path, *, device: str | int | None = None) -> None:
    sd = require_sounddevice()
    audio, sample_rate = read_wav_float32(path)
    if audio.size == 0:
        return
    sd.play(audio, sample_rate, device=device)
    sd.wait()


def segment_frames(
    frames: Iterable[np.ndarray],
    config: EndpointingConfig | None = None,
    *,
    flush: bool = True,
) -> list[CapturedUtterance]:
    endpoint = EnergyEndpointDetector(config)
    utterances: list[CapturedUtterance] = []
    for frame in frames:
        utterance = endpoint.accept(frame)
        if utterance is not None:
            utterances.append(utterance)
    if flush:
        utterance = endpoint.flush()
        if utterance is not None:
            utterances.append(utterance)
    return utterances


def write_utterance_wav(
    utterance: CapturedUtterance,
    directory: Path | None = None,
    *,
    prefix: str = "voice_turn_",
) -> Path:
    if directory is None:
        target_dir = Path(tempfile.mkdtemp(prefix="voice-listen-"))
    else:
        target_dir = directory.expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)

    path = target_dir / f"{prefix}{time.strftime('%Y%m%d_%H%M%S')}_{time.monotonic_ns()}.wav"
    write_pcm16_wav(path, utterance.audio, utterance.sample_rate)
    return path


def write_pcm16_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(normalize_audio_frame(audio), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def resample_audio(frame: np.ndarray, source_sample_rate: int, target_sample_rate: int) -> np.ndarray:
    audio = normalize_audio_frame(frame)
    if source_sample_rate == target_sample_rate or audio.size == 0:
        return audio

    from scipy import signal

    divisor = math.gcd(int(source_sample_rate), int(target_sample_rate))
    up = int(target_sample_rate) // divisor
    down = int(source_sample_rate) // divisor
    return signal.resample_poly(audio, up, down).astype(np.float32, copy=False)


def read_wav_float32(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    if channels > 1:
        audio = audio.reshape(-1, channels)
    return np.clip(audio, -1.0, 1.0), sample_rate


def normalize_audio_frame(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame, dtype=np.float32)
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim == 1:
        return array
    return array.reshape(-1, array.shape[-1]).mean(axis=1).astype(np.float32)


def concatenate_audio(frames: Iterable[np.ndarray]) -> np.ndarray:
    chunks = [normalize_audio_frame(frame) for frame in frames]
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32, copy=False)


def frame_rms(frame: np.ndarray) -> float:
    normalized = normalize_audio_frame(frame)
    if normalized.size == 0:
        return 0.0
    return float(math.sqrt(float(np.mean(np.square(normalized, dtype=np.float64)))))
