"""Subprocess bridges for local TTS runtimes."""

from __future__ import annotations

import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from local_safety_assistant.stack.config import MeloTtsConfig, MossTtsConfig, PiperTtsConfig


@dataclass(frozen=True)
class TtsResult:
    text: str
    audio_paths: tuple[Path, ...]
    command: tuple[str, ...]
    elapsed_seconds: float
    stdout: str
    stderr: str


class MeloTtsBridge:
    """Invoke `MeloTTS.cpp/build/meloTTS_ov` with temporary UTF-8 input files."""

    def __init__(
        self,
        config: MeloTtsConfig,
        *,
        tts_device: str = "CPU",
        bert_device: str = "CPU",
        denoise_device: str = "CPU",
    ) -> None:
        self.config = config
        self.tts_device = tts_device
        self.bert_device = bert_device
        self.denoise_device = denoise_device

    def build_command(self, input_file: Path, output_prefix: Path) -> tuple[str, ...]:
        command = [
            str(self.config.binary),
            "--model_dir",
            str(self.config.model_dir),
            "--input_file",
            str(input_file),
            "--output_filename",
            str(output_prefix),
            "--language",
            self.config.language,
            "--speed",
            str(self.config.speed),
            "--tts_device",
            self.tts_device,
            "--bert_device",
            self.bert_device,
            "--nf_device",
            self.denoise_device,
            "--quantize",
            _bool_arg(self.config.quantize),
            "--disable_bert",
            _bool_arg(self.config.disable_bert),
            "--disable_nf",
            _bool_arg(self.config.disable_nf),
        ]
        return tuple(command)

    def synthesize(self, text: str, *, output_name: str | None = None) -> TtsResult:
        if not self.config.binary.exists():
            raise FileNotFoundError(f"MeloTTS binary does not exist: {self.config.binary}")
        if not self.config.model_dir.exists():
            raise FileNotFoundError(f"MeloTTS model directory does not exist: {self.config.model_dir}")

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        stem = output_name or f"voice_{int(time.time() * 1000)}"
        output_prefix = self.config.output_dir / stem

        with tempfile.TemporaryDirectory(prefix="melotts_input_") as temp_dir:
            input_file = Path(temp_dir) / "input.txt"
            prepared_text = prepare_safety_tts_text(text)
            input_file.write_text(prepared_text + "\n", encoding="utf-8")
            command = self.build_command(input_file, output_prefix)
            started = time.perf_counter()
            completed = subprocess.run(
                command,
                cwd=self.config.binary.parent.parent,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
            )
            elapsed = time.perf_counter() - started

        if completed.returncode != 0:
            raise RuntimeError(
                "MeloTTS failed with exit code "
                f"{completed.returncode}: {completed.stderr.strip() or completed.stdout.strip()}"
            )

        audio_paths = tuple(sorted(self.config.output_dir.glob(f"{stem}_*.wav")))
        if not audio_paths:
            expected = _expected_output_path(output_prefix, self.config.language)
            if expected.exists():
                audio_paths = (expected,)
        if not audio_paths:
            raise RuntimeError(f"MeloTTS finished but no WAV output matched: {output_prefix}_*.wav")

        return TtsResult(
            text=prepared_text,
            audio_paths=audio_paths,
            command=command,
            elapsed_seconds=elapsed,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


class MossTtsBridge:
    """Invoke the isolated MOSS-TTS-Nano ONNX CLI for one-shot synthesis."""

    def __init__(self, config: MossTtsConfig) -> None:
        self.config = config

    def build_command(self, input_file: Path, output_file: Path) -> tuple[str, ...]:
        command = [
            str(self.config.executable),
            "generate",
            "--backend",
            "onnx",
            "--onnx-model-dir",
            str(self.config.model_dir),
            "--text-file",
            str(input_file),
            "--output-audio-path",
            str(output_file),
            "--mode",
            "voice_clone",
            "--voice",
            self.config.voice,
            "--execution-provider",
            self.config.execution_provider,
            "--cpu-threads",
            str(self.config.cpu_threads),
            "--max-new-frames",
            str(self.config.max_new_frames),
            "--voice-clone-max-text-tokens",
            str(self.config.voice_clone_max_text_tokens),
            "--sample-mode",
            self.config.sample_mode,
            "--realtime-streaming-decode",
            str(self.config.realtime_streaming_decode),
            "--text-temperature",
            str(self.config.text_temperature),
            "--text-top-p",
            str(self.config.text_top_p),
            "--text-top-k",
            str(self.config.text_top_k),
            "--audio-temperature",
            str(self.config.audio_temperature),
            "--audio-top-p",
            str(self.config.audio_top_p),
            "--audio-top-k",
            str(self.config.audio_top_k),
            "--audio-repetition-penalty",
            str(self.config.audio_repetition_penalty),
        ]
        if self.config.prompt_audio is not None:
            command.extend(["--prompt-speech", str(self.config.prompt_audio)])
        if self.config.cpu_affinity:
            command[:0] = ["taskset", "--cpu-list", self.config.cpu_affinity]
        return tuple(command)

    def synthesize(self, text: str, *, output_name: str | None = None) -> TtsResult:
        if not self.config.executable.exists():
            raise FileNotFoundError(f"MOSS-TTS-Nano executable does not exist: {self.config.executable}")
        if not self.config.source_dir.exists():
            raise FileNotFoundError(f"MOSS-TTS-Nano source directory does not exist: {self.config.source_dir}")
        if not self.config.model_dir.exists():
            raise FileNotFoundError(f"MOSS-TTS-Nano ONNX model directory does not exist: {self.config.model_dir}")
        if self.config.prompt_audio is not None and not self.config.prompt_audio.exists():
            raise FileNotFoundError(f"MOSS-TTS-Nano prompt audio does not exist: {self.config.prompt_audio}")

        output_dir = self.config.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = output_name or f"moss_voice_{int(time.time() * 1000)}"
        output_file = output_dir / (stem if stem.endswith(".wav") else f"{stem}.wav")

        with tempfile.TemporaryDirectory(prefix="moss_tts_input_") as temp_dir:
            input_file = Path(temp_dir) / "input.txt"
            prepared_text = prepare_safety_tts_text(text)
            input_file.write_text(prepared_text + "\n", encoding="utf-8")
            command = self.build_command(input_file, output_file)
            started = time.perf_counter()
            completed = subprocess.run(
                command,
                cwd=self.config.source_dir,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
            )
            elapsed = time.perf_counter() - started

        if completed.returncode != 0:
            raise RuntimeError(
                "MOSS-TTS-Nano failed with exit code "
                f"{completed.returncode}: {completed.stderr.strip() or completed.stdout.strip()}"
            )
        if not output_file.exists():
            raise RuntimeError(f"MOSS-TTS-Nano finished but did not write WAV output: {output_file}")

        return TtsResult(
            text=prepared_text,
            audio_paths=(output_file,),
            command=command,
            elapsed_seconds=elapsed,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


class PiperTtsBridge:
    """Invoke the optional sherpa-onnx Piper runner in an isolated Python environment."""

    def __init__(self, config: PiperTtsConfig) -> None:
        self.config = config

    def build_command(self, input_file: Path, output_file: Path) -> tuple[str, ...]:
        if self.config.model_dir is None:
            raise ValueError("--piper-model-dir is required when --tts-engine piper is selected")
        command = [
            str(_absolute_path(self.config.python)),
            str(_absolute_path(self.config.runner)),
            "--model-dir",
            str(_absolute_path(self.config.model_dir)),
            "--text-file",
            str(_absolute_path(input_file)),
            "--output-file",
            str(_absolute_path(output_file)),
            "--speed",
            str(self.config.speed),
            "--silence-scale",
            str(self.config.silence_scale),
            "--threads",
            str(self.config.threads),
        ]
        if self.config.espeak_data_dir is not None:
            command.extend(["--espeak-data-dir", str(_absolute_path(self.config.espeak_data_dir))])
        return tuple(command)

    def synthesize(self, text: str, *, output_name: str | None = None) -> TtsResult:
        if not self.config.python.exists():
            raise FileNotFoundError(f"Piper Python executable does not exist: {self.config.python}")
        if not self.config.runner.exists():
            raise FileNotFoundError(f"Piper runner does not exist: {self.config.runner}")
        if self.config.model_dir is None:
            raise FileNotFoundError("--piper-model-dir is required when using Piper TTS")
        if not self.config.model_dir.exists():
            raise FileNotFoundError(f"Piper model directory does not exist: {self.config.model_dir}")
        if self.config.espeak_data_dir is not None and not self.config.espeak_data_dir.exists():
            raise FileNotFoundError(f"Piper espeak-ng-data directory does not exist: {self.config.espeak_data_dir}")

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        stem = output_name or f"piper_voice_{int(time.time() * 1000)}"
        output_file = self.config.output_dir / f"{stem}.wav"

        with tempfile.TemporaryDirectory(prefix="piper_tts_input_") as temp_dir:
            input_file = Path(temp_dir) / "input.txt"
            prepared_text = prepare_safety_tts_text(text)
            input_file.write_text(prepared_text + "\n", encoding="utf-8")
            command = self.build_command(input_file, output_file)
            started = time.perf_counter()
            completed = subprocess.run(
                command,
                cwd=self.config.runner.parent,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
            )
            elapsed = time.perf_counter() - started

        if completed.returncode != 0:
            raise RuntimeError(
                "Piper TTS failed with exit code "
                f"{completed.returncode}: {completed.stderr.strip() or completed.stdout.strip()}"
            )
        if not output_file.exists():
            raise RuntimeError(f"Piper TTS finished but did not write WAV output: {output_file}")

        return TtsResult(
            text=prepared_text,
            audio_paths=(output_file,),
            command=command,
            elapsed_seconds=elapsed,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


def _bool_arg(value: bool) -> str:
    return "true" if value else "false"


def _expected_output_path(output_prefix: Path, language: str) -> Path:
    suffix = "ZH-MIX-EN" if language.upper() == "ZH" else "EN-US"
    return output_prefix.with_name(f"{output_prefix.name}_{suffix}.wav")


def _absolute_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def prepare_melotts_text(text: str) -> str:
    return prepare_safety_tts_text(text)


def prepare_safety_tts_text(text: str) -> str:
    """Make short Chinese assistant replies easier for local TTS engines to phrase."""

    prepared = " ".join(text.strip().split())
    if not prepared:
        return prepared
    prepared = _normalize_spoken_symbols(prepared)
    prepared = _expand_project_acronyms(prepared)
    prepared = _add_chinese_phrase_pauses(prepared)
    if not prepared:
        return prepared
    return _ensure_sentence_punctuation(prepared)


def _normalize_spoken_symbols(text: str) -> str:
    replacements = {
        ",": "，",
        ";": "；",
        ":": "：",
        "?": "？",
        "!": "！",
        "／": "/",
    }
    normalized = text
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    normalized = re.sub(r"(?<![A-Za-z0-9])([A-Z]{2,})\s*/\s*([A-Z]{2,})(?![A-Za-z0-9])", r"\1 和 \2", normalized)
    return normalized


def _expand_project_acronyms(text: str) -> str:
    replacements = {
        "OpenVINO": "Open VINO",
        "Qwen3.5": "通义千问三点五",
        "Qwen3": "通义千问三",
    }
    expanded = text
    for source, target in replacements.items():
        expanded = expanded.replace(source, target)
    return re.sub(r"(?<![A-Za-z0-9])(ROS|NPU|CPU|GPU|ASR|TTS|LLM|API|JSON|USB|IMU)(?![A-Za-z0-9])", _spell_acronym, expanded)


def _spell_acronym(match: re.Match[str]) -> str:
    return " ".join(match.group(1))


def _add_chinese_phrase_pauses(text: str) -> str:
    paused = text
    paused = re.sub(r"(?<![，；。！？])时(?=(?:立即|进入|停止|降低|限速|触发|执行|需要))", "时，", paused)
    paused = re.sub(r"(?<![，；。！？])后(?=(?:请|再|继续|通知|确认|复位))", "后，", paused)
    paused = re.sub(r"(?<![，；。！？])，?(?=请确认)", "，", paused)
    paused = re.sub(r"(?<![，；。！？])并(?=(?:通知|要求|进入|停止|降低|复位))", "，并", paused)
    paused = re.sub(r"(?<![，；。！？])否则", "，否则", paused)
    paused = re.sub(r"([。！？；])+", lambda match: match.group(0)[0], paused)
    paused = re.sub(r"，+", "，", paused)
    return paused.strip("，； ")


def _ensure_sentence_punctuation(text: str) -> str:
    if text[-1] in "。！？":
        return text
    if any(word in text for word in ("急停", "立即停止", "紧急停止")):
        return f"{text}！"
    return f"{text}。"
