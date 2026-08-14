"""Shared paths and model aliases for the local safety assistant."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".ov_cache"


@dataclass(frozen=True)
class ModelAlias:
    name: str
    path: Path
    purpose: str


MODEL_ALIASES: dict[str, ModelAlias] = {
    "qwen35-2b": ModelAlias(
        name="qwen35-2b",
        path=MODELS_DIR / "Qwen3.5-2B-int4-ov",
        purpose="Primary small local VLM/text model for rule-editing smoke tests.",
    ),
    "qwen35-4b": ModelAlias(
        name="qwen35-4b",
        path=MODELS_DIR / "Qwen3.5-4B-int4-ov",
        purpose="Third-party Qwen3.5/Qwopus 4B VLM benchmark candidate.",
    ),
    "qwen35-9b": ModelAlias(
        name="qwen35-9b",
        path=MODELS_DIR / "Qwen3.5-9B-int4-ov",
        purpose="Larger local VLM/text model for device/resource validation.",
    ),
    "whisper-large-v3": ModelAlias(
        name="whisper-large-v3",
        path=MODELS_DIR / "asr" / "whisper-large-v3-int4-ov",
        purpose="Local ASR model for offline speech transcription.",
    ),
    "whisper-large-v3-turbo": ModelAlias(
        name="whisper-large-v3-turbo",
        path=MODELS_DIR / "asr" / "whisper-large-v3-turbo-int4-ov",
        purpose="Default low-latency ASR model for the voice assistant stack.",
    ),
    "qwen3-8b": ModelAlias(
        name="qwen3-8b",
        path=MODELS_DIR / "Qwen3-8B-int4-cw-ov",
        purpose="Older comparison LLM model.",
    ),
    "deepseek-1.5b": ModelAlias(
        name="deepseek-1.5b",
        path=MODELS_DIR / "DeepSeek-R1-Distill-Qwen-1.5B-int4-gq-ov",
        purpose="Older comparison reasoning model.",
    ),
}


DEFAULT_LLM_ALIAS = "qwen35-2b"
DEFAULT_ASR_ALIAS = "whisper-large-v3-turbo"
