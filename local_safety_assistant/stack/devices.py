"""Device allocation policy for ASR, LLM, and C++ MeloTTS stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


KNOWN_DEVICES = ("CPU", "GPU", "NPU")


@dataclass(frozen=True)
class StageDevicePlan:
    stage: str
    selected: str
    fallback: tuple[str, ...]
    reason: str

    @property
    def run_order(self) -> tuple[str, ...]:
        return (self.selected, *self.fallback)


@dataclass(frozen=True)
class StackDevicePlan:
    asr: StageDevicePlan
    llm_small: StageDevicePlan
    llm_large: StageDevicePlan
    tts: StageDevicePlan
    tts_bert: StageDevicePlan
    tts_denoise: StageDevicePlan

    def stages(self) -> tuple[StageDevicePlan, ...]:
        return (
            self.asr,
            self.llm_small,
            self.llm_large,
            self.tts,
            self.tts_bert,
            self.tts_denoise,
        )

    def as_dict(self) -> dict[str, dict[str, object]]:
        return {
            plan.stage: {
                "selected": plan.selected,
                "fallback": list(plan.fallback),
                "run_order": list(plan.run_order),
                "reason": plan.reason,
            }
            for plan in self.stages()
        }


def normalize_devices(devices: Iterable[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for device in devices:
        normalized = str(device).upper()
        if normalized in KNOWN_DEVICES and normalized not in seen:
            seen.append(normalized)
    return tuple(seen)


def available_openvino_devices() -> tuple[str, ...]:
    import openvino as ov

    return normalize_devices(ov.Core().available_devices)


def _choose(stage: str, available: tuple[str, ...], preferred: tuple[str, ...], reason: str) -> StageDevicePlan:
    ordered = tuple(device for device in preferred if device in available)
    if not ordered:
        ordered = ("CPU",)
    return StageDevicePlan(stage=stage, selected=ordered[0], fallback=ordered[1:], reason=reason)


def build_device_plan(available_devices: Iterable[str]) -> StackDevicePlan:
    """Build the default low-latency allocation plan.

    The plan intentionally spreads work across engines:
    ASR uses GPU first for measured Whisper turbo latency, Qwen3.5 2B uses GPU
    first for interactive turns, Qwen3.5 9B uses CPU first as a stable quality
    fallback, and
    MeloTTS keeps the acoustic model on CPU while using NPU/GPU for supported
    preprocessing and denoise subgraphs.
    """

    available = normalize_devices(available_devices)
    if not available:
        available = ("CPU",)

    return StackDevicePlan(
        asr=_choose(
            "asr_whisper_turbo",
            available,
            ("GPU", "CPU", "NPU"),
            "Whisper turbo is fastest on this host's GPU; CPU remains the stable first fallback.",
        ),
        llm_small=_choose(
            "llm_qwen35_2b",
            available,
            ("GPU", "CPU", "NPU"),
            "Qwen3.5 2B is the interactive model; GPU is fastest on this host and avoids the observed NPU compiler fallback delay.",
        ),
        llm_large=_choose(
            "llm_qwen35_9b",
            available,
            ("CPU", "GPU", "NPU"),
            "Qwen3.5 9B is the quality model; CPU loads reliably on this host while GPU compile can exceed the realtime budget.",
        ),
        tts=_choose(
            "tts_melo_acoustic",
            available,
            ("CPU", "GPU"),
            "MeloTTS acoustic model supports CPU/GPU, but CPU is the reliable default on this host; NPU is not supported.",
        ),
        tts_bert=_choose(
            "tts_melo_bert",
            available,
            ("CPU", "GPU", "NPU"),
            "MeloTTS BERT can support NPU, but this local model set lacks the static-shape BIN; CPU is the stable default.",
        ),
        tts_denoise=_choose(
            "tts_melo_deepfilter",
            available,
            ("NPU", "GPU", "CPU"),
            "DeepFilterNet supports NPU/GPU/CPU when compiled in; this keeps denoise off the ASR CPU path.",
        ),
    )


def format_device_plan(plan: StackDevicePlan) -> str:
    lines = ["OpenVINO voice stack device plan:"]
    for stage in plan.stages():
        fallback = " -> ".join(stage.fallback) if stage.fallback else "-"
        lines.append(f"- {stage.stage}: primary={stage.selected}, fallback={fallback}")
        lines.append(f"  reason: {stage.reason}")
    return "\n".join(lines)
