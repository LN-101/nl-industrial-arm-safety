"""Command-line smoke tests for local OpenVINO ASR and Qwen models."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import openvino as ov
import openvino_genai as ov_genai

from local_safety_assistant.config import (
    DEFAULT_ASR_ALIAS,
    DEFAULT_CACHE_DIR,
    DEFAULT_LLM_ALIAS,
    MODEL_ALIASES,
    MODELS_DIR,
    PROJECT_ROOT,
)


TEXT_MODEL_KINDS = {"llm", "vlm"}


@dataclass(frozen=True)
class ModelInfo:
    path: Path
    kind: str
    model_type: str
    architecture: str
    total_bin_bytes: int
    largest_bin_bytes: int
    openvino_xml_count: int
    exists: bool
    error: str = ""


def bytes_to_gib(value: int) -> float:
    return value / (1024**3)


def format_size(value: int) -> str:
    if value <= 0:
        return "-"
    if value >= 1024**3:
        return f"{bytes_to_gib(value):.2f} GiB"
    return f"{value / (1024**2):.1f} MiB"


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


def get_property(core: ov.Core, device: str, name: str) -> Any:
    try:
        return core.get_property(device, name)
    except Exception:
        return None


def supported_properties(core: ov.Core, device: str) -> dict[str, Any]:
    props = get_property(core, device, "SUPPORTED_PROPERTIES")
    return props if isinstance(props, dict) else {}


def resolve_model_path(value: str | Path) -> Path:
    raw = str(value)
    if raw in MODEL_ALIASES:
        return MODEL_ALIASES[raw].path.resolve()

    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()

    repo_relative = (PROJECT_ROOT / path).resolve()
    if repo_relative.exists():
        return repo_relative
    return path.resolve()


def alias_names_for_path(path: Path) -> list[str]:
    resolved = path.resolve()
    return [
        alias
        for alias, model_alias in MODEL_ALIASES.items()
        if model_alias.path.resolve() == resolved
    ]


def detect_model(path: Path, forced_kind: str = "auto") -> ModelInfo:
    path = path.expanduser().resolve()
    if not path.exists():
        return ModelInfo(path, "missing", "", "", 0, 0, 0, False, "path does not exist")

    config = read_json(path / "config.json")
    model_type = str(config.get("model_type") or "")
    architectures = config.get("architectures") or []
    architecture = ", ".join(str(item) for item in architectures)

    bin_files = sorted(path.glob("openvino*.bin"))
    xml_files = sorted(path.glob("openvino*.xml"))
    bin_sizes = [item.stat().st_size for item in bin_files if item.exists()]
    total_bin_bytes = sum(bin_sizes)
    largest_bin_bytes = max(bin_sizes) if bin_sizes else 0

    if forced_kind != "auto":
        kind = forced_kind
    elif model_type == "whisper" or (path / "openvino_encoder_model.xml").exists():
        kind = "asr"
    elif model_type == "qwen3_5" or (path / "openvino_vision_embeddings_model.xml").exists():
        kind = "vlm"
    elif (path / "openvino_language_model.xml").exists() or (path / "openvino_model.xml").exists():
        kind = "llm"
    else:
        kind = "unknown"

    return ModelInfo(
        path=path,
        kind=kind,
        model_type=model_type,
        architecture=architecture,
        total_bin_bytes=total_bin_bytes,
        largest_bin_bytes=largest_bin_bytes,
        openvino_xml_count=len(xml_files),
        exists=True,
    )


def model_inventory(models_dir: Path) -> list[ModelInfo]:
    candidates: dict[Path, ModelInfo] = {}
    for alias in MODEL_ALIASES.values():
        info = detect_model(alias.path)
        candidates[info.path] = info

    if models_dir.exists():
        for config_path in models_dir.rglob("config.json"):
            model_dir = config_path.parent.resolve()
            if any(model_dir.glob("openvino*.xml")):
                candidates[model_dir] = detect_model(model_dir)

    return sorted(candidates.values(), key=lambda item: str(item.path))


def print_inventory(models_dir: Path, json_output: bool) -> None:
    rows = model_inventory(models_dir)
    if json_output:
        print(
            json.dumps(
                [
                    {
                        **asdict(row),
                        "path": str(row.path),
                        "aliases": alias_names_for_path(row.path),
                        "total_bin_size": format_size(row.total_bin_bytes),
                        "largest_bin_size": format_size(row.largest_bin_bytes),
                    }
                    for row in rows
                ],
                indent=2,
                ensure_ascii=True,
            )
        )
        return

    print(f"Model inventory under {models_dir.resolve()}")
    for row in rows:
        aliases = ", ".join(alias_names_for_path(row.path)) or "-"
        status = "ok" if row.exists and row.kind != "unknown" else row.error or row.kind
        print(f"\n{row.path.relative_to(PROJECT_ROOT) if row.path.is_relative_to(PROJECT_ROOT) else row.path}")
        print(f"  aliases: {aliases}")
        print(f"  kind: {row.kind}")
        print(f"  model_type: {row.model_type or '-'}")
        print(f"  architecture: {row.architecture or '-'}")
        print(f"  OpenVINO XML files: {row.openvino_xml_count}")
        print(f"  total weights: {format_size(row.total_bin_bytes)}")
        print(f"  largest weight file: {format_size(row.largest_bin_bytes)}")
        print(f"  status: {status}")


def print_devices(core: ov.Core, json_output: bool = False) -> None:
    if json_output:
        devices = []
        for device in core.available_devices:
            devices.append(
                {
                    "name": device,
                    "full_name": get_property(core, device, "FULL_DEVICE_NAME"),
                    "type": str(get_property(core, device, "DEVICE_TYPE")),
                    "architecture": str(get_property(core, device, "DEVICE_ARCHITECTURE")),
                    "capabilities": get_property(core, device, "OPTIMIZATION_CAPABILITIES"),
                    "gpu_total_mem": get_property(core, device, "GPU_DEVICE_TOTAL_MEM_SIZE"),
                    "gpu_max_alloc_mem": get_property(core, device, "GPU_DEVICE_MAX_ALLOC_MEM_SIZE"),
                    "npu_total_mem": get_property(core, device, "NPU_DEVICE_TOTAL_MEM_SIZE"),
                    "npu_driver": get_property(core, device, "NPU_DRIVER_VERSION"),
                    "npu_compiler": get_property(core, device, "NPU_COMPILER_VERSION"),
                }
            )
        print(json.dumps(devices, indent=2, ensure_ascii=True))
        return

    print("OpenVINO:", ov.get_version())
    print("OpenVINO GenAI:", getattr(ov_genai, "__version__", "unknown"))
    print("Python:", sys.executable)
    print("Available devices:", ", ".join(core.available_devices) or "none")

    for device in core.available_devices:
        print(f"\n[{device}]")
        for key in (
            "FULL_DEVICE_NAME",
            "DEVICE_TYPE",
            "DEVICE_ARCHITECTURE",
            "OPTIMIZATION_CAPABILITIES",
            "NPU_DRIVER_VERSION",
            "NPU_COMPILER_VERSION",
            "NPU_DEVICE_TOTAL_MEM_SIZE",
            "GPU_DEVICE_TOTAL_MEM_SIZE",
            "GPU_DEVICE_MAX_ALLOC_MEM_SIZE",
        ):
            value = get_property(core, device, key)
            if value is None:
                continue
            if isinstance(value, int) and key.endswith(("MEM_SIZE", "ALLOC_MEM_SIZE")):
                print(f"{key}: {bytes_to_gib(value):.2f} GiB")
            else:
                print(f"{key}: {value}")


def generic_device_properties(core: ov.Core, device: str, cache_dir: Path) -> dict[str, Any]:
    props: dict[str, Any] = {}
    supported = supported_properties(core, device)

    if "CACHE_DIR" in supported:
        cache_dir.mkdir(parents=True, exist_ok=True)
        props["CACHE_DIR"] = str(cache_dir)

    if "PERFORMANCE_HINT" in supported:
        props["PERFORMANCE_HINT"] = "LATENCY"

    if device == "GPU":
        if "GPU_ENABLE_LARGE_ALLOCATIONS" in supported:
            props["GPU_ENABLE_LARGE_ALLOCATIONS"] = "YES"
        if "GPU_ENABLE_SDPA_OPTIMIZATION" in supported:
            props["GPU_ENABLE_SDPA_OPTIMIZATION"] = "YES"

    return props


def npu_text_properties(args: argparse.Namespace, cache_dir: Path) -> dict[str, Any]:
    (cache_dir / "npu").mkdir(parents=True, exist_ok=True)
    props: dict[str, Any] = {
        "CACHE_DIR": str(cache_dir / "npu"),
        "PERFORMANCE_HINT": "LATENCY",
        "MAX_PROMPT_LEN": args.max_prompt_len,
        "MIN_RESPONSE_LEN": min(args.min_response_len, args.max_new_tokens),
        "PREFILL_HINT": args.npu_prefill_hint,
        "GENERATE_HINT": args.npu_generate_hint,
    }

    if args.npu_compiler_type:
        props["NPU_COMPILER_TYPE"] = args.npu_compiler_type

    return props


def pipeline_properties(
    core: ov.Core,
    device: str,
    kind: str,
    cache_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if device == "NPU" and kind in TEXT_MODEL_KINDS:
        return npu_text_properties(args, cache_dir)
    return generic_device_properties(core, device, cache_dir / device.lower())


def gpu_is_safe_to_try(core: ov.Core, info: ModelInfo) -> tuple[bool, str]:
    max_alloc = get_property(core, "GPU", "GPU_DEVICE_MAX_ALLOC_MEM_SIZE")
    supported = supported_properties(core, "GPU")
    model_size = info.largest_bin_bytes

    if not model_size or not max_alloc:
        return True, ""

    if "GPU_ENABLE_LARGE_ALLOCATIONS" in supported:
        return True, ""

    required = int(model_size * 1.20)
    if required > max_alloc:
        return (
            False,
            "GPU single-allocation limit is too small "
            f"({format_size(max_alloc)} max allocation, {format_size(model_size)} largest weight file).",
        )
    return True, ""


def known_unsafe_for_npu(info: ModelInfo) -> str:
    name = info.path.name.lower()
    if "7b" in name and "nf4" in name:
        return (
            "This 7B NF4 model previously reset the NPU on this machine "
            "(intel_vpu TDR / DEVICE_LOST). Use --force-unstable-devices to try it anyway."
        )
    return ""


def candidate_devices(
    core: ov.Core,
    requested: str,
    info: ModelInfo,
    force_unstable_devices: bool,
) -> list[str]:
    available = set(core.available_devices)
    requested = requested.upper()

    if requested == "AUTO":
        if info.kind == "asr":
            devices = [device for device in ("CPU", "GPU", "NPU") if device in available]
        else:
            devices = [device for device in ("NPU", "GPU", "CPU") if device in available]
    else:
        if requested not in available:
            raise RuntimeError(
                f"Requested device {requested!r} is not available. "
                f"Available devices: {', '.join(core.available_devices) or 'none'}"
            )
        devices = [requested]
        if requested != "CPU" and "CPU" in available:
            devices.append("CPU")

    if "GPU" in devices and not force_unstable_devices:
        safe, reason = gpu_is_safe_to_try(core, info)
        if not safe:
            print(f"Skipping GPU: {reason}")
            devices.remove("GPU")

    if "NPU" in devices and not force_unstable_devices:
        reason = known_unsafe_for_npu(info)
        if reason:
            print(f"Skipping NPU: {reason}")
            devices.remove("NPU")

    return devices


def print_failure_hint(device: str, info: ModelInfo, error: Exception) -> None:
    message = str(error)
    first_line = next((line for line in message.splitlines() if line.strip()), repr(error))
    print(f"\n{device} failed: {type(error).__name__}: {first_line}")

    if info.kind == "vlm" and "input_ids" in message:
        print("Hint: this model exposes a VLM graph; use the auto/VLM pipeline, not LLMPipeline.")
    elif device == "NPU" and "StopLocationVerifierPass" in message:
        print("Hint: the NPU compiler rejected this graph; keep CPU/GPU fallback enabled for this model.")
    elif device == "NPU" and ("DEVICE_LOST" in message or "device hung" in message):
        print("Hint: the NPU runtime reset during inference. CPU fallback is the stable path.")
    elif device == "GPU":
        print("Hint: GPU failures on large INT4 models are usually allocation or compile-memory limits.")


def extract_generated_text(result: Any) -> str:
    if isinstance(result, str):
        return result

    texts = getattr(result, "texts", None)
    if texts:
        return "\n".join(str(item) for item in texts)

    text = getattr(result, "text", None)
    if text:
        return str(text)

    return str(result)


def run_text_generation(args: argparse.Namespace) -> int:
    model_path = resolve_model_path(args.model)
    info = detect_model(model_path, args.pipeline)
    cache_dir = args.cache_dir.expanduser().resolve()
    core = ov.Core()

    print_devices(core)

    if not info.exists:
        print(f"Model path does not exist: {info.path}", file=sys.stderr)
        return 2
    if info.kind not in TEXT_MODEL_KINDS:
        print(f"Model is {info.kind!r}, not a text generation model: {info.path}", file=sys.stderr)
        return 2

    if args.disable_npu_l0:
        os.environ["DISABLE_OPENVINO_GENAI_NPU_L0"] = "1"

    try:
        devices = candidate_devices(core, args.device, info, args.force_unstable_devices)
    except Exception as error:
        print(error, file=sys.stderr)
        return 2

    if not devices:
        print("No usable device found.", file=sys.stderr)
        return 2

    aliases = ", ".join(alias_names_for_path(info.path)) or "-"
    print(f"\nModel: {info.path}")
    print(f"Aliases: {aliases}")
    print(f"Detected kind: {info.kind}")
    print(f"Model type: {info.model_type or '-'}")
    print(f"Total weights: {format_size(info.total_bin_bytes)}")
    print(f"Largest weight file: {format_size(info.largest_bin_bytes)}")
    print("Run order:", " -> ".join(devices))

    errors: list[tuple[str, Exception]] = []
    for device in devices:
        try:
            run_text_generation_on_device(info, device, args.prompt, args.max_new_tokens, cache_dir, args)
            return 0
        except Exception as error:
            errors.append((device, error))
            print_failure_hint(device, info, error)
            if args.verbose:
                traceback.print_exc()

    print("\nAll candidate devices failed.", file=sys.stderr)
    for device, error in errors:
        print(f"- {device}: {type(error).__name__}: {error}", file=sys.stderr)
    return 1


def run_text_generation_on_device(
    info: ModelInfo,
    device: str,
    prompt: str,
    max_new_tokens: int,
    cache_dir: Path,
    args: argparse.Namespace,
) -> None:
    core = ov.Core()
    props = pipeline_properties(core, device, info.kind, cache_dir, args)

    print(f"\nLoading {info.kind.upper()} model on {device}...")
    if props:
        print("Device properties:", props)

    start_load = time.time()
    if info.kind == "vlm":
        pipe = ov_genai.VLMPipeline(info.path, device, **props)
    else:
        pipe = ov_genai.LLMPipeline(info.path, device, **props)
    print(f"Loaded on {device} in {time.time() - start_load:.2f}s")

    print(f"\nPrompt: {prompt}\n")
    token_count = 0
    first_token_time: float | None = None

    def streamer(subword: str) -> bool:
        nonlocal token_count, first_token_time
        if first_token_time is None:
            first_token_time = time.time()
        token_count += 1
        print(subword, end="", flush=True)
        return False

    start = time.time()
    result = pipe.generate(prompt, max_new_tokens=max_new_tokens, streamer=streamer)
    end = time.time()

    if token_count == 0:
        print(extract_generated_text(result), end="", flush=True)

    ttft = first_token_time - start if first_token_time else 0
    gen_time = end - first_token_time if first_token_time else end - start
    tps = token_count / gen_time if token_count and gen_time > 0 else 0

    print("\n\n--- Stats ---")
    print(f"Device: {device}")
    print(f"Pipeline: {info.kind}")
    print(f"Tokens streamed: {token_count}")
    print(f"TTFT: {ttft:.3f}s")
    print(f"Generation time: {gen_time:.3f}s")
    print(f"Tokens/s: {tps:.2f}")
    print(f"Total time: {end - start:.3f}s")


def load_wav_mono_16k(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Audio sample not found: {path}")

    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_rate != 16000:
        raise ValueError(
            f"Audio sample is {sample_rate} Hz. WhisperPipeline expects 16000 Hz audio; "
            "resample the WAV before running this smoke test."
        )

    if sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return np.clip(audio, -1.0, 1.0)


def run_asr(args: argparse.Namespace) -> int:
    model_path = resolve_model_path(args.model)
    info = detect_model(model_path)
    cache_dir = args.cache_dir.expanduser().resolve()
    core = ov.Core()

    print_devices(core)

    if not info.exists:
        print(f"ASR model path does not exist: {info.path}", file=sys.stderr)
        return 2
    if info.kind != "asr":
        print(f"Model is {info.kind!r}, not an ASR model: {info.path}", file=sys.stderr)
        return 2

    if args.audio_sample is None and not args.load_only:
        print("\nASR model path is valid, but no audio sample was provided.")
        print("Provide --audio-sample pointing to a 16 kHz PCM WAV file, or use --load-only.")
        return 2

    audio: np.ndarray | None = None
    if args.audio_sample is not None:
        try:
            audio = load_wav_mono_16k(args.audio_sample.expanduser().resolve())
        except Exception as error:
            print(f"Invalid audio sample: {error}", file=sys.stderr)
            return 2

    try:
        devices = candidate_devices(core, args.device, info, args.force_unstable_devices)
    except Exception as error:
        print(error, file=sys.stderr)
        return 2

    if not devices:
        print("No usable device found.", file=sys.stderr)
        return 2

    print(f"\nASR model: {info.path}")
    print(f"Detected kind: {info.kind}")
    print(f"Total weights: {format_size(info.total_bin_bytes)}")
    print("Run order:", " -> ".join(devices))

    errors: list[tuple[str, Exception]] = []
    for device in devices:
        try:
            run_asr_on_device(info, device, audio, cache_dir, args)
            return 0
        except Exception as error:
            errors.append((device, error))
            print_failure_hint(device, info, error)
            if args.verbose:
                traceback.print_exc()

    print("\nAll ASR candidate devices failed.", file=sys.stderr)
    for device, error in errors:
        print(f"- {device}: {type(error).__name__}: {error}", file=sys.stderr)
    return 1


def run_asr_on_device(
    info: ModelInfo,
    device: str,
    audio: np.ndarray | None,
    cache_dir: Path,
    args: argparse.Namespace,
) -> None:
    core = ov.Core()
    props = pipeline_properties(core, device, "asr", cache_dir, args)

    print(f"\nLoading Whisper model on {device}...")
    if props:
        print("Device properties:", props)

    start_load = time.time()
    pipe = ov_genai.WhisperPipeline(info.path, device, **props)
    print(f"Loaded on {device} in {time.time() - start_load:.2f}s")

    if args.load_only:
        print("Load-only ASR smoke test passed.")
        return

    assert audio is not None
    kwargs: dict[str, Any] = {}
    if args.language:
        language = args.language
        if not language.startswith("<|"):
            language = f"<|{language}|>"
        kwargs["language"] = language
    if args.return_timestamps:
        kwargs["return_timestamps"] = True

    start = time.time()
    result = pipe.generate(audio.tolist(), **kwargs)
    end = time.time()

    print("\n--- Transcript ---")
    print(extract_generated_text(result))
    print("\n--- Stats ---")
    print(f"Device: {device}")
    print(f"Audio samples: {len(audio)}")
    print(f"Audio seconds: {len(audio) / 16000:.2f}")
    print(f"Total time: {end - start:.3f}s")


def add_text_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", "--model-path", default=DEFAULT_LLM_ALIAS, help="Model alias or local path.")
    parser.add_argument("--device", default="AUTO", help="AUTO, CPU, GPU, or NPU.")
    parser.add_argument("--prompt", default="Explain what OpenVINO is in one sentence.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-prompt-len", type=int, default=128)
    parser.add_argument("--min-response-len", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--pipeline", choices=("auto", "llm", "vlm"), default="auto")
    parser.add_argument("--force-unstable-devices", action="store_true")
    parser.add_argument("--disable-npu-l0", action="store_true")
    parser.add_argument("--npu-compiler-type", choices=("DRIVER", "PREFER_PLUGIN"))
    parser.add_argument("--npu-prefill-hint", default="STATIC", choices=("STATIC", "DYNAMIC"))
    parser.add_argument("--npu-generate-hint", default="FAST_COMPILE", choices=("BEST_PERF", "FAST_COMPILE"))
    parser.add_argument("--verbose", action="store_true")


def add_asr_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", "--model-path", default=DEFAULT_ASR_ALIAS, help="ASR model alias or local path.")
    parser.add_argument("--device", default="AUTO", help="AUTO, CPU, GPU, or NPU. ASR AUTO prefers CPU first.")
    parser.add_argument("--audio-sample", type=Path, help="16 kHz PCM WAV file to transcribe.")
    parser.add_argument("--language", help="Whisper language token, e.g. zh or en.")
    parser.add_argument("--return-timestamps", action="store_true")
    parser.add_argument("--load-only", action="store_true", help="Load the ASR pipeline without transcribing audio.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--pipeline", choices=("auto", "asr"), default="auto")
    parser.add_argument("--force-unstable-devices", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-prompt-len", type=int, default=128)
    parser.add_argument("--min-response-len", type=int, default=4)
    parser.add_argument("--npu-compiler-type", choices=("DRIVER", "PREFER_PLUGIN"))
    parser.add_argument("--npu-prefill-hint", default="STATIC", choices=("STATIC", "DYNAMIC"))
    parser.add_argument("--npu-generate-hint", default="FAST_COMPILE", choices=("BEST_PERF", "FAST_COMPILE"))
    parser.add_argument("--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local OpenVINO model inventory and smoke tests.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    devices = subparsers.add_parser("devices", help="List OpenVINO devices.")
    devices.add_argument("--json", action="store_true")

    inventory = subparsers.add_parser("inventory", help="List local OpenVINO model inventory.")
    inventory.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    inventory.add_argument("--devices", action="store_true", help="Print devices before inventory.")
    inventory.add_argument("--json", action="store_true")

    generate = subparsers.add_parser("generate", help="Run a text-generation smoke test.")
    add_text_generation_args(generate)

    asr = subparsers.add_parser("asr", help="Run an ASR smoke test.")
    add_asr_args(asr)

    return parser


def legacy_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Compatibility wrapper for text generation smoke tests.")
    add_text_generation_args(parser)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    core = ov.Core()
    if args.list_devices:
        print_devices(core, json_output=args.json)
        return 0

    return run_text_generation(args)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {"devices", "inventory", "generate", "asr"}

    if not argv or argv[0] not in commands:
        return legacy_main(argv)

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "devices":
        print_devices(ov.Core(), json_output=args.json)
        return 0

    if args.command == "inventory":
        if args.devices and not args.json:
            print_devices(ov.Core())
            print()
        print_inventory(args.models_dir, args.json)
        return 0

    if args.command == "generate":
        return run_text_generation(args)

    if args.command == "asr":
        return run_asr(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
