#!/usr/bin/env python3
"""Run a Piper ONNX voice through sherpa-onnx and write one WAV file."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any


DEFAULT_SILENCE_SCALE = 1.0


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(args: argparse.Namespace) -> str:
    if args.text_file is not None:
        return args.text_file.read_text(encoding="utf-8").strip()
    return args.text.strip()


def write_tokens(config: dict[str, Any], tokens_path: Path) -> None:
    phoneme_id_map = config["phoneme_id_map"]
    lines = [f"{symbol} {ids[0]}" for symbol, ids in phoneme_id_map.items()]
    tokens_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def metadata_dict(model: Any) -> dict[str, str]:
    return {item.key: item.value for item in model.metadata_props}


def set_metadata(model: Any, values: dict[str, Any]) -> bool:
    current = metadata_dict(model)
    changed = False
    for key, value in values.items():
        value_text = str(value)
        if current.get(key) == value_text:
            continue
        changed = True
        for item in model.metadata_props:
            if item.key == key:
                item.value = value_text
                break
        else:
            item = model.metadata_props.add()
            item.key = key
            item.value = value_text
    return changed


def ensure_sherpa_model(model_dir: Path, config: dict[str, Any]) -> tuple[Path, Path]:
    import onnx

    source_model = model_dir / "model.onnx"
    sherpa_model = model_dir / "model.sherpa.onnx"
    tokens_path = model_dir / "tokens.txt"

    if not source_model.exists():
        raise FileNotFoundError(f"Missing Piper ONNX model: {source_model}")
    if not sherpa_model.exists():
        shutil.copy2(source_model, sherpa_model)

    write_tokens(config, tokens_path)

    model = onnx.load(sherpa_model)
    changed = set_metadata(
        model,
        {
            "model_type": "vits",
            "comment": "piper",
            "language": config["language"]["name_english"],
            "voice": config["espeak"]["voice"],
            "has_espeak": 1,
            "n_speakers": config["num_speakers"],
            "sample_rate": config["audio"]["sample_rate"],
        },
    )
    if changed:
        onnx.save(model, sherpa_model)

    return sherpa_model, tokens_path


def build_tts(
    model_path: Path,
    tokens_path: Path,
    espeak_data_dir: Path,
    config_json: dict[str, Any],
    threads: int,
    silence_scale: float,
) -> Any:
    import sherpa_onnx

    inference = config_json["inference"]
    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(model_path),
                tokens=str(tokens_path),
                data_dir=str(espeak_data_dir),
                noise_scale=inference["noise_scale"],
                noise_scale_w=inference["noise_w"],
                length_scale=inference["length_scale"],
            ),
            provider="cpu",
            debug=False,
            num_threads=threads,
        ),
        max_num_sentences=1,
        silence_scale=silence_scale,
    )
    if not config.validate():
        raise RuntimeError(f"Invalid sherpa-onnx Piper config: {config}")
    return sherpa_onnx.OfflineTts(config)


def synthesize(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import sherpa_onnx
    import soundfile as sf

    model_dir = args.model_dir
    config_json = read_json(model_dir / "config.json")
    text = read_text(args)
    if not text:
        raise ValueError("Piper TTS text is empty")
    if args.espeak_data_dir is None:
        raise ValueError("--espeak-data-dir is required for Piper TTS")
    if args.silence_scale <= 0:
        raise ValueError("--silence-scale must be greater than 0")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    model_path, tokens_path = ensure_sherpa_model(model_dir, config_json)

    load_start = time.perf_counter()
    tts = build_tts(model_path, tokens_path, args.espeak_data_dir, config_json, args.threads, args.silence_scale)
    load_seconds = time.perf_counter() - load_start

    gen_config = sherpa_onnx.GenerationConfig()
    gen_config.sid = 0
    gen_config.speed = args.speed
    gen_config.silence_scale = args.silence_scale

    generate_start = time.perf_counter()
    audio = tts.generate(text, gen_config)
    generate_seconds = time.perf_counter() - generate_start
    if len(audio.samples) == 0:
        raise RuntimeError("Piper TTS generated empty audio")

    samples = np.asarray(audio.samples, dtype=np.float32)
    audio_seconds = len(samples) / audio.sample_rate
    sf.write(args.output_file, samples, samplerate=audio.sample_rate, subtype="PCM_16")

    return {
        "text": text,
        "output_file": str(args.output_file),
        "model_path": str(model_path),
        "tokens_path": str(tokens_path),
        "sample_rate": audio.sample_rate,
        "audio_seconds": audio_seconds,
        "speed": args.speed,
        "silence_scale": args.silence_scale,
        "load_seconds": load_seconds,
        "generate_seconds": generate_seconds,
        "rtf": generate_seconds / audio_seconds,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthesize one WAV with Piper zh_CN ONNX via sherpa-onnx.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--espeak-data-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--text")
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--silence-scale", type=float, default=DEFAULT_SILENCE_SCALE)
    parser.add_argument("--threads", type=int, default=4)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if bool(args.text) == bool(args.text_file):
        parser.error("Provide exactly one of --text or --text-file")
    payload = synthesize(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
