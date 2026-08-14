#!/usr/bin/env python3
"""Benchmark local OpenVINO Whisper ASR models on FLEURS Chinese samples."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import tarfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import openvino as ov
import openvino_genai as ov_genai
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "benchmarks" / "asr_fleurs_cmn_hans_cn"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "asr_results"
DEFAULT_MODELS = {
    "large-v3-int4": PROJECT_ROOT / "models" / "asr" / "whisper-large-v3-int4-ov",
    "turbo-int4": PROJECT_ROOT / "models" / "asr" / "whisper-large-v3-turbo-int4-ov",
}
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".ov_cache" / "asr_benchmark"
PUNCTUATION_RE = re.compile(r"[\s\.,!?;:'\"，。！？；：“”‘’、（）()\[\]【】《》<>…—\-·|]+")
HAN_RE = re.compile(r"[\u4e00-\u9fff]")


@dataclass(frozen=True)
class Sample:
    sample_id: str
    filename: str
    transcript: str
    num_samples: int
    gender: str
    audio_path: Path


@dataclass
class SampleResult:
    sample_id: str
    filename: str
    gender: str
    audio_seconds: float
    reference: str
    prediction: str
    normalized_reference: str
    normalized_prediction: str
    han_reference: str
    han_prediction: str
    char_errors: int
    reference_chars: int
    cer: float
    han_char_errors: int
    han_reference_chars: int
    han_cer: float
    inference_seconds: float
    rtf: float


@dataclass
class ModelResult:
    name: str
    path: str
    device: str
    load_seconds: float
    total_inference_seconds: float
    total_audio_seconds: float
    rtf: float
    cer: float
    han_cer: float
    mean_sample_seconds: float
    median_sample_seconds: float
    samples: list[SampleResult]


def normalize_chinese(text: str) -> str:
    return PUNCTUATION_RE.sub("", text).lower()


def han_only(text: str) -> str:
    return "".join(HAN_RE.findall(text))


def edit_distance(reference: str, prediction: str) -> int:
    previous = list(range(len(prediction) + 1))
    for row_index, ref_char in enumerate(reference, start=1):
        current = [row_index]
        for col_index, pred_char in enumerate(prediction, start=1):
            substitution = previous[col_index - 1] + (ref_char != pred_char)
            insertion = current[col_index - 1] + 1
            deletion = previous[col_index] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


def safe_extract_member(tar_file: tarfile.TarFile, member: tarfile.TarInfo, destination: Path) -> None:
    target = (destination / member.name).resolve()
    destination = destination.resolve()
    if not target.is_relative_to(destination):
        raise ValueError(f"Refusing to extract unsafe tar member: {member.name}")
    tar_file.extract(member, destination)


def parse_samples(dataset_dir: Path) -> list[dict[str, str]]:
    tsv_path = dataset_dir / "dev.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(f"Missing transcript file: {tsv_path}")

    rows: list[dict[str, str]] = []
    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for raw in reader:
            if len(raw) < 7:
                continue
            rows.append(
                {
                    "sample_id": raw[0],
                    "filename": raw[1],
                    "transcript": raw[2],
                    "num_samples": raw[5],
                    "gender": raw[6],
                }
            )
    return rows


def choose_rows(rows: list[dict[str, str]], count: int, max_audio_seconds: float) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    seen_transcripts: set[str] = set()
    for row in rows:
        transcript = row["transcript"]
        if transcript in seen_transcripts:
            continue
        audio_seconds = int(row["num_samples"]) / 16000.0
        if audio_seconds > max_audio_seconds:
            continue
        selected.append(row)
        seen_transcripts.add(transcript)
        if len(selected) >= count:
            break
    if len(selected) < count:
        raise RuntimeError(
            f"Only selected {len(selected)} samples under {max_audio_seconds:.1f}s; "
            f"requested {count}."
        )
    return selected


def prepare_samples(dataset_dir: Path, count: int, max_audio_seconds: float) -> list[Sample]:
    rows = choose_rows(parse_samples(dataset_dir), count, max_audio_seconds)
    sample_dir = dataset_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    missing = [row for row in rows if not (sample_dir / row["filename"]).exists()]
    if missing:
        tar_path = dataset_dir / "dev.tar.gz"
        if not tar_path.exists():
            raise FileNotFoundError(f"Missing audio archive: {tar_path}")
        wanted = {f"dev/{row['filename']}" for row in missing}
        with tarfile.open(tar_path, "r:gz") as tar_file:
            members = [member for member in tar_file.getmembers() if member.name in wanted]
            found = {member.name for member in members}
            missing_names = sorted(wanted - found)
            if missing_names:
                raise FileNotFoundError(f"Archive is missing expected samples: {missing_names}")
            for member in members:
                safe_extract_member(tar_file, member, sample_dir)
                extracted = sample_dir / member.name
                extracted.rename(sample_dir / Path(member.name).name)

    return [
        Sample(
            sample_id=row["sample_id"],
            filename=row["filename"],
            transcript=row["transcript"],
            num_samples=int(row["num_samples"]),
            gender=row["gender"],
            audio_path=sample_dir / row["filename"],
        )
        for row in rows
    ]


def load_audio(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if sample_rate != 16000:
        raise ValueError(f"{path} is {sample_rate} Hz, expected 16000 Hz")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.clip(audio, -1.0, 1.0)


def extract_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    text = getattr(result, "text", None)
    if text:
        return str(text)
    texts = getattr(result, "texts", None)
    if texts:
        return "\n".join(str(item) for item in texts)
    return str(result)


def device_properties(device: str, cache_dir: Path) -> dict[str, Any]:
    core = ov.Core()
    supported = core.get_property(device, "SUPPORTED_PROPERTIES")
    props: dict[str, Any] = {}
    if isinstance(supported, dict) and "CACHE_DIR" in supported:
        device_cache = cache_dir / device.lower()
        device_cache.mkdir(parents=True, exist_ok=True)
        props["CACHE_DIR"] = str(device_cache)
    if isinstance(supported, dict) and "PERFORMANCE_HINT" in supported:
        props["PERFORMANCE_HINT"] = "LATENCY"
    if device == "GPU" and isinstance(supported, dict):
        if "GPU_ENABLE_LARGE_ALLOCATIONS" in supported:
            props["GPU_ENABLE_LARGE_ALLOCATIONS"] = "YES"
        if "GPU_ENABLE_SDPA_OPTIMIZATION" in supported:
            props["GPU_ENABLE_SDPA_OPTIMIZATION"] = "YES"
    return props


def benchmark_model(
    name: str,
    model_path: Path,
    device: str,
    samples: list[Sample],
    language: str,
    cache_dir: Path,
) -> ModelResult:
    if not model_path.exists():
        raise FileNotFoundError(f"Model does not exist: {model_path}")

    props = device_properties(device, cache_dir)
    start_load = time.perf_counter()
    pipe = ov_genai.WhisperPipeline(model_path, device, **props)
    load_seconds = time.perf_counter() - start_load

    if not language.startswith("<|"):
        language = f"<|{language}|>"

    sample_results: list[SampleResult] = []
    for sample in samples:
        audio = load_audio(sample.audio_path)
        audio_seconds = len(audio) / 16000.0
        start = time.perf_counter()
        result = pipe.generate(audio.tolist(), language=language)
        inference_seconds = time.perf_counter() - start
        prediction = extract_text(result)
        normalized_reference = normalize_chinese(sample.transcript)
        normalized_prediction = normalize_chinese(prediction)
        han_reference = han_only(sample.transcript)
        han_prediction = han_only(prediction)
        errors = edit_distance(normalized_reference, normalized_prediction)
        reference_chars = len(normalized_reference)
        han_errors = edit_distance(han_reference, han_prediction)
        han_reference_chars = len(han_reference)
        sample_results.append(
            SampleResult(
                sample_id=sample.sample_id,
                filename=sample.filename,
                gender=sample.gender,
                audio_seconds=audio_seconds,
                reference=sample.transcript,
                prediction=prediction,
                normalized_reference=normalized_reference,
                normalized_prediction=normalized_prediction,
                han_reference=han_reference,
                han_prediction=han_prediction,
                char_errors=errors,
                reference_chars=reference_chars,
                cer=errors / reference_chars if reference_chars else 0.0,
                han_char_errors=han_errors,
                han_reference_chars=han_reference_chars,
                han_cer=han_errors / han_reference_chars if han_reference_chars else 0.0,
                inference_seconds=inference_seconds,
                rtf=inference_seconds / audio_seconds if audio_seconds else 0.0,
            )
        )

    total_audio_seconds = sum(item.audio_seconds for item in sample_results)
    total_inference_seconds = sum(item.inference_seconds for item in sample_results)
    total_errors = sum(item.char_errors for item in sample_results)
    total_reference_chars = sum(item.reference_chars for item in sample_results)
    total_han_errors = sum(item.han_char_errors for item in sample_results)
    total_han_reference_chars = sum(item.han_reference_chars for item in sample_results)
    sample_times = [item.inference_seconds for item in sample_results]

    return ModelResult(
        name=name,
        path=str(model_path),
        device=device,
        load_seconds=load_seconds,
        total_inference_seconds=total_inference_seconds,
        total_audio_seconds=total_audio_seconds,
        rtf=total_inference_seconds / total_audio_seconds if total_audio_seconds else 0.0,
        cer=total_errors / total_reference_chars if total_reference_chars else 0.0,
        han_cer=total_han_errors / total_han_reference_chars if total_han_reference_chars else 0.0,
        mean_sample_seconds=statistics.mean(sample_times),
        median_sample_seconds=statistics.median(sample_times),
        samples=sample_results,
    )


def write_markdown(results: list[ModelResult], output_path: Path, dataset_dir: Path) -> None:
    lines = [
        "# ASR Benchmark Results",
        "",
        f"Dataset: Google FLEURS `cmn_hans_cn` dev split under `{dataset_dir}`.",
        "",
        "| Model | Device | Load s | Audio s | Inference s | RTF | Mixed CER | Han CER | Mean sample s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            "| "
            f"{result.name} | {result.device} | {result.load_seconds:.2f} | "
            f"{result.total_audio_seconds:.2f} | {result.total_inference_seconds:.2f} | "
            f"{result.rtf:.3f} | {result.cer:.3%} | {result.han_cer:.3%} | "
            f"{result.mean_sample_seconds:.2f} |"
        )
    lines.extend(["", "## Sample Outputs", ""])
    for result in results:
        lines.extend([f"### {result.name}", ""])
        for sample in result.samples[:5]:
            lines.append(
                f"- `{sample.filename}` mixed CER {sample.cer:.1%}, "
                f"Han CER {sample.han_cer:.1%}, {sample.inference_seconds:.2f}s"
            )
            lines.append(f"  - ref: {sample.reference}")
            lines.append(f"  - hyp: {sample.prediction}")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_model_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value).expanduser()
        return path.name, path
    name, raw_path = value.split("=", 1)
    return name, Path(raw_path).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--device", default="CPU", help="OpenVINO device, e.g. CPU or GPU.")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--max-audio-seconds", type=float, default=12.0)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--model",
        action="append",
        help="Model spec as name=path. Defaults to large-v3-int4 and turbo-int4.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.model:
        models = [parse_model_arg(value) for value in args.model]
    else:
        models = [(name, path) for name, path in DEFAULT_MODELS.items()]

    samples = prepare_samples(dataset_dir, args.sample_count, args.max_audio_seconds)
    results = [
        benchmark_model(name, path.expanduser().resolve(), args.device.upper(), samples, args.language, args.cache_dir)
        for name, path in models
    ]

    payload = {
        "dataset": {
            "name": "google/fleurs",
            "config": "cmn_hans_cn",
            "split": "dev",
            "license": "cc-by-4.0",
            "dataset_dir": str(dataset_dir),
            "sample_count": len(samples),
        },
        "device": args.device.upper(),
        "language": args.language,
        "models": [asdict(result) for result in results],
    }
    json_path = results_dir / f"fleurs_cmn_hans_cn_{args.device.lower()}_{len(samples)}.json"
    md_path = results_dir / f"fleurs_cmn_hans_cn_{args.device.lower()}_{len(samples)}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(results, md_path, dataset_dir)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    for result in results:
        print(
            f"{result.name}: mixed_CER={result.cer:.3%}, Han_CER={result.han_cer:.3%}, RTF={result.rtf:.3f}, "
            f"inference={result.total_inference_seconds:.2f}s/audio={result.total_audio_seconds:.2f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
