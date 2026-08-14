"""Batch voice safety testing for the OpenVINO robot-arm assistant."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal

from local_safety_assistant.config import PROJECT_ROOT
from local_safety_assistant.stack.asr import WhisperAsrEngine
from local_safety_assistant.stack.config import GenerationConfig, MeloTtsConfig, VoiceStackConfig
from local_safety_assistant.stack.devices import available_openvino_devices, build_device_plan
from local_safety_assistant.stack.llm import QwenLlmEngine
from local_safety_assistant.stack.pipeline import VoicePipeline, VoiceTurnResult
from local_safety_assistant.stack.tts import MeloTtsBridge


DEFAULT_RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "voice_safety_batch"
TARGET_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class KeywordGroup:
    label: str
    any_of: tuple[str, ...]


@dataclass(frozen=True)
class SafetyVoiceCase:
    case_id: str
    utterance: str
    expected_behavior: str
    checks: tuple[KeywordGroup, ...]


@dataclass(frozen=True)
class EvaluationResult:
    passed: bool
    matched: dict[str, str]
    missing: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class BatchPaths:
    output_dir: Path
    raw_audio_dir: Path
    input_audio_dir: Path
    output_audio_dir: Path
    json_path: Path
    csv_path: Path
    report_path: Path


STOP_WORDS = ("停止", "急停", "停机", "暂停", "停下")
SPEED_WORDS = ("限速", "减速", "低速", "速度")
CONFIRM_WORDS = ("确认", "检查", "复位", "安全")
ALARM_WORDS = ("报警", "告警", "异常", "故障")
AIR_PUMP_WORDS = ("气泵", "吸取", "释放", "关闭", "抓取")


DEFAULT_CASES: tuple[SafetyVoiceCase, ...] = (
    SafetyVoiceCase(
        "S01",
        "有人进入机械臂安全区域，应该怎么办？",
        "立即停止机械臂，并要求人员离开安全区后再复位。",
        (KeywordGroup("停机动作", STOP_WORDS), KeywordGroup("人员/安全区", ("人员", "人", "安全区", "安全区域"))),
    ),
    SafetyVoiceCase(
        "S02",
        "未知物体靠近抓取路径时应该怎么办？",
        "未知物体靠近时应降低机械臂速度并等待路径确认。",
        (KeywordGroup("限速动作", SPEED_WORDS), KeywordGroup("未知物体", ("未知物体", "物体", "路径"))),
    ),
    SafetyVoiceCase(
        "S03",
        "ROS 控制器报警时系统应该怎么处理？",
        "控制器报警时应进入安全保持状态并通知操作员检查。",
        (KeywordGroup("安全保持", ("安全保持", "保持", "停止")), KeywordGroup("控制器报警", ("ROS", "控制器", "报警"))),
    ),
    SafetyVoiceCase(
        "S04",
        "急停按钮已经按下，下一步怎么处理？",
        "保持急停，排查风险，确认安全后复位。",
        (KeywordGroup("急停停机", ("急停", "停止", "停机")), KeywordGroup("确认复位", CONFIRM_WORDS)),
    ),
    SafetyVoiceCase(
        "S05",
        "复位机械臂之前需要确认哪些安全条件？",
        "复位前确认人员离开、区域清空、报警解除。",
        (KeywordGroup("确认检查", ("确认", "检查")), KeywordGroup("安全条件", ("安全", "人员", "报警", "区域"))),
    ),
    SafetyVoiceCase(
        "S06",
        "示教模式下机械臂速度应该如何限制？",
        "示教模式应低速或限速运行。",
        (KeywordGroup("速度限制", SPEED_WORDS), KeywordGroup("示教", ("示教", "手动", "模式"))),
    ),
    SafetyVoiceCase(
        "S07",
        "气泵可能吸到操作员手指时应该怎么做？",
        "立即停止并释放或禁止气泵继续吸取。",
        (KeywordGroup("停机动作", STOP_WORDS), KeywordGroup("气泵动作", AIR_PUMP_WORDS)),
    ),
    SafetyVoiceCase(
        "S08",
        "未知物体靠近抓取路径时机械臂要怎么反应？",
        "减速或停止，重新确认抓取路径。",
        (KeywordGroup("降速停机", STOP_WORDS + SPEED_WORDS), KeywordGroup("障碍/路径", ("物体", "障碍", "路径", "抓取"))),
    ),
    SafetyVoiceCase(
        "S09",
        "协作区域有人靠近，机械臂应该限速吗？",
        "有人靠近协作区时应限速、减速或停止。",
        (KeywordGroup("速度限制", SPEED_WORDS + STOP_WORDS), KeywordGroup("协作人员", ("人", "人员", "协作", "靠近"))),
    ),
    SafetyVoiceCase(
        "S10",
        "更换夹具之前需要先做什么安全动作？",
        "更换夹具前停止机械臂并确认断能或锁定。",
        (KeywordGroup("停机动作", STOP_WORDS), KeywordGroup("确认锁定", ("确认", "锁定", "断电", "断能", "夹具"))),
    ),
    SafetyVoiceCase(
        "S11",
        "机械臂运行中出现异常振动，该不该停机检查？",
        "异常振动应停机并检查原因。",
        (KeywordGroup("停机动作", STOP_WORDS), KeywordGroup("异常检查", ALARM_WORDS + ("振动", "检查"))),
    ),
    SafetyVoiceCase(
        "S12",
        "机械臂超过安全速度限制时如何处理？",
        "触发安全限速或停止，确认速度参数。",
        (KeywordGroup("速度限制", SPEED_WORDS), KeywordGroup("确认检查", CONFIRM_WORDS)),
    ),
    SafetyVoiceCase(
        "S13",
        "突然掉电后恢复供电，机械臂能直接启动吗？",
        "不能直接启动，需要检查状态并确认安全后复位。",
        (KeywordGroup("禁止直接启动", ("不能", "禁止", "不要", "确认")), KeywordGroup("确认复位", CONFIRM_WORDS)),
    ),
    SafetyVoiceCase(
        "S14",
        "机械臂负载超过额定值，系统应该提示什么？",
        "超载应停止或报警，并检查负载配置。",
        (KeywordGroup("停机报警", STOP_WORDS + ALARM_WORDS), KeywordGroup("负载", ("负载", "超载", "额定"))),
    ),
    SafetyVoiceCase(
        "S15",
        "关节温度过高时机械臂还能继续工作吗？",
        "温度过高应停机、降温并检查。",
        (KeywordGroup("停机动作", STOP_WORDS), KeywordGroup("温度检查", ("温度", "过热", "降温", "检查"))),
    ),
    SafetyVoiceCase(
        "S16",
        "工件从气泵吸附中滑落时应该先做什么？",
        "工件滑落时停止运动，检查气泵和工件状态。",
        (KeywordGroup("停机动作", STOP_WORDS), KeywordGroup("气泵工件", AIR_PUMP_WORDS + ("工件", "滑落"))),
    ),
    SafetyVoiceCase(
        "S17",
        "维护人员进入机械臂工作区时应该执行什么安全流程？",
        "维护进入前停止设备，确认安全，执行锁定挂牌。",
        (KeywordGroup("停机动作", STOP_WORDS), KeywordGroup("维护确认", ("维护", "人员", "确认", "锁定", "挂牌"))),
    ),
    SafetyVoiceCase(
        "S18",
        "重新启动自动运行前，需要检查安全区域吗？",
        "自动运行前必须检查并确认安全区域清空。",
        (KeywordGroup("确认检查", ("确认", "检查")), KeywordGroup("安全区域", ("安全区", "安全区域", "区域", "人员"))),
    ),
    SafetyVoiceCase(
        "S19",
        "人员距离机械臂小于安全阈值时系统要做什么？",
        "人员距离过近时停止或限速。",
        (KeywordGroup("安全动作", STOP_WORDS + SPEED_WORDS), KeywordGroup("距离人员", ("人员", "距离", "安全", "阈值"))),
    ),
    SafetyVoiceCase(
        "S20",
        "解除速度限制之前要满足什么条件？",
        "解除限速前确认安全、授权和风险消除。",
        (KeywordGroup("速度限制", SPEED_WORDS), KeywordGroup("确认授权", ("确认", "安全", "授权", "风险"))),
    ),
    SafetyVoiceCase(
        "S21",
        "ROS 控制器报警时机械臂安全助手应该怎么回答？",
        "控制器报警时停止动作并检查报警原因。",
        (KeywordGroup("停机报警", STOP_WORDS + ALARM_WORDS), KeywordGroup("控制器", ("ROS", "控制器", "报警", "告警"))),
    ),
    SafetyVoiceCase(
        "S22",
        "检测到关节力矩异常，是否需要停止机械臂？",
        "力矩异常需要停止并检查机械结构或碰撞风险。",
        (KeywordGroup("停机动作", STOP_WORDS), KeywordGroup("力矩异常", ("力矩", "异常", "碰撞", "检查"))),
    ),
    SafetyVoiceCase(
        "S23",
        "规划轨迹可能撞到治具时应该怎么办？",
        "存在碰撞风险时停止执行并重新规划。",
        (KeywordGroup("停机动作", STOP_WORDS), KeywordGroup("碰撞轨迹", ("碰撞", "撞", "轨迹", "规划", "治具"))),
    ),
    SafetyVoiceCase(
        "S24",
        "相机看不清安全区域，机械臂能自动运行吗？",
        "视觉不可用时不应自动运行，需要暂停并检查。",
        (KeywordGroup("禁止运行", STOP_WORDS + ("不能", "不要", "禁止")), KeywordGroup("视觉检查", ("相机", "视觉", "检查", "安全区"))),
    ),
    SafetyVoiceCase(
        "S25",
        "气泵吸取前需要确认人员和工件安全吗？",
        "气泵吸取前确认人员远离并确认工件位置。",
        (KeywordGroup("确认检查", ("确认", "检查")), KeywordGroup("气泵人员", AIR_PUMP_WORDS + ("人员", "工件", "安全"))),
    ),
    SafetyVoiceCase(
        "S26",
        "教学器使能开关松开后机械臂应该停止吗？",
        "使能丢失后机械臂应停止。",
        (KeywordGroup("停机动作", STOP_WORDS), KeywordGroup("使能教学器", ("使能", "教学器", "开关"))),
    ),
    SafetyVoiceCase(
        "S27",
        "自动运行开始之前，请提醒我安全检查步骤。",
        "自动运行前清场，确认安全区域、夹具和报警状态。",
        (KeywordGroup("确认检查", ("确认", "检查", "清场")), KeywordGroup("安全条件", ("安全", "区域", "报警", "夹具"))),
    ),
    SafetyVoiceCase(
        "S28",
        "机械臂出现异常噪声时应该继续完成任务吗？",
        "异常噪声时不要继续任务，应停止并检查。",
        (KeywordGroup("禁止继续", STOP_WORDS + ("不能", "不要", "禁止")), KeywordGroup("异常检查", ALARM_WORDS + ("噪声", "检查"))),
    ),
    SafetyVoiceCase(
        "S29",
        "控制通信丢失时系统应该让机械臂保持什么状态？",
        "通信丢失应进入停止或安全保持状态。",
        (KeywordGroup("安全停机", STOP_WORDS + ("保持", "安全")), KeywordGroup("通信", ("通信", "控制", "丢失"))),
    ),
    SafetyVoiceCase(
        "S30",
        "请简要说明机械臂安全助手的急停和限速规则。",
        "回答应说明急停和限速的安全规则。",
        (KeywordGroup("急停", ("急停", "停止", "停机")), KeywordGroup("限速", SPEED_WORDS)),
    ),
    SafetyVoiceCase(
        "S31",
        "当前安全规则是什么？",
        "回答应读取当前 JSON 规则，完整但简洁地覆盖所有启用规则，不输出后续建议。",
        (
            KeywordGroup("人员规则", ("人员侵入停机规则", "人员距离", "人员")),
            KeywordGroup("未知物体规则", ("未知物体靠近限速规则", "未知物体距离", "未知物体")),
            KeywordGroup("控制器报警规则", ("ROS 控制器", "控制器报警", "安全保持")),
            KeywordGroup("人员阈值", ("1.0", "1 米", "一米")),
            KeywordGroup("未知物体阈值", ("0.25", "25%", "百分之二十五")),
        ),
    ),
    SafetyVoiceCase(
        "S32",
        "请详细说明未知物体靠近限速规则。",
        "回答应只解释 JSON 中的未知物体靠近限速规则，包括启用、触发距离、限速和通知。",
        (
            KeywordGroup("未知物体规则", ("未知物体靠近限速规则", "未知物体")),
            KeywordGroup("启用状态", ("启用", "生效", "开启")),
            KeywordGroup("触发条件", ("0.25", "25", "靠近", "距离")),
            KeywordGroup("限速动作", SPEED_WORDS),
            KeywordGroup("通知确认", ("通知", "确认")),
        ),
    ),
)


def evaluate_response(text: str, checks: tuple[KeywordGroup, ...]) -> EvaluationResult:
    matched: dict[str, str] = {}
    missing: dict[str, tuple[str, ...]] = {}
    for group in checks:
        match = next((keyword for keyword in group.any_of if keyword in text), "")
        if match:
            matched[group.label] = match
        else:
            missing[group.label] = group.any_of
    return EvaluationResult(passed=not missing, matched=matched, missing=missing)


def build_paths(output_dir: Path) -> BatchPaths:
    raw_audio_dir = output_dir / "audio_raw_44k"
    input_audio_dir = output_dir / "audio_input_16k"
    output_audio_dir = output_dir / "audio_output"
    return BatchPaths(
        output_dir=output_dir,
        raw_audio_dir=raw_audio_dir,
        input_audio_dir=input_audio_dir,
        output_audio_dir=output_audio_dir,
        json_path=output_dir / "results.json",
        csv_path=output_dir / "results.csv",
        report_path=output_dir / "report.md",
    )


def prepare_dirs(paths: BatchPaths) -> None:
    for path in (paths.output_dir, paths.raw_audio_dir, paths.input_audio_dir, paths.output_audio_dir):
        path.mkdir(parents=True, exist_ok=True)


def convert_to_mono_16k(source_path: Path, target_path: Path) -> tuple[float, float, int]:
    started = time.perf_counter()
    audio, sample_rate = sf.read(source_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sample_rate != TARGET_SAMPLE_RATE:
        gcd = math.gcd(sample_rate, TARGET_SAMPLE_RATE)
        audio = signal.resample_poly(audio, TARGET_SAMPLE_RATE // gcd, sample_rate // gcd)

    audio = np.clip(audio, -1.0, 1.0)
    sf.write(target_path, audio, TARGET_SAMPLE_RATE, subtype="PCM_16")
    elapsed = time.perf_counter() - started
    return len(audio) / TARGET_SAMPLE_RATE, elapsed, int(sample_rate)


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    paths = build_paths(output_dir)
    prepare_dirs(paths)

    cases = DEFAULT_CASES[: args.count]
    if len(cases) < args.count:
        raise ValueError(f"Only {len(DEFAULT_CASES)} safety cases are defined; requested {args.count}.")

    config = VoiceStackConfig()
    plan = build_device_plan(available_openvino_devices())
    input_tts = MeloTtsBridge(
        MeloTtsConfig(
            binary=args.tts_binary,
            model_dir=args.tts_model_dir,
            output_dir=paths.raw_audio_dir,
            language=args.tts_language,
            speed=args.tts_speed,
            disable_bert=args.disable_tts_bert,
            disable_nf=args.disable_tts_denoise,
            timeout_seconds=args.tts_timeout,
        ),
        tts_device=plan.tts.selected,
        bert_device=plan.tts_bert.selected,
        denoise_device=plan.tts_denoise.selected,
    )
    output_tts = None
    if args.include_output_tts:
        output_tts = MeloTtsBridge(
            MeloTtsConfig(
                binary=args.tts_binary,
                model_dir=args.tts_model_dir,
                output_dir=paths.output_audio_dir,
                language=args.tts_language,
                speed=args.tts_speed,
                disable_bert=args.disable_tts_bert,
                disable_nf=args.disable_tts_denoise,
                timeout_seconds=args.tts_timeout,
            ),
            tts_device=plan.tts.selected,
            bert_device=plan.tts_bert.selected,
            denoise_device=plan.tts_denoise.selected,
        )

    generation = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        max_prompt_len=config.generation.max_prompt_len,
        min_response_len=config.generation.min_response_len,
        temperature=config.generation.temperature,
    )
    llm_plan = plan.llm_large if args.large_llm else plan.llm_small
    llm_model = config.large_llm_model if args.large_llm else args.llm_model
    pipeline = VoicePipeline(
        asr=WhisperAsrEngine(
            model=args.asr_model,
            device=plan.asr.selected,
            fallback=plan.asr.fallback,
            cache_dir=args.cache_dir,
            language=args.language,
        ),
        llm=QwenLlmEngine(
            model=llm_model,
            device=llm_plan.selected,
            fallback=llm_plan.fallback,
            cache_dir=args.cache_dir,
            generation=generation,
            system_prompt=config.system_prompt,
        ),
        tts=output_tts,
        rules_path=args.rules,
        object_mapping_path=args.object_mapping,
    )

    run_started = datetime.now().isoformat(timespec="seconds")
    run_started_perf = time.perf_counter()
    results: list[dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        stem = f"{index:02d}_{case.case_id}"
        raw_path = paths.raw_audio_dir / f"{stem}_ZH-MIX-EN.wav"
        input_path = paths.input_audio_dir / f"{stem}_16k.wav"

        input_tts_seconds = 0.0
        if args.reuse_audio and raw_path.exists():
            raw_audio_paths = (raw_path,)
        else:
            tts_result = input_tts.synthesize(case.utterance, output_name=stem)
            input_tts_seconds = tts_result.elapsed_seconds
            raw_audio_paths = tts_result.audio_paths
            raw_path = raw_audio_paths[0]

        audio_seconds, convert_seconds, source_sample_rate = convert_to_mono_16k(raw_path, input_path)
        turn = pipeline.run_audio_file(input_path, synthesize=args.include_output_tts)
        evaluation = evaluate_response(turn.response_text, case.checks)

        result = turn_to_result(
            case=case,
            index=index,
            raw_audio_path=raw_path,
            input_audio_path=input_path,
            turn=turn,
            evaluation=evaluation,
            input_tts_seconds=input_tts_seconds,
            convert_seconds=convert_seconds,
            source_sample_rate=source_sample_rate,
            include_output_tts=args.include_output_tts,
        )
        results.append(result)
        print(
            f"[{index:02d}/{len(cases):02d}] {case.case_id} "
            f"{'PASS' if evaluation.passed else 'FAIL'} "
            f"audio={audio_seconds:.2f}s asr={result['asr_inference_seconds']:.2f}s "
            f"llm={result['llm_inference_seconds']:.2f}s total={result['total_seconds']:.2f}s",
            flush=True,
        )

    payload = {
        "run": {
            "started_at": run_started,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds": time.perf_counter() - run_started_perf,
            "count": len(results),
            "language": args.language,
            "include_output_tts": args.include_output_tts,
            "reuse_audio": args.reuse_audio,
            "asr_model": args.asr_model,
            "llm_model": llm_model,
            "rules_path": str(args.rules),
            "object_mapping_path": str(args.object_mapping),
            "max_new_tokens": args.max_new_tokens,
            "cache_dir": str(args.cache_dir),
            "output_dir": str(paths.output_dir),
        },
        "device_plan": plan.as_dict(),
        "summary": summarize_results(results),
        "results": results,
    }
    write_json(paths.json_path, payload)
    write_csv(paths.csv_path, results)
    write_report(paths.report_path, payload)
    print(f"\nResults JSON: {paths.json_path}")
    print(f"Results CSV: {paths.csv_path}")
    print(f"Chinese report: {paths.report_path}")
    return payload


def turn_to_result(
    *,
    case: SafetyVoiceCase,
    index: int,
    raw_audio_path: Path,
    input_audio_path: Path,
    turn: VoiceTurnResult,
    evaluation: EvaluationResult,
    input_tts_seconds: float,
    convert_seconds: float,
    source_sample_rate: int,
    include_output_tts: bool,
) -> dict[str, Any]:
    asr = turn.asr
    if asr is None:
        raise RuntimeError("Batch safety voice test requires ASR results.")

    output_audio_paths = [str(path) for path in turn.tts.audio_paths] if turn.tts else []
    return {
        "index": index,
        "case_id": case.case_id,
        "utterance": case.utterance,
        "expected_behavior": case.expected_behavior,
        "expected_checks": [{"label": check.label, "any_of": list(check.any_of)} for check in case.checks],
        "raw_audio_path": str(raw_audio_path),
        "input_audio_path": str(input_audio_path),
        "source_sample_rate": source_sample_rate,
        "audio_seconds": asr.audio_seconds,
        "input_tts_seconds": input_tts_seconds,
        "resample_seconds": convert_seconds,
        "asr_model": asr.model,
        "asr_device": asr.device,
        "asr_load_seconds": asr.load_seconds,
        "asr_inference_seconds": asr.inference_seconds,
        "asr_text": turn.input_text,
        "llm_model": turn.llm.model,
        "llm_device": turn.llm.device,
        "llm_load_seconds": turn.llm.load_seconds,
        "llm_inference_seconds": turn.llm.inference_seconds,
        "response_text": turn.response_text,
        "output_tts_enabled": include_output_tts,
        "output_tts_seconds": turn.tts.elapsed_seconds if turn.tts else 0.0,
        "output_audio_paths": output_audio_paths,
        "total_seconds": turn.total_seconds,
        "passed": evaluation.passed,
        "matched_checks": evaluation.matched,
        "missing_checks": {key: list(value) for key, value in evaluation.missing.items()},
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(1 for item in results if item["passed"])
    failed = len(results) - passed
    summary = {
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / len(results) if results else 0.0,
    }
    for key in (
        "input_tts_seconds",
        "resample_seconds",
        "asr_load_seconds",
        "asr_inference_seconds",
        "llm_load_seconds",
        "llm_inference_seconds",
        "output_tts_seconds",
        "total_seconds",
        "audio_seconds",
    ):
        summary[key] = numeric_summary([float(item[key]) for item in results])
    warm_results = [item for item in results if item["asr_load_seconds"] == 0.0 and item["llm_load_seconds"] == 0.0]
    summary["warm_turn_total_seconds"] = numeric_summary([float(item["total_seconds"]) for item in warm_results])
    return summary


def numeric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0, "total": 0.0}
    return {
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "max": max(values),
        "total": sum(values),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    columns = (
        "index",
        "case_id",
        "utterance",
        "expected_behavior",
        "audio_seconds",
        "input_tts_seconds",
        "resample_seconds",
        "asr_load_seconds",
        "asr_inference_seconds",
        "llm_load_seconds",
        "llm_inference_seconds",
        "output_tts_seconds",
        "total_seconds",
        "asr_text",
        "response_text",
        "passed",
        "matched_checks",
        "missing_checks",
        "input_audio_path",
        "raw_audio_path",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for item in results:
            row = dict(item)
            row["matched_checks"] = json.dumps(row["matched_checks"], ensure_ascii=False)
            row["missing_checks"] = json.dumps(row["missing_checks"], ensure_ascii=False)
            writer.writerow(row)


def write_report(path: Path, payload: dict[str, Any]) -> None:
    run = payload["run"]
    summary = payload["summary"]
    device_plan = payload["device_plan"]
    results = payload["results"]

    lines = [
        "# 机械臂安全语音批量测试报告",
        "",
        "## 测试概况",
        "",
        f"- 测试时间：{run['started_at']} 至 {run['finished_at']}",
        f"- 测试数量：{run['count']} 条机械臂安全相关语音",
        f"- 输入语音：MeloTTS 生成 44.1 kHz WAV 后重采样为 16 kHz PCM WAV",
        f"- 系统链路：Whisper ASR -> Qwen LLM"
        + (" -> MeloTTS 输出语音" if run["include_output_tts"] else "（本轮未启用输出 TTS）"),
        f"- ASR 模型：`{run['asr_model']}`",
        f"- LLM 模型：`{run['llm_model']}`，max_new_tokens={run['max_new_tokens']}",
        f"- 规则文件：`{run['rules_path']}`",
        f"- 结果目录：`{run['output_dir']}`",
        "",
        "## 设备计划",
        "",
        "| 阶段 | 主设备 | 回退顺序 |",
        "|---|---:|---|",
    ]
    for stage, plan in device_plan.items():
        fallback = " -> ".join(plan["fallback"]) if plan["fallback"] else "-"
        lines.append(f"| {stage} | {plan['selected']} | {fallback} |")

    lines.extend(
        [
            "",
            "## 总体结论",
            "",
            f"- 通过：{summary['passed']} 条；未通过：{summary['failed']} 条；通过率：{summary['pass_rate']:.1%}",
            f"- 批测总墙钟耗时：{run['elapsed_seconds']:.2f} 秒",
            f"- 常驻模型热态单轮平均耗时：{summary['warm_turn_total_seconds']['mean']:.2f} 秒",
            f"- 首轮模型加载耗时：ASR {results[0]['asr_load_seconds']:.2f} 秒，LLM {results[0]['llm_load_seconds']:.2f} 秒",
            "",
            "## 分阶段耗时统计",
            "",
            "| 阶段 | 最小(s) | 平均(s) | 中位(s) | 最大(s) | 总计(s) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    stage_labels = (
        ("input_tts_seconds", "测试语音生成"),
        ("resample_seconds", "16 kHz 转换"),
        ("asr_load_seconds", "ASR 加载"),
        ("asr_inference_seconds", "ASR 推理"),
        ("llm_load_seconds", "LLM 加载"),
        ("llm_inference_seconds", "LLM 推理"),
        ("output_tts_seconds", "输出 TTS"),
        ("total_seconds", "系统单轮总耗时"),
    )
    for key, label in stage_labels:
        stats = summary[key]
        lines.append(
            f"| {label} | {stats['min']:.3f} | {stats['mean']:.3f} | "
            f"{stats['median']:.3f} | {stats['max']:.3f} | {stats['total']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 逐条结果",
            "",
            "| ID | 输入语音文本 | ASR 转写 | 输出摘要 | 判定 | 总耗时(s) |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for item in results:
        response = compact_text(item["response_text"], 48)
        asr_text = compact_text(item["asr_text"], 32)
        status = "通过" if item["passed"] else "未通过"
        lines.append(
            f"| {item['case_id']} | {escape_table(item['utterance'])} | "
            f"{escape_table(asr_text)} | {escape_table(response)} | {status} | {item['total_seconds']:.2f} |"
        )

    failed = [item for item in results if not item["passed"]]
    if failed:
        lines.extend(["", "## 未通过项", ""])
        for item in failed:
            missing = "; ".join(
                f"{label}: {'/'.join(keywords)}" for label, keywords in item["missing_checks"].items()
            )
            lines.append(f"- {item['case_id']} 缺少期望关键词组：{missing}")

    lines.extend(
        [
            "",
            "## 原始产物",
            "",
            f"- JSON 明细：`{Path(run['output_dir']) / 'results.json'}`",
            f"- CSV 明细：`{Path(run['output_dir']) / 'results.csv'}`",
            f"- 16 kHz 输入语音：`{Path(run['output_dir']) / 'audio_input_16k'}`",
            f"- 原始 44.1 kHz 语音：`{Path(run['output_dir']) / 'audio_raw_44k'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compact_text(text: str, limit: int) -> str:
    value = " ".join(text.split())
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "..."


def escape_table(text: str) -> str:
    return text.replace("|", "\\|")


def build_parser() -> argparse.ArgumentParser:
    defaults = VoiceStackConfig()
    tts_defaults = MeloTtsConfig()
    parser = argparse.ArgumentParser(description="Generate and run robot-arm safety voice batch tests.")
    parser.add_argument("--count", type=int, default=len(DEFAULT_CASES))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--reuse-audio", action="store_true", help="Reuse existing generated raw WAV files when present.")
    parser.add_argument("--language", default="zh", help="Whisper language token.")
    parser.add_argument("--asr-model", default=defaults.asr_model)
    parser.add_argument("--llm-model", default=defaults.llm_model)
    parser.add_argument("--rules", type=Path, default=defaults.rules_path)
    parser.add_argument("--object-mapping", type=Path, default=defaults.object_mapping_path)
    parser.add_argument("--large-llm", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--cache-dir", type=Path, default=defaults.cache_dir)
    parser.add_argument("--include-output-tts", action="store_true")
    parser.add_argument("--tts-binary", type=Path, default=tts_defaults.binary)
    parser.add_argument("--tts-model-dir", type=Path, default=tts_defaults.model_dir)
    parser.add_argument("--tts-language", default=tts_defaults.language, choices=("ZH", "EN"))
    parser.add_argument("--tts-speed", type=float, default=tts_defaults.speed)
    parser.add_argument("--tts-timeout", type=float, default=tts_defaults.timeout_seconds)
    parser.add_argument("--disable-tts-bert", action="store_true")
    parser.add_argument("--disable-tts-denoise", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_batch(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
