"""CLI for the OpenVINO real-time voice stack."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from local_safety_assistant.stack.asr import WhisperAsrEngine
from local_safety_assistant.stack.config import (
    DEFAULT_MOSS_YANGMI_PROMPT_AUDIO,
    MeloTtsConfig,
    MossTtsConfig,
    PiperTtsConfig,
    VoiceStackConfig,
)
from local_safety_assistant.stack.devices import available_openvino_devices, build_device_plan, format_device_plan
from local_safety_assistant.stack.llm import QwenLlmEngine
from local_safety_assistant.stack.microphone import (
    EndpointingConfig,
    MicrophoneConfig,
    iter_microphone_utterances,
    list_audio_devices,
    play_wav_file,
    require_sounddevice,
    write_utterance_wav,
)
from local_safety_assistant.stack.pipeline import (
    RULE_EDIT_STRATEGY_ONE_PASS,
    RULE_EDIT_STRATEGY_TWO_PASS,
    VoicePipeline,
    VoiceTurnResult,
)
from local_safety_assistant.stack.ros2_bridge import (
    Ros2BridgeConfig,
    Ros2VoiceBridge,
    build_voice_ros2_plan,
    plans_to_json,
    sync_estop_plans_to_arm_rules,
)
from local_safety_assistant.stack.tts import MeloTtsBridge, MossTtsBridge, PiperTtsBridge
from local_safety_assistant.stack.vision import Ros2TriggerVisionSnapshotProvider


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenVINO ASR -> Qwen3.5 -> local TTS voice stack.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Print the OpenVINO device allocation plan.")
    plan.add_argument("--json", action="store_true")

    audio_devices = subparsers.add_parser("audio-devices", help="List sounddevice input/output devices.")
    audio_devices.add_argument("--json", action="store_true")

    text_turn = subparsers.add_parser("text-turn", help="Run a text-only LLM turn with optional TTS.")
    add_runtime_args(text_turn)
    text_turn.add_argument("--text", required=True)
    text_turn.add_argument("--skip-tts", action="store_true")

    audio_file = subparsers.add_parser("audio-file", help="Run ASR -> LLM -> optional TTS on a 16 kHz WAV file.")
    add_runtime_args(audio_file)
    audio_file.add_argument("--audio", type=Path, required=True)
    audio_file.add_argument("--language", help="Whisper language token, e.g. zh or en.")
    audio_file.add_argument("--skip-tts", action="store_true")

    ros2_text_turn = subparsers.add_parser("ros2-text-turn", help="Run a text turn and publish ROS2 topic messages.")
    add_runtime_args(ros2_text_turn)
    add_ros2_bridge_args(ros2_text_turn)
    ros2_text_turn.add_argument("--text", required=True)
    ros2_text_turn.add_argument("--skip-tts", action="store_true")

    ros2_audio_file = subparsers.add_parser(
        "ros2-audio-file",
        help="Run ASR -> LLM -> optional TTS on a WAV file and publish ROS2 topic messages.",
    )
    add_runtime_args(ros2_audio_file)
    add_ros2_bridge_args(ros2_audio_file)
    ros2_audio_file.add_argument("--audio", type=Path, required=True)
    ros2_audio_file.add_argument("--language", help="Whisper language token, e.g. zh or en.")
    ros2_audio_file.add_argument("--skip-tts", action="store_true")

    listen = subparsers.add_parser("listen", help="Run no-wake realtime microphone voice turns.")
    add_runtime_args(listen)
    add_ros2_bridge_args(listen)
    add_listen_args(listen)

    tts = subparsers.add_parser("tts", help="Synthesize text through the configured local TTS engine.")
    add_tts_args(tts)
    tts.add_argument("--text", required=True)
    tts.add_argument("--output-name")

    return parser


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    defaults = VoiceStackConfig()
    parser.add_argument("--llm-model", default=defaults.llm_model)
    parser.add_argument("--large-llm", action="store_true", help="Use qwen35-9b instead of the default 2B model.")
    parser.add_argument("--max-new-tokens", type=int, default=defaults.generation.max_new_tokens)
    parser.add_argument("--asr-model", default=defaults.asr_model)
    parser.add_argument("--rules", type=Path, default=defaults.rules_path)
    parser.add_argument("--object-mapping", type=Path, default=defaults.object_mapping_path)
    parser.add_argument("--arm-rules", type=Path, default=defaults.arm_rules_path)
    parser.add_argument("--cache-dir", type=Path, default=defaults.cache_dir)
    parser.add_argument("--vision-snapshot-service", default=defaults.vision.snapshot_service)
    parser.add_argument("--vision-snapshot-timeout", type=float, default=defaults.vision.snapshot_timeout_seconds)
    parser.add_argument(
        "--rule-edit-strategy",
        choices=(
            RULE_EDIT_STRATEGY_ONE_PASS.replace("_", "-"),
            RULE_EDIT_STRATEGY_TWO_PASS.replace("_", "-"),
        ),
        default=defaults.rule_edit_strategy,
        help="Select the same-2B rule patch generation strategy for rule-edit smoke tests.",
    )
    add_tts_args(parser)


def add_tts_args(parser: argparse.ArgumentParser) -> None:
    defaults = MeloTtsConfig()
    moss_defaults = MossTtsConfig()
    piper_defaults = PiperTtsConfig()
    parser.add_argument("--tts-engine", choices=("moss", "melo", "piper"), default="moss")
    parser.add_argument("--tts-binary", type=Path, default=defaults.binary)
    parser.add_argument("--tts-model-dir", type=Path, default=defaults.model_dir)
    parser.add_argument("--tts-output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--tts-language", default=defaults.language, choices=("ZH", "EN"))
    parser.add_argument("--tts-speed", type=float, default=defaults.speed)
    parser.add_argument("--tts-timeout", type=float, default=defaults.timeout_seconds)
    parser.add_argument("--disable-tts-bert", action="store_true")
    parser.add_argument("--disable-tts-denoise", action="store_true")
    parser.add_argument("--moss-executable", type=Path, default=moss_defaults.executable)
    parser.add_argument("--moss-source-dir", type=Path, default=moss_defaults.source_dir)
    parser.add_argument("--moss-model-dir", type=Path, default=moss_defaults.model_dir)
    parser.add_argument("--moss-voice", default=moss_defaults.voice)
    parser.add_argument("--moss-prompt-audio", type=Path, default=moss_defaults.prompt_audio)
    parser.add_argument(
        "--moss-use-yangmi-prompt-audio",
        action="store_true",
        help="Use the legacy zh_11.wav prompt-audio cloning preset instead of the built-in voice codes.",
    )
    parser.add_argument("--moss-cpu-threads", type=int, default=moss_defaults.cpu_threads)
    parser.add_argument("--moss-cpus", default=moss_defaults.cpu_affinity)
    parser.add_argument("--moss-execution-provider", choices=("cpu", "cuda"), default=moss_defaults.execution_provider)
    parser.add_argument("--moss-max-new-frames", type=int, default=moss_defaults.max_new_frames)
    parser.add_argument("--moss-voice-clone-max-text-tokens", type=int, default=moss_defaults.voice_clone_max_text_tokens)
    parser.add_argument(
        "--moss-realtime-streaming-decode",
        type=int,
        choices=(0, 1),
        default=moss_defaults.realtime_streaming_decode,
    )
    parser.add_argument("--moss-timeout", type=float, default=moss_defaults.timeout_seconds)
    parser.add_argument("--piper-python", type=Path, default=piper_defaults.python)
    parser.add_argument("--piper-runner", type=Path, default=piper_defaults.runner)
    parser.add_argument("--piper-model-dir", type=Path, default=piper_defaults.model_dir)
    parser.add_argument("--piper-espeak-data-dir", type=Path, default=piper_defaults.espeak_data_dir)
    parser.add_argument("--piper-speed", type=float, default=piper_defaults.speed)
    parser.add_argument("--piper-silence-scale", type=float, default=piper_defaults.silence_scale)
    parser.add_argument("--piper-threads", type=int, default=piper_defaults.threads)
    parser.add_argument("--piper-timeout", type=float, default=piper_defaults.timeout_seconds)


def add_ros2_bridge_args(parser: argparse.ArgumentParser) -> None:
    defaults = Ros2BridgeConfig()
    parser.add_argument("--dry-run-ros2", action="store_true", help="Print planned ROS2 messages without publishing.")
    parser.add_argument("--direct-estop-topic", action="store_true", help="Publish Bool directly to /emergency_stop.")
    parser.add_argument("--ros2-node-name", default=defaults.node_name)
    parser.add_argument("--voice-source", default=defaults.source)
    parser.add_argument("--transcript-topic", default=defaults.transcript_topic)
    parser.add_argument("--response-topic", default=defaults.response_topic)
    parser.add_argument("--estop-request-topic", default=defaults.estop_request_topic)
    parser.add_argument("--estop-bool-topic", default=defaults.estop_bool_topic)
    parser.add_argument("--goal-topic", default=defaults.goal_topic)
    parser.add_argument("--ros2-wait", type=float, default=defaults.wait_for_subscribers_seconds)
    parser.add_argument("--no-transcript-topic", action="store_true")
    parser.add_argument("--no-response-topic", action="store_true")
    parser.add_argument("--no-command-topics", action="store_true")


def add_listen_args(parser: argparse.ArgumentParser) -> None:
    endpoint_defaults = EndpointingConfig()
    mic_defaults = MicrophoneConfig()
    parser.add_argument("--no-wake", action="store_true", help="Listen continuously without wake-word detection.")
    parser.add_argument("--no-ros2", action="store_true", help="Do not publish ROS2 messages for completed turns.")
    parser.add_argument("--language", help="Whisper language token, e.g. zh or en.")
    parser.add_argument("--skip-tts", action="store_true")
    parser.add_argument("--no-play-tts", action="store_true", help="Generate TTS WAV files without playing them.")
    parser.add_argument("--play-device", help="sounddevice output device name or numeric id for TTS playback.")
    parser.add_argument("--max-turns", type=int, help="Stop after this many completed utterances.")
    parser.add_argument("--mic-device", help="sounddevice input device name or numeric id.")
    parser.add_argument("--mic-sample-rate", type=int, help="Input device sample rate. Defaults to device default.")
    parser.add_argument("--mic-block-ms", type=float, default=mic_defaults.block_seconds * 1000.0)
    parser.add_argument("--speech-threshold", type=float, default=endpoint_defaults.speech_threshold)
    parser.add_argument("--min-speech-seconds", type=float, default=endpoint_defaults.min_speech_seconds)
    parser.add_argument("--trailing-silence-seconds", type=float, default=endpoint_defaults.trailing_silence_seconds)
    parser.add_argument("--pre-roll-seconds", type=float, default=endpoint_defaults.pre_roll_seconds)
    parser.add_argument("--max-utterance-seconds", type=float, default=endpoint_defaults.max_utterance_seconds)
    parser.add_argument(
        "--listen-audio-dir",
        type=Path,
        help="Directory for captured utterance WAV files. Defaults to temporary directories.",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if args.command == "plan":
        return command_plan(args)
    if args.command == "audio-devices":
        return command_audio_devices(args)
    if args.command == "text-turn":
        return command_text_turn(args)
    if args.command == "audio-file":
        return command_audio_file(args)
    if args.command == "ros2-text-turn":
        return command_ros2_text_turn(args)
    if args.command == "ros2-audio-file":
        return command_ros2_audio_file(args)
    if args.command == "listen":
        return command_listen(args)
    if args.command == "tts":
        return command_tts(args)

    parser.print_help()
    return 2


def command_plan(args: argparse.Namespace) -> int:
    plan = build_device_plan(available_openvino_devices())
    if args.json:
        print(json.dumps(plan.as_dict(), indent=2, ensure_ascii=True))
    else:
        print(format_device_plan(plan))
    return 0


def command_audio_devices(args: argparse.Namespace) -> int:
    try:
        devices = list_audio_devices()
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps([asdict(device) for device in devices], indent=2, ensure_ascii=True))
        return 0

    print("Audio devices:")
    for device in devices:
        direction = []
        if device.max_input_channels:
            direction.append(f"in={device.max_input_channels}")
        if device.max_output_channels:
            direction.append(f"out={device.max_output_channels}")
        flags = ", ".join(direction) or "no I/O"
        print(f"{device.index}: {device.name} ({flags}, default_sr={device.default_sample_rate:g})")
    return 0


def command_text_turn(args: argparse.Namespace) -> int:
    pipeline = build_pipeline(args, include_asr=False, include_tts=not args.skip_tts)
    result = pipeline.run_text_turn(args.text, synthesize=not args.skip_tts)
    print_turn_result(result)
    return 0


def command_audio_file(args: argparse.Namespace) -> int:
    pipeline = build_pipeline(args, include_asr=True, include_tts=not args.skip_tts)
    result = pipeline.run_audio_file(args.audio, synthesize=not args.skip_tts)
    print_turn_result(result)
    return 0


def command_ros2_text_turn(args: argparse.Namespace) -> int:
    pipeline = build_pipeline(args, include_asr=False, include_tts=not args.skip_tts)
    result = pipeline.run_text_turn(args.text, synthesize=not args.skip_tts)
    print_turn_result(result)
    publish_or_print_ros2_plan(result, args)
    return 0


def command_ros2_audio_file(args: argparse.Namespace) -> int:
    pipeline = build_pipeline(args, include_asr=True, include_tts=not args.skip_tts)
    result = pipeline.run_audio_file(args.audio, synthesize=not args.skip_tts)
    print_turn_result(result)
    publish_or_print_ros2_plan(result, args)
    return 0


def command_listen(args: argparse.Namespace) -> int:
    if not args.no_wake:
        print("The listen command currently supports only --no-wake mode.", file=sys.stderr)
        return 2
    if args.max_turns is not None and args.max_turns <= 0:
        print("--max-turns must be greater than 0.", file=sys.stderr)
        return 2

    try:
        require_sounddevice()
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 1

    pipeline = build_pipeline(args, include_asr=True, include_tts=not args.skip_tts)
    mic_config = build_microphone_config(args)
    endpoint_config = build_endpointing_config(args, mic_config.sample_rate)

    print("Listening without wake word. Press Ctrl+C to stop.")
    completed = 0
    try:
        for utterance in iter_microphone_utterances(mic_config, endpoint_config):
            completed += 1
            print(
                f"\nTurn {completed}: captured {utterance.audio_seconds:.2f}s "
                f"(speech {utterance.speech_seconds:.2f}s, rms {utterance.rms:.4f}, reason {utterance.reason})"
            )
            audio_path = write_utterance_wav(utterance, args.listen_audio_dir)
            result = pipeline.run_audio_file(audio_path, synthesize=not args.skip_tts)
            print_turn_result(result)
            if should_play_tts(args, result):
                play_tts_result(result, args)
            if not args.no_ros2:
                publish_or_print_ros2_plan(result, args)
            if args.max_turns is not None and completed >= args.max_turns:
                break
    except KeyboardInterrupt:
        print("\nStopped.")
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


def command_tts(args: argparse.Namespace) -> int:
    plan = build_device_plan(available_openvino_devices()) if args.tts_engine == "melo" else None
    bridge = build_tts_engine(args, plan=plan)
    result = bridge.synthesize(args.text, output_name=args.output_name)
    print("Generated audio:")
    for path in result.audio_paths:
        print(f"- {path}")
    print(f"Elapsed: {result.elapsed_seconds:.3f}s")
    return 0


def build_pipeline(args: argparse.Namespace, *, include_asr: bool, include_tts: bool) -> VoicePipeline:
    plan = build_device_plan(available_openvino_devices())
    config = VoiceStackConfig()
    generation = config.generation.__class__(
        max_new_tokens=args.max_new_tokens,
        max_prompt_len=config.generation.max_prompt_len,
        min_response_len=config.generation.min_response_len,
        temperature=config.generation.temperature,
    )
    llm_plan = plan.llm_large if args.large_llm else plan.llm_small
    model = config.large_llm_model if args.large_llm else args.llm_model
    llm = QwenLlmEngine(
        model=model,
        device=llm_plan.selected,
        fallback=llm_plan.fallback,
        cache_dir=args.cache_dir,
        generation=generation,
        system_prompt=config.system_prompt,
    )
    asr = None
    if include_asr:
        asr = WhisperAsrEngine(
            model=args.asr_model,
            device=plan.asr.selected,
            fallback=plan.asr.fallback,
            cache_dir=args.cache_dir,
            language=getattr(args, "language", None),
        )

    tts = None
    if include_tts:
        tts = build_tts_engine(args, plan=plan)

    return VoicePipeline(
        asr=asr,
        llm=llm,
        tts=tts,
        rules_path=args.rules,
        object_mapping_path=args.object_mapping,
        arm_rules_path=args.arm_rules,
        rule_edit_strategy=args.rule_edit_strategy,
        vision_snapshot_provider=Ros2TriggerVisionSnapshotProvider(
            service_name=args.vision_snapshot_service,
            timeout_seconds=args.vision_snapshot_timeout,
        ),
        require_confirmation_for_side_effects=bool(
            getattr(args, "require_confirmation_for_side_effects", False)
        ),
    )


def build_tts_config(args: argparse.Namespace) -> MeloTtsConfig:
    return MeloTtsConfig(
        binary=args.tts_binary,
        model_dir=args.tts_model_dir,
        output_dir=args.tts_output_dir,
        language=args.tts_language,
        speed=args.tts_speed,
        disable_bert=args.disable_tts_bert,
        disable_nf=args.disable_tts_denoise,
        timeout_seconds=args.tts_timeout,
    )


def build_piper_tts_config(args: argparse.Namespace) -> PiperTtsConfig:
    return PiperTtsConfig(
        python=args.piper_python,
        runner=args.piper_runner,
        model_dir=args.piper_model_dir,
        espeak_data_dir=args.piper_espeak_data_dir,
        output_dir=args.tts_output_dir,
        speed=args.piper_speed,
        silence_scale=args.piper_silence_scale,
        threads=args.piper_threads,
        timeout_seconds=args.piper_timeout,
    )


def build_moss_tts_config(args: argparse.Namespace) -> MossTtsConfig:
    prompt_audio = args.moss_prompt_audio
    if prompt_audio is None and getattr(args, "moss_use_yangmi_prompt_audio", False):
        prompt_audio = DEFAULT_MOSS_YANGMI_PROMPT_AUDIO
    return MossTtsConfig(
        executable=args.moss_executable,
        source_dir=args.moss_source_dir,
        model_dir=args.moss_model_dir,
        output_dir=args.tts_output_dir,
        voice=args.moss_voice,
        prompt_audio=prompt_audio,
        cpu_threads=args.moss_cpu_threads,
        cpu_affinity=args.moss_cpus,
        execution_provider=args.moss_execution_provider,
        max_new_frames=args.moss_max_new_frames,
        voice_clone_max_text_tokens=args.moss_voice_clone_max_text_tokens,
        realtime_streaming_decode=args.moss_realtime_streaming_decode,
        timeout_seconds=args.moss_timeout,
    )


def build_tts_engine(args: argparse.Namespace, *, plan=None) -> MeloTtsBridge | MossTtsBridge | PiperTtsBridge:
    if args.tts_engine == "moss":
        return MossTtsBridge(build_moss_tts_config(args))
    if args.tts_engine == "piper":
        return PiperTtsBridge(build_piper_tts_config(args))
    if plan is None:
        plan = build_device_plan(available_openvino_devices())
    return MeloTtsBridge(
        build_tts_config(args),
        tts_device=plan.tts.selected,
        bert_device=plan.tts_bert.selected,
        denoise_device=plan.tts_denoise.selected,
    )


def build_microphone_config(args: argparse.Namespace) -> MicrophoneConfig:
    return MicrophoneConfig(
        sample_rate=args.mic_sample_rate,
        block_seconds=max(args.mic_block_ms, 1.0) / 1000.0,
        device=parse_microphone_device(args.mic_device),
    )


def build_endpointing_config(args: argparse.Namespace, sample_rate: int) -> EndpointingConfig:
    return EndpointingConfig(
        sample_rate=MicrophoneConfig().target_sample_rate,
        speech_threshold=args.speech_threshold,
        min_speech_seconds=args.min_speech_seconds,
        trailing_silence_seconds=args.trailing_silence_seconds,
        pre_roll_seconds=args.pre_roll_seconds,
        max_utterance_seconds=args.max_utterance_seconds,
    )


def parse_microphone_device(value: str | None) -> str | int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def should_play_tts(args: argparse.Namespace, result: VoiceTurnResult) -> bool:
    return bool(result.tts and not args.skip_tts and not args.no_play_tts)


def play_tts_result(result: VoiceTurnResult, args: argparse.Namespace) -> None:
    if result.tts is None:
        return
    device = parse_microphone_device(args.play_device)
    for path in result.tts.audio_paths:
        print(f"Playing TTS: {path}")
        play_wav_file(path, device=device)


def build_ros2_bridge_config(args: argparse.Namespace) -> Ros2BridgeConfig:
    return Ros2BridgeConfig(
        node_name=args.ros2_node_name,
        transcript_topic=args.transcript_topic,
        response_topic=args.response_topic,
        estop_request_topic=args.estop_request_topic,
        estop_bool_topic=args.estop_bool_topic,
        goal_topic=args.goal_topic,
        source=args.voice_source,
        use_estop_request=not args.direct_estop_topic,
        publish_transcript=not args.no_transcript_topic,
        publish_response=not args.no_response_topic,
        publish_commands=not args.no_command_topics,
        wait_for_subscribers_seconds=args.ros2_wait,
    )


def publish_or_print_ros2_plan(result: VoiceTurnResult, args: argparse.Namespace) -> None:
    bridge = Ros2VoiceBridge(build_ros2_bridge_config(args))
    plans = build_voice_ros2_plan(result, bridge.config)
    sync_estop_plans_to_arm_rules(plans, bridge.config, args.arm_rules)
    if args.dry_run_ros2:
        print("\nROS2 plan:")
        print(plans_to_json(plans))
        return

    bridge.publish_plans(plans)
    print("\nROS2 published:")
    print(plans_to_json(plans))


def print_turn_result(result: VoiceTurnResult) -> None:
    print("Input:")
    print(result.input_text)
    print("\nResponse:")
    print(result.response_text)
    print("\nTimings:")
    summary = {
        "total_seconds": result.total_seconds,
        "asr": asdict(result.asr) if result.asr else None,
        "llm": llm_result_to_json(result.llm),
        "tts": {
            "audio_paths": [str(path) for path in result.tts.audio_paths],
            "elapsed_seconds": result.tts.elapsed_seconds,
        }
        if result.tts
        else None,
        "rule_update": {
            "rules_path": str(result.rule_update.rules_path),
            "rule_id": result.rule_update.rule_id,
            "version": result.rule_update.version,
            "strategy": result.rule_update.strategy,
            "patch_model": result.rule_update.patch_llm.model,
            "patch_device": result.rule_update.patch_llm.device,
        }
        if result.rule_update
        else None,
        "object_mapping_update": {
            "object_mapping_path": str(result.object_mapping_update.object_mapping_path),
            "marker": result.object_mapping_update.marker,
            "object_name": result.object_mapping_update.object_name,
            "version": result.object_mapping_update.version,
        }
        if result.object_mapping_update
        else None,
        "object_mapping_query": {
            "object_mapping_path": str(result.object_mapping_query.object_mapping_path),
            "marker": result.object_mapping_query.marker,
            "object_name": result.object_mapping_query.object_name,
            "version": result.object_mapping_query.version,
            "enabled": result.object_mapping_query.enabled,
        }
        if result.object_mapping_query
        else None,
        "object_mapping_table_query": {
            "object_mapping_path": str(result.object_mapping_table_query.object_mapping_path),
            "version": result.object_mapping_table_query.version,
            "mappings": [asdict(mapping) for mapping in result.object_mapping_table_query.mappings],
        }
        if result.object_mapping_table_query
        else None,
        "object_grasp_intent": {
            "object_mapping_path": str(result.object_grasp_intent.object_mapping_path),
            "marker": result.object_grasp_intent.marker,
            "object_name": result.object_grasp_intent.object_name,
            "version": result.object_grasp_intent.version,
            "target_source": result.object_grasp_intent.target_source,
            "original_text": result.object_grasp_intent.original_text,
        }
        if result.object_grasp_intent
        else None,
        "arm_runtime_query": {
            "arm_rules_path": str(result.arm_runtime_query.arm_rules_path),
            "capture_requested": result.arm_runtime_query.capture_requested,
            "capture_goal": result.arm_runtime_query.capture_goal,
            "capture_object": result.arm_runtime_query.capture_object,
            "stop_requested": result.arm_runtime_query.stop_requested,
            "recover_requested": result.arm_runtime_query.recover_requested,
            "decelerate": result.arm_runtime_query.decelerate,
            "safety_distance": result.arm_runtime_query.safety_distance,
        }
        if result.arm_runtime_query
        else None,
        "arm_deceleration_request": {
            "arm_rules_path": str(result.arm_deceleration_request.arm_rules_path),
            "target_speed_percent": result.arm_deceleration_request.target_speed_percent,
            "arm_decelerate": result.arm_deceleration_request.arm_decelerate,
        }
        if result.arm_deceleration_request
        else None,
        "vision_artifacts": [artifact.as_dict() for artifact in result.vision_artifacts],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=True))


def llm_result_to_json(result) -> dict:
    payload = asdict(result)
    try:
        json.dumps(payload.get("parsed"), ensure_ascii=True)
    except TypeError:
        payload["parsed"] = str(payload.get("parsed"))
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
