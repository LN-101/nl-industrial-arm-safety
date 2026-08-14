"""Initial local safety assistant entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import openvino as ov

from local_safety_assistant.config import MODEL_ALIASES, PROJECT_ROOT
from local_safety_assistant.model_testbed import detect_model, format_size
from local_safety_assistant.rules import RuleValidationError, load_rule_document
from local_safety_assistant.stack.config import MeloTtsConfig, VoiceStackConfig
from local_safety_assistant.stack.devices import build_device_plan, format_device_plan


DEFAULT_RULES_PATH = PROJECT_ROOT / "Code" / "config" / "safety_rules.example.json"


def print_status(args: argparse.Namespace) -> int:
    print("Local safety assistant status")
    print(f"Python environment should be qwen35_env for this project.")

    core = ov.Core()
    print("OpenVINO devices:", ", ".join(core.available_devices) or "none")
    print()
    print(format_device_plan(build_device_plan(core.available_devices)))

    print("\nModels")
    for alias, model_alias in MODEL_ALIASES.items():
        info = detect_model(model_alias.path)
        status = "ok" if info.exists and info.kind != "unknown" else info.error or info.kind
        print(f"- {alias}: {info.kind} ({format_size(info.total_bin_bytes)}) [{status}]")

    stack_config = VoiceStackConfig()
    tts_config = MeloTtsConfig()
    print("\nVoice stack")
    print(f"- ASR default: {stack_config.asr_model}")
    print(f"- LLM default: {stack_config.llm_model}")
    print(f"- LLM large: {stack_config.large_llm_model}")
    print(f"- MeloTTS binary: {tts_config.binary} [{'ok' if tts_config.binary.exists() else 'missing'}]")
    print(f"- MeloTTS models: {tts_config.model_dir} [{'ok' if tts_config.model_dir.exists() else 'missing'}]")

    print("\nRules")
    try:
        document = load_rule_document(args.rules)
    except FileNotFoundError:
        print(f"- missing rule file: {args.rules}")
        return 2
    except RuleValidationError as error:
        print(f"- invalid rule file: {error}")
        return 2

    print(f"- file: {args.rules}")
    print(f"- version: {document['version']}")
    print(f"- rules: {len(document['rules'])}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local safety assistant project entrypoint.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="Validate local runtime configuration.")
    status.add_argument("--rules", type=Path, default=DEFAULT_RULES_PATH)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "status":
        return print_status(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
