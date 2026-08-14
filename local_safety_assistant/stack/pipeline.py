"""ASR -> LLM -> TTS pipeline orchestration."""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Protocol

from local_safety_assistant.arm_rules import (
    ArmDecelerationRequestResult,
    ArmRulesValidationError,
    request_arm_deceleration,
    sync_personnel_distance_to_arm_rules,
)
from local_safety_assistant.confirmation import (
    PendingConfirmation,
    build_object_grasp_execution_confirmation,
    build_object_mapping_update_confirmation,
    build_rule_edit_confirmation,
    build_speed_change_confirmation,
)
from local_safety_assistant.object_mapping import (
    ObjectMappingValidationError,
    get_object_mapping,
    load_object_mapping_document,
    normalize_marker,
    normalize_object_name,
    preview_object_mapping_update_changes,
    resolve_object_grasp_target,
    update_object_mapping,
)
from local_safety_assistant.rules import (
    PERSONNEL_DISTANCE_MAX_M,
    PERSONNEL_DISTANCE_MIN_M,
    RuleValidationError,
    apply_rule_patch,
    load_rule_document,
    preview_rule_patch_changes,
)
from local_safety_assistant.stack.asr import AsrResult
from local_safety_assistant.stack.llm import LlmResult, strip_thinking_text
from local_safety_assistant.stack.tts import TtsResult
from local_safety_assistant.stack.vision import (
    VisionImageArtifact,
    VisionSnapshotProvider,
    format_vision_snapshot_error,
)
from local_safety_assistant.workspace_snapshot import (
    WorkspaceArmRuntimeSummary,
    WorkspaceObjectMappingEntry,
    WorkspaceSnapshot,
    load_workspace_snapshot,
)


LOGGER = logging.getLogger(__name__)


class AsrEngine(Protocol):
    def transcribe_wav(self, audio_path: Path) -> AsrResult: ...


class LlmEngine(Protocol):
    def generate(self, user_text: str) -> LlmResult: ...


class TtsEngine(Protocol):
    def synthesize(self, text: str, *, output_name: str | None = None) -> TtsResult: ...


@dataclass(frozen=True)
class RuleUpdateResult:
    rules_path: Path
    rule_id: str
    version: int
    patch: dict[str, Any]
    patch_llm: LlmResult
    strategy: str


@dataclass(frozen=True)
class ObjectMappingUpdateResult:
    object_mapping_path: Path
    marker: str
    object_name: str
    version: int


@dataclass(frozen=True)
class ObjectMappingQueryResult:
    object_mapping_path: Path
    marker: str
    object_name: str
    version: int
    enabled: bool


@dataclass(frozen=True)
class ObjectMappingTableEntry:
    marker: str
    object_name: str
    enabled: bool


@dataclass(frozen=True)
class ObjectMappingTableQueryResult:
    object_mapping_path: Path
    version: int
    mappings: tuple[ObjectMappingTableEntry, ...]


@dataclass(frozen=True)
class ObjectGraspIntentResult:
    object_mapping_path: Path
    marker: str
    object_name: str
    version: int
    target_source: str
    original_text: str


@dataclass(frozen=True)
class ArmRuntimeQueryResult:
    arm_rules_path: Path
    capture_requested: bool
    capture_goal: str
    capture_object: str | None
    stop_requested: bool
    recover_requested: bool
    decelerate: str
    safety_distance: str


@dataclass(frozen=True)
class VoiceTurnResult:
    input_text: str
    response_text: str
    asr: AsrResult | None
    llm: LlmResult
    tts: TtsResult | None
    total_seconds: float
    rule_update: RuleUpdateResult | None = None
    object_mapping_update: ObjectMappingUpdateResult | None = None
    object_mapping_query: ObjectMappingQueryResult | None = None
    object_mapping_table_query: ObjectMappingTableQueryResult | None = None
    object_grasp_intent: ObjectGraspIntentResult | None = None
    arm_runtime_query: ArmRuntimeQueryResult | None = None
    arm_deceleration_request: ArmDecelerationRequestResult | None = None
    vision_artifacts: tuple[VisionImageArtifact, ...] = ()
    pending_confirmation: PendingConfirmation | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentToolRequest:
    name: str
    argument: str = ""
    arguments: dict[str, Any] | None = None


@dataclass(frozen=True)
class AgentRouteDecision:
    final_content: str | None = None
    tool_request: AgentToolRequest | None = None


RULE_EDIT_STRATEGY_ONE_PASS = "one_pass"
RULE_EDIT_STRATEGY_TWO_PASS = "two_pass"
RULE_EDIT_STRATEGIES = (RULE_EDIT_STRATEGY_ONE_PASS, RULE_EDIT_STRATEGY_TWO_PASS)


def _deterministic_llm_result(prompt: str, text: str) -> LlmResult:
    return LlmResult(
        prompt=prompt,
        text=text,
        model="deterministic-router",
        device="none",
        load_seconds=0.0,
        inference_seconds=0.0,
    )


def _llm_structured_text(result: LlmResult) -> str:
    text = strip_thinking_text(result.text).strip()
    if text or result.parsed is None:
        return text
    try:
        return json.dumps(result.parsed, ensure_ascii=False)
    except TypeError:
        return str(result.parsed)


def _log_route_stage(
    user_text: str,
    result: LlmResult,
    route_text: str,
    decision: AgentRouteDecision | None,
) -> None:
    decision_type = "none"
    tool_name = None
    if decision is not None:
        if decision.tool_request is not None:
            decision_type = "tool_call"
            tool_name = decision.tool_request.name
        else:
            decision_type = "final"
    _log_pipeline_event(
        "agent_route",
        user_text=user_text,
        model=result.model,
        device=result.device,
        inference_seconds=result.inference_seconds,
        route_text=route_text,
        decision_type=decision_type,
        tool_name=tool_name,
    )


def _log_rule_read_stage(
    user_text: str,
    result: LlmResult,
    model_text: str,
    response_text: str,
    *,
    used_fallback: bool,
) -> None:
    _log_pipeline_event(
        "rule_read",
        user_text=user_text,
        model=result.model,
        device=result.device,
        inference_seconds=result.inference_seconds,
        model_text=model_text,
        used_fallback=used_fallback,
        response_text=response_text,
    )


def _log_pipeline_event(event: str, **payload: Any) -> None:
    sanitized = {key: _diagnostic_value(value) for key, value in payload.items()}
    LOGGER.info("%s %s", event, json.dumps(sanitized, ensure_ascii=False, sort_keys=True))


def _diagnostic_value(value: Any, *, max_chars: int = 2000) -> Any:
    if not isinstance(value, str):
        return value
    compact = " ".join(value.split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[:max_chars]}...[truncated]"


class VoicePipeline:
    def __init__(
        self,
        *,
        llm: LlmEngine,
        asr: AsrEngine | None = None,
        tts: TtsEngine | None = None,
        rules_path: Path | None = None,
        object_mapping_path: Path | None = None,
        arm_rules_path: Path | None = None,
        rule_edit_strategy: str = RULE_EDIT_STRATEGY_TWO_PASS,
        vision_snapshot_provider: VisionSnapshotProvider | None = None,
        require_confirmation_for_side_effects: bool = False,
        pending_confirmation_provider: Callable[[], PendingConfirmation | None] | None = None,
    ) -> None:
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.rules_path = rules_path
        self.object_mapping_path = object_mapping_path
        self.arm_rules_path = arm_rules_path
        self.rule_edit_strategy = normalize_rule_edit_strategy(rule_edit_strategy)
        self.vision_snapshot_provider = vision_snapshot_provider
        self.require_confirmation_for_side_effects = require_confirmation_for_side_effects
        self.pending_confirmation_provider = pending_confirmation_provider

    def run_text_turn(self, text: str, *, synthesize: bool = True) -> VoiceTurnResult:
        started = time.perf_counter()
        normalized_text = normalize_asr_text(text)
        return self._run_normalized_turn(
            normalized_text,
            asr=None,
            started=started,
            synthesize=synthesize,
        )

    def run_audio_file(self, audio_path: Path, *, synthesize: bool = True) -> VoiceTurnResult:
        if self.asr is None:
            raise RuntimeError("ASR engine is required for audio-file turns.")
        started = time.perf_counter()
        asr_result = self.asr.transcribe_wav(audio_path)
        normalized_text = normalize_asr_text(asr_result.text)
        return self._run_normalized_turn(
            normalized_text,
            asr=asr_result,
            started=started,
            synthesize=synthesize,
        )

    def _run_normalized_turn(
        self,
        normalized_text: str,
        *,
        asr: AsrResult | None,
        started: float,
        synthesize: bool,
    ) -> VoiceTurnResult:
        if asr is not None and should_ignore_asr_noise(normalized_text):
            return self._run_ignored_asr_noise_turn(
                normalized_text,
                asr=asr,
                started=started,
            )
        if normalized_text in _EXACT_LLM_BYPASS_COMMANDS:
            response_text = deterministic_spoken_response(normalized_text)
            tts_result = self._maybe_synthesize(response_text, synthesize)
            return VoiceTurnResult(
                input_text=normalized_text,
                response_text=response_text,
                asr=asr,
                llm=_deterministic_llm_result(normalized_text, response_text),
                tts=tts_result,
                total_seconds=time.perf_counter() - started,
            )
        if should_reject_rule_authoring(normalized_text):
            return self._run_unsupported_rule_authoring_turn(
                normalized_text,
                asr=asr,
                started=started,
                synthesize=synthesize,
            )
        mapping_guidance = build_object_mapping_update_guidance(normalized_text)
        if mapping_guidance is not None:
            tts_result = self._maybe_synthesize(mapping_guidance, synthesize)
            return VoiceTurnResult(
                input_text=normalized_text,
                response_text=mapping_guidance,
                asr=asr,
                llm=_deterministic_llm_result(normalized_text, mapping_guidance),
                tts=tts_result,
                total_seconds=time.perf_counter() - started,
            )
        if should_guide_workspace_snapshot(normalized_text):
            return self._run_workspace_guidance_turn(
                normalized_text,
                asr=asr,
                started=started,
                synthesize=synthesize,
            )
        if should_query_arm_runtime(normalized_text):
            return self._run_arm_runtime_query_turn(
                normalized_text,
                primary_result=None,
                asr=asr,
                started=started,
                synthesize=synthesize,
            )
        if should_request_arm_deceleration(normalized_text):
            return self._run_arm_deceleration_turn(
                normalized_text,
                primary_result=None,
                asr=asr,
                started=started,
                synthesize=synthesize,
            )

        route_result = self._generate_agent_route(normalized_text)
        route_text = _llm_structured_text(route_result)
        route_decision = parse_agent_route_decision(route_text)
        _log_route_stage(normalized_text, route_result, route_text, route_decision)
        if route_decision is not None:
            return self._run_agent_route_decision(
                route_decision,
                normalized_text,
                primary_result=route_result,
                asr=asr,
                started=started,
                synthesize=synthesize,
            )
        if should_query_object_mapping_table(normalized_text):
            return self._run_object_mapping_table_query_turn(
                normalized_text,
                primary_result=route_result,
                asr=asr,
                started=started,
                synthesize=synthesize,
            )
        if should_query_object_mapping(normalized_text):
            return self._run_object_mapping_query_turn(
                normalized_text,
                primary_result=route_result,
                tool_request=None,
                asr=asr,
                started=started,
                synthesize=synthesize,
            )
        if should_update_object_mapping(normalized_text):
            return self._run_object_mapping_update_turn(
                normalized_text,
                primary_result=route_result,
                tool_request=None,
                asr=asr,
                started=started,
                synthesize=synthesize,
            )
        if should_query_arm_runtime(normalized_text):
            return self._run_arm_runtime_query_turn(
                normalized_text,
                primary_result=route_result,
                asr=asr,
                started=started,
                synthesize=synthesize,
            )
        if should_analyze_vision(normalized_text):
            return self._run_vision_analysis_turn(
                normalized_text,
                primary_result=route_result,
                asr=asr,
                started=started,
                synthesize=synthesize,
            )
        if should_grasp_object(normalized_text):
            return self._run_object_grasp_turn(
                normalized_text,
                primary_result=route_result,
                asr=asr,
                started=started,
                synthesize=synthesize,
            )

        return self._run_legacy_normalized_turn(
            normalized_text,
            asr=asr,
            started=started,
            synthesize=synthesize,
        )

    def _generate_agent_route(self, user_text: str) -> LlmResult:
        route_prompt = build_agent_router_prompt(user_text)
        generate_structured = getattr(self.llm, "generate_structured_json", None)
        if callable(generate_structured):
            return generate_structured(
                route_prompt,
                json_schema=AGENT_ROUTE_JSON_SCHEMA,
                regex=AGENT_ROUTE_REGEX,
                max_new_tokens=128,
            )
        return self.llm.generate(route_prompt)

    def _run_agent_route_decision(
        self,
        route_decision: AgentRouteDecision,
        user_text: str,
        *,
        primary_result: LlmResult,
        asr: AsrResult | None,
        started: float,
        synthesize: bool,
    ) -> VoiceTurnResult:
        if route_decision.tool_request is not None:
            if route_decision.tool_request.name == AGENT_TOOL_EDIT_RULES:
                if not should_edit_rules(user_text):
                    return self._run_rule_edit_clarification_turn(
                        user_text,
                        llm_result=primary_result,
                        asr=asr,
                        started=started,
                        synthesize=synthesize,
                    )
                if should_read_rules(user_text):
                    return self._run_rule_read_turn(
                        user_text,
                        primary_result=primary_result,
                        asr=asr,
                        started=started,
                        synthesize=synthesize,
                    )
            return self._run_agent_tool_turn(
                route_decision.tool_request,
                user_text,
                primary_result=primary_result,
                asr=asr,
                started=started,
                synthesize=synthesize,
            )

        if should_query_object_mapping_table(user_text):
            return self._run_object_mapping_table_query_turn(
                user_text,
                primary_result=primary_result,
                asr=asr,
                started=started,
                synthesize=synthesize,
            )

        if should_query_object_mapping(user_text):
            return self._run_object_mapping_query_turn(
                user_text,
                primary_result=primary_result,
                tool_request=None,
                asr=asr,
                started=started,
                synthesize=synthesize,
            )

        if should_update_object_mapping(user_text):
            return self._run_object_mapping_update_turn(
                user_text,
                primary_result=primary_result,
                tool_request=None,
                asr=asr,
                started=started,
                synthesize=synthesize,
            )

        if should_query_arm_runtime(user_text):
            return self._run_arm_runtime_query_turn(
                user_text,
                primary_result=primary_result,
                asr=asr,
                started=started,
                synthesize=synthesize,
            )

        if should_analyze_vision(user_text):
            return self._run_vision_analysis_turn(
                user_text,
                primary_result=primary_result,
                asr=asr,
                started=started,
                synthesize=synthesize,
            )

        if should_grasp_object(user_text):
            return self._run_object_grasp_turn(
                user_text,
                primary_result=primary_result,
                asr=asr,
                started=started,
                synthesize=synthesize,
            )

        response_text = build_spoken_response(user_text, route_decision.final_content or "")
        tts_result = self._maybe_synthesize(response_text, synthesize)
        return VoiceTurnResult(
            input_text=user_text,
            response_text=response_text,
            asr=asr,
            llm=primary_result,
            tts=tts_result,
            total_seconds=time.perf_counter() - started,
        )

    def _run_legacy_normalized_turn(
        self,
        normalized_text: str,
        *,
        asr: AsrResult | None,
        started: float,
        synthesize: bool,
    ) -> VoiceTurnResult:
        rule_edit_intent = should_edit_rules(normalized_text)
        if rule_edit_intent and self.rule_edit_strategy == RULE_EDIT_STRATEGY_ONE_PASS:
            return self._run_rule_edit_turn(
                normalized_text,
                primary_result=None,
                tool_request=None,
                strategy=RULE_EDIT_STRATEGY_ONE_PASS,
                asr=asr,
                started=started,
                synthesize=synthesize,
            )

        llm_result = self.llm.generate(normalized_text)
        model_text = _llm_structured_text(llm_result)
        final_content = parse_agent_final_response(model_text)
        if final_content is not None:
            model_text = final_content
        tool_request = parse_agent_tool_request(model_text)
        rule_read_intent = should_read_rules(normalized_text)
        if rule_edit_intent:
            return self._run_rule_edit_turn(
                normalized_text,
                primary_result=llm_result,
                tool_request=tool_request if tool_request and tool_request.name == AGENT_TOOL_EDIT_RULES else None,
                strategy=RULE_EDIT_STRATEGY_TWO_PASS,
                asr=asr,
                started=started,
                synthesize=synthesize,
            )

        if tool_request is not None:
            if tool_request.name == AGENT_TOOL_EDIT_RULES and rule_read_intent:
                return self._run_rule_read_turn(
                    normalized_text,
                    primary_result=llm_result,
                    asr=asr,
                    started=started,
                    synthesize=synthesize,
                )
            return self._run_agent_tool_turn(
                tool_request,
                normalized_text,
                primary_result=llm_result,
                asr=asr,
                started=started,
                synthesize=synthesize,
            )
        if rule_read_intent:
            return self._run_rule_read_turn(
                normalized_text,
                primary_result=llm_result,
                asr=asr,
                started=started,
                synthesize=synthesize,
            )

        response_text = build_spoken_response(normalized_text, model_text)
        tts_result = self._maybe_synthesize(response_text, synthesize)
        return VoiceTurnResult(
            input_text=normalized_text,
            response_text=response_text,
            asr=asr,
            llm=llm_result,
            tts=tts_result,
            total_seconds=time.perf_counter() - started,
        )

    def _run_ignored_asr_noise_turn(
        self,
        user_text: str,
        *,
        asr: AsrResult,
        started: float,
    ) -> VoiceTurnResult:
        diagnostic_text = _diagnostic_value(user_text, max_chars=120) or "<empty>"
        LOGGER.info("已忽略疑似 ASR 噪声：%s", diagnostic_text)
        return VoiceTurnResult(
            input_text=user_text,
            response_text="",
            asr=asr,
            llm=_deterministic_llm_result(user_text, ""),
            tts=None,
            total_seconds=time.perf_counter() - started,
            metadata={"no_answer": True, "reason": "asr_noise"},
        )

    def _run_unsupported_rule_authoring_turn(
        self,
        user_text: str,
        *,
        asr: AsrResult | None,
        started: float,
        synthesize: bool,
    ) -> VoiceTurnResult:
        response_text = "首版只支持修改已有安全规则，不支持新增或删除规则。"
        tts_result = self._maybe_synthesize(response_text, synthesize)
        return VoiceTurnResult(
            input_text=user_text,
            response_text=response_text,
            asr=asr,
            llm=_deterministic_llm_result(user_text, response_text),
            tts=tts_result,
            total_seconds=time.perf_counter() - started,
        )

    def _run_agent_tool_turn(
        self,
        tool_request: AgentToolRequest,
        user_text: str,
        *,
        primary_result: LlmResult,
        asr: AsrResult | None,
        started: float,
        synthesize: bool,
    ) -> VoiceTurnResult:
        if should_grasp_object(user_text):
            return self._run_object_grasp_turn(
                user_text,
                primary_result=primary_result,
                asr=asr,
                started=started,
                synthesize=synthesize,
            )
        if (
            tool_request.name in {AGENT_TOOL_GET_OBJECT_MAPPING, AGENT_TOOL_UPDATE_OBJECT_MAPPING}
            and should_query_object_mapping_table(user_text)
        ):
            return self._run_object_mapping_table_query_turn(
                user_text,
                primary_result=primary_result,
                asr=asr,
                started=started,
                synthesize=synthesize,
            )
        if tool_request.name == AGENT_TOOL_UPDATE_OBJECT_MAPPING and should_query_object_mapping(user_text):
            return self._run_object_mapping_query_turn(
                user_text,
                primary_result=primary_result,
                tool_request=tool_request,
                asr=asr,
                started=started,
                synthesize=synthesize,
            )
        if tool_request.name == AGENT_TOOL_GET_OBJECT_MAPPING and should_update_object_mapping(user_text):
            return self._run_object_mapping_update_turn(
                user_text,
                primary_result=primary_result,
                tool_request=tool_request,
                asr=asr,
                started=started,
                synthesize=synthesize,
            )
        if tool_request.name == AGENT_TOOL_LOAD_RULES:
            return self._run_rule_read_turn(
                user_text,
                primary_result=primary_result,
                asr=asr,
                started=started,
                synthesize=synthesize,
            )
        if tool_request.name == AGENT_TOOL_EDIT_RULES:
            return self._run_rule_edit_turn(
                user_text,
                primary_result=primary_result,
                tool_request=tool_request,
                strategy=self.rule_edit_strategy,
                asr=asr,
                started=started,
                synthesize=synthesize,
            )
        if tool_request.name == AGENT_TOOL_ANALYZE_VISION:
            return self._run_vision_analysis_turn(
                user_text,
                primary_result=primary_result,
                asr=asr,
                started=started,
                synthesize=synthesize,
            )
        if tool_request.name == AGENT_TOOL_GET_OBJECT_MAPPING:
            return self._run_object_mapping_query_turn(
                user_text,
                primary_result=primary_result,
                tool_request=tool_request,
                asr=asr,
                started=started,
                synthesize=synthesize,
            )
        if tool_request.name == AGENT_TOOL_UPDATE_OBJECT_MAPPING:
            return self._run_object_mapping_update_turn(
                user_text,
                primary_result=primary_result,
                tool_request=tool_request,
                asr=asr,
                started=started,
                synthesize=synthesize,
            )
        if should_update_object_mapping(user_text):
            return self._run_object_mapping_update_turn(
                user_text,
                primary_result=primary_result,
                tool_request=None,
                asr=asr,
                started=started,
                synthesize=synthesize,
            )

        response_text = build_unknown_tool_response(tool_request.name, user_text)
        tts_result = self._maybe_synthesize(response_text, synthesize)
        return VoiceTurnResult(
            input_text=user_text,
            response_text=response_text,
            asr=asr,
            llm=primary_result,
            tts=tts_result,
            total_seconds=time.perf_counter() - started,
        )

    def _run_workspace_guidance_turn(
        self,
        user_text: str,
        *,
        asr: AsrResult | None,
        started: float,
        synthesize: bool,
    ) -> VoiceTurnResult:
        snapshot = self._load_workspace_snapshot()
        response_text = build_workspace_snapshot_guidance_response(snapshot)
        tts_result = self._maybe_synthesize(response_text, synthesize)
        return VoiceTurnResult(
            input_text=user_text,
            response_text=response_text,
            asr=asr,
            llm=_deterministic_llm_result(user_text, response_text),
            tts=tts_result,
            total_seconds=time.perf_counter() - started,
        )

    def _run_arm_runtime_query_turn(
        self,
        user_text: str,
        *,
        primary_result: LlmResult | None,
        asr: AsrResult | None,
        started: float,
        synthesize: bool,
    ) -> VoiceTurnResult:
        snapshot = self._load_workspace_snapshot(
            include_rules=False,
            include_object_mapping=False,
            include_arm_runtime=True,
        )
        response_text = build_arm_runtime_query_response(snapshot)
        tts_result = self._maybe_synthesize(response_text, synthesize)
        arm_runtime = snapshot.arm_runtime
        return VoiceTurnResult(
            input_text=user_text,
            response_text=response_text,
            asr=asr,
            llm=primary_result or _deterministic_llm_result(user_text, response_text),
            tts=tts_result,
            total_seconds=time.perf_counter() - started,
            arm_runtime_query=_arm_runtime_query_result(arm_runtime) if arm_runtime else None,
        )

    def _run_arm_deceleration_turn(
        self,
        user_text: str,
        *,
        primary_result: LlmResult | None,
        asr: AsrResult | None,
        started: float,
        synthesize: bool,
    ) -> VoiceTurnResult:
        target_percent, error_text = extract_arm_deceleration_target_percent(user_text)
        if error_text is not None:
            response_text = error_text
            tts_result = self._maybe_synthesize(response_text, synthesize)
            return VoiceTurnResult(
                input_text=user_text,
                response_text=response_text,
                asr=asr,
                llm=primary_result or _deterministic_llm_result(user_text, response_text),
                tts=tts_result,
                total_seconds=time.perf_counter() - started,
            )

        assert target_percent is not None
        if self.arm_rules_path is None:
            response_text = "这是机械臂速度调整请求，但当前未配置机械臂运行时 JSON 文件。"
            tts_result = self._maybe_synthesize(response_text, synthesize)
            return VoiceTurnResult(
                input_text=user_text,
                response_text=response_text,
                asr=asr,
                llm=primary_result or _deterministic_llm_result(user_text, response_text),
                tts=tts_result,
                total_seconds=time.perf_counter() - started,
            )

        arm_decelerate = format_arm_deceleration_fraction(target_percent)
        if self.require_confirmation_for_side_effects:
            confirmation = build_speed_change_confirmation(
                user_text,
                target_speed=format_arm_deceleration_percent(target_percent),
                target_speed_percent=target_percent,
                arm_decelerate=arm_decelerate,
                arm_rules_path=str(self.arm_rules_path),
            )
            response_text = build_confirmation_required_response(confirmation)
            tts_result = self._maybe_synthesize(response_text, synthesize)
            return VoiceTurnResult(
                input_text=user_text,
                response_text=response_text,
                asr=asr,
                llm=primary_result or _deterministic_llm_result(user_text, response_text),
                tts=tts_result,
                total_seconds=time.perf_counter() - started,
                pending_confirmation=confirmation,
            )

        try:
            result = request_arm_deceleration(
                self.arm_rules_path,
                target_speed_percent=target_percent,
            )
        except (OSError, ValueError) as error:
            response_text = build_arm_deceleration_rejection_response(error)
            tts_result = self._maybe_synthesize(response_text, synthesize)
            return VoiceTurnResult(
                input_text=user_text,
                response_text=response_text,
                asr=asr,
                llm=primary_result or _deterministic_llm_result(user_text, response_text),
                tts=tts_result,
                total_seconds=time.perf_counter() - started,
            )

        response_text = build_arm_deceleration_success_response(
            result.target_speed_percent,
        )
        tts_result = self._maybe_synthesize(response_text, synthesize)
        return VoiceTurnResult(
            input_text=user_text,
            response_text=response_text,
            asr=asr,
            llm=primary_result or _deterministic_llm_result(user_text, response_text),
            tts=tts_result,
            total_seconds=time.perf_counter() - started,
            arm_deceleration_request=result,
        )

    def _run_rule_read_turn(
        self,
        user_text: str,
        *,
        primary_result: LlmResult,
        asr: AsrResult | None,
        started: float,
        synthesize: bool,
    ) -> VoiceTurnResult:
        if self.rules_path is None:
            response_text = "当前未配置安全规则文件，无法读取当前规则。"
            tts_result = self._maybe_synthesize(response_text, synthesize)
            return VoiceTurnResult(
                input_text=user_text,
                response_text=response_text,
                asr=asr,
                llm=primary_result,
                tts=tts_result,
                total_seconds=time.perf_counter() - started,
            )

        try:
            document = load_rule_document(self.rules_path)
        except (OSError, ValueError) as error:
            response_text = f"读取安全规则失败：{error}"
            tts_result = self._maybe_synthesize(response_text, synthesize)
            return VoiceTurnResult(
                input_text=user_text,
                response_text=response_text,
                asr=asr,
                llm=primary_result,
                tts=tts_result,
                total_seconds=time.perf_counter() - started,
            )

        rule_prompt = build_rule_read_prompt(user_text, document)
        tool_result = self.llm.generate(rule_prompt)
        tool_text = _llm_structured_text(tool_result)
        final_content = parse_agent_final_response(tool_text)
        if final_content is not None:
            tool_text = final_content
        fallback_response = build_rule_read_fallback_response(document)
        used_fallback = parse_agent_tool_request(tool_text) is not None
        if used_fallback:
            response_text = fallback_response
        else:
            response_text = build_rule_read_response(user_text, tool_text, document)
            used_fallback = response_text == fallback_response
        _log_rule_read_stage(user_text, tool_result, tool_text, response_text, used_fallback=used_fallback)
        tts_result = self._maybe_synthesize(response_text, synthesize)
        return VoiceTurnResult(
            input_text=user_text,
            response_text=response_text,
            asr=asr,
            llm=tool_result,
            tts=tts_result,
            total_seconds=time.perf_counter() - started,
        )

    def _run_rule_edit_turn(
        self,
        user_text: str,
        *,
        primary_result: LlmResult | None,
        tool_request: AgentToolRequest | None,
        strategy: str,
        asr: AsrResult | None,
        started: float,
        synthesize: bool,
    ) -> VoiceTurnResult:
        if self.rules_path is None:
            response_text = "这是规则修改请求，但当前未配置安全规则文件。"
            tts_result = self._maybe_synthesize(response_text, synthesize)
            return VoiceTurnResult(
                input_text=user_text,
                response_text=response_text,
                asr=asr,
                llm=primary_result or _deterministic_llm_result(user_text, response_text),
                tts=tts_result,
                total_seconds=time.perf_counter() - started,
            )

        try:
            document = load_rule_document(self.rules_path)
        except (OSError, ValueError) as error:
            response_text = f"读取安全规则失败：{error}"
            tts_result = self._maybe_synthesize(response_text, synthesize)
            return VoiceTurnResult(
                input_text=user_text,
                response_text=response_text,
                asr=asr,
                llm=primary_result or _deterministic_llm_result(user_text, response_text),
                tts=tts_result,
                total_seconds=time.perf_counter() - started,
            )

        patch_result = primary_result
        patch_payload: dict[str, Any] | None = None
        if tool_request is not None and tool_request.arguments:
            patch_payload = _normalize_rule_patch_payload(user_text, document, tool_request.arguments)
        else:
            patch_result = self._generate_rule_edit_patch(user_text, document, strategy=strategy)
            patch_text = _llm_structured_text(patch_result)
            try:
                patch_payload = extract_rule_patch_payload(patch_text)
            except (RuleValidationError, ValueError) as error:
                patch_payload = _repair_rule_patch_payload(user_text, document, _extract_first_json_value(patch_text))
                if patch_payload is None:
                    return self._run_rejected_rule_patch_turn(
                        user_text,
                        reason=error,
                        llm_result=patch_result,
                        asr=asr,
                        started=started,
                        synthesize=synthesize,
                    )

        if patch_result is None:
            patch_result = self._generate_rule_edit_patch(user_text, document, strategy=strategy)
            patch_text = _llm_structured_text(patch_result)
            try:
                patch_payload = extract_rule_patch_payload(patch_text)
            except (RuleValidationError, ValueError) as error:
                patch_payload = _repair_rule_patch_payload(user_text, document, _extract_first_json_value(patch_text))
                if patch_payload is None:
                    return self._run_rejected_rule_patch_turn(
                        user_text,
                        reason=error,
                        llm_result=patch_result,
                        asr=asr,
                        started=started,
                        synthesize=synthesize,
                    )

        assert patch_payload is not None
        if not _rule_patch_rule_exists(document, patch_payload):
            repaired_payload = _repair_rule_patch_payload(
                user_text,
                document,
                patch_payload,
                repair_existing_patch=True,
            )
            if repaired_payload is not None:
                patch_payload = repaired_payload
        if not _rule_patch_is_grounded_in_request(user_text, document, patch_payload):
            return self._run_rule_edit_clarification_turn(
                user_text,
                llm_result=patch_result,
                asr=asr,
                started=started,
                synthesize=synthesize,
            )
        if self.require_confirmation_for_side_effects:
            try:
                preview = preview_rule_patch_changes(document, patch_payload)
            except (OSError, ValueError) as error:
                return self._run_rejected_rule_patch_turn(
                    user_text,
                    reason=error,
                    llm_result=patch_result,
                    asr=asr,
                    started=started,
                    synthesize=synthesize,
                )
            if not preview.changed:
                response_text = build_rule_edit_noop_response(document, patch_payload, preview.changes)
                tts_result = self._maybe_synthesize(response_text, synthesize)
                return VoiceTurnResult(
                    input_text=user_text,
                    response_text=response_text,
                    asr=asr,
                    llm=patch_result,
                    tts=tts_result,
                    total_seconds=time.perf_counter() - started,
                )
            rule_id = str(patch_payload.get("rule_id") or patch_payload.get("id"))
            rule_name = _rule_edit_target_label(document, rule_id)
            change_summary = _rule_patch_value_transition_summary(preview.changes, changed_only=True)
            confirmation = build_rule_edit_confirmation(
                user_text,
                rule_id=rule_id,
                rule_name=rule_name,
                patch=patch_payload,
                rules_path=str(self.rules_path),
                change_summary=change_summary,
            )
            response_text = build_confirmation_required_response(confirmation)
            tts_result = self._maybe_synthesize(response_text, synthesize)
            return VoiceTurnResult(
                input_text=user_text,
                response_text=response_text,
                asr=asr,
                llm=patch_result,
                tts=tts_result,
                total_seconds=time.perf_counter() - started,
                pending_confirmation=confirmation,
            )

        updated_document: dict[str, Any] | None = None
        try:
            preview = preview_rule_patch_changes(document, patch_payload)
            if not preview.changed:
                response_text = build_rule_edit_noop_response(document, patch_payload, preview.changes)
                tts_result = self._maybe_synthesize(response_text, synthesize)
                return VoiceTurnResult(
                    input_text=user_text,
                    response_text=response_text,
                    asr=asr,
                    llm=patch_result,
                    tts=tts_result,
                    total_seconds=time.perf_counter() - started,
                )
            updated_document = apply_rule_patch(self.rules_path, patch_payload)
        except (OSError, ValueError) as error:
            repaired_payload = _repair_rule_patch_payload(user_text, document, patch_payload, repair_existing_patch=True)
            if repaired_payload is not None and repaired_payload != patch_payload:
                try:
                    repaired_preview = preview_rule_patch_changes(document, repaired_payload)
                    if not repaired_preview.changed:
                        response_text = build_rule_edit_noop_response(document, repaired_payload, repaired_preview.changes)
                        tts_result = self._maybe_synthesize(response_text, synthesize)
                        return VoiceTurnResult(
                            input_text=user_text,
                            response_text=response_text,
                            asr=asr,
                            llm=patch_result,
                            tts=tts_result,
                            total_seconds=time.perf_counter() - started,
                        )
                    updated_document = apply_rule_patch(self.rules_path, repaired_payload)
                except (OSError, ValueError):
                    repaired_payload = None
                else:
                    patch_payload = repaired_payload
            if repaired_payload is None:
                return self._run_rejected_rule_patch_turn(
                    user_text,
                    reason=error,
                    llm_result=patch_result,
                    asr=asr,
                    started=started,
                    synthesize=synthesize,
                )
            if updated_document is None:
                return self._run_rejected_rule_patch_turn(
                    user_text,
                    reason=error,
                    llm_result=patch_result,
                    asr=asr,
                    started=started,
                    synthesize=synthesize,
                )

        response_text = build_rule_edit_success_response(document, updated_document, patch_payload)
        sync_message = self._sync_personnel_distance_to_arm_rules(document, updated_document)
        if sync_message:
            response_text = f"{response_text}{sync_message}"
        rule_id = str(patch_payload.get("rule_id") or patch_payload.get("id"))
        tts_result = self._maybe_synthesize(response_text, synthesize)
        return VoiceTurnResult(
            input_text=user_text,
            response_text=response_text,
            asr=asr,
            llm=patch_result,
            tts=tts_result,
            total_seconds=time.perf_counter() - started,
            rule_update=RuleUpdateResult(
                rules_path=self.rules_path,
                rule_id=rule_id,
                version=int(updated_document["version"]),
                patch=patch_payload,
                patch_llm=patch_result,
                strategy=strategy,
            ),
        )

    def _run_object_mapping_update_turn(
        self,
        user_text: str,
        *,
        primary_result: LlmResult,
        tool_request: AgentToolRequest | None,
        asr: AsrResult | None,
        started: float,
        synthesize: bool,
    ) -> VoiceTurnResult:
        if self.object_mapping_path is None:
            response_text = "这是物体标记映射更新请求，但当前未配置映射文件。"
            tts_result = self._maybe_synthesize(response_text, synthesize)
            return VoiceTurnResult(
                input_text=user_text,
                response_text=response_text,
                asr=asr,
                llm=primary_result,
                tts=tts_result,
                total_seconds=time.perf_counter() - started,
            )

        marker, object_name, error_text = extract_object_mapping_update(
            user_text,
            tool_request.arguments if tool_request else None,
        )
        if error_text is not None:
            response_text = error_text
            tts_result = self._maybe_synthesize(response_text, synthesize)
            return VoiceTurnResult(
                input_text=user_text,
                response_text=response_text,
                asr=asr,
                llm=primary_result,
                tts=tts_result,
                total_seconds=time.perf_counter() - started,
            )

        assert marker is not None
        assert object_name is not None
        if self.require_confirmation_for_side_effects:
            try:
                document = load_object_mapping_document(self.object_mapping_path)
                preview = preview_object_mapping_update_changes(
                    document,
                    marker=marker,
                    object_name=object_name,
                )
            except (OSError, ValueError) as error:
                response_text = build_object_mapping_update_rejection_response(error)
                tts_result = self._maybe_synthesize(response_text, synthesize)
                return VoiceTurnResult(
                    input_text=user_text,
                    response_text=response_text,
                    asr=asr,
                    llm=primary_result,
                    tts=tts_result,
                    total_seconds=time.perf_counter() - started,
                )
            if not preview.changed:
                response_text = build_object_mapping_update_noop_response(
                    preview.marker,
                    preview.previous_object,
                )
                tts_result = self._maybe_synthesize(response_text, synthesize)
                return VoiceTurnResult(
                    input_text=user_text,
                    response_text=response_text,
                    asr=asr,
                    llm=primary_result,
                    tts=tts_result,
                    total_seconds=time.perf_counter() - started,
                )
            confirmation = build_object_mapping_update_confirmation(
                user_text,
                marker=preview.marker,
                object_name=preview.new_object,
                object_mapping_path=str(self.object_mapping_path),
                previous_object=preview.previous_object,
            )
            response_text = build_confirmation_required_response(confirmation)
            tts_result = self._maybe_synthesize(response_text, synthesize)
            return VoiceTurnResult(
                input_text=user_text,
                response_text=response_text,
                asr=asr,
                llm=primary_result,
                tts=tts_result,
                total_seconds=time.perf_counter() - started,
                pending_confirmation=confirmation,
            )

        try:
            document = load_object_mapping_document(self.object_mapping_path)
            preview = preview_object_mapping_update_changes(
                document,
                marker=marker,
                object_name=object_name,
            )
            if not preview.changed:
                response_text = build_object_mapping_update_noop_response(
                    preview.marker,
                    preview.previous_object,
                )
                tts_result = self._maybe_synthesize(response_text, synthesize)
                return VoiceTurnResult(
                    input_text=user_text,
                    response_text=response_text,
                    asr=asr,
                    llm=primary_result,
                    tts=tts_result,
                    total_seconds=time.perf_counter() - started,
                )
            updated_document = update_object_mapping(
                self.object_mapping_path,
                marker=preview.marker,
                object_name=preview.new_object,
            )
        except (OSError, ValueError) as error:
            response_text = build_object_mapping_update_rejection_response(error)
            tts_result = self._maybe_synthesize(response_text, synthesize)
            return VoiceTurnResult(
                input_text=user_text,
                response_text=response_text,
                asr=asr,
                llm=primary_result,
                tts=tts_result,
                total_seconds=time.perf_counter() - started,
            )

        version = updated_document.get("version")
        response_text = build_object_mapping_update_success_response(
            preview.marker,
            preview.new_object,
            previous_object=preview.previous_object,
        )
        tts_result = self._maybe_synthesize(response_text, synthesize)
        return VoiceTurnResult(
            input_text=user_text,
            response_text=response_text,
            asr=asr,
            llm=primary_result,
            tts=tts_result,
            total_seconds=time.perf_counter() - started,
            object_mapping_update=ObjectMappingUpdateResult(
                object_mapping_path=self.object_mapping_path,
                marker=preview.marker,
                object_name=preview.new_object,
                version=version if isinstance(version, int) else -1,
            ),
        )

    def _run_object_mapping_query_turn(
        self,
        user_text: str,
        *,
        primary_result: LlmResult,
        tool_request: AgentToolRequest | None,
        asr: AsrResult | None,
        started: float,
        synthesize: bool,
    ) -> VoiceTurnResult:
        if self.object_mapping_path is None:
            response_text = "这是物体标记映射查询请求，但当前未配置映射文件。"
            tts_result = self._maybe_synthesize(response_text, synthesize)
            return VoiceTurnResult(
                input_text=user_text,
                response_text=response_text,
                asr=asr,
                llm=primary_result,
                tts=tts_result,
                total_seconds=time.perf_counter() - started,
            )

        marker, error_text = extract_object_mapping_query(
            user_text,
            tool_request.arguments if tool_request else None,
        )
        if error_text is not None:
            response_text = error_text
            tts_result = self._maybe_synthesize(response_text, synthesize)
            return VoiceTurnResult(
                input_text=user_text,
                response_text=response_text,
                asr=asr,
                llm=primary_result,
                tts=tts_result,
                total_seconds=time.perf_counter() - started,
            )

        assert marker is not None
        try:
            mapping = get_object_mapping(self.object_mapping_path, marker=marker)
        except (OSError, ValueError) as error:
            response_text = build_object_mapping_query_rejection_response(error)
            tts_result = self._maybe_synthesize(response_text, synthesize)
            return VoiceTurnResult(
                input_text=user_text,
                response_text=response_text,
                asr=asr,
                llm=primary_result,
                tts=tts_result,
                total_seconds=time.perf_counter() - started,
            )

        object_name = str(mapping.get("object", "")).strip()
        version = mapping.get("version")
        enabled = mapping.get("enabled", True)
        response_text = build_object_mapping_query_success_response(
            marker,
            object_name,
            enabled=enabled if isinstance(enabled, bool) else True,
        )
        tts_result = self._maybe_synthesize(response_text, synthesize)
        return VoiceTurnResult(
            input_text=user_text,
            response_text=response_text,
            asr=asr,
            llm=primary_result,
            tts=tts_result,
            total_seconds=time.perf_counter() - started,
            object_mapping_query=ObjectMappingQueryResult(
                object_mapping_path=self.object_mapping_path,
                marker=marker,
                object_name=object_name,
                version=version if isinstance(version, int) else -1,
                enabled=enabled if isinstance(enabled, bool) else True,
            ),
        )

    def _run_object_mapping_table_query_turn(
        self,
        user_text: str,
        *,
        primary_result: LlmResult,
        asr: AsrResult | None,
        started: float,
        synthesize: bool,
    ) -> VoiceTurnResult:
        if self.object_mapping_path is None:
            response_text = "这是物体标记映射表查询请求，但当前未配置映射文件。"
            tts_result = self._maybe_synthesize(response_text, synthesize)
            return VoiceTurnResult(
                input_text=user_text,
                response_text=response_text,
                asr=asr,
                llm=primary_result,
                tts=tts_result,
                total_seconds=time.perf_counter() - started,
            )

        snapshot = self._load_workspace_snapshot(
            include_rules=False,
            include_object_mapping=True,
            include_arm_runtime=False,
        )
        if snapshot.object_mapping is None:
            error = snapshot.error_for("object_mapping")
            reason = ValueError(error.message) if error is not None else ValueError("映射文件不可用。")
            response_text = build_object_mapping_query_rejection_response(reason)
            tts_result = self._maybe_synthesize(response_text, synthesize)
            return VoiceTurnResult(
                input_text=user_text,
                response_text=response_text,
                asr=asr,
                llm=primary_result,
                tts=tts_result,
                total_seconds=time.perf_counter() - started,
            )

        mappings = _object_mapping_table_entries_from_snapshot(snapshot.object_mapping.entries)
        response_text = build_object_mapping_table_query_success_response(mappings)
        tts_result = self._maybe_synthesize(response_text, synthesize)
        return VoiceTurnResult(
            input_text=user_text,
            response_text=response_text,
            asr=asr,
            llm=primary_result,
            tts=tts_result,
            total_seconds=time.perf_counter() - started,
            object_mapping_table_query=ObjectMappingTableQueryResult(
                object_mapping_path=self.object_mapping_path,
                version=snapshot.object_mapping.version,
                mappings=mappings,
            ),
        )

    def _run_object_grasp_turn(
        self,
        user_text: str,
        *,
        primary_result: LlmResult,
        asr: AsrResult | None,
        started: float,
        synthesize: bool,
    ) -> VoiceTurnResult:
        if self.object_mapping_path is None:
            response_text = "这是物体抓取请求，但当前未配置物体映射文件。"
            tts_result = self._maybe_synthesize(response_text, synthesize)
            return VoiceTurnResult(
                input_text=user_text,
                response_text=response_text,
                asr=asr,
                llm=primary_result,
                tts=tts_result,
                total_seconds=time.perf_counter() - started,
            )

        marker, object_name, target_source, error_text = extract_object_grasp_target(user_text)
        if error_text is not None:
            response_text = error_text
            tts_result = self._maybe_synthesize(response_text, synthesize)
            return VoiceTurnResult(
                input_text=user_text,
                response_text=response_text,
                asr=asr,
                llm=primary_result,
                tts=tts_result,
                total_seconds=time.perf_counter() - started,
            )

        try:
            resolved = resolve_object_grasp_target(
                self.object_mapping_path,
                marker=marker,
                object_name=object_name,
            )
        except (OSError, ValueError) as error:
            response_text = build_object_grasp_rejection_response(error)
            tts_result = self._maybe_synthesize(response_text, synthesize)
            return VoiceTurnResult(
                input_text=user_text,
                response_text=response_text,
                asr=asr,
                llm=primary_result,
                tts=tts_result,
                total_seconds=time.perf_counter() - started,
            )

        resolved_marker = str(resolved.get("marker", "")).strip()
        resolved_object = normalize_object_name(str(resolved.get("object", "")))
        resolved_source = str(resolved.get("target_source") or target_source or "unknown")
        version = resolved.get("version")
        response_text = build_object_grasp_success_response(
            resolved_marker,
            resolved_object,
            target_source=resolved_source,
        )
        if self.require_confirmation_for_side_effects:
            confirmation = build_object_grasp_execution_confirmation(
                user_text,
                marker=resolved_marker,
                object_name=resolved_object,
            )
            response_text = build_confirmation_required_response(confirmation)
            tts_result = self._maybe_synthesize(response_text, synthesize)
            return VoiceTurnResult(
                input_text=user_text,
                response_text=response_text,
                asr=asr,
                llm=primary_result,
                tts=tts_result,
                total_seconds=time.perf_counter() - started,
                pending_confirmation=confirmation,
            )
        tts_result = self._maybe_synthesize(response_text, synthesize)
        return VoiceTurnResult(
            input_text=user_text,
            response_text=response_text,
            asr=asr,
            llm=primary_result,
            tts=tts_result,
            total_seconds=time.perf_counter() - started,
            object_grasp_intent=ObjectGraspIntentResult(
                object_mapping_path=self.object_mapping_path,
                marker=resolved_marker,
                object_name=resolved_object,
                version=version if isinstance(version, int) else -1,
                target_source=resolved_source,
                original_text=user_text,
            ),
        )

    def _run_vision_analysis_turn(
        self,
        user_text: str,
        *,
        primary_result: LlmResult,
        asr: AsrResult | None,
        started: float,
        synthesize: bool,
    ) -> VoiceTurnResult:
        if self.vision_snapshot_provider is None:
            response_text = "当前未配置视觉快照服务，无法获取相机图像。"
            tts_result = self._maybe_synthesize(response_text, synthesize)
            return VoiceTurnResult(
                input_text=user_text,
                response_text=response_text,
                asr=asr,
                llm=primary_result,
                tts=tts_result,
                total_seconds=time.perf_counter() - started,
            )

        try:
            artifact = self.vision_snapshot_provider.capture_snapshot()
        except Exception as error:
            response_text = f"视觉服务当前不可用：{format_vision_snapshot_error(error)}"
            tts_result = self._maybe_synthesize(response_text, synthesize)
            return VoiceTurnResult(
                input_text=user_text,
                response_text=response_text,
                asr=asr,
                llm=primary_result,
                tts=tts_result,
                total_seconds=time.perf_counter() - started,
            )

        generate_with_image = getattr(self.llm, "generate_with_image", None)
        if not callable(generate_with_image):
            response_text = "已获取相机图像，但当前 LLM 未配置视觉分析能力。"
            tts_result = self._maybe_synthesize(response_text, synthesize)
            return VoiceTurnResult(
                input_text=user_text,
                response_text=response_text,
                asr=asr,
                llm=primary_result,
                tts=tts_result,
                total_seconds=time.perf_counter() - started,
                vision_artifacts=(artifact,),
            )

        try:
            vision_prompt = build_vision_analysis_prompt(user_text, artifact)
            vision_result = generate_with_image(
                vision_prompt,
                artifact.image_path,
                max_new_tokens=220,
                system_prompt=VISION_ANALYSIS_SYSTEM_PROMPT,
            )
            response_text = sanitize_spoken_response(_llm_structured_text(vision_result))
            if is_unusable_vision_analysis_response(response_text):
                response_text = VISION_ANALYSIS_FALLBACK_RESPONSE
        except Exception as error:
            response_text = f"已获取相机图像，但视觉分析失败：{error}"
            tts_result = self._maybe_synthesize(response_text, synthesize)
            return VoiceTurnResult(
                input_text=user_text,
                response_text=response_text,
                asr=asr,
                llm=primary_result,
                tts=tts_result,
                total_seconds=time.perf_counter() - started,
                vision_artifacts=(artifact,),
            )

        tts_result = self._maybe_synthesize(response_text, synthesize)
        return VoiceTurnResult(
            input_text=user_text,
            response_text=response_text,
            asr=asr,
            llm=vision_result,
            tts=tts_result,
            total_seconds=time.perf_counter() - started,
            vision_artifacts=(artifact,),
        )

    def _generate_rule_edit_patch(self, user_text: str, document: dict[str, Any], *, strategy: str) -> LlmResult:
        rule_prompt = build_rule_edit_patch_prompt(user_text, document, strategy=strategy)
        generate_structured = getattr(self.llm, "generate_structured_json", None)
        if callable(generate_structured):
            return generate_structured(
                rule_prompt,
                json_schema=RULE_EDIT_TOOL_JSON_SCHEMA,
                regex=RULE_EDIT_TOOL_REGEX,
                max_new_tokens=256,
            )
        return self.llm.generate(rule_prompt)

    def _run_rejected_rule_patch_turn(
        self,
        user_text: str,
        *,
        reason: Exception,
        llm_result: LlmResult,
        asr: AsrResult | None,
        started: float,
        synthesize: bool,
    ) -> VoiceTurnResult:
        response_text = build_rule_edit_rejection_response(reason)
        tts_result = self._maybe_synthesize(response_text, synthesize)
        return VoiceTurnResult(
            input_text=user_text,
            response_text=response_text,
            asr=asr,
            llm=llm_result,
            tts=tts_result,
            total_seconds=time.perf_counter() - started,
        )

    def _run_rule_edit_clarification_turn(
        self,
        user_text: str,
        *,
        llm_result: LlmResult,
        asr: AsrResult | None,
        started: float,
        synthesize: bool,
    ) -> VoiceTurnResult:
        response_text = _RULE_EDIT_CLARIFICATION_RESPONSE
        tts_result = self._maybe_synthesize(response_text, synthesize)
        return VoiceTurnResult(
            input_text=user_text,
            response_text=response_text,
            asr=asr,
            llm=llm_result,
            tts=tts_result,
            total_seconds=time.perf_counter() - started,
        )

    def _load_workspace_snapshot(
        self,
        *,
        include_rules: bool = True,
        include_object_mapping: bool = True,
        include_arm_runtime: bool = True,
    ) -> WorkspaceSnapshot:
        pending_confirmation = None
        if self.pending_confirmation_provider is not None:
            try:
                pending_confirmation = self.pending_confirmation_provider()
            except Exception as error:
                _log_pipeline_event("workspace_snapshot_pending_error", error=str(error))
        return load_workspace_snapshot(
            rules_path=self.rules_path if include_rules else None,
            object_mapping_path=self.object_mapping_path if include_object_mapping else None,
            arm_rules_path=self.arm_rules_path if include_arm_runtime else None,
            pending_confirmation=pending_confirmation,
        )

    def _sync_personnel_distance_to_arm_rules(
        self,
        previous_document: dict[str, Any],
        updated_document: dict[str, Any],
    ) -> str:
        if self.arm_rules_path is None:
            return ""
        try:
            result = sync_personnel_distance_to_arm_rules(
                previous_document,
                updated_document,
                self.arm_rules_path,
            )
        except (OSError, ValueError) as error:
            _log_pipeline_event(
                "personnel_distance_arm_rule_sync_failed",
                arm_rules_path=str(self.arm_rules_path),
                error=str(error),
            )
            return f" 但机械臂运行时安全距离未同步：{error}"
        if not result.synced or result.distance_m is None:
            return ""
        distance_text = _format_compact_number(result.distance_m)
        return f" 机械臂运行时安全距离已同步为 {distance_text} 米。"

    def _maybe_synthesize(self, text: str, synthesize: bool) -> TtsResult | None:
        if not synthesize or self.tts is None:
            return None
        return self.tts.synthesize(text)


ASR_CORRECTIONS: tuple[tuple[str, str], ...] = (
    ("詳細", "详细"),
    ("詳情", "详情"),
    ("說", "说"),
    ("當前", "当前"),
    ("現有", "现有"),
    ("規則", "规则"),
    ("安全規則", "安全规则"),
    ("安全策略", "安全策略"),
    ("安全配置", "安全配置"),
    ("啟用", "启用"),
    ("什麼", "什么"),
    ("哪幾條", "哪几条"),
    ("閾值", "阈值"),
    ("條件", "条件"),
    ("動作", "动作"),
    ("防護門", "防护门"),
    ("光柵", "光栅"),
    ("機械臂", "机械臂"),
    ("機器臂", "机械臂"),
    ("你好机械皮出手", "你好机械臂助手"),
    ("机械皮出手", "机械臂助手"),
    ("机器臂", "机械臂"),
    ("机械皮", "机械臂"),
    ("机械比", "机械臂"),
    ("机械壁", "机械臂"),
    ("机械币", "机械臂"),
    ("机械必", "机械臂"),
    ("气崩", "气泵"),
    ("气碰", "气泵"),
    ("气棒", "气泵"),
    ("汽泵", "气泵"),
    ("光山", "光栅"),
    ("光删", "光栅"),
    ("光珊", "光栅"),
    ("线速", "限速"),
    ("限诉", "限速"),
    ("县速", "限速"),
    ("机停", "急停"),
    ("急亭", "急停"),
    ("急婷", "急停"),
    ("付位", "复位"),
    ("腹位", "复位"),
    ("防户门", "防护门"),
    ("安全去", "安全区"),
)

RULE_ACTION_WORDS = (
    "修改",
    "更改",
    "变更",
    "更新",
    "调整",
    "设置",
    "设为",
    "调到",
    "改到",
    "改成",
    "改为",
    "恢复",
    "保持",
    "新增",
    "添加",
    "增加",
    "删除",
    "移除",
    "禁用",
    "停用",
    "启用",
    "关闭",
    "打开",
)
RULE_UNSUPPORTED_AUTHORING_WORDS = (
    "新增",
    "添加",
    "增加规则",
    "增加安全规则",
    "删除",
    "移除",
)
RULE_AUTHORING_SUBJECT_WORDS = (
    "规则",
    "安全规则",
    "安全策略",
    "安全配置",
)
RULE_SUBJECT_WORDS = (
    "规则",
    "安全策略",
    "安全配置",
    "限速",
    "速度",
    "阈值",
    "距离",
    "急停",
    "停机",
    "动作",
    "气泵",
)
RULE_EDIT_SPECIFIC_SUBJECT_WORDS = (
    "人员",
    "人机",
    "未知物体",
    "障碍物",
    "防护门",
    "光栅",
    "控制器",
    "报警",
    "示教",
    "限速",
    "速度",
    "阈值",
    "距离",
    "急停",
    "停机",
    "动作",
    "气泵",
)
RULE_EDIT_ASSIGNMENT_WORDS = (
    "修改为",
    "更改为",
    "调整为",
    "设置为",
    "设置成",
    "调整到",
    "设为",
    "改成",
    "改为",
    "调到",
    "改到",
)
RULE_EDIT_STATE_WORDS = (
    "禁用",
    "停用",
    "启用",
    "恢复",
    "关闭",
    "打开",
    "开启",
)
RULE_EDIT_TOPIC_GROUPS = (
    ("人员", "人机", "保护距离", "安全距离"),
    ("未知物体", "未知障碍", "障碍物", "物体靠近"),
    ("防护门",),
    ("安全光栅", "光栅"),
    ("控制器", "通信异常", "控制器报警"),
    ("示教",),
    ("限速", "速度", "减速"),
    ("急停", "停机", "停止"),
    ("气泵",),
)
RULE_EDITOR_ROUTE = "ROUTE:rule_editor_9b"
AGENT_TOOL_PREFIX = "TOOL:"
AGENT_TOOL_LOAD_RULES = "load_rules"
AGENT_TOOL_EDIT_RULES = "edit_rules"
AGENT_TOOL_ANALYZE_VISION = "analyze_environment_vision"
AGENT_TOOL_GET_OBJECT_MAPPING = "get_object_mapping"
AGENT_TOOL_UPDATE_OBJECT_MAPPING = "update_object_mapping"
AGENT_TOOL_ALIASES = {
    "edit_object_mapping": AGENT_TOOL_UPDATE_OBJECT_MAPPING,
    "modify_object_mapping": AGENT_TOOL_UPDATE_OBJECT_MAPPING,
    "set_object_mapping": AGENT_TOOL_UPDATE_OBJECT_MAPPING,
    "object_mapping_edit": AGENT_TOOL_UPDATE_OBJECT_MAPPING,
}
AGENT_TOOL_NAMES = (
    AGENT_TOOL_LOAD_RULES,
    AGENT_TOOL_EDIT_RULES,
    AGENT_TOOL_ANALYZE_VISION,
    AGENT_TOOL_GET_OBJECT_MAPPING,
    AGENT_TOOL_UPDATE_OBJECT_MAPPING,
)
_AGENT_TOOL_NAME_PATTERN = "|".join(re.escape(name) for name in AGENT_TOOL_NAMES)
AGENT_ROUTE_REGEX = (
    r'\{[\s\S]*"type"\s*:\s*"(?:final|tool_call)"[\s\S]*'
    r'(?:("content"\s*:\s*"[\s\S]*")|("name"\s*:\s*"(?:'
    + _AGENT_TOOL_NAME_PATTERN
    + r')"))[\s\S]*\}'
)
AGENT_ROUTE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
    "required": ["type"],
    "properties": {
        "type": {"enum": ["final", "tool_call"]},
        "content": {"type": "string"},
        "name": {"enum": list(AGENT_TOOL_NAMES)},
        "arguments": {"type": "object"},
    },
}
RULE_EDIT_TOOL_REGEX = r'\{[\s\S]*"name"\s*:\s*"edit_rules"[\s\S]*"arguments"\s*:\s*\{[\s\S]*"changes"\s*:\s*\{[\s\S]*\}[\s\S]*\}[\s\S]*\}'
RULE_EDIT_TOOL_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["type", "name", "arguments"],
    "properties": {
        "type": {"const": "tool_call"},
        "name": {"const": AGENT_TOOL_EDIT_RULES},
        "arguments": {
            "type": "object",
            "additionalProperties": True,
            "required": ["rule_id", "changes"],
            "properties": {
                "mode": {"const": "patch"},
                "rule_id": {"type": "string", "minLength": 1},
                "changes": {
                    "type": "object",
                    "minProperties": 1,
                    "additionalProperties": {"type": ["string", "number", "boolean"]},
                },
            },
        },
    },
}
CAPABILITY_RESPONSE = (
    "我可以处理机械臂急停、停机、复位、限速、气泵和安全区相关语音指令，"
    "也可以解释或修改安全规则、维护标号物体映射、解析抓取目标，并在明确请求时调用视觉分析当前工作环境。"
)
_FALLBACK_RESPONSE = "收到。"
VISION_ANALYSIS_SYSTEM_PROMPT = (
    "你是本地机械臂安全语音助手的视觉分析模块。只根据当前输入图像进行观察和安全相关描述。"
    "不要执行工具路由、不要确认机械臂操作。"
    "不要输出 JSON、Markdown、编号列表、隐藏推理或 <think> 标签。"
)
VISION_ANALYSIS_FALLBACK_RESPONSE = "已获取相机图像，但视觉模型未生成有效分析，请重试。"
_GENERIC_VISION_ACKNOWLEDGEMENTS = {
    "收到",
    "已收到",
    "好的",
    "好",
    "可以",
    "明白",
    "明白了",
    "ok",
    "okay",
}
_RULE_EDIT_REJECTION_GUIDANCE = "如果确实需要修改，可以先更改安全限定的数值，或者把对应已有规则的启用状态暂时设为禁用，并在风险解除后恢复启用。"
_RULE_EDIT_CLARIFICATION_RESPONSE = (
    "请明确要修改哪条已有安全规则，以及目标值或状态。"
    "例如：把人员安全距离调整为0.4米，或暂时禁用人员侵入规则。"
)
_RULE_PATCH_FIELD_LABELS = {
    "enabled": "启用状态",
    "conditions.person_distance_m.lt": "人员保护距离阈值",
    "conditions.unknown_object_distance_m.lt": "未知物体靠近距离阈值",
    "conditions.guard_door_open.eq": "防护门打开触发条件",
    "conditions.light_curtain_blocked.eq": "安全光栅遮挡触发条件",
    "conditions.ros_controller_alarm.eq": "控制器报警触发条件",
    "conditions.teach_mode.eq": "示教模式触发条件",
    "action.max_speed_scale": "最大速度比例",
    "action.requires_reset": "是否要求复位",
    "action.requires_manual_check": "是否要求人工检查",
    "action.notify_operator": "是否通知操作员",
    "action.severity": "严重级别",
    "action.type": "动作类型",
    "conditions": "触发条件配置",
    "action": "动作配置",
    "id": "规则标识",
    "name": "规则名称",
    "description": "规则描述",
}
_RULE_FOLLOWUP_MARKERS = (
    "建议后续提问",
    "后续建议",
    "建议后续",
    "后续具体提问建议",
    "建议提问",
    "建议追问",
    "建议继续问",
    "继续追问",
    "后续可以具体问",
    "后续可以问",
    "可以继续具体问",
    "可以继续问",
    "可以追问",
)

_VISION_REQUEST_PATTERNS = (
    "调用视觉",
    "启动视觉",
    "视觉分析",
    "用视觉",
    "看当前画面",
    "看一下当前画面",
    "看下当前画面",
    "分析当前工作环境",
    "分析下当前工作环境",
    "分析当前工作现场",
    "分析下当前工作现场",
    "拍一张",
    "拍张",
    "当前画面",
    "相机画面",
    "当前图像",
    "当前图片",
)

_WORKSPACE_SNAPSHOT_SUBJECT_WORDS = (
    "工作区",
    "工作空间",
    "工作环境",
    "工作现场",
    "工位",
    "现场环境",
)
_WORKSPACE_SNAPSHOT_QUERY_WORDS = (
    "什么情况",
    "情况",
    "状态",
    "概况",
    "怎么样",
    "当前",
    "现在",
    "整体",
    "汇总",
    "说一说",
    "说说",
    "讲讲",
)
_ARM_RUNTIME_QUERY_SUBJECT_WORDS = (
    "机械臂抓取请求",
    "抓取请求",
    "执行请求",
    "执行状态",
    "arm_rules",
    "arm runtime",
    "机械臂 json",
    "机械臂json",
    "机械臂 JSON",
    "机械臂JSON",
)
_ARM_RUNTIME_QUERY_WORDS = (
    "当前",
    "现在",
    "查询",
    "查看",
    "读取",
    "状态",
    "是什么",
    "什么",
    "有没有",
)

_OBJECT_MAPPING_UPDATE_WORDS = (
    "改成",
    "改为",
    "更新为",
    "设为",
    "设置为",
    "变成",
    "现在是",
    "以后叫",
    "对应的是",
    "对应为",
    "对应到",
    "对应",
    "标到",
    "标记到",
    "贴到",
    "贴在",
    "命名为",
    "叫做",
    "叫",
)
_OBJECT_MAPPING_UPDATE_SUBJECT_WORDS = (
    "映射",
    "物体",
    "对象",
    "东西",
    "工件",
)
_OBJECT_MAPPING_QUERY_WORDS = (
    "查询",
    "查看",
    "查一下",
    "查下",
    "看一下",
    "看下",
    "什么",
    "哪个",
    "哪一个",
    "哪些",
    "吗",
    "么",
    "?",
    "？",
)
_OBJECT_MAPPING_TABLE_SUBJECT_WORDS = (
    "物体映射表",
    "物体标记映射表",
    "标号物体映射表",
    "标号映射表",
    "标记映射表",
    "物体映射",
    "标号映射",
    "标记映射",
)
_OBJECT_MAPPING_TABLE_QUERY_WORDS = _OBJECT_MAPPING_QUERY_WORDS + (
    "当前",
    "现在",
    "全部",
    "所有",
    "完整",
    "列表",
    "列出",
    "说一说",
    "说说",
    "讲讲",
    "介绍",
    "说明",
    "给我",
)
_OBJECT_MAPPING_TABLE_MUTATION_WORDS = (
    "改",
    "修改",
    "更新",
    "新增",
    "添加",
    "删除",
    "移除",
    "清空",
    "重置",
    "禁用",
    "启用",
    "关闭",
    "打开",
    "设置",
)
_OBJECT_NAME_PLACEHOLDER_WORDS = (
    "物体",
    "对象",
    "东西",
    "新的物体",
    "新物体",
    "这个",
    "那个",
)
_LABELED_MARKER_RE = re.compile(r"标号\s*([A-Z])(?![A-Za-z])", re.IGNORECASE)
_BARE_MARKER_RE = re.compile(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", re.IGNORECASE)
_OBJECT_GRASP_ACTION_WORDS = (
    "帮我抓取",
    "帮忙抓取",
    "请抓取",
    "抓取",
    "拿取",
    "取来",
    "递给我",
    "拿给我",
    "给我",
    "我需要",
)
_OBJECT_GRASP_EXCLUDED_PHRASES = (
    "给我看看",
    "给我看一下",
    "给我看下",
    "给我说明",
    "给我解释",
    "我需要说明",
    "我需要解释",
    "我需要了解",
)
_OBJECT_GRASP_LEADING_FILLERS = (
    "一下",
    "下",
    "这个",
    "那个",
    "这件",
    "那件",
    "一个",
    "一件",
    "一块",
    "一根",
    "一把",
)
_OBJECT_GRASP_TRAILING_FILLERS = (
    "给我",
    "过来",
    "一下",
    "下",
    "吧",
)

_CAPABILITY_PATTERNS = (
    "你能做什么",
    "你可以做什么",
    "你会做什么",
    "能做什么",
    "可以做什么",
    "有什么功能",
    "功能",
)
_QUESTION_WORDS = (
    "吗",
    "么",
    "能不能",
    "可不可以",
    "是否",
    "多少",
    "?",
    "？",
)
_EXPLANATION_WORDS = (
    "说明",
    "解释",
    "规则",
    "是什么",
    "为什么",
    "如何",
    "怎么",
    "介绍",
    "含义",
    "原理",
)
_ESTOP_RELEASE_PHRASES = (
    "解除急停",
    "取消急停",
    "释放急停",
    "复位急停",
    "清除急停",
    "解除紧急停止",
    "取消紧急停止",
)
_ESTOP_TRIGGER_PHRASES = (
    "急停",
    "紧急停止",
    "立即停止",
    "立刻停止",
    "马上停止",
    "停止机械臂",
    "停下机械臂",
    "停止机器人",
    "停止运动",
    "停机",
)
_EXACT_LLM_BYPASS_COMMANDS = frozenset(("急停", "解除急停"))
_RESET_PHRASES = (
    "复位",
    "回零",
)
_SPEED_LIMIT_PHRASES = (
    "限速",
    "减速",
    "降低速度",
)
_ARM_DECELERATION_COMMAND_PHRASES = (
    "速度限制",
    "限制速度",
    "速度降到",
    "速度降至",
    "速度降低到",
    "速度降低至",
    "降低速度",
    "减速",
    "限速",
    "降速",
)
_ARM_DECELERATION_RULE_CONTEXT_WORDS = (
    "规则",
    "安全策略",
    "安全配置",
    "阈值",
)
_ARM_DECELERATION_QUERY_WORDS = (
    "当前",
    "现在",
    "查询",
    "查看",
    "读取",
    "多少",
    "什么",
    "状态",
)
_ARM_DECELERATION_DELTA_WORDS = (
    "百分点",
)
_ARM_DECELERATION_NUMBER_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)(?![\d.])")
_AIR_PUMP_PHRASES = (
    "气泵",
    "吸取",
    "释放",
    "开泵",
    "关泵",
)
_ASSISTANT_ADDRESS_PATTERNS = (
    "机械臂助手",
    "安全助手",
)
_INTERNAL_RESPONSE_MARKERS = (
    "修正语音识别错误",
    "语音识别错误",
    "误识别",
    "ASR",
    "asr",
    "纠正",
    "更正",
    "正在修正",
    "正在纠正",
    "内部处理",
)
_META_RESPONSE_MARKERS = (
    "准备播报",
    "准备回复",
    "播报以下",
    "以下安全操作内容",
    "启动安全语音助手",
    "当前安全规则已加载",
    "安全规则已加载",
    "当前规则已加载",
    "规则已加载",
    "规则文档已加载",
    "当前安全规则已读取",
    "安全规则已读取",
    "当前规则已读取",
    "已读取当前规则",
    "已获取当前规则",
    "安全规则已就绪",
    "已确认安全规则",
    "将立即处理您的请求",
    "交给 9B",
    "规则编辑器",
)
_RULE_READ_ACK_MARKERS = (
    "已加载",
    "已经加载",
    "已读取",
    "已经读取",
    "读取完成",
    "已获取",
    "已经获取",
    "已查到",
    "已找到",
)
_HEADING_LINES = (
    "安全指令",
    "响应",
    "response",
    "输入",
    "input",
)

_ASR_PUNCTUATION_ARTIFACTS = set(",，、。?？!！;；:：")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_JAPANESE_KANA_RE = re.compile(r"[\u3040-\u30ff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_MARKER_REPETITIVE_CJK_NOISE_RE = re.compile(r"^(?:[A-D]|标号[A-D])([\u3400-\u4dbf\u4e00-\u9fff])\1{2,}$", re.IGNORECASE)
_RAW_JSON_FIELD_KEYS = {
    "id",
    "name",
    "description",
    "enabled",
    "conditions",
    "condition",
    "action",
    "rules",
    "version",
    "type",
    "tool",
    "arguments",
    "argument",
    "response_text",
    "ros2_plan",
}


def normalize_asr_text(text: str) -> str:
    normalized = " ".join(text.strip().split())
    normalized = convert_traditional_to_simplified(normalized)
    for source, target in ASR_CORRECTIONS:
        normalized = normalized.replace(source, target)
    normalized = _remove_asr_punctuation_artifacts(normalized)
    normalized = " ".join(normalized.split())
    return normalized


def _remove_asr_punctuation_artifacts(text: str) -> str:
    cleaned: list[str] = []
    for index, char in enumerate(text):
        if char in _ASR_PUNCTUATION_ARTIFACTS:
            continue
        if char == "." and not _is_internal_dot(text, index):
            continue
        cleaned.append(char)
    return "".join(cleaned).strip()


def _is_internal_dot(text: str, index: int) -> bool:
    return 0 < index < len(text) - 1 and text[index - 1].isalnum() and text[index + 1].isalnum()


def should_ignore_asr_noise(text: str) -> bool:
    compact = text.strip()
    if not compact:
        return True
    if _is_marker_repetitive_cjk_noise(compact):
        return True
    if _JAPANESE_KANA_RE.search(compact) and not _CJK_RE.search(compact):
        return True
    if _CJK_RE.search(compact):
        return False
    return bool(_LATIN_RE.search(compact))


def _is_marker_repetitive_cjk_noise(text: str) -> bool:
    nospace = re.sub(r"\s+", "", text)
    return bool(_MARKER_REPETITIVE_CJK_NOISE_RE.fullmatch(nospace))


def convert_traditional_to_simplified(text: str) -> str:
    converter = _opencc_t2s_converter()
    if converter is None:
        return text
    try:
        return str(converter.convert(text))
    except Exception:
        return text


@lru_cache(maxsize=1)
def _opencc_t2s_converter() -> Any | None:
    try:
        from opencc import OpenCC
    except ImportError:
        return None
    try:
        return OpenCC("t2s")
    except Exception:
        return None


def build_spoken_response(user_text: str, model_text: str) -> str:
    deterministic = deterministic_spoken_response(user_text)
    if deterministic is not None:
        return deterministic
    return sanitize_spoken_response(model_text)


def build_rule_read_response(user_text: str, model_text: str, document: dict[str, Any]) -> str:
    if parse_agent_tool_request(model_text) is not None:
        return build_rule_read_fallback_response(document)
    if _is_raw_json_or_code_response(model_text):
        return build_rule_read_fallback_response(document)

    response_text = sanitize_spoken_response(model_text)
    response_text = _strip_raw_rule_ids(response_text, document)
    response_text = _strip_rule_read_followup_tail(response_text)
    if _is_unusable_rule_read_response(response_text):
        return build_rule_read_fallback_response(document)
    return response_text


def deterministic_spoken_response(text: str) -> str | None:
    normalized = normalize_asr_text(text)
    compact = normalized.lower()
    if any(pattern.lower() in compact for pattern in _CAPABILITY_PATTERNS):
        return CAPABILITY_RESPONSE
    if _looks_explanatory_or_question(normalized):
        return None
    if any(phrase in normalized for phrase in _ESTOP_RELEASE_PHRASES):
        return "已收到解除急停请求，请确认现场安全后再执行。"
    if any(phrase in normalized for phrase in _ESTOP_TRIGGER_PHRASES):
        return "机械臂急停！"
    if any(phrase in normalized for phrase in _RESET_PHRASES):
        return "机械臂复位。"
    if any(phrase in normalized for phrase in _SPEED_LIMIT_PHRASES):
        return "机械臂已限速。"
    if any(phrase in normalized for phrase in _AIR_PUMP_PHRASES):
        return "气泵指令已收到。"
    if any(pattern in normalized for pattern in _ASSISTANT_ADDRESS_PATTERNS):
        return "你好，我是机械臂安全助手。"
    return None


def sanitize_spoken_response(text: str) -> str:
    stripped = strip_route_marker(strip_thinking_text(text))
    if _is_unusable_spoken_text(stripped) or _is_raw_json_or_code_response(stripped):
        return _FALLBACK_RESPONSE
    cleaned_lines: list[str] = []
    for line in stripped.splitlines():
        cleaned = _clean_spoken_line(line)
        if cleaned:
            cleaned_lines.append(cleaned)
    response = _join_spoken_lines(cleaned_lines)
    response = re.sub(r"\s+", " ", response).strip()
    response = _strip_stray_semicolon_artifacts(response)
    response = response.strip("；:：,， ")
    if _is_unusable_spoken_text(response):
        return _FALLBACK_RESPONSE
    return response or _FALLBACK_RESPONSE


def _is_unusable_spoken_text(text: str) -> bool:
    compact = re.sub(r"\s+", "", text).strip().lower()
    return compact in {"", "[]", "{}", "null", "none"}


def _clean_spoken_line(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    if _is_raw_json_or_code_line(stripped):
        return ""
    marker_text = stripped.strip("*# `")
    marker_key = marker_text.strip(":：").lower()
    if marker_key in _HEADING_LINES:
        return ""
    if any(marker in stripped for marker in _INTERNAL_RESPONSE_MARKERS):
        return ""
    if marker_text.lower() == RULE_EDITOR_ROUTE.lower():
        return ""
    if stripped.lower().startswith(AGENT_TOOL_PREFIX.lower()):
        return ""
    if "→" in stripped or "->" in stripped:
        return ""
    if any(marker in stripped for marker in _META_RESPONSE_MARKERS) and _is_meta_only_rule_read_text(stripped):
        return ""

    stripped = re.sub(r"^#{1,6}\s*", "", stripped)
    stripped = re.sub(r"^\s*(?:[-*•]\s+|\d+[.、]\s*)", "", stripped)
    stripped = stripped.replace("**", "").replace("__", "").replace("`", "")
    stripped = stripped.strip()
    if stripped.strip(":：").lower() in _HEADING_LINES:
        return ""
    if re.fullmatch(r"[；;。！？?!，,、]+", stripped):
        return ""
    return stripped


def _is_raw_json_or_code_response(text: str) -> bool:
    stripped = _strip_json_fence(text)
    if not stripped:
        return False
    parsed = _parse_complete_json_value(stripped)
    if isinstance(parsed, (dict, list)):
        return True
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if not lines:
        return False
    raw_line_count = sum(1 for line in lines if _is_raw_json_or_code_line(line))
    return raw_line_count == len(lines)


def _is_raw_json_or_code_line(line: str) -> bool:
    stripped = line.strip().strip(",")
    if not stripped:
        return True
    if stripped in {"{", "}", "[", "]", "},", "],"}:
        return True
    if stripped.startswith("```"):
        return True
    if re.fullmatch(r"[{}\[\],]+", stripped):
        return True
    candidate = stripped.lstrip("{[").strip()
    field_match = re.match(r"^[\"']?([A-Za-z_][A-Za-z0-9_]*)[\"']?\s*[:=]", candidate)
    return bool(field_match and field_match.group(1).lower() in _RAW_JSON_FIELD_KEYS)


def _parse_complete_json_value(text: str) -> Any | None:
    stripped = _strip_json_fence(text)
    decoder = json.JSONDecoder()
    try:
        value, end = decoder.raw_decode(stripped)
    except json.JSONDecodeError:
        return None
    if stripped[end:].strip():
        return None
    return value


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```").strip()
        stripped = stripped.removesuffix("```").strip()
    return stripped


def _join_spoken_lines(lines: list[str]) -> str:
    response = ""
    for line in lines:
        if not response:
            response = line
            continue
        if response[-1] in "。！？；" or line[0] in "。！？；，、":
            response += line.lstrip("；; ")
        else:
            response += f"；{line}"
    return response


def _strip_stray_semicolon_artifacts(text: str) -> str:
    cleaned = re.sub(r"([。！？])；+(?=\s*[\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9])", r"\1", text)
    cleaned = re.sub(r"([:：])；+(?=\s*[\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9])", r"\1", cleaned)
    cleaned = re.sub(r"；{2,}", "；", cleaned)
    return cleaned


def _strip_rule_read_followup_tail(response_text: str) -> str:
    marker_index = _first_rule_followup_marker_index(response_text)
    if marker_index < 0:
        return response_text

    prefix = response_text[:marker_index].rstrip(" ；;，,、")
    if not prefix:
        return ""
    if prefix[-1] not in "。！？":
        prefix += "。"
    return prefix


def _first_rule_followup_marker_index(text: str) -> int:
    indexes = []
    for marker in _RULE_FOLLOWUP_MARKERS:
        index = text.find(marker)
        if index >= 0:
            indexes.append(index)
    if not indexes:
        return -1
    return min(indexes)


def _looks_explanatory_or_question(text: str) -> bool:
    return any(word in text for word in _QUESTION_WORDS) or any(word in text for word in _EXPLANATION_WORDS)


def _is_unusable_rule_read_response(response_text: str) -> bool:
    compact = re.sub(r"\s+", "", response_text)
    if not compact or compact == _FALLBACK_RESPONSE:
        return True
    if any(marker in compact for marker in _RULE_READ_ACK_MARKERS) and _is_meta_only_rule_read_text(compact):
        return True
    return False


def _strip_raw_rule_ids(response_text: str, document: dict[str, Any]) -> str:
    rules = document.get("rules")
    if not isinstance(rules, list):
        return response_text
    rule_ids = [
        re.escape(rule["id"])
        for rule in rules
        if isinstance(rule, dict) and isinstance(rule.get("id"), str) and rule["id"].strip()
    ]
    if not rule_ids:
        return response_text
    rule_id_pattern = "|".join(rule_ids)
    cleaned = re.sub(
        rf"[（(]\s*(?:id\s*[:：]\s*)?(?:{rule_id_pattern})\s*[）)]",
        "",
        response_text,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        rf"\s*id\s*[:：]\s*(?:{rule_id_pattern})\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or response_text


def _is_meta_only_rule_read_text(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if len(compact) <= 24:
        return True
    useful_markers = (
        "人员",
        "未知物体",
        "防护门",
        "光栅",
        "控制器",
        "示教",
        "气泵",
        "急停",
        "停机",
        "停止",
        "限速",
        "减速",
        "阈值",
        "触发",
        "条件",
        "动作",
        "复位",
        "报警",
        "米",
        "%",
        "百分",
    )
    matches = sum(1 for marker in useful_markers if marker in compact)
    return matches < 2


def should_edit_rules(text: str) -> bool:
    compact = normalize_asr_text(text).lower()
    read_only_words = (
        "有哪些",
        "有什么",
        "哪几条",
        "多少条",
        "列表",
        "清单",
        "明细",
        "详情",
        "详细",
        "列出",
        "显示",
        "读取",
        "查看",
        "是什么",
        "状态",
        "说明",
        "解释",
        "介绍",
        "讲讲",
        "如何",
        "怎么",
        "怎么办",
    )
    has_action = any(word.lower() in compact for word in RULE_ACTION_WORDS)
    has_subject = any(word.lower() in compact for word in RULE_SUBJECT_WORDS)
    has_specific_subject = any(word.lower() in compact for word in RULE_EDIT_SPECIFIC_SUBJECT_WORDS)
    if not has_action or not has_subject or not has_specific_subject:
        return False

    has_assignment = any(word.lower() in compact for word in RULE_EDIT_ASSIGNMENT_WORDS)
    if any(word.lower() in compact for word in read_only_words) and not has_assignment:
        return False
    if any(word.lower() in compact for word in _QUESTION_WORDS) and not has_assignment:
        return False
    return _rule_edit_has_explicit_target(compact)


def _rule_edit_has_explicit_target(compact_text: str) -> bool:
    target_fragment = _rule_edit_target_fragment(compact_text)
    if target_fragment != compact_text:
        cleaned = target_fragment.strip("。.!！?？,，;；:：、 ")
        return bool(cleaned) and cleaned not in {"什么", "多少", "哪个", "哪条", "如何", "怎么"}
    return any(word.lower() in compact_text for word in RULE_EDIT_STATE_WORDS)


def _rule_edit_target_fragment(text: str) -> str:
    compact = re.sub(r"\s+", "", normalize_asr_text(text)).lower()
    matches: list[tuple[int, int]] = []
    for word in RULE_EDIT_ASSIGNMENT_WORDS:
        start = compact.rfind(word.lower())
        if start >= 0:
            matches.append((start, len(word)))
    if not matches:
        return compact
    start, length = max(matches)
    return compact[start + length :]


def _rule_edit_subject_fragment(text: str) -> str:
    compact = re.sub(r"\s+", "", normalize_asr_text(text)).lower()
    starts = [compact.find(word.lower()) for word in RULE_EDIT_ASSIGNMENT_WORDS]
    starts = [start for start in starts if start >= 0]
    return compact[: min(starts)] if starts else compact


def should_reject_rule_authoring(text: str) -> bool:
    compact = normalize_asr_text(text).lower()
    has_authoring_action = any(word.lower() in compact for word in RULE_UNSUPPORTED_AUTHORING_WORDS)
    has_rule_subject = any(word.lower() in compact for word in RULE_AUTHORING_SUBJECT_WORDS)
    return has_authoring_action and has_rule_subject


def normalize_rule_edit_strategy(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in RULE_EDIT_STRATEGIES:
        allowed = ", ".join(strategy.replace("_", "-") for strategy in RULE_EDIT_STRATEGIES)
        raise ValueError(f"Unknown rule edit strategy {value!r}; expected one of: {allowed}")
    return normalized


RULE_READ_CONTEXT_WORDS = (
    "当前",
    "现在",
    "现有",
    "已有",
    "已经配置",
    "已配置",
    "启用",
    "正在启用",
    "全部",
    "所有",
)
RULE_READ_LIST_WORDS = (
    "有哪些",
    "有什么",
    "哪几条",
    "多少条",
    "列表",
    "清单",
    "明细",
    "详情",
    "详细",
    "列出",
    "显示",
    "读取",
    "查看",
)
RULE_READ_STATE_PHRASES = (
    "安全规则是什么",
    "规则是什么",
    "规则状态",
    "规则配置",
)
RULE_READ_EXPLANATION_WORDS = (
    "说明",
    "解释",
    "介绍",
    "讲讲",
    "说说",
    "说一说",
    "详情",
    "详细",
    "怎么",
    "如何",
    "触发",
    "条件",
    "动作",
    "阈值",
)
RULE_READ_SUBJECT_WORDS = (
    "规则",
    "安全策略",
    "安全配置",
)
RULE_READ_TOPIC_WORDS = (
    "急停",
    "停机",
    "限速",
    "速度",
    "阈值",
    "距离",
    "气泵",
    "人员",
    "未知物体",
    "控制器",
    "报警",
    "示教",
)


def should_read_rules(text: str) -> bool:
    if should_edit_rules(text):
        return False

    compact = normalize_asr_text(text).lower()
    has_rule_subject = any(word.lower() in compact for word in RULE_READ_SUBJECT_WORDS)
    has_topic = any(word.lower() in compact for word in RULE_READ_TOPIC_WORDS)
    has_explanation = any(word.lower() in compact for word in RULE_READ_EXPLANATION_WORDS)
    if has_rule_subject and has_explanation:
        return True
    if has_topic and has_explanation:
        return True
    if not has_rule_subject:
        return False
    if any(word.lower() in compact for word in RULE_READ_CONTEXT_WORDS):
        return True
    if any(word.lower() in compact for word in RULE_READ_LIST_WORDS):
        return True
    return any(phrase.lower() in compact for phrase in RULE_READ_STATE_PHRASES)


def should_analyze_vision(text: str) -> bool:
    compact = normalize_asr_text(text).lower()
    return any(pattern.lower() in compact for pattern in _VISION_REQUEST_PATTERNS)


def should_query_arm_runtime(text: str) -> bool:
    normalized = normalize_asr_text(text)
    compact = normalized.lower()
    if should_read_rules(normalized) or should_edit_rules(normalized):
        return False
    if should_analyze_vision(normalized):
        return False
    has_subject = any(subject.lower() in compact for subject in _ARM_RUNTIME_QUERY_SUBJECT_WORDS)
    has_query = any(word.lower() in compact for word in _ARM_RUNTIME_QUERY_WORDS)
    return has_subject and has_query


def should_request_arm_deceleration(text: str) -> bool:
    normalized = normalize_asr_text(text)
    compact = _compact_lower(normalized)
    if not _contains_arm_deceleration_phrase(compact):
        return False
    if should_read_rules(normalized) or should_edit_rules(normalized):
        return False
    if any(word.lower() in compact for word in _ARM_DECELERATION_RULE_CONTEXT_WORDS):
        return False
    if _looks_explanatory_or_question(normalized):
        return False
    if _number_text_after_arm_deceleration_phrase(compact) is not None:
        return True
    if any(word.lower() in compact for word in _ARM_DECELERATION_QUERY_WORDS):
        return False
    return True


def extract_arm_deceleration_target_percent(text: str) -> tuple[float | None, str | None]:
    normalized = normalize_asr_text(text)
    compact = _compact_lower(normalized)
    if any(word.lower() in compact for word in _ARM_DECELERATION_DELTA_WORDS):
        return (
            None,
            "这次没有执行速度调整：请说目标速度百分比，例如“减速到30%”，不要使用“降低10个百分点”这类差值说法。",
        )

    value_text = _number_text_after_arm_deceleration_phrase(compact)
    if value_text is None:
        return None, "请说清楚目标速度百分比，例如：减速到30%。本次未写入机械臂 JSON 执行请求。"

    try:
        target_percent = float(value_text)
    except ValueError:
        return None, "目标速度百分比无法识别。本次未写入机械臂 JSON 执行请求。"
    try:
        _validate_arm_deceleration_target_percent(target_percent)
    except ArmRulesValidationError as error:
        return None, f"{error}本次未写入机械臂 JSON 执行请求。"
    return target_percent, None


def format_arm_deceleration_percent(target_percent: float) -> str:
    parsed = _validate_arm_deceleration_target_percent(target_percent)
    return f"{parsed:g}%"


def format_arm_deceleration_fraction(target_percent: float) -> str:
    parsed = _validate_arm_deceleration_target_percent(target_percent)
    return f"{parsed / 100.0:g}"


def _compact_lower(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _contains_arm_deceleration_phrase(compact_text: str) -> bool:
    return any(phrase.lower() in compact_text for phrase in _ARM_DECELERATION_COMMAND_PHRASES)


def _number_text_after_arm_deceleration_phrase(compact_text: str) -> str | None:
    phrase_ends: list[int] = []
    for phrase in sorted(_ARM_DECELERATION_COMMAND_PHRASES, key=len, reverse=True):
        normalized_phrase = phrase.lower()
        start = compact_text.find(normalized_phrase)
        while start >= 0:
            phrase_ends.append(start + len(normalized_phrase))
            start = compact_text.find(normalized_phrase, start + 1)
    for phrase_end in sorted(phrase_ends):
        match = _ARM_DECELERATION_NUMBER_RE.search(compact_text, phrase_end)
        if match is not None:
            return match.group(1)
    return None


def _validate_arm_deceleration_target_percent(value: float) -> float:
    if isinstance(value, bool):
        raise ArmRulesValidationError("目标速度百分比必须是数字。")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ArmRulesValidationError("目标速度百分比必须是数字。") from exc
    if not 0.0 <= parsed <= 100.0:
        raise ArmRulesValidationError("目标速度百分比必须在 0% 到 100% 之间。")
    return parsed


def should_guide_workspace_snapshot(text: str) -> bool:
    normalized = normalize_asr_text(text)
    compact = normalized.lower()
    if not compact:
        return False
    if (
        should_read_rules(normalized)
        or should_edit_rules(normalized)
        or should_query_object_mapping_table(normalized)
        or should_query_object_mapping(normalized)
        or should_update_object_mapping(normalized)
        or should_query_arm_runtime(normalized)
        or should_request_arm_deceleration(normalized)
        or should_analyze_vision(normalized)
        or should_grasp_object(normalized)
    ):
        return False
    has_subject = any(subject.lower() in compact for subject in _WORKSPACE_SNAPSHOT_SUBJECT_WORDS)
    has_query = any(word.lower() in compact for word in _WORKSPACE_SNAPSHOT_QUERY_WORDS)
    return has_subject and has_query


def should_query_object_mapping(text: str) -> bool:
    normalized = normalize_asr_text(text)
    if _find_labeled_marker(normalized) is None:
        return False
    compact = normalized.lower()
    return any(word.lower() in compact for word in _OBJECT_MAPPING_QUERY_WORDS)


def should_query_object_mapping_table(text: str) -> bool:
    normalized = normalize_asr_text(text)
    if _find_labeled_marker(normalized) is not None:
        return False
    compact = normalized.lower()
    if not any(subject.lower() in compact for subject in _OBJECT_MAPPING_TABLE_SUBJECT_WORDS):
        return False
    if any(word.lower() in compact for word in _OBJECT_MAPPING_TABLE_MUTATION_WORDS):
        return False
    return any(word.lower() in compact for word in _OBJECT_MAPPING_TABLE_QUERY_WORDS)


def should_update_object_mapping(text: str) -> bool:
    normalized = normalize_asr_text(text)
    compact = normalized.lower()
    if any(word.lower() in compact for word in _OBJECT_MAPPING_QUERY_WORDS):
        return False
    marker = _find_labeled_marker(normalized)
    if marker is None:
        return False
    if not _has_object_mapping_update_subject(normalized, marker):
        return False
    return any(word.lower() in compact for word in _OBJECT_MAPPING_UPDATE_WORDS)


def should_grasp_object(text: str) -> bool:
    normalized = normalize_asr_text(text)
    compact = normalized.lower()
    if not compact:
        return False
    if should_read_rules(normalized) or should_edit_rules(normalized):
        return False
    if (
        should_query_object_mapping_table(normalized)
        or should_query_object_mapping(normalized)
        or should_update_object_mapping(normalized)
    ):
        return False
    if should_analyze_vision(normalized):
        return False
    if any(phrase.lower() in compact for phrase in _OBJECT_GRASP_EXCLUDED_PHRASES):
        return False
    if _looks_explanatory_or_question(normalized):
        return False
    return any(word.lower() in compact for word in _OBJECT_GRASP_ACTION_WORDS)


def extract_object_grasp_target(user_text: str) -> tuple[str | None, str | None, str | None, str | None]:
    marker = _find_labeled_marker(user_text)
    if marker is not None:
        marker = normalize_marker(marker)
        if marker not in {"A", "B", "C", "D"}:
            return None, None, None, "当前只支持标号A、标号B、标号C、标号D。"
        return marker, None, "marker", None

    if _find_bare_marker(user_text) is not None:
        return None, None, None, "请使用“标号A、标号B、标号C 或 标号D”说明抓取目标，不能只说 A、B、C 或 D。"

    object_name = _extract_object_grasp_object_name(user_text)
    if not _is_acceptable_object_name(object_name):
        return None, None, None, "请说明要抓取的目标，例如标号A到标号D，或已经配置的物体名称。"
    return None, object_name, "object_name", None


def extract_object_mapping_query(
    user_text: str,
    arguments: dict[str, Any] | None = None,
) -> tuple[str | None, str | None]:
    return _extract_required_labeled_marker(user_text, arguments)


def extract_object_mapping_update(
    user_text: str,
    arguments: dict[str, Any] | None = None,
) -> tuple[str | None, str | None, str | None]:
    marker, marker_error = _extract_required_labeled_marker(user_text, arguments)
    if marker_error is not None:
        return None, None, marker_error
    guidance = build_object_mapping_update_guidance(user_text)
    if guidance is not None:
        return marker, None, guidance

    object_name = _extract_object_name_from_text(user_text, marker)
    object_name = normalize_object_name(object_name or "")
    if not _is_acceptable_object_name(object_name):
        return marker, None, "请说明这个字母现在对应哪个物体。"

    return marker, object_name, None


def _extract_required_labeled_marker(
    user_text: str,
    arguments: dict[str, Any] | None = None,
) -> tuple[str | None, str | None]:
    marker = _find_labeled_marker(user_text)
    if marker is None:
        return None, "请使用“标号A、标号B、标号C 或 标号D”说明要操作的标号。"

    marker = normalize_marker(marker)
    if marker not in {"A", "B", "C", "D"}:
        return None, "当前只支持标号A、标号B、标号C、标号D。"

    argument_marker = _extract_marker_from_arguments(arguments)
    if argument_marker is not None and normalize_marker(argument_marker) != marker:
        return None, "用户原文中的标号与工具参数不一致，请重新说明标号A、标号B、标号C 或 标号D。"
    return marker, None


def _extract_marker_from_arguments(arguments: dict[str, Any] | None) -> str | None:
    if not isinstance(arguments, dict):
        return None
    for key in ("marker", "letter", "label"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _object_mapping_table_entries_from_snapshot(
    entries: tuple[WorkspaceObjectMappingEntry, ...],
) -> tuple[ObjectMappingTableEntry, ...]:
    return tuple(
        ObjectMappingTableEntry(
            marker=entry.marker,
            object_name=entry.object_name,
            enabled=entry.enabled,
        )
        for entry in entries
    )


def _find_labeled_marker(text: str) -> str | None:
    normalized = normalize_asr_text(text)
    match = _LABELED_MARKER_RE.search(normalized)
    return match.group(1).upper() if match else None


def _find_bare_marker(text: str) -> str | None:
    normalized = _LABELED_MARKER_RE.sub("", normalize_asr_text(text))
    match = _BARE_MARKER_RE.search(normalized)
    return match.group(1).upper() if match else None


def _has_object_mapping_update_subject(text: str, marker: str | None = None) -> bool:
    normalized = normalize_asr_text(text)
    target_text = normalized
    if marker is not None:
        for match in _LABELED_MARKER_RE.finditer(normalized):
            if match.group(1).upper() == marker:
                target_text = normalized[match.end() :]
                break
    compact = target_text.lower()
    return any(word.lower() in compact for word in _OBJECT_MAPPING_UPDATE_SUBJECT_WORDS)


def _extract_object_name_from_text(user_text: str, marker: str) -> str | None:
    normalized = normalize_asr_text(user_text)
    marker_match = None
    for match in _LABELED_MARKER_RE.finditer(normalized):
        if match.group(1).upper() == marker:
            marker_match = match
            break
    if marker_match is None:
        return None

    tail = normalized[marker_match.end() :]
    for phrase in sorted(_OBJECT_MAPPING_UPDATE_WORDS, key=len, reverse=True):
        index = tail.find(phrase)
        if index < 0:
            continue
        return _clean_extracted_object_name(tail[index + len(phrase) :])
    return None


def _clean_extracted_object_name(value: str) -> str:
    cleaned = value.strip(" \t\r\n:：,，;；.。!！?？")
    if "叫做" in cleaned:
        cleaned = cleaned.rsplit("叫做", 1)[1]
    elif "名叫" in cleaned:
        cleaned = cleaned.rsplit("名叫", 1)[1]
    elif "叫" in cleaned:
        cleaned = cleaned.rsplit("叫", 1)[1]
    cleaned = re.sub(r"^(?:新的?|这个|那个)?(?:物体|对象|东西|工件)(?:上)?", "", cleaned)
    cleaned = cleaned.strip(" \t\r\n:：,，;；.。!！?？为是到")
    cleaned = re.split(r"[，,。；;！!?？]", cleaned, maxsplit=1)[0]
    return normalize_object_name(cleaned)


def _extract_object_grasp_object_name(user_text: str) -> str:
    normalized = normalize_asr_text(user_text)
    for phrase in sorted(_OBJECT_GRASP_ACTION_WORDS, key=len, reverse=True):
        index = normalized.find(phrase)
        if index < 0:
            continue
        return _clean_extracted_grasp_object_name(normalized[index + len(phrase) :])
    return ""


def _clean_extracted_grasp_object_name(value: str) -> str:
    cleaned = value.strip(" \t\r\n:：,，;；.。!！?？")
    changed = True
    while changed:
        changed = False
        for filler in sorted(_OBJECT_GRASP_LEADING_FILLERS, key=len, reverse=True):
            if cleaned.startswith(filler):
                cleaned = cleaned[len(filler) :].strip()
                changed = True
                break
    cleaned = re.split(r"[，,。；;！!?？]", cleaned, maxsplit=1)[0].strip()
    changed = True
    while changed:
        changed = False
        for filler in sorted(_OBJECT_GRASP_TRAILING_FILLERS, key=len, reverse=True):
            if cleaned.endswith(filler):
                cleaned = cleaned[: -len(filler)].strip()
                changed = True
                break
    return normalize_object_name(cleaned)


def _is_acceptable_object_name(value: str) -> bool:
    compact = value.strip()
    if not compact:
        return False
    if any(word in compact for word in _OBJECT_MAPPING_QUERY_WORDS):
        return False
    if compact in _OBJECT_NAME_PLACEHOLDER_WORDS:
        return False
    return True


def parse_agent_final_response(text: str) -> str | None:
    value = _extract_first_json_value(text)
    if not isinstance(value, dict):
        return None
    if str(value.get("type", "")).strip().lower() != "final":
        return None
    content = value.get("content")
    return content.strip() if isinstance(content, str) and content.strip() else None


def parse_agent_tool_request(text: str) -> AgentToolRequest | None:
    stripped = strip_thinking_text(text).strip()
    if not stripped:
        return None

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if not lines:
        return None

    marker = lines[0].strip("*# `")
    marker_lower = marker.lower()
    if marker_lower.startswith(AGENT_TOOL_PREFIX.lower()) or marker_lower.startswith("tool："):
        separator = ":" if ":" in marker else "："
        raw = marker.split(separator, 1)[1].strip()
        if not raw:
            return None
        parts = raw.split(maxsplit=1)
        return _agent_tool_request(parts[0], parts[1] if len(parts) > 1 else "", None, allow_unknown=True)

    value = _extract_first_json_value(stripped)
    if isinstance(value, dict):
        return _agent_tool_request_from_mapping(value)
    return None


def parse_agent_route_decision(text: str) -> AgentRouteDecision | None:
    final_content = parse_agent_final_response(text)
    if final_content is not None:
        return AgentRouteDecision(final_content=final_content)

    tool_request = parse_agent_tool_request(text)
    if tool_request is not None:
        return AgentRouteDecision(tool_request=tool_request)

    value = _extract_first_json_value(text)
    if not isinstance(value, dict):
        return None

    route = str(value.get("route") or value.get("intent") or value.get("action") or "").strip().lower()
    arguments = _normalize_tool_arguments(value.get("arguments"))
    if route in {"load_rules", "read_rules", "rule_read"}:
        return AgentRouteDecision(
            tool_request=AgentToolRequest(AGENT_TOOL_LOAD_RULES, arguments=arguments),
        )
    if route in {"edit_rules", "rule_edit"}:
        return AgentRouteDecision(
            tool_request=AgentToolRequest(AGENT_TOOL_EDIT_RULES, arguments=arguments),
        )
    if route in {"analyze_environment_vision", "capture_vision_snapshot", "vision", "analyze_vision"}:
        return AgentRouteDecision(
            tool_request=AgentToolRequest(AGENT_TOOL_ANALYZE_VISION, arguments=arguments),
        )
    if route in {"get_object_mapping", "read_object_mapping", "object_mapping_query", "query_object_mapping"}:
        return AgentRouteDecision(
            tool_request=AgentToolRequest(AGENT_TOOL_GET_OBJECT_MAPPING, arguments=arguments),
        )
    if route in {
        "update_object_mapping",
        "edit_object_mapping",
        "modify_object_mapping",
        "set_object_mapping",
        "object_mapping_edit",
        "object_mapping_update",
        "object_marker_update",
    }:
        return AgentRouteDecision(
            tool_request=AgentToolRequest(AGENT_TOOL_UPDATE_OBJECT_MAPPING, arguments=arguments),
        )
    if route in {"reject_rule_authoring", "reject", "unsupported_rule_authoring"}:
        content = value.get("content")
        if not isinstance(content, str) or not content.strip():
            content = "首版只支持修改已有安全规则，不支持新增或删除规则。"
        return AgentRouteDecision(final_content=content.strip())
    if route in {"final", "respond", "answer"}:
        content = value.get("content") or value.get("response") or value.get("answer")
        if isinstance(content, str) and content.strip():
            return AgentRouteDecision(final_content=content.strip())
    return None


def _agent_tool_request_from_mapping(value: dict[str, Any]) -> AgentToolRequest | None:
    if str(value.get("type", "")).strip().lower() == "final":
        return None

    tool_calls = value.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        first_call = tool_calls[0]
        if isinstance(first_call, dict):
            nested_request = _agent_tool_request_from_mapping(first_call)
            if nested_request is not None:
                return nested_request

    function = value.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        arguments = _normalize_tool_arguments(function.get("arguments"))
        if isinstance(name, str):
            return _agent_tool_request(name, "", arguments, allow_unknown=True)

    name = value.get("tool") or value.get("name")
    arguments = _normalize_tool_arguments(value.get("arguments"))
    argument = value.get("argument") or value.get("input") or ""
    explicit_tool = str(value.get("type", "")).strip().lower() == "tool_call" or "tool" in value
    if isinstance(name, str):
        return _agent_tool_request(name, str(argument), arguments, allow_unknown=explicit_tool)
    return None


def _normalize_rule_patch_payload(user_text: str, document: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    if _looks_like_rule_patch(payload):
        return payload
    repaired = _repair_rule_patch_payload(user_text, document, payload)
    if repaired is not None:
        return repaired
    return payload


def _repair_rule_patch_payload(
    user_text: str,
    document: dict[str, Any],
    raw_value: Any,
    *,
    repair_existing_patch: bool = False,
) -> dict[str, Any] | None:
    payload = _extract_rule_edit_arguments(raw_value)
    if payload is None:
        return None
    if _looks_like_rule_patch(payload) and not repair_existing_patch:
        return payload if isinstance(payload, dict) else None

    legacy_rule = _extract_single_legacy_rule(payload)
    context_text = f"{normalize_asr_text(user_text)} {json.dumps(payload, ensure_ascii=False)}".lower()
    if legacy_rule is not None:
        context_text = f"{context_text} {json.dumps(legacy_rule, ensure_ascii=False)}".lower()

    target_path: str | None = None
    if any(term in context_text for term in ("人员", "person", "保护距离", "安全距离")):
        target_path = "conditions.person_distance_m.lt"
    elif any(term in context_text for term in ("未知物体", "unknown_object", "物体靠近")):
        target_path = "conditions.unknown_object_distance_m.lt"
    if target_path is None:
        return None

    value = _extract_distance_patch_value(user_text, payload)
    if value is None:
        return None

    rule_id = _find_rule_id_with_scalar_path(document, target_path)
    if rule_id is None:
        return None
    return {"rule_id": rule_id, "changes": {target_path: value}}


def _extract_rule_edit_arguments(value: Any) -> Any:
    if not isinstance(value, dict):
        return None
    name = value.get("tool") or value.get("name")
    if isinstance(name, str) and name.strip().lower() == AGENT_TOOL_EDIT_RULES:
        return value.get("arguments")
    return value


def _extract_single_legacy_rule(payload: dict[str, Any]) -> dict[str, Any] | None:
    rules = payload.get("rules")
    if isinstance(rules, list) and len(rules) == 1 and isinstance(rules[0], dict):
        return rules[0]
    if "conditions" in payload or "action" in payload:
        return payload
    return None


def _extract_distance_patch_value(user_text: str, payload: dict[str, Any]) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|米|公尺)", normalize_asr_text(user_text), flags=re.IGNORECASE)
    if match:
        return float(match.group(1))

    legacy_rule = _extract_single_legacy_rule(payload)
    if legacy_rule is None:
        return None
    conditions = legacy_rule.get("conditions")
    if isinstance(conditions, dict):
        for key in ("distance_m", "person_distance_m", "unknown_object_distance_m"):
            value = conditions.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
            if isinstance(value, dict):
                nested = value.get("lt")
                if isinstance(nested, (int, float)) and not isinstance(nested, bool):
                    return float(nested)
    return None


def _find_rule_id_with_scalar_path(document: dict[str, Any], path: str) -> str | None:
    parts = tuple(path.split("."))
    rules = document.get("rules")
    if not isinstance(rules, list):
        return None
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        current: Any = rule
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                break
            current = current[part]
        else:
            rule_id = rule.get("id")
            if isinstance(rule_id, str) and rule_id.strip() and not isinstance(current, (dict, list)):
                return rule_id
    return None


def _rule_patch_is_grounded_in_request(
    user_text: str,
    document: dict[str, Any],
    patch: dict[str, Any],
) -> bool:
    rule_id = patch.get("rule_id") or patch.get("id")
    changes = patch.get("changes")
    if not isinstance(rule_id, str) or not rule_id.strip() or not isinstance(changes, dict) or not changes:
        return False
    if not _rule_patch_target_is_grounded(user_text, document, rule_id):
        return False
    return all(
        isinstance(path, str) and _rule_patch_value_is_grounded(user_text, path, value)
        for path, value in changes.items()
    )


def _rule_patch_rule_exists(document: dict[str, Any], patch: dict[str, Any]) -> bool:
    rule_id = patch.get("rule_id") or patch.get("id")
    rules = document.get("rules")
    return isinstance(rule_id, str) and isinstance(rules, list) and any(
        isinstance(rule, dict) and rule.get("id") == rule_id for rule in rules
    )


def _rule_patch_target_is_grounded(user_text: str, document: dict[str, Any], rule_id: str) -> bool:
    rules = document.get("rules")
    if not isinstance(rules, list):
        return False

    subject_text = _rule_edit_subject_fragment(user_text)
    target_rule: dict[str, Any] | None = None
    for rule in rules:
        if not isinstance(rule, dict) or rule.get("id") != rule_id:
            continue
        target_rule = rule
        break
    if target_rule is None:
        return False

    rule_name = target_rule.get("name") or target_rule.get("title")
    if isinstance(rule_name, str) and rule_name.strip() and rule_name.strip().lower() in subject_text:
        return True
    if rule_id.lower() in subject_text:
        return True

    mentioned_groups = [
        group
        for group in RULE_EDIT_TOPIC_GROUPS
        if any(marker.lower() in subject_text for marker in group)
    ]
    if not mentioned_groups:
        return False

    candidates = [
        rule
        for rule in rules
        if isinstance(rule, dict)
        and all(any(marker.lower() in _rule_grounding_text(rule) for marker in group) for group in mentioned_groups)
    ]
    return len(candidates) == 1 and candidates[0].get("id") == rule_id


def _rule_grounding_text(rule: dict[str, Any]) -> str:
    text = json.dumps(rule, ensure_ascii=False).lower()
    aliases = []
    alias_sources = (
        (("person", "人员", "person_distance"), "人员 人机 保护距离 安全距离"),
        (("unknown_object", "未知物体", "unknown object"), "未知物体 未知障碍 障碍物 物体靠近"),
        (("guard_door", "防护门"), "防护门"),
        (("light_curtain", "安全光栅", "光栅"), "安全光栅 光栅"),
        (("controller", "控制器"), "控制器 通信异常 控制器报警"),
        (("teach_mode", "示教"), "示教"),
        (("limit_speed", "max_speed_scale", "限速"), "限速 速度 减速"),
        (("stop_motion", "emergency_stop", "停机", "急停"), "急停 停机 停止"),
        (("air_pump", "气泵"), "气泵"),
    )
    for sources, expanded in alias_sources:
        if any(source in text for source in sources):
            aliases.append(expanded)
    return f"{text} {' '.join(aliases)}"


def _rule_patch_value_is_grounded(user_text: str, path: str, value: Any) -> bool:
    target_text = _rule_edit_target_fragment(user_text)
    if isinstance(value, bool):
        if path == "enabled":
            true_markers = ("启用", "恢复", "打开", "开启", "保持")
            false_markers = ("禁用", "停用", "关闭", "不启用")
        else:
            true_markers = ("true", "是", "需要", "要求", "启用", "打开", "开启", "通知", "遮挡", "报警")
            false_markers = (
                "false",
                "否",
                "不需要",
                "不要求",
                "无需",
                "禁用",
                "关闭",
                "不通知",
                "未遮挡",
            )
        has_false_marker = any(marker in target_text for marker in false_markers)
        if not value:
            return has_false_marker
        return not has_false_marker and any(marker in target_text for marker in true_markers)

    if isinstance(value, (int, float)):
        number_matches = re.findall(r"[-+]?\d+(?:\.\d+)?", target_text)
        for raw_value in number_matches:
            requested_value = float(raw_value)
            if math.isclose(float(value), requested_value, rel_tol=0.0, abs_tol=1e-9):
                return True
            if ("%" in target_text or "百分" in target_text) and math.isclose(
                float(value), requested_value / 100.0, rel_tol=0.0, abs_tol=1e-9
            ):
                return True
        return False

    # String-valued fields still pass through the storage-layer allowlist and
    # validation. Numeric and boolean hallucinations are the unsafe cases this
    # grounding boundary must reject before confirmation or write.
    return True


def _normalize_tool_arguments(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip().startswith("{"):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(decoded, dict):
            return decoded
    return None


def _agent_tool_request(
    name: str,
    argument: str = "",
    arguments: dict[str, Any] | None = None,
    *,
    allow_unknown: bool = False,
) -> AgentToolRequest | None:
    normalized = name.strip().strip("。.,，;；:：").lower()
    normalized = AGENT_TOOL_ALIASES.get(normalized, normalized)
    if normalized in AGENT_TOOL_NAMES or (allow_unknown and normalized):
        return AgentToolRequest(normalized, argument.strip(), arguments)
    return None


def routes_to_rule_editor(text: str) -> bool:
    return text.lstrip().lower().startswith(RULE_EDITOR_ROUTE.lower())


def strip_route_marker(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].strip().lower() == RULE_EDITOR_ROUTE.lower():
        return "\n".join(lines[1:]).strip()
    return text.strip()


def build_agent_router_prompt(user_text: str) -> str:
    return (
        "你是机械臂安全语音助手的结构化工具路由器。"
        "必须只输出一个 JSON object，不要 Markdown，不要解释，不要输出思考过程或 <think> 标签。"
        "只能使用以下两种输出类型：\n"
        "1. 普通最终回复：{\"type\":\"final\",\"content\":\"<适合直接语音播报的简短中文>\"}\n"
        "2. 工具调用：{\"type\":\"tool_call\",\"name\":\"load_rules\",\"arguments\":{}} "
        "或 {\"type\":\"tool_call\",\"name\":\"edit_rules\",\"arguments\":{}} "
        "或 {\"type\":\"tool_call\",\"name\":\"analyze_environment_vision\",\"arguments\":{}} "
        "或 {\"type\":\"tool_call\",\"name\":\"get_object_mapping\",\"arguments\":{}} "
        "或 {\"type\":\"tool_call\",\"name\":\"update_object_mapping\",\"arguments\":{}}\n"
        "当用户询问或查询当前规则、已有规则、启用规则、规则列表、规则详情、规则状态、规则配置、"
        "某条规则说明、触发条件、动作或阈值时，无论用户是问「是什么」、「有哪些」、「怎么样」、「介绍一下」、「讲讲」、「说说」，"
        "一律输出 load_rules。常见问法包括但不限于："
        "「当前/现在/已有安全规则是什么」「规则有哪些」「全部规则」「规则列表」「介绍一下规则」"
        "「人员侵入触发条件是什么」「限速阈值多少」「控制器报警动作是什么」等，"
        "都输出 load_rules。"
        "当用户要求修改、调整、设为、调到、改到、启用、禁用、关闭或打开已有安全规则、阈值、距离、速度或动作时，"
        "输出 edit_rules；这里只做路由，arguments 保持空对象，不要生成 patch。"
        "当用户明确要求调用视觉、视觉分析、看当前画面、拍一张当前工位图或分析当前工作环境时，"
        "输出 analyze_environment_vision；这里只做路由，arguments 保持空对象。"
        "当用户只是笼统询问当前工作区、工作环境、工位或现场是什么情况时，不要调用视觉，"
        "输出 final，引导用户改问当前安全规则、当前物体映射表、当前机械臂抓取请求，"
        "或明确说调用视觉分析当前画面。"
        "当用户询问当前、全部或完整的物体映射表、标号映射表，而没有指定单个标号时，"
        "例如「说一说当前的物体映射表」「查看全部物体映射」，输出 get_object_mapping。"
        "当用户用「标号A、标号B、标号C、标号D」查询物体映射，"
        "例如「查询标号A」「标号B 对应什么」时，输出 get_object_mapping。"
        "当用户用「标号A、标号B、标号C、标号D」并明确说要修改映射、物体、对象或工件名称时，"
        "例如「把标号A的映射改成红色方块」「标号A的物体现在是扳手」「标号B贴到新的物体上叫夹具」时，"
        "输出 update_object_mapping；这里只做路由，可以保持 arguments 为空对象。"
        "如果用户只说「把标号A改成红色方块」而没有说明映射或物体，不要调用工具，输出 final 提示用户改说「把标号A的映射改成红色方块」。"
        "当用户说「抓取」「给我」「我需要」并带有标号或物体名称时，这是抓取目标解析请求，"
        "不要调用物体映射查询或更新工具，输出 final；程序会用确定性映射解析抓取目标。"
        "如果用户只说 A、B、C、D 而没有说「标号」，不要调用物体映射工具，输出 final 让用户改用标号A到标号D。"
        "当用户要求新增、创建、添加、删除或移除规则时，输出 final，内容说明首版只支持修改已有安全规则。"
        "当用户只是确认、感谢、评价或寒暄时，输出 final，用自然简短中文回应。"
        "急停、复位、限速、气泵等直接操作指令如果不是规则读写请求，输出 final。"
        "禁止输出数组、空对象、空列表、TOOL 前缀、ROUTE 标记或自然语言 JSON 之外的文字。\n\n"
        f"用户输入：{user_text}\n\n"
        "输出："
    )


def build_vision_analysis_prompt(user_text: str, artifact: VisionImageArtifact) -> str:
    metadata = json.dumps(artifact.metadata, ensure_ascii=False, sort_keys=True)
    return (
        "<image>\n"
        "请分析这张当前 RGB 快照，并只输出适合语音播报的 4 到 6 句简洁中文。"
        "第一句简述画面中的整体场景。"
        "第二句描述主要可见对象及其大致位置关系。"
        "重点观察：人员是否进入或靠近机械臂工作区、是否有障碍物或未知物体靠近路径、"
        "工位是否有明显遮挡/杂乱/危险状态、是否需要操作员进一步确认。"
        "不要编造图片中看不见的细节；不确定时明确说不确定。"
        "不要 Markdown，不要编号，不要输出 JSON。"
        f"用户请求：{user_text}\n"
        f"图像元数据：{metadata}\n"
    )


def is_unusable_vision_analysis_response(text: str) -> bool:
    compact = re.sub(r"[\s。.!！?？,，;；:：、~…\"'`]+", "", strip_thinking_text(text)).lower()
    if not compact:
        return True
    if compact in _GENERIC_VISION_ACKNOWLEDGEMENTS:
        return True
    if len(compact) < 8:
        return True
    return any(compact.startswith(prefix) and len(compact) <= 12 for prefix in _GENERIC_VISION_ACKNOWLEDGEMENTS)


def build_rule_edit_patch_prompt(
    user_text: str,
    document: dict[str, Any],
    *,
    strategy: str,
) -> str:
    current_rules = json.dumps(document, ensure_ascii=False, indent=2)
    strategy_label = "one-pass" if strategy == RULE_EDIT_STRATEGY_ONE_PASS else "two-pass"
    min_personnel_distance = _format_compact_number(PERSONNEL_DISTANCE_MIN_M)
    max_personnel_distance = _format_compact_number(PERSONNEL_DISTANCE_MAX_M)
    return (
        "用户要求修改机械臂安全规则。你是本项目的 2B 规则 patch 生成器。"
        "必须只输出一个 JSON object，不要 Markdown，不要解释，不要输出思考过程或 <think> 标签。"
        "输出必须是 edit_rules 工具调用 envelope，格式如下："
        '{"type":"tool_call","name":"edit_rules","arguments":{"rule_id":"<existing_rule_id>",'
        '"changes":{"<existing.scalar.path>": <new_scalar_value>}}}。'
        "只能修改当前规则文档中已经存在的规则和已经存在的标量字段。"
        "禁止新增规则、删除规则、新增字段、删除字段、修改 id/name/description/action.type，"
        "禁止替换 conditions 或 action 整个对象。"
        "changes 的 key 使用点号路径，例如 enabled、conditions.person_distance_m.lt、"
        "action.max_speed_scale、action.requires_reset。"
        "如果用户说人员安全距离或保护距离，优先在现有规则中查找 conditions.person_distance_m.lt；"
        f"人员保护距离阈值只能设置为 {min_personnel_distance} 到 {max_personnel_distance} 米之间的数字；"
        "如果用户要暂时禁用或恢复人员安全距离规则，只能修改同一条已有规则的 enabled 为 false 或 true。"
        "如果用户说未知物体靠近距离，优先查找 conditions.unknown_object_distance_m.lt。"
        "不要输出 safety_distance、distance_m 或 rules 数组这类新规则/旧格式。"
        "新值类型必须匹配当前字段类型；boolean 只能用 true/false，数字只能用 JSON number。\n\n"
        f"策略：{strategy_label}\n\n"
        f"用户请求：{user_text}\n\n"
        f"当前规则文档：\n{current_rules}\n\n"
        "输出要求：只输出一个 edit_rules JSON object。"
    )


def build_rule_read_prompt(user_text: str, document: dict[str, Any]) -> str:
    scope = _rule_read_question_scope(user_text)
    candidate_rules = _rule_read_candidate_rules(user_text, document, scope)
    prompt_document = _rule_read_prompt_document(document, candidate_rules)
    current_rules = json.dumps(prompt_document, ensure_ascii=False, indent=2)
    focus_instruction = _rule_read_focus_instruction(user_text, document, scope, candidate_rules)
    document_label = "当前规则文档"
    if candidate_rules:
        document_label = "当前规则文档（已按用户问题筛选为候选规则）"
    if scope == "specific":
        scope_instruction = (
            "问题类型提示：这是具体规则问题。只回答 JSON 中最相关的一条或一类规则，"
            "禁止输出后续建议、建议提问、整体建议或其它规则名。"
            "只允许使用规则的中文 name 或自然中文描述，严禁输出 id、字段名、snake_case 或“id:”字样。"
        )
    else:
        scope_instruction = (
            "问题类型提示：这是规则总览问题。需要完整覆盖所有启用规则，"
            "禁止输出后续建议、建议提问、整体建议或其它规则名。"
        )
    return (
        "这是工具结果后的最终回答阶段，禁止再调用任何 TOOL。"
        "你已经通过 load_rules 工具读取了当前机械臂安全规则文档。"
        "请自己理解 JSON 中每条规则的 id、name、description、conditions、action 和 enabled，"
        "只根据这份 JSON 回答用户，不要补充 JSON 中没有的规则、阈值或动作。"
        "回答必须是自然、简短、完整的中文，适合直接语音播报；不要输出 Markdown、工具名、JSON，"
        "也不要照抄英文 snake_case 标识、字段名或枚举值。\n\n"
        f"用户问题：{user_text}\n\n"
        f"{scope_instruction}\n\n"
        f"{focus_instruction}\n\n"
        f"{document_label}：\n{current_rules}\n\n"
        "回答要求：如果问题笼统询问当前规则、规则列表或全部规则，请覆盖每条启用规则的核心风险、"
        "触发条件和动作，按不同安全方面归纳，每条规则只用一个短分句，整体控制在一百八十个汉字以内；"
        "不要输出后续建议、建议提问或“整体安全建议”这类泛泛建议。"
        "如果问题指向某一条或某一类规则，请在 JSON 中找出最相关规则，只解释该规则的启用状态、"
        "触发条件、动作和关键阈值，控制在一百二十个汉字以内；不要枚举或暗示无关规则，"
        "不要补充通用安全建议、操作步骤或 JSON 之外的流程，也不要输出后续建议或建议提问。"
    )


def _rule_read_question_scope(user_text: str) -> str:
    compact = normalize_asr_text(user_text).lower()
    has_topic = any(word.lower() in compact for word in RULE_READ_TOPIC_WORDS)
    has_focus_trigger = any(trigger.lower() in compact for trigger, _ in RULE_READ_FOCUS_TERMS)
    broad_words = (
        "当前安全规则",
        "已有规则",
        "启用规则详情",
        "全部",
        "所有",
        "有哪些",
        "有什么",
        "哪几条",
        "多少条",
        "列表",
        "清单",
        "总结",
        "概括",
        "汇总",
    )
    if (has_topic or has_focus_trigger) and not any(word.lower() in compact for word in broad_words):
        return "specific"
    return "broad"


RULE_READ_FOCUS_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("限速", ("限速", "速度", "limit_speed", "max_speed")),
    ("速度", ("限速", "速度", "limit_speed", "max_speed")),
    ("防护门", ("防护门", "安全门", "guard_door")),
    ("光栅", ("光栅", "light_curtain")),
    ("人员", ("人员", "person")),
    ("未知物体", ("未知物体", "unknown_object")),
    ("控制器", ("控制器", "controller", "ros")),
    ("报警", ("报警", "告警", "alarm")),
    ("示教", ("示教", "teach")),
    ("急停", ("急停", "紧急停止", "stop_motion")),
    ("停机", ("停机", "停止", "stop_motion")),
)


def _rule_read_candidate_rules(user_text: str, document: dict[str, Any], scope: str) -> list[dict[str, Any]]:
    if scope != "specific":
        return []

    compact = normalize_asr_text(user_text).lower()
    active_terms: list[str] = []
    for trigger, terms in RULE_READ_FOCUS_TERMS:
        if trigger.lower() in compact:
            active_terms.extend(term.lower() for term in terms)
    if not active_terms:
        return []

    candidates: list[dict[str, Any]] = []
    rules = document.get("rules")
    if isinstance(rules, list):
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            rule_text = json.dumps(rule, ensure_ascii=False).lower()
            if any(term in rule_text for term in active_terms):
                candidates.append(rule)
    return candidates


def _rule_read_prompt_document(document: dict[str, Any], candidate_rules: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidate_rules:
        return document
    prompt_document = dict(document)
    prompt_document["rules"] = candidate_rules
    prompt_document["filtered_from_rule_count"] = len(document.get("rules", [])) if isinstance(document.get("rules"), list) else None
    return prompt_document


def _rule_read_focus_instruction(
    user_text: str,
    document: dict[str, Any],
    scope: str,
    candidate_rules: list[dict[str, Any]],
) -> str:
    if scope != "specific":
        return "候选规则提示：总览问题不预筛选规则。"

    if not candidate_rules:
        if _rule_read_candidate_rules(user_text, document, scope):
            return "候选规则提示：没有预筛选到候选规则，请只按 JSON 语义选择最相关规则。"
        return "候选规则提示：用户没有给出可预筛选的具体安全主题，请只按问题语义选择最相关规则。"

    candidates = [_rule_display_name(rule, index) for index, rule in enumerate(candidate_rules, start=1)]

    count_instruction = "候选规则超过一条，必须逐条覆盖全部候选规则。" if len(candidates) > 1 else ""
    return (
        "候选规则提示：本问题只应回答这些候选规则："
        f"{'、'.join(candidates)}。"
        f"{count_instruction}"
        "不要提及候选之外的规则；不要输出任何英文 id 或字段名。"
    )


def build_rule_read_fallback_response(document: dict[str, Any]) -> str:
    rules = document.get("rules")
    if not isinstance(rules, list) or not rules:
        return "当前没有配置安全规则。"

    names = [_rule_display_name(rule, index) for index, rule in enumerate(rules, start=1) if isinstance(rule, dict)]
    if not names:
        return "当前没有可播报的安全规则。"
    preview = "、".join(names[:6])
    if len(names) > 6:
        preview += f"等 {len(names)} 条"
    return (
        "规则文件已读取，但模型没有生成可用的中文总结。"
        f"当前共有 {len(names)} 条规则，涉及：{preview}。"
        "规则文档包含这些规则的触发条件和动作配置。"
    )


def build_rule_edit_success_response(
    previous_document: dict[str, Any],
    updated_document: dict[str, Any],
    patch: dict[str, Any],
) -> str:
    rule_id = patch.get("rule_id") or patch.get("id")
    rule_label = _rule_edit_target_label(previous_document, rule_id)
    preview = preview_rule_patch_changes(previous_document, patch)
    change_summary = _rule_patch_value_transition_summary(preview.changes, changed_only=True)
    version = updated_document.get("version")
    version_text = str(version) if isinstance(version, int) else "未知"
    if not change_summary:
        return f"安全规则已更新：已修改{rule_label}，当前版本 {version_text}。"
    return f"安全规则已更新：{rule_label}的{change_summary}，当前版本 {version_text}。"


def build_rule_edit_noop_response(
    previous_document: dict[str, Any],
    patch: dict[str, Any],
    changes: Any,
) -> str:
    rule_id = patch.get("rule_id") or patch.get("id")
    rule_label = _rule_edit_target_label(previous_document, rule_id)
    current_summary = _rule_patch_value_transition_summary(changes)
    if current_summary:
        return f"安全规则无需修改：{rule_label}的{current_summary}。未进入确认阶段，规则文件未写入。"
    return f"安全规则无需修改：{rule_label}当前值已经和请求一致。未进入确认阶段，规则文件未写入。"


def build_confirmation_required_response(confirmation: PendingConfirmation) -> str:
    checklist = "；".join(item for item in confirmation.checklist if item)
    if checklist:
        return f"这个操作需要确认：{confirmation.summary}。{confirmation.prompt}确认要点：{checklist}。"
    return f"这个操作需要确认：{confirmation.summary}。{confirmation.prompt}"


def build_workspace_snapshot_guidance_response(snapshot: WorkspaceSnapshot) -> str:
    prefix = "这是宽泛工作区问题，我不会一次性展开所有状态。"
    pending_text = ""
    if snapshot.pending_confirmation is not None:
        pending_text = f"当前有待确认操作：{snapshot.pending_confirmation.summary}。"
    options = "当前安全规则是什么、当前物体映射表、当前机械臂抓取请求"
    return f"{prefix}{pending_text}可以继续问：{options}；需要画面时请明确说“调用视觉分析当前画面”。"


def build_arm_runtime_query_response(snapshot: WorkspaceSnapshot) -> str:
    arm_runtime = snapshot.arm_runtime
    if arm_runtime is None:
        error = snapshot.error_for("arm_runtime")
        if error is not None:
            return f"读取机械臂 JSON 执行请求失败：{error.message}"
        return "当前未配置机械臂 JSON 执行请求文件，无法读取抓取请求。"

    target = f"标号{arm_runtime.capture_goal}" if arm_runtime.capture_goal else "未指定目标"
    if arm_runtime.capture_object:
        target += f"（{arm_runtime.capture_object}）"
    capture_text = "有待执行抓取请求" if arm_runtime.capture_requested else "没有待执行抓取请求"
    extra_states = []
    if arm_runtime.stop_requested:
        extra_states.append("停止请求已置位")
    if arm_runtime.recover_requested:
        extra_states.append("恢复请求已置位")
    state_text = f"；{'，'.join(extra_states)}" if extra_states else ""
    return (
        f"当前机械臂 JSON 执行请求：{capture_text}，目标{target}；"
        f"减速参数 {arm_runtime.decelerate or '未知'}，安全距离 {arm_runtime.safety_distance or '未知'} 米"
        f"{state_text}。"
    )


def _arm_runtime_query_result(summary: WorkspaceArmRuntimeSummary) -> ArmRuntimeQueryResult:
    return ArmRuntimeQueryResult(
        arm_rules_path=summary.path,
        capture_requested=summary.capture_requested,
        capture_goal=summary.capture_goal,
        capture_object=summary.capture_object,
        stop_requested=summary.stop_requested,
        recover_requested=summary.recover_requested,
        decelerate=summary.decelerate,
        safety_distance=summary.safety_distance,
    )


def build_arm_deceleration_success_response(target_percent: float) -> str:
    target_text = format_arm_deceleration_percent(target_percent)
    return f"已确认机械臂目标速度为 {target_text}。已写入机械臂 JSON 执行请求。"


def build_arm_deceleration_rejection_response(reason: Exception) -> str:
    return f"机械臂速度调整请求未执行：{reason}。未写入机械臂 JSON 执行请求。"


def build_object_mapping_update_guidance(user_text: str) -> str | None:
    normalized = normalize_asr_text(user_text)
    compact = normalized.lower()
    marker = _find_labeled_marker(normalized)
    if marker is None:
        return None
    if any(word.lower() in compact for word in _OBJECT_MAPPING_QUERY_WORDS):
        return None
    if not any(word.lower() in compact for word in _OBJECT_MAPPING_UPDATE_WORDS):
        return None
    if _has_object_mapping_update_subject(normalized, marker):
        return None
    object_name = _extract_object_name_from_text(normalized, marker) or "工具箱"
    return (
        "这次没有执行修改。"
        f"请明确说要修改标号{marker}的映射或物体，例如："
        f"把标号{marker}的映射改为{object_name}，或把标号{marker}的物体改为{object_name}。"
    )


def build_unknown_tool_response(tool_name: str, user_text: str) -> str:
    guidance = build_object_mapping_update_guidance(user_text)
    if guidance is not None:
        return guidance
    if "object_mapping" in tool_name:
        example_marker = _find_labeled_marker(user_text) or "D"
        example_object = _extract_object_name_from_text(user_text, example_marker) or "工具箱"
        return (
            "这次没有执行修改。"
            f"要修改物体映射，请明确说：把标号{example_marker}的映射改为{example_object}。"
            "也可以说：查询标号A，查看当前映射。"
        )
    return (
        "这次没有执行操作。"
        "请改用明确指令，例如：查询标号A、把标号D的映射改为工具箱、解除急停，或抓取标号A。"
    )


def build_rule_edit_rejection_response(reason: Exception) -> str:
    restriction = _rule_validation_restriction(reason)
    return (
        "规则修改请求未通过安全验证，未写入规则文件。"
        f"限制规则：{restriction}"
        f"{_RULE_EDIT_REJECTION_GUIDANCE}"
    )


def build_object_mapping_update_success_response(
    marker: str,
    object_name: str,
    *,
    previous_object: str | None = None,
) -> str:
    if previous_object is not None and normalize_object_name(previous_object) != normalize_object_name(object_name):
        return (
            f"已更新映射：{marker} 现在对应{object_name}，原来对应{previous_object}。"
            f"请确认现场 {marker} 标贴已经贴在该物体上。"
        )
    return f"已更新映射：{marker} 现在对应{object_name}。请确认现场 {marker} 标贴已经贴在该物体上。"


def build_object_mapping_update_noop_response(marker: str, object_name: str) -> str:
    return f"物体映射无需修改：标号{marker} 当前已经对应{object_name}。未进入确认阶段，映射文件未写入。"


def build_object_mapping_query_success_response(marker: str, object_name: str, *, enabled: bool = True) -> str:
    if enabled:
        return f"当前映射：标号{marker} 对应{object_name}。"
    return f"当前映射：标号{marker} 对应{object_name}，但这个标号当前未启用。"


def build_object_mapping_table_query_success_response(mappings: tuple[ObjectMappingTableEntry, ...]) -> str:
    if not mappings:
        return "当前物体映射表为空。"
    parts = []
    for mapping in mappings:
        text = f"标号{mapping.marker} 对应{mapping.object_name}"
        if not mapping.enabled:
            text += "（当前未启用）"
        parts.append(text)
    return f"当前物体映射表：{'；'.join(parts)}。"


def build_object_grasp_success_response(marker: str, object_name: str, *, target_source: str) -> str:
    if target_source == "object_name":
        return f"已识别抓取目标：{object_name}，对应标号{marker}。等待后续执行模块处理。"
    return f"已识别抓取目标：标号{marker}，对应{object_name}。等待后续执行模块处理。"


def build_object_grasp_rejection_response(reason: Exception) -> str:
    if isinstance(reason, FileNotFoundError):
        return f"抓取目标未确认：找不到物体映射文件 {reason.filename or reason}。"
    if isinstance(reason, ObjectMappingValidationError):
        return f"抓取目标未确认：{reason}"
    return f"抓取目标未确认：{reason}"


def build_object_mapping_query_rejection_response(reason: Exception) -> str:
    if isinstance(reason, FileNotFoundError):
        return f"物体标记映射查询失败：找不到映射文件 {reason.filename or reason}。"
    if isinstance(reason, ObjectMappingValidationError):
        return f"物体标记映射查询失败：{reason}"
    return f"物体标记映射查询失败：{reason}"


def build_object_mapping_update_rejection_response(reason: Exception) -> str:
    if isinstance(reason, FileNotFoundError):
        return f"物体标记映射更新失败，未写入映射文件：找不到映射文件 {reason.filename or reason}。"
    if isinstance(reason, ObjectMappingValidationError):
        return f"物体标记映射更新失败，未写入映射文件：{reason}"
    return f"物体标记映射更新失败，未写入映射文件：{reason}"


def _rule_display_name(rule: dict[str, Any], index: int) -> str:
    for key in ("name", "title", "description"):
        value = rule.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    rule_id = rule.get("id")
    if isinstance(rule_id, str) and rule_id.strip():
        return rule_id.strip()
    return f"第 {index} 条规则"


def _rule_edit_target_label(document: dict[str, Any], rule_id: Any) -> str:
    rules = document.get("rules")
    if not isinstance(rules, list):
        return "目标规则"
    for index, rule in enumerate(rules, start=1):
        if not isinstance(rule, dict) or rule.get("id") != rule_id:
            continue
        for key in ("name", "title"):
            value = rule.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return f"第 {index} 条规则"
    return "目标规则"


def _rule_patch_change_summary(changes: Any) -> str:
    if not isinstance(changes, dict) or not changes:
        return ""
    summaries = []
    for path, value in changes.items():
        if not isinstance(path, str) or not path.strip():
            continue
        normalized_path = _normalize_rule_patch_path_label(path)
        field_label = _rule_patch_field_label(normalized_path)
        value_label = _rule_patch_value_label(normalized_path, value)
        summaries.append(f"{field_label}改为{value_label}")
    return "、".join(summaries)


def _rule_patch_value_transition_summary(changes: Any, *, changed_only: bool = False) -> str:
    if not changes:
        return ""
    summaries = []
    for change in changes:
        path = getattr(change, "path", "")
        previous_value = getattr(change, "previous_value", None)
        new_value = getattr(change, "new_value", None)
        if changed_only and previous_value == new_value:
            continue
        if not isinstance(path, str) or not path.strip():
            continue
        normalized_path = _normalize_rule_patch_path_label(path)
        field_label = _rule_patch_field_label(normalized_path)
        previous_label = _rule_patch_value_label(normalized_path, previous_value)
        new_label = _rule_patch_value_label(normalized_path, new_value)
        if previous_value == new_value:
            summaries.append(f"{field_label}当前已是{previous_label}")
        else:
            summaries.append(f"{field_label}改为{new_label}（原值{previous_label}）")
    return "、".join(summaries)


def _rule_patch_field_label(path: str) -> str:
    if path in _RULE_PATCH_FIELD_LABELS:
        return _RULE_PATCH_FIELD_LABELS[path]
    if path.startswith("conditions."):
        return "触发条件字段"
    if path.startswith("action."):
        return "动作字段"
    return "目标字段"


def _rule_patch_value_label(path: str, value: Any) -> str:
    if path == "enabled" and isinstance(value, bool):
        return "启用" if value else "禁用"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = _format_compact_number(value)
        if path.startswith("conditions.") and path.endswith(".lt"):
            return f"{number} 米"
        if path == "action.max_speed_scale":
            return f"{number}（{_format_percent(value)}）"
        return number
    if isinstance(value, str):
        return value.strip() or "空字符串"
    return str(value)


def _format_compact_number(value: int | float) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _format_percent(value: int | float) -> str:
    percent = float(value) * 100
    return f"{percent:g}%"


def _rule_validation_restriction(reason: Exception) -> str:
    text = str(reason).strip()
    path = _extract_rule_validation_path(text)
    field_label = _rule_patch_field_label(path) if path else "目标字段"

    if "Personnel safety distance threshold" in text:
        min_text = _format_compact_number(PERSONNEL_DISTANCE_MIN_M)
        max_text = _format_compact_number(PERSONNEL_DISTANCE_MAX_M)
        if "between" in text:
            return f"人员保护距离阈值必须在 {min_text} 到 {max_text} 米之间。"
        if "finite number" in text or "number in meters" in text:
            return f"人员保护距离阈值必须是以米为单位的有限数字，允许范围是 {min_text} 到 {max_text} 米。"
    if "path is not allowed" in text:
        return f"{field_label}属于受保护字段，不能通过语音直接修改。"
    if "cannot replace object or list field" in text or "dotted scalar paths, not objects" in text:
        return f"只能修改已有的单个标量字段，不能整体替换{field_label}。"
    if "path does not exist" in text:
        return f"只能修改当前规则中已经存在的字段，{field_label}不存在。"
    if "target rule does not exist" in text:
        return "只能修改当前规则文件里已经存在的规则，目标规则不存在。"
    if "value type does not match" in text:
        return f"{field_label}的新值类型必须和当前字段一致。"
    if "value must be scalar" in text:
        return f"{field_label}的新值必须是数字、布尔值或文本，不能是对象或列表。"
    if "must contain rule_id and changes" in text or "must contain non-empty string field 'rule_id'" in text:
        return "修改指令必须同时指定一个已有目标规则和要修改的字段。"
    if "non-empty object field 'changes'" in text:
        return "修改指令必须包含至少一个要修改的已有字段。"
    if "path crosses non-object field" in text:
        return "修改路径必须逐级对应当前规则中的对象字段，不能穿过非对象字段。"
    if "path is invalid" in text or "change paths must be non-empty strings" in text:
        return "修改字段路径必须是非空的点号路径，并且只能指向已有标量字段。"
    if "did not return an edit_rules patch envelope" in text or "Model did not return a JSON object" in text:
        return "模型没有生成受支持的规则修改指令；只能提交单条已有规则的已有标量字段修改。"
    return "只能修改已有规则的已有标量字段，不能新增、删除或整体替换规则结构。"


def _extract_rule_validation_path(text: str) -> str:
    if ":" not in text:
        return ""
    candidate = text.rsplit(":", 1)[1].strip().strip("'\"。.")
    if not candidate:
        return ""
    return _normalize_rule_patch_path_label(candidate)


def _normalize_rule_patch_path_label(path: str) -> str:
    return ".".join(part.strip() for part in path.strip().split(".") if part.strip())


def extract_rule_patch_payload(text: str) -> dict[str, Any]:
    value = _extract_first_json_object(text)
    if _looks_like_rule_patch(value):
        return value

    name = value.get("tool") or value.get("name")
    arguments = value.get("arguments")
    if isinstance(name, str) and name.strip().lower() == AGENT_TOOL_EDIT_RULES and isinstance(arguments, dict):
        if _looks_like_rule_patch(arguments):
            return arguments
        raise RuleValidationError("edit_rules arguments must contain rule_id and changes.")

    raise RuleValidationError("Rule patch generator did not return an edit_rules patch envelope.")


def _looks_like_rule_patch(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    rule_id = value.get("rule_id") or value.get("id")
    return isinstance(rule_id, str) and isinstance(value.get("changes"), dict)


def extract_rule_replacement(text: str) -> dict[str, Any]:
    return _extract_first_json_object(text)


def _extract_first_json_object(text: str) -> dict[str, Any]:
    value = _extract_first_json_value(text)
    if isinstance(value, dict):
        return value
    raise ValueError("Model did not return a JSON object.")


def _extract_first_json_value(text: str) -> Any | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```").strip()
        stripped = stripped.removesuffix("```").strip()

    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        return value
    return None
