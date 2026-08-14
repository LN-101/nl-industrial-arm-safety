"""OpenVINO GenAI Qwen3.5 text generation engine."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from local_safety_assistant.stack.config import GenerationConfig


@dataclass(frozen=True)
class LlmResult:
    prompt: str
    text: str
    model: str
    device: str
    load_seconds: float
    inference_seconds: float
    parsed: Any | None = None
    finish_reason: str = ""


class QwenLlmEngine:
    """Lazy OpenVINO GenAI pipeline wrapper for Qwen3.5 2B/9B."""

    def __init__(
        self,
        *,
        model: str,
        device: str,
        cache_dir: Path,
        generation: GenerationConfig,
        system_prompt: str,
        fallback: tuple[str, ...] = (),
    ) -> None:
        self.model = model
        self.device = device
        self.fallback = fallback
        self.cache_dir = cache_dir
        self.generation = generation
        self.system_prompt = system_prompt
        self._pipe: Any | None = None
        self._loaded_device: str | None = None
        self._load_seconds = 0.0

    def generate(self, user_text: str) -> LlmResult:
        return self._generate(user_text)

    def generate_structured_json(
        self,
        user_text: str,
        *,
        json_schema: dict[str, Any] | str | None = None,
        regex: str | None = None,
        max_new_tokens: int | None = None,
    ) -> LlmResult:
        return self._generate(
            user_text,
            json_schema=json_schema,
            regex=regex,
            max_new_tokens=max_new_tokens,
        )

    def generate_with_image(
        self,
        user_text: str,
        image_path: Path,
        *,
        max_new_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> LlmResult:
        try:
            import numpy as np
            import openvino as ov
            from PIL import Image
        except ImportError as error:
            raise RuntimeError("Vision analysis requires numpy, Pillow, and OpenVINO.") from error

        resolved_image = image_path.expanduser().resolve()
        if not resolved_image.is_file():
            raise FileNotFoundError(f"Vision analysis image does not exist: {resolved_image}")
        with Image.open(resolved_image) as image:
            image_tensor = ov.Tensor(np.asarray(image.convert("RGB"), dtype=np.uint8))

        errors: list[str] = []
        for device in (self.device, *self.fallback):
            try:
                pipe, load_seconds = self._load(device)
                prompt = build_prompt(
                    system_prompt or self.system_prompt,
                    user_text,
                    tokenizer=_safe_tokenizer(pipe),
                )
                started = time.perf_counter()
                result = pipe.generate(
                    prompt,
                    image=image_tensor,
                    max_new_tokens=max_new_tokens or self.generation.max_new_tokens,
                    do_sample=False,
                )
                elapsed = time.perf_counter() - started
                return LlmResult(
                    prompt=prompt,
                    text=strip_thinking_text(_extract_generated_text(result)).strip(),
                    model=self.model,
                    device=device,
                    load_seconds=load_seconds,
                    inference_seconds=elapsed,
                    parsed=_extract_parsed_result(result),
                    finish_reason=_extract_finish_reason(result),
                )
            except Exception as error:
                errors.append(f"{device}: {type(error).__name__}: {error}")
                self._pipe = None
                self._loaded_device = None
        raise RuntimeError("All vision LLM devices failed: " + " | ".join(errors))

    def _generate(
        self,
        user_text: str,
        *,
        json_schema: dict[str, Any] | str | None = None,
        regex: str | None = None,
        max_new_tokens: int | None = None,
    ) -> LlmResult:
        errors: list[str] = []
        for device in (self.device, *self.fallback):
            try:
                pipe, load_seconds = self._load(device)
                import openvino_genai as ov_genai

                prompt = build_prompt(self.system_prompt, user_text, tokenizer=_safe_tokenizer(pipe))
                started = time.perf_counter()
                kwargs = _build_generation_kwargs(
                    ov_genai,
                    max_new_tokens=max_new_tokens or self.generation.max_new_tokens,
                    json_schema=json_schema,
                    regex=regex,
                )
                result = pipe.generate(prompt, **kwargs)
                elapsed = time.perf_counter() - started
                return LlmResult(
                    prompt=prompt,
                    text=strip_thinking_text(_extract_generated_text(result)).strip(),
                    model=self.model,
                    device=device,
                    load_seconds=load_seconds,
                    inference_seconds=elapsed,
                    parsed=_extract_parsed_result(result),
                    finish_reason=_extract_finish_reason(result),
                )
            except Exception as error:
                errors.append(f"{device}: {type(error).__name__}: {error}")
                self._pipe = None
                self._loaded_device = None
        raise RuntimeError("All LLM devices failed: " + " | ".join(errors))

    def warm_load(self) -> dict[str, Any]:
        errors: list[str] = []
        for device in (self.device, *self.fallback):
            try:
                _, load_seconds = self._load(device)
                return {
                    "status": "ready",
                    "model": self.model,
                    "device": device,
                    "load_seconds": load_seconds,
                }
            except Exception as error:
                errors.append(f"{device}: {type(error).__name__}: {error}")
                self._pipe = None
                self._loaded_device = None
        raise RuntimeError("All LLM devices failed during warmup: " + " | ".join(errors))

    def _load(self, device: str) -> tuple[Any, float]:
        if self._pipe is not None and self._loaded_device == device:
            return self._pipe, 0.0

        import openvino as ov
        import openvino_genai as ov_genai
        from local_safety_assistant.model_testbed import (
            detect_model,
            pipeline_properties,
            resolve_model_path,
        )

        model_path = resolve_model_path(self.model)
        info = detect_model(model_path)
        if not info.exists:
            raise FileNotFoundError(f"LLM model path does not exist: {info.path}")
        if info.kind not in {"llm", "vlm"}:
            raise RuntimeError(f"Model is {info.kind!r}, not a text generation model: {info.path}")

        args = SimpleNamespace(
            max_prompt_len=self.generation.max_prompt_len,
            min_response_len=self.generation.min_response_len,
            max_new_tokens=self.generation.max_new_tokens,
            npu_prefill_hint="STATIC",
            npu_generate_hint="FAST_COMPILE",
            npu_compiler_type=None,
        )
        props = pipeline_properties(ov.Core(), device, info.kind, self.cache_dir, args)

        started = time.perf_counter()
        if info.kind == "vlm":
            self._pipe = ov_genai.VLMPipeline(info.path, device, **props)
        else:
            self._pipe = ov_genai.LLMPipeline(info.path, device, **props)
        self._loaded_device = device
        self._load_seconds = time.perf_counter() - started
        return self._pipe, self._load_seconds


def build_prompt(system_prompt: str, user_text: str, tokenizer: Any | None = None) -> str:
    system_content = (
        f"{system_prompt.strip()}\n"
        "Runtime invariant: enable_thinking=false. Do not output hidden reasoning, "
        "<think> tags, or analysis text."
    )
    user_content = user_text.strip()
    if tokenizer is not None:
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        try:
            return str(
                tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    extra_context={"enable_thinking": False},
                )
            )
        except TypeError:
            return str(tokenizer.apply_chat_template(messages, True, "", None, {"enable_thinking": False}))

    return (
        "<|im_start|>system\n"
        f"{system_content}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_content}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )


def strip_thinking_text(text: str) -> str:
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    stripped = re.sub(r"<think>.*", "", stripped, flags=re.DOTALL | re.IGNORECASE)
    stripped = stripped.replace("</think>", "")
    stripped = stripped.replace("<|im_start|>assistant", "")
    stripped = stripped.replace("<|im_end|>", "")
    return stripped.strip()


def _safe_tokenizer(pipe: Any) -> Any | None:
    try:
        return pipe.get_tokenizer()
    except Exception:
        return None


def _build_generation_kwargs(
    ov_genai: Any,
    *,
    max_new_tokens: int,
    json_schema: dict[str, Any] | str | None,
    regex: str | None,
) -> dict[str, Any]:
    if json_schema is None and regex is None:
        return {"max_new_tokens": max_new_tokens}

    try:
        generation_config = ov_genai.GenerationConfig()
        generation_config.max_new_tokens = max_new_tokens
        structured = ov_genai.StructuredOutputConfig()
        if json_schema is not None and hasattr(structured, "json_schema"):
            structured.json_schema = (
                json.dumps(json_schema, ensure_ascii=False) if isinstance(json_schema, dict) else json_schema
            )
        elif regex is not None and hasattr(structured, "regex"):
            structured.regex = regex
        else:
            return {"max_new_tokens": max_new_tokens}
        generation_config.structured_output_config = structured
        return {"generation_config": generation_config}
    except Exception:
        return {"max_new_tokens": max_new_tokens}


def _extract_generated_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    texts = getattr(result, "texts", None)
    if texts:
        return "\n".join(str(item) for item in texts)
    text = getattr(result, "text", None)
    if text:
        return str(text)
    return str(result)


def _extract_parsed_result(result: Any) -> Any | None:
    return getattr(result, "parsed", None)


def _extract_finish_reason(result: Any) -> str:
    finish_reason = getattr(result, "finish_reason", None)
    if finish_reason is None:
        finish_reason = getattr(result, "finish_reasons", None)
    if finish_reason is None:
        return ""
    return str(finish_reason)


@lru_cache(maxsize=1)
def genai_structured_tool_capabilities() -> dict[str, Any]:
    try:
        import openvino_genai as ov_genai
    except Exception as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"}

    snapshot: dict[str, Any] = {
        "available": True,
        "generation_config": hasattr(ov_genai, "GenerationConfig"),
        "structured_output_config": hasattr(ov_genai, "StructuredOutputConfig"),
        "supports_parsers": False,
        "supports_json_schema": False,
        "supports_regex": False,
        "supports_tool_call_finish_reason": False,
        "error": "",
    }
    try:
        generation_config = ov_genai.GenerationConfig()
        structured = ov_genai.StructuredOutputConfig()
        snapshot["supports_parsers"] = hasattr(generation_config, "parsers")
        snapshot["supports_json_schema"] = hasattr(structured, "json_schema")
        snapshot["supports_regex"] = hasattr(structured, "regex")
        finish_reason = getattr(ov_genai, "GenerationFinishReason", None)
        snapshot["supports_tool_call_finish_reason"] = bool(
            finish_reason is not None and hasattr(finish_reason, "TOOL_CALL")
        )
    except Exception as error:
        snapshot["error"] = f"{type(error).__name__}: {error}"
    return snapshot
