"""Runtime configuration for the OpenVINO voice stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from local_safety_assistant.config import DEFAULT_CACHE_DIR, PROJECT_ROOT
from local_safety_assistant.stack.vision import (
    DEFAULT_VISION_SNAPSHOT_SERVICE,
    DEFAULT_VISION_SNAPSHOT_TIMEOUT_SECONDS,
)


DEFAULT_ASR_MODEL = "whisper-large-v3-turbo"
DEFAULT_LLM_MODEL = "qwen35-2b"
DEFAULT_LLM_LARGE_MODEL = "qwen35-9b"
DEFAULT_AUDIO_OUTPUT_DIR = PROJECT_ROOT / ".runtime" / "tts"
DEFAULT_MELO_ROOT = PROJECT_ROOT / "MeloTTS.cpp"
DEFAULT_MOSS_ENV = PROJECT_ROOT / ".runtime" / "moss_tts_env"
DEFAULT_MOSS_SOURCE_DIR = PROJECT_ROOT / ".runtime" / "src" / "MOSS-TTS-Nano"
DEFAULT_MOSS_MODEL_DIR = PROJECT_ROOT / "models" / "tts"
DEFAULT_MOSS_YANGMI_PROMPT_AUDIO = DEFAULT_MOSS_SOURCE_DIR / "assets" / "audio" / "zh_11.wav"
DEFAULT_PIPER_EVAL_PYTHON = PROJECT_ROOT / ".runtime" / "tts_eval_env" / "bin" / "python"
DEFAULT_PIPER_RUNNER = PROJECT_ROOT / "Code" / "piper_tts.py"
DEFAULT_PIPER_SILENCE_SCALE = 1.0
DEFAULT_RULES_PATH = PROJECT_ROOT / "Code" / "config" / "safety_rules.example.json"
DEFAULT_OBJECT_MAPPING_PATH = PROJECT_ROOT / "Code" / "config" / "object_mapping.example.json"
DEFAULT_ARM_RULES_PATH = PROJECT_ROOT / "Code" / "config" / "arm_rules.json"
DEFAULT_RULE_EDIT_STRATEGY = "two-pass"

DEFAULT_SYSTEM_PROMPT = (
    "你是本项目的本地机械臂安全语音助手，运行在 OpenVINO + Whisper ASR + "
    "Qwen3.5 + 本地 TTS 语音链路中。只用简短中文回答，适合直接语音播报。"
    "回答必须围绕机械臂安全操作：急停、停机、复位、限速、气泵、"
    "人员进入安全区、ROS 控制器报警、规则解释、物体标号映射、"
    "抓取目标解析和明确请求下的当前环境视觉分析。"
    "ASR 可能错听的项目词"
    "只在内部归一化，不要说明纠错过程；例如机器臂/机械比/机械壁=机械臂，"
    "气崩/气碰/气棒=气泵，线速/限诉=限速。"
    "用户只是确认、感谢、评价或寒暄时，像真人助手一样自然简短回应，"
    "不要调用工具，不要提规则编辑器或规则状态。"
    "如果输入明确说明已经通过 load_rules 工具读取规则，或包含当前规则文档，"
    "这是工具结果后的最终回答阶段；禁止再次输出 TOOL:load_rules，必须直接根据 JSON 回答。"
    "你要自行理解 JSON 里的 id、name、description、conditions、action、enabled，"
    "用自然中文总结，不要照抄英文 snake_case 标识、字段名或枚举值。"
    "笼统询问全部规则时，覆盖每条启用规则的风险、触发条件和动作，完整但不啰嗦，"
    "每条规则只用短分句，不要输出后续建议、建议提问或整体安全建议。"
    "询问某一条或某一类规则时，只解释 JSON 中最相关规则的状态、触发条件、动作和阈值，"
    "不要枚举无关规则，不要补充通用安全建议或 JSON 外流程。"
    "如果用户询问或查询当前规则、已有规则、启用规则、规则列表、规则详情、规则状态、规则配置，"
    "或问「规则是什么」、「有哪些规则」、「介绍一下规则」、「讲讲规则」、某条规则的说明/触发条件/动作/阈值，"
    "不要凭记忆回答；只输出 JSON：{\"type\":\"tool_call\",\"name\":\"load_rules\",\"arguments\":{}}。"
    "用户要求修改、启用或禁用已有安全规则时，不要直接写规则文件；"
    "只输出 JSON：{\"type\":\"tool_call\",\"name\":\"edit_rules\",\"arguments\":{}}。"
    "用户要求新增或删除安全规则时，首版不支持，直接简短拒绝，不要调用工具。"
    "用户明确要求调用视觉、视觉分析当前画面或分析当前工作环境时，"
    "只输出 JSON：{\"type\":\"tool_call\",\"name\":\"analyze_environment_vision\",\"arguments\":{}}。"
    "用户只是笼统询问当前工作区、工作环境、工位或现场是什么情况时，不要调用视觉，"
    "应简短引导用户改问当前安全规则、当前物体映射表、当前机械臂抓取请求，"
    "或明确说调用视觉分析当前画面。"
    "用户用标号A、标号B、标号C、标号D 查询当前物体映射时，"
    "只输出 JSON：{\"type\":\"tool_call\",\"name\":\"get_object_mapping\",\"arguments\":{}}。"
    "用户用标号A、标号B、标号C、标号D 改变标号与物体名称的对应关系时，"
    "例如把标号A改成红色方块、标号A现在是扳手、标号B贴到新的物体上叫夹具，"
    "只输出 JSON：{\"type\":\"tool_call\",\"name\":\"update_object_mapping\",\"arguments\":{}}。"
    "用户说抓取、给我、我需要并带有标号或已映射物体名称时，这是抓取目标解析请求，"
    "不要调用物体映射查询或更新工具；程序会用确定性映射解析抓取目标。"
    "用户只说 A、B、C、D 而没有说标号时，不要调用物体映射工具，提示用户改用标号A到标号D。"
    "不要输出思考过程，不要输出 <think> 标签。"
)


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 180
    max_prompt_len: int = 4096
    min_response_len: int = 4
    temperature: float = 0.1


@dataclass(frozen=True)
class MeloTtsConfig:
    binary: Path = DEFAULT_MELO_ROOT / "build" / "meloTTS_ov"
    model_dir: Path = DEFAULT_MELO_ROOT / "ov_models"
    output_dir: Path = DEFAULT_AUDIO_OUTPUT_DIR
    language: str = "ZH"
    speed: float = 0.8
    quantize: bool = True
    disable_bert: bool = False
    disable_nf: bool = False
    timeout_seconds: float = 120.0


@dataclass(frozen=True)
class MossTtsConfig:
    executable: Path = DEFAULT_MOSS_ENV / "bin" / "moss-tts-nano"
    source_dir: Path = DEFAULT_MOSS_SOURCE_DIR
    model_dir: Path = DEFAULT_MOSS_MODEL_DIR
    output_dir: Path = DEFAULT_AUDIO_OUTPUT_DIR
    voice: str = "Xiaoyu"
    prompt_audio: Path | None = None
    # Benchmarked on Core Ultra 5 225U (2P+8E+2LPE): 3 threads streams fastest
    # (~0.95x realtime); 4 drops to ~0.78x and 8 collapses to ~0.37x.
    cpu_threads: int = 3
    cpu_affinity: str | None = None
    execution_provider: str = "cpu"
    max_new_frames: int = 375
    voice_clone_max_text_tokens: int = 75
    sample_mode: str = "fixed"
    realtime_streaming_decode: int = 1
    text_temperature: float = 1.0
    text_top_p: float = 1.0
    text_top_k: int = 50
    audio_temperature: float = 0.8
    audio_top_p: float = 0.95
    audio_top_k: int = 25
    audio_repetition_penalty: float = 1.2
    timeout_seconds: float = 180.0


@dataclass(frozen=True)
class PiperTtsConfig:
    python: Path = DEFAULT_PIPER_EVAL_PYTHON
    runner: Path = DEFAULT_PIPER_RUNNER
    model_dir: Path | None = None
    espeak_data_dir: Path | None = None
    output_dir: Path = DEFAULT_AUDIO_OUTPUT_DIR
    speed: float = 1.0
    silence_scale: float = DEFAULT_PIPER_SILENCE_SCALE
    threads: int = 4
    timeout_seconds: float = 120.0


@dataclass(frozen=True)
class VisionConfig:
    snapshot_service: str = DEFAULT_VISION_SNAPSHOT_SERVICE
    snapshot_timeout_seconds: float = DEFAULT_VISION_SNAPSHOT_TIMEOUT_SECONDS


@dataclass(frozen=True)
class VoiceStackConfig:
    asr_model: str = DEFAULT_ASR_MODEL
    llm_model: str = DEFAULT_LLM_MODEL
    large_llm_model: str = DEFAULT_LLM_LARGE_MODEL
    rules_path: Path = DEFAULT_RULES_PATH
    object_mapping_path: Path = DEFAULT_OBJECT_MAPPING_PATH
    arm_rules_path: Path = DEFAULT_ARM_RULES_PATH
    cache_dir: Path = DEFAULT_CACHE_DIR / "voice_stack"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    rule_edit_strategy: str = DEFAULT_RULE_EDIT_STRATEGY
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    tts: MeloTtsConfig = field(default_factory=MeloTtsConfig)
    moss_tts: MossTtsConfig = field(default_factory=MossTtsConfig)
    piper_tts: PiperTtsConfig = field(default_factory=PiperTtsConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
