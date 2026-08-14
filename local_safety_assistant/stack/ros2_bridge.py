"""ROS2 topic bridge for completed voice turns."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from local_safety_assistant.arm_rules import ArmEstopRequestResult, request_arm_estop
from local_safety_assistant.stack.pipeline import VoiceTurnResult, normalize_asr_text, should_edit_rules


ROS2_STRING = "std_msgs/String"
ROS2_BOOL = "std_msgs/Bool"
ROS2_POINT = "geometry_msgs/Point"

DEFAULT_TRANSCRIPT_TOPIC = "/voice/transcript"
DEFAULT_RESPONSE_TOPIC = "/voice/assistant_response"
DEFAULT_ESTOP_REQUEST_TOPIC = "/safety/estop/request"
DEFAULT_ESTOP_BOOL_TOPIC = "/emergency_stop"
DEFAULT_GOAL_TOPIC = "/goal"
DEFAULT_SOURCE = "voice_assistant"
DEFAULT_CAMERA_ESTOP_SOURCE = "min_distance_camera"

_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")

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
    "可以",
)
_QUESTION_WORDS = (
    "吗",
    "能不能",
    "可不可以",
    "是否",
)
_COMMAND_FORCE_WORDS = (
    "立即",
    "立刻",
    "马上",
    "现在",
    "执行",
    "触发",
    "启动",
    "拉起",
    "发布",
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
_GOAL_WORDS = (
    "目标",
    "坐标",
    "位置",
    "移动",
    "移到",
    "移动到",
    "到达",
    "去到",
)


@dataclass(frozen=True)
class Ros2BridgeConfig:
    node_name: str = "voice_ros2_bridge"
    transcript_topic: str = DEFAULT_TRANSCRIPT_TOPIC
    response_topic: str = DEFAULT_RESPONSE_TOPIC
    estop_request_topic: str = DEFAULT_ESTOP_REQUEST_TOPIC
    estop_bool_topic: str = DEFAULT_ESTOP_BOOL_TOPIC
    goal_topic: str = DEFAULT_GOAL_TOPIC
    source: str = DEFAULT_SOURCE
    use_estop_request: bool = True
    publish_transcript: bool = True
    publish_response: bool = True
    publish_commands: bool = True
    qos_depth: int = 10
    wait_for_subscribers_seconds: float = 0.2
    estop_reset_sources: tuple[str, ...] = (DEFAULT_CAMERA_ESTOP_SOURCE,)


@dataclass(frozen=True)
class EstopCommand:
    active: bool
    reason: str


@dataclass(frozen=True)
class GoalCommand:
    x: float
    y: float
    z: float
    reason: str


@dataclass(frozen=True)
class Ros2MessagePlan:
    topic: str
    message_type: str
    payload: Any
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "message_type": self.message_type,
            "payload": self.payload,
            "reason": self.reason,
        }


def build_voice_ros2_plan(
    result: VoiceTurnResult,
    config: Ros2BridgeConfig | None = None,
) -> list[Ros2MessagePlan]:
    bridge_config = config or Ros2BridgeConfig()
    plans: list[Ros2MessagePlan] = []

    if bridge_config.publish_transcript and result.input_text:
        plans.append(
            Ros2MessagePlan(
                topic=bridge_config.transcript_topic,
                message_type=ROS2_STRING,
                payload=result.input_text,
                reason="voice transcript",
            )
        )

    if bridge_config.publish_response and result.response_text:
        plans.append(
            Ros2MessagePlan(
                topic=bridge_config.response_topic,
                message_type=ROS2_STRING,
                payload=result.response_text,
                reason="assistant response",
            )
        )

    if (
        not bridge_config.publish_commands
        or result.rule_update is not None
        or result.object_mapping_query is not None
        or result.object_mapping_table_query is not None
        or result.object_mapping_update is not None
        or result.object_grasp_intent is not None
        or result.arm_runtime_query is not None
        or result.arm_deceleration_request is not None
    ):
        return plans

    estop = detect_estop_command(result.input_text)
    if estop is not None:
        plans.append(_build_estop_plan(estop, bridge_config))
        return plans

    goal = detect_goal_command(result.input_text)
    if goal is not None:
        plans.append(
            Ros2MessagePlan(
                topic=bridge_config.goal_topic,
                message_type=ROS2_POINT,
                payload={"x": goal.x, "y": goal.y, "z": goal.z},
                reason=goal.reason,
            )
        )

    return plans


def detect_estop_command(text: str) -> EstopCommand | None:
    normalized = normalize_asr_text(text)
    if not normalized or should_edit_rules(normalized) or _looks_explanatory(normalized):
        return None

    for phrase in _ESTOP_RELEASE_PHRASES:
        if phrase in normalized:
            return EstopCommand(active=False, reason=normalized)

    for phrase in _ESTOP_TRIGGER_PHRASES:
        if phrase in normalized:
            return EstopCommand(active=True, reason=normalized)

    return None


def detect_goal_command(text: str) -> GoalCommand | None:
    normalized = normalize_asr_text(text)
    if not normalized or should_edit_rules(normalized) or _looks_explanatory(normalized):
        return None
    if not any(word in normalized for word in _GOAL_WORDS):
        return None

    values = [float(match.group(0)) for match in _NUMBER_RE.finditer(normalized)]
    if len(values) < 3:
        return None
    x, y, z = values[:3]
    return GoalCommand(x=x, y=y, z=z, reason=normalized)


def plans_to_json(plans: list[Ros2MessagePlan]) -> str:
    return json.dumps([plan.as_dict() for plan in plans], indent=2, ensure_ascii=False)


def sync_estop_plans_to_arm_rules(
    plans: Sequence[Ros2MessagePlan],
    config: Ros2BridgeConfig,
    arm_rules_path: Path,
) -> ArmEstopRequestResult | None:
    command = estop_command_from_plans(plans, config)
    if command is None:
        return None
    return request_arm_estop(arm_rules_path, active=command.active)


def estop_command_from_plans(
    plans: Sequence[Ros2MessagePlan],
    config: Ros2BridgeConfig,
) -> EstopCommand | None:
    command: EstopCommand | None = None
    for plan in plans:
        detected = estop_command_from_plan(plan, config)
        if detected is not None:
            command = detected
    return command


def estop_command_from_plan(
    plan: Ros2MessagePlan,
    config: Ros2BridgeConfig,
) -> EstopCommand | None:
    if plan.topic == config.estop_request_topic and plan.message_type == ROS2_STRING:
        try:
            payload = json.loads(str(plan.payload))
        except (TypeError, ValueError):
            return None
        active = payload.get("active") if isinstance(payload, dict) else None
        if isinstance(active, bool):
            return EstopCommand(active=active, reason=str(payload.get("reason") or plan.reason))
        return None
    if plan.topic == config.estop_bool_topic and plan.message_type == ROS2_BOOL and isinstance(plan.payload, bool):
        return EstopCommand(active=plan.payload, reason=plan.reason)
    return None


class Ros2VoiceBridge:
    def __init__(self, config: Ros2BridgeConfig | None = None) -> None:
        self.config = config or Ros2BridgeConfig()

    def publish_turn(self, result: VoiceTurnResult) -> list[Ros2MessagePlan]:
        plans = build_voice_ros2_plan(result, self.config)
        self.publish_plans(plans)
        return plans

    def publish_plans(self, plans: list[Ros2MessagePlan]) -> None:
        if not plans:
            return

        try:
            import rclpy
            from geometry_msgs.msg import Point
            from rclpy.executors import SingleThreadedExecutor
            from std_msgs.msg import Bool, String
        except ImportError as exc:
            raise RuntimeError(
                "ROS2 bridge requires rclpy and ROS2 message packages. "
                "Source the ROS2 environment or rerun with --dry-run-ros2."
            ) from exc

        created_context = not rclpy.ok()
        if created_context:
            rclpy.init(args=None)

        node = rclpy.create_node(self.config.node_name)
        executor = SingleThreadedExecutor(context=node.context)
        executor.add_node(node)
        try:
            publishers: dict[tuple[str, str], Any] = {}
            for plan in plans:
                key = (plan.topic, plan.message_type)
                if key not in publishers:
                    message_class = _message_class(plan.message_type, String=String, Bool=Bool, Point=Point)
                    publishers[key] = node.create_publisher(message_class, plan.topic, self.config.qos_depth)

            deadline = time.monotonic() + max(self.config.wait_for_subscribers_seconds, 0.0)
            while time.monotonic() < deadline and rclpy.ok():
                executor.spin_once(timeout_sec=0.05)

            for plan in plans:
                publisher = publishers[(plan.topic, plan.message_type)]
                publisher.publish(_build_ros2_message(plan, String=String, Bool=Bool, Point=Point))

            executor.spin_once(timeout_sec=0.05)
        finally:
            executor.remove_node(node)
            executor.shutdown(timeout_sec=0.0)
            node.destroy_node()
            if created_context and rclpy.ok():
                rclpy.shutdown()


def _build_estop_plan(command: EstopCommand, config: Ros2BridgeConfig) -> Ros2MessagePlan:
    if not config.use_estop_request:
        return Ros2MessagePlan(
            topic=config.estop_bool_topic,
            message_type=ROS2_BOOL,
            payload=command.active,
            reason=command.reason,
        )

    payload_data: dict[str, Any] = {
        "source": config.source,
        "active": command.active,
        "latch": command.active,
        "reason": command.reason,
    }
    if not command.active:
        reset_sources = [
            source.strip()
            for source in config.estop_reset_sources
            if isinstance(source, str) and source.strip()
        ]
        if reset_sources:
            payload_data["reset_sources"] = reset_sources
    payload = json.dumps(payload_data, ensure_ascii=False)
    return Ros2MessagePlan(
        topic=config.estop_request_topic,
        message_type=ROS2_STRING,
        payload=payload,
        reason=command.reason,
    )


def _looks_explanatory(text: str) -> bool:
    if any(question_word in text for question_word in _QUESTION_WORDS):
        return True
    if any(force_word in text for force_word in _COMMAND_FORCE_WORDS):
        return False
    return any(word in text for word in _EXPLANATION_WORDS)


def _message_class(message_type: str, *, String: Any, Bool: Any, Point: Any) -> Any:
    if message_type == ROS2_STRING:
        return String
    if message_type == ROS2_BOOL:
        return Bool
    if message_type == ROS2_POINT:
        return Point
    raise ValueError(f"Unsupported ROS2 message type: {message_type}")


def _build_ros2_message(plan: Ros2MessagePlan, *, String: Any, Bool: Any, Point: Any) -> Any:
    if plan.message_type == ROS2_STRING:
        msg = String()
        msg.data = str(plan.payload)
        return msg
    if plan.message_type == ROS2_BOOL:
        msg = Bool()
        msg.data = bool(plan.payload)
        return msg
    if plan.message_type == ROS2_POINT:
        msg = Point()
        msg.x = float(plan.payload["x"])
        msg.y = float(plan.payload["y"])
        msg.z = float(plan.payload["z"])
        return msg
    raise ValueError(f"Unsupported ROS2 message type: {plan.message_type}")
