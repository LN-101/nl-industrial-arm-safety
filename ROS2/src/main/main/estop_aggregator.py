"""Aggregate multi-source emergency-stop requests into a Bool topic."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any


DEFAULT_REQUEST_TOPIC = '/safety/estop/request'
DEFAULT_OUTPUT_TOPIC = '/emergency_stop'
DEFAULT_QOS_DEPTH = 10


@dataclass(frozen=True)
class EstopRequest:
    source: str
    active: bool
    latch: bool
    reason: str = ''


def parse_estop_request(text: str) -> tuple[EstopRequest | None, str | None]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f'invalid JSON: {exc}'

    if not isinstance(payload, dict):
        return None, 'payload must be a JSON object'

    source = payload.get('source')
    if not isinstance(source, str) or not source.strip():
        return None, "field 'source' must be a non-empty string"

    active = payload.get('active')
    if not isinstance(active, bool):
        return None, "field 'active' must be a boolean"

    latch_value = payload.get('latch', active)
    if not isinstance(latch_value, bool):
        return None, "field 'latch' must be a boolean"

    reason = payload.get('reason', '')
    if reason is None:
        reason_text = ''
    elif isinstance(reason, str):
        reason_text = reason.strip()
    else:
        reason_text = str(reason)

    return (
        EstopRequest(
            source=source.strip(),
            active=active,
            latch=latch_value,
            reason=reason_text,
        ),
        None,
    )


class MultiSourceEstopState:
    def __init__(self) -> None:
        self._sources: dict[str, bool] = {}

    @property
    def active_sources(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                source
                for source, active in self._sources.items()
                if active
            )
        )

    @property
    def effective_active(self) -> bool:
        return any(self._sources.values())

    def apply(self, request: EstopRequest) -> bool:
        if request.active:
            self._sources[request.source] = True
        elif not request.latch:
            self._sources[request.source] = False
        return self.effective_active


class EstopAggregatorNode:
    def __init__(self) -> None:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import Bool, String

        class _Node(Node):
            def __init__(self) -> None:
                super().__init__('estop_aggregator')
                self.declare_parameter('request_topic', DEFAULT_REQUEST_TOPIC)
                self.declare_parameter('output_topic', DEFAULT_OUTPUT_TOPIC)
                self.declare_parameter('qos_depth', DEFAULT_QOS_DEPTH)

                request_topic = str(
                    self.get_parameter('request_topic').value
                    or DEFAULT_REQUEST_TOPIC
                )
                output_topic = str(
                    self.get_parameter('output_topic').value
                    or DEFAULT_OUTPUT_TOPIC
                )
                qos_depth = int(
                    self.get_parameter('qos_depth').value
                    or DEFAULT_QOS_DEPTH
                )

                self._state = MultiSourceEstopState()
                self._publisher = self.create_publisher(
                    Bool,
                    output_topic,
                    qos_depth,
                )
                self._subscription = self.create_subscription(
                    String,
                    request_topic,
                    self._on_request,
                    qos_depth,
                )
                self.get_logger().info(
                    'estop aggregator listening on '
                    f'{request_topic}, publishing Bool to {output_topic}'
                )

            def _on_request(self, msg: Any) -> None:
                request, error = parse_estop_request(
                    str(getattr(msg, 'data', ''))
                )
                if error is not None or request is None:
                    self.get_logger().warn(
                        f'ignored invalid estop request: {error}'
                    )
                    return

                effective = self._state.apply(request)
                out = Bool()
                out.data = effective
                self._publisher.publish(out)
                active_sources = ','.join(self._state.active_sources) or 'none'
                self.get_logger().info(
                    f'estop source={request.source} active={request.active} '
                    f'latch={request.latch} effective={effective} '
                    f'active_sources={active_sources} '
                    f'reason={request.reason}'
                )

        self._rclpy = rclpy
        self._node = _Node()

    def spin(self) -> None:
        self._rclpy.spin(self._node)

    def destroy(self) -> None:
        self._node.destroy_node()


def main(args: list[str] | None = None) -> None:
    import rclpy

    rclpy.init(args=args)
    node = EstopAggregatorNode()
    try:
        node.spin()
    finally:
        node.destroy()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
