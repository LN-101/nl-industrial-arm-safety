"""ROS2 snapshot node for RGB camera images."""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from local_safety_assistant.config import PROJECT_ROOT
from local_safety_assistant.stack.vision import DEFAULT_VISION_SNAPSHOT_SERVICE


DEFAULT_VISION_IMAGE_TOPIC = "/camera/color/image_raw"
DEFAULT_VISION_SNAPSHOT_DIR = PROJECT_ROOT / ".runtime" / "vision_snapshots"
DEFAULT_MAX_IMAGE_AGE_SECONDS = 2.0


@dataclass(frozen=True)
class VisionSnapshotNodeConfig:
    image_topic: str = DEFAULT_VISION_IMAGE_TOPIC
    service_name: str = DEFAULT_VISION_SNAPSHOT_SERVICE
    output_dir: Path = DEFAULT_VISION_SNAPSHOT_DIR
    max_image_age_seconds: float = DEFAULT_MAX_IMAGE_AGE_SECONDS
    jpeg_quality: int = 92


def image_message_to_rgb_array(message: Any) -> np.ndarray:
    """Convert common sensor_msgs/Image encodings to an RGB uint8 array."""

    height = int(message.height)
    width = int(message.width)
    step = int(message.step)
    encoding = str(message.encoding).lower()
    if height <= 0 or width <= 0 or step <= 0:
        raise ValueError("Image message dimensions must be positive.")

    channels_by_encoding = {
        "rgb8": 3,
        "bgr8": 3,
        "rgba8": 4,
        "bgra8": 4,
        "mono8": 1,
    }
    channels = channels_by_encoding.get(encoding)
    if channels is None:
        raise ValueError(f"Unsupported image encoding for RGB snapshot: {message.encoding}")

    expected_row_bytes = width * channels
    if step < expected_row_bytes:
        raise ValueError(
            f"Image message step {step} is smaller than expected row bytes {expected_row_bytes}."
        )

    raw = np.frombuffer(bytes(message.data), dtype=np.uint8)
    expected_total_bytes = step * height
    if raw.size < expected_total_bytes:
        raise ValueError(
            f"Image message data is too short: {raw.size} bytes, expected at least {expected_total_bytes}."
        )

    rows = raw[:expected_total_bytes].reshape((height, step))[:, :expected_row_bytes]
    array = rows.reshape((height, width, channels))
    if encoding == "rgb8":
        return array.copy()
    if encoding == "bgr8":
        return array[:, :, ::-1].copy()
    if encoding == "rgba8":
        return array[:, :, :3].copy()
    if encoding == "bgra8":
        return array[:, :, [2, 1, 0]].copy()
    return np.repeat(array, 3, axis=2)


def save_image_message_as_jpeg(message: Any, output_path: Path, *, quality: int) -> None:
    from PIL import Image

    array = image_message_to_rgb_array(message)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(output_path, format="JPEG", quality=quality)


def snapshot_payload(image_path: Path, message: Any, *, topic: str) -> str:
    header = getattr(message, "header", None)
    frame_id = str(getattr(header, "frame_id", "") or "")
    stamp = getattr(header, "stamp", None)
    stamp_text = _stamp_to_text(stamp)
    return json.dumps(
        {
            "image_path": str(image_path),
            "stamp": stamp_text,
            "frame_id": frame_id,
            "topic": topic,
            "encoding": str(getattr(message, "encoding", "")),
        },
        ensure_ascii=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ROS2 RGB camera snapshot Trigger provider.")
    parser.add_argument("--image-topic", default=DEFAULT_VISION_IMAGE_TOPIC)
    parser.add_argument("--service-name", default=DEFAULT_VISION_SNAPSHOT_SERVICE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_VISION_SNAPSHOT_DIR)
    parser.add_argument("--max-image-age-seconds", type=float, default=DEFAULT_MAX_IMAGE_AGE_SECONDS)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = VisionSnapshotNodeConfig(
        image_topic=args.image_topic,
        service_name=args.service_name,
        output_dir=args.output_dir,
        max_image_age_seconds=args.max_image_age_seconds,
        jpeg_quality=max(1, min(int(args.jpeg_quality), 100)),
    )
    run_snapshot_node(config)
    return 0


def run_snapshot_node(config: VisionSnapshotNodeConfig) -> None:
    try:
        import rclpy
        from sensor_msgs.msg import Image
        from std_srvs.srv import Trigger
    except ImportError as exc:
        raise RuntimeError(
            "Vision snapshot node requires rclpy, sensor_msgs, and std_srvs. "
            "Source the ROS2 environment before running this command."
        ) from exc

    rclpy.init(args=None)
    node = rclpy.create_node("vision_snapshot_provider")
    snapshot_node = _VisionSnapshotNode(node, config, Image=Image, Trigger=Trigger)
    try:
        node.get_logger().info(
            f"Vision snapshot service {config.service_name} subscribed to {config.image_topic}"
        )
        rclpy.spin(node)
    finally:
        snapshot_node.destroy()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


class _VisionSnapshotNode:
    def __init__(self, node: Any, config: VisionSnapshotNodeConfig, *, Image: Any, Trigger: Any) -> None:
        self.node = node
        self.config = config
        self._lock = threading.Lock()
        self._latest_message: Any | None = None
        self._latest_monotonic: float | None = None
        self._subscription = node.create_subscription(Image, config.image_topic, self._on_image, 10)
        self._service = node.create_service(Trigger, config.service_name, self._on_trigger)

    def destroy(self) -> None:
        self.node.destroy_subscription(self._subscription)
        self.node.destroy_service(self._service)

    def _on_image(self, message: Any) -> None:
        with self._lock:
            self._latest_message = message
            self._latest_monotonic = time.monotonic()

    def _on_trigger(self, request: Any, response: Any) -> Any:  # noqa: ARG002
        with self._lock:
            message = self._latest_message
            captured_at = self._latest_monotonic

        if message is None or captured_at is None:
            response.success = False
            response.message = f"No image has been received on {self.config.image_topic}."
            return response

        age_seconds = time.monotonic() - captured_at
        if age_seconds > self.config.max_image_age_seconds:
            response.success = False
            response.message = (
                f"Latest image is stale: {age_seconds:.2f}s old on {self.config.image_topic}."
            )
            return response

        filename = f"{_safe_name(self.config.image_topic)}_{int(time.time() * 1000)}.jpg"
        image_path = self.config.output_dir / filename
        try:
            save_image_message_as_jpeg(message, image_path, quality=self.config.jpeg_quality)
        except Exception as error:
            response.success = False
            response.message = f"Failed to save snapshot: {type(error).__name__}: {error}"
            return response

        response.success = True
        response.message = snapshot_payload(image_path.resolve(), message, topic=self.config.image_topic)
        return response


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip("/")) or "camera"


def _stamp_to_text(stamp: Any) -> str:
    if stamp is None:
        return ""
    sec = int(getattr(stamp, "sec", 0) or 0)
    nanosec = int(getattr(stamp, "nanosec", 0) or 0)
    if sec <= 0 and nanosec <= 0:
        return ""
    return f"{sec}.{nanosec:09d}"


if __name__ == "__main__":
    raise SystemExit(main())
