"""Vision snapshot helpers for voice-triggered environment analysis."""

from __future__ import annotations

import json
import mimetypes
import secrets
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


DEFAULT_VISION_SNAPSHOT_SERVICE = "/vision/capture_snapshot"
DEFAULT_VISION_SNAPSHOT_TIMEOUT_SECONDS = 3.0
DEFAULT_VISION_CLIENT_NODE = "voice_vision_snapshot_client"
ORBBEC_UDEV_RULES_FILE = "/etc/udev/rules.d/99-obsensor-libusb.rules"
ORBBEC_USB_PERMISSION_ERROR_SUMMARY = "Orbbec/Gemini 摄像头 USB 打开失败"
SUPPORTED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})
_ORBBEC_USB_ERROR_SIGNATURES = (
    "usbenumerator openusbdevice failed",
    "openusbdevice failed",
    "access denied (insufficient permissions)",
    "device_unavailable",
)


@dataclass(frozen=True)
class VisionImageArtifact:
    image_path: Path
    source: str = "vision_snapshot"
    caption: str = "当前视觉快照"
    mime_type: str = "image/jpeg"
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "image_path": str(self.image_path),
            "source": self.source,
            "caption": self.caption,
            "mime_type": self.mime_type,
            "metadata": dict(self.metadata),
        }


class VisionSnapshotProvider(Protocol):
    def capture_snapshot(self) -> VisionImageArtifact: ...


def format_vision_snapshot_error(error: BaseException | str) -> str:
    """Return an operator-facing message for known snapshot runtime failures."""

    message = str(error)
    if ORBBEC_USB_PERMISSION_ERROR_SUMMARY in message:
        return message
    lowered = message.lower()
    if any(signature in lowered for signature in _ORBBEC_USB_ERROR_SIGNATURES):
        return (
            f"{ORBBEC_USB_PERMISSION_ERROR_SUMMARY}，通常是 Orbbec udev 权限规则未安装"
            "或安装后未重新拔插设备。请安装 pyorbbecsdk 自带的 "
            f"99-obsensor-libusb.rules 到 {ORBBEC_UDEV_RULES_FILE}，"
            "重新加载 udev 后拔插摄像头并重启视觉服务。"
            f"原始错误：{message}"
        )
    return message


def snapshot_from_trigger_message(message: str, *, source: str = "ros2_trigger") -> VisionImageArtifact:
    """Parse a std_srvs/Trigger message payload into a validated image artifact."""

    try:
        payload = json.loads(message)
    except json.JSONDecodeError as error:
        raise ValueError("Vision snapshot Trigger response message must be JSON.") from error
    if not isinstance(payload, dict):
        raise ValueError("Vision snapshot Trigger response JSON must be an object.")

    raw_path = payload.get("image_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("Vision snapshot Trigger response JSON must contain image_path.")

    image_path = validate_snapshot_image_path(Path(raw_path))
    metadata = {key: value for key, value in payload.items() if key != "image_path"}
    return VisionImageArtifact(
        image_path=image_path,
        source=source,
        caption="当前视觉快照",
        mime_type=_image_mime_type(image_path),
        metadata=metadata,
    )


def validate_snapshot_image_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Vision snapshot image does not exist: {resolved}")
    if resolved.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_SUFFIXES))
        raise ValueError(f"Vision snapshot image must use one of these suffixes: {supported}")
    return resolved


def copy_vision_artifact_to_dir(artifact: VisionImageArtifact, target_dir: Path) -> VisionImageArtifact:
    source_path = validate_snapshot_image_path(artifact.image_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    suffix = source_path.suffix.lower() or ".jpg"
    target_path = target_dir / f"vision_{int(time.time() * 1000)}_{secrets.token_hex(4)}{suffix}"
    shutil.copy2(source_path, target_path)
    return VisionImageArtifact(
        image_path=target_path.resolve(),
        source=artifact.source,
        caption=artifact.caption,
        mime_type=_image_mime_type(target_path),
        metadata=dict(artifact.metadata),
    )


class Ros2TriggerVisionSnapshotProvider:
    def __init__(
        self,
        *,
        service_name: str = DEFAULT_VISION_SNAPSHOT_SERVICE,
        timeout_seconds: float = DEFAULT_VISION_SNAPSHOT_TIMEOUT_SECONDS,
        node_name: str = DEFAULT_VISION_CLIENT_NODE,
    ) -> None:
        self.service_name = service_name
        self.timeout_seconds = timeout_seconds
        self.node_name = node_name

    def capture_snapshot(self) -> VisionImageArtifact:
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from std_srvs.srv import Trigger
        except ImportError as exc:
            raise RuntimeError(
                "ROS2 vision snapshot requires rclpy and std_srvs. "
                "Source the ROS2 environment or configure a fake provider for tests."
            ) from exc

        created_context = not rclpy.ok()
        if created_context:
            rclpy.init(args=None)

        node = rclpy.create_node(self.node_name)
        executor = SingleThreadedExecutor(context=node.context)
        executor.add_node(node)
        try:
            client = node.create_client(Trigger, self.service_name)
            if not client.wait_for_service(timeout_sec=self.timeout_seconds):
                raise RuntimeError(
                    f"Vision snapshot service is not available: {self.service_name}"
                )
            future = client.call_async(Trigger.Request())
            executor.spin_until_future_complete(future, timeout_sec=self.timeout_seconds)
            if not future.done():
                raise RuntimeError(
                    f"Vision snapshot service timed out after {self.timeout_seconds:.1f}s: {self.service_name}"
                )
            response = future.result()
            if response is None:
                raise RuntimeError(f"Vision snapshot service returned no response: {self.service_name}")
            if not bool(getattr(response, "success", False)):
                message = format_vision_snapshot_error(
                    str(getattr(response, "message", "") or "snapshot request failed")
                )
                raise RuntimeError(f"Vision snapshot service failed: {message}")
            return snapshot_from_trigger_message(
                str(getattr(response, "message", "")),
                source=self.service_name,
            )
        finally:
            executor.remove_node(node)
            executor.shutdown(timeout_sec=0.0)
            node.destroy_node()
            if created_context and rclpy.ok():
                rclpy.shutdown()


def _image_mime_type(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"
