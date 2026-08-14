"""HTTP server for the mobile hotspot control surface."""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import os
import re
import ssl
import sys
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from local_safety_assistant.web.service import (
    WEB_VOICE_UPLOAD_MAX_BYTES,
    WebUiConfig,
    WebUiService,
    format_byte_limit,
)
from local_safety_assistant.web.ui import render_index_html


DEFAULT_MAX_REQUEST_BYTES = 25 * 1024 * 1024
MOSS_CPU_LIST_PATTERN = re.compile(r"^[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*$")


def _positive_buffer_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("MOSS PCM buffer seconds must be a number") from error
    if not 0.1 <= seconds <= 30.0:
        raise argparse.ArgumentTypeError("MOSS PCM buffer seconds must be from 0.1 through 30.0")
    return seconds


def _optional_cpu_list(value: str) -> str | None:
    normalized = value.strip()
    if not normalized:
        return None
    if MOSS_CPU_LIST_PATTERN.fullmatch(normalized) is None:
        raise argparse.ArgumentTypeError("MOSS CPU list must look like 0-3 or 0,2,4-6")
    requested_cpus: set[int] = set()
    for part in normalized.split(","):
        bounds = tuple(int(item) for item in part.split("-", maxsplit=1))
        start = bounds[0]
        end = bounds[-1]
        if end < start:
            raise argparse.ArgumentTypeError("MOSS CPU list ranges must be ascending")
        requested_cpus.update(range(start, end + 1))
    available_cpus = set(os.sched_getaffinity(0))
    unavailable = sorted(requested_cpus - available_cpus)
    if unavailable:
        raise argparse.ArgumentTypeError(f"MOSS CPU list includes unavailable CPUs: {unavailable}")
    return normalized


class WebUiHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], service: WebUiService) -> None:
        super().__init__(server_address, WebUiRequestHandler)
        self.service = service


class WebUiRequestHandler(BaseHTTPRequestHandler):
    server: WebUiHTTPServer

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(
                render_index_html(
                    self.server.service.config.title,
                    moss_pcm_buffer_seconds=self.server.service.config.moss_pcm_buffer_seconds,
                )
            )
            return
        if parsed.path == "/api/status":
            self._send_json(self.server.service.status(self._session_token()))
            return
        if parsed.path.startswith("/api/chat-stream/"):
            if not self._require_auth():
                return
            suffix = parsed.path.removeprefix("/api/chat-stream/")
            if suffix.endswith("/audio"):
                self._handle_chat_stream_audio(suffix.removesuffix("/audio"))
                return
            self._handle_chat_stream_status(parsed.path.removeprefix("/api/chat-stream/"))
            return
        if parsed.path == "/emergency-alert-audio":
            self._handle_emergency_alert_audio()
            return
        if parsed.path.startswith("/audio/"):
            self._handle_audio(parsed.path.removeprefix("/audio/"))
            return
        if parsed.path.startswith("/images/"):
            if not self._require_auth():
                return
            self._handle_image(parsed.path.removeprefix("/images/"))
            return
        self._send_error_json(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/login":
            self._handle_login()
            return
        if parsed.path == "/api/logout":
            self._handle_logout()
            return
        if not self._require_auth():
            return
        if parsed.path == "/api/chat":
            self._handle_chat()
            return
        if parsed.path == "/api/chat-stream":
            self._handle_chat_stream()
            return
        if parsed.path == "/api/voice":
            self._handle_voice()
            return
        if parsed.path == "/api/voice-stream":
            self._handle_voice_stream()
            return
        if parsed.path == "/api/turn/cancel":
            self._handle_turn_cancel()
            return
        if parsed.path == "/api/confirmation/confirm":
            self._handle_confirmation_confirm()
            return
        if parsed.path == "/api/confirmation/cancel":
            self._handle_confirmation_cancel()
            return
        if parsed.path == "/api/estop":
            self._handle_estop()
            return
        if parsed.path == "/api/estop/external":
            self._handle_estop_external()
            return
        self._send_error_json(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        sys.stdout.write(f"[web-ui] {self.address_string()} - {format % args}\n")

    def _handle_login(self) -> None:
        try:
            payload = self._read_json_body()
        except ValueError as error:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(error))
            return
        username = str(payload.get("username", ""))
        password = str(payload.get("password", ""))
        token = self.server.service.login(username, password)
        if token is None:
            self._send_error_json(HTTPStatus.UNAUTHORIZED, "用户名或密码错误")
            return
        self._send_json(
            {
                "ok": True,
                "authenticated": True,
                "status_line": "已登录",
                "message": "登录成功",
            },
            cookie_token=token,
        )

    def _handle_logout(self) -> None:
        self.server.service.logout(self._session_token())
        self._send_json(
            {
                "ok": True,
                "authenticated": False,
                "status_line": "已退出",
            },
            clear_cookie=True,
        )

    def _handle_chat(self) -> None:
        try:
            payload = self._read_json_body()
        except ValueError as error:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(error))
            return
        text = str(payload.get("text", ""))
        try:
            result = self.server.service.handle_chat(text)
        except FileNotFoundError as error:
            self._send_error_json(HTTPStatus.SERVICE_UNAVAILABLE, str(error))
            return
        except RuntimeError as error:
            self._send_error_json(HTTPStatus.SERVICE_UNAVAILABLE, str(error))
            return
        except ValueError as error:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(error))
            return
        self._send_json(result.as_dict())

    def _handle_chat_stream(self) -> None:
        try:
            payload = self._read_json_body()
        except ValueError as error:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(error))
            return
        text = str(payload.get("text", ""))
        try:
            result = self.server.service.handle_chat_progressive(text)
        except FileNotFoundError as error:
            self._send_error_json(HTTPStatus.SERVICE_UNAVAILABLE, str(error))
            return
        except RuntimeError as error:
            self._send_error_json(HTTPStatus.SERVICE_UNAVAILABLE, str(error))
            return
        except ValueError as error:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(error))
            return
        self._send_json(result)

    def _handle_voice(self) -> None:
        try:
            audio_bytes = self._read_raw_body(
                max_bytes=WEB_VOICE_UPLOAD_MAX_BYTES,
                too_large_error=f"Voice upload too large; max {format_byte_limit(WEB_VOICE_UPLOAD_MAX_BYTES)}",
            )
        except ValueError as error:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(error))
            return
        content_type = self.headers.get("Content-Type")
        try:
            result = self.server.service.handle_voice(audio_bytes, content_type=content_type)
        except FileNotFoundError as error:
            self._send_error_json(HTTPStatus.SERVICE_UNAVAILABLE, str(error))
            return
        except RuntimeError as error:
            self._send_error_json(HTTPStatus.SERVICE_UNAVAILABLE, str(error))
            return
        except ValueError as error:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(error))
            return
        self._send_json(result.as_dict())

    def _handle_voice_stream(self) -> None:
        try:
            audio_bytes = self._read_raw_body(
                max_bytes=WEB_VOICE_UPLOAD_MAX_BYTES,
                too_large_error=f"Voice upload too large; max {format_byte_limit(WEB_VOICE_UPLOAD_MAX_BYTES)}",
            )
        except ValueError as error:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(error))
            return
        content_type = self.headers.get("Content-Type")
        try:
            result = self.server.service.handle_voice_progressive(audio_bytes, content_type=content_type)
        except FileNotFoundError as error:
            self._send_error_json(HTTPStatus.SERVICE_UNAVAILABLE, str(error))
            return
        except RuntimeError as error:
            self._send_error_json(HTTPStatus.SERVICE_UNAVAILABLE, str(error))
            return
        except ValueError as error:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(error))
            return
        self._send_json(result)

    def _handle_turn_cancel(self) -> None:
        try:
            self._read_json_body()
        except ValueError as error:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(error))
            return
        result = self.server.service.cancel_active_web_turns()
        self._send_json(result.as_dict())

    def _handle_chat_stream_status(self, turn_id: str) -> None:
        result = self.server.service.progressive_turn_status(Path(turn_id).name)
        if result is None:
            self._send_error_json(HTTPStatus.NOT_FOUND, "Turn not found")
            return
        self._send_json(result)

    def _handle_chat_stream_audio(self, turn_id: str) -> None:
        try:
            result = self.server.service.progressive_turn_audio(Path(turn_id).name)
        except RuntimeError as error:
            self._send_error_json(HTTPStatus.SERVICE_UNAVAILABLE, str(error))
            return
        if result is None:
            self._send_error_json(HTTPStatus.NOT_FOUND, "Audio stream not found")
            return
        self._send_audio_stream(result)

    def _handle_estop(self) -> None:
        result = self.server.service.handle_estop()
        status = HTTPStatus.OK if result.ok else HTTPStatus.SERVICE_UNAVAILABLE
        self._send_json(result.as_dict(), status=status)

    def _handle_estop_external(self) -> None:
        try:
            payload = self._read_json_body()
        except ValueError as error:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(error))
            return
        try:
            result = self.server.service.ingest_external_estop_event(payload)
        except ValueError as error:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(error))
            return
        self._send_json(result)

    def _handle_emergency_alert_audio(self) -> None:
        path = self.server.service.serve_emergency_alert_audio_path()
        if path is None:
            self._send_error_json(HTTPStatus.NOT_FOUND, "Emergency alert audio not found")
            return
        media_type = mimetypes.guess_type(path.name)[0] or "audio/wav"
        self._send_file(path, media_type)

    def _handle_confirmation_confirm(self) -> None:
        try:
            payload = self._read_json_body()
        except ValueError as error:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(error))
            return
        confirmation_id = str(payload.get("id") or payload.get("confirmation_id") or "")
        if not confirmation_id:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "confirmation id is required")
            return
        result = self.server.service.confirm_pending(confirmation_id)
        status = HTTPStatus.OK if result.status == "confirmed" else HTTPStatus.BAD_REQUEST
        self._send_json(result.as_dict(), status=status)

    def _handle_confirmation_cancel(self) -> None:
        try:
            payload = self._read_json_body()
        except ValueError as error:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(error))
            return
        confirmation_id = payload.get("id") or payload.get("confirmation_id")
        result = self.server.service.cancel_pending(str(confirmation_id) if confirmation_id else None)
        self._send_json(result.as_dict())

    def _handle_audio(self, filename: str) -> None:
        path = self.server.service.serve_audio_path(filename)
        if path is None:
            self._send_error_json(HTTPStatus.NOT_FOUND, "Audio not found")
            return
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send_file(path, media_type)

    def _handle_image(self, filename: str) -> None:
        path = self.server.service.serve_image_path(filename)
        if path is None:
            self._send_error_json(HTTPStatus.NOT_FOUND, "Image not found")
            return
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send_file(path, media_type)

    def _require_auth(self) -> bool:
        if self.server.service.is_authenticated(self._session_token()):
            return True
        self._send_error_json(HTTPStatus.UNAUTHORIZED, "请先登录")
        return False

    def _send_html(self, html: str, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(
        self,
        payload: dict[str, Any],
        *,
        status: HTTPStatus = HTTPStatus.OK,
        cookie_token: str | None = None,
        clear_cookie: bool = False,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        if cookie_token is not None:
            self.send_header("Set-Cookie", self._cookie_header(cookie_token))
        elif clear_cookie:
            self.send_header("Set-Cookie", self._clear_cookie_header())
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"ok": False, "error": message}, status=status)

    def _send_file(self, path: Path, media_type: str) -> None:
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_audio_stream(self, result: Any) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("X-Audio-Codec", "pcm_s16le")
        self.send_header("X-Audio-Sample-Rate", str(result.sample_rate))
        self.send_header("X-Audio-Channels", str(result.channels))
        self.send_header("X-Stream-Id", result.stream_id)
        self.end_headers()
        try:
            for chunk in result.iterator:
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return
        finally:
            close = getattr(result.iterator, "close", None)
            if callable(close):
                close()

    def _read_json_body(self) -> dict[str, Any]:
        body = self._read_raw_body()
        if not body:
            return {}
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("Invalid JSON body") from error
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def _read_raw_body(
        self,
        *,
        max_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
        too_large_error: str | None = None,
    ) -> bytes:
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
        except ValueError as error:
            raise ValueError("Invalid Content-Length header") from error
        if length > max_bytes:
            raise ValueError(too_large_error or f"Request body too large; max {format_byte_limit(max_bytes)}")
        return self.rfile.read(length) if length else b""

    def _session_token(self) -> str | None:
        cookie_header = self.headers.get("Cookie")
        if not cookie_header:
            return None
        cookie = SimpleCookie()
        cookie.load(cookie_header)
        morsel = cookie.get("ai_ov_web_session")
        if morsel is None:
            return None
        return morsel.value

    def _cookie_header(self, token: str) -> str:
        cookie = SimpleCookie()
        cookie["ai_ov_web_session"] = token
        cookie["ai_ov_web_session"]["path"] = "/"
        cookie["ai_ov_web_session"]["httponly"] = True
        cookie["ai_ov_web_session"]["samesite"] = "Lax"
        cookie["ai_ov_web_session"]["max-age"] = str(self.server.service.config.session_ttl_seconds)
        return cookie.output(header="").strip()

    def _clear_cookie_header(self) -> str:
        cookie = SimpleCookie()
        cookie["ai_ov_web_session"] = ""
        cookie["ai_ov_web_session"]["path"] = "/"
        cookie["ai_ov_web_session"]["httponly"] = True
        cookie["ai_ov_web_session"]["samesite"] = "Lax"
        cookie["ai_ov_web_session"]["max-age"] = "0"
        return cookie.output(header="").strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mobile hotspot web UI for the local safety assistant.")
    parser.add_argument("--host", default=WebUiConfig().host)
    parser.add_argument("--port", type=int, default=WebUiConfig().port)
    parser.add_argument("--title", default=WebUiConfig().title)
    parser.add_argument("--admin-username", default=WebUiConfig().admin_username)
    parser.add_argument("--admin-password", default=WebUiConfig().admin_password)
    parser.add_argument("--session-ttl-seconds", type=int, default=WebUiConfig().session_ttl_seconds)
    parser.add_argument("--runtime-dir", type=Path, default=WebUiConfig().runtime_dir)
    parser.add_argument("--tts-engine", choices=("auto", "moss", "melo", "piper"), default=WebUiConfig().tts_engine)
    parser.add_argument(
        "--moss-pcm-buffer-seconds",
        type=_positive_buffer_seconds,
        default=WebUiConfig().moss_pcm_buffer_seconds,
    )
    parser.add_argument("--moss-cpus", type=_optional_cpu_list, default=WebUiConfig().moss_cpu_affinity)
    parser.add_argument("--dry-run-ros2", action="store_true")
    parser.add_argument("--direct-estop-topic", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true", help="Start serving without preloading local ASR/LLM/TTS runtimes.")
    parser.add_argument(
        "--emergency-alert-audio",
        type=Path,
        help=(
            "Pre-generated alert audio file the browser plays directly when an external "
            "emergency stop is detected. Default: built-in emergency alert audio"
        ),
    )
    parser.add_argument("--ssl-certfile", type=Path, help="TLS certificate file for HTTPS microphone access.")
    parser.add_argument("--ssl-keyfile", type=Path, help="TLS private key file. Optional if included in certfile.")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.ssl_keyfile is not None and args.ssl_certfile is None:
        parser.error("--ssl-keyfile requires --ssl-certfile")
    service = WebUiService.build_default(
        WebUiConfig(
            host=args.host,
            port=args.port,
            title=args.title,
            admin_username=args.admin_username,
            admin_password=args.admin_password,
            session_ttl_seconds=args.session_ttl_seconds,
            runtime_dir=args.runtime_dir,
            tts_engine=args.tts_engine,
            moss_pcm_buffer_seconds=args.moss_pcm_buffer_seconds,
            moss_cpu_affinity=args.moss_cpus,
            ros2_dry_run=args.dry_run_ros2,
            direct_estop_topic=args.direct_estop_topic,
            warm_start=not args.skip_warmup,
            emergency_alert_audio=args.emergency_alert_audio,
        )
    )
    if service.config.warm_start:
        print("Warmup: loading Web voice stack components (ASR, qwen35-2b, MOSS TTS; excluding qwen35-9b).")
        warmup = service.warm_start()
        print(f"Warmup complete in {warmup['elapsed_seconds']:.3f}s; excluded: {', '.join(warmup['excluded_models'])}.")
    if not args.dry_run_ros2:
        try:
            service.start_external_estop_listener()
            print("External emergency-stop alert listener subscribed to /safety/estop/request.")
        except RuntimeError as error:
            print(f"External emergency-stop alert listener unavailable: {error}")
    server = WebUiHTTPServer((service.config.host, service.config.port), service)
    scheme = "http"
    if args.ssl_certfile is not None:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(
            certfile=str(args.ssl_certfile.expanduser()),
            keyfile=str(args.ssl_keyfile.expanduser()) if args.ssl_keyfile is not None else None,
        )
        server.socket = context.wrap_socket(server.socket, server_side=True)
        scheme = "https"
    address = server.server_address
    print(f"Web UI listening on {scheme}://{address[0]}:{address[1]}/")
    print(f"Login: {service.config.admin_username} / {service.config.admin_password}")
    print("Web UI ready: warmup complete and HTTP server accepting traffic." if service.config.warm_start else "Web UI ready: warmup skipped and HTTP server accepting traffic.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
        service.shutdown()
    return 0
