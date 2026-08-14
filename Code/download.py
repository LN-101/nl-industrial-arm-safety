#!/usr/bin/env python3
"""Optional downloader for local OpenVINO model assets.

The smoke tests are offline once models exist under ./models. This script is
only for fetching or refreshing those local model directories.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_REPOS = {
    "asr-whisper-large-v3": (
        "OpenVINO/whisper-large-v3-int4-ov",
        Path("models/asr/whisper-large-v3-int4-ov"),
    ),
    "asr-whisper-large-v3-turbo": (
        "OpenVINO/whisper-large-v3-turbo-int4-ov",
        Path("models/asr/whisper-large-v3-turbo-int4-ov"),
    ),
    "qwen35-2b": (
        "OpenVINO/Qwen3.5-2B-int4-ov",
        Path("models/Qwen3.5-2B-int4-ov"),
    ),
    "qwen35-9b": (
        "OpenVINO/Qwen3.5-9B-int4-ov",
        Path("models/Qwen3.5-9B-int4-ov"),
    ),
    "qwen3-8b": (
        "OpenVINO/Qwen3-8B-int4-cw-ov",
        Path("models/Qwen3-8B-int4-cw-ov"),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download OpenVINO models into ./models.")
    parser.add_argument("model", choices=sorted(MODEL_REPOS), help="Model alias to download.")
    parser.add_argument("--local-dir", type=Path, help="Override the default local model directory.")
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Optional Hugging Face token. Defaults to HF_TOKEN if set.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_id, default_dir = MODEL_REPOS[args.model]
    local_dir = args.local_dir or default_dir

    print(f"Downloading {repo_id} -> {local_dir}")
    snapshot_download(repo_id, local_dir=local_dir, token=args.token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
