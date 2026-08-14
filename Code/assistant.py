#!/usr/bin/env python3
"""Compatibility entrypoint for the local safety assistant scaffold."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_safety_assistant.app import main


if __name__ == "__main__":
    raise SystemExit(main())
