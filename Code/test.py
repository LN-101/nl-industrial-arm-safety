#!/usr/bin/env python3
"""Compatibility entrypoint for the local OpenVINO model testbed.

Run with:
    ./qwen35_env/bin/python Code/test.py --list-devices
    ./qwen35_env/bin/python Code/test.py inventory
    ./qwen35_env/bin/python Code/test.py generate --model qwen35-2b
"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_safety_assistant.model_testbed import main


if __name__ == "__main__":
    raise SystemExit(main())
