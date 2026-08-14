from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_safety_assistant.stack.asr import DEFAULT_ASR_HOTWORDS, WhisperAsrEngine


class FakeWhisperPipe:
    def __init__(self) -> None:
        self.generate_kwargs: dict[str, str] = {}

    def generate(self, audio: list[float], **kwargs: str) -> SimpleNamespace:
        self.generate_kwargs = kwargs
        return SimpleNamespace(text="机械臂安全规则")


class WhisperAsrEngineTest(unittest.TestCase):
    def transcribe_with(self, hotwords: tuple[str, ...] | str | None = DEFAULT_ASR_HOTWORDS) -> FakeWhisperPipe:
        pipe = FakeWhisperPipe()
        engine = WhisperAsrEngine(
            model="whisper",
            device="CPU",
            cache_dir=Path("/tmp/cache"),
            language="zh",
            hotwords=hotwords,
        )

        with patch.object(engine, "_load", return_value=(pipe, 0.0)):
            result = engine.transcribe_audio(np.zeros(160, dtype=np.float32), audio_seconds=0.01)

        self.assertEqual(result.text, "机械臂安全规则")
        return pipe

    def test_default_hotwords_are_passed_to_whisper(self) -> None:
        pipe = self.transcribe_with()

        self.assertEqual(
            pipe.generate_kwargs["hotwords"],
            "机械臂，安全，规则，急停，距离，人员，限制，气泵，映射，A，B，C，D，ABCD，标号A，标号B，标号C，标号D",
        )
        self.assertEqual(pipe.generate_kwargs["language"], "<|zh|>")

    def test_hotword_string_can_be_overridden(self) -> None:
        pipe = self.transcribe_with("气泵，光栅")

        self.assertEqual(pipe.generate_kwargs["hotwords"], "气泵，光栅")

    def test_hotwords_can_be_disabled(self) -> None:
        pipe = self.transcribe_with(None)

        self.assertNotIn("hotwords", pipe.generate_kwargs)


if __name__ == "__main__":
    unittest.main()
