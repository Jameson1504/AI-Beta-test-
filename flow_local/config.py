"""Configuration for flow_local: everything a user might want to tune,
in one dataclass with sane offline-first defaults."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class FlowConfig:
    # Activation
    hotkey: str = "<ctrl>+<alt>+space"     # push-to-talk combo (pynput format)
    mode: str = "push_to_talk"             # "push_to_talk" | "toggle"

    # Audio capture
    sample_rate: int = 16_000
    channels: int = 1
    max_utterance_seconds: float = 60.0

    # ASR (faster-whisper / CTranslate2, fully local)
    model_size: str = "base.en"            # tiny.en/base.en/small.en/medium.en/...
    device: str = "cpu"                    # "cpu" | "cuda"
    compute_type: str = "int8"             # int8 (fast on CPU) | float16 (GPU)
    language: str = "en"

    # Cleanup
    remove_filler_words: bool = True
    auto_punctuate_capitalize: bool = True
    use_local_llm_cleanup: bool = False    # opt-in, requires local Ollama at 127.0.0.1
    llm_cleanup_model: str = "llama3.2"

    # Injection
    injection_mode: str = "type"           # "type" (keystrokes) | "paste" (clipboard)
    type_delay_seconds: float = 0.0

    @classmethod
    def load(cls, path: str | Path) -> "FlowConfig":
        data = json.loads(Path(path).read_text())
        return cls(**data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")
