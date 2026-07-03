"""Local speech-to-text via faster-whisper (CTranslate2 Whisper). Runs
entirely on-device — CPU by default, GPU if configured — with no network
calls at inference time (the model is downloaded once, ahead of time, to a
local cache)."""

from __future__ import annotations

import numpy as np

from .config import FlowConfig


class Transcriber:
    def __init__(self, config: FlowConfig):
        from faster_whisper import WhisperModel  # noqa: local import

        self.config = config
        self._model = WhisperModel(
            config.model_size,
            device=config.device,
            compute_type=config.compute_type,
        )

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        segments, _info = self._model.transcribe(
            audio,
            language=self.config.language,
            vad_filter=True,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()
