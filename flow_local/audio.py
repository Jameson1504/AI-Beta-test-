"""Local microphone capture. Audio is buffered in-process and handed to the
transcriber as a numpy array — it is never written to a socket."""

from __future__ import annotations

import numpy as np

from .config import FlowConfig


class AudioRecorder:
    """Records mono float32 audio from the default input device while
    `start`/`stop` bracket an utterance. Import of `sounddevice` is deferred
    to construction time so the rest of the package stays importable (and
    testable) on machines/containers with no audio hardware."""

    def __init__(self, config: FlowConfig):
        import sounddevice as sd  # noqa: local import, see class docstring

        self._sd = sd
        self.config = config
        self._chunks: list[np.ndarray] = []
        self._stream = None

    def _callback(self, indata, frames, time_info, status):
        self._chunks.append(indata[:, 0].copy())

    def start(self) -> None:
        self._chunks = []
        self._stream = self._sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            callback=self._callback,
            dtype="float32",
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(self._chunks)
        max_samples = int(self.config.max_utterance_seconds * self.config.sample_rate)
        return audio[:max_samples]
