"""Wires the pieces together: hotkey -> record -> transcribe -> clean up ->
inject. This is the whole "product" — everything downstream of it is CLI
plumbing."""

from __future__ import annotations

import logging
import time

from .audio import AudioRecorder
from .cleanup import TranscriptProcessor
from .config import FlowConfig
from .hotkey import HotkeyListener
from .inject import TextInjector
from .transcribe import Transcriber

logger = logging.getLogger("flow_local")


class FlowApp:
    def __init__(self, config: FlowConfig):
        self.config = config
        self.recorder = AudioRecorder(config)
        self.transcriber = Transcriber(config)
        self.processor = TranscriptProcessor(config)
        self.injector = TextInjector(config)
        self.hotkey = HotkeyListener(config, on_start=self._on_start, on_stop=self._on_stop)
        self._last_inserted_len = 0

    def _on_start(self) -> None:
        logger.info("listening...")
        self.recorder.start()

    def _on_stop(self) -> None:
        audio = self.recorder.stop()
        raw_text = self.transcriber.transcribe(audio)
        if not raw_text:
            return
        result = self.processor.process(raw_text)

        if result.action == "insert":
            if self.config.use_local_llm_cleanup:
                from .llm_cleanup import refine_with_local_llm

                result.text = refine_with_local_llm(result.text, self.config.llm_cleanup_model)
            self.injector.inject(result.text)
            self._last_inserted_len = len(result.text)
        elif result.action == "clear_last":
            self.injector.clear_last(self._last_inserted_len)
            self._last_inserted_len = 0
        elif result.action == "newline":
            self.injector.inject("\n")
        elif result.action == "new_paragraph":
            self.injector.inject("\n\n")

    def run(self) -> None:
        logger.info(
            "flow_local ready — hold %s to dictate (100%% local, no network)",
            self.config.hotkey,
        )
        self.hotkey.start()
        try:
            while True:
                time.sleep(0.25)
        except KeyboardInterrupt:
            self.hotkey.stop()
