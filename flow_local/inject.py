"""Writes transcribed text into whatever field has focus.

Wispr Flow does this through OS accessibility APIs with a clipboard
fallback. `flow_local` uses the portable equivalent available without extra
OS permissions: simulated keystrokes for most text, or clipboard-paste for
speed on long text. Both act only on the machine's own input queue/clipboard
— there is no accessibility-tree or window inspection, so no other app's
content is ever read.
"""

from __future__ import annotations

import time

from .config import FlowConfig


class TextInjector:
    def __init__(self, config: FlowConfig):
        self.config = config
        self._keyboard = None
        self._pyperclip = None

    def _keyboard_controller(self):
        if self._keyboard is None:
            from pynput.keyboard import Controller  # noqa: local import

            self._keyboard = Controller()
        return self._keyboard

    def inject(self, text: str) -> None:
        if not text:
            return
        if self.config.injection_mode == "paste":
            self._paste(text)
        else:
            self._type(text)

    def _type(self, text: str) -> None:
        keyboard = self._keyboard_controller()
        for char in text:
            keyboard.type(char)
            if self.config.type_delay_seconds:
                time.sleep(self.config.type_delay_seconds)

    def _paste(self, text: str) -> None:
        if self._pyperclip is None:
            import pyperclip  # noqa: local import

            self._pyperclip = pyperclip
        previous = self._pyperclip.paste()
        self._pyperclip.copy(text)
        keyboard = self._keyboard_controller()
        from pynput.keyboard import Key  # noqa: local import

        with keyboard.pressed(Key.ctrl):
            keyboard.press("v")
            keyboard.release("v")
        self._pyperclip.copy(previous)

    def clear_last(self, char_count: int) -> None:
        """Best-effort undo for "scratch that": backspace the last
        injected utterance."""
        keyboard = self._keyboard_controller()
        from pynput.keyboard import Key  # noqa: local import

        for _ in range(char_count):
            keyboard.press(Key.backspace)
            keyboard.release(Key.backspace)
