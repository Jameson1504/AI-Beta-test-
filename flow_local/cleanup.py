"""Turns a raw ASR transcript into "what you'd have typed": strips filler
words, fixes capitalization/punctuation, and recognizes a small set of
spoken editing commands (Wispr Flow's "Command Mode"). Pure text
processing — no model, no network — so it's fast and fully testable.

An optional local-LLM cleanup pass can be layered on top (see
`llm_cleanup.py`) for anyone who wants closer-to-Flow quality and is willing
to run a local model for it; it is off by default and this module works
standalone without it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import FlowConfig

_FILLER_WORDS = (
    "um", "umm", "uh", "uhh", "uhm", "erm", "hmm",
)
_FILLER_PATTERN = re.compile(
    r"\b(?:" + "|".join(_FILLER_WORDS) + r")\b[,]?", re.IGNORECASE
)
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_SENTENCE_END = re.compile(r"[.!?]$")

# Spoken commands recognized in Command Mode, checked against the full
# (lowercased, stripped) utterance so they don't fire on partial matches
# inside normal dictation.
_COMMANDS = {
    "scratch that": "clear_last",
    "delete that": "clear_last",
    "new line": "newline",
    "new paragraph": "new_paragraph",
}


@dataclass
class ProcessedResult:
    action: str  # "insert" | "clear_last" | "newline" | "new_paragraph"
    text: str = ""


def _strip_fillers(text: str) -> str:
    text = _FILLER_PATTERN.sub("", text)
    return _MULTI_SPACE.sub(" ", text).strip()


def _punctuate_capitalize(text: str) -> str:
    if not text:
        return text
    text = text[0].upper() + text[1:]
    text = re.sub(r"\bi\b", "I", text)
    if not _SENTENCE_END.search(text):
        text += "."
    return text


class TranscriptProcessor:
    def __init__(self, config: FlowConfig):
        self.config = config

    def process(self, raw_text: str) -> ProcessedResult:
        normalized = raw_text.strip().lower().strip(" .!?")
        if normalized in _COMMANDS:
            return ProcessedResult(action=_COMMANDS[normalized])

        text = raw_text.strip()
        if self.config.remove_filler_words:
            text = _strip_fillers(text)
        if self.config.auto_punctuate_capitalize:
            text = _punctuate_capitalize(text)
        return ProcessedResult(action="insert", text=text)
