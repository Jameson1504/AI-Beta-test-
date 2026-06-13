"""From-scratch tokenizers.

Two simple, dependency-free tokenizers:

* ``ByteTokenizer`` — maps text to its raw UTF-8 bytes (vocab = 256). Lossless
  for *any* input, never produces unknown tokens. A great default.
* ``CharTokenizer`` — builds a vocab from the unique characters in a corpus.
  Smaller vocab on small/clean data, but can't encode unseen characters.

Both expose the same ``encode`` / ``decode`` interface and can ``save``/``load``
their state so training and generation use an identical mapping.
"""

from __future__ import annotations

import json
from typing import List


class ByteTokenizer:
    """UTF-8 byte-level tokenizer. Fixed vocab of 256, encodes anything."""

    kind = "byte"

    def __init__(self) -> None:
        self.vocab_size = 256

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: List[int]) -> str:
        return bytes(int(i) & 0xFF for i in ids).decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"kind": self.kind}, f)

    @classmethod
    def load(cls, path: str) -> "ByteTokenizer":
        return cls()


class CharTokenizer:
    """Character-level tokenizer with a vocab learned from a corpus."""

    kind = "char"

    def __init__(self, chars: List[str]) -> None:
        self.itos = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.vocab_size = len(self.itos)

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    def encode(self, text: str) -> List[int]:
        # Unknown chars are skipped to stay robust; warn the caller via no token.
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids if 0 <= int(i) < self.vocab_size)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"kind": self.kind, "itos": self.itos}, f)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path) as f:
            data = json.load(f)
        return cls(data["itos"])


def build_tokenizer(kind: str, text: str = "", vocab_size: int = 4096):
    """Factory: ``kind`` is ``"byte"``, ``"char"`` or ``"bpe"``.

    For ``"bpe"`` a tokenizer is *trained* on ``text`` up to ``vocab_size`` and
    includes the chat special tokens by default.
    """
    if kind == "byte":
        return ByteTokenizer()
    if kind == "char":
        return CharTokenizer.from_text(text)
    if kind == "bpe":
        from .bpe import BPETokenizer
        from .chat import CHAT_SPECIAL_TOKENS
        return BPETokenizer.train(text, vocab_size, CHAT_SPECIAL_TOKENS)
    raise ValueError(f"unknown tokenizer kind {kind!r}")


def load_tokenizer(path: str):
    """Load whichever tokenizer was saved at ``path``."""
    with open(path) as f:
        kind = json.load(f).get("kind", "byte")
    if kind == "char":
        return CharTokenizer.load(path)
    if kind == "bpe":
        from .bpe import BPETokenizer
        return BPETokenizer.load(path)
    return ByteTokenizer.load(path)
