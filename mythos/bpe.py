"""From-scratch byte-level Byte-Pair Encoding (BPE) tokenizer.

This is the GPT-2/GPT-style tokenizer built from scratch: it learns a vocabulary
of merges over UTF-8 bytes, so each token covers ~3-4 characters instead of one.
That means a given context window holds ~4x more text than the byte tokenizer.

It also supports *special tokens* (e.g. ``<|user|>``, ``<|assistant|>``) which
are needed for chat / instruction formatting and are never split.

Train one with:
    python -m mythos.bpe --data data/input.txt --vocab-size 4096 --out tok.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from typing import Dict, List, Tuple

# GPT-2-style pre-tokenization. Python's ``re`` lacks \p{L}; \w (+UNICODE) is a
# close, dependency-free approximation that keeps contractions/spaces sensible.
SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+"""


def _merge(symbols: Tuple[int, ...], pair: Tuple[int, int], idx: int) -> Tuple[int, ...]:
    out: List[int] = []
    i = 0
    n = len(symbols)
    while i < n:
        if i < n - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            out.append(idx)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


class BPETokenizer:
    kind = "bpe"

    def __init__(self, merges: Dict[Tuple[int, int], int],
                 special_tokens: Dict[str, int], pattern: str = SPLIT_PATTERN) -> None:
        self.merges = merges
        self.special_tokens = dict(special_tokens)
        self.pattern = re.compile(pattern)
        self._pattern_str = pattern
        self.vocab = self._build_vocab()
        self._inv_special = {v: k for k, v in self.special_tokens.items()}

    # -- construction -------------------------------------------------------- #
    def _build_vocab(self) -> Dict[int, bytes]:
        vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in sorted(self.merges.items(), key=lambda kv: kv[1]):
            vocab[idx] = vocab[a] + vocab[b]
        return vocab

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special_tokens)

    @classmethod
    def train(cls, text: str, vocab_size: int,
              special_tokens: List[str] | None = None,
              pattern: str = SPLIT_PATTERN, verbose: bool = False) -> "BPETokenizer":
        special_tokens = special_tokens or []
        assert vocab_size >= 256 + len(special_tokens), "vocab_size too small"
        num_merges = vocab_size - 256 - len(special_tokens)

        # Pre-tokenize into words and count frequencies; BPE then operates on the
        # unique words weighted by count (far faster than per-character scans).
        words = re.compile(pattern).findall(text)
        word_freq = Counter(words)
        splits: Dict[str, Tuple[int, ...]] = {
            w: tuple(w.encode("utf-8")) for w in word_freq
        }

        merges: Dict[Tuple[int, int], int] = {}
        for i in range(num_merges):
            pair_counts: Counter = Counter()
            for w, freq in word_freq.items():
                sym = splits[w]
                for a, b in zip(sym, sym[1:]):
                    pair_counts[(a, b)] += freq
            if not pair_counts:
                break
            best = max(pair_counts, key=pair_counts.get)
            idx = 256 + i
            merges[best] = idx
            splits = {w: _merge(sym, best, idx) for w, sym in splits.items()}
            if verbose and (i % 200 == 0):
                print(f"  merge {i + 1}/{num_merges}: {best} -> {idx}")

        base = 256 + len(merges)
        specials = {tok: base + j for j, tok in enumerate(special_tokens)}
        return cls(merges, specials, pattern)

    # -- encode / decode ----------------------------------------------------- #
    def _encode_word(self, word: bytes) -> List[int]:
        symbols = tuple(word)
        while len(symbols) >= 2:
            pairs = set(zip(symbols, symbols[1:]))
            candidate = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if candidate not in self.merges:
                break
            symbols = _merge(symbols, candidate, self.merges[candidate])
        return list(symbols)

    def _encode_ordinary(self, text: str) -> List[int]:
        ids: List[int] = []
        for word in self.pattern.findall(text):
            ids.extend(self._encode_word(word.encode("utf-8")))
        return ids

    def encode(self, text: str, allowed_special: str = "all") -> List[int]:
        if not self.special_tokens or allowed_special == "none":
            return self._encode_ordinary(text)
        specials = self.special_tokens
        splitter = "(" + "|".join(re.escape(s) for s in specials) + ")"
        ids: List[int] = []
        for chunk in re.split(splitter, text):
            if chunk in specials:
                ids.append(specials[chunk])
            elif chunk:
                ids.extend(self._encode_ordinary(chunk))
        return ids

    def decode(self, ids: List[int]) -> str:
        parts: List[bytes] = []
        for i in ids:
            i = int(i)
            if i in self.vocab:
                parts.append(self.vocab[i])
            elif i in self._inv_special:
                parts.append(self._inv_special[i].encode("utf-8"))
        return b"".join(parts).decode("utf-8", errors="replace")

    def token_id(self, special: str) -> int:
        return self.special_tokens[special]

    # -- persistence --------------------------------------------------------- #
    def save(self, path: str) -> None:
        data = {
            "kind": self.kind,
            "pattern": self._pattern_str,
            "merges": [[a, b, idx] for (a, b), idx in
                       sorted(self.merges.items(), key=lambda kv: kv[1])],
            "special_tokens": self.special_tokens,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path) as f:
            data = json.load(f)
        merges = {(a, b): idx for a, b, idx in data["merges"]}
        return cls(merges, data.get("special_tokens", {}),
                   data.get("pattern", SPLIT_PATTERN))


def main() -> None:
    p = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")
    p.add_argument("--data", required=True)
    p.add_argument("--vocab-size", type=int, default=4096)
    p.add_argument("--out", default="tokenizer.json")
    p.add_argument("--no-chat-tokens", action="store_true",
                   help="omit chat special tokens (<|user|> etc.)")
    args = p.parse_args()

    from .chat import CHAT_SPECIAL_TOKENS
    specials = [] if args.no_chat_tokens else CHAT_SPECIAL_TOKENS
    with open(args.data, encoding="utf-8") as f:
        text = f.read()
    print(f"[bpe] training vocab_size={args.vocab_size} on {len(text)} chars ...")
    tok = BPETokenizer.train(text, args.vocab_size, specials, verbose=True)
    tok.save(args.out)
    sample = "Hello, world! How are you?"
    print(f"[bpe] done. vocab_size={tok.vocab_size}. saved -> {args.out}")
    print(f"[bpe] sample encode {sample!r} -> {len(tok.encode(sample))} tokens "
          f"(vs {len(sample.encode('utf-8'))} bytes)")


if __name__ == "__main__":
    main()
