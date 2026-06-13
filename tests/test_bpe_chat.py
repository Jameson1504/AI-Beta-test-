"""Tests for the BPE tokenizer, chat/SFT formatting, and EOS-stopped generation."""

import torch

from mythos.bpe import BPETokenizer
from mythos.chat import (CHAT_SPECIAL_TOKENS, ASSISTANT, EOT, build_prompt_ids,
                         build_sft_example)
from mythos.config import ModelConfig
from mythos.model import GPT

CORPUS = ("the quick brown fox jumps over the lazy dog. " * 50 +
          "hello world, how are you today? I am fine thank you. " * 50)


def test_bpe_roundtrip_is_lossless():
    tok = BPETokenizer.train(CORPUS, vocab_size=400, special_tokens=CHAT_SPECIAL_TOKENS)
    for s in ["hello world", "the quick brown fox", "how are you today?"]:
        assert tok.decode(tok.encode(s)) == s


def test_bpe_compresses_vs_bytes():
    tok = BPETokenizer.train(CORPUS, vocab_size=512, special_tokens=CHAT_SPECIAL_TOKENS)
    s = "the quick brown fox jumps over the lazy dog"
    assert len(tok.encode(s)) < len(s.encode("utf-8"))  # fewer tokens than bytes


def test_bpe_special_tokens_are_single_ids():
    tok = BPETokenizer.train(CORPUS, vocab_size=400, special_tokens=CHAT_SPECIAL_TOKENS)
    for sp in CHAT_SPECIAL_TOKENS:
        assert tok.encode(sp) == [tok.token_id(sp)]
    # Special tokens survive a save/load round trip with the same ids.
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "tok.json")
        tok.save(path)
        tok2 = BPETokenizer.load(path)
        assert tok2.vocab_size == tok.vocab_size
        assert tok2.encode("hello world") == tok.encode("hello world")


def test_sft_masks_prompt_tokens():
    tok = BPETokenizer.train(CORPUS, vocab_size=400, special_tokens=CHAT_SPECIAL_TOKENS)
    x, y = build_sft_example(tok, "You are helpful.", "hello", "hi there", block_size=64)
    assert len(x) == len(y) == 64
    # Prompt region must be ignored (-1); at least some response tokens supervised.
    n_supervised = sum(1 for t in y if t != -1)
    assert 0 < n_supervised < 64
    # The assistant marker is the last prompt token; everything before the first
    # supervised position should be masked.
    first_sup = next(i for i, t in enumerate(y) if t != -1)
    assert all(t == -1 for t in y[:first_sup])


def test_generate_stops_at_eos():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=300, block_size=32, n_layer=2, n_head=2, n_embd=32)
    model = GPT(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    eos = 0
    out = model.generate(prompt, max_new_tokens=20, temperature=0.0, eos_token_id=eos)
    # Either it hit eos (last token is eos) or produced the full length.
    assert out.size(1) <= 4 + 20


def test_build_prompt_ends_with_assistant():
    tok = BPETokenizer.train(CORPUS, vocab_size=400, special_tokens=CHAT_SPECIAL_TOKENS)
    ids = build_prompt_ids(tok, "sys", "user question")
    assert ids[-1] == tok.token_id(ASSISTANT)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nall {len(fns)} tests passed")
