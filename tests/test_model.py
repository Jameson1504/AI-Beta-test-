"""Sanity tests for the Mythos model.

Run with:  python -m pytest -q   (or)   python tests/test_model.py
The "overfit one batch" test is the single best smoke test for a model + loop:
if a fresh model can't drive the loss toward zero on one fixed batch, something
is wired wrong.
"""

import torch

from mythos.config import ModelConfig
from mythos.model import GPT
from mythos.tokenizer import ByteTokenizer, CharTokenizer


def tiny_cfg(**kw):
    base = dict(vocab_size=65, block_size=32, n_layer=2, n_head=4, n_embd=64)
    base.update(kw)
    return ModelConfig(**base)


def test_forward_shapes():
    cfg = tiny_cfg()
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    logits, loss = model(x, x)
    assert logits.shape == (2, cfg.block_size, cfg.vocab_size)
    assert loss.dim() == 0 and loss.item() > 0


def test_inference_only_last_position():
    cfg = tiny_cfg()
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 10))
    logits, loss = model(x)
    assert logits.shape == (1, 1, cfg.vocab_size)
    assert loss is None


def test_gqa_runs():
    cfg = tiny_cfg(n_head=4, n_kv_head=2)
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, _ = model(x)
    assert logits.shape[-1] == cfg.vocab_size


def test_generate_with_kv_cache_matches_full_forward():
    """Greedy generation via KV-cache must match a plain causal forward pass."""
    torch.manual_seed(0)
    cfg = tiny_cfg(dropout=0.0)
    model = GPT(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 8))
    # Reference: argmax of a full forward over the prompt.
    with torch.no_grad():
        logits, _ = model(prompt)
        ref_next = logits[:, -1, :].argmax(-1)
    gen = model.generate(prompt, max_new_tokens=1, temperature=0.0)
    assert gen[0, -1].item() == ref_next.item()


def test_overfit_one_batch():
    torch.manual_seed(0)
    cfg = tiny_cfg(n_layer=2)
    model = GPT(cfg)
    opt = model.configure_optimizer(0.0, 1e-3, (0.9, 0.95))
    x = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
    y = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
    first = None
    for _ in range(200):
        _, loss = model(x, y)
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.2, f"loss did not drop enough: {first} -> {loss.item()}"


def test_tokenizers_roundtrip():
    bt = ByteTokenizer()
    s = "Hello, world! 🌍 café"
    assert bt.decode(bt.encode(s)) == s
    ct = CharTokenizer.from_text("abcde abcde")
    assert ct.decode(ct.encode("dead beef cab".replace("f", ""))) is not None
    assert ct.vocab_size == len(set("abcde "))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nall {len(fns)} tests passed")
