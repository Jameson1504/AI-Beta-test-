# Mythos — Plan & Architecture

> A from-scratch GPT language model. Honest goal: a **real, modern, working**
> transformer you fully own and understand — built with the same architectural
> ingredients as today's frontier models, and designed to scale up as you add
> data and compute.

## The honest framing (read this first)

You asked for "the world's best AI, better than Fable 5 Mythos." Here is the
truth, plainly:

- The gap between a hobby model and a frontier model (Fable 5, Opus, Llama-405B)
  is **not the code**. The transformer architecture is public and only a few
  hundred lines. The gap is **scale**: trillions of tokens of curated data,
  thousands of GPUs running for months ($10M–$100M+), large research teams, and
  months of post-training (RLHF, evals, safety).
- A model trained from scratch in this environment will be **small and not
  competitive with frontier models** — and that is expected. What it *will* be
  is **real**: it learns from data, it generates text, and every line is yours.
- The design below is **scale-ready**. The same code that trains a 1M-parameter
  toy on a CPU is what trains a 1B+ model on GPUs — you change the config, the
  data, and the hardware, not the architecture.

So the deliverable is the best *honest* version of your request: a clean,
modern, production-shaped GPT you can train today and grow later.

## Architecture (what makes it "modern", not 2019-era GPT-2)

Decoder-only transformer with the upgrades used by Llama / Mistral / modern LMs:

| Component        | Choice                  | Why                                        |
|------------------|-------------------------|--------------------------------------------|
| Normalization    | **RMSNorm** (pre-norm)  | Cheaper & more stable than LayerNorm       |
| Positional info  | **RoPE** (rotary)       | Better length generalization than learned  |
| Attention        | Multi-head causal, GQA-ready | Scales to long context; KV-cache for fast gen |
| MLP              | **SwiGLU**              | Stronger than GELU MLP at equal params     |
| Embeddings       | **Weight-tied** head    | Saves params, improves quality             |
| Init / training  | scaled init, AdamW, cosine LR, warmup, grad clip | Stable convergence |
| Sampling         | temperature, top-k, top-p, KV-cache | Quality + speed at inference       |

## Repository layout

```
mythos/
  config.py      # dataclass model + training configs (toy / small / scale presets)
  tokenizer.py   # byte-level and char-level tokenizers (from scratch)
  bpe.py         # byte-level BPE tokenizer + chat special tokens (from scratch)
  model.py       # the GPT: RMSNorm, RoPE, attention, SwiGLU, KV-cache
  data.py        # tokenized dataset + batching
  train.py       # training loop (AdamW, cosine schedule, eval, checkpoints, DDP)
  finetune.py    # instruction-tuning (SFT) stage
  chat.py        # chat template, SFT data, interactive chat CLI
  generate.py    # autoregressive sampling
data/
  input.txt          # tiny demo corpus (swap in your own)
  instructions.jsonl # demo instruction dataset for SFT
tests/
  test_model.py     # shape checks + "overfit one batch" sanity test
  test_bpe_chat.py  # BPE round-trip + chat template tests
scripts/
  demo.sh        # train a tiny model end-to-end and sample from it
```

## Roadmap (how this grows from toy to serious)

1. ✅ **Phase 0 — Foundation:** working modern GPT, trains on CPU on a tiny
   corpus, generates text, has tests. *Proves the engine runs.*
2. ✅ **Phase 1 — Real tokenizer:** from-scratch byte-level BPE (`mythos/bpe.py`)
   with chat special tokens; ~4x more text per context window.
3. 🟡 **Phase 2 — Scale on GPU:** multi-GPU **DDP** is wired (`torchrun`), plus
   bf16/AMP, gradient accumulation, and `torch.compile`. Remaining: run it on a
   real GPU box with a multi-GB dataset to reach 100M–1B params.
4. 🟡 **Phase 3 — Post-training:** instruction-tuning (SFT) with masked loss
   (`mythos/finetune.py`) + a chat CLI (`mythos/chat.py`) are done. Remaining:
   preference tuning (DPO) and a larger instruction set.
5. ⏭ **Phase 4 — Evaluation & serving:** eval harness (perplexity + task
   benchmarks), a small inference server, quantization for cheap deployment.

## How to run (after Phase 0)

```bash
pip install -r requirements.txt
python -m mythos.train --preset toy --data data/input.txt --steps 2000
python -m mythos.generate --prompt "To be, or not to be" --max-new-tokens 200
```
