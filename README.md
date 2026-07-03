# Mythos 🜂

A **from-scratch, modern GPT language model** in clean PyTorch — the same
architectural ingredients as today's frontier models, in a few hundred readable
lines you fully own and can scale up.

```
token embedding → N×[ RMSNorm → RoPE causal attention → RMSNorm → SwiGLU ] → RMSNorm → tied head
```

## A straight answer about "better than Fable 5"

This project began as "build the world's best AI, better than Fable 5 Mythos."
Here's the honest version of that goal:

- **You can't out-train a frontier model in a coding session.** The transformer
  architecture is public and small — the moat is *scale*: trillions of tokens,
  thousands of GPUs for months ($10M–$100M+), big teams, and months of
  post-training. No amount of clever code closes that gap alone.
- **What you _can_ own is the real thing at small scale.** Mythos is a genuine,
  correct, modern LM. It learns from data and generates text, and the *exact
  same code* scales from a 1M-param CPU toy to a 1B+ GPU model — you change the
  config and the hardware, not the architecture.

So Mythos is the best *honest* version of the request: a real foundation you
understand completely, built to grow. See [`PLAN.md`](PLAN.md) for the full
architecture rationale and the roadmap from toy → GPU-scale → instruction-tuned.

## What's modern about it

| Component     | Mythos uses        | vs. GPT-2 (2019)      |
|---------------|--------------------|-----------------------|
| Norm          | RMSNorm (pre-norm) | LayerNorm             |
| Positions     | RoPE (rotary)      | learned absolute      |
| MLP           | SwiGLU             | GELU MLP              |
| Attention     | causal MHA, **GQA-ready**, **KV-cache** | MHA only |
| Head          | weight-tied        | weight-tied           |
| Training      | AdamW · cosine LR · warmup · grad-clip · AMP/bf16 · grad-accum | — |
| Sampling      | temperature · top-k · top-p · KV-cache | — |

## Quickstart

```bash
pip install -r requirements.txt

# Train a ~1M-param model from scratch (CPU-friendly, a few minutes)
python -m mythos.train --preset toy --data data/input.txt --steps 2000

# Generate from your trained model
python -m mythos.generate --ckpt checkpoints --prompt "ROMEO:" \
    --max-new-tokens 300 --temperature 0.8 --top-k 50

# Or do both at once
bash scripts/demo.sh
```

A sample corpus (`data/input.txt`, ~1MB of Shakespeare) is included. Swap in any
UTF-8 text file to train on your own data.

### Verified working

After 500 CPU steps on the toy preset
(`python -m mythos.train --preset toy --data data/input.txt --steps 500`),
training loss falls **5.49 → 1.87** and the model emits Shakespeare-shaped text
(character names, dialogue, line breaks). The default `--steps 2000` (what
`scripts/demo.sh` runs) trains longer for cleaner output.

## Presets

| Preset  | Params | Context | Where it runs        |
|---------|--------|---------|----------------------|
| `toy`   | ~1M    | 128     | CPU, minutes         |
| `small` | ~25M   | 256     | one GPU              |
| `base`  | ~120M  | 512     | one GPU + real data  |

Override anything on the CLI: `--steps`, `--batch-size`, `--lr`, `--block-size`,
`--device`, `--compile`.

## Advanced: BPE tokenizer, multi-GPU, and chat fine-tuning

### Byte-level BPE tokenizer (from scratch)

Each BPE token covers ~3-4 characters, so a context window holds ~4x more text
than the byte tokenizer. It also carries chat special tokens (`<|user|>`, …).

```bash
# Train a tokenizer once...
python -m mythos.bpe --data data/input.txt --vocab-size 4096 --out tok.json
# ...then pretrain with it (or just pass --tokenizer bpe to train inline)
python -m mythos.train --preset small --tokenizer bpe --vocab-size 4096 \
    --data data/input.txt
```

### Multi-GPU training (DDP)

One process per GPU via `torchrun`. Logging/checkpointing happen on rank 0;
gradients sync once per optimizer step (and only on the last micro-step when
accumulating). Backend is chosen automatically (`nccl` on GPU, `gloo` on CPU).

```bash
torchrun --standalone --nproc_per_node=4 -m mythos.train \
    --preset base --data data/input.txt --compile
```

### Instruction-tuning → a chat assistant

Turn a pretrained base model into a chat model by training on
(instruction, response) pairs, with the loss masked to the assistant response.
Requires a BPE tokenizer with chat tokens (the default).

```bash
# 1) pretrain a base model with the BPE tokenizer
python -m mythos.train --preset small --tokenizer bpe --data data/input.txt --out checkpoints
# 2) instruction-tune it
python -m mythos.finetune --init checkpoints --data data/instructions.jsonl --out checkpoints-sft
# 3) chat with it
python -m mythos.chat --ckpt checkpoints-sft
```

The chat template is `<|system|>…<|user|>…<|assistant|>…<|endoftext|>`, and
generation stops at `<|endoftext|>`. Bring your own JSONL of
`{"instruction", "input"?, "output"}` (Alpaca-style) or `{"system"?, "user", "response"}`.

> Note: quality tracks scale. The included toy data + a 1M-param model just
> proves the *pipeline* end-to-end; real answers need a bigger model, more data,
> and more steps.

## Project layout

```
mythos/
  config.py      # dataclass configs + presets
  tokenizer.py   # byte- and char-level tokenizers (from scratch)
  bpe.py         # byte-level BPE tokenizer + special tokens (from scratch)
  model.py       # the GPT: RMSNorm, RoPE, attention, SwiGLU, KV-cache
  data.py        # tokenized dataset + batching
  train.py       # AdamW · cosine LR · eval · checkpoints · multi-GPU (DDP)
  finetune.py    # instruction-tuning (SFT) stage
  chat.py        # chat template, SFT data, interactive chat CLI
  generate.py    # autoregressive sampling
data/input.txt          # demo pretraining corpus (swap your own)
data/instructions.jsonl # demo instruction dataset for SFT
tests/                  # model + tokenizer/chat tests
scripts/demo.sh
```

## Tests

```bash
pip install -r requirements-dev.txt   # installs pytest
python -m pytest -q                   # runs tests/test_model.py + tests/test_bpe_chat.py
```

Includes the two best smoke tests for an LM: *generation-with-KV-cache matches a
full forward pass*, and *the model can overfit a single batch*.

## Scaling up (the roadmap)

1. ✅ **BPE tokenizer** → ~4x more text per context window, richer vocab.
2. ✅ **Multi-GPU (DDP)** → data-parallel training across GPUs with `torchrun`.
3. ✅ **Instruction-tuning** → masked-loss SFT + a chat CLI.
4. ⏭ **More scale**: `--preset base --compile`, bf16, gradient accumulation, a
   real multi-GB dataset → 100M–1B params on a GPU box.
5. ⏭ **Preference tuning (DPO)** and an eval harness (perplexity + tasks).
6. ⏭ **Serving**: a small inference server + quantization for cheap deployment.

Full detail in [`PLAN.md`](PLAN.md).

## License

MIT — do anything you like with it.
