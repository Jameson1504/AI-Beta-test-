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

After 500 CPU steps on the toy preset, training loss falls **5.49 → 1.87** and
the model emits Shakespeare-shaped text (character names, dialogue, line breaks).
Train longer / bigger for cleaner output.

## Presets

| Preset  | Params | Context | Where it runs        |
|---------|--------|---------|----------------------|
| `toy`   | ~1M    | 128     | CPU, minutes         |
| `small` | ~25M   | 256     | one GPU              |
| `base`  | ~120M  | 512     | one GPU + real data  |

Override anything on the CLI: `--steps`, `--batch-size`, `--lr`, `--block-size`,
`--device`, `--compile`.

## Project layout

```
mythos/
  config.py      # dataclass configs + presets
  tokenizer.py   # byte-level and char-level tokenizers (from scratch)
  model.py       # the GPT: RMSNorm, RoPE, attention, SwiGLU, KV-cache
  data.py        # tokenized dataset + batching
  train.py       # AdamW · cosine LR · eval · checkpoints
  generate.py    # autoregressive sampling
data/input.txt   # demo corpus (swap your own)
tests/test_model.py
scripts/demo.sh
```

## Tests

```bash
PYTHONPATH=. python tests/test_model.py     # or: python -m pytest -q
```

Includes the two best smoke tests for an LM: *generation-with-KV-cache matches a
full forward pass*, and *the model can overfit a single batch*.

## Scaling up (the roadmap, briefly)

1. **Bigger data + BPE tokenizer** → longer context, richer vocab.
2. **GPU**: `--preset base --compile`, bf16, gradient accumulation, multi-GPU
   (DDP/FSDP) → 100M–1B params.
3. **Post-training**: instruction-tune on (prompt, response) pairs so it follows
   instructions like a chat assistant; optional preference tuning (DPO).
4. **Eval & serve**: perplexity + task benchmarks, a small inference server,
   quantization.

Full detail in [`PLAN.md`](PLAN.md).

## License

MIT — do anything you like with it.
