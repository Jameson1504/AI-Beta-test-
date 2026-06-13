"""Instruction-tuning (SFT) stage: turn a pretrained Mythos model into a
chat assistant by training on (instruction, response) pairs.

It loads a pretrained checkpoint (which must have used a BPE tokenizer with chat
special tokens), then continues training with the loss masked to the assistant
response. This is the same step that turns a base GPT into a chat model.

Usage:
    python -m mythos.finetune --init checkpoints --data data/instructions.jsonl \
        --steps 1000 --out checkpoints-sft
"""

from __future__ import annotations

import argparse
import math
import os
import time

import torch

from .chat import SFTData
from .config import ModelConfig
from .model import GPT
from .tokenizer import load_tokenizer


def cosine_lr(step, max_steps, lr, min_lr, warmup):
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (lr - min_lr)


@torch.no_grad()
def eval_loss(model, data, batch_size, iters=20):
    model.eval()
    losses = torch.zeros(iters)
    for k in range(iters):
        x, y = data.get_batch("train", batch_size)
        _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def main() -> None:
    p = argparse.ArgumentParser(description="Instruction-tune a Mythos model.")
    p.add_argument("--init", default="checkpoints", help="pretrained checkpoint dir")
    p.add_argument("--data", default="data/instructions.jsonl")
    p.add_argument("--out", default="checkpoints-sft")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.manual_seed(args.seed)

    # Load pretrained weights + the exact tokenizer it was trained with.
    ckpt = torch.load(os.path.join(args.init, "ckpt.pt"), map_location=device, weights_only=False)
    mcfg = ModelConfig(**ckpt["model_config"])
    model = GPT(mcfg).to(device)
    model.load_state_dict(ckpt["model"])
    tokenizer = load_tokenizer(os.path.join(args.init, "tokenizer.json"))
    if not hasattr(tokenizer, "token_id"):
        raise SystemExit(
            "Pretrained tokenizer has no chat special tokens. Pretrain with a BPE "
            "tokenizer (chat tokens are included by default) before fine-tuning.")

    data = SFTData(args.data, tokenizer, mcfg.block_size, device)
    print(f"[sft] {len(data)} examples | {model.num_params()/1e6:.2f}M params | device {device}")

    optimizer = model.configure_optimizer(args.weight_decay, args.lr, (0.9, 0.95),
                                           "cuda" if device.startswith("cuda") else "cpu")
    os.makedirs(args.out, exist_ok=True)
    tokenizer.save(os.path.join(args.out, "tokenizer.json"))

    t0 = time.time()
    model.train()
    for step in range(args.steps + 1):
        lr = cosine_lr(step, args.steps, args.lr, args.min_lr, args.warmup)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if step % 100 == 0 or step == args.steps:
            print(f"[step {step:>5}] sft loss {eval_loss(model, data, args.batch_size):.4f} "
                  f"| lr {lr:.2e} | {time.time()-t0:.1f}s")
            torch.save({"model": model.state_dict(), "model_config": mcfg.__dict__,
                        "step": step}, os.path.join(args.out, "ckpt.pt"))
        if step == args.steps:
            break

        x, y = data.get_batch("train", args.batch_size)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    print(f"[sft] done. checkpoint -> {args.out}/  (chat with: python -m mythos.chat --ckpt {args.out})")


if __name__ == "__main__":
    main()
