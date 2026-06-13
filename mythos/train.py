"""Training loop for Mythos.

Usage:
    python -m mythos.train --preset toy --data data/input.txt --steps 2000

Features: AdamW, cosine LR schedule with warmup, gradient clipping, gradient
accumulation, periodic eval, automatic mixed precision on GPU, and checkpointing
(model + tokenizer + config) so generation reuses an identical setup.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from contextlib import nullcontext

import torch

from .config import build_configs
from .data import TextData
from .model import GPT
from .tokenizer import build_tokenizer


def pick_device(choice: str) -> str:
    if choice != "auto":
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(choice: str, device: str):
    if choice == "auto":
        if device == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32
    return {"float32": torch.float32, "bfloat16": torch.bfloat16,
            "float16": torch.float16}[choice]


def lr_at(step: int, tcfg) -> float:
    if step < tcfg.warmup_steps:
        return tcfg.learning_rate * (step + 1) / max(1, tcfg.warmup_steps)
    if step >= tcfg.max_steps:
        return tcfg.min_lr
    progress = (step - tcfg.warmup_steps) / max(1, tcfg.max_steps - tcfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return tcfg.min_lr + coeff * (tcfg.learning_rate - tcfg.min_lr)


@torch.no_grad()
def evaluate(model, data, tcfg) -> dict:
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(tcfg.eval_iters)
        for k in range(tcfg.eval_iters):
            x, y = data.get_batch(split, tcfg.batch_size)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Train a Mythos GPT from scratch.")
    p.add_argument("--preset", default="toy", help="toy | small | base")
    p.add_argument("--data", dest="data_path", default=None)
    p.add_argument("--tokenizer", default=None, choices=[None, "byte", "char"])
    p.add_argument("--steps", dest="max_steps", type=int, default=None)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    p.add_argument("--lr", dest="learning_rate", type=float, default=None)
    p.add_argument("--block-size", dest="block_size", type=int, default=None)
    p.add_argument("--out", dest="out_dir", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    mcfg, tcfg = build_configs(
        args.preset,
        data_path=args.data_path, tokenizer=args.tokenizer, max_steps=args.max_steps,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        block_size=args.block_size, out_dir=args.out_dir, device=args.device,
        seed=args.seed,
    )
    if args.compile:
        tcfg.compile = True

    torch.manual_seed(tcfg.seed)
    device = pick_device(tcfg.device)
    dtype = pick_dtype(tcfg.dtype, device)
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    print(f"[mythos] device={device} dtype={str(dtype).split('.')[-1]} preset={args.preset}")

    # Tokenizer + data. The tokenizer sets the model's vocab size.
    with open(tcfg.data_path, "r", encoding="utf-8") as f:
        corpus = f.read()
    tokenizer = build_tokenizer(tcfg.tokenizer, corpus)
    mcfg.vocab_size = tokenizer.vocab_size
    data = TextData(tcfg.data_path, tokenizer, mcfg.block_size, tcfg.val_fraction, device)

    model = GPT(mcfg).to(device)
    print(f"[mythos] parameters: {model.num_params()/1e6:.2f}M "
          f"(vocab={mcfg.vocab_size}, ctx={mcfg.block_size})")
    if tcfg.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = model.configure_optimizer(
        tcfg.weight_decay, tcfg.learning_rate, (tcfg.beta1, tcfg.beta2), device_type)

    amp = (device_type == "cuda" and dtype in (torch.bfloat16, torch.float16))
    ctx = torch.autocast(device_type, dtype=dtype) if amp else nullcontext()
    scaler = torch.amp.GradScaler(device_type, enabled=(amp and dtype == torch.float16))

    os.makedirs(tcfg.out_dir, exist_ok=True)
    tokenizer.save(os.path.join(tcfg.out_dir, "tokenizer.json"))

    best_val = float("inf")
    t0 = time.time()
    model.train()
    for step in range(tcfg.max_steps + 1):
        lr = lr_at(step, tcfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if step % tcfg.eval_interval == 0 or step == tcfg.max_steps:
            m = evaluate(model, data, tcfg)
            dt = time.time() - t0
            print(f"[step {step:>6}] train {m['train']:.4f} | val {m['val']:.4f} "
                  f"| lr {lr:.2e} | {dt:.1f}s")
            if m["val"] < best_val:
                best_val = m["val"]
                save_checkpoint(model, mcfg, tcfg, step, best_val)

        if step == tcfg.max_steps:
            break

        optimizer.zero_grad(set_to_none=True)
        for micro in range(tcfg.grad_accum_steps):
            x, y = data.get_batch("train", tcfg.batch_size)
            with ctx:
                _, loss = model(x, y)
                loss = loss / tcfg.grad_accum_steps
            scaler.scale(loss).backward()
        if tcfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % tcfg.log_interval == 0 and step > 0:
            print(f"  step {step}: loss {loss.item()*tcfg.grad_accum_steps:.4f}")

    print(f"[mythos] done. best val loss {best_val:.4f}. "
          f"checkpoint in {tcfg.out_dir}/")


def save_checkpoint(model, mcfg, tcfg, step, val_loss) -> None:
    raw = getattr(model, "_orig_mod", model)  # unwrap torch.compile
    torch.save({
        "model": raw.state_dict(),
        "model_config": mcfg.__dict__,
        "step": step,
        "val_loss": val_loss,
    }, os.path.join(tcfg.out_dir, "ckpt.pt"))


if __name__ == "__main__":
    main()
