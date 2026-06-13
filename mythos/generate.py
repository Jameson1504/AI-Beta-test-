"""Generate text from a trained Mythos checkpoint.

Usage:
    python -m mythos.generate --prompt "To be, or not to be" --max-new-tokens 200
"""

from __future__ import annotations

import argparse
import os

import torch

from .config import ModelConfig
from .model import GPT
from .tokenizer import load_tokenizer


def load_model(ckpt_dir: str, device: str):
    ckpt = torch.load(os.path.join(ckpt_dir, "ckpt.pt"), map_location=device, weights_only=False)
    mcfg = ModelConfig(**ckpt["model_config"])
    model = GPT(mcfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tokenizer = load_tokenizer(os.path.join(ckpt_dir, "tokenizer.json"))
    return model, tokenizer, ckpt


def main() -> None:
    p = argparse.ArgumentParser(description="Sample text from a Mythos model.")
    p.add_argument("--ckpt", default="checkpoints")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.manual_seed(args.seed)

    model, tokenizer, ckpt = load_model(args.ckpt, device)
    print(f"[mythos] loaded checkpoint (step {ckpt.get('step')}, "
          f"val_loss {ckpt.get('val_loss'):.4f})\n")

    ids = tokenizer.encode(args.prompt) or tokenizer.encode("\n")
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    for s in range(args.num_samples):
        out = model.generate(idx, args.max_new_tokens, args.temperature,
                             args.top_k, args.top_p)
        text = tokenizer.decode(out[0].tolist())
        print(f"--- sample {s + 1} ---\n{text}\n")


if __name__ == "__main__":
    main()
