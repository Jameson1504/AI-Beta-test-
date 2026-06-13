"""Chat formatting, supervised-finetuning (SFT) examples, and an interactive CLI.

The chat template wraps turns in special tokens so the model learns role
structure:

    <|system|>{system}<|user|>{user}<|assistant|>{response}<|endoftext|>

During SFT, the loss is masked to the **assistant response only** (plus its
closing <|endoftext|>): the model is graded on what it should *say*, not on
re-predicting the prompt.

Run a trained chat model with:
    python -m mythos.chat --ckpt checkpoints-sft
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

import torch

# Special tokens included when training a BPE tokenizer for chat.
EOT = "<|endoftext|>"
SYSTEM = "<|system|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
CHAT_SPECIAL_TOKENS = [EOT, SYSTEM, USER, ASSISTANT]

DEFAULT_SYSTEM = "You are Mythos, a helpful, concise assistant."


def build_prompt_ids(tokenizer, system: str, user: str) -> List[int]:
    """Token ids for a prompt ending right before the assistant should speak."""
    ids: List[int] = []
    ids.append(tokenizer.token_id(SYSTEM))
    ids += tokenizer.encode(system, allowed_special="none")
    ids.append(tokenizer.token_id(USER))
    ids += tokenizer.encode(user, allowed_special="none")
    ids.append(tokenizer.token_id(ASSISTANT))
    return ids


def build_sft_example(tokenizer, system: str, user: str, response: str,
                      block_size: int) -> Tuple[List[int], List[int]]:
    """Return (input_ids, target_ids) with targets masked (-1) outside the
    assistant response. Sequence is truncated/padded to ``block_size + 1``."""
    prompt = build_prompt_ids(tokenizer, system, user)
    resp = tokenizer.encode(response, allowed_special="none") + [tokenizer.token_id(EOT)]

    ids = prompt + resp
    supervised = [False] * len(prompt) + [True] * len(resp)

    ids = ids[: block_size + 1]
    supervised = supervised[: block_size + 1]

    pad_id = tokenizer.token_id(EOT)
    while len(ids) < block_size + 1:
        ids.append(pad_id)
        supervised.append(False)

    x = ids[:-1]
    y = [tok if sup else -1 for tok, sup in zip(ids[1:], supervised[1:])]
    return x, y


def load_jsonl(path: str) -> List[dict]:
    import json
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class SFTData:
    """Builds masked (x, y) batches from an instruction JSONL file.

    Each row may use ``{"instruction", "input"?, "output"}`` (Alpaca-style) or
    ``{"system"?, "user", "response"}``.
    """

    def __init__(self, path: str, tokenizer, block_size: int, device: str = "cpu") -> None:
        rows = load_jsonl(path)
        self.examples: List[Tuple[List[int], List[int]]] = []
        for r in rows:
            system = r.get("system", DEFAULT_SYSTEM)
            user = r.get("user") or r.get("instruction", "")
            if r.get("input"):
                user = f"{user}\n\n{r['input']}"
            response = r.get("response") or r.get("output", "")
            if not user or not response:
                continue
            self.examples.append(build_sft_example(tokenizer, system, user, response, block_size))
        if not self.examples:
            raise ValueError(f"no usable examples in {path}")
        self.device = device

    def __len__(self) -> int:
        return len(self.examples)

    def get_batch(self, split: str, batch_size: int):
        ix = torch.randint(0, len(self.examples), (batch_size,))
        x = torch.tensor([self.examples[i][0] for i in ix], dtype=torch.long)
        y = torch.tensor([self.examples[i][1] for i in ix], dtype=torch.long)
        return x.to(self.device), y.to(self.device)


def chat_cli() -> None:
    from .generate import load_model

    p = argparse.ArgumentParser(description="Chat with a fine-tuned Mythos model.")
    p.add_argument("--ckpt", default="checkpoints-sft")
    p.add_argument("--system", default=DEFAULT_SYSTEM)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    model, tokenizer, ckpt = load_model(args.ckpt, device)
    if not hasattr(tokenizer, "token_id"):
        raise SystemExit("This checkpoint's tokenizer has no chat special tokens. "
                         "Train a BPE tokenizer with chat tokens and re-run SFT.")
    eot = tokenizer.token_id(EOT)
    print(f"[mythos] chat ready (step {ckpt.get('step')}). Ctrl-C to exit.\n")

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            break
        if not user:
            continue
        ids = build_prompt_ids(tokenizer, args.system, user)
        idx = torch.tensor([ids[-model.cfg.block_size:]], dtype=torch.long, device=device)
        out = model.generate(idx, args.max_new_tokens, args.temperature,
                             args.top_k, args.top_p, eos_token_id=eot)
        reply = tokenizer.decode(out[0, len(ids):].tolist()).split(EOT)[0].strip()
        print(f"mythos> {reply}\n")


if __name__ == "__main__":
    chat_cli()
