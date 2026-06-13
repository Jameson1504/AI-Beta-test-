"""Tokenized dataset + batching.

Loads a UTF-8 text file, tokenizes it once into a flat tensor of token ids,
splits into train/val, and serves random contiguous (x, y) windows where ``y``
is ``x`` shifted by one — the standard next-token-prediction setup.
"""

from __future__ import annotations

from typing import Tuple

import torch


class TextData:
    def __init__(self, path: str, tokenizer, block_size: int,
                 val_fraction: float = 0.1, device: str = "cpu") -> None:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        if not text:
            raise ValueError(f"data file {path!r} is empty")

        ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        if ids.numel() <= block_size + 1:
            raise ValueError(
                f"corpus has only {ids.numel()} tokens; need > block_size+1 "
                f"({block_size + 1}). Use a bigger file or smaller block_size."
            )
        n_val = max(block_size + 1, int(len(ids) * val_fraction))
        self.train_ids = ids[:-n_val]
        self.val_ids = ids[-n_val:]
        self.block_size = block_size
        self.device = device

    def get_batch(self, split: str, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.train_ids if split == "train" else self.val_ids
        max_start = len(data) - self.block_size - 1
        ix = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([data[i:i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + self.block_size] for i in ix])
        if self.device.startswith("cuda"):
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y
