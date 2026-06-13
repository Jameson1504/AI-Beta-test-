"""Configuration dataclasses and presets for Mythos.

A *preset* bundles a model size with sensible training defaults. The same code
path trains every preset — only these numbers (and your hardware/data) change as
you scale from a CPU toy to a GPU-trained model.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ModelConfig:
    """Architecture hyperparameters for the GPT."""

    vocab_size: int = 256          # set by the tokenizer (256 for byte-level)
    block_size: int = 256          # max context length (tokens)
    n_layer: int = 6               # number of transformer blocks
    n_head: int = 6                # number of attention query heads
    n_kv_head: Optional[int] = None  # GQA: KV heads (defaults to n_head = MHA)
    n_embd: int = 384              # model / residual stream width
    mlp_ratio: float = 4.0         # hidden width of the SwiGLU MLP (per gate)
    dropout: float = 0.0           # dropout prob (0 is best for small data + many epochs)
    rope_theta: float = 10000.0    # RoPE base frequency
    norm_eps: float = 1e-5         # RMSNorm epsilon
    tie_weights: bool = True       # share token-embedding and output projection

    def __post_init__(self) -> None:
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


@dataclass
class TrainConfig:
    """Optimization & loop hyperparameters."""

    # data / io
    data_path: str = "data/input.txt"
    out_dir: str = "checkpoints"
    tokenizer: str = "byte"        # "byte" or "char"

    # optimization
    batch_size: int = 32
    grad_accum_steps: int = 1
    max_steps: int = 2000
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # schedule / logging
    eval_interval: int = 250
    eval_iters: int = 50
    log_interval: int = 25
    val_fraction: float = 0.1

    # system
    device: str = "auto"          # "auto" | "cpu" | "cuda" | "mps"
    dtype: str = "auto"           # "auto" | "float32" | "bfloat16" | "float16"
    compile: bool = False         # torch.compile (great on GPU, skip on CPU)
    seed: int = 1337


def _model(**kw) -> ModelConfig:
    return ModelConfig(**kw)


# Presets: pick with --preset on the CLI. Sizes are approximate param counts.
PRESETS: dict[str, dict] = {
    # ~1M params — trains in minutes on a CPU, proves the engine works.
    "toy": {
        "model": _model(block_size=128, n_layer=4, n_head=4, n_embd=128),
        "train": {"batch_size": 32, "max_steps": 2000, "learning_rate": 3e-4},
    },
    # ~25M params — GPT-2-ish "small-small"; comfortable on a single GPU.
    "small": {
        "model": _model(block_size=256, n_layer=8, n_head=8, n_embd=512),
        "train": {"batch_size": 64, "max_steps": 50_000, "learning_rate": 3e-4},
    },
    # ~120M params — GPT-2 scale. Needs a GPU and a real dataset.
    "base": {
        "model": _model(block_size=512, n_layer=12, n_head=12, n_embd=768),
        "train": {"batch_size": 32, "grad_accum_steps": 8, "max_steps": 200_000,
                   "learning_rate": 6e-4, "compile": True, "dtype": "bfloat16"},
    },
}


def build_configs(preset: str, **overrides) -> tuple[ModelConfig, TrainConfig]:
    """Return (ModelConfig, TrainConfig) for a preset, applying CLI overrides."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {list(PRESETS)}")
    spec = PRESETS[preset]
    model = ModelConfig(**asdict(spec["model"]))
    train = TrainConfig(**spec["train"])
    for k, v in overrides.items():
        if v is None:
            continue
        if hasattr(model, k):
            setattr(model, k, v)
        if hasattr(train, k):
            setattr(train, k, v)
    model.__post_init__()
    return model, train
