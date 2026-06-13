"""Mythos: a from-scratch, modern decoder-only transformer (GPT) language model.

Same architectural ingredients as today's frontier models (RMSNorm, RoPE,
SwiGLU, weight-tied embeddings, KV-cache) in a small, readable, scale-ready
codebase you fully own.
"""

from .config import ModelConfig, TrainConfig, PRESETS
from .model import GPT

__all__ = ["ModelConfig", "TrainConfig", "PRESETS", "GPT"]
__version__ = "0.1.0"
