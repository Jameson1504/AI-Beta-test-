"""The Mythos GPT: a modern decoder-only transformer, from scratch.

Architecture (matching the choices used by Llama / Mistral / modern LMs):

    token embedding
    -> N x TransformerBlock:
         x = x + Attention(RMSNorm(x))      # causal, RoPE, GQA-ready, KV-cache
         x = x + SwiGLU_MLP(RMSNorm(x))
    -> RMSNorm
    -> linear head (weight-tied to the token embedding)

Everything here is plain PyTorch and only a few hundred lines. The same module
trains a 1M-param toy on a CPU and a 1B-param model on GPUs.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean-subtraction, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm.type_as(x) * self.weight


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype):
    """Precompute cos/sin tables for rotary position embeddings."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)            # (seq_len, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)     # (seq_len, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin, pos: int = 0):
    """Apply RoPE to q,k of shape (B, n_head, T, head_dim).

    ``pos`` is the absolute position of the first token (for KV-cache decoding).
    """
    T = q.size(-2)
    cos = cos[pos:pos + T].unsqueeze(0).unsqueeze(0)   # (1,1,T,hd)
    sin = sin[pos:pos + T].unsqueeze(0).unsqueeze(0)
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


class Attention(nn.Module):
    """Causal multi-head self-attention with RoPE, GQA support and KV-cache."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.head_dim
        self.n_rep = self.n_head // self.n_kv_head

        self.q_proj = nn.Linear(cfg.n_embd, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, kv_cache=None, pos: int = 0):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin, pos=pos)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            if past_k is not None:
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            new_cache = (k, v)
        else:
            new_cache = None

        # Grouped-query attention: replicate KV heads to match query heads.
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # With a cache and a single new token there's nothing to mask causally.
        is_causal = kv_cache is None or q.size(2) == k.size(2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal and T > 1,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(y), new_cache


class SwiGLU(nn.Module):
    """SwiGLU feed-forward: down( silu(gate(x)) * up(x) )."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = int(cfg.mlp_ratio * cfg.n_embd)
        # round to a multiple of 64 for hardware friendliness
        hidden = 64 * ((hidden + 63) // 64)
        self.gate_proj = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.up_proj = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, kv_cache=None, pos: int = 0):
        h, new_cache = self.attn(self.attn_norm(x), cos, sin, kv_cache, pos)
        x = x + h
        x = x + self.mlp(self.mlp_norm(x))
        return x, new_cache


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        # RoPE tables (not parameters; rebuilt on device/dtype as needed).
        cos, sin = build_rope_cache(cfg.block_size, cfg.head_dim, cfg.rope_theta,
                                    torch.device("cpu"), torch.float32)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scaled init for residual projections (GPT-2 trick) for deep stability.
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and self.cfg.tie_weights:
            n -= self.tok_emb.weight.numel()
        return n

    def _rope(self, device, dtype):
        if self.rope_cos.device != device or self.rope_cos.dtype != dtype:
            cos, sin = build_rope_cache(self.cfg.block_size, self.cfg.head_dim,
                                        self.cfg.rope_theta, device, dtype)
            self.rope_cos, self.rope_sin = cos, sin
        return self.rope_cos, self.rope_sin

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        x = self.drop(self.tok_emb(idx))
        cos, sin = self._rope(x.device, x.dtype)
        for block in self.blocks:
            x, _ = block(x, cos, sin)
        x = self.norm(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
            return logits, loss
        # Inference: only compute logits for the final position (saves compute).
        logits = self.lm_head(x[:, [-1], :])
        return logits, None

    def configure_optimizer(self, weight_decay, lr, betas, device_type="cpu"):
        """AdamW with decay on 2-D weights only (not norms/biases/embeddings)."""
        decay, no_decay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused = device_type == "cuda" and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        return torch.optim.AdamW(groups, lr=lr, betas=betas, **({"fused": True} if fused else {}))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None,
                 eos_token_id=None):
        """Autoregressive sampling with a KV-cache for O(T) generation.

        Stops early once every sequence in the batch has emitted ``eos_token_id``.
        """
        self.eval()
        device = idx.device
        cos, sin = self._rope(device, torch.float32)
        caches: List = [(None, None) for _ in self.blocks]

        # Prime the cache with the prompt (cropped to the context window).
        cond = idx[:, -self.cfg.block_size:]
        pos = 0
        x = self.drop(self.tok_emb(cond))
        for i, block in enumerate(self.blocks):
            x, caches[i] = block(x, cos, sin, kv_cache=caches[i], pos=pos)
        logits = self.lm_head(self.norm(x)[:, [-1], :])
        pos += cond.size(1)

        for _ in range(max_new_tokens):
            next_id = self._sample(logits[:, -1, :], temperature, top_k, top_p)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_token_id is not None and (next_id == eos_token_id).all():
                break
            if pos >= self.cfg.block_size:
                # Context full: restart the cache from the most recent window.
                cond = idx[:, -self.cfg.block_size:]
                caches = [(None, None) for _ in self.blocks]
                pos = 0
                x = self.drop(self.tok_emb(cond))
                for i, block in enumerate(self.blocks):
                    x, caches[i] = block(x, cos, sin, kv_cache=caches[i], pos=pos)
                logits = self.lm_head(self.norm(x)[:, [-1], :])
                pos += cond.size(1)
                continue
            x = self.drop(self.tok_emb(next_id))
            for i, block in enumerate(self.blocks):
                x, caches[i] = block(x, cos, sin, kv_cache=caches[i], pos=pos)
            logits = self.lm_head(self.norm(x)[:, [-1], :])
            pos += 1
        return idx

    @staticmethod
    def _sample(logits, temperature, top_k, top_p):
        if temperature <= 0:                      # greedy
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / temperature
        if top_k is not None:
            k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, k)
            logits[logits < v[:, [-1]]] = -float("inf")
        if top_p is not None:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumprobs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            remove = cumprobs > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            logits = logits.scatter(1, sorted_idx, sorted_logits.masked_fill(remove, -float("inf")))
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
