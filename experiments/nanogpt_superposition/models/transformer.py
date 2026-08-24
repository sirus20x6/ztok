"""Small pre-norm decoder Transformer with RoPE and causal attention."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .common import LMOutput, ModelConfig, RMSNorm, SwiGLU, initialize_weights


def _rope(x: torch.Tensor, positions: torch.Tensor, base: float) -> torch.Tensor:
    dimension = x.shape[-1]
    if dimension % 2:
        raise ValueError("RoPE head dimension must be even")
    frequency = base ** (
        -torch.arange(0, dimension, 2, device=x.device, dtype=torch.float32) / dimension
    )
    angle = positions.float()[:, None] * frequency[None, :]
    cosine = angle.cos().to(dtype=x.dtype)[None, :, None, :]
    sine = angle.sin().to(dtype=x.dtype)[None, :, None, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
    ).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.head_width = config.width // config.heads
        self.rope_base = config.rope_base
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.output = nn.Linear(config.width, config.width, bias=False)
        self.dropout = config.dropout

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        *,
        return_attention: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, length, width = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, self.head_width)
        query, key, value = qkv.unbind(dim=2)
        query = _rope(query, positions, self.rope_base).transpose(1, 2)
        key = _rope(key, positions, self.rope_base).transpose(1, 2)
        value = value.transpose(1, 2)

        attention = None
        if return_attention:
            scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_width)
            mask = torch.ones(length, length, device=x.device, dtype=torch.bool).tril()
            scores = scores.masked_fill(~mask, float("-inf"))
            attention = scores.softmax(dim=-1)
            output = attention @ value
        else:
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        output = output.transpose(1, 2).reshape(batch, length, width)
        return self.output(output), attention


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.width)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.width)
        self.mlp = SwiGLU(config.width, config.mlp_multiple, config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        *,
        return_attention: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        update, attention = self.attention(
            self.attention_norm(x), positions, return_attention=return_attention
        )
        x = x + update
        return x + self.mlp(self.mlp_norm(x)), attention


class TransformerLM(nn.Module):
    architecture = "transformer"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.layers)
        )
        self.norm = RMSNorm(config.width)
        self.output = nn.Linear(config.width, config.vocab_size, bias=False)
        self.apply(initialize_weights)
        if config.tie_embeddings:
            self.output.weight = self.token_embedding.weight

    def forward(
        self,
        token_ids: torch.Tensor | None = None,
        *,
        input_embeddings: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        return_diagnostics: bool = False,
        **_: Any,
    ) -> LMOutput:
        if (token_ids is None) == (input_embeddings is None):
            raise ValueError("provide exactly one of token_ids or input_embeddings")
        x = (
            self.token_embedding(token_ids)
            if input_embeddings is None
            else input_embeddings
        )
        if x.ndim != 3:
            raise ValueError("model input must have shape [batch, sequence, width]")
        if x.shape[1] > self.config.max_sequence_length:
            raise ValueError("sequence exceeds max_sequence_length")
        if positions is None:
            positions = torch.arange(x.shape[1], device=x.device)
        if positions.shape != (x.shape[1],):
            raise ValueError("positions must have shape [sequence]")

        attentions: list[torch.Tensor] = []
        layer_norms: list[float] = []
        for block in self.blocks:
            x, attention = block(x, positions, return_attention=return_diagnostics)
            if return_diagnostics:
                assert attention is not None
                attentions.append(attention.detach())
                layer_norms.append(float(x.detach().float().norm(dim=-1).mean()))
        hidden = self.norm(x)
        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                "attention": attentions,
                "layer_representation_norm": layer_norms,
                "output_entropy": float(
                    torch.distributions.Categorical(
                        logits=self.output(hidden[:, -1]).detach()
                    )
                    .entropy()
                    .mean()
                ),
            }
        return LMOutput(
            logits=self.output(hidden),
            hidden=hidden,
            diagnostics=diagnostics,
        )
