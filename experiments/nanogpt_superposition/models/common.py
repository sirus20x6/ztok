"""Shared model contracts and accounting utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    width: int = 256
    layers: int = 6
    heads: int = 8
    mlp_multiple: int = 3
    max_sequence_length: int = 2048
    tie_embeddings: bool = True
    dropout: float = 0.0
    rope_base: float = 10_000.0

    def validate(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if self.width <= 0 or self.layers <= 0 or self.heads <= 0:
            raise ValueError("width, layers, and heads must be positive")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if self.max_sequence_length < 2:
            raise ValueError("max_sequence_length must be at least 2")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LMOutput:
    logits: torch.Tensor
    hidden: torch.Tensor
    state: Any = None
    diagnostics: dict[str, Any] | None = None


class RMSNorm(nn.Module):
    def __init__(self, width: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.epsilon)
        return (x.float() * scale).to(dtype=x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, width: int, multiple: int, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = _round_multiple(int(width * multiple * 2 / 3), 64)
        self.gate = nn.Linear(width, hidden, bias=False)
        self.value = nn.Linear(width, hidden, bias=False)
        self.output = nn.Linear(hidden, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.output(F.silu(self.gate(x)) * self.value(x)))


def _round_multiple(value: int, multiple: int) -> int:
    return multiple * ((value + multiple - 1) // multiple)


def initialize_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)


def count_parameters(model: nn.Module) -> dict[str, int]:
    embedding_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, nn.Embedding)
        for parameter in module.parameters(recurse=False)
    }
    total = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    embedding = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) in embedding_ids
    )
    return {"total": total, "embedding": embedding, "non_embedding": total - embedding}


def estimate_training_flops(
    architecture: str,
    config: ModelConfig,
    batch_size: int,
    positions: int,
    *,
    learned_fusion_flops: int = 0,
) -> int:
    """Transparent approximate forward+backward FLOPs for budget matching.

    The 6*N rule accounts for dense parameter matmuls. Transformer attention
    adds the quadratic score/value work; RWKV adds linear recurrent mixing.
    This estimate is logged alongside measured GPU-seconds and is never used
    as a substitute for timing.
    """

    dense_parameters = 12 * config.layers * config.width * config.width
    dense = 6 * batch_size * positions * dense_parameters
    if architecture == "transformer":
        sequence = (
            12 * batch_size * config.layers * positions * positions * config.width
        )
    elif architecture == "rwkv":
        sequence = 18 * batch_size * config.layers * positions * config.width
    else:
        raise ValueError(f"unknown architecture: {architecture}")
    output = 6 * batch_size * positions * config.width * config.vocab_size
    return int(dense + sequence + output + learned_fusion_flops)


def cross_entropy_bits_per_byte(
    loss_nats: float,
    predicted_tokens: int,
    source_bytes: int,
) -> float:
    if source_bytes <= 0:
        raise ValueError("source_bytes must be positive")
    return loss_nats * predicted_tokens / (source_bytes * 0.6931471805599453)
