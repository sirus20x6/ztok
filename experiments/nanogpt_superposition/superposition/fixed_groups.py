"""Vectorized model-side execution of ztok fixed superposition plans."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class FusionKind(str, Enum):
    MEAN = "mean"
    NORM_PRESERVING_MEAN = "norm_preserving_mean"
    LEARNED = "learned"

    @classmethod
    def parse(cls, value: str) -> FusionKind:
        return cls(value.replace("-", "_"))


class LearnedFixedFusion(nn.Module):
    """Ordered fixed-size fusion; parameters/FLOPs are explicitly reportable."""

    def __init__(self, width: int, group_size: int) -> None:
        super().__init__()
        self.width = width
        self.group_size = group_size
        self.position = nn.Parameter(torch.zeros(group_size, width))
        self.projection = nn.Sequential(
            nn.Linear(width * group_size, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )

    def forward(self, grouped: torch.Tensor) -> torch.Tensor:
        if grouped.shape[-2:] != (self.group_size, self.width):
            raise ValueError("learned fusion received an incompatible group shape")
        positioned = grouped + self.position
        return self.projection(positioned.flatten(-2))

    def estimated_forward_flops(self, batch: int, groups: int) -> int:
        return int(2 * batch * groups * self.width * self.width * (self.group_size + 1))


def group_embeddings(
    embeddings: torch.Tensor,
    group_size: int,
    fusion: FusionKind | str,
    learned_fusion: LearnedFixedFusion | None = None,
) -> torch.Tensor:
    if embeddings.ndim != 3:
        raise ValueError("embeddings must have shape [batch, sequence, width]")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if embeddings.shape[1] % group_size:
        raise ValueError("sequence length must be divisible by group_size")
    kind = fusion if isinstance(fusion, FusionKind) else FusionKind.parse(fusion)
    grouped = embeddings.reshape(
        embeddings.shape[0],
        embeddings.shape[1] // group_size,
        group_size,
        embeddings.shape[2],
    )
    if kind is FusionKind.MEAN:
        return grouped.mean(dim=2)
    if kind is FusionKind.NORM_PRESERVING_MEAN:
        raw = grouped.mean(dim=2)
        target_norm = grouped.float().norm(dim=-1).mean(dim=2, keepdim=True)
        return F.normalize(raw.float(), dim=-1).to(raw.dtype) * target_norm.to(
            raw.dtype
        )
    if learned_fusion is None:
        raise ValueError("learned fusion requires a LearnedFixedFusion module")
    return learned_fusion(grouped)


def scale_source_gradient(
    fused_embeddings: torch.Tensor, multiplier: float
) -> torch.Tensor:
    """Scale only the gradient flowing upstream from a fused input activation."""
    if multiplier <= 0:
        raise ValueError("source gradient multiplier must be positive")
    if multiplier != 1.0 and fused_embeddings.requires_grad:
        fused_embeddings.register_hook(lambda gradient: gradient * multiplier)
    return fused_embeddings


def grouped_positions(
    length: int, group_size: int, device: torch.device
) -> torch.Tensor:
    if length % group_size:
        raise ValueError("length must be divisible by group_size")
    # Preserve the normalized source-range center without forcing a tokenizer
    # positional policy. Both models receive this only where they support it.
    return (
        torch.arange(length // group_size, device=device, dtype=torch.float32)
        * group_size
        + (group_size - 1) / 2
    )


def bag_target_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Unordered multiset cross-entropy for all tokens in the next group.

    Repeated tokens retain multiplicity. This is the stable paper-like bag
    condition used in screening; ordered targets use a separate auxiliary head.
    """

    if logits.shape[:-1] != targets.shape[:-1]:
        raise ValueError("logits and target group axes do not align")
    log_probabilities = logits.log_softmax(dim=-1)
    selected = (
        log_probabilities.unsqueeze(-2)
        .expand(*targets.shape, logits.shape[-1])
        .gather(-1, targets.unsqueeze(-1))
        .squeeze(-1)
    )
    return -selected.mean()


def ordered_target_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape[:-1] != targets.shape:
        raise ValueError("ordered logits and targets do not align")
    return F.cross_entropy(logits.flatten(0, -2), targets.flatten())


@dataclass
class ObjectiveResult:
    loss: torch.Tensor
    model_positions: int
    source_tokens: int
    target_tokens: int
    diagnostics: dict[str, Any]


class FixedObjective(nn.Module):
    def __init__(
        self,
        width: int,
        vocab_size: int,
        group_size: int,
        fusion: FusionKind | str,
        target: str = "bag",
    ) -> None:
        super().__init__()
        if group_size < 2:
            raise ValueError("fixed superposition group_size must be at least 2")
        if target not in {"bag", "ordered"}:
            raise ValueError("target must be bag or ordered")
        self.group_size = group_size
        self.fusion = (
            fusion if isinstance(fusion, FusionKind) else FusionKind.parse(fusion)
        )
        self.target = target
        self.learned_fusion = (
            LearnedFixedFusion(width, group_size)
            if self.fusion is FusionKind.LEARNED
            else None
        )
        self.ordered_head = (
            nn.Linear(width, group_size * vocab_size, bias=False)
            if target == "ordered"
            else None
        )
        self.vocab_size = vocab_size

    def forward(
        self,
        model: nn.Module,
        token_ids: torch.Tensor,
        *,
        source_gradient_multiplier: float = 1.0,
        return_diagnostics: bool = False,
    ) -> ObjectiveResult:
        usable = token_ids.shape[1] - token_ids.shape[1] % self.group_size
        if usable < 2 * self.group_size:
            raise ValueError("batch needs at least two complete groups")
        token_ids = token_ids[:, :usable]
        all_groups = token_ids.reshape(
            token_ids.shape[0], usable // self.group_size, self.group_size
        )
        source_groups = all_groups[:, :-1]
        target_groups = all_groups[:, 1:]
        flat_sources = source_groups.flatten(1)
        source_embeddings = model.token_embedding(flat_sources)
        fused = group_embeddings(
            source_embeddings,
            self.group_size,
            self.fusion,
            self.learned_fusion,
        )
        fused = scale_source_gradient(fused, source_gradient_multiplier)
        positions = grouped_positions(
            flat_sources.shape[1], self.group_size, token_ids.device
        )
        output = model(
            input_embeddings=fused,
            positions=positions,
            return_diagnostics=return_diagnostics,
        )
        if self.target == "bag":
            loss = bag_target_loss(output.logits, target_groups)
        else:
            assert self.ordered_head is not None
            ordered = self.ordered_head(output.hidden).reshape(
                *output.hidden.shape[:2], self.group_size, self.vocab_size
            )
            loss = ordered_target_loss(ordered, target_groups)
        diagnostics: dict[str, Any] = {
            "group_size": self.group_size,
            "fusion": self.fusion.value,
            "target": self.target,
            "superposed_source_gradient_multiplier": source_gradient_multiplier,
            "positions_per_source_token": fused.shape[1] / token_ids.shape[1],
        }
        if output.diagnostics is not None:
            for key, value in output.diagnostics.items():
                if key == "attention":
                    # Every coarse input is a superposed position. Retain the
                    # layerwise mean attention for persistence diagnostics,
                    # without serializing full O(T^2) matrices.
                    diagnostics["attention_to_superposed_positions"] = [
                        float(layer.detach().float().mean()) for layer in value
                    ]
                else:
                    diagnostics[key] = value
        return ObjectiveResult(
            loss=loss,
            model_positions=fused.shape[0] * fused.shape[1],
            source_tokens=token_ids.numel(),
            target_tokens=target_groups.numel(),
            diagnostics=diagnostics,
        )

    def auxiliary_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def learned_fusion_flops(self, batch: int, groups: int) -> int:
        if self.learned_fusion is None:
            return 0
        return self.learned_fusion.estimated_forward_flops(batch, groups)


def audit_ztok_fixed_plan(
    pipeline: Any, text: str | bytes, group_size: int
) -> dict[str, Any]:
    """Assert the vectorized trainer contract against the real ztok plan API."""

    from ztok.superposition import FixedSuperpositionConfig, build_fixed_plan

    ordinary = pipeline.encode(text)
    plan = build_fixed_plan(
        pipeline,
        text,
        FixedSuperpositionConfig(group_size=group_size),
    )
    if list(plan.original_ids) != ordinary:
        raise AssertionError("ztok superposition plan changed ordinary token IDs")
    flattened = [
        source.token_index for group in plan.groups for source in group.sources
    ]
    if flattened != list(range(len(ordinary))):
        raise AssertionError("ztok plan is not a complete contiguous partition")
    return {
        "schema": plan.schema,
        "original_token_count": plan.original_token_count,
        "output_token_count": plan.output_token_count,
        "group_size": group_size,
    }
