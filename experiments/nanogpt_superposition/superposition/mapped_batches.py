"""Model-side execution for fully aligned dataset-superposition superbatches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .soft_tokens import norm_preserving_weighted_fusion


@dataclass
class MappedObjectiveResult:
    loss: torch.Tensor
    model_positions: int
    source_tokens: int
    target_tokens: int
    diagnostics: dict[str, Any]


def _pool_span(vectors: torch.Tensor) -> torch.Tensor:
    weights = torch.ones(len(vectors), device=vectors.device, dtype=vectors.dtype)
    return norm_preserving_weighted_fusion(vectors, weights).to(vectors.dtype)


class MappedSuperbatchObjective(nn.Module):
    """Fuse structurally aligned spans across examples into one causal sequence.

    This first reference adapter accepts only complete, monotonic alignments.
    Plans containing unique units must use ordinary rows until a branch-aware
    representation is implemented.
    """

    def forward(
        self,
        model: nn.Module,
        token_ids: torch.Tensor,
        span_ranges: torch.Tensor,
        fusion_weights: torch.Tensor | None = None,
        *,
        return_diagnostics: bool = False,
    ) -> MappedObjectiveResult:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [examples, sequence]")
        if span_ranges.ndim != 3 or span_ranges.shape[0] != token_ids.shape[0]:
            raise ValueError("span_ranges must have shape [examples, slots, 2]")
        if span_ranges.shape[-1] != 2:
            raise ValueError("span_ranges must contain [start, end] pairs")
        examples, slots = span_ranges.shape[:2]
        if examples < 2 or slots < 2:
            raise ValueError("a mapped superbatch needs two examples and two slots")
        if fusion_weights is None:
            fusion_weights = torch.ones(
                examples,
                slots,
                device=token_ids.device,
                dtype=model.token_embedding.weight.dtype,
            )
        if fusion_weights.shape != (examples, slots):
            raise ValueError("fusion_weights must have shape [examples, slots]")
        if torch.any(fusion_weights < 0) or not torch.isfinite(fusion_weights).all():
            raise ValueError("fusion weights must be finite and non-negative")

        embedded = model.token_embedding(token_ids)
        fused_slots = []
        target_groups: list[list[int]] = []
        source_token_count = 0
        previous_ends = [-1] * examples
        for slot in range(slots):
            pooled = []
            target = []
            for example in range(examples):
                start = int(span_ranges[example, slot, 0])
                end = int(span_ranges[example, slot, 1])
                if not 0 <= start < end <= token_ids.shape[1]:
                    raise ValueError("span range is empty or out of bounds")
                if start < previous_ends[example]:
                    raise ValueError("span ranges must be monotonic")
                previous_ends[example] = end
                pooled.append(_pool_span(embedded[example, start:end]))
                ids = token_ids[example, start:end]
                target.extend(int(token) for token in ids)
                source_token_count += end - start
            stacked = torch.stack(pooled)
            fused_slots.append(
                norm_preserving_weighted_fusion(
                    stacked, fusion_weights[:, slot].to(stacked.dtype)
                ).to(stacked.dtype)
            )
            target_groups.append(target)

        inputs = torch.stack(fused_slots[:-1]).unsqueeze(0)
        output = model(
            input_embeddings=inputs,
            return_diagnostics=return_diagnostics,
        )
        log_probabilities = output.logits[0].log_softmax(dim=-1)
        losses = []
        target_count = 0
        for position, targets in enumerate(target_groups[1:]):
            indices = torch.tensor(targets, device=token_ids.device)
            losses.append(-log_probabilities[position, indices].mean())
            target_count += len(targets)
        loss = torch.stack(losses).mean()
        diagnostics: dict[str, Any] = {
            "schema": "ztok.dataset_superposition.v1",
            "example_count": examples,
            "aligned_slot_count": slots,
            "fusion": "norm_preserving_mean",
            "mapping": "complete_monotonic_alignment",
            "positions_per_example_span": (slots - 1) / (examples * (slots - 1)),
        }
        if output.diagnostics is not None:
            diagnostics.update(output.diagnostics)
        return MappedObjectiveResult(
            loss=loss,
            model_positions=slots - 1,
            source_tokens=source_token_count,
            target_tokens=target_count,
            diagnostics=diagnostics,
        )
