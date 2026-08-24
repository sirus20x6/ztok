"""Apply ztok superposition descriptors to PyTorch embeddings.

The tokenizer only emits an operation plan. It deliberately does not own an
embedding table or depend on PyTorch. For fixed plans, ``token_embeddings``
contains the ordinary embedding-table lookups. For CCSS plans, source indices
instead address contextual span vectors pooled from a documented encoder
layer and pooling policy (for example, a selected Qwen middle/late layer).
"""

from __future__ import annotations

from typing import Any


def apply_superposition_plan(token_embeddings: Any, plan: Any) -> Any:
    """Fuse ``[source, hidden]`` embeddings according to a ztok plan.

    ``plan`` may be the typed ``ztok.superposition.SuperpositionPlan`` or any
    object exposing compatible ``groups`` and ``sources`` attributes.
    """

    import torch
    import torch.nn.functional as functional

    if token_embeddings.ndim != 2:
        raise ValueError("token_embeddings must have shape [source, hidden]")

    outputs = []
    for group in plan.groups:
        indices = torch.tensor(
            [source.token_index for source in group.sources],
            device=token_embeddings.device,
            dtype=torch.long,
        )
        weights = torch.tensor(
            [source.weight for source in group.sources],
            device=token_embeddings.device,
            dtype=token_embeddings.dtype,
        )
        source_embeddings = token_embeddings.index_select(0, indices)
        fusion = getattr(group.fusion, "value", group.fusion)

        if fusion == "mean":
            fused = source_embeddings.mean(dim=0)
        elif fusion == "weighted_mean":
            fused = (
                source_embeddings * weights[:, None]
            ).sum(dim=0) / weights.sum()
        elif fusion == "norm_preserving_mean":
            raw = (
                source_embeddings * weights[:, None]
            ).sum(dim=0) / weights.sum()
            target_norm = (
                source_embeddings.norm(dim=-1) * weights
            ).sum() / weights.sum()
            fused = functional.normalize(raw, dim=-1) * target_norm
        else:
            raise ValueError(f"unsupported fusion kind: {fusion!r}")
        outputs.append(fused)

    if not outputs:
        return token_embeddings.new_empty((0, token_embeddings.shape[-1]))
    return torch.stack(outputs)


__all__ = ["apply_superposition_plan"]
