"""Sparse soft-token construction and teacher-forced exposure utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class SparseCandidates:
    token_ids: torch.Tensor
    probabilities: torch.Tensor
    entropy: float
    selected_mass: float


def select_sparse_candidates(
    logits: torch.Tensor,
    *,
    top_p: float = 0.90,
    max_k: int = 8,
    logit_margin: float = 5.0,
) -> SparseCandidates:
    if logits.ndim != 1:
        raise ValueError("logits must be one-dimensional")
    if not 0 < top_p <= 1 or max_k <= 0 or logit_margin < 0:
        raise ValueError("invalid sparse-candidate configuration")
    probabilities = logits.float().softmax(dim=-1)
    sorted_probability, sorted_ids = probabilities.sort(descending=True)
    sorted_logits = logits.float()[sorted_ids]
    cumulative = sorted_probability.cumsum(dim=0)
    # Include the token that crosses top-p, bounded by max_k and margin.
    count = int((cumulative < top_p).sum()) + 1
    count = min(max(count, 1), max_k, logits.numel())
    margin_ok = sorted_logits[:count] >= sorted_logits[0] - logit_margin
    count = max(int(margin_ok.sum()), 1)
    ids = sorted_ids[:count]
    selected = probabilities[ids]
    entropy = float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
    return SparseCandidates(ids, selected, entropy, float(selected.sum()))


def norm_preserving_weighted_fusion(
    embeddings: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    if embeddings.ndim != 2 or weights.shape != (embeddings.shape[0],):
        raise ValueError(
            "expected embeddings [candidate,width] and weights [candidate]"
        )
    if torch.any(weights < 0) or not torch.isfinite(weights).all():
        raise ValueError("weights must be finite and non-negative")
    denominator = weights.sum()
    if denominator <= 0:
        raise ValueError("weights must have positive mass")
    normalized = weights / denominator
    raw = (embeddings.float() * normalized[:, None]).sum(dim=0)
    target_norm = (embeddings.float().norm(dim=-1) * normalized).sum()
    return F.normalize(raw, dim=-1) * target_norm


class SoftTokenMetadata(nn.Module):
    def __init__(self, width: int, buckets: int = 8) -> None:
        super().__init__()
        if buckets < 2:
            raise ValueError("buckets must be at least two")
        self.buckets = buckets
        self.type_embedding = nn.Parameter(torch.zeros(width))
        self.entropy = nn.Embedding(buckets, width)
        self.mass = nn.Embedding(buckets, width)
        self.dispersion = nn.Embedding(buckets, width)
        nn.init.normal_(self.type_embedding, std=0.02)
        nn.init.normal_(self.entropy.weight, std=0.02)
        nn.init.normal_(self.mass.weight, std=0.02)
        nn.init.normal_(self.dispersion.weight, std=0.02)

    def _bucket(self, value: torch.Tensor) -> torch.Tensor:
        return (value.clamp(0, 1) * (self.buckets - 1)).round().long()

    def forward(
        self,
        fused: torch.Tensor,
        *,
        normalized_entropy: torch.Tensor,
        cluster_mass: torch.Tensor,
        dispersion: torch.Tensor,
    ) -> torch.Tensor:
        return (
            fused
            + self.type_embedding
            + self.entropy(self._bucket(normalized_entropy))
            + self.mass(self._bucket(cluster_mass))
            + self.dispersion(self._bucket(dispersion))
        )


@dataclass(frozen=True)
class CorruptionResult:
    embeddings: torch.Tensor
    replaced: torch.Tensor
    mean_cluster_size: float


def teacher_forced_soft_corruption(
    token_ids: torch.Tensor,
    embedding_table: torch.Tensor,
    *,
    rate: float,
    generator: torch.Generator,
    true_weight: float = 0.70,
    alternatives: int = 2,
    metadata: SoftTokenMetadata | None = None,
) -> CorruptionResult:
    """Replace selected inputs with mixtures dominated by their true token.

    Neighbors come from cosine similarity in the selected semantic space. The
    operation is intended as E0 exposure, not a claim of lexical equivalence.
    """

    if token_ids.ndim != 2:
        raise ValueError("token_ids must have shape [batch, sequence]")
    if not 0 <= rate <= 1 or not 0 < true_weight <= 1 or alternatives < 1:
        raise ValueError("invalid corruption configuration")
    ordinary = F.embedding(token_ids, embedding_table)
    if rate == 0:
        return CorruptionResult(
            ordinary, torch.zeros_like(token_ids, dtype=torch.bool), 0.0
        )
    replace = torch.rand(token_ids.shape, generator=generator, device="cpu") < rate
    replace = replace.to(token_ids.device)
    if not replace.any():
        return CorruptionResult(ordinary, replace, 0.0)
    normalized_table = F.normalize(embedding_table.detach().float(), dim=-1)
    unique = token_ids[replace].unique()
    neighbor_map: dict[int, torch.Tensor] = {}
    for token in unique.tolist():
        similarity = normalized_table @ normalized_table[token]
        similarity[token] = float("-inf")
        neighbor_map[token] = similarity.topk(
            min(alternatives, len(similarity) - 1)
        ).indices
    output = ordinary.clone()
    for row, column in replace.nonzero(as_tuple=False).tolist():
        token = int(token_ids[row, column])
        neighbors = neighbor_map[token]
        alternate_weight = (1.0 - true_weight) / len(neighbors)
        vectors = torch.cat(
            (embedding_table[token : token + 1], embedding_table[neighbors]), dim=0
        )
        weights = torch.tensor(
            [true_weight] + [alternate_weight] * len(neighbors),
            device=vectors.device,
            dtype=vectors.dtype,
        )
        fused = norm_preserving_weighted_fusion(vectors, weights).to(output.dtype)
        if metadata is not None:
            probability = weights.float() / weights.float().sum()
            normalized_entropy = (
                -(probability * probability.clamp_min(1e-12).log()).sum()
                / torch.tensor(len(probability), device=vectors.device).float().log()
            )
            normalized_vectors = F.normalize(vectors.float(), dim=-1)
            dispersion = (1.0 - normalized_vectors @ normalized_vectors.T).max()
            fused = metadata(
                fused[None],
                normalized_entropy=normalized_entropy[None],
                cluster_mass=torch.ones(1, device=vectors.device),
                dispersion=dispersion.clamp(0, 1)[None],
            )[0]
        output[row, column] = fused
    return CorruptionResult(output, replace, float(1 + alternatives))
