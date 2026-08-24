"""Conservative complete-link clustering for sparse next-token candidates."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .soft_tokens import SparseCandidates, norm_preserving_weighted_fusion


@dataclass(frozen=True)
class ClusterConfig:
    min_cluster_mass: float = 0.60
    max_pair_cosine_distance: float = 0.08
    max_transition_distance: float | None = None
    protected_pairs: frozenset[tuple[int, int]] = field(default_factory=frozenset)


@dataclass(frozen=True)
class ClusterDecision:
    accepted: bool
    token_ids: tuple[int, ...]
    probabilities: tuple[float, ...]
    mass: float
    maximum_cosine_distance: float
    transition_dispersion: float | None
    reason: str


def _protected(left: int, right: int, pairs: frozenset[tuple[int, int]]) -> bool:
    return (left, right) in pairs or (right, left) in pairs


def _maximum_distance(vectors: torch.Tensor) -> float:
    if len(vectors) < 2:
        return 0.0
    normalized = F.normalize(vectors.float(), dim=-1)
    distance = 1.0 - normalized @ normalized.T
    return float(distance.max())


def select_safe_cluster(
    candidates: SparseCandidates,
    semantic_vectors: torch.Tensor,
    config: ClusterConfig,
    *,
    transition_vectors: torch.Tensor | None = None,
) -> ClusterDecision:
    if semantic_vectors.shape[0] != len(candidates.token_ids):
        raise ValueError("semantic vector rows must match candidates")
    if transition_vectors is not None and transition_vectors.shape[0] != len(
        candidates.token_ids
    ):
        raise ValueError("transition vector rows must match candidates")
    if not 0 <= config.min_cluster_mass <= 1:
        raise ValueError("min_cluster_mass must be in [0, 1]")

    best: tuple[list[int], float, float, float | None] | None = None
    for seed in range(len(candidates.token_ids)):
        members = [seed]
        order = sorted(
            (index for index in range(len(candidates.token_ids)) if index != seed),
            key=lambda index: (-float(candidates.probabilities[index]), index),
        )
        for index in order:
            token = int(candidates.token_ids[index])
            if any(
                _protected(
                    token, int(candidates.token_ids[member]), config.protected_pairs
                )
                for member in members
            ):
                continue
            proposed = members + [index]
            if (
                _maximum_distance(semantic_vectors[proposed])
                > config.max_pair_cosine_distance
            ):
                continue
            if (
                transition_vectors is not None
                and config.max_transition_distance is not None
                and _maximum_distance(transition_vectors[proposed])
                > config.max_transition_distance
            ):
                continue
            members = proposed
        mass = float(candidates.probabilities[members].sum())
        semantic_distance = _maximum_distance(semantic_vectors[members])
        transition_distance = (
            _maximum_distance(transition_vectors[members])
            if transition_vectors is not None
            else None
        )
        key = (mass, -semantic_distance, -min(members))
        if best is None or key > (best[1], -best[2], -min(best[0])):
            best = (members, mass, semantic_distance, transition_distance)
    assert best is not None
    members, mass, maximum_distance, transition_distance = best
    accepted = mass >= config.min_cluster_mass and len(members) > 1
    if len(members) == 1:
        reason = "no_complete_link_partner"
    elif mass < config.min_cluster_mass:
        reason = "cluster_mass_below_threshold"
    else:
        reason = "complete_link_safety_passed"
    return ClusterDecision(
        accepted=accepted,
        token_ids=tuple(int(candidates.token_ids[index]) for index in members),
        probabilities=tuple(
            float(candidates.probabilities[index]) for index in members
        ),
        mass=mass,
        maximum_cosine_distance=maximum_distance,
        transition_dispersion=transition_distance,
        reason=reason,
    )


def fused_cluster_embedding(
    decision: ClusterDecision, embedding_table: torch.Tensor
) -> torch.Tensor:
    if not decision.accepted:
        raise ValueError("cannot fuse a rejected cluster")
    ids = torch.tensor(decision.token_ids, device=embedding_table.device)
    weights = torch.tensor(
        decision.probabilities,
        device=embedding_table.device,
        dtype=embedding_table.dtype,
    )
    return norm_preserving_weighted_fusion(embedding_table[ids], weights)
