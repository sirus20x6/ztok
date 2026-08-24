from __future__ import annotations

import torch
from superposition.semantic_clusters import ClusterConfig, select_safe_cluster
from superposition.soft_tokens import (
    SoftTokenMetadata,
    SparseCandidates,
    select_sparse_candidates,
    teacher_forced_soft_corruption,
)


def test_sparse_candidates_respect_k_mass_and_margin() -> None:
    candidates = select_sparse_candidates(
        torch.tensor([5.0, 4.5, 4.0, -10.0]), top_p=0.9, max_k=2, logit_margin=2
    )
    assert candidates.token_ids.tolist() == [0, 1]
    assert candidates.selected_mass > 0.8


def test_complete_link_cluster_rejects_semantic_chain() -> None:
    candidates = SparseCandidates(
        token_ids=torch.tensor([10, 11, 12]),
        probabilities=torch.tensor([0.4, 0.35, 0.25]),
        entropy=1.0,
        selected_mass=1.0,
    )
    vectors = torch.tensor([[1.0, 0.0], [0.95, 0.31], [0.80, 0.60]])
    decision = select_safe_cluster(
        candidates,
        vectors,
        ClusterConfig(min_cluster_mass=0.70, max_pair_cosine_distance=0.06),
    )
    assert decision.accepted
    assert decision.token_ids == (10, 11)
    assert 12 not in decision.token_ids


def test_protected_pair_forces_discrete_fallback() -> None:
    candidates = SparseCandidates(
        token_ids=torch.tensor([1, 2]),
        probabilities=torch.tensor([0.6, 0.4]),
        entropy=0.5,
        selected_mass=1.0,
    )
    decision = select_safe_cluster(
        candidates,
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        ClusterConfig(protected_pairs=frozenset({(1, 2)})),
    )
    assert not decision.accepted


def test_teacher_forced_metadata_receives_gradients() -> None:
    torch.manual_seed(4)
    table = torch.nn.Parameter(torch.randn(16, 8))
    metadata = SoftTokenMetadata(8)
    result = teacher_forced_soft_corruption(
        torch.tensor([[1, 2, 3, 4]]),
        table,
        rate=1.0,
        generator=torch.Generator().manual_seed(5),
        metadata=metadata,
    )
    result.embeddings.square().mean().backward()
    assert result.replaced.all()
    assert metadata.type_embedding.grad is not None
    assert torch.isfinite(metadata.type_embedding.grad).all()
