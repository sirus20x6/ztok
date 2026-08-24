import pytest
import torch
from torch import nn

from models.common import LMOutput
from superposition.statement_primer import (
    _aligned_expansion,
    collate_statement_primer_batch,
    norm_preserving_fusion,
    statement_primer_objective,
)


class FakePipeline:
    branches = {
        "Gary is bright.": [1, 2, 3, 4],
        "Gary is radiant.": [1, 2, 5, 4],
        "Harry is bright.": [8, 9, 2, 3, 4],
    }

    def encode(self, text: str) -> list[int]:
        return self.branches[text]


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(16, 8)
        self.output = nn.Linear(8, 16, bias=False)

    def forward(self, *, input_embeddings: torch.Tensor) -> LMOutput:
        return LMOutput(
            logits=self.output(input_embeddings), hidden=input_embeddings
        )


def expansion_record(*statements: str) -> dict:
    return {
        "expansion_id": "test",
        "statement_kind": "fact",
        "expanded_slot_kind": "predicate",
        "alternatives": [
            {"statements": [statement], "weight": 1 / len(statements)}
            for statement in statements
        ],
    }


def test_exact_one_token_alternatives_become_a_primer_example() -> None:
    expansion = _aligned_expansion(
        expansion_record("Gary is bright.", "Gary is radiant."), FakePipeline()
    )

    assert expansion is not None
    assert expansion.expanded_token_index == 2
    assert expansion.alternative_token_ids == (3, 5)
    assert expansion.weights == pytest.approx((0.5, 0.5))


def test_different_length_branches_are_conservatively_rejected() -> None:
    assert (
        _aligned_expansion(
            expansion_record("Gary is bright.", "Harry is bright."), FakePipeline()
        )
        is None
    )


def test_norm_preserving_fusion_matches_weighted_source_norm() -> None:
    embeddings = torch.tensor([[[3.0, 0.0], [0.0, 4.0]]])
    weights = torch.tensor([[0.25, 0.75]])
    fused = norm_preserving_fusion(embeddings, weights)

    assert fused.norm(dim=-1).item() == pytest.approx(3.75)


def test_primer_objective_backpropagates_through_fused_alternatives() -> None:
    expansion = _aligned_expansion(
        expansion_record("Gary is bright.", "Gary is radiant."), FakePipeline()
    )
    assert expansion is not None
    batch = collate_statement_primer_batch([expansion], device=torch.device("cpu"))
    model = TinyModel()

    result = statement_primer_objective(model, batch)
    result.loss.backward()

    assert torch.isfinite(result.loss)
    assert result.diagnostics["set_target_positions"] == 1
    assert model.token_embedding.weight.grad is not None
    assert model.token_embedding.weight.grad[3].abs().sum() > 0
    assert model.token_embedding.weight.grad[5].abs().sum() > 0
