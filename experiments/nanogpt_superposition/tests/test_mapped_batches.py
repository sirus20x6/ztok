import torch
from models.common import ModelConfig
from models.transformer import TransformerLM
from superposition.mapped_batches import MappedSuperbatchObjective


def test_mapped_superbatch_fuses_across_examples_and_trains() -> None:
    model = TransformerLM(
        ModelConfig(
            vocab_size=32,
            width=32,
            layers=1,
            heads=2,
            max_sequence_length=8,
        )
    )
    tokens = torch.tensor(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ]
    )
    ranges = torch.tensor(
        [
            [[0, 1], [1, 2], [2, 4]],
            [[0, 1], [1, 2], [2, 4]],
        ]
    )
    result = MappedSuperbatchObjective()(model, tokens, ranges)
    result.loss.backward()

    assert torch.isfinite(result.loss)
    assert result.model_positions == 2
    assert result.source_tokens == 8
    assert result.target_tokens == 6
    assert result.diagnostics["positions_per_example_span"] == 0.5
    assert model.token_embedding.weight.grad is not None


def test_mapped_superbatch_rejects_non_monotonic_ranges() -> None:
    model = TransformerLM(
        ModelConfig(
            vocab_size=16,
            width=16,
            layers=1,
            heads=2,
            max_sequence_length=8,
        )
    )
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    ranges = torch.tensor(
        [
            [[1, 2], [0, 1]],
            [[0, 1], [1, 2]],
        ]
    )
    try:
        MappedSuperbatchObjective()(model, tokens, ranges)
    except ValueError as error:
        assert "monotonic" in str(error)
    else:
        raise AssertionError("non-monotonic mapping was accepted")
