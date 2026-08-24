from __future__ import annotations

import torch
from models.common import ModelConfig
from models.transformer import TransformerLM
from superposition.fixed_groups import (
    FixedObjective,
    bag_target_loss,
    group_embeddings,
    scale_source_gradient,
)
from superposition.recovery import RecoverySchedule


def test_mean_and_norm_preserving_fusion() -> None:
    values = torch.tensor([[[3.0, 0.0], [0.0, 4.0], [1.0, 0.0], [1.0, 0.0]]])
    mean = group_embeddings(values, 2, "mean")
    norm = group_embeddings(values, 2, "norm_preserving_mean")
    torch.testing.assert_close(mean[0, 0], torch.tensor([1.5, 2.0]))
    torch.testing.assert_close(norm[0, 0], torch.tensor([2.1, 2.8]))


def test_bag_target_is_order_invariant() -> None:
    logits = torch.randn(2, 3, 20)
    targets = torch.randint(0, 20, (2, 3, 4))
    first = bag_target_loss(logits, targets)
    second = bag_target_loss(logits, targets.flip(-1))
    torch.testing.assert_close(first, second)


def test_source_gradient_scaling_does_not_scale_downstream_parameters() -> None:
    source = torch.tensor([[1.0, 2.0]], requires_grad=True)
    downstream = torch.tensor([[3.0], [4.0]], requires_grad=True)
    scaled = scale_source_gradient(source, 2**0.5)
    (scaled @ downstream).sum().backward()
    torch.testing.assert_close(source.grad, torch.tensor([[3.0, 4.0]]) * (2**0.5))
    torch.testing.assert_close(downstream.grad, torch.tensor([[1.0], [2.0]]))


def test_fixed_bag_and_ordered_objectives_train() -> None:
    config = ModelConfig(
        vocab_size=32, width=32, layers=1, heads=2, max_sequence_length=32
    )
    tokens = torch.randint(0, 32, (2, 17))
    for target in ("bag", "ordered"):
        model = TransformerLM(config)
        objective = FixedObjective(32, 32, 4, "norm_preserving_mean", target)
        result = objective(model, tokens)
        result.loss.backward()
        assert torch.isfinite(result.loss)
        assert result.model_positions == 2 * 3
        assert result.source_tokens == 2 * 16


def test_recovery_schedules_use_budget_fraction() -> None:
    schedule = RecoverySchedule.parse("50/25/25")
    assert schedule.phase(49, 100) == "coarse"
    assert schedule.phase(50, 100) == "mixed"
    assert schedule.phase(75, 100) == "ordinary"
    assert schedule.use_superposition(60, 100, 0.49)
    assert not schedule.use_superposition(60, 100, 0.51)

    direct_recovery = RecoverySchedule.parse("50/0/50")
    assert direct_recovery.phase(49, 100) == "coarse"
    assert direct_recovery.phase(50, 100) == "ordinary"
    assert direct_recovery.learning_rate_multiplier(49, 100, 6.5) == 6.5
    assert direct_recovery.learning_rate_multiplier(50, 100, 6.5) == 1.0
    assert direct_recovery.learning_rate_multiplier(49, 100, 8.5, 6.0) == 8.5
    assert direct_recovery.learning_rate_multiplier(50, 100, 8.5, 6.0) == 6.0


def test_coarse_learning_rate_multiplier_tapers_through_mixed_phase() -> None:
    schedule = RecoverySchedule.parse("50/25/25")
    multiplier = 2**0.5
    assert schedule.learning_rate_multiplier(0, 100, multiplier) == multiplier
    assert schedule.learning_rate_multiplier(49, 100, multiplier) == multiplier
    assert schedule.learning_rate_multiplier(50, 100, multiplier) == multiplier
    assert (
        schedule.learning_rate_multiplier(62.5, 100, multiplier)
        == (multiplier + 1.0) / 2
    )
    assert schedule.learning_rate_multiplier(75, 100, multiplier) == 1.0
    assert schedule.learning_rate_multiplier(100, 100, multiplier) == 1.0


def test_learning_rate_multiplier_tapers_between_configured_phase_rates() -> None:
    schedule = RecoverySchedule.parse("50/25/25")
    assert schedule.learning_rate_multiplier(50, 100, 8.0, 6.0) == 8.0
    assert schedule.learning_rate_multiplier(62.5, 100, 8.0, 6.0) == 7.0
    assert schedule.learning_rate_multiplier(75, 100, 8.0, 6.0) == 6.0
