from __future__ import annotations

import pytest

from config import TrainingConfig


def test_batch_schedule_resolves_from_budget_fraction() -> None:
    config = TrainingConfig(
        batch_size=8,
        batch_size_schedule=(
            {"start_fraction": 0.0, "batch_size": 8},
            {"start_fraction": 0.5, "batch_size": 16},
        ),
    )
    config.validate()

    assert config.batch_size_at(0, 100) == 8
    assert config.batch_size_at(49, 100) == 8
    assert config.batch_size_at(50, 100) == 16
    assert config.batch_size_at(100, 100) == 16


def test_batch_schedule_rejects_ambiguous_stages() -> None:
    with pytest.raises(ValueError, match="first scheduled batch"):
        TrainingConfig(
            batch_size=8,
            batch_size_schedule=(
                {"start_fraction": 0.0, "batch_size": 16},
            ),
        ).validate()


def test_train_shape_schedule_resolves_context_and_batch_together() -> None:
    config = TrainingConfig(
        batch_size=8,
        context_tokens=1904,
        train_shape_schedule=(
            {"start_fraction": 0.0, "batch_size": 24, "context_tokens": 512},
            {"start_fraction": 1 / 3, "batch_size": 12, "context_tokens": 1024},
            {"start_fraction": 2 / 3, "batch_size": 8, "context_tokens": 1904},
        ),
    )
    config.validate()

    assert config.training_shape_at(0, 300) == (24, 512)
    assert config.training_shape_at(100, 300) == (12, 1024)
    assert config.training_shape_at(200, 300) == (8, 1904)


def test_train_shape_schedule_is_exclusive_with_batch_schedule() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        TrainingConfig(
            batch_size=8,
            batch_size_schedule=(
                {"start_fraction": 0.0, "batch_size": 8},
            ),
            train_shape_schedule=(
                {"start_fraction": 0.0, "batch_size": 24, "context_tokens": 512},
            ),
        ).validate()
