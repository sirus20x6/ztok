from __future__ import annotations

import pytest

from config import ExperimentConfig, TrainingConfig
from models.common import ModelConfig
from train import _learning_rate


def _config(training: TrainingConfig) -> ExperimentConfig:
    return ExperimentConfig(
        schema="ztok.nanogpt_superposition.config.v1",
        run_name="schedule-test",
        seed=1,
        architecture="rwkv",
        data_dir="data",
        output_dir="output",
        model=ModelConfig(vocab_size=16),
        training=training,
    )


def test_late_linear_schedule_plateaus_then_cools_over_final_fifth() -> None:
    config = _config(
        TrainingConfig(
            learning_rate=3e-4,
            min_learning_rate=3e-5,
            warmup_steps=10,
            lr_schedule_basis="budget",
            lr_decay_kind="late_linear",
            cooldown_fraction=0.2,
        )
    )

    assert _learning_rate(config, 9, 10, 100) == pytest.approx(3e-4)
    assert _learning_rate(config, 10, 79, 100) == pytest.approx(3e-4)
    assert _learning_rate(config, 10, 80, 100) == pytest.approx(3e-4)
    assert _learning_rate(config, 10, 90, 100) == pytest.approx(1.65e-4)
    assert _learning_rate(config, 10, 100, 100) == pytest.approx(3e-5)
