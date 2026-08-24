from __future__ import annotations

import torch
from config import ExperimentConfig, TrainingConfig
from models.common import ModelConfig
from train import (
    _checkpoint_config_matches,
    _cpu_rng_state,
    _restore_training_counters,
)


def test_checkpoint_rng_state_is_restored_to_cpu() -> None:
    state = torch.Generator().get_state()
    if torch.cuda.is_available():
        state = state.cuda()
    restored = _cpu_rng_state(state)
    assert restored.device.type == "cpu"
    assert restored.dtype == torch.uint8
    torch.Generator().set_state(restored)


def test_resume_restores_only_cumulative_training_counters() -> None:
    counters = {"step": 0, "source_bytes": 0}
    saved = {
        "step": 12,
        "source_bytes": 34,
        "validation_bits_per_byte": 1.5,
        "wall_seconds": 10.0,
    }
    _restore_training_counters(counters, saved)
    assert counters == {"step": 12, "source_bytes": 34}


def test_pre_primer_checkpoint_restores_ordinary_step() -> None:
    counters = {"step": 0, "ordinary_step": 0, "primer_step": 0}
    _restore_training_counters(counters, {"step": 12})
    assert counters == {"step": 12, "ordinary_step": 12, "primer_step": 0}


def test_pre_primer_default_config_remains_resume_compatible() -> None:
    config = ExperimentConfig(
        schema="ztok.nanogpt_superposition.config.v1",
        run_name="test",
        seed=1,
        architecture="rwkv",
        data_dir="data",
        output_dir="output",
        model=ModelConfig(vocab_size=16),
        training=TrainingConfig(),
    )
    saved = config.to_dict()
    for key in (
        "primer_path",
        "primer_epochs",
        "primer_batch_size",
        "primer_eval_interval",
    ):
        saved["training"].pop(key)

    assert _checkpoint_config_matches(saved, config)
