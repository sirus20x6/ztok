from __future__ import annotations

import torch

from config import ExperimentConfig, TrainingConfig
from models.common import ModelConfig
from models.rwkv_lab_adapter import RWKVLabLM
from train import make_optimizer, output_head_is_split, split_tied_output_head


def test_late_head_split_clones_weight_and_adam_state(monkeypatch) -> None:
    monkeypatch.setenv("RWKV8_FORCE_PYREF", "1")
    model_config = ModelConfig(
        vocab_size=32,
        width=32,
        layers=2,
        heads=2,
        mlp_multiple=2,
        max_sequence_length=16,
    )
    config = ExperimentConfig(
        schema="ztok.nanogpt_superposition.config.v1",
        run_name="head-split-test",
        seed=1,
        architecture="rwkv",
        data_dir="unused",
        output_dir="unused",
        model=model_config,
        training=TrainingConfig(),
    )
    model = RWKVLabLM(model_config)
    optimizer, parameters = make_optimizer(model, [], config)
    embedding = model.token_embedding.weight
    embedding.grad = torch.ones_like(embedding)
    optimizer.step()
    source_state = optimizer.state[embedding]

    assert split_tied_output_head(model, optimizer, parameters) is True
    head = model.output.weight
    assert output_head_is_split(model)
    assert head is not embedding
    assert torch.equal(head, embedding)
    assert any(parameter is head for parameter in parameters)
    assert optimizer.state[head]["exp_avg"] is not source_state["exp_avg"]
    assert torch.equal(optimizer.state[head]["exp_avg"], source_state["exp_avg"])
    assert split_tied_output_head(model, optimizer, parameters) is False


def test_spectral_muon_routes_internal_matrices_and_keeps_head_on_adam(
    monkeypatch,
) -> None:
    monkeypatch.setenv("RWKV8_FORCE_PYREF", "1")
    model_config = ModelConfig(
        vocab_size=32,
        width=32,
        layers=2,
        heads=2,
        mlp_multiple=2,
        max_sequence_length=16,
    )
    config = ExperimentConfig(
        schema="ztok.nanogpt_superposition.config.v1",
        run_name="muon-test",
        seed=1,
        architecture="rwkv",
        data_dir="unused",
        output_dir="unused",
        model=model_config,
        training=TrainingConfig(
            learning_rate=0.02,
            optimizer_kind="spectral_muon",
            spectral_muon={
                "learning_rate": 0.02,
                "fallback_multiplier": 0.015,
                "batched": True,
                "adam_beta1": 0.9,
                "adam_beta2": 0.95,
            },
        ),
    )
    model = RWKVLabLM(model_config)

    optimizer, parameters = make_optimizer(model, [], config)

    assert len(parameters) == len(list(model.parameters()))
    assert len(optimizer.param_groups) == 2
    matrix, fallback = optimizer.param_groups
    assert matrix["use_muon"] is True
    assert matrix["lr"] == 0.02
    assert fallback["use_muon"] is False
    assert fallback["lr"] == 0.0003
    fallback_ids = {id(parameter) for parameter in fallback["params"]}
    assert id(model.token_embedding.weight) in fallback_ids
    assert id(model.output.weight) in fallback_ids
