from __future__ import annotations

import torch
from models.common import ModelConfig, count_parameters
from models.rwkv import RWKVLM
from models.transformer import TransformerLM


def config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        width=32,
        layers=2,
        heads=2,
        mlp_multiple=2,
        max_sequence_length=32,
    )


def test_transformer_accepts_ids_and_identical_embeddings() -> None:
    torch.manual_seed(1)
    model = TransformerLM(config()).eval()
    ids = torch.randint(0, 64, (2, 12))
    by_id = model(ids).logits
    by_embedding = model(input_embeddings=model.token_embedding(ids)).logits
    torch.testing.assert_close(by_id, by_embedding)


def test_rwkv_reference_full_sequence_matches_recurrent_steps() -> None:
    torch.manual_seed(2)
    model = RWKVLM(config()).eval()
    ids = torch.randint(0, 64, (2, 10))
    full = model(ids)
    state = None
    logits = []
    for index in range(ids.shape[1]):
        output = model.step(model.token_embedding(ids[:, index]), state)
        state = output.state
        logits.append(output.logits)
    torch.testing.assert_close(
        full.logits, torch.cat(logits, dim=1), rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        full.state.flattened(), state.flattened(), rtol=1e-5, atol=1e-6
    )


def test_parameter_accounting_splits_embeddings() -> None:
    counts = count_parameters(TransformerLM(config()))
    assert counts["total"] == counts["embedding"] + counts["non_embedding"]
    assert counts["embedding"] == 64 * 32


def test_models_remain_finite_after_backward() -> None:
    ids = torch.randint(0, 64, (2, 8))
    for model in (TransformerLM(config()), RWKVLM(config())):
        loss = torch.nn.functional.cross_entropy(
            model(ids[:, :-1]).logits.flatten(0, 1), ids[:, 1:].flatten()
        )
        loss.backward()
        assert torch.isfinite(loss)
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
