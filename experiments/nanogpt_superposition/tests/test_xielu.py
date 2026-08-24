import pytest
import torch
import torch.nn.functional as F

from models.common import ModelConfig, count_parameters
from models.rwkv_lab_adapter import RWKVLabLM, XIELU


def test_xielu_reference_values_and_initial_alphas() -> None:
    activation = XIELU()
    values = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])

    alpha_p, alpha_n = activation.effective_alphas()
    actual = activation(values)
    expected = torch.where(
        values > 0,
        0.8 * values.square() + 0.5 * values,
        0.8 * torch.expm1(torch.clamp_max(values, -1e-6))
        - 0.8 * values
        + 0.5 * values,
    )

    torch.testing.assert_close(alpha_p, torch.tensor([0.8]))
    torch.testing.assert_close(alpha_n, torch.tensor([0.8]))
    torch.testing.assert_close(actual, expected)


def test_xielu_curvature_parameters_receive_gradients() -> None:
    activation = XIELU()
    activation(torch.tensor([-1.0, 1.0])).sum().backward()

    assert activation.alpha_p.grad is not None
    assert activation.alpha_n.grad is not None
    assert activation.alpha_p.grad.abs().sum() > 0
    assert activation.alpha_n.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_kernel_is_exact_at_bfloat16_precision() -> None:
    try:
        import xielu  # noqa: F401
    except ImportError:
        pytest.skip("optional xIELU CUDA extension is unavailable")
    torch.manual_seed(7)
    values = (
        torch.randn(4, 128, device="cuda", dtype=torch.bfloat16) * 2
    ).requires_grad_()
    gradient = torch.randn_like(values)
    reference = XIELU().to(device="cuda", dtype=torch.bfloat16)
    optimized = XIELU(backend="cuda").to(device="cuda", dtype=torch.bfloat16)
    optimized.load_state_dict(reference.state_dict())
    fused_fallback = XIELU(backend="fused").to(
        device="cuda", dtype=torch.bfloat16
    )
    fused_fallback.load_state_dict(reference.state_dict())

    expected = reference(values)
    expected.backward(gradient)
    expected_input_gradient = values.grad.detach().clone()
    values.grad = None
    actual = optimized(values)
    actual.backward(gradient)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(values.grad, expected_input_gradient, rtol=0, atol=0)
    torch.testing.assert_close(
        optimized.alpha_p.grad, reference.alpha_p.grad, rtol=0, atol=0
    )
    torch.testing.assert_close(
        optimized.alpha_n.grad, reference.alpha_n.grad, rtol=0, atol=0
    )
    torch.testing.assert_close(fused_fallback(values.detach()), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_fused_linear_epilogue_matches_standalone_xielu() -> None:
    try:
        import triton  # noqa: F401
        import xielu  # noqa: F401
    except ImportError:
        pytest.skip("optional Triton/xIELU CUDA dependencies are unavailable")
    from models.fused_xielu import (
        _triton_linear_xielu,
        fused_xielu_channel_mix,
    )

    torch.manual_seed(11)
    rows, width, expanded, output_width = 256, 64, 128, 64
    source = torch.randn(
        rows, width, device="cuda", dtype=torch.bfloat16
    ).requires_grad_()
    key_weight = torch.randn(
        expanded, width, device="cuda", dtype=torch.bfloat16
    ).requires_grad_()
    value_weight = torch.randn(
        output_width, expanded, device="cuda", dtype=torch.bfloat16
    ).requires_grad_()
    activation = XIELU(backend="cuda").to(device="cuda", dtype=torch.bfloat16)

    reference_pre = F.linear(source, key_weight)
    reference_post = activation(reference_pre)
    fused_pair = _triton_linear_xielu(
        source,
        key_weight,
        activation.alpha_p,
        activation.alpha_n,
        beta=activation.beta,
        eps=-activation.eps,
    )
    assert fused_pair is not None
    for actual, expected in zip(fused_pair, (reference_pre, reference_post), strict=True):
        relative_error = (
            (actual.float() - expected.float()).norm()
            / expected.float().norm().clamp_min(1e-12)
        )
        assert relative_error < 2e-3

    gradient = torch.randn(
        rows, output_width, device="cuda", dtype=torch.bfloat16
    )
    reference_output = F.linear(reference_post, value_weight)
    reference_output.backward(gradient)
    reference_gradients = tuple(
        value.grad.detach().clone()
        for value in (
            source,
            key_weight,
            value_weight,
            activation.alpha_p,
            activation.alpha_n,
        )
    )

    for value in (
        source,
        key_weight,
        value_weight,
        activation.alpha_p,
        activation.alpha_n,
    ):
        value.grad = None
    fused_output = fused_xielu_channel_mix(
        source,
        key_weight,
        value_weight,
        activation.alpha_p,
        activation.alpha_n,
    )
    fused_output.backward(gradient)
    for actual, expected in zip(
        (
            source.grad,
            key_weight.grad,
            value_weight.grad,
            activation.alpha_p.grad,
            activation.alpha_n.grad,
        ),
        reference_gradients,
        strict=True,
    ):
        relative_error = (
            (actual.float() - expected.float()).norm()
            / expected.float().norm().clamp_min(1e-12)
        )
        assert relative_error < 2e-3


def test_xielu_changes_only_channel_activation_and_adds_two_scalars_per_layer() -> None:
    config = ModelConfig(
        vocab_size=32,
        width=32,
        layers=2,
        heads=2,
        mlp_multiple=2,
        max_sequence_length=16,
    )
    torch.manual_seed(123)
    baseline = RWKVLabLM(config)
    torch.manual_seed(123)
    xielu = RWKVLabLM(config, channel_activation="xielu")

    baseline_state = baseline.state_dict()
    xielu_state = xielu.state_dict()
    for name, value in baseline_state.items():
        torch.testing.assert_close(xielu_state[name], value, rtol=0, atol=0)
    extra = set(xielu_state) - set(baseline_state)
    assert extra == {
        "core.blocks.0.ffn.xielu.alpha_n",
        "core.blocks.0.ffn.xielu.alpha_p",
        "core.blocks.1.ffn.xielu.alpha_n",
        "core.blocks.1.ffn.xielu.alpha_p",
    }
    assert count_parameters(xielu)["total"] == count_parameters(baseline)["total"] + 4


def test_xielu_rwkv_parallel_and_recurrent_paths_are_finite(monkeypatch) -> None:
    monkeypatch.setenv("RWKV8_FORCE_PYREF", "1")
    config = ModelConfig(
        vocab_size=32,
        width=32,
        layers=2,
        heads=2,
        mlp_multiple=2,
        max_sequence_length=16,
    )
    model = RWKVLabLM(config, channel_activation="xielu")
    token_ids = torch.tensor([[1, 2, 3, 4]])

    model.train()
    parallel = model(token_ids)
    model.eval()
    recurrent = model(token_ids)

    assert torch.isfinite(parallel.logits).all()
    assert torch.isfinite(recurrent.logits).all()
    assert recurrent.state is not None


def test_hidden_states_skips_head_without_changing_parallel_hidden(monkeypatch) -> None:
    monkeypatch.setenv("RWKV8_FORCE_PYREF", "1")
    config = ModelConfig(
        vocab_size=32,
        width=32,
        layers=2,
        heads=2,
        mlp_multiple=2,
        max_sequence_length=16,
    )
    model = RWKVLabLM(config, channel_activation="xielu")
    token_ids = torch.tensor([[1, 2, 3, 4]])

    model.train()
    expected = model(token_ids).hidden
    actual = model.hidden_states(token_ids)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
