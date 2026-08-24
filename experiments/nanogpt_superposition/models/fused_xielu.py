"""Fused RWKV ChannelMix up-projection and xIELU epilogue.

The Triton forward kernel computes ``x @ key_weight.T`` and applies xIELU
while the GEMM accumulator tile is still resident.  It stores both the BF16
pre-activation (needed by xIELU's backward) and activated value (needed by the
down-projection backward).  Unsupported shapes use the qualified standalone
xIELU CUDA operation, so recurrent inference and small tests remain correct.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - optional CUDA dependency
    triton = tl = None
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _linear_xielu_pointer_kernel(
        input_ptr,
        weight_ptr,
        pre_ptr,
        post_ptr,
        alpha_p_ptr,
        alpha_n_ptr,
        M,
        N,
        K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
        BETA: tl.constexpr,
        EPS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        tiles_m = tl.cdiv(M, BLOCK_M)
        tiles_n = tl.cdiv(N, BLOCK_N)
        tiles_per_group = GROUP_M * tiles_n
        first_m = (pid // tiles_per_group) * GROUP_M
        group_size_m = tl.minimum(tiles_m - first_m, GROUP_M)
        pid_m = first_m + (pid % group_size_m)
        pid_n = (pid % tiles_per_group) // group_size_m

        offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_k = tl.arange(0, BLOCK_K)
        input_offsets = offsets_m[:, None] * K + offsets_k[None, :]
        weight_offsets = offsets_n[:, None] * K + offsets_k[None, :]
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_offset in range(0, K, BLOCK_K):
            input_block = tl.load(
                input_ptr + input_offsets,
                mask=offsets_m[:, None] < M,
                other=0.0,
            )
            weight_block = tl.load(
                weight_ptr + weight_offsets,
                mask=offsets_n[:, None] < N,
                other=0.0,
            )
            accumulator = tl.dot(input_block, weight_block.T, accumulator)
            input_offsets += BLOCK_K
            weight_offsets += BLOCK_K

        raw_alpha_p = tl.load(alpha_p_ptr).to(tl.float32)
        raw_alpha_n = tl.load(alpha_n_ptr).to(tl.float32)
        alpha_p = tl.where(
            raw_alpha_p > 20.0,
            raw_alpha_p,
            tl.where(
                raw_alpha_p < -20.0,
                0.0,
                tl.log(1.0 + tl.exp(raw_alpha_p)),
            ),
        )
        softplus_alpha_n = tl.where(
            raw_alpha_n > 20.0,
            raw_alpha_n,
            tl.where(
                raw_alpha_n < -20.0,
                0.0,
                tl.log(1.0 + tl.exp(raw_alpha_n)),
            ),
        )
        alpha_n = BETA + softplus_alpha_n
        pre = accumulator.to(tl.bfloat16)
        value = pre.to(tl.float32)
        positive = value * (alpha_p * value + BETA)
        negative = (
            alpha_n * (tl.exp(tl.minimum(value, EPS)) - 1.0)
            - softplus_alpha_n * value
        )
        activated = tl.where(value > 0.0, positive, negative)
        output_offsets = offsets_m[:, None] * N + offsets_n[None, :]
        output_mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
        tl.store(pre_ptr + output_offsets, pre, mask=output_mask)
        tl.store(
            post_ptr + output_offsets,
            activated.to(tl.bfloat16),
            mask=output_mask,
        )

def _triton_linear_xielu(
    value: torch.Tensor,
    weight: torch.Tensor,
    alpha_p: torch.Tensor,
    alpha_n: torch.Tensor,
    *,
    beta: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return fused ``(pre, post)`` or ``None`` for the safe fallback."""
    block_m = 128
    block_n = 128
    block_k = 32
    if (
        not _HAS_TRITON
        or not value.is_cuda
        or value.ndim != 2
        or weight.ndim != 2
        or value.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or alpha_p.dtype != torch.bfloat16
        or alpha_n.dtype != torch.bfloat16
        or value.shape[1] != weight.shape[1]
        or value.shape[1] % block_k
    ):
        return None

    rows, contraction = value.shape
    columns = weight.shape[0]
    pre = torch.empty(rows, columns, device=value.device, dtype=value.dtype)
    post = torch.empty_like(pre)
    grid = (triton.cdiv(rows, block_m) * triton.cdiv(columns, block_n),)
    _linear_xielu_pointer_kernel[grid](
        value,
        weight,
        pre,
        post,
        alpha_p,
        alpha_n,
        rows,
        columns,
        contraction,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        BETA=beta,
        EPS=eps,
        num_stages=3,
        num_warps=8,
    )
    return pre, post


class _FusedXIELUChannelMix(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        value,
        key_weight,
        value_weight,
        alpha_p,
        alpha_n,
        beta,
        eps,
    ):
        original_shape = value.shape
        # A custom autograd function is opaque to autocast. Mirror BF16
        # ``F.linear`` autocast explicitly, while returning the input gradient
        # in the caller's original dtype.
        ctx.input_dtype = value.dtype
        flat = (
            value.to(dtype=key_weight.dtype)
            .reshape(-1, original_shape[-1])
            .contiguous()
        )
        fused = _triton_linear_xielu(
            flat,
            key_weight,
            alpha_p,
            alpha_n,
            beta=float(beta),
            eps=float(eps),
        )
        ctx.used_fused_epilogue = fused is not None
        if fused is None:
            pre = F.linear(flat, key_weight)
            post = torch.ops.xielu.forward(
                pre.contiguous(), alpha_p, alpha_n, float(beta), float(eps)
            )
        else:
            pre, post = fused
        output = F.linear(post, value_weight)
        ctx.save_for_backward(
            flat, key_weight, value_weight, pre, post, alpha_p, alpha_n
        )
        ctx.original_shape = original_shape
        ctx.beta = float(beta)
        ctx.eps = float(eps)
        return output.reshape(*original_shape[:-1], value_weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        (
            flat,
            key_weight,
            value_weight,
            pre,
            post,
            alpha_p,
            alpha_n,
        ) = ctx.saved_tensors
        grad = (
            grad_output.to(dtype=value_weight.dtype)
            .reshape(-1, grad_output.shape[-1])
            .contiguous()
        )
        grad_value_weight = grad.transpose(0, 1) @ post
        grad_post = (grad @ value_weight).contiguous()
        grad_pre, grad_alpha_p, grad_alpha_n = torch.ops.xielu.backward(
            pre.contiguous(),
            grad_post,
            alpha_p,
            alpha_n,
            ctx.beta,
            ctx.eps,
        )
        grad_key_weight = grad_pre.transpose(0, 1) @ flat
        grad_input = grad_pre @ key_weight
        return (
            grad_input.reshape(ctx.original_shape).to(dtype=ctx.input_dtype),
            grad_key_weight,
            grad_value_weight,
            grad_alpha_p,
            grad_alpha_n,
            None,
            None,
        )


def fused_xielu_channel_mix(
    value: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    alpha_p: torch.Tensor,
    alpha_n: torch.Tensor,
    *,
    beta: float = 0.5,
    eps: float = -1e-6,
) -> torch.Tensor:
    """Execute ``Linear -> xIELU -> Linear`` with a fused first epilogue."""
    return _FusedXIELUChannelMix.apply(
        value,
        key_weight,
        value_weight,
        alpha_p,
        alpha_n,
        beta,
        eps,
    )


__all__ = ["fused_xielu_channel_mix", "_triton_linear_xielu"]
