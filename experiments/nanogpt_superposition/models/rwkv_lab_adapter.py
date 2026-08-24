"""Adapter for moe-mla's established RWKV-7 training implementation.

The experiment remains in ztok, while the actual RWKV block and recurrent
state semantics come from the local `rwkv_lab.rwkv_pretrain.RWKV7Small` used by
the existing trainer/dashboard. Set `MOE_MLA_ROOT` when the repository is not
at `/thearray/git/moe-mla`.
"""

from __future__ import annotations

import math
import os
import sys
from types import MethodType
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .common import LMOutput, ModelConfig


def _load_rwkv7_small():
    root = Path(os.environ.get("MOE_MLA_ROOT", "/thearray/git/moe-mla"))
    source = root / "src"
    if not source.exists():
        raise RuntimeError(
            f"moe-mla source not found at {source}; set MOE_MLA_ROOT or use rwkv_backend=reference"
        )
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from rwkv_lab.rwkv_pretrain import RWKV7Small

    return RWKV7Small


class XIELU(nn.Module):
    """Paper-reference xIELU with per-layer trainable curvature scalars.

    Huang and Schlag initialize both effective alphas to 0.8, constrain them
    with softplus, and fix beta at 0.5.  Float32 elementwise arithmetic mirrors
    the optimized CUDA reference, which loads BF16 values and computes in
    float before writing BF16 outputs.
    """

    def __init__(
        self,
        alpha_p_init: float = 0.8,
        alpha_n_init: float = 0.8,
        beta: float = 0.5,
        eps: float = 1e-6,
        backend: str = "reference",
    ) -> None:
        super().__init__()
        if backend not in {"reference", "cuda", "fused"}:
            raise ValueError("xIELU backend must be reference, cuda, or fused")
        if alpha_p_init <= 0 or alpha_n_init <= beta:
            raise ValueError("xIELU initial alphas must satisfy alpha_p>0 and alpha_n>beta")
        self.alpha_p = nn.Parameter(
            torch.tensor([math.log(math.expm1(alpha_p_init))])
        )
        self.alpha_n = nn.Parameter(
            torch.tensor([math.log(math.expm1(alpha_n_init - beta))])
        )
        self.beta = beta
        self.eps = eps
        self.backend = backend
        if backend in {"cuda", "fused"}:
            try:
                from xielu import xielu as xielu_cuda
            except ImportError as error:
                raise RuntimeError(
                    "xielu_cuda requires the pinned nathanrchn/kernels extension"
                ) from error
            self._cuda_operation = xielu_cuda

    def effective_alphas(self) -> tuple[torch.Tensor, torch.Tensor]:
        return F.softplus(self.alpha_p.float()), self.beta + F.softplus(
            self.alpha_n.float()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self.backend in {"cuda", "fused"}
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and self.alpha_p.dtype == torch.bfloat16
            and self.alpha_n.dtype == torch.bfloat16
            and x.numel() % 128 == 0
        ):
            return self._cuda_operation(
                x.contiguous(),
                self.alpha_p,
                self.alpha_n,
                self.beta,
                -self.eps,
            )
        x_float = x.float()
        alpha_p, alpha_n = self.effective_alphas()
        positive = alpha_p * x_float.square() + self.beta * x_float
        negative = (
            alpha_n * torch.expm1(torch.clamp_max(x_float, -self.eps))
            - alpha_n * x_float
            + self.beta * x_float
        )
        return torch.where(x_float > 0, positive, negative).to(dtype=x.dtype)


def _xielu_channel_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
    cache_params=None,
    cache_position=None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    hidden_gate: torch.Tensor | None = None,
    shift_state: torch.Tensor | None = None,
    reset_mask: torch.Tensor | None = None,
    return_state: bool = False,
    **kwargs: Any,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """ChannelMix forward with only ReLU-squared replaced by xIELU."""
    if cache_params is not None:
        has_previous = getattr(cache_params, "has_previous_state", None)
        if callable(has_previous) and has_previous():
            raise NotImplementedError(
                "xIELU ChannelMix supports explicit recurrent state, not cache_params"
            )
    prepared_mixed = kwargs.pop("_megakernel_channel_mix", None)
    if prepared_mixed is not None:
        if prepared_mixed.shape != hidden_states.shape:
            raise ValueError("prepared ChannelMix input must match hidden states")
        mixed = prepared_mixed
    else:
        previous = torch.zeros_like(hidden_states)
        if shift_state is not None:
            previous[:, :1] = shift_state.to(hidden_states.dtype).reshape(
                hidden_states.shape[0], 1, hidden_states.shape[-1]
            )
        if hidden_states.shape[1] > 1:
            previous[:, 1:] = hidden_states[:, :-1]
        if reset_mask is not None:
            if (
                reset_mask.shape != hidden_states.shape[:2]
                or not torch.all(reset_mask[:, 0])
            ):
                raise ValueError(
                    "reset_mask must be [batch,time] and reset the first token"
                )
            previous = previous.masked_fill(reset_mask[..., None], 0.0)
        xk = module.x_k.to(dtype=hidden_states.dtype).view(1, 1, -1)
        mixed = hidden_states + (previous - hidden_states) * xk
    if module.xielu.backend == "fused" and hidden_gate is None:
        from .fused_xielu import fused_xielu_channel_mix

        output = fused_xielu_channel_mix(
            mixed,
            module.key.weight,
            module.value.weight,
            module.xielu.alpha_p,
            module.xielu.alpha_n,
            beta=module.xielu.beta,
            eps=-module.xielu.eps,
        )
    else:
        activated = module.xielu(module.key(mixed))
        if hidden_gate is not None:
            activated = activated * hidden_gate
        output = module.value(activated)
    if return_state:
        return output, hidden_states[:, -1:]
    return output


def _install_xielu_channel_mix(core: nn.Module, *, backend: str) -> int:
    installed = 0
    for block in core.blocks:
        channel_mix = block.ffn
        required = ("x_k", "key", "value", "ffn_hidden_size")
        if not all(hasattr(channel_mix, name) for name in required):
            raise TypeError("xIELU requires native RWKV ChannelMix blocks")
        channel_mix.xielu = XIELU(backend=backend)
        channel_mix.forward = MethodType(_xielu_channel_forward, channel_mix)
        installed += 1
    return installed


@dataclass
class RWKVLabState:
    blocks: tuple[dict[str, Any], ...]

    def flattened(self) -> torch.Tensor:
        tensors: list[torch.Tensor] = []

        def visit(value: Any) -> None:
            if isinstance(value, torch.Tensor):
                tensor = value.float().reshape(value.shape[0], -1)
                tensors.append(tensor)
            elif isinstance(value, dict):
                for key in sorted(value):
                    visit(value[key])
            elif isinstance(value, (list, tuple)):
                for item in value:
                    visit(item)

        visit(self.blocks)
        if not tensors:
            raise ValueError("RWKV-Lab state contained no tensors")
        return torch.cat(tensors, dim=-1)


class RWKVLabLM(nn.Module):
    architecture = "rwkv"
    backend = "rwkv_lab_rwkv7"

    def __init__(
        self,
        config: ModelConfig,
        *,
        channel_activation: str = "relu_squared",
    ) -> None:
        super().__init__()
        config.validate()
        if channel_activation not in {
            "relu_squared",
            "xielu",
            "xielu_cuda",
            "xielu_fused",
        }:
            raise ValueError("unsupported RWKV channel activation")
        self.config = config
        self.channel_activation = channel_activation
        rwkv7_small = _load_rwkv7_small()
        self.core = rwkv7_small(
            config.vocab_size,
            config.width,
            config.layers,
            config.width // config.heads,
            {},
            ffn_hidden=config.mlp_multiple * config.width,
        )
        if config.tie_embeddings:
            self.core.head.weight = self.core.emb.weight
        self.xielu_layer_count = (
            _install_xielu_channel_mix(
                self.core,
                backend=(
                    "fused"
                    if channel_activation == "xielu_fused"
                    else "cuda"
                    if channel_activation == "xielu_cuda"
                    else "reference"
                ),
            )
            if channel_activation in {"xielu", "xielu_cuda", "xielu_fused"}
            else 0
        )

    @property
    def token_embedding(self) -> nn.Embedding:
        return self.core.emb

    @property
    def output(self) -> nn.Linear:
        return self.core.head

    def hidden_states(
        self,
        token_ids: torch.Tensor | None = None,
        *,
        input_embeddings: torch.Tensor | None = None,
        reset_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the post-norm LM hidden states without projecting logits."""
        if (token_ids is None) == (input_embeddings is None):
            raise ValueError("provide exactly one of token_ids or input_embeddings")
        if token_ids is not None:
            return self.core(
                token_ids,
                hidden_only=True,
                reset_mask=reset_mask,
            )
        assert input_embeddings is not None
        x = input_embeddings
        value_first = None
        for block in self.core.blocks:
            x, value_first = block(
                x,
                value_first,
                ids=None,
                e0=None,
                reset_mask=reset_mask,
            )
        return self.core.ln_out(x)

    def _standard_embeddings(
        self,
        embeddings: torch.Tensor,
        reset_mask: torch.Tensor | None,
    ) -> LMOutput:
        x = embeddings
        value_first = None
        for block in self.core.blocks:
            x, value_first = block(
                x,
                value_first,
                ids=None,
                e0=None,
                reset_mask=reset_mask,
            )
        hidden = self.core.ln_out(x)
        return LMOutput(logits=self.core.head(hidden), hidden=hidden)

    def _recurrent_embeddings(
        self,
        embeddings: torch.Tensor,
        state: RWKVLabState | None,
    ) -> LMOutput:
        states = state.blocks if state is not None else (None,) * len(self.core.blocks)
        if len(states) != len(self.core.blocks):
            raise ValueError("RWKV-Lab state depth mismatch")
        x = embeddings
        value_first = None
        next_states = []
        for block, block_state in zip(self.core.blocks, states, strict=True):
            x, value_first, block_state = block.forward_recurrent(
                x,
                value_first,
                block_state,
                ids=None,
                e0=None,
            )
            next_states.append(block_state)
        hidden = self.core.ln_out(x)
        return LMOutput(
            logits=self.core.head(hidden),
            hidden=hidden,
            state=RWKVLabState(tuple(next_states)),
        )

    @staticmethod
    def _value_norm(value: Any) -> float:
        tensors: list[torch.Tensor] = []

        def visit(item: Any) -> None:
            if isinstance(item, torch.Tensor):
                tensors.append(item.detach().float().reshape(item.shape[0], -1))
            elif isinstance(item, dict):
                for key in sorted(item):
                    visit(item[key])
            elif isinstance(item, (list, tuple)):
                for child in item:
                    visit(child)

        visit(value)
        if not tensors:
            return 0.0
        return float(torch.cat(tensors, dim=-1).norm(dim=-1).mean())

    def forward(
        self,
        token_ids: torch.Tensor | None = None,
        *,
        input_embeddings: torch.Tensor | None = None,
        state: RWKVLabState | None = None,
        reset_mask: torch.Tensor | None = None,
        return_diagnostics: bool = False,
        **_: Any,
    ) -> LMOutput:
        if (token_ids is None) == (input_embeddings is None):
            raise ValueError("provide exactly one of token_ids or input_embeddings")
        if token_ids is not None and state is None and self.training:
            logits, hidden = self.core(
                token_ids,
                return_hidden=True,
                reset_mask=reset_mask,
            )
            return LMOutput(logits=logits, hidden=hidden)
        embeddings = (
            self.token_embedding(token_ids)
            if input_embeddings is None
            else input_embeddings
        )
        if embeddings.ndim != 3:
            raise ValueError("model input must have shape [batch, sequence, width]")
        # Training uses the established parallel scan. Evaluation uses the
        # exact recurrent-state API so branch-equivalence can inspect state.
        output = (
            self._standard_embeddings(embeddings, reset_mask)
            if self.training and state is None
            else self._recurrent_embeddings(embeddings, state)
        )
        if return_diagnostics:
            diagnostic_state = output.state
            if diagnostic_state is None:
                # The established parallel training scan does not return its
                # final state by default. Probe the same embeddings through
                # the exact recurrent API only at sparse telemetry steps.
                with torch.no_grad():
                    diagnostic_state = self._recurrent_embeddings(
                        embeddings.detach(), None
                    ).state
            assert diagnostic_state is not None
            output.diagnostics = {
                "backend": self.backend,
                "recurrent_state_norm": float(
                    diagnostic_state.flattened().norm(dim=-1).mean()
                ),
                "time_mix_state_norm": [
                    self._value_norm(layer.get("wkv"))
                    for layer in diagnostic_state.blocks
                ],
                "channel_mix_state_norm": [
                    self._value_norm(layer.get("ffn_shift"))
                    for layer in diagnostic_state.blocks
                ],
                "layer_representation_norm": [
                    float(output.hidden.detach().float().norm(dim=-1).mean())
                ],
            }
        return output
