"""Auditable RWKV-v4-style reference model.

The WKV recurrence follows the established per-channel exponentially decayed
weighted-value formulation and has exactly matching full-sequence and recurrent
paths. The Python time loop favors correctness over kernel speed; serious runs
should substitute a validated fused WKV kernel without changing this contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .common import LMOutput, ModelConfig, RMSNorm, initialize_weights


@dataclass
class RWKVLayerState:
    time_previous: torch.Tensor
    time_numerator: torch.Tensor
    time_denominator: torch.Tensor
    channel_previous: torch.Tensor


@dataclass
class RWKVState:
    layers: tuple[RWKVLayerState, ...]

    def flattened(self) -> torch.Tensor:
        values = []
        for layer in self.layers:
            values.extend(
                (
                    layer.time_previous,
                    layer.time_numerator,
                    layer.time_denominator,
                    layer.channel_previous,
                )
            )
        return torch.cat(values, dim=-1)


class TimeMix(nn.Module):
    def __init__(self, width: int, layer_index: int, layer_count: int) -> None:
        super().__init__()
        ratio = layer_index / max(layer_count - 1, 1)
        coordinate = torch.linspace(0.0, 1.0, width)
        self.mix_key = nn.Parameter(coordinate.pow(1.0 - 0.3 * ratio))
        self.mix_value = nn.Parameter(coordinate.pow(1.0 - 0.3 * ratio) + 0.05 * ratio)
        self.mix_receptance = nn.Parameter(coordinate.pow(0.5 * (1.0 - ratio)))
        self.time_decay = nn.Parameter(torch.linspace(-3.0, 0.0, width))
        self.time_first = nn.Parameter(torch.full((width,), math_log(0.3)))
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.receptance = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)

    def step(
        self,
        x: torch.Tensor,
        previous: torch.Tensor,
        numerator: torch.Tensor,
        denominator: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        key_input = x * self.mix_key + previous * (1.0 - self.mix_key)
        value_input = x * self.mix_value + previous * (1.0 - self.mix_value)
        receptance_input = x * self.mix_receptance + previous * (
            1.0 - self.mix_receptance
        )
        key = self.key(key_input).float().clamp(-30.0, 30.0)
        value = self.value(value_input).float()
        receptance = torch.sigmoid(self.receptance(receptance_input).float())

        current_weight = torch.exp(key)
        output_weight = torch.exp((key + self.time_first.float()).clamp(-30.0, 30.0))
        wkv = (numerator.float() + output_weight * value) / (
            denominator.float() + output_weight + 1e-9
        )
        decay = torch.exp(-torch.exp(self.time_decay.float()))
        next_numerator = decay * numerator.float() + current_weight * value
        next_denominator = decay * denominator.float() + current_weight
        update = self.output((receptance * wkv).to(dtype=x.dtype))
        return update, x, next_numerator, next_denominator


class ChannelMix(nn.Module):
    def __init__(
        self, width: int, layer_index: int, layer_count: int, multiple: int
    ) -> None:
        super().__init__()
        ratio = layer_index / max(layer_count - 1, 1)
        coordinate = torch.linspace(0.0, 1.0, width)
        self.mix_key = nn.Parameter(coordinate.pow(1.0 - 0.3 * ratio))
        self.mix_receptance = nn.Parameter(coordinate.pow(0.5 * (1.0 - ratio)))
        hidden = multiple * width
        self.key = nn.Linear(width, hidden, bias=False)
        self.value = nn.Linear(hidden, width, bias=False)
        self.receptance = nn.Linear(width, width, bias=False)

    def step(
        self, x: torch.Tensor, previous: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_input = x * self.mix_key + previous * (1.0 - self.mix_key)
        receptance_input = x * self.mix_receptance + previous * (
            1.0 - self.mix_receptance
        )
        key = F.relu(self.key(key_input)).square()
        update = torch.sigmoid(self.receptance(receptance_input)) * self.value(key)
        return update, x


class RWKVBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_index: int) -> None:
        super().__init__()
        self.time_norm = RMSNorm(config.width)
        self.time_mix = TimeMix(config.width, layer_index, config.layers)
        self.channel_norm = RMSNorm(config.width)
        self.channel_mix = ChannelMix(
            config.width, layer_index, config.layers, config.mlp_multiple
        )

    def step(
        self, x: torch.Tensor, state: RWKVLayerState
    ) -> tuple[torch.Tensor, RWKVLayerState]:
        normalized = self.time_norm(x)
        update, time_previous, numerator, denominator = self.time_mix.step(
            normalized,
            state.time_previous,
            state.time_numerator,
            state.time_denominator,
        )
        x = x + update
        normalized = self.channel_norm(x)
        update, channel_previous = self.channel_mix.step(
            normalized, state.channel_previous
        )
        return x + update, RWKVLayerState(
            time_previous=time_previous,
            time_numerator=numerator,
            time_denominator=denominator,
            channel_previous=channel_previous,
        )


class RWKVLM(nn.Module):
    architecture = "rwkv"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.input_norm = RMSNorm(config.width)
        self.blocks = nn.ModuleList(
            RWKVBlock(config, index) for index in range(config.layers)
        )
        self.norm = RMSNorm(config.width)
        self.output = nn.Linear(config.width, config.vocab_size, bias=False)
        self.apply(initialize_weights)
        if config.tie_embeddings:
            self.output.weight = self.token_embedding.weight

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> RWKVState:
        layers = []
        for _ in self.blocks:
            value = torch.zeros(
                batch_size, self.config.width, device=device, dtype=dtype
            )
            accumulator = torch.zeros(
                batch_size, self.config.width, device=device, dtype=torch.float32
            )
            layers.append(
                RWKVLayerState(value, accumulator, accumulator.clone(), value.clone())
            )
        return RWKVState(tuple(layers))

    @staticmethod
    def _reset(state: RWKVLayerState, mask: torch.Tensor) -> RWKVLayerState:
        keep = (~mask).to(dtype=state.time_previous.dtype)[:, None]
        keep_float = keep.float()
        return RWKVLayerState(
            state.time_previous * keep,
            state.time_numerator * keep_float,
            state.time_denominator * keep_float,
            state.channel_previous * keep,
        )

    def step(
        self,
        input_embedding: torch.Tensor,
        state: RWKVState | None = None,
        *,
        reset_mask: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> LMOutput:
        if input_embedding.ndim != 2:
            raise ValueError("input_embedding must have shape [batch, width]")
        if state is None:
            state = self.initial_state(
                input_embedding.shape[0], input_embedding.device, input_embedding.dtype
            )
        if len(state.layers) != len(self.blocks):
            raise ValueError("state layer count does not match model")
        x = self.input_norm(input_embedding)
        next_layers = []
        layer_norms = []
        state_norms = []
        channel_state_norms = []
        for block, layer_state in zip(self.blocks, state.layers, strict=True):
            if reset_mask is not None:
                layer_state = self._reset(layer_state, reset_mask)
            x, layer_state = block.step(x, layer_state)
            next_layers.append(layer_state)
            if return_diagnostics:
                layer_norms.append(float(x.detach().float().norm(dim=-1).mean()))
                state_norms.append(
                    float(layer_state.time_numerator.detach().norm(dim=-1).mean())
                )
                channel_state_norms.append(
                    float(
                        layer_state.channel_previous.detach()
                        .float()
                        .norm(dim=-1)
                        .mean()
                    )
                )
        hidden = self.norm(x)
        logits = self.output(hidden)
        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                "layer_representation_norm": layer_norms,
                "time_mix_state_norm": state_norms,
                "channel_mix_state_norm": channel_state_norms,
                "recurrent_state_norm": float(
                    RWKVState(tuple(next_layers))
                    .flattened()
                    .detach()
                    .norm(dim=-1)
                    .mean()
                ),
                "output_entropy": float(
                    torch.distributions.Categorical(logits=logits.detach())
                    .entropy()
                    .mean()
                ),
            }
        return LMOutput(
            logits=logits[:, None, :],
            hidden=hidden[:, None, :],
            state=RWKVState(tuple(next_layers)),
            diagnostics=diagnostics,
        )

    def forward(
        self,
        token_ids: torch.Tensor | None = None,
        *,
        input_embeddings: torch.Tensor | None = None,
        state: RWKVState | None = None,
        reset_mask: torch.Tensor | None = None,
        return_diagnostics: bool = False,
        **_: Any,
    ) -> LMOutput:
        if (token_ids is None) == (input_embeddings is None):
            raise ValueError("provide exactly one of token_ids or input_embeddings")
        embeddings = (
            self.token_embedding(token_ids)
            if input_embeddings is None
            else input_embeddings
        )
        if embeddings.ndim != 3:
            raise ValueError("model input must have shape [batch, sequence, width]")
        if reset_mask is not None and reset_mask.shape != embeddings.shape[:2]:
            raise ValueError("reset_mask must have shape [batch, sequence]")
        if state is None:
            state = self.initial_state(
                embeddings.shape[0], embeddings.device, embeddings.dtype
            )
        logits, hidden = [], []
        diagnostics: dict[str, list[Any]] = {
            "layer_representation_norm": [],
            "time_mix_state_norm": [],
            "channel_mix_state_norm": [],
            "recurrent_state_norm": [],
            "output_entropy": [],
        }
        for index in range(embeddings.shape[1]):
            output = self.step(
                embeddings[:, index],
                state,
                reset_mask=None if reset_mask is None else reset_mask[:, index],
                return_diagnostics=return_diagnostics,
            )
            state = output.state
            logits.append(output.logits)
            hidden.append(output.hidden)
            if return_diagnostics:
                assert output.diagnostics is not None
                for key, value in output.diagnostics.items():
                    diagnostics[key].append(value)
        summarized = None
        if return_diagnostics:
            summarized = {}
            for key, values in diagnostics.items():
                if not values:
                    summarized[key] = []
                elif isinstance(values[0], list):
                    summarized[key] = [
                        sum(float(row[index]) for row in values) / len(values)
                        for index in range(len(values[0]))
                    ]
                else:
                    summarized[key] = sum(float(value) for value in values) / len(
                        values
                    )
        return LMOutput(
            logits=torch.cat(logits, dim=1),
            hidden=torch.cat(hidden, dim=1),
            state=state,
            diagnostics=summarized,
        )


def math_log(value: float) -> float:
    # Kept local so module import does not depend on NumPy.
    import math

    return math.log(value)
