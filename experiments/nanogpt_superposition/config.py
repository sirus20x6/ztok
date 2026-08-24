"""Versioned experiment configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from .models.common import ModelConfig
except ImportError:
    from models.common import ModelConfig


@dataclass(frozen=True)
class TrainingConfig:
    representation: str = "ordinary"
    group_size: int = 1
    fusion: str = "norm_preserving_mean"
    target: str = "bag"
    schedule: str = "50/25/25"
    batch_size: int = 16
    batch_size_schedule: tuple[dict[str, Any], ...] = ()
    train_shape_schedule: tuple[dict[str, Any], ...] = ()
    context_tokens: int = 256
    context_source_bytes_target: int | None = None
    max_steps: int = 100
    budget_kind: str = "steps"
    budget_value: float | None = None
    eval_interval: int = 25
    eval_batches: int = 8
    checkpoint_interval: int = 50
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 10
    lr_schedule_basis: str = "steps"
    lr_decay_kind: str = "cosine"
    cooldown_fraction: float = 0.2
    head_split_fraction: float | None = None
    coarse_lr_multiplier: float = 1.0
    ordinary_lr_multiplier: float = 1.0
    coarse_superposed_source_gradient_multiplier: float = 1.0
    mixed_superposed_source_gradient_multiplier: float = 1.0
    weight_decay: float = 0.1
    optimizer_foreach: bool = False
    optimizer_fused: bool = False
    optimizer_kind: str = "adamw"
    spectral_muon: dict[str, Any] | None = None
    spectral_muon_apply_recovery_multiplier: bool = False
    prefetch_batches: int = 0
    synchronize_each_step: bool = True
    telemetry_interval: int = 1
    compile_model: bool = False
    fused_linear_cross_entropy: bool = False
    grad_clip: float = 1.0
    mixed_precision: bool = False
    device: str = "cpu"
    soft_token_rate: float = 0.0
    primer_path: str | None = None
    primer_epochs: int = 0
    primer_batch_size: int = 256
    primer_eval_interval: int = 100

    def validate(self) -> None:
        if self.representation not in {"ordinary", "fixed"}:
            raise ValueError("representation must be ordinary or fixed")
        if self.representation == "fixed" and self.group_size not in {2, 4, 8, 16}:
            raise ValueError("fixed group_size must be 2, 4, 8, or 16")
        if self.representation == "ordinary" and self.group_size != 1:
            raise ValueError("ordinary representation requires group_size=1")
        if self.target not in {"bag", "ordered"}:
            raise ValueError("target must be bag or ordered")
        if self.budget_kind not in {"steps", "source_bytes", "flops", "gpu_seconds"}:
            raise ValueError("unsupported budget_kind")
        if self.budget_kind != "steps" and (
            self.budget_value is None or self.budget_value <= 0
        ):
            raise ValueError("non-step budgets require a positive budget_value")
        if self.lr_schedule_basis not in {"steps", "budget"}:
            raise ValueError("lr_schedule_basis must be steps or budget")
        if self.lr_decay_kind not in {"cosine", "late_linear"}:
            raise ValueError("lr_decay_kind must be cosine or late_linear")
        if (
            not isinstance(self.cooldown_fraction, (int, float))
            or isinstance(self.cooldown_fraction, bool)
            or not 0.0 < float(self.cooldown_fraction) <= 1.0
        ):
            raise ValueError("cooldown_fraction must be in (0, 1]")
        if self.lr_decay_kind == "late_linear" and self.lr_schedule_basis != "budget":
            raise ValueError("late_linear decay requires lr_schedule_basis=budget")
        if self.head_split_fraction is not None and (
            not isinstance(self.head_split_fraction, (int, float))
            or isinstance(self.head_split_fraction, bool)
            or not 0.0 < float(self.head_split_fraction) < 1.0
        ):
            raise ValueError("head_split_fraction must be in (0, 1)")
        if self.coarse_lr_multiplier <= 0:
            raise ValueError("coarse_lr_multiplier must be positive")
        if self.ordinary_lr_multiplier <= 0:
            raise ValueError("ordinary_lr_multiplier must be positive")
        if self.coarse_superposed_source_gradient_multiplier <= 0:
            raise ValueError(
                "coarse_superposed_source_gradient_multiplier must be positive"
            )
        if self.mixed_superposed_source_gradient_multiplier <= 0:
            raise ValueError(
                "mixed_superposed_source_gradient_multiplier must be positive"
            )
        if not isinstance(self.optimizer_foreach, bool) or not isinstance(
            self.optimizer_fused, bool
        ):
            raise TypeError("optimizer foreach and fused controls must be boolean")
        if self.optimizer_foreach and self.optimizer_fused:
            raise ValueError("optimizer foreach and fused controls are exclusive")
        if self.optimizer_kind not in {"adamw", "spectral_muon"}:
            raise ValueError("optimizer_kind must be adamw or spectral_muon")
        if self.optimizer_kind == "spectral_muon":
            if not isinstance(self.spectral_muon, dict):
                raise TypeError("spectral_muon optimizer requires its resolved configuration")
            if self.optimizer_foreach or self.optimizer_fused:
                raise ValueError("SpectralMuon does not use AdamW foreach/fused controls")
        elif self.spectral_muon is not None:
            raise ValueError("spectral_muon configuration requires optimizer_kind=spectral_muon")
        if not isinstance(self.spectral_muon_apply_recovery_multiplier, bool):
            raise TypeError("spectral_muon_apply_recovery_multiplier must be boolean")
        if (
            not isinstance(self.prefetch_batches, int)
            or isinstance(self.prefetch_batches, bool)
            or self.prefetch_batches not in {0, 1}
        ):
            raise ValueError("prefetch_batches must be zero or one")
        if not isinstance(self.synchronize_each_step, bool):
            raise TypeError("synchronize_each_step must be boolean")
        if (
            not isinstance(self.telemetry_interval, int)
            or isinstance(self.telemetry_interval, bool)
            or not 1 <= self.telemetry_interval <= 1024
        ):
            raise ValueError("telemetry_interval must be between 1 and 1,024")
        if not isinstance(self.compile_model, bool):
            raise TypeError("compile_model must be boolean")
        if not isinstance(self.fused_linear_cross_entropy, bool):
            raise TypeError("fused_linear_cross_entropy must be boolean")
        if self.max_steps <= 0 or self.batch_size <= 0 or self.context_tokens < 2:
            raise ValueError("batch, context, and max_steps must be positive")
        if not isinstance(self.batch_size_schedule, (list, tuple)):
            raise TypeError("batch_size_schedule must be a sequence of stages")
        previous_fraction = -1.0
        for index, stage in enumerate(self.batch_size_schedule):
            if not isinstance(stage, dict) or set(stage) != {
                "start_fraction",
                "batch_size",
            }:
                raise ValueError(
                    "batch_size_schedule stages require only start_fraction and batch_size"
                )
            fraction = stage["start_fraction"]
            batch_size = stage["batch_size"]
            if (
                not isinstance(fraction, (int, float))
                or isinstance(fraction, bool)
                or not 0.0 <= float(fraction) < 1.0
            ):
                raise ValueError("batch schedule fractions must be in [0, 1)")
            if (
                not isinstance(batch_size, int)
                or isinstance(batch_size, bool)
                or not 1 <= batch_size <= 512
            ):
                raise ValueError("scheduled batch sizes must be between 1 and 512")
            if float(fraction) <= previous_fraction:
                raise ValueError("batch schedule fractions must be strictly increasing")
            if index == 0 and float(fraction) != 0.0:
                raise ValueError("batch_size_schedule must start at fraction zero")
            if index == 0 and batch_size != self.batch_size:
                raise ValueError("the first scheduled batch must equal batch_size")
            previous_fraction = float(fraction)
        if self.batch_size_schedule and self.train_shape_schedule:
            raise ValueError(
                "batch_size_schedule and train_shape_schedule are mutually exclusive"
            )
        if not isinstance(self.train_shape_schedule, (list, tuple)):
            raise TypeError("train_shape_schedule must be a sequence of stages")
        previous_fraction = -1.0
        previous_context = 0
        for index, stage in enumerate(self.train_shape_schedule):
            if not isinstance(stage, dict) or set(stage) != {
                "start_fraction",
                "batch_size",
                "context_tokens",
            }:
                raise ValueError(
                    "train_shape_schedule stages require only start_fraction, "
                    "batch_size, and context_tokens"
                )
            fraction = stage["start_fraction"]
            batch_size = stage["batch_size"]
            context_tokens = stage["context_tokens"]
            if (
                not isinstance(fraction, (int, float))
                or isinstance(fraction, bool)
                or not 0.0 <= float(fraction) < 1.0
            ):
                raise ValueError("shape schedule fractions must be in [0, 1)")
            if (
                not isinstance(batch_size, int)
                or isinstance(batch_size, bool)
                or not 1 <= batch_size <= 512
            ):
                raise ValueError("scheduled batch sizes must be between 1 and 512")
            if (
                not isinstance(context_tokens, int)
                or isinstance(context_tokens, bool)
                or context_tokens < 2
            ):
                raise ValueError("scheduled contexts must be at least two tokens")
            if float(fraction) <= previous_fraction:
                raise ValueError("shape schedule fractions must be strictly increasing")
            if index == 0 and float(fraction) != 0.0:
                raise ValueError("train_shape_schedule must start at fraction zero")
            if context_tokens < previous_context:
                raise ValueError("scheduled contexts must be non-decreasing")
            if self.representation == "fixed" and context_tokens % self.group_size:
                raise ValueError(
                    "scheduled contexts must be divisible by fixed group_size"
                )
            previous_fraction = float(fraction)
            previous_context = context_tokens
        if self.representation == "fixed" and self.context_tokens % self.group_size:
            raise ValueError("context_tokens must be divisible by group_size")
        if not 0 <= self.soft_token_rate <= 1:
            raise ValueError("soft_token_rate must be in [0,1]")
        if self.primer_epochs < 0:
            raise ValueError("primer_epochs must be non-negative")
        if self.primer_epochs and not self.primer_path:
            raise ValueError("primer_path is required when primer_epochs is positive")
        if self.primer_batch_size <= 0 or self.primer_eval_interval <= 0:
            raise ValueError("primer batch size and eval interval must be positive")

    def batch_size_at(self, progress: float, total_budget: float) -> int:
        if total_budget <= 0:
            raise ValueError("batch schedule requires a positive total budget")
        fraction = min(max(float(progress) / float(total_budget), 0.0), 1.0)
        selected = self.batch_size
        for stage in self.batch_size_schedule:
            if fraction < float(stage["start_fraction"]):
                break
            selected = int(stage["batch_size"])
        return selected

    def training_shape_at(
        self, progress: float, total_budget: float
    ) -> tuple[int, int]:
        if total_budget <= 0:
            raise ValueError("shape schedule requires a positive total budget")
        if not self.train_shape_schedule:
            return self.batch_size_at(progress, total_budget), self.context_tokens
        fraction = min(max(float(progress) / float(total_budget), 0.0), 1.0)
        selected = self.train_shape_schedule[0]
        for stage in self.train_shape_schedule:
            if fraction < float(stage["start_fraction"]):
                break
            selected = stage
        return int(selected["batch_size"]), int(selected["context_tokens"])


@dataclass(frozen=True)
class ExperimentConfig:
    schema: str
    run_name: str
    seed: int
    architecture: str
    data_dir: str
    output_dir: str
    model: ModelConfig
    training: TrainingConfig
    tokenizer_condition: str = "ordinary_bpe"
    rwkv_backend: str = "rwkv_lab"
    rwkv_channel_activation: str = "relu_squared"
    notes: str = ""

    def validate(self) -> None:
        if self.schema != "ztok.nanogpt_superposition.config.v1":
            raise ValueError("unsupported experiment config schema")
        if self.architecture not in {"transformer", "rwkv"}:
            raise ValueError("architecture must be transformer or rwkv")
        if self.rwkv_backend not in {"rwkv_lab", "reference"}:
            raise ValueError("rwkv_backend must be rwkv_lab or reference")
        if self.rwkv_channel_activation not in {
            "relu_squared",
            "xielu",
            "xielu_cuda",
            "xielu_fused",
        }:
            raise ValueError(
                "rwkv_channel_activation must be relu_squared, xielu, "
                "xielu_cuda, or xielu_fused"
            )
        if self.architecture != "rwkv" and self.rwkv_channel_activation != "relu_squared":
            raise ValueError("RWKV channel activations require architecture=rwkv")
        if self.rwkv_backend != "rwkv_lab" and self.rwkv_channel_activation != "relu_squared":
            raise ValueError("xIELU currently requires rwkv_backend=rwkv_lab")
        if self.tokenizer_condition not in {"ordinary_bpe", "superbpe", "byte_debug"}:
            raise ValueError("unsupported tokenizer_condition")
        self.model.validate()
        self.training.validate()
        if self.training.context_tokens > self.model.max_sequence_length:
            raise ValueError("training context exceeds model max_sequence_length")
        for stage in self.training.train_shape_schedule:
            if int(stage["context_tokens"]) > self.model.max_sequence_length:
                raise ValueError(
                    "scheduled context exceeds model max_sequence_length"
                )
        if self.training.head_split_fraction is not None:
            if not self.model.tie_embeddings:
                raise ValueError("late head splitting requires tied embeddings")
            if self.architecture != "rwkv" or self.rwkv_backend != "rwkv_lab":
                raise ValueError("late head splitting requires the RWKV-Lab backend")
            if self.training.optimizer_kind != "adamw":
                raise ValueError("late head splitting currently requires AdamW")
            if self.training.compile_model:
                raise ValueError("late head splitting is incompatible with static compilation")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: Path | str) -> ExperimentConfig:
    raw = json.loads(Path(path).read_text())
    model = ModelConfig(**raw.pop("model"))
    training = TrainingConfig(**raw.pop("training"))
    config = ExperimentConfig(model=model, training=training, **raw)
    config.validate()
    return config
