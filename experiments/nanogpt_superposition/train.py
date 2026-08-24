#!/usr/bin/env python3
"""Matched-budget Transformer/RWKV training harness."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from config import ExperimentConfig, load_config
from dataset import DeterministicBatchPrefetcher, PackedCorpus
from models.common import (
    count_parameters,
    cross_entropy_bits_per_byte,
    estimate_training_flops,
)
from models.rwkv import RWKVLM
from models.rwkv_lab_adapter import RWKVLabLM
from models.transformer import TransformerLM
from superposition.fixed_groups import FixedObjective, audit_ztok_fixed_plan
from superposition.recovery import RecoverySchedule
from superposition.soft_tokens import SoftTokenMetadata, teacher_forced_soft_corruption
from superposition.statement_primer import (
    collate_statement_primer_batch,
    load_statement_expansions,
    statement_primer_batches,
    statement_primer_objective,
)


class ProfileStepNotifier:
    """Child-side bridge to TrainVM's authority-owned step profiler."""

    def __init__(self) -> None:
        step_fd = os.environ.get("TRAINVM_PROFILE_STEP_FD")
        acknowledgement_fd = os.environ.get("TRAINVM_PROFILE_ACK_FD")
        raw_boundaries = os.environ.get("TRAINVM_PROFILE_SYNC_COUNTS", "")
        raw_total = os.environ.get("TRAINVM_PROFILE_TOTAL_COUNT")
        if (
            step_fd is None
            and acknowledgement_fd is None
            and not raw_boundaries
            and raw_total is None
        ):
            self._step_fd = None
            self._acknowledgement_fd = None
            self._boundaries: frozenset[int] = frozenset()
            self._count = 0
            self._total = 0
            return
        if step_fd is None or acknowledgement_fd is None or raw_total is None:
            raise RuntimeError("TrainVM profile step bridge is incomplete")
        self._step_fd = int(step_fd)
        self._acknowledgement_fd = int(acknowledgement_fd)
        self._boundaries = frozenset(
            int(value) for value in raw_boundaries.split(",") if value
        )
        if any(value <= 0 for value in self._boundaries):
            raise RuntimeError("TrainVM profile synchronization counts must be positive")
        self._count = 0
        self._total = int(raw_total)
        if self._total <= 0 or any(value > self._total for value in self._boundaries):
            raise RuntimeError("TrainVM profile total count is invalid")

    def step(self, optimizer_step: int) -> None:
        if self._step_fd is None:
            return
        if self._count >= self._total:
            return
        self._count += 1
        os.write(self._step_fd, f"{optimizer_step}\n".encode("ascii"))
        if self._count in self._boundaries:
            assert self._acknowledgement_fd is not None
            if os.read(self._acknowledgement_fd, 1) != b"1":
                raise RuntimeError("TrainVM profile boundary acknowledgement failed")


def git_metadata() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    fingerprint = hashlib.sha256()
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=root,
        capture_output=True,
        check=True,
    ).stdout
    fingerprint.update(diff)
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=root,
        capture_output=True,
        check=True,
    ).stdout.split(b"\0")
    for encoded_path in sorted(path for path in untracked if path):
        path = root / os.fsdecode(encoded_path)
        if not path.is_file():
            continue
        fingerprint.update(encoded_path)
        fingerprint.update(b"\0")
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                fingerprint.update(chunk)
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "working_tree_sha256": fingerprint.hexdigest(),
    }


def make_model(config: ExperimentConfig) -> torch.nn.Module:
    if config.architecture == "transformer":
        return TransformerLM(config.model)
    return (
        RWKVLabLM(
            config.model,
            channel_activation=config.rwkv_channel_activation,
        )
        if config.rwkv_backend == "rwkv_lab"
        else RWKVLM(config.model)
    )


def make_optimizer(
    model: torch.nn.Module,
    auxiliary_parameters: list[torch.nn.Parameter],
    config: ExperimentConfig,
) -> tuple[torch.optim.Optimizer, list[torch.nn.Parameter]]:
    parameters = list(model.parameters()) + auxiliary_parameters
    training = config.training
    if training.optimizer_kind == "adamw":
        return (
            torch.optim.AdamW(
                parameters,
                lr=training.learning_rate,
                weight_decay=training.weight_decay,
                betas=(0.9, 0.95),
                foreach=training.optimizer_foreach,
                fused=training.optimizer_fused,
            ),
            parameters,
        )

    if not isinstance(model, RWKVLabLM) or training.spectral_muon is None:
        raise TypeError("SpectralMuon requires the RWKV-Lab model backend")
    from rwkv_lab.spectral_muon import SpectralMuon

    resolved = dict(training.spectral_muon)
    matrix_learning_rate = float(resolved.pop("learning_rate"))
    fallback_multiplier = float(resolved.pop("fallback_multiplier"))
    if "adam_beta1" in resolved or "adam_beta2" in resolved:
        if "adam_beta1" not in resolved or "adam_beta2" not in resolved:
            raise ValueError("resolved SpectralMuon Adam betas are incomplete")
        resolved["adam_betas"] = (
            float(resolved.pop("adam_beta1")),
            float(resolved.pop("adam_beta2")),
        )
    if matrix_learning_rate != training.learning_rate:
        raise ValueError("SpectralMuon schedule peak disagrees with resolved matrix LR")
    excluded = {id(model.token_embedding.weight), id(model.output.weight)}
    matrix_parameters: list[torch.nn.Parameter] = []
    fallback_parameters: list[torch.nn.Parameter] = []
    for parameter in model.parameters():
        if (
            id(parameter) not in excluded
            and parameter.ndim == 2
            and min(parameter.shape) > 1
        ):
            matrix_parameters.append(parameter)
        else:
            fallback_parameters.append(parameter)
    fallback_parameters.extend(auxiliary_parameters)
    groups = [
        {
            "params": matrix_parameters,
            "lr": matrix_learning_rate,
            "schedule_multiplier": 1.0,
            "use_muon": True,
        },
        {
            "params": fallback_parameters,
            "lr": matrix_learning_rate * fallback_multiplier,
            "schedule_multiplier": fallback_multiplier,
            "use_muon": False,
        },
    ]
    return SpectralMuon(groups, weight_decay=0.0, **resolved), parameters


def split_tied_output_head(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    parameters: list[torch.nn.Parameter],
) -> bool:
    """Untie an RWKV-Lab output head without discarding learned Adam state."""
    if not isinstance(model, RWKVLabLM):
        raise TypeError("late head splitting requires the RWKV-Lab model")
    source = model.token_embedding.weight
    if model.output.weight is not source:
        return False
    output_weight = torch.nn.Parameter(
        source.detach().clone(), requires_grad=source.requires_grad
    )
    model.output.weight = output_weight
    matching_groups = [
        group
        for group in optimizer.param_groups
        if any(parameter is source for parameter in group["params"])
    ]
    if len(matching_groups) != 1:
        raise RuntimeError("tied embedding parameter must belong to one optimizer group")
    matching_groups[0]["params"].append(output_weight)
    if source in optimizer.state:
        optimizer.state[output_weight] = {
            key: value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
            for key, value in optimizer.state[source].items()
        }
    parameters.append(output_weight)
    return True


def output_head_is_split(model: torch.nn.Module) -> bool:
    return (
        isinstance(model, RWKVLabLM)
        and model.output.weight is not model.token_embedding.weight
    )


def ordinary_objective(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    *,
    soft_rate: float,
    corruption_generator: torch.Generator,
    soft_metadata: SoftTokenMetadata | None,
    fused_linear_cross_entropy: bool = False,
) -> tuple[torch.Tensor, int, int, dict[str, Any]]:
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    diagnostics: dict[str, Any] = {"soft_token_rate": 0.0}
    if soft_rate:
        corruption = teacher_forced_soft_corruption(
            inputs,
            model.token_embedding.weight,
            rate=soft_rate,
            generator=corruption_generator,
            metadata=soft_metadata,
        )
        model_input = {"input_embeddings": corruption.embeddings}
        diagnostics = {
            "soft_token_rate": float(corruption.replaced.float().mean()),
            "mean_soft_cluster_size": corruption.mean_cluster_size,
        }
    else:
        model_input = {"token_ids": inputs}
    if fused_linear_cross_entropy:
        if not isinstance(model, RWKVLabLM):
            raise TypeError(
                "fused_linear_cross_entropy currently requires the RWKV-Lab backend"
            )
        from rwkv_lab.fused_ce import lmhead_cross_entropy

        hidden = model.hidden_states(**model_input)
        loss = lmhead_cross_entropy(hidden, model.output, targets, fused=True)
    else:
        output = model(**model_input)
        loss = F.cross_entropy(output.logits.flatten(0, 1), targets.flatten())
    return loss, inputs.numel(), targets.numel(), diagnostics


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    corpus: PackedCorpus,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    generator = torch.Generator().manual_seed(config.seed + 991)
    losses = []
    predicted_tokens = 0
    source_bytes = 0
    for _ in range(config.training.eval_batches):
        batch = corpus.sample(
            config.training.batch_size,
            config.training.context_tokens + 1,
            generator=generator,
            device=device,
        )
        inputs, targets = batch.token_ids[:, :-1], batch.token_ids[:, 1:]
        autocast = (
            torch.autocast(device_type=device.type, dtype=torch.bfloat16)
            if config.training.mixed_precision and device.type in {"cuda", "cpu"}
            else nullcontext()
        )
        with autocast:
            output = model(inputs)
            loss = F.cross_entropy(output.logits.flatten(0, 1), targets.flatten())
        losses.append(float(loss))
        predicted_tokens += targets.numel()
        source_bytes += batch.source_bytes
    model.train()
    loss = sum(losses) / len(losses)
    return {
        "ordinary_validation_loss": loss,
        "ordinary_validation_perplexity": math.exp(min(loss, 20)),
        "validation_bits_per_byte": cross_entropy_bits_per_byte(
            loss, predicted_tokens, source_bytes
        ),
        "validation_source_bytes": source_bytes,
    }


def _budget_progress(
    kind: str,
    *,
    step: int,
    source_bytes: int,
    flops: int,
    elapsed: float,
) -> float:
    return {
        "steps": float(step),
        "source_bytes": float(source_bytes),
        "flops": float(flops),
        "gpu_seconds": float(elapsed),
    }[kind]


def _total_budget(config: ExperimentConfig) -> float:
    return (
        float(config.training.max_steps)
        if config.training.budget_kind == "steps"
        else float(config.training.budget_value)
    )


def _learning_rate(
    config: ExperimentConfig,
    step: int,
    progress: float | None = None,
    total_budget: float | None = None,
) -> float:
    training = config.training
    if step < training.warmup_steps:
        return training.learning_rate * (step + 1) / max(training.warmup_steps, 1)
    if training.lr_schedule_basis == "budget":
        if progress is None or total_budget is None:
            raise ValueError("budget-relative LR requires progress and total_budget")
        ratio = min(max(progress / total_budget, 0.0), 1.0)
    else:
        ratio = min(
            (step - training.warmup_steps)
            / max(training.max_steps - training.warmup_steps, 1),
            1.0,
        )
    if training.lr_decay_kind == "late_linear":
        cooldown_start = 1.0 - float(training.cooldown_fraction)
        if ratio < cooldown_start:
            return training.learning_rate
        cooldown_progress = (ratio - cooldown_start) / max(
            float(training.cooldown_fraction), 1.0e-12
        )
        multiplier = 1.0 - min(max(cooldown_progress, 0.0), 1.0)
    else:
        multiplier = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return training.min_learning_rate + multiplier * (
        training.learning_rate - training.min_learning_rate
    )


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    fixed: torch.nn.Module | None,
    soft_metadata: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    counters: dict[str, Any],
    generator: torch.Generator,
    corruption_generator: torch.Generator,
    tokenizer: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "ztok.nanogpt_superposition.checkpoint.v1",
            "config": config.to_dict(),
            "model": model.state_dict(),
            "fixed_objective": fixed.state_dict() if fixed is not None else None,
            "soft_metadata": soft_metadata.state_dict()
            if soft_metadata is not None
            else None,
            "optimizer": optimizer.state_dict(),
            "counters": counters,
            "sampling_generator": generator.get_state(),
            "corruption_generator": corruption_generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "tokenizer": tokenizer,
            **git_metadata(),
        },
        path,
    )
    step = int(counters["step"])
    sidecar = path.parent / f"step_{step:06d}"
    sidecar.mkdir(parents=True, exist_ok=True)
    (sidecar / "config.json").write_text(
        json.dumps(
            {
                **config.to_dict(),
                "parameters": count_parameters(model),
                "source_bytes": counters["source_bytes"],
                "model_positions": counters["model_positions"],
                "estimated_flops": counters["estimated_flops"],
            },
            indent=2,
        )
        + "\n"
    )
    dashboard_checkpoint = sidecar / "ckpt.pt"
    if dashboard_checkpoint.exists():
        dashboard_checkpoint.unlink()
    try:
        os.link(path, dashboard_checkpoint)
    except OSError:
        shutil.copy2(path, dashboard_checkpoint)


def _cpu_rng_state(state: torch.Tensor) -> torch.Tensor:
    """PyTorch RNG restoration APIs require CPU byte tensors."""
    return state.detach().cpu()


def _restore_training_counters(counters: dict[str, Any], saved: dict[str, Any]) -> None:
    """Restore cumulative counters without leaking final-evaluation fields."""
    counters.update({key: saved[key] for key in counters if key in saved})
    if "ordinary_step" in counters and "ordinary_step" not in saved:
        counters["ordinary_step"] = saved["step"]


def _checkpoint_config_matches(
    saved: dict[str, Any], config: ExperimentConfig
) -> bool:
    """Allow pre-primer checkpoints to resume with new default-only fields."""
    normalized = json.loads(json.dumps(saved))
    training = normalized.get("training", {})
    defaults = {
        "primer_path": None,
        "primer_epochs": 0,
        "primer_batch_size": 256,
        "primer_eval_interval": 100,
        "batch_size_schedule": (),
        "train_shape_schedule": (),
        "lr_decay_kind": "cosine",
        "cooldown_fraction": 0.2,
        "head_split_fraction": None,
    }
    for key, value in defaults.items():
        training.setdefault(key, value)
    normalized.setdefault("rwkv_channel_activation", "relu_squared")
    current = json.loads(json.dumps(config.to_dict()))
    return normalized == current


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--extend-gpu-seconds-to",
        type=float,
        help="with --resume, continue to this cumulative device-time budget",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.extend_gpu_seconds_to is not None and args.resume is None:
        raise SystemExit("--extend-gpu-seconds-to requires --resume")
    config = load_config(args.config)
    profile_steps = ProfileStepNotifier()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed & 0xFFFFFFFF)
    random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device(config.training.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    train_corpus = PackedCorpus(config.data_dir, "train")
    validation_corpus = PackedCorpus(config.data_dir, "validation")
    vocabulary_size = train_corpus.metadata["tokenizer"]["vocabulary_size"]
    expected_tokenizer_kind = {
        "ordinary_bpe": "bpe",
        "superbpe": "superbpe",
        "byte_debug": "byte",
    }[config.tokenizer_condition]
    actual_tokenizer_kind = train_corpus.metadata["tokenizer"]["kind"]
    if actual_tokenizer_kind != expected_tokenizer_kind:
        raise SystemExit(
            f"config tokenizer condition {config.tokenizer_condition} expects "
            f"{expected_tokenizer_kind}, dataset reports {actual_tokenizer_kind}"
        )
    if config.model.vocab_size < vocabulary_size:
        raise SystemExit(
            f"model vocab {config.model.vocab_size} is smaller than dataset requirement {vocabulary_size}"
        )
    model = make_model(config)
    rwkv_lab_bfloat16 = (
        config.architecture == "rwkv"
        and config.rwkv_backend == "rwkv_lab"
        and config.training.mixed_precision
        and device.type == "cuda"
    )
    model = model.to(
        device=device,
        dtype=torch.bfloat16 if rwkv_lab_bfloat16 else None,
    )
    training_model = (
        torch.compile(model, dynamic=False, fullgraph=False)
        if config.training.compile_model
        else model
    )
    fixed = None
    if config.training.representation == "fixed":
        fixed = FixedObjective(
            config.model.width,
            config.model.vocab_size,
            config.training.group_size,
            config.training.fusion,
            config.training.target,
        ).to(device)
    soft_metadata = (
        SoftTokenMetadata(config.model.width).to(device)
        if config.training.soft_token_rate > 0
        else None
    )
    auxiliary_parameters = (list(fixed.parameters()) if fixed else []) + (
        list(soft_metadata.parameters()) if soft_metadata else []
    )
    optimizer, parameters = make_optimizer(model, auxiliary_parameters, config)
    parameter_counts = count_parameters(model)
    parameter_counts["auxiliary"] = (fixed.auxiliary_parameters() if fixed else 0) + (
        sum(value.numel() for value in soft_metadata.parameters())
        if soft_metadata
        else 0
    )
    parameter_counts["total_with_auxiliary"] = (
        parameter_counts["total"] + parameter_counts["auxiliary"]
    )
    if args.dry_run:
        print(
            json.dumps(
                {"config": config.to_dict(), "parameters": parameter_counts}, indent=2
            )
        )
        return

    primer_expansions = []
    primer_diagnostics = None
    if config.training.primer_epochs:
        import ztok

        tokenizer = train_corpus.metadata["tokenizer"]
        if tokenizer["path"] is None:
            raise SystemExit("the statement primer requires a persisted tokenizer")
        with ztok.Pipeline.from_path(tokenizer["path"]) as pipeline:
            primer_expansions, primer_diagnostics = load_statement_expansions(
                config.training.primer_path, pipeline
            )
        if not primer_expansions:
            raise SystemExit("the statement primer contains no usable expansions")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2) + "\n"
    )
    metadata = {
        "schema": "ztok.nanogpt_superposition.run.v1",
        "config": config.to_dict(),
        "parameters": parameter_counts,
        "dataset": train_corpus.metadata,
        **git_metadata(),
        "torch_version": torch.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else os.uname().machine,
        "parameter_dtype": str(next(model.parameters()).dtype),
        "autocast_dtype": (
            "torch.bfloat16" if config.training.mixed_precision else None
        ),
        "dashboard_contract": "moe-mla.trainboard.train_jsonl.v1",
        "statement_superposition_primer": primer_diagnostics,
        "continuation": (
            {
                "checkpoint": str(args.resume.resolve()),
                "extend_gpu_seconds_to": args.extend_gpu_seconds_to,
            }
            if args.resume
            else None
        ),
    }
    (output_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")

    generator = torch.Generator().manual_seed(config.seed + 1)
    corruption_generator = torch.Generator().manual_seed(config.seed + 2)
    counters: dict[str, Any] = {
        "step": 0,
        "ordinary_step": 0,
        "ordinary_examples": 0,
        "primer_step": 0,
        "source_bytes": 0,
        "primer_source_bytes": 0,
        "ordinary_token_equivalents": 0,
        "primer_token_equivalents": 0,
        "model_positions": 0,
        "primer_model_positions": 0,
        "estimated_flops": 0,
        "primer_estimated_flops": 0,
        "elapsed_seconds": 0.0,
        "primer_elapsed_seconds": 0.0,
    }
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if not _checkpoint_config_matches(checkpoint["config"], config):
            raise SystemExit("resume config differs from checkpoint")
        saved_head_split = (
            config.training.head_split_fraction is not None
            and _budget_progress(
                config.training.budget_kind,
                step=checkpoint["counters"].get("ordinary_step", 0),
                source_bytes=checkpoint["counters"].get("source_bytes", 0),
                flops=checkpoint["counters"].get("estimated_flops", 0),
                elapsed=checkpoint["counters"].get("elapsed_seconds", 0.0),
            )
            >= _total_budget(config) * config.training.head_split_fraction
        )
        if saved_head_split:
            if not split_tied_output_head(model, optimizer, parameters):
                raise RuntimeError("resume expected a tied model before topology restoration")
        model.load_state_dict(checkpoint["model"])
        if fixed is not None:
            fixed.load_state_dict(checkpoint["fixed_objective"])
        if soft_metadata is not None:
            soft_metadata.load_state_dict(checkpoint["soft_metadata"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        _restore_training_counters(counters, checkpoint["counters"])
        if "ordinary_examples" not in checkpoint["counters"]:
            counters["ordinary_examples"] = (
                counters["ordinary_step"] * config.training.batch_size
            )
        generator.set_state(_cpu_rng_state(checkpoint["sampling_generator"]))
        corruption_generator.set_state(
            _cpu_rng_state(checkpoint["corruption_generator"])
        )
        torch.set_rng_state(_cpu_rng_state(checkpoint["torch_rng"]))
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(
                [_cpu_rng_state(state) for state in checkpoint["cuda_rng"]]
            )
        if args.extend_gpu_seconds_to is not None:
            if config.training.lr_schedule_basis == "budget":
                raise SystemExit(
                    "cannot redefine a budget-relative LR schedule during extension; "
                    "declare the combined budget up front or use a step-relative schedule"
                )
            if args.extend_gpu_seconds_to <= counters["elapsed_seconds"]:
                raise SystemExit(
                    "extension target must exceed checkpoint cumulative GPU-seconds"
                )

    if fixed is not None:
        import ztok

        tokenizer = train_corpus.metadata["tokenizer"]
        pipeline = (
            ztok.Pipeline.byte_id()
            if tokenizer["path"] is None
            else ztok.Pipeline.from_path(tokenizer["path"])
        )
        with pipeline:
            metadata["ztok_plan_audit"] = audit_ztok_fixed_plan(
                pipeline, b"0123456789abcdef", config.training.group_size
            )

    schedule = RecoverySchedule.parse(config.training.schedule)
    budget_kind = (
        "gpu_seconds"
        if args.extend_gpu_seconds_to is not None
        else config.training.budget_kind
    )
    total_budget = (
        args.extend_gpu_seconds_to
        if args.extend_gpu_seconds_to is not None
        else _total_budget(config)
    )
    if budget_kind == "gpu_seconds" and not config.training.synchronize_each_step:
        raise SystemExit(
            "asynchronous step timing is incompatible with a gpu_seconds budget"
        )
    metrics_path = output_dir / "metrics.jsonl"
    dashboard_path = output_dir / "train.jsonl"
    start_wall = time.perf_counter()
    last_phase = None
    model.train()
    training_model.train()
    log_mode = "a" if args.resume else "w"
    with (
        metrics_path.open(log_mode, encoding="utf-8", buffering=1) as metrics,
        dashboard_path.open(log_mode, encoding="utf-8", buffering=1) as dashboard,
    ):
        dashboard.write(
            json.dumps(
                {
                    "kind": "startup",
                    "step": counters["step"],
                    "architecture": config.architecture,
                    "rwkv_backend": config.rwkv_backend,
                    "rwkv_channel_activation": config.rwkv_channel_activation,
                    "tokenizer": config.tokenizer_condition,
                    "representation": config.training.representation,
                    "group_size": config.training.group_size,
                    "fusion": config.training.fusion,
                    "target": config.training.target,
                    "params": parameter_counts["total_with_auxiliary"],
                    "resumed": bool(args.resume),
                }
            )
            + "\n"
        )
        primer_plan = []
        for epoch in range(config.training.primer_epochs):
            primer_plan.extend(
                statement_primer_batches(
                    primer_expansions,
                    batch_size=config.training.primer_batch_size,
                    seed=config.seed + 3000 + epoch,
                )
            )
        while counters["primer_step"] < len(primer_plan):
            records = primer_plan[counters["primer_step"]]
            primer_batch = collate_statement_primer_batch(records, device=device)
            optimizer.zero_grad(set_to_none=True)
            step_started = time.perf_counter()
            autocast = (
                torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                if config.training.mixed_precision and device.type in {"cuda", "cpu"}
                else nullcontext()
            )
            with autocast:
                primer_result = statement_primer_objective(model, primer_batch)
            primer_result.loss.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(parameters, config.training.grad_clip)
            )
            base_learning_rate = _learning_rate(config, counters["primer_step"])
            learning_rate_multiplier = config.training.coarse_lr_multiplier
            learning_rate = base_learning_rate * learning_rate_multiplier
            for group in optimizer.param_groups:
                group["lr"] = learning_rate * group.get(
                    "schedule_multiplier", 1.0
                )
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_seconds = time.perf_counter() - step_started
            primer_batch_size = primer_batch.token_ids.shape[0]
            positions_per_batch = (
                primer_result.model_positions // primer_batch_size
            )
            step_flops = estimate_training_flops(
                config.architecture,
                config.model,
                primer_batch_size,
                positions_per_batch,
            )
            counters["step"] += 1
            counters["primer_step"] += 1
            counters["primer_source_bytes"] += primer_batch.explicit_source_bytes
            counters["primer_token_equivalents"] += primer_batch.token_ids.numel()
            counters["primer_model_positions"] += primer_result.model_positions
            counters["primer_estimated_flops"] += step_flops
            counters["primer_elapsed_seconds"] += step_seconds
            row = {
                "schema": "ztok.nanogpt_superposition.metric.v1",
                **counters,
                "step_seconds": step_seconds,
                "architecture": config.architecture,
                "rwkv_channel_activation": config.rwkv_channel_activation,
                "tokenizer": config.tokenizer_condition,
                "phase": "primer",
                "lr_schedule_phase": "coarse",
                "mode": "statement_superposition",
                "group_size": None,
                "loss": float(primer_result.loss.detach()),
                "gradient_norm": gradient_norm,
                "base_learning_rate": base_learning_rate,
                "learning_rate_multiplier": learning_rate_multiplier,
                "learning_rate": learning_rate,
                "step_estimated_flops": step_flops,
                "primer_batch_size": primer_batch_size,
                "primer_batch_explicit_source_bytes": (
                    primer_batch.explicit_source_bytes
                ),
                "peak_vram_bytes": torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else 0,
                **primer_result.diagnostics,
            }
            if (
                counters["primer_step"] % config.training.primer_eval_interval == 0
                or counters["primer_step"] == 1
                or counters["primer_step"] == len(primer_plan)
            ):
                row.update(validate(model, validation_corpus, config, device))
            metrics.write(json.dumps(row) + "\n")
            dashboard.write(
                json.dumps(
                    {
                        "kind": "train",
                        "step": counters["step"],
                        "ordinary_step": counters["ordinary_step"],
                        "primer_step": counters["primer_step"],
                        "loss": float(primer_result.loss.detach()),
                        "lr": learning_rate,
                        "lr_multiplier": learning_rate_multiplier,
                        "gnorm": gradient_norm,
                        "tok_per_sec": int(
                            primer_batch.token_ids.numel()
                            / max(step_seconds, 1e-9)
                        ),
                        "phase": "primer",
                        "mode": "statement_superposition",
                        "primer_source_bytes": counters["primer_source_bytes"],
                        "primer_model_positions": counters["primer_model_positions"],
                        "primer_estimated_flops": counters[
                            "primer_estimated_flops"
                        ],
                        "peak_vram_bytes": row["peak_vram_bytes"],
                    }
                )
                + "\n"
            )
            if "ordinary_validation_loss" in row:
                dashboard.write(
                    json.dumps(
                        {
                            "kind": "eval",
                            "step": counters["step"],
                            "ordinary_step": counters["ordinary_step"],
                            "primer_step": counters["primer_step"],
                            "loss": row["ordinary_validation_loss"],
                            "val_loss": row["ordinary_validation_loss"],
                            "ppl": row["ordinary_validation_perplexity"],
                            "bits_per_byte": row["validation_bits_per_byte"],
                            "tokens": config.training.eval_batches
                            * config.training.batch_size
                            * config.training.context_tokens,
                            "source_bytes": row["validation_source_bytes"],
                            "phase": "primer",
                        }
                    )
                    + "\n"
                )
            print(json.dumps(row))

        if primer_plan and counters["ordinary_step"] == 0:
            _save_checkpoint(
                output_dir / "primer_checkpoint.pt",
                model,
                fixed,
                soft_metadata,
                optimizer,
                config,
                counters,
                generator,
                corruption_generator,
                train_corpus.metadata["tokenizer"],
            )
            dashboard.write(
                json.dumps(
                    {
                        "kind": "checkpoint",
                        "step": counters["step"],
                        "phase": "primer_complete",
                    }
                )
                + "\n"
            )

        active_batch_size, active_context_tokens = config.training.training_shape_at(
            0.0, total_budget
        )
        batch_prefetcher = (
            DeterministicBatchPrefetcher(
                train_corpus,
                batch_size=active_batch_size,
                sequence_tokens=active_context_tokens + 1,
                generator=generator,
                pin_memory=device.type == "cuda",
            )
            if config.training.prefetch_batches
            else None
        )
        telemetry_window_started = None
        telemetry_window_steps = 0
        while counters["ordinary_step"] < config.training.max_steps:
            progress = _budget_progress(
                budget_kind,
                step=counters["ordinary_step"],
                source_bytes=counters["source_bytes"],
                flops=counters["estimated_flops"],
                elapsed=counters["elapsed_seconds"],
            )
            if progress >= total_budget:
                break
            head_split_event = False
            if (
                config.training.head_split_fraction is not None
                and progress / total_budget >= config.training.head_split_fraction
            ):
                head_split_event = split_tied_output_head(
                    model, optimizer, parameters
                )
                if head_split_event:
                    updated_counts = count_parameters(model)
                    updated_counts["auxiliary"] = parameter_counts["auxiliary"]
                    updated_counts["total_with_auxiliary"] = (
                        updated_counts["total"] + updated_counts["auxiliary"]
                    )
                    parameter_counts.clear()
                    parameter_counts.update(updated_counts)
            scheduled_batch_size, scheduled_context_tokens = (
                config.training.training_shape_at(
                    progress, total_budget
                )
            )
            if (
                scheduled_batch_size != active_batch_size
                or scheduled_context_tokens != active_context_tokens
            ):
                if batch_prefetcher is not None:
                    batch_prefetcher.close()
                    batch_prefetcher = DeterministicBatchPrefetcher(
                        train_corpus,
                        batch_size=scheduled_batch_size,
                        sequence_tokens=scheduled_context_tokens + 1,
                        generator=generator,
                        pin_memory=device.type == "cuda",
                    )
                active_batch_size = scheduled_batch_size
                active_context_tokens = scheduled_context_tokens
            if batch_prefetcher is not None:
                random_value, batch = batch_prefetcher.next(device)
            else:
                random_value = float(torch.rand((), generator=generator))
                batch = train_corpus.sample(
                    active_batch_size,
                    active_context_tokens + 1,
                    generator=generator,
                    device=device,
                )
            superposed = fixed is not None and schedule.use_superposition(
                progress, total_budget, random_value
            )
            phase = (
                schedule.phase(progress, total_budget)
                if fixed is not None
                else "ordinary"
            )
            optimizer.zero_grad(set_to_none=True)
            step_started = time.perf_counter()
            if telemetry_window_steps == 0:
                telemetry_window_started = step_started
            autocast = (
                torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                if config.training.mixed_precision and device.type in {"cuda", "cpu"}
                else nullcontext()
            )
            with autocast:
                if superposed:
                    assert fixed is not None
                    want_diagnostics = (
                        counters["ordinary_step"]
                        % config.training.eval_interval
                        == 0
                        or phase != last_phase
                    )
                    result = fixed(
                        training_model,
                        batch.token_ids,
                        source_gradient_multiplier=(
                            config.training.coarse_superposed_source_gradient_multiplier
                            if phase == "coarse"
                            else config.training.mixed_superposed_source_gradient_multiplier
                            if phase == "mixed"
                            else 1.0
                        ),
                        return_diagnostics=want_diagnostics,
                    )
                    loss = result.loss
                    model_positions = result.model_positions
                    target_tokens = result.target_tokens
                    detail = result.diagnostics
                else:
                    fraction = min(progress / total_budget, 1.0)
                    if fraction < 0.20:
                        soft_rate = 0.0
                    elif fraction < 0.50:
                        soft_rate = config.training.soft_token_rate * 0.25
                    elif fraction < 0.80:
                        soft_rate = config.training.soft_token_rate * 0.50
                    else:
                        soft_rate = config.training.soft_token_rate
                    loss, model_positions, target_tokens, detail = ordinary_objective(
                        training_model,
                        batch.token_ids,
                        soft_rate=soft_rate,
                        corruption_generator=corruption_generator,
                        soft_metadata=soft_metadata,
                        fused_linear_cross_entropy=(
                            config.training.fused_linear_cross_entropy
                        ),
                    )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters, config.training.grad_clip
            )
            base_learning_rate = _learning_rate(
                config,
                counters["ordinary_token_equivalents"]
                / (
                    config.training.batch_size
                    * (config.training.context_tokens + 1)
                ),
                progress,
                total_budget,
            )
            lr_schedule_phase = schedule.phase(progress, total_budget)
            learning_rate_multiplier = schedule.learning_rate_multiplier(
                progress,
                total_budget,
                config.training.coarse_lr_multiplier,
                config.training.ordinary_lr_multiplier,
            )
            if (
                config.training.optimizer_kind == "spectral_muon"
                and not config.training.spectral_muon_apply_recovery_multiplier
            ):
                learning_rate_multiplier = 1.0
            learning_rate = base_learning_rate * learning_rate_multiplier
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.step()
            positions_per_batch = model_positions // active_batch_size
            learned_flops = (
                fixed.learned_fusion_flops(
                    active_batch_size, positions_per_batch
                )
                if superposed and fixed is not None
                else 0
            )
            step_flops = estimate_training_flops(
                config.architecture,
                config.model,
                active_batch_size,
                positions_per_batch,
                learned_fusion_flops=learned_flops,
            )
            completed_ordinary_step = counters["ordinary_step"] + 1
            phase_transition = phase != last_phase
            next_progress = _budget_progress(
                budget_kind,
                step=completed_ordinary_step,
                source_bytes=counters["source_bytes"] + batch.source_bytes,
                flops=counters["estimated_flops"] + step_flops,
                elapsed=counters["elapsed_seconds"],
            )
            evaluation_due = (
                completed_ordinary_step % config.training.eval_interval == 0
                or completed_ordinary_step == 1
            )
            checkpoint_due = (
                completed_ordinary_step % config.training.checkpoint_interval == 0
            )
            terminal_due = (
                completed_ordinary_step >= config.training.max_steps
                or next_progress >= total_budget
            )
            telemetry_window_steps += 1
            telemetry_due = (
                config.training.synchronize_each_step
                or telemetry_window_steps >= config.training.telemetry_interval
                or evaluation_due
                or checkpoint_due
                or terminal_due
                or phase_transition
                or head_split_event
            )
            if config.training.synchronize_each_step:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                step_seconds = time.perf_counter() - step_started
                elapsed_increment = step_seconds
                telemetry_window_steps = 0
                telemetry_window_started = None
            elif telemetry_due:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                if telemetry_window_started is None:
                    raise RuntimeError("telemetry window lost its start time")
                elapsed_increment = time.perf_counter() - telemetry_window_started
                step_seconds = elapsed_increment / telemetry_window_steps
                telemetry_window_steps = 0
                telemetry_window_started = None
            else:
                elapsed_increment = 0.0
                step_seconds = 0.0
            counters["step"] += 1
            counters["ordinary_step"] += 1
            counters["ordinary_examples"] += active_batch_size
            counters["source_bytes"] += batch.source_bytes
            counters["ordinary_token_equivalents"] += batch.token_ids.numel()
            counters["model_positions"] += model_positions
            counters["estimated_flops"] += step_flops
            counters["elapsed_seconds"] += elapsed_increment
            profile_steps.step(counters["ordinary_step"])
            if not telemetry_due:
                last_phase = phase
                continue
            loss_value = float(loss.detach())
            gradient_norm_value = float(gradient_norm.detach())
            row = {
                "schema": "ztok.nanogpt_superposition.metric.v1",
                **counters,
                "step_seconds": step_seconds,
                "batch_source_bytes": batch.source_bytes,
                "batch_size": active_batch_size,
                "context_tokens": active_context_tokens,
                "mean_context_source_bytes": batch.source_bytes
                / active_batch_size,
                "architecture": config.architecture,
                "rwkv_channel_activation": config.rwkv_channel_activation,
                "tokenizer": config.tokenizer_condition,
                "phase": phase,
                "lr_schedule_phase": lr_schedule_phase,
                "mode": "fixed" if superposed else "ordinary",
                "group_size": config.training.group_size if superposed else 1,
                "loss": loss_value,
                "training_bits_per_byte": cross_entropy_bits_per_byte(
                    loss_value, target_tokens, batch.source_bytes
                ),
                "gradient_norm": gradient_norm_value,
                "base_learning_rate": base_learning_rate,
                "learning_rate_multiplier": learning_rate_multiplier,
                "learning_rate": learning_rate,
                "step_estimated_flops": step_flops,
                "positions_per_source_token": model_positions / batch.token_ids.numel(),
                "peak_vram_bytes": torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else 0,
                "phase_transition": phase_transition,
                "head_split_active": output_head_is_split(model),
                "head_split_event": head_split_event,
                **detail,
            }
            last_phase = phase
            if evaluation_due:
                row.update(validate(model, validation_corpus, config, device))
            metrics.write(json.dumps(row) + "\n")
            metrics.flush()
            dashboard.write(
                json.dumps(
                    {
                        "kind": "train",
                        "step": counters["step"],
                        "ordinary_step": counters["ordinary_step"],
                        "ordinary_examples": counters["ordinary_examples"],
                        "primer_step": counters["primer_step"],
                        "loss": loss_value,
                        "lr": learning_rate,
                        "lr_multiplier": learning_rate_multiplier,
                        "gnorm": gradient_norm_value,
                        "tok_per_sec": int(
                            batch.token_ids.numel() / max(step_seconds, 1e-9)
                        ),
                        "bytes_per_sec": batch.source_bytes / max(step_seconds, 1e-9),
                        "batch_size": active_batch_size,
                        "context_tokens": active_context_tokens,
                        "bits_per_byte": row["training_bits_per_byte"],
                        "source_bytes": counters["source_bytes"],
                        "mean_context_source_bytes": row["mean_context_source_bytes"],
                        "model_positions": counters["model_positions"],
                        "estimated_flops": counters["estimated_flops"],
                        "phase": phase,
                        "lr_schedule_phase": lr_schedule_phase,
                        "mode": row["mode"],
                        "rwkv_channel_activation": config.rwkv_channel_activation,
                        "group_size": row["group_size"],
                        "peak_vram_bytes": row["peak_vram_bytes"],
                        "head_split_active": row["head_split_active"],
                        "head_split_event": row["head_split_event"],
                    }
                )
                + "\n"
            )
            if "ordinary_validation_loss" in row:
                dashboard.write(
                    json.dumps(
                        {
                            "kind": "eval",
                            "step": counters["step"],
                            "ordinary_step": counters["ordinary_step"],
                            "primer_step": counters["primer_step"],
                            "loss": row["ordinary_validation_loss"],
                            "val_loss": row["ordinary_validation_loss"],
                            "ppl": row["ordinary_validation_perplexity"],
                            "bits_per_byte": row["validation_bits_per_byte"],
                            "tokens": config.training.eval_batches
                            * config.training.batch_size
                            * config.training.context_tokens,
                            "source_bytes": row["validation_source_bytes"],
                        }
                    )
                    + "\n"
                )
            print(json.dumps(row))
            if checkpoint_due:
                _save_checkpoint(
                    output_dir / "checkpoint.pt",
                    model,
                    fixed,
                    soft_metadata,
                    optimizer,
                    config,
                    counters,
                    generator,
                    corruption_generator,
                    train_corpus.metadata["tokenizer"],
                )
                dashboard.write(
                    json.dumps({"kind": "checkpoint", "step": counters["step"]}) + "\n"
                )
        if batch_prefetcher is not None:
            batch_prefetcher.close()
    counters["wall_seconds"] = time.perf_counter() - start_wall
    counters["estimated_flops_per_ordinary_source_token"] = counters[
        "estimated_flops"
    ] / max(counters["ordinary_token_equivalents"], 1)
    final_validation = validate(model, validation_corpus, config, device)
    counters.update(final_validation)
    _save_checkpoint(
        output_dir / "checkpoint.pt",
        model,
        fixed,
        soft_metadata,
        optimizer,
        config,
        counters,
        generator,
        corruption_generator,
        train_corpus.metadata["tokenizer"],
    )
    with dashboard_path.open("a", encoding="utf-8") as dashboard:
        dashboard.write(
            json.dumps(
                {
                    "kind": "eval",
                    "step": counters["step"],
                    "loss": final_validation["ordinary_validation_loss"],
                    "val_loss": final_validation["ordinary_validation_loss"],
                    "ppl": final_validation["ordinary_validation_perplexity"],
                    "bits_per_byte": final_validation["validation_bits_per_byte"],
                }
            )
            + "\n"
        )
        dashboard.write(
            json.dumps({"kind": "checkpoint", "step": counters["step"]}) + "\n"
        )
    (output_dir / "summary.json").write_text(
        json.dumps({**metadata, "results": counters}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
