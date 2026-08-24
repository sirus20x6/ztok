#!/usr/bin/env python3
"""Tiny decoder-only plumbing test for fixed token superposition.

This is a falsification-oriented sanity check, not a paper reproduction.
It compares ordinary training, fixed groups of 2/4, and ordinary-token
recovery on the same deterministic 10k+ sequence corpus.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional

import ztok
from ztok.superposition import FixedSuperpositionConfig, build_fixed_plan


class TinyDecoder(nn.Module):
    def __init__(self, vocab: int, sequence_length: int, width: int, layers: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab, width)
        self.position_embedding = nn.Embedding(sequence_length, width)
        block = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=width * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(block, num_layers=layers)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab, bias=False)

    def hidden(self, embeddings: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = embeddings + self.position_embedding(positions)
        length = x.shape[1]
        mask = torch.triu(
            torch.ones(length, length, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        return self.decoder(x, mask=mask, is_causal=True)

    def ordinary_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.hidden(self.token_embedding(tokens), positions)
        return self.head(self.norm(hidden))

    def group_logits(self, tokens: torch.Tensor, group_size: int) -> torch.Tensor:
        batch, length = tokens.shape
        usable = length - (length % group_size)
        tokens = tokens[:, :usable]
        x = self.token_embedding(tokens).reshape(
            batch, usable // group_size, group_size, -1
        )
        raw = x.mean(dim=2)
        target_norm = x.norm(dim=-1).mean(dim=2, keepdim=True)
        fused = functional.normalize(raw, dim=-1) * target_norm
        source_positions = torch.arange(usable, device=tokens.device).reshape(
            usable // group_size, group_size
        )
        position_vectors = self.position_embedding(source_positions).mean(dim=1)
        length = fused.shape[1]
        mask = torch.triu(
            torch.ones(length, length, device=tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        hidden = self.decoder(
            fused + position_vectors,
            mask=mask,
            is_causal=True,
        )
        return self.head(self.norm(hidden))


def corpus(count: int, length: int, vocab: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, vocab, (count, 1), generator=generator)
    steps = torch.randint(1, 8, (count, 1), generator=generator)
    positions = torch.arange(length).reshape(1, -1)
    structured = (starts + positions * steps) % vocab
    noise = torch.randint(0, vocab, (count, length), generator=generator)
    replace = torch.rand((count, length), generator=generator) < 0.08
    return torch.where(replace, noise, structured).long()


def ordinary_loss(model: TinyDecoder, batch: torch.Tensor) -> torch.Tensor:
    logits = model.ordinary_logits(batch[:, :-1])
    return functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        batch[:, 1:].reshape(-1),
    )


def group_loss(
    model: TinyDecoder, batch: torch.Tensor, group_size: int
) -> torch.Tensor:
    usable = batch.shape[1] - (batch.shape[1] % group_size)
    groups = batch[:, :usable].reshape(batch.shape[0], -1, group_size)
    logits = model.group_logits(batch[:, :usable], group_size)[:, :-1]
    targets = groups[:, 1:]
    log_probabilities = logits.log_softmax(dim=-1)
    selected = log_probabilities.gather(
        dim=-1, index=targets
    )
    return -selected.mean()


@torch.no_grad()
def validate(model: TinyDecoder, validation: torch.Tensor, batch_size: int) -> float:
    model.eval()
    losses = []
    for start in range(0, min(len(validation), batch_size * 8), batch_size):
        losses.append(
            ordinary_loss(model, validation[start : start + batch_size]).item()
        )
    model.train()
    return sum(losses) / len(losses)


def train_condition(
    name: str,
    train: torch.Tensor,
    validation: torch.Tensor,
    *,
    group_size: int,
    phase_steps: int,
    recovery_steps: int,
    batch_size: int,
    width: int,
    layers: int,
    vocab: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = TinyDecoder(vocab, train.shape[1], width, layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    generator = torch.Generator().manual_seed(seed + 1)

    def run(steps: int, grouped: bool) -> tuple[float, float, float]:
        started = time.perf_counter()
        final_loss = math.nan
        source_tokens = 0
        for _ in range(steps):
            indices = torch.randint(
                0, len(train), (batch_size,), generator=generator
            )
            batch = train[indices].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = (
                group_loss(model, batch, group_size)
                if grouped
                else ordinary_loss(model, batch)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            final_loss = loss.item()
            source_tokens += batch.numel()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        return final_loss, elapsed, source_tokens / elapsed

    initial_validation = validate(model, validation.to(device), batch_size)
    phase_loss, phase_seconds, phase_tokens_per_second = run(
        phase_steps, grouped=group_size > 1
    )
    pre_recovery_validation = validate(model, validation.to(device), batch_size)
    recovery_loss, recovery_seconds, recovery_tokens_per_second = run(
        recovery_steps, grouped=False
    )
    final_validation = validate(model, validation.to(device), batch_size)
    return {
        "condition": name,
        "group_size": group_size,
        "sequence_length_reduction": 1.0 - 1.0 / group_size,
        "initial_ordinary_validation_loss": initial_validation,
        "phase_training_loss": phase_loss,
        "ordinary_validation_loss_before_recovery": pre_recovery_validation,
        "recovery_training_loss": recovery_loss,
        "ordinary_validation_loss_after_recovery": final_validation,
        "phase_seconds": phase_seconds,
        "recovery_seconds": recovery_seconds,
        "total_seconds": phase_seconds + recovery_seconds,
        "phase_source_tokens_per_second": phase_tokens_per_second,
        "recovery_tokens_per_second": recovery_tokens_per_second,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", type=int, default=10_000)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--phase-steps", type=int, default=100)
    parser.add_argument("--recovery-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    if args.sequences < 10_000:
        raise SystemExit("--sequences must be at least 10000 for the recorded sanity test")
    if args.sequence_length % 4:
        raise SystemExit("--sequence-length must be divisible by 4")

    # Exercise the actual ztok plan path before the model-side vectorization.
    with ztok.Pipeline.byte_id() as pipeline:
        probe = bytes((32 + index % 90 for index in range(args.sequence_length)))
        ordinary = pipeline.encode(probe)
        for group_size in (2, 4):
            plan = build_fixed_plan(
                pipeline, probe, FixedSuperpositionConfig(group_size=group_size)
            )
            assert list(plan.original_ids) == ordinary
            assert plan.output_token_count == len(ordinary) // group_size

    data = corpus(args.sequences, args.sequence_length, 256, args.seed)
    split = max(1, int(len(data) * 0.9))
    train, validation = data[:split], data[split:]
    device = torch.device(args.device)
    conditions = [
        train_condition(
            name,
            train,
            validation,
            group_size=group_size,
            phase_steps=args.phase_steps,
            recovery_steps=args.recovery_steps,
            batch_size=args.batch_size,
            width=args.width,
            layers=args.layers,
            vocab=256,
            device=device,
            seed=args.seed,
        )
        for name, group_size in (("ordinary_baseline", 1), ("fixed_2", 2), ("fixed_4", 4))
    ]
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).strip()
    )
    baseline_final = conditions[0]["ordinary_validation_loss_after_recovery"]
    baseline_wall = conditions[0]["total_seconds"]
    for result in conditions:
        result["final_loss_delta_vs_baseline"] = (
            result["ordinary_validation_loss_after_recovery"] - baseline_final
        )
        result["relative_final_loss_delta_vs_baseline"] = (
            result["final_loss_delta_vs_baseline"] / baseline_final
        )
        result["lower_wall_clock_cost"] = result["total_seconds"] < baseline_wall
        result["reaches_baseline_within_1pct"] = (
            result["ordinary_validation_loss_after_recovery"]
            <= baseline_final * 1.01
        )
    report = {
        "schema": "ztok.fixed_superposition_sanity.v1",
        "git_commit": commit,
        "git_dirty": dirty,
        "seed": args.seed,
        "device": str(device),
        "tokenizer_identifier": "ztok byte_id / identity / identity",
        "vocabulary_identifier": "byte values 0..255",
        "embedding_model": "trainable TinyDecoder token embedding",
        "embedding_layer": "input embedding",
        "span_extraction_policy": "fixed contiguous token windows",
        "dataset_sequences": args.sequences,
        "sequence_length": args.sequence_length,
        "phase_steps": args.phase_steps,
        "recovery_steps": args.recovery_steps,
        "model": {"width": args.width, "layers": args.layers, "vocabulary": 256},
        "conditions": conditions,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "note": "Bag loss averages next-group token log-probabilities; this is a plumbing sanity check, not a paper reproduction.",
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Fixed token-superposition language-model sanity check",
        "",
        f"- Commit: `{commit}` ({'dirty' if dirty else 'clean'})",
        f"- Dataset: {args.sequences} sequences × {args.sequence_length} source tokens",
        f"- Device: `{device}`",
        "",
        "| Condition | Reduction | Phase tok/s | Pre-recovery val loss | Final ordinary val loss | Δ vs baseline | Wall time |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in conditions:
        lines.append(
            f"| {result['condition']} | {result['sequence_length_reduction']:.1%} "
            f"| {result['phase_source_tokens_per_second']:.0f} "
            f"| {result['ordinary_validation_loss_before_recovery']:.4f} "
            f"| {result['ordinary_validation_loss_after_recovery']:.4f} "
            f"| {result['final_loss_delta_vs_baseline']:+.4f} "
            f"| {result['total_seconds']:.2f}s |"
        )
    lines.extend(
        [
            "",
            "The tokenizer IDs used by the plan were asserted identical to ordinary byte-ID encoding. The training loss and embedding operations remain outside tokenizer core.",
            "",
        ]
    )
    args.markdown_output.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
