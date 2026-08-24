"""ProofWriter one-slot semantic-superposition primer utilities.

The expansion inventory contains sets of statements that are identical except
for one logically valid slot.  This module turns each set into one causal-LM
example: the differing input token is replaced by a norm-preserving weighted
mixture and the preceding output position predicts the complete alternative
set.  Shared continuation tokens retain the ordinary causal objective.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class StatementExpansion:
    expansion_id: str
    token_ids: tuple[int, ...]
    expanded_token_index: int
    alternative_token_ids: tuple[int, ...]
    weights: tuple[float, ...]
    explicit_source_bytes: int
    statement_kind: str
    slot_kind: str


@dataclass(frozen=True)
class StatementPrimerBatch:
    token_ids: torch.Tensor
    expanded_token_indices: torch.Tensor
    alternative_token_ids: torch.Tensor
    alternative_weights: torch.Tensor
    explicit_source_bytes: int


@dataclass
class StatementPrimerResult:
    loss: torch.Tensor
    model_positions: int
    target_tokens: int
    diagnostics: dict[str, Any]


def _aligned_expansion(record: dict[str, Any], pipeline: Any) -> StatementExpansion | None:
    alternatives = record.get("alternatives", [])
    if len(alternatives) < 2:
        return None
    statements = [str(value["statements"][0]) for value in alternatives]
    branches = [tuple(int(token) for token in pipeline.encode(text)) for text in statements]
    if not branches or len({len(branch) for branch in branches}) != 1:
        return None
    if len(branches[0]) < 2:
        return None
    differing = [
        index
        for index, values in enumerate(zip(*branches, strict=True))
        if len(set(values)) > 1
    ]
    if len(differing) != 1:
        return None
    expanded_index = differing[0]
    # A token outside the causal input cannot teach a fused state.
    if expanded_index >= len(branches[0]) - 1:
        return None
    alternative_ids = tuple(branch[expanded_index] for branch in branches)
    if len(set(alternative_ids)) != len(alternative_ids):
        return None
    raw_weights = [float(value.get("weight", 1.0)) for value in alternatives]
    weight_sum = sum(raw_weights)
    if weight_sum <= 0:
        return None
    weights = tuple(weight / weight_sum for weight in raw_weights)
    return StatementExpansion(
        expansion_id=str(record["expansion_id"]),
        token_ids=branches[0],
        expanded_token_index=expanded_index,
        alternative_token_ids=alternative_ids,
        weights=weights,
        explicit_source_bytes=sum(len(text.encode("utf-8")) for text in statements),
        statement_kind=str(record.get("statement_kind", "unknown")),
        slot_kind=str(record.get("expanded_slot_kind", "unknown")),
    )


def load_statement_expansions(
    path: Path | str,
    pipeline: Any,
) -> tuple[list[StatementExpansion], dict[str, Any]]:
    """Load and conservatively retain exact one-token branch alignments."""
    expansions: list[StatementExpansion] = []
    total = 0
    rejected = 0
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            total += 1
            expansion = _aligned_expansion(json.loads(line), pipeline)
            if expansion is None:
                rejected += 1
            else:
                expansions.append(expansion)
    diagnostics = {
        "inventory_expansions": total,
        "usable_expansions": len(expansions),
        "rejected_unaligned_expansions": rejected,
        "usable_fraction": len(expansions) / total if total else 0.0,
        "explicit_source_bytes_per_epoch": sum(
            expansion.explicit_source_bytes for expansion in expansions
        ),
    }
    return expansions, diagnostics


def statement_primer_batches(
    expansions: Iterable[StatementExpansion],
    *,
    batch_size: int,
    seed: int,
) -> Iterator[list[StatementExpansion]]:
    """Yield deterministic shuffled, equal-length batches without padding."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    buckets: dict[int, list[StatementExpansion]] = defaultdict(list)
    for expansion in expansions:
        buckets[len(expansion.token_ids)].append(expansion)
    rng = random.Random(seed)
    batches: list[list[StatementExpansion]] = []
    for length in sorted(buckets):
        bucket = buckets[length]
        rng.shuffle(bucket)
        batches.extend(
            bucket[start : start + batch_size]
            for start in range(0, len(bucket), batch_size)
        )
    rng.shuffle(batches)
    yield from batches


def collate_statement_primer_batch(
    records: list[StatementExpansion],
    *,
    device: torch.device,
) -> StatementPrimerBatch:
    if not records:
        raise ValueError("cannot collate an empty primer batch")
    sequence_length = len(records[0].token_ids)
    if any(len(record.token_ids) != sequence_length for record in records):
        raise ValueError("primer batches require equal sequence lengths")
    maximum_alternatives = max(len(record.alternative_token_ids) for record in records)
    alternative_ids = torch.zeros(
        (len(records), maximum_alternatives), dtype=torch.long, device=device
    )
    weights = torch.zeros(
        (len(records), maximum_alternatives), dtype=torch.float32, device=device
    )
    for row, record in enumerate(records):
        count = len(record.alternative_token_ids)
        alternative_ids[row, :count] = torch.tensor(
            record.alternative_token_ids, dtype=torch.long, device=device
        )
        weights[row, :count] = torch.tensor(
            record.weights, dtype=torch.float32, device=device
        )
    return StatementPrimerBatch(
        token_ids=torch.tensor(
            [record.token_ids for record in records], dtype=torch.long, device=device
        ),
        expanded_token_indices=torch.tensor(
            [record.expanded_token_index for record in records],
            dtype=torch.long,
            device=device,
        ),
        alternative_token_ids=alternative_ids,
        alternative_weights=weights,
        explicit_source_bytes=sum(record.explicit_source_bytes for record in records),
    )


def norm_preserving_fusion(
    embeddings: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Fuse ``[batch, alternatives, width]`` while preserving mean norm."""
    float_embeddings = embeddings.float()
    weighted = (float_embeddings * weights[..., None]).sum(dim=1)
    target_norm = (
        float_embeddings.norm(dim=-1) * weights
    ).sum(dim=1, keepdim=True)
    fused = F.normalize(weighted, dim=-1) * target_norm
    return fused.to(dtype=embeddings.dtype)


def statement_primer_objective(
    model: torch.nn.Module,
    batch: StatementPrimerBatch,
) -> StatementPrimerResult:
    """Train one fused logical slot plus its shared causal continuation."""
    inputs = batch.token_ids[:, :-1]
    targets = batch.token_ids[:, 1:]
    input_embeddings = model.token_embedding(inputs).clone()
    alternative_embeddings = model.token_embedding(batch.alternative_token_ids)
    fused = norm_preserving_fusion(
        alternative_embeddings, batch.alternative_weights
    )
    rows = torch.arange(inputs.shape[0], device=inputs.device)
    input_embeddings[rows, batch.expanded_token_indices] = fused
    output = model(input_embeddings=input_embeddings)

    per_position = F.cross_entropy(
        output.logits.flatten(0, 1), targets.flatten(), reduction="none"
    ).view_as(targets)
    set_target_rows = batch.expanded_token_indices > 0
    replaced_positions = torch.zeros_like(per_position, dtype=torch.bool)
    if bool(set_target_rows.any()):
        selected_rows = rows[set_target_rows]
        selected_positions = batch.expanded_token_indices[set_target_rows] - 1
        log_probs = F.log_softmax(
            output.logits[selected_rows, selected_positions].float(), dim=-1
        )
        selected_ids = batch.alternative_token_ids[set_target_rows]
        selected_weights = batch.alternative_weights[set_target_rows]
        set_losses = -(
            log_probs.gather(1, selected_ids) * selected_weights
        ).sum(dim=1)
        per_position[selected_rows, selected_positions] = set_losses.to(
            dtype=per_position.dtype
        )
        replaced_positions[selected_rows, selected_positions] = True

    loss = per_position.mean()
    alternatives_per_example = (batch.alternative_weights > 0).sum(dim=1).float()
    return StatementPrimerResult(
        loss=loss,
        model_positions=inputs.numel(),
        target_tokens=targets.numel(),
        diagnostics={
            "set_target_positions": int(replaced_positions.sum()),
            "mean_alternatives": float(alternatives_per_example.mean()),
            "maximum_alternatives": int(alternatives_per_example.max()),
            "mean_primer_sequence_tokens": float(batch.token_ids.shape[1]),
        },
    )
