#!/usr/bin/env python3
"""Compare fused feedback with explicit discrete branches."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from models.common import ModelConfig
from models.rwkv import RWKVLM
from models.rwkv_lab_adapter import RWKVLabLM
from models.transformer import TransformerLM
from superposition.semantic_clusters import ClusterConfig, select_safe_cluster
from superposition.soft_tokens import (
    SparseCandidates,
    norm_preserving_weighted_fusion,
)


@dataclass(frozen=True)
class HorizonResult:
    horizon: int
    state_error: float
    hidden_error: float
    logit_kl: float


@dataclass(frozen=True)
class BranchResult:
    horizons: tuple[HorizonResult, ...]
    continuation_loss_fused: float
    continuation_loss_explicit: float
    continuation_loss_delta: float
    candidate_count: int
    candidate_embedding_diameter: float


def _state_vector(output: object) -> torch.Tensor:
    state = output.state
    if state is not None and hasattr(state, "flattened"):
        return state.flattened()
    return output.hidden[:, -1].float()


def _run(
    model: torch.nn.Module,
    prefix_embeddings: torch.Tensor,
    candidate_embedding: torch.Tensor,
    continuation_embeddings: torch.Tensor,
    horizon: int,
) -> object:
    values = [prefix_embeddings, candidate_embedding[None, None, :]]
    if horizon:
        values.append(continuation_embeddings[:horizon][None, :, :])
    return model(input_embeddings=torch.cat(values, dim=1))


def _relative_error(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        (actual - reference).float().norm() / reference.float().norm().clamp_min(1e-9)
    )


@torch.no_grad()
def transition_representations(
    model: torch.nn.Module,
    context_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> torch.Tensor:
    table = model.token_embedding.weight
    prefix = table[context_ids][None]
    vectors = []
    for candidate in table[candidate_ids]:
        output = model(
            input_embeddings=torch.cat((prefix, candidate[None, None]), dim=1)
        )
        vectors.append(_state_vector(output)[0])
    return torch.stack(vectors)


@torch.no_grad()
def branch_equivalence(
    model: torch.nn.Module,
    context_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
    probabilities: torch.Tensor,
    continuation_ids: torch.Tensor,
    *,
    horizons: Iterable[int] = (0, 1, 4, 16, 64),
) -> BranchResult:
    if context_ids.ndim != 1 or candidate_ids.ndim != 1 or probabilities.ndim != 1:
        raise ValueError(
            "context, candidates, and probabilities must be one-dimensional"
        )
    if candidate_ids.shape != probabilities.shape or not len(candidate_ids):
        raise ValueError("candidate IDs and probabilities must align and be nonempty")
    weights = probabilities.float() / probabilities.float().sum()
    table = model.token_embedding.weight
    prefix = table[context_ids][None, :, :]
    candidates = table[candidate_ids]
    continuation = table[continuation_ids]
    fused = norm_preserving_weighted_fusion(candidates, weights)
    maximum_horizon = len(continuation_ids)
    requested = sorted({min(max(int(value), 0), maximum_horizon) for value in horizons})
    results = []
    for horizon in requested:
        fused_output = _run(model, prefix, fused, continuation, horizon)
        branches = [
            _run(model, prefix, candidate, continuation, horizon)
            for candidate in candidates
        ]
        explicit_state = sum(
            weight * _state_vector(output)
            for weight, output in zip(weights, branches, strict=True)
        )
        explicit_hidden = sum(
            weight * output.hidden[:, -1].float()
            for weight, output in zip(weights, branches, strict=True)
        )
        explicit_probability = sum(
            weight * output.logits[:, -1].float().softmax(dim=-1)
            for weight, output in zip(weights, branches, strict=True)
        )
        fused_log_probability = fused_output.logits[:, -1].float().log_softmax(dim=-1)
        kl = F.kl_div(
            fused_log_probability,
            explicit_probability,
            reduction="batchmean",
        )
        results.append(
            HorizonResult(
                horizon=horizon,
                state_error=_relative_error(
                    _state_vector(fused_output), explicit_state
                ),
                hidden_error=_relative_error(
                    fused_output.hidden[:, -1].float(), explicit_hidden
                ),
                logit_kl=float(kl),
            )
        )

    fused_losses = []
    branch_log_likelihood = torch.zeros(
        len(candidates), device=candidates.device, dtype=torch.float32
    )
    for target_index in range(len(continuation_ids)):
        fused_output = _run(model, prefix, fused, continuation, target_index)
        target = continuation_ids[target_index : target_index + 1]
        fused_losses.append(F.cross_entropy(fused_output.logits[:, -1], target))
        for branch_index, candidate in enumerate(candidates):
            output = _run(model, prefix, candidate, continuation, target_index)
            branch_log_likelihood[branch_index] += (
                output.logits[:, -1].float().log_softmax(dim=-1)[0, int(target.item())]
            )
    fused_loss = float(torch.stack(fused_losses).mean()) if fused_losses else 0.0
    explicit_loss = (
        float(
            -torch.logsumexp(
                weights.clamp_min(1e-12).log() + branch_log_likelihood, dim=0
            )
            / len(continuation_ids)
        )
        if len(continuation_ids)
        else 0.0
    )
    normalized = F.normalize(candidates.float(), dim=-1)
    diameter = float((1.0 - normalized @ normalized.T).max())
    return BranchResult(
        horizons=tuple(results),
        continuation_loss_fused=fused_loss,
        continuation_loss_explicit=explicit_loss,
        continuation_loss_delta=fused_loss - explicit_loss,
        candidate_count=len(candidate_ids),
        candidate_embedding_diameter=diameter,
    )


def _load_model(checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    config = ModelConfig(**payload["config"]["model"])
    architecture = payload["config"]["architecture"]
    if architecture == "transformer":
        model = TransformerLM(config)
    elif payload["config"].get("rwkv_backend", "rwkv_lab") == "rwkv_lab":
        model = RWKVLabLM(
            config,
            channel_activation=payload["config"].get(
                "rwkv_channel_activation", "relu_squared"
            ),
        )
    else:
        model = RWKVLM(config)
    model.load_state_dict(payload["model"])
    return model.to(device).eval(), payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model, payload = _load_model(args.checkpoint, torch.device(args.device))
    tokenizer = payload.get("tokenizer")
    if not tokenizer or "path" not in tokenizer:
        raise SystemExit("checkpoint must record tokenizer metadata")
    import ztok

    pipeline = (
        ztok.Pipeline.byte_id()
        if tokenizer["path"] is None
        else ztok.Pipeline.from_path(tokenizer["path"])
    )
    cases = json.loads(args.cases.read_text())["cases"]
    rows = []
    with pipeline:
        for case in cases:
            context = pipeline.encode(case["context"])
            alternatives = [pipeline.encode(value) for value in case["candidates"]]
            if any(len(value) != 1 for value in alternatives):
                rows.append({**case, "status": "skipped_non_single_token"})
                continue
            continuation = pipeline.encode(
                case.get(
                    "continuation", " continuation text for divergence measurement."
                )
            )
            candidate_ids = torch.tensor(
                [value[0] for value in alternatives], device=args.device
            )
            candidate_probability = torch.full(
                (len(alternatives),),
                1.0 / len(alternatives),
                device=args.device,
            )
            protected = frozenset()
            if case["class"] in {"contradiction", "binding", "exact_sensitive"}:
                protected = frozenset(
                    (int(candidate_ids[left]), int(candidate_ids[right]))
                    for left in range(len(candidate_ids))
                    for right in range(left + 1, len(candidate_ids))
                )
            raw_decision = select_safe_cluster(
                SparseCandidates(
                    candidate_ids,
                    candidate_probability,
                    entropy=float(
                        -(candidate_probability * candidate_probability.log()).sum()
                    ),
                    selected_mass=1.0,
                ),
                model.token_embedding.weight[candidate_ids],
                ClusterConfig(protected_pairs=protected),
            )
            output_decision = select_safe_cluster(
                SparseCandidates(
                    candidate_ids,
                    candidate_probability,
                    entropy=float(
                        -(candidate_probability * candidate_probability.log()).sum()
                    ),
                    selected_mass=1.0,
                ),
                model.output.weight[candidate_ids],
                ClusterConfig(protected_pairs=protected),
            )
            transition_vectors = transition_representations(
                model,
                torch.tensor(context, device=args.device),
                candidate_ids,
            )
            transition_decision = select_safe_cluster(
                SparseCandidates(
                    candidate_ids,
                    candidate_probability,
                    entropy=float(
                        -(candidate_probability * candidate_probability.log()).sum()
                    ),
                    selected_mass=1.0,
                ),
                model.token_embedding.weight[candidate_ids],
                ClusterConfig(
                    protected_pairs=protected,
                    max_transition_distance=0.08,
                ),
                transition_vectors=transition_vectors,
            )
            result = branch_equivalence(
                model,
                torch.tensor(context, device=args.device),
                candidate_ids,
                candidate_probability,
                torch.tensor(continuation, device=args.device),
            )
            clustered_metrics = {}
            for label, decision in (
                ("embedding", raw_decision),
                ("transition", transition_decision),
            ):
                if not decision.accepted:
                    clustered_metrics[label] = None
                    continue
                selected = torch.tensor(decision.token_ids, device=args.device)
                selected_probability = torch.tensor(
                    decision.probabilities,
                    device=args.device,
                )
                clustered_metrics[label] = asdict(
                    branch_equivalence(
                        model,
                        torch.tensor(context, device=args.device),
                        selected,
                        selected_probability,
                        torch.tensor(continuation, device=args.device),
                    )
                )
            expected_safe = case["class"] in {"tight_lexical", "morphology"}
            rows.append(
                {
                    **case,
                    "status": "ok",
                    "expected_safe": expected_safe,
                    "cluster_decision": asdict(transition_decision),
                    "embedding_cluster_decision": asdict(raw_decision),
                    "output_cluster_decision": asdict(output_decision),
                    "transition_cluster_decision": asdict(transition_decision),
                    "clustered_metrics": clustered_metrics,
                    "metrics": asdict(result),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema": "ztok.branch_equivalence.v1",
                "checkpoint": str(args.checkpoint),
                "architecture": payload["config"]["architecture"],
                "rwkv_backend": payload["config"].get("rwkv_backend"),
                "git_commit": payload.get("git_commit"),
                "cases": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
