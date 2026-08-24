#!/usr/bin/env python3
"""Ordinary and guarded soft-token generation from a recovered checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from branch_equivalence import _load_model
from superposition.semantic_clusters import (
    ClusterConfig,
    fused_cluster_embedding,
    select_safe_cluster,
)
from superposition.soft_tokens import SoftTokenMetadata, select_sparse_candidates


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_ids: list[int],
    *,
    mode: str,
    max_new_tokens: int,
    checkpoint_interval: int,
    cluster_config: ClusterConfig,
    soft_metadata: SoftTokenMetadata | None = None,
) -> dict:
    if mode not in {"ordinary", "assisted", "latent"}:
        raise ValueError("mode must be ordinary, assisted, or latent")
    device = model.token_embedding.weight.device
    history = model.token_embedding(torch.tensor(prompt_ids, device=device))[None, :, :]
    emitted: list[int | None] = []
    decisions = []
    consecutive_soft = 0
    for _ in range(max_new_tokens):
        output = model(input_embeddings=history)
        logits = output.logits[0, -1]
        sparse = select_sparse_candidates(logits)
        force_discrete = (
            mode == "ordinary"
            or (mode == "assisted" and consecutive_soft >= 1)
            or (mode == "latent" and consecutive_soft >= checkpoint_interval)
        )
        decision = select_safe_cluster(
            sparse,
            model.token_embedding.weight[sparse.token_ids],
            cluster_config,
        )
        if decision.accepted and not force_discrete:
            embedding = fused_cluster_embedding(
                decision, model.token_embedding.weight
            ).to(history.dtype)
            if soft_metadata is None:
                raise ValueError(
                    "soft-token generation requires trained metadata embeddings"
                )
            probability = torch.tensor(
                decision.probabilities,
                device=device,
                dtype=torch.float32,
            )
            probability = probability / probability.sum()
            normalized_entropy = (
                -(probability * probability.clamp_min(1e-12).log()).sum()
                / torch.tensor(len(probability), device=device).float().log()
            )
            embedding = soft_metadata(
                embedding[None],
                normalized_entropy=normalized_entropy[None],
                cluster_mass=torch.tensor([decision.mass], device=device),
                dispersion=torch.tensor(
                    [min(max(decision.maximum_cosine_distance, 0.0), 1.0)],
                    device=device,
                ),
            )[0]
            history = torch.cat((history, embedding[None, None]), dim=1)
            emitted.append(None)
            consecutive_soft += 1
            action = "soft"
        else:
            token = int(sparse.token_ids[0])
            history = torch.cat(
                (history, model.token_embedding.weight[token][None, None]), dim=1
            )
            emitted.append(token)
            consecutive_soft = 0
            action = "discrete_checkpoint" if force_discrete else "discrete_fallback"
        decisions.append(
            {
                "action": action,
                "candidate_ids": sparse.token_ids.tolist(),
                "selected_mass": sparse.selected_mass,
                "cluster": {
                    "accepted": decision.accepted,
                    "token_ids": decision.token_ids,
                    "mass": decision.mass,
                    "dispersion": decision.maximum_cosine_distance,
                    "reason": decision.reason,
                },
            }
        )
    return {
        "mode": mode,
        "prompt_ids": prompt_ids,
        "emitted_ids": emitted,
        "soft_steps": sum(value is None for value in emitted),
        "discrete_fallback_rate": sum(value is not None for value in emitted)
        / max(len(emitted), 1),
        "decisions": decisions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument(
        "--mode", choices=("ordinary", "assisted", "latent"), default="ordinary"
    )
    parser.add_argument("--checkpoint-interval", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model, checkpoint = _load_model(args.checkpoint, torch.device(args.device))
    exposure_rate = checkpoint["config"]["training"].get("soft_token_rate", 0.0)
    if args.mode != "ordinary" and exposure_rate <= 0:
        raise SystemExit(
            "assisted/latent inference requires a checkpoint trained with soft-token exposure"
        )
    soft_metadata = None
    if exposure_rate > 0:
        soft_metadata = SoftTokenMetadata(model.config.width).to(args.device)
        soft_metadata.load_state_dict(checkpoint["soft_metadata"])
        soft_metadata.eval()
    tokenizer = checkpoint.get("tokenizer")
    if tokenizer is None:
        raise SystemExit("checkpoint has no tokenizer metadata")
    import ztok

    pipeline = (
        ztok.Pipeline.byte_id()
        if tokenizer.get("path") is None
        else ztok.Pipeline.from_path(tokenizer["path"])
    )
    rows = []
    with pipeline:
        for prompt in args.prompt:
            result = generate(
                model,
                pipeline.encode(prompt),
                mode=args.mode,
                max_new_tokens=args.max_new_tokens,
                checkpoint_interval=args.checkpoint_interval,
                cluster_config=ClusterConfig(),
                soft_metadata=soft_metadata,
            )
            discrete = [value for value in result["emitted_ids"] if value is not None]
            decoded = pipeline.decode(discrete)
            result["discrete_text_only"] = (
                decoded.decode("utf-8", errors="replace")
                if isinstance(decoded, bytes)
                else decoded
            )
            rows.append({"prompt": prompt, **result})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"schema": "ztok.soft_generation.v1", "generations": rows}, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
