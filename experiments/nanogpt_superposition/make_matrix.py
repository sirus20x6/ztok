#!/usr/bin/env python3
"""Materialize paired screening configs for trainboard-visible runs."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ordinary-data", required=True)
    parser.add_argument("--superbpe-data", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs-root", default="/thearray/git/moe-mla/runs")
    parser.add_argument("--run-prefix", default="ztok-superposition")
    parser.add_argument("--seed", type=int, default=260506546)
    parser.add_argument("--full-matrix", action="store_true")
    parser.add_argument(
        "--budget-kind",
        choices=("steps", "source_bytes", "flops", "gpu_seconds"),
        default="source_bytes",
    )
    parser.add_argument("--budget-value", type=float, default=400000000)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--context-bytes", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument(
        "--schedule",
        choices=("50/25/25", "50/0/50", "75/15/10", "30/30/40"),
        default="50/25/25",
    )
    parser.add_argument(
        "--lr-schedule-basis", choices=("steps", "budget"), default="budget"
    )
    parser.add_argument("--coarse-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--ordinary-lr-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--coarse-superposed-source-gradient-multiplier", type=float, default=1.0
    )
    parser.add_argument(
        "--mixed-superposed-source-gradient-multiplier", type=float, default=1.0
    )
    args = parser.parse_args()
    ordinary_metadata = json.loads(
        (Path(args.ordinary_data) / "metadata.json").read_text()
    )
    superbpe_metadata = json.loads(
        (Path(args.superbpe_data) / "metadata.json").read_text()
    )
    for split in ("train", "validation"):
        if (
            ordinary_metadata["splits"][split]["split_hash"]
            != superbpe_metadata["splits"][split]["split_hash"]
        ):
            raise SystemExit(f"ordinary/SuperBPE {split} document split hashes differ")
    if (
        ordinary_metadata["source"]["document_order_hash"]
        != superbpe_metadata["source"]["document_order_hash"]
    ):
        raise SystemExit("ordinary/SuperBPE source document order hashes differ")

    def context_tokens(metadata: dict) -> int:
        bytes_per_token = metadata["splits"]["train"]["bytes_per_token"]
        tokens = max(16, round(args.context_bytes / bytes_per_token))
        return max(16, tokens - tokens % 8)

    contexts = {
        "ordinary_bpe": context_tokens(ordinary_metadata),
        "superbpe": context_tokens(superbpe_metadata),
    }
    base = {
        "schema": "ztok.nanogpt_superposition.config.v1",
        "seed": args.seed,
        "rwkv_backend": "rwkv_lab",
        "model": {
            "vocab_size": 32768,
            "width": 448,
            "layers": 10,
            "heads": 7,
            "mlp_multiple": 3,
            "max_sequence_length": max(contexts.values()),
            "tie_embeddings": True,
            "dropout": 0.0,
            "rope_base": 10000.0,
        },
        "training": {
            "representation": "ordinary",
            "group_size": 1,
            "fusion": "norm_preserving_mean",
            "target": "bag",
            "schedule": args.schedule,
            "batch_size": args.batch_size,
            "context_tokens": contexts["ordinary_bpe"],
            "context_source_bytes_target": args.context_bytes,
            "max_steps": args.max_steps,
            "budget_kind": args.budget_kind,
            "budget_value": (
                None if args.budget_kind == "steps" else args.budget_value
            ),
            "eval_interval": args.eval_interval,
            "eval_batches": args.eval_batches,
            "checkpoint_interval": args.checkpoint_interval,
            "learning_rate": 0.0003,
            "min_learning_rate": 0.00003,
            "warmup_steps": 100,
            "lr_schedule_basis": args.lr_schedule_basis,
            "coarse_lr_multiplier": args.coarse_lr_multiplier,
            "ordinary_lr_multiplier": args.ordinary_lr_multiplier,
            "coarse_superposed_source_gradient_multiplier": args.coarse_superposed_source_gradient_multiplier,
            "mixed_superposed_source_gradient_multiplier": args.mixed_superposed_source_gradient_multiplier,
            "weight_decay": 0.1,
            "grad_clip": 1.0,
            "mixed_precision": True,
            "device": "cuda",
            "soft_token_rate": 0.0,
        },
        "notes": "One-seed screening only; repeat winners with three seeds and a larger model.",
    }
    if args.full_matrix:
        conditions = [
            (tokenizer, representation, group, fusion)
            for tokenizer in ("ordinary_bpe", "superbpe")
            for representation, group in (("ordinary", 1), ("fixed", 2), ("fixed", 4))
            for fusion in (
                ("mean", "norm_preserving_mean")
                if representation == "fixed"
                else ("norm_preserving_mean",)
            )
        ]
    else:
        conditions = [
            ("ordinary_bpe", "ordinary", 1, "norm_preserving_mean"),
            ("ordinary_bpe", "fixed", 2, "norm_preserving_mean"),
            ("ordinary_bpe", "fixed", 4, "norm_preserving_mean"),
            ("superbpe", "ordinary", 1, "norm_preserving_mean"),
            ("superbpe", "fixed", 2, "norm_preserving_mean"),
        ]
    args.output.mkdir(parents=True, exist_ok=True)
    for architecture in ("transformer", "rwkv"):
        for tokenizer, representation, group, fusion in conditions:
            config = copy.deepcopy(base)
            budget_label = {
                "steps": "b1steps",
                "source_bytes": "b2bytes",
                "flops": "b3flops",
                "gpu_seconds": "b3seconds",
            }[args.budget_kind]
            name = f"{args.run_prefix}-{architecture}-{tokenizer}-{representation}{group}-{fusion}-{budget_label}"
            config.update(
                {
                    "run_name": name,
                    "architecture": architecture,
                    "tokenizer_condition": tokenizer,
                    "data_dir": args.ordinary_data
                    if tokenizer == "ordinary_bpe"
                    else args.superbpe_data,
                    "output_dir": str(Path(args.runs_root) / name),
                }
            )
            config["training"].update(
                {
                    "representation": representation,
                    "group_size": group,
                    "fusion": fusion,
                    "context_tokens": contexts[tokenizer],
                }
            )
            path = args.output / f"{name}.json"
            path.write_text(json.dumps(config, indent=2) + "\n")
            print(path)


if __name__ == "__main__":
    main()
