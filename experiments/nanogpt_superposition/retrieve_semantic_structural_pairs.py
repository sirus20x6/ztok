#!/usr/bin/env python3
"""Retrieve semantically similar examples, then require structural compatibility."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from contextualize_dataset import DEFAULT_MODEL, DEFAULT_REVISION
from retrieve_structural_pairs import structural_features, weighted_jaccard
from transformers import AutoModel, AutoTokenizer

DEFAULT_PROMPT = "Represent this sentence for paraphrase retrieval: {text}"


def embed_examples(
    texts: list[str],
    *,
    model_name: str,
    revision: str,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, use_fast=True, padding_side="left"
    )
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = AutoModel.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            prompts = [
                DEFAULT_PROMPT.format(text=text)
                for text in texts[start : start + batch_size]
            ]
            encoded = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            output = model(**encoded, return_dict=True)
            vectors.append(output.last_hidden_state[:, -1].float().cpu())
    return F.normalize(torch.cat(vectors), dim=-1)


def retrieve_pairs(
    examples: list[dict[str, Any]],
    embeddings: torch.Tensor,
    *,
    semantic_top_k: int,
    minimum_semantic_score: float,
    minimum_structural_score: float,
    minimum_content_spans: int,
    max_pairs: int,
    similarity_device: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if embeddings.ndim != 2 or embeddings.shape[0] != len(examples):
        raise ValueError("embeddings must have one row per example")
    if semantic_top_k <= 0 or max_pairs <= 0:
        raise ValueError("semantic_top_k and max_pairs must be positive")
    normalized = F.normalize(embeddings.float(), dim=-1)
    if similarity_device is not None:
        normalized = normalized.to(similarity_device)
    features = [structural_features(example) for example in examples]
    feature_counts = [sum(feature.values()) for feature in features]
    candidate_pairs: dict[tuple[int, int], float] = {}
    width = min(semantic_top_k + 1, len(examples))
    for start in range(0, len(examples), 512):
        scores = normalized[start : start + 512] @ normalized.T
        values, indices = scores.topk(width, dim=1)
        for local_index, (row_values, row_indices) in enumerate(
            zip(values, indices, strict=True)
        ):
            left = start + local_index
            for score, right_tensor in zip(row_values.tolist(), row_indices.tolist()):
                right = int(right_tensor)
                if left == right or score < minimum_semantic_score:
                    continue
                pair = (min(left, right), max(left, right))
                candidate_pairs[pair] = max(score, candidate_pairs.get(pair, -1.0))

    candidates = []
    for (left, right), semantic_score in candidate_pairs.items():
        if (
            feature_counts[left] < minimum_content_spans
            or feature_counts[right] < minimum_content_spans
        ):
            continue
        structural_score = weighted_jaccard(features[left], features[right])
        if structural_score < minimum_structural_score:
            continue
        exact_duplicate = examples[left]["text"] == examples[right]["text"]
        candidates.append(
            (
                semantic_score,
                structural_score,
                exact_duplicate,
                left,
                right,
            )
        )

    selected = []
    used: set[int] = set()
    for semantic_score, structural_score, exact_duplicate, left, right in sorted(
        candidates,
        key=lambda item: (
            -item[0],
            -item[1],
            examples[item[3]]["example_id"],
            examples[item[4]]["example_id"],
        ),
    ):
        if left in used or right in used:
            continue
        used.update((left, right))
        selected.append(
            {
                "left_example_id": examples[left]["example_id"],
                "right_example_id": examples[right]["example_id"],
                "semantic_score": semantic_score,
                "structural_score": structural_score,
                "exact_duplicate": exact_duplicate,
                "left_feature_count": feature_counts[left],
                "right_feature_count": feature_counts[right],
            }
        )
        if len(selected) >= max_pairs:
            break
    return selected, {
        "semantic_candidate_pair_count": len(candidate_pairs),
        "structurally_compatible_pair_count": len(candidates),
        "selected_pair_count": len(selected),
        "selected_exact_duplicate_count": sum(
            pair["exact_duplicate"] for pair in selected
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--semantic-top-k", type=int, default=32)
    parser.add_argument("--minimum-semantic-score", type=float, default=0.75)
    parser.add_argument("--minimum-structural-score", type=float, default=0.35)
    parser.add_argument("--minimum-content-spans", type=int, default=3)
    parser.add_argument("--max-pairs", type=int, default=64)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    embeddings = embed_examples(
        [str(example["text"]) for example in payload["examples"]],
        model_name=args.model,
        revision=args.revision,
        device=args.device,
        batch_size=args.batch_size,
    )
    selected, counts = retrieve_pairs(
        payload["examples"],
        embeddings,
        semantic_top_k=args.semantic_top_k,
        minimum_semantic_score=args.minimum_semantic_score,
        minimum_structural_score=args.minimum_structural_score,
        minimum_content_spans=args.minimum_content_spans,
        max_pairs=args.max_pairs,
        similarity_device=args.device,
    )
    selected_ids = {
        example_id
        for pair in selected
        for example_id in (pair["left_example_id"], pair["right_example_id"])
    }
    output = copy.deepcopy(payload)
    output["dataset_id"] = (
        f"{payload.get('dataset_id', args.input.stem)}-semantic-retrieved-{len(selected)}"
    )
    output["examples"] = [
        example
        for example in output["examples"]
        if example["example_id"] in selected_ids
    ]
    output["retrieval_pairs"] = selected
    output["retrieval_adapter"] = {
        "kind": "qwen_sentence_embedding_then_weighted_structural_jaccard",
        "model": args.model,
        "revision": args.revision,
        "prompt": DEFAULT_PROMPT,
        "pooling": "last_nonpadding_token",
        "semantic_top_k": args.semantic_top_k,
        "minimum_semantic_score": args.minimum_semantic_score,
        "minimum_structural_score": args.minimum_structural_score,
        "minimum_content_spans": args.minimum_content_spans,
        "disjoint_pairs": True,
        "source_example_count": len(payload["examples"]),
        "source_span_count": sum(
            len(example["spans"]) for example in payload["examples"]
        ),
        "selected_example_count": len(output["examples"]),
        "selected_span_count": sum(
            len(example["spans"]) for example in output["examples"]
        ),
        **counts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["retrieval_adapter"], indent=2))


if __name__ == "__main__":
    main()
