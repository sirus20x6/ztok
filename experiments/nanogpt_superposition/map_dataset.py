#!/usr/bin/env python3
"""Build an auditable semantic/grammatical superbatch plan from JSON spans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from superposition.dataset_mapping import (
    DatasetMappingConfig,
    RelationEdge,
    SpanNode,
    build_dataset_superposition_plan,
)

INPUT_SCHEMA = "ztok.dataset_semantic_spans.v1"


def load_exchange(
    payload: dict[str, Any],
) -> tuple[list[SpanNode], list[RelationEdge], torch.Tensor]:
    if payload.get("schema") != INPUT_SCHEMA:
        raise ValueError(f"expected schema {INPUT_SCHEMA}")
    spans = []
    embeddings = []
    edges = []
    seen_examples: set[int] = set()
    dimensions: int | None = None
    for example in payload.get("examples", []):
        example_id = int(example["example_id"])
        example_tokens = example.get("token_ids")
        if example_id in seen_examples:
            raise ValueError(f"duplicate example_id: {example_id}")
        seen_examples.add(example_id)
        for raw in example.get("spans", []):
            embedding = [float(value) for value in raw["embedding"]]
            if not embedding:
                raise ValueError("span embeddings must be non-empty")
            dimensions = dimensions or len(embedding)
            if len(embedding) != dimensions:
                raise ValueError("span embedding dimensions differ")
            embedding_index = len(embeddings)
            embeddings.append(embedding)
            spans.append(
                SpanNode(
                    example_id=example_id,
                    span_id=str(raw["span_id"]),
                    token_start=int(raw["token_start"]),
                    token_end=int(raw["token_end"]),
                    embedding_index=embedding_index,
                    grammatical_role=raw.get("grammatical_role"),
                    semantic_role=raw.get("semantic_role"),
                    semantic_value=raw.get("semantic_value"),
                    entity_type=raw.get("entity_type"),
                    binding_type=raw.get("binding_type"),
                    source_token_ids=(
                        tuple(
                            int(token)
                            for token in example_tokens[
                                int(raw["token_start"]) : int(raw["token_end"])
                            ]
                        )
                        if example_tokens is not None
                        else None
                    ),
                    confidence=float(raw.get("confidence", 1.0)),
                    protected=bool(raw.get("protected", False)),
                )
            )
        for raw in example.get("relations", []):
            edges.append(
                RelationEdge(
                    example_id=example_id,
                    source_span_id=str(raw["source_span_id"]),
                    relation=str(raw["relation"]),
                    target_span_id=str(raw["target_span_id"]),
                )
            )
    if not embeddings:
        raise ValueError("exchange contains no spans")
    return spans, edges, torch.tensor(embeddings, dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    spans, edges, embeddings = load_exchange(payload)
    config_payload = json.loads(args.config.read_text()) if args.config else {}
    allowed_pairs = (
        {
            tuple(sorted((int(pair["left_example_id"]), int(pair["right_example_id"]))))
            for pair in payload["retrieval_pairs"]
        }
        if "retrieval_pairs" in payload
        else None
    )
    plan = build_dataset_superposition_plan(
        spans,
        edges,
        embeddings,
        DatasetMappingConfig(**config_payload),
        allowed_example_pairs=allowed_pairs,
    )
    retrieval_by_pair = {
        tuple(
            sorted((int(pair["left_example_id"]), int(pair["right_example_id"])))
        ): pair
        for pair in payload.get("retrieval_pairs", [])
    }
    for superbatch in plan["superbatches"]:
        retrieval = retrieval_by_pair.get(tuple(superbatch["example_ids"]))
        if retrieval is not None:
            superbatch["retrieval"] = retrieval
    retrieval_adapter = payload.get("retrieval_adapter", {})
    source_span_count = retrieval_adapter.get("source_span_count")
    if source_span_count is not None:
        source_span_count = int(source_span_count)
        unselected_span_count = source_span_count - plan["span_count"]
        if unselected_span_count < 0:
            raise ValueError(
                "retrieval source_span_count is smaller than selected spans"
            )
        full_output_count = (
            plan["execution"]["retained_output_unit_count"] + unselected_span_count
        )
        plan["execution"].update(
            {
                "source_dataset_example_count": int(
                    retrieval_adapter["source_example_count"]
                ),
                "source_dataset_span_count": source_span_count,
                "unselected_ordinary_span_count": unselected_span_count,
                "source_dataset_output_unit_count": full_output_count,
                "source_dataset_compression_ratio": (
                    full_output_count / source_span_count
                ),
            }
        )
    plan["dataset_id"] = payload.get("dataset_id", args.input.stem)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    if args.inspect:
        print(
            json.dumps(
                {
                    "schema": plan["schema"],
                    "dataset_id": plan["dataset_id"],
                    "examples": plan["example_count"],
                    "spans": plan["span_count"],
                    "clusters": len(plan["clusters"]),
                    "superbatches": len(plan["superbatches"]),
                    "unique_spans": len(plan["unique_spans"]),
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
