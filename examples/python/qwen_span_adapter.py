#!/usr/bin/env python3
"""Export contextual Qwen span vectors as ztok.semantic_spans.v1 JSONL.

This optional adapter lives outside tokenizer core. The ztok tokenizer file
must correspond exactly to the Qwen vocabulary so the IDs passed to the model
are valid. Record the model revision when producing benchmark data.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel

import ztok


def overlapping_token_range(
    starts: list[int], ends: list[int], byte_start: int, byte_end: int
) -> tuple[int, int]:
    indices = [
        index
        for index, (start, end) in enumerate(zip(starts, ends))
        if end > byte_start and start < byte_end
    ]
    if not indices:
        raise ValueError(f"byte span [{byte_start},{byte_end}) has no token overlap")
    return indices[0], indices[-1] + 1


def pool(hidden: torch.Tensor, token_start: int, token_end: int) -> list[float]:
    return (
        hidden[token_start:token_end]
        .float()
        .mean(dim=0)
        .cpu()
        .tolist()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--layers", default="18,30")
    parser.add_argument("--average-layers", default="18,24,30")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    args = parser.parse_args()

    layers = [int(item) for item in args.layers.split(",") if item]
    average_layers = [
        int(item) for item in args.average_layers.split(",") if item
    ]
    if not layers or not average_layers:
        raise SystemExit("--layers and --average-layers must each be non-empty")
    requested = sorted(set(layers + average_layers))
    dtype = getattr(torch, args.dtype)
    model = AutoModel.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(args.device)
    model.eval()
    pipeline = ztok.Pipeline.from_path(args.tokenizer)
    outputs = {
        f"layer-{layer}": args.output_prefix.with_suffix(f".layer-{layer}.jsonl")
        for layer in layers
    }
    outputs[f"average-{'-'.join(map(str, average_layers))}"] = (
        args.output_prefix.with_suffix(
            f".average-{'-'.join(map(str, average_layers))}.jsonl"
        )
    )
    handles = {name: path.open("w") for name, path in outputs.items()}
    try:
        with args.input.open() as source, torch.inference_mode():
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                per_representation = {
                    name: {
                        "schema": "ztok.semantic_spans.v1",
                        "image_id": record["image_id"],
                        "embedding_adapter": {
                            "model": args.model,
                            "revision": args.revision,
                            "representation": name,
                            "pooling": "mean over ztok byte-overlap token range",
                        },
                        "captions": [],
                    }
                    for name in handles
                }
                original_token_count = 0
                for caption in record["captions"]:
                    text = caption["text"]
                    ids, overlays = pipeline.encode_with_overlays(
                        text,
                        [
                            ztok.OVERLAY_BYTE_START,
                            ztok.OVERLAY_BYTE_END,
                        ],
                    )
                    starts = overlays[ztok.OVERLAY_BYTE_START]
                    ends = overlays[ztok.OVERLAY_BYTE_END]
                    original_token_count += len(ids)
                    input_ids = torch.tensor(
                        ids, device=args.device, dtype=torch.long
                    ).unsqueeze(0)
                    result = model(input_ids=input_ids, output_hidden_states=True)
                    hidden_states = result.hidden_states
                    if max(requested) >= len(hidden_states):
                        raise ValueError(
                            f"requested layer {max(requested)} but model exposed "
                            f"{len(hidden_states)} hidden-state tensors"
                        )
                    average_hidden = torch.stack(
                        [hidden_states[layer][0] for layer in average_layers]
                    ).mean(dim=0)
                    tokens = [
                        {
                            "token_index": index,
                            "token_id": token_id,
                            "byte_start": starts[index],
                            "byte_end": ends[index],
                        }
                        for index, token_id in enumerate(ids)
                    ]
                    caption_outputs = {}
                    for name in handles:
                        caption_outputs[name] = {
                            "caption_id": caption["caption_id"],
                            "section": caption.get("section"),
                            "text": text,
                            "tokens": tokens,
                            "spans": [],
                        }
                    for source_span in caption.get("spans", []):
                        token_start, token_end = overlapping_token_range(
                            starts,
                            ends,
                            source_span["byte_start"],
                            source_span["byte_end"],
                        )
                        for layer in layers:
                            span = copy.deepcopy(source_span)
                            span["token_start"] = token_start
                            span["token_end"] = token_end
                            span["embedding"] = pool(
                                hidden_states[layer][0], token_start, token_end
                            )
                            caption_outputs[f"layer-{layer}"]["spans"].append(span)
                        average_name = (
                            f"average-{'-'.join(map(str, average_layers))}"
                        )
                        span = copy.deepcopy(source_span)
                        span["token_start"] = token_start
                        span["token_end"] = token_end
                        span["embedding"] = pool(
                            average_hidden, token_start, token_end
                        )
                        caption_outputs[average_name]["spans"].append(span)
                    for name, caption_output in caption_outputs.items():
                        if caption_output["section"] is None:
                            del caption_output["section"]
                        per_representation[name]["captions"].append(
                            caption_output
                        )
                for name, document in per_representation.items():
                    document["original_token_count"] = original_token_count
                    handles[name].write(
                        json.dumps(document, ensure_ascii=False) + "\n"
                    )
                print(f"processed line {line_number}", flush=True)
    finally:
        pipeline.close()
        for handle in handles.values():
            handle.close()


if __name__ == "__main__":
    main()
