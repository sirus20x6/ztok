#!/usr/bin/env python3
"""Attach frozen-teacher contextual span vectors to a dataset exchange file."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"


def find_surface(text: str, span: dict[str, Any]) -> tuple[int, int]:
    if "char_start" in span and "char_end" in span:
        start, end = int(span["char_start"]), int(span["char_end"])
        if not 0 <= start < end <= len(text):
            raise ValueError(f"invalid character range for {span['span_id']}")
        return start, end
    surface = str(span.get("text") or str(span["span_id"]).split("-", 1)[-1])
    start = text.casefold().find(surface.casefold())
    if start < 0:
        raise ValueError(f"cannot locate {surface!r} for {span['span_id']}")
    return start, start + len(surface)


def overlapping_token_indices(
    offsets: list[tuple[int, int]], char_start: int, char_end: int
) -> list[int]:
    indices = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > start and end > char_start and start < char_end
    ]
    if not indices:
        raise ValueError(f"teacher tokens do not cover [{char_start}, {char_end})")
    return indices


def teacher_embedding_eligible(span: dict[str, Any]) -> bool:
    """Whether a span denotes complete Unicode text suitable for the teacher."""
    return bool(span.get("teacher_embedding_eligible", True))


def contextualize(
    payload: dict[str, Any],
    *,
    model_name: str,
    revision: str,
    layers: tuple[int, ...],
    device: str,
    representation: str = "contextual_hidden_mean",
    batch_size: int = 16,
) -> dict[str, Any]:
    if not layers:
        raise ValueError("at least one teacher layer is required")
    if representation not in {"contextual_hidden_mean", "semantic_prompt_last_token"}:
        raise ValueError("unsupported teacher representation")
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
    output = copy.deepcopy(payload)
    hidden_size = int(model.config.hidden_size)
    resolved_layers: tuple[int, ...] | None = None
    with torch.inference_mode():
        if representation == "contextual_hidden_mean":
            for example in output["examples"]:
                text = str(example["text"])
                encoded = tokenizer(
                    text,
                    return_offsets_mapping=True,
                    return_tensors="pt",
                    truncation=False,
                )
                offsets = [
                    tuple(map(int, pair)) for pair in encoded.pop("offset_mapping")[0]
                ]
                inputs = {key: value.to(device) for key, value in encoded.items()}
                model_output = model(
                    **inputs, output_hidden_states=True, return_dict=True
                )
                hidden_states = model_output.hidden_states
                resolved = tuple(
                    layer if layer >= 0 else len(hidden_states) + layer
                    for layer in layers
                )
                if any(index < 0 or index >= len(hidden_states) for index in resolved):
                    raise ValueError(
                        f"layers {layers} resolve outside hidden-state depth"
                    )
                if resolved_layers is None:
                    resolved_layers = resolved
                elif resolved_layers != resolved:
                    raise ValueError("teacher returned inconsistent hidden-state depth")
                for span in example["spans"]:
                    if not teacher_embedding_eligible(span):
                        span["embedding"] = [0.0] * hidden_size
                        continue
                    char_start, char_end = find_surface(text, span)
                    indices = overlapping_token_indices(offsets, char_start, char_end)
                    vector = (
                        torch.stack(
                            [
                                hidden_states[index][0, indices].float().mean(dim=0)
                                for index in resolved
                            ]
                        )
                        .mean(dim=0)
                        .cpu()
                    )
                    span["char_start"] = char_start
                    span["char_end"] = char_end
                    span["embedding"] = vector.tolist()
        else:
            records = []
            for example in output["examples"]:
                text = str(example["text"])
                for span in example["spans"]:
                    if not teacher_embedding_eligible(span):
                        span["embedding"] = [0.0] * hidden_size
                        continue
                    char_start, char_end = find_surface(text, span)
                    span["char_start"] = char_start
                    span["char_end"] = char_end
                    prompt = (
                        "Classify the meaning and role of the marked span. "
                        f"Sentence: {text}\nMarked span: {text[char_start:char_end]}"
                    )
                    records.append((span, prompt))
            for start in range(0, len(records), batch_size):
                batch = records[start : start + batch_size]
                encoded = tokenizer(
                    [prompt for _, prompt in batch],
                    padding=True,
                    return_tensors="pt",
                    truncation=False,
                ).to(device)
                model_output = model(
                    **encoded, output_hidden_states=True, return_dict=True
                )
                hidden_states = model_output.hidden_states
                resolved = tuple(
                    layer if layer >= 0 else len(hidden_states) + layer
                    for layer in layers
                )
                if any(index < 0 or index >= len(hidden_states) for index in resolved):
                    raise ValueError(
                        f"layers {layers} resolve outside hidden-state depth"
                    )
                if resolved_layers is None:
                    resolved_layers = resolved
                elif resolved_layers != resolved:
                    raise ValueError("teacher returned inconsistent hidden-state depth")
                vectors = (
                    torch.stack(
                        [hidden_states[index][:, -1].float() for index in resolved]
                    )
                    .mean(dim=0)
                    .cpu()
                )
                for (span, _), vector in zip(batch, vectors, strict=True):
                    span["embedding"] = vector.tolist()
    output["embedding_adapter"] = {
        "model": model_name,
        "revision": revision,
        "requested_layers": list(layers),
        "resolved_hidden_state_indices": list(resolved_layers or ()),
        "representation": representation,
        "pooling": (
            "mean_teacher_tokens_overlapping_character_span"
            if representation == "contextual_hidden_mean"
            else "last_nonpadding_token_of_span_in_context_semantic_prompt"
        ),
        "prompt": (
            None
            if representation == "contextual_hidden_mean"
            else "Classify the meaning and role of the marked span. Sentence: {text}\\nMarked span: {span}"
        ),
        "dtype": str(dtype).removeprefix("torch."),
        "device": device,
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--layer", type=int, action="append")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--representation",
        choices=("contextual_hidden_mean", "semantic_prompt_last_token"),
        default="contextual_hidden_mean",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    result = contextualize(
        json.loads(args.input.read_text()),
        model_name=args.model,
        revision=args.revision,
        layers=tuple(args.layer or (-1,)),
        device=args.device,
        representation=args.representation,
        batch_size=args.batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["embedding_adapter"], indent=2))


if __name__ == "__main__":
    main()
