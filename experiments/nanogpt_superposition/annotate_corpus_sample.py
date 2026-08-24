#!/usr/bin/env python3
"""Create ztok-aligned grammatical spans from a deterministic corpus sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import spacy
import ztok

PROTECTED_WORDS = {
    "no",
    "not",
    "never",
    "left",
    "right",
    "above",
    "below",
    "behind",
    "front",
    "before",
    "after",
    "true",
    "false",
}
PROTECTED_POS = {
    "ADP",
    "AUX",
    "CCONJ",
    "DET",
    "NUM",
    "PART",
    "PRON",
    "PUNCT",
    "SCONJ",
    "SPACE",
    "SYM",
}


def semantic_role(pos: str, dependency: str) -> str:
    if pos in {"NOUN", "PROPN", "PRON"}:
        return "entity"
    if pos in {"VERB", "AUX"}:
        return "action"
    if pos == "ADJ":
        return "attribute"
    if pos == "ADV":
        return "modifier"
    if pos == "ADP":
        return "relation"
    if pos == "NUM":
        return "count"
    if dependency == "neg":
        return "negation"
    return "syntax"


def byte_range(text: str, char_start: int, char_end: int) -> tuple[int, int]:
    return (
        len(text[:char_start].encode("utf-8")),
        len(text[:char_end].encode("utf-8")),
    )


def token_range(
    starts: list[int], ends: list[int], byte_start: int, byte_end: int
) -> tuple[int, int]:
    indices = [
        index
        for index, (start, end) in enumerate(zip(starts, ends, strict=True))
        if end > byte_start and start < byte_end
    ]
    if not indices:
        raise ValueError(f"ztok tokens do not cover [{byte_start}, {byte_end})")
    return indices[0], indices[-1] + 1


def assign_ztok_tokens(
    starts: list[int],
    ends: list[int],
    byte_spans: list[tuple[int, int]],
) -> tuple[list[list[int]], list[int]]:
    assignments = [[] for _ in byte_spans]
    unassigned = []
    for token_index, (token_start, token_end) in enumerate(
        zip(starts, ends, strict=True)
    ):
        overlaps = [
            max(0, min(token_end, span_end) - max(token_start, span_start))
            for span_start, span_end in byte_spans
        ]
        maximum = max(overlaps, default=0)
        if maximum <= 0:
            unassigned.append(token_index)
            continue
        assignments[overlaps.index(maximum)].append(token_index)
    return assignments, unassigned


def annotate_sentence(
    example_id: int, sentence: Any, pipeline: ztok.Pipeline
) -> dict[str, Any]:
    text = sentence.text.strip()
    ids, overlays = pipeline.encode_with_overlays(
        text, [ztok.OVERLAY_BYTE_START, ztok.OVERLAY_BYTE_END]
    )
    starts = overlays[ztok.OVERLAY_BYTE_START]
    ends = overlays[ztok.OVERLAY_BYTE_END]
    spans = []
    span_for_token: dict[int, str] = {}
    sentence_start = (
        sentence.start_char + len(sentence.text) - len(sentence.text.lstrip())
    )
    candidates = []
    for local_index, token in enumerate(sentence):
        if token.is_space:
            continue
        surface = token.text
        relative_start = token.idx - sentence_start
        relative_end = relative_start + len(surface)
        if relative_start < 0 or relative_end > len(text):
            continue
        start_byte, end_byte = byte_range(text, relative_start, relative_end)
        candidates.append(
            (
                local_index,
                token,
                surface,
                relative_start,
                relative_end,
                start_byte,
                end_byte,
            )
        )
    assignments, unassigned = assign_ztok_tokens(
        starts, ends, [(item[5], item[6]) for item in candidates]
    )
    for candidate, assigned in zip(candidates, assignments, strict=True):
        if not assigned:
            continue
        (
            local_index,
            token,
            surface,
            relative_start,
            relative_end,
            start_byte,
            end_byte,
        ) = candidate
        if assigned != list(range(assigned[0], assigned[-1] + 1)):
            raise ValueError("one grammatical span received non-contiguous ztok tokens")
        start_token, end_token = assigned[0], assigned[-1] + 1
        span_id = f"e{example_id}-s{local_index}"
        span_for_token[token.i] = span_id
        role = semantic_role(token.pos_, token.dep_)
        protected = (
            token.pos_ in PROTECTED_POS
            or token.dep_ == "neg"
            or token.lower_ in PROTECTED_WORDS
            or role == "syntax"
        )
        semantic_value = token.lemma_.casefold() if protected else None
        binding_type = None
        if token.pos_ in {"ADJ", "ADV"} and token.head != token:
            binding_type = semantic_role(token.head.pos_, token.head.dep_)
        spans.append(
            {
                "span_id": span_id,
                "text": surface,
                "char_start": relative_start,
                "char_end": relative_end,
                "byte_start": start_byte,
                "byte_end": end_byte,
                "token_start": start_token,
                "token_end": end_token,
                "token_byte_start": starts[start_token],
                "token_byte_end": ends[end_token - 1],
                "grammatical_role": token.dep_,
                "semantic_role": role,
                "semantic_value": semantic_value,
                "entity_type": token.ent_type_ or None,
                "binding_type": binding_type,
                "confidence": 1.0,
                "protected": protected,
                "embedding": [0.0],
            }
        )
    encoded_text = text.encode("utf-8")
    for token_index in unassigned:
        start_byte, end_byte = starts[token_index], ends[token_index]
        spans.append(
            {
                "span_id": f"e{example_id}-z{token_index}",
                "text": encoded_text[start_byte:end_byte].decode(
                    "utf-8", errors="replace"
                ),
                "byte_start": start_byte,
                "byte_end": end_byte,
                "token_start": token_index,
                "token_end": token_index + 1,
                "token_byte_start": start_byte,
                "token_byte_end": end_byte,
                "grammatical_role": "tokenizer_residual",
                "semantic_role": "syntax",
                "semantic_value": None,
                "entity_type": None,
                "binding_type": None,
                "confidence": 1.0,
                "protected": True,
                "teacher_embedding_eligible": False,
                "embedding": [0.0],
            }
        )
    spans.sort(
        key=lambda span: (span["token_start"], span["token_end"], span["span_id"])
    )
    relations = []
    for token in sentence:
        if token.is_space:
            continue
        if token.head == token:
            continue
        source = span_for_token.get(token.i)
        target = span_for_token.get(token.head.i)
        if source is None or target is None:
            continue
        relations.append(
            {
                "source_span_id": source,
                "relation": token.dep_,
                "target_span_id": target,
            }
        )
    return {
        "example_id": example_id,
        "text": text,
        "token_ids": ids,
        "spans": spans,
        "relations": relations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=32)
    parser.add_argument("--max-document-characters", type=int, default=4000)
    parser.add_argument("--max-sentence-tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=260506546)
    args = parser.parse_args()

    nlp = spacy.load("en_core_web_sm")
    parquet = pq.ParquetFile(args.parquet)
    examples = []
    with ztok.Pipeline.from_path(args.tokenizer) as pipeline:
        for batch in parquet.iter_batches(columns=["text", "id"], batch_size=64):
            for row in batch.to_pylist():
                document = nlp(str(row["text"])[: args.max_document_characters])
                for sentence in document.sents:
                    words = [token for token in sentence if not token.is_space]
                    if not 5 <= len(words) <= args.max_sentence_tokens:
                        continue
                    if sum(token.is_alpha for token in words) < 4:
                        continue
                    example = annotate_sentence(len(examples), sentence, pipeline)
                    example["source_document_id"] = str(row["id"])
                    examples.append(example)
                    break
                if len(examples) >= args.max_examples:
                    break
            if len(examples) >= args.max_examples:
                break
    payload = {
        "schema": "ztok.dataset_semantic_spans.v1",
        "dataset_id": f"fineweb-edu-structural-sample-{len(examples)}",
        "seed": args.seed,
        "annotation_adapter": {
            "model": "en_core_web_sm",
            "version": "3.8.0",
            "segmentation": "spacy_tokens_mapped_to_ztok_byte_offsets",
            "semantic_roles": "deterministic_pos_dependency_rules",
        },
        "source": {
            "path": str(args.parquet.resolve()),
            "tokenizer": str(args.tokenizer.resolve()),
        },
        "examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"dataset_id": payload["dataset_id"], "examples": len(examples)}))


if __name__ == "__main__":
    main()
