#!/usr/bin/env python3
"""Prepare deterministic document-bounded ztok training arrays."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


@dataclass(frozen=True)
class SourceDocument:
    text: bytes
    corpus_tokens: int | None = None


def _open_binary(path: Path):
    if path.suffix != ".zst":
        return path.open("rb")
    try:
        import zstandard
    except ImportError as error:
        raise RuntimeError(".zst input requires the zstandard package") from error
    source = path.open("rb")
    reader = zstandard.ZstdDecompressor().stream_reader(source)
    buffered = io.BufferedReader(reader)

    class ManagedReader:
        def __enter__(self):
            return buffered

        def __exit__(self, *_):
            buffered.close()
            source.close()

    return ManagedReader()


def _iter_documents(
    paths: list[Path],
    *,
    line_documents: bool,
    jsonl_text_field: str | None,
    parquet_text_field: str | None = None,
    parquet_token_field: str | None = None,
) -> Iterator[SourceDocument]:
    for path in paths:
        if path.suffix == ".parquet":
            if parquet_text_field is None:
                raise RuntimeError("Parquet input requires --parquet-text-field")
            try:
                import pyarrow.parquet as pq
            except ImportError as error:
                raise RuntimeError("Parquet input requires pyarrow") from error
            columns = [parquet_text_field]
            token_root = (
                parquet_token_field.split(".", 1)[0] if parquet_token_field else None
            )
            if token_root and token_root not in columns:
                columns.append(token_root)
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(columns=columns, batch_size=1024):
                for row in batch.to_pylist():
                    value = row[parquet_text_field]
                    document = (
                        value.encode("utf-8")
                        if isinstance(value, str)
                        else bytes(value)
                    )
                    if not document.strip():
                        continue
                    corpus_tokens = None
                    if parquet_token_field:
                        token_value = row
                        for component in parquet_token_field.split("."):
                            token_value = token_value[component]
                        corpus_tokens = int(token_value)
                    yield SourceDocument(document, corpus_tokens)
            continue
        with _open_binary(path) as source:
            if jsonl_text_field is not None:
                for line in source:
                    if not line.strip():
                        continue
                    value = json.loads(line)[jsonl_text_field]
                    document = (
                        value.encode("utf-8")
                        if isinstance(value, str)
                        else bytes(value)
                    )
                    if document.strip():
                        yield SourceDocument(document)
            elif line_documents:
                for document in source:
                    if document.strip():
                        yield SourceDocument(document)
            else:
                document = source.read()
                if document.strip():
                    yield SourceDocument(document)


def _split(document: bytes, seed: int, validation_fraction: float) -> str:
    digest = hashlib.sha256(seed.to_bytes(8, "little") + document).digest()
    value = int.from_bytes(digest[:8], "little") / 2**64
    return "validation" if value < validation_fraction else "train"


def _tokenizer(path: Path | None):
    import ztok

    return ztok.Pipeline.byte_id() if path is None else ztok.Pipeline.from_path(path)


def _encode_document(
    pipeline, document: bytes
) -> tuple[list[int], list[int], list[int]]:
    import ztok

    ids, overlays = pipeline.encode_with_overlays(
        document, [ztok.OVERLAY_BYTE_START, ztok.OVERLAY_BYTE_END]
    )
    starts = overlays[ztok.OVERLAY_BYTE_START]
    ends = overlays[ztok.OVERLAY_BYTE_END]
    if not (len(ids) == len(starts) == len(ends)):
        raise RuntimeError("ztok returned misaligned offsets")
    if any(end < start or end > len(document) for start, end in zip(starts, ends)):
        raise RuntimeError("ztok returned an invalid byte span")
    return ids, starts, ends


def _write_split(output: Path, name: str, rows: list[tuple]) -> dict:
    token_count = sum(len(row[0]) for row in rows)
    tokens = np.empty(token_count, dtype=np.uint32)
    byte_starts = np.empty(token_count, dtype=np.uint32)
    byte_ends = np.empty(token_count, dtype=np.uint32)
    documents = np.empty((len(rows), 4), dtype=np.uint64)
    cursor = 0
    source_bytes = 0
    hashes = hashlib.sha256()
    for index, (ids, starts, ends, byte_count, digest) in enumerate(rows):
        length = len(ids)
        tokens[cursor : cursor + length] = ids
        byte_starts[cursor : cursor + length] = starts
        byte_ends[cursor : cursor + length] = ends
        documents[index] = (cursor, cursor + length, byte_count, int(digest[:16], 16))
        cursor += length
        source_bytes += byte_count
        hashes.update(bytes.fromhex(digest))
    tokens.tofile(output / f"{name}.tokens.bin")
    byte_starts.tofile(output / f"{name}.byte_starts.bin")
    byte_ends.tofile(output / f"{name}.byte_ends.bin")
    np.save(output / f"{name}.documents.npy", documents, allow_pickle=False)
    return {
        "documents": len(rows),
        "tokens": token_count,
        "source_bytes": source_bytes,
        "bytes_per_token": source_bytes / token_count if token_count else 0.0,
        "split_hash": hashes.hexdigest(),
    }


class SplitWriter:
    def __init__(self, output: Path, name: str) -> None:
        self.output = output
        self.name = name
        self.tokens = (output / f"{name}.tokens.bin").open("wb")
        self.byte_starts = (output / f"{name}.byte_starts.bin").open("wb")
        self.byte_ends = (output / f"{name}.byte_ends.bin").open("wb")
        self.documents: list[tuple[int, int, int, int]] = []
        self.token_count = 0
        self.source_bytes = 0
        self.split_hash = hashlib.sha256()

    def append(
        self,
        ids: list[int],
        starts: list[int],
        ends: list[int],
        byte_count: int,
        digest: str,
    ) -> None:
        begin = self.token_count
        np.asarray(ids, dtype=np.uint32).tofile(self.tokens)
        np.asarray(starts, dtype=np.uint32).tofile(self.byte_starts)
        np.asarray(ends, dtype=np.uint32).tofile(self.byte_ends)
        self.token_count += len(ids)
        self.source_bytes += byte_count
        self.documents.append(
            (begin, self.token_count, byte_count, int(digest[:16], 16))
        )
        self.split_hash.update(bytes.fromhex(digest))

    def finish(self) -> dict:
        self.tokens.close()
        self.byte_starts.close()
        self.byte_ends.close()
        np.save(
            self.output / f"{self.name}.documents.npy",
            np.asarray(self.documents, dtype=np.uint64).reshape(-1, 4),
            allow_pickle=False,
        )
        return {
            "documents": len(self.documents),
            "tokens": self.token_count,
            "source_bytes": self.source_bytes,
            "bytes_per_token": (
                self.source_bytes / self.token_count if self.token_count else 0.0
            ),
            "split_hash": self.split_hash.hexdigest(),
        }

    def close(self) -> None:
        for handle in (self.tokens, self.byte_starts, self.byte_ends):
            if not handle.closed:
                handle.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--tiny-shakespeare", action="store_true")
    parser.add_argument("--line-documents", action="store_true")
    parser.add_argument(
        "--jsonl-text-field",
        help="stream JSONL or JSONL.zst and read this field as each document",
    )
    parser.add_argument(
        "--parquet-text-field",
        help="stream this Parquet string column as documents",
    )
    parser.add_argument(
        "--parquet-token-field",
        help="optional dotted Parquet field containing source-corpus token counts",
    )
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument(
        "--tokenizer-kind", choices=("bpe", "superbpe", "byte"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=260506546)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--language", default="en")
    parser.add_argument("--max-documents", type=int)
    parser.add_argument(
        "--max-corpus-tokens",
        type=int,
        help="stop before the first document that would exceed this source token budget",
    )
    parser.add_argument(
        "--max-source-bytes",
        type=int,
        help="stop before the first document that would exceed this byte budget",
    )
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1:
        raise SystemExit("--validation-fraction must be in (0,1)")
    inputs = list(args.input)
    if args.tiny_shakespeare:
        cache = args.output.parent / "tiny_shakespeare.txt"
        if not cache.exists():
            cache.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, cache)
        inputs.append(cache)
    if not inputs:
        raise SystemExit("provide --input or --tiny-shakespeare")
    if args.tokenizer_kind != "byte" and args.tokenizer is None:
        raise SystemExit("BPE and SuperBPE preparation require --tokenizer")

    args.output.mkdir(parents=True, exist_ok=True)
    writers = {name: SplitWriter(args.output, name) for name in ("train", "validation")}
    maximum_id = 0
    document_order_hash = hashlib.sha256()
    source_document_count = 0
    source_corpus_tokens = 0
    selected_source_bytes = 0
    try:
        with _tokenizer(args.tokenizer) as pipeline:
            documents = _iter_documents(
                inputs,
                line_documents=args.line_documents,
                jsonl_text_field=args.jsonl_text_field,
                parquet_text_field=args.parquet_text_field,
                parquet_token_field=args.parquet_token_field,
            )
            for source in documents:
                document = source.text
                if (
                    args.max_documents is not None
                    and source_document_count >= args.max_documents
                ):
                    break
                if args.max_corpus_tokens is not None and source.corpus_tokens is None:
                    raise RuntimeError(
                        "--max-corpus-tokens requires --parquet-token-field"
                    )
                if (
                    args.max_corpus_tokens is not None
                    and source_document_count
                    and source_corpus_tokens + source.corpus_tokens
                    > args.max_corpus_tokens
                ):
                    break
                if (
                    args.max_source_bytes is not None
                    and source_document_count
                    and selected_source_bytes + len(document) > args.max_source_bytes
                ):
                    break
                source_document_count += 1
                source_corpus_tokens += source.corpus_tokens or 0
                selected_source_bytes += len(document)
                digest = hashlib.sha256(document).hexdigest()
                document_order_hash.update(bytes.fromhex(digest))
                ids, starts, ends = _encode_document(pipeline, document)
                if len(ids) < 2:
                    continue
                split = _split(document, args.seed, args.validation_fraction)
                writers[split].append(ids, starts, ends, len(document), digest)
                maximum_id = max(maximum_id, max(ids))
        split_metadata = {name: writer.finish() for name, writer in writers.items()}
    finally:
        for writer in writers.values():
            writer.close()
    if (
        not split_metadata["train"]["documents"]
        or not split_metadata["validation"]["documents"]
    ):
        raise SystemExit(
            "deterministic split produced an empty partition; add documents or adjust --validation-fraction"
        )
    tokenizer_hash = None
    if args.tokenizer is not None:
        tokenizer_hash = hashlib.sha256(args.tokenizer.read_bytes()).hexdigest()
    metadata = {
        "schema": "ztok.nanogpt_superposition.dataset.v1",
        "seed": args.seed,
        "tokenizer": {
            "kind": args.tokenizer_kind,
            "path": str(args.tokenizer.resolve()) if args.tokenizer else None,
            "sha256": tokenizer_hash,
            "vocabulary_size": maximum_id + 1,
        },
        "source": {
            "paths": [str(path.resolve()) for path in inputs],
            "document_order_hash": document_order_hash.hexdigest(),
            "document_count": source_document_count,
            "corpus_token_count": source_corpus_tokens or None,
            "selected_source_bytes": selected_source_bytes,
            "language_distribution": {args.language: 1.0},
        },
        "splits": split_metadata,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
