"""Streaming encode tests for the Python binding.

Wraps the post-1.18 C ABI's `ztok_stream_*` family. The streaming output
should match `Pipeline.encode` byte-for-byte when the pre-tokenizer is a
real one (cl100k splits on whitespace, giving safe-cut boundaries on
every span) and may differ ONLY at unsafe chunk seams with the identity
pre-tokenizer + BPE-merge models — same caveat documented in
src/stream.zig.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import ztok


def _collect(stream) -> list[int]:
    """Materialize a generator of id lists into a single flat list."""

    out: list[int] = []
    for batch in stream:
        assert isinstance(batch, list)
        out.extend(batch)
    return out


def test_stream_single_chunk_matches_encode(bpe_pipeline: ztok.Pipeline) -> None:
    text = "hello world the quick brown fox"
    want = bpe_pipeline.encode(text)
    got = _collect(bpe_pipeline.encode_stream(text))
    assert got == want


def test_stream_many_small_chunks_matches_encode(bpe_pipeline: ztok.Pipeline) -> None:
    # Force lots of feeds by setting a tiny chunk size — the encoder
    # should still emit ids identical to a single-shot encode because
    # the cl100k pre-tokenizer finds safe whitespace cuts on every span.
    text = "hello world the quick brown fox hello world the quick brown fox"
    want = bpe_pipeline.encode(text)
    got = _collect(bpe_pipeline.encode_stream(text, chunk_size=4))
    assert got == want


def test_stream_mid_utf8_codepoint_is_deferred() -> None:
    # "héllo" — the é is 2 bytes (0xC3 0xA9). Feeding the bytes in two
    # halves where the cut falls between the two bytes of é must not
    # produce a garbage id; the stream encoder defers the trailing 0xC3
    # to the next feed.
    with ztok.Pipeline.byte_id() as pipe:
        text = "h\xe9llo"
        # bytes manually to control the cut precisely.
        raw = text.encode("utf-8")
        want = pipe.encode(text)

        # Drive the underlying C API directly to control the cut point.
        # encode_stream(chunk_size=2) chops at byte 2 (mid-é).
        got = _collect(pipe.encode_stream(text, chunk_size=2))
        assert got == want
        assert len(raw) == 6  # "h" + 0xC3 0xA9 + "llo"


def test_stream_empty_input_yields_nothing() -> None:
    with ztok.Pipeline.byte_id() as pipe:
        out = list(pipe.encode_stream(""))
        # Empty input produces zero non-empty yields.
        assert out == []


def test_stream_multiline_matches_encode(bpe_pipeline: ztok.Pipeline) -> None:
    text = "hello world\nthe quick brown fox\nfoo bar baz\n"
    want = bpe_pipeline.encode(text)
    got = _collect(bpe_pipeline.encode_stream(text, chunk_size=8))
    assert got == want
