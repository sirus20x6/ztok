"""Tests for Pipeline.encode_with_overlays (ztok_encode_with_overlays)."""

from __future__ import annotations

import pytest

import ztok


def test_ids_match_plain_encode(bpe_pipeline: ztok.Pipeline) -> None:
    text = "hello world"
    plain = bpe_pipeline.encode(text)
    ids, overlays = bpe_pipeline.encode_with_overlays(
        text, [ztok.OVERLAY_BYTE_START, ztok.OVERLAY_BYTE_END]
    )
    # Requesting overlays must not change tokenization.
    assert ids == plain
    assert set(overlays.keys()) == {ztok.OVERLAY_BYTE_START, ztok.OVERLAY_BYTE_END}


def test_channel_lengths_equal_ids(bpe_pipeline: ztok.Pipeline) -> None:
    text = "the quick brown fox"
    ids, overlays = bpe_pipeline.encode_with_overlays(
        text,
        [
            ztok.OVERLAY_BYTE_START,
            ztok.OVERLAY_BYTE_END,
            ztok.OVERLAY_BOUNDARY,
            ztok.OVERLAY_PROVENANCE,
        ],
    )
    for kind, values in overlays.items():
        assert len(values) == len(ids), f"channel {kind} length mismatch"


def test_byte_spans_are_sensible(bpe_pipeline: ztok.Pipeline) -> None:
    text = "hello world"
    ids, overlays = bpe_pipeline.encode_with_overlays(
        text, [ztok.OVERLAY_BYTE_START, ztok.OVERLAY_BYTE_END]
    )
    starts = overlays[ztok.OVERLAY_BYTE_START]
    ends = overlays[ztok.OVERLAY_BYTE_END]
    n = len(text.encode("utf-8"))
    # Each span is non-empty, in-bounds, and end > start.
    for s, e in zip(starts, ends):
        assert 0 <= s < e <= n, f"bad span ({s}, {e}) for input of {n} bytes"
    # Spans tile the input left-to-right: first starts at 0, last ends at n,
    # and each token picks up exactly where the previous left off.
    assert starts[0] == 0
    assert ends[-1] == n
    for prev_end, s in zip(ends, starts[1:]):
        assert s == prev_end


def test_opcode_domain_channel_is_all_zero(bpe_pipeline: ztok.Pipeline) -> None:
    # No domain normalizer plugin is loaded, so OPCODE is zero-filled.
    text = "hello world"
    ids, overlays = bpe_pipeline.encode_with_overlays(text, [ztok.OVERLAY_OPCODE])
    opcode = overlays[ztok.OVERLAY_OPCODE]
    assert len(opcode) == len(ids)
    assert all(v == 0 for v in opcode)


def test_byte_id_single_byte_spans() -> None:
    with ztok.Pipeline.byte_id() as pipe:
        ids, overlays = pipe.encode_with_overlays(
            "hi", [ztok.OVERLAY_BYTE_START, ztok.OVERLAY_BYTE_END]
        )
        assert ids == [0x68, 0x69]
        assert overlays[ztok.OVERLAY_BYTE_START] == [0, 1]
        assert overlays[ztok.OVERLAY_BYTE_END] == [1, 2]


def test_empty_input_returns_empty_channels() -> None:
    with ztok.Pipeline.byte_id() as pipe:
        ids, overlays = pipe.encode_with_overlays(
            "", [ztok.OVERLAY_BYTE_START, ztok.OVERLAY_OPCODE]
        )
        assert ids == []
        assert overlays == {ztok.OVERLAY_BYTE_START: [], ztok.OVERLAY_OPCODE: []}


def test_no_channels_returns_just_ids(bpe_pipeline: ztok.Pipeline) -> None:
    ids, overlays = bpe_pipeline.encode_with_overlays("hello world", [])
    assert ids == bpe_pipeline.encode("hello world")
    assert overlays == {}
