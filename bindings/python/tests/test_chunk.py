"""Token-window chunking tests for the Python binding.

Run over a byte_id pipeline (each input byte = one token) so chunk
boundaries are predictable: "abcdefghij" is 10 tokens, one per byte.
"""

from __future__ import annotations

import pytest

import ztok


@pytest.fixture
def byte_pipeline() -> ztok.Pipeline:
    pipe = ztok.Pipeline.byte_id()
    yield pipe
    pipe.close()


def test_chunk_non_overlapping(byte_pipeline: ztok.Pipeline) -> None:
    chunks = byte_pipeline.chunk("abcdefghij", max_tokens=4, overlap=0)
    # 10 tokens / window 4, stride 4 -> [0,4) [4,8) [8,10).
    assert len(chunks) == 3
    want = [(0, 4, 0, 4, 4), (4, 8, 4, 8, 4), (8, 10, 8, 10, 2)]
    for c, (ts, te, bs, be, n) in zip(chunks, want):
        assert (c.token_start, c.token_end) == (ts, te)
        assert (c.byte_start, c.byte_end) == (bs, be)
        assert len(c.ids) == n


def test_chunk_overlap(byte_pipeline: ztok.Pipeline) -> None:
    chunks = byte_pipeline.chunk("abcdefghij", max_tokens=4, overlap=2)
    assert len(chunks) >= 2
    # stride = 2, so the last 2 ids of chunk[i] equal the first 2 of
    # chunk[i+1].
    for a, b in zip(chunks, chunks[1:]):
        if len(a.ids) >= 2 and len(b.ids) >= 2:
            assert a.ids[-2:] == b.ids[:2]


def test_chunk_empty_and_bad_args(byte_pipeline: ztok.Pipeline) -> None:
    assert byte_pipeline.chunk("", max_tokens=4) == []
    with pytest.raises(ztok.ZtokInvalidInputError):
        byte_pipeline.chunk("abc", max_tokens=0)
    with pytest.raises(ztok.ZtokInvalidInputError):
        byte_pipeline.chunk("abc", max_tokens=4, overlap=4)
