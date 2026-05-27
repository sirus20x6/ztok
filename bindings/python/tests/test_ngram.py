"""Engram n-gram hashing tests for the Python binding.

Mirrors src/ngram.zig's contract: deterministic multi-head token-n-gram
hashes, row-major [position][head], with positions = len(ids) - n + 1.
"""

from __future__ import annotations

import ztok


def test_ngram_length_math() -> None:
    ids = [1, 2, 3, 4, 5]
    # 5 ids, n=2 -> 4 positions; heads=3 -> 12 hashes.
    out = ztok.ngram_hash(ids, n=2, heads=3)
    assert len(out) == 4 * 3


def test_ngram_deterministic() -> None:
    ids = [7, 8, 9, 10, 11, 12]
    a = ztok.ngram_hash(ids, n=3, heads=4)
    b = ztok.ngram_hash(ids, n=3, heads=4)
    assert a == b
    assert all(isinstance(h, int) and h >= 0 for h in a)


def test_ngram_head_independence() -> None:
    # The heads of a single position should not all collide.
    out = ztok.ngram_hash([42, 43, 44], n=2, heads=4)
    first_position = out[:4]
    assert len(set(first_position)) > 1


def test_ngram_short_and_bad_args() -> None:
    # Stream shorter than one window -> empty.
    assert ztok.ngram_hash([1, 2], n=3, heads=2) == []
    # Degenerate args -> empty (no error).
    assert ztok.ngram_hash([], n=1, heads=1) == []
    assert ztok.ngram_hash([1, 2, 3], n=0, heads=1) == []
    assert ztok.ngram_hash([1, 2, 3], n=2, heads=0) == []


def test_ngram_batch_matches_single() -> None:
    streams = [
        [1, 2, 3, 4],
        [],            # empty -> no hashes
        [9],           # shorter than window -> no hashes
        [5, 6, 7, 8, 9],
    ]
    with ztok.BatchPool(workers=2) as pool:
        batched = ztok.ngram_hash_batch(pool, streams, n=2, heads=3)
    assert len(batched) == len(streams)
    for s, got in zip(streams, batched):
        assert got == ztok.ngram_hash(s, n=2, heads=3)
