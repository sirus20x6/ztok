"""BatchPool encode-batch tests, including leak checks via weakref."""

from __future__ import annotations

import gc
import weakref

import pytest

import ztok


def test_batch_pool_workers_resolves_auto() -> None:
    with ztok.BatchPool(workers=0) as pool:
        assert pool.workers >= 1


def test_batch_pool_explicit_worker_count() -> None:
    with ztok.BatchPool(workers=3) as pool:
        assert pool.workers == 3


def test_batch_pool_closed_pool_raises() -> None:
    pool = ztok.BatchPool(workers=2)
    pool.close()
    with pytest.raises(ztok.ZtokError):
        _ = pool.workers


def test_batch_encode_matches_single_encode(bpe_pipeline: ztok.Pipeline) -> None:
    inputs = ["hello world", " the quick brown fox", "foo bar baz"]
    expected = [bpe_pipeline.encode(s) for s in inputs]
    with ztok.BatchPool(workers=4) as pool:
        got = bpe_pipeline.encode_batch(pool, inputs)
    assert got == expected


def test_batch_encode_1000_strings_no_leaks(bpe_pipeline: ztok.Pipeline) -> None:
    inputs = ["hello world"] * 1000

    class _Holder:
        # Plain `list` doesn't support weakref; wrap so we can prove
        # the results object itself is reclaimable after we drop it.
        __slots__ = ("data", "__weakref__")

        def __init__(self, data):  # type: ignore[no-untyped-def]
            self.data = data

    with ztok.BatchPool(workers=8) as pool:
        holder = _Holder(bpe_pipeline.encode_batch(pool, inputs))
    assert len(holder.data) == 1000
    assert all(r == holder.data[0] for r in holder.data)
    # Weakref-style leak guard: the per-id-buffer C allocations were
    # freed inside encode_batch (via ztok_ids_free). Python-side, the
    # only references are the plain int lists we returned, which the
    # garbage collector reclaims cleanly once we drop the holder.
    ref = weakref.ref(holder)
    del holder
    gc.collect()
    assert ref() is None


def test_batch_encode_empty_inputs(bpe_pipeline: ztok.Pipeline) -> None:
    with ztok.BatchPool(workers=2) as pool:
        assert bpe_pipeline.encode_batch(pool, []) == []


def test_batch_encode_with_empty_string(bpe_pipeline: ztok.Pipeline) -> None:
    inputs = ["hello", "", "world"]
    with ztok.BatchPool(workers=2) as pool:
        results = bpe_pipeline.encode_batch(pool, inputs)
    assert len(results) == 3
    assert results[1] == []  # empty input -> empty ids
    assert results[0] == bpe_pipeline.encode("hello")
    assert results[2] == bpe_pipeline.encode("world")


def test_pool_finalizer_releases_on_gc() -> None:
    # Drop the pool without explicit close; the weakref.finalize hook
    # should still tear down the C worker pool when Python collects it.
    pool = ztok.BatchPool(workers=2)
    finalizer_alive = pool._finalizer.alive
    assert finalizer_alive
    ref = weakref.ref(pool)
    del pool
    gc.collect()
    assert ref() is None
