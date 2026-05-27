"""RWKV "World" tokenizer tests for the Python binding.

Loads the real rwkv_vocab_v20230424.txt fixture (skipped when absent)
and checks ztok reproduces the canonical reference encodings, matching
the in-tree gate in src/rwkv_world.zig.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import ztok

# bindings/python/tests/ -> repo root -> bench/vocabs/...
_VOCAB = (
    Path(__file__).resolve().parents[3]
    / "bench"
    / "vocabs"
    / "rwkv_vocab_v20230424.txt"
)


@pytest.fixture(scope="module")
def rwkv_pipeline() -> ztok.Pipeline:
    if not _VOCAB.exists():
        pytest.skip(f"RWKV vocab fixture not present at {_VOCAB}")
    pipe = ztok.Pipeline.from_rwkv(str(_VOCAB))
    yield pipe
    pipe.close()


# Golden id sequences captured from BlinkDL's canonical reference
# tokenizer (see bench/rwkv_parity.py / src/rwkv_world.zig).
GOLDEN = [
    ("Hello, world!", [33155, 45, 40213, 34]),
    ("emoji 😀🚀✨ test", [34295, 33, 3319, 153, 129, 3319, 155, 129, 10059, 32223]),
    ("0 1 2 10 99 100", [49, 284, 285, 3483, 3572, 3483, 49]),
]


@pytest.mark.parametrize("text, want", GOLDEN)
def test_rwkv_matches_reference(rwkv_pipeline: ztok.Pipeline, text, want) -> None:
    assert rwkv_pipeline.encode(text) == want


def test_rwkv_round_trips(rwkv_pipeline: ztok.Pipeline) -> None:
    for text, _ in GOLDEN:
        ids = rwkv_pipeline.encode(text)
        assert rwkv_pipeline.decode(ids) == text
