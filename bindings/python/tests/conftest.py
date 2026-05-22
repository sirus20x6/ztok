"""Shared pytest fixtures for the ztok Python binding tests.

We build a tiny synthetic .tiktoken vocab covering all 256 single bytes
plus a handful of merges (matching the fixture used in src/c_api.zig's
own tests). Saved once per test session.
"""

from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path

import pytest

import ztok


_EXTRAS = [
    "he", "hel", "hell", "hello",
    " w", " wo", " wor", " worl", " world",
    "th", "the", " th", " the",
    "fo", "foo", "bar", "baz",
    " quick", " brown", " fox",
]


def _write_tiktoken_vocab(path: Path) -> None:
    lines: list[str] = []
    rank = 0
    for b in range(256):
        lines.append(base64.b64encode(bytes([b])).decode() + f" {rank}")
        rank += 1
    for extra in _EXTRAS:
        lines.append(base64.b64encode(extra.encode()).decode() + f" {rank}")
        rank += 1
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture(scope="session")
def tiktoken_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    p = tmp_path_factory.mktemp("ztok") / "synthetic_cl100k.tiktoken"
    _write_tiktoken_vocab(p)
    return p


@pytest.fixture
def bpe_pipeline(tiktoken_path: Path) -> ztok.Pipeline:
    pipe = ztok.Pipeline.from_tiktoken(str(tiktoken_path), cl100k=True)
    yield pipe
    pipe.close()
