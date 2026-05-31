"""Tests for Pipeline.fingerprint() — the 32-byte tokenizer fingerprint.

Mirrors the rust/dotnet/java fingerprint tests: determinism + 32-byte
length, plus a cross-binding golden value for the byte_id pipeline.
"""

from __future__ import annotations

import ztok


# Golden fingerprint for the default byte_id pipeline. Computed directly
# from libztok's ztok_fingerprint and shared across the rust/dotnet/java/
# nodejs/ruby/go bindings to confirm cross-binding agreement.
GOLDEN_BYTE_ID = "201ecf86554b5a970471e0189d7e78dc2c3df24519f7d5ebb0caacc86701e77c"


def test_fingerprint_is_32_bytes() -> None:
    with ztok.Pipeline.byte_id() as pipe:
        fp = pipe.fingerprint()
        assert isinstance(fp, bytes)
        assert len(fp) == 32
        # Not all zeros (would suggest a hashing bug).
        assert any(b != 0 for b in fp)


def test_fingerprint_is_deterministic() -> None:
    with ztok.Pipeline.byte_id() as a, ztok.Pipeline.byte_id() as b:
        assert a.fingerprint() == b.fingerprint()


def test_fingerprint_golden_byte_id() -> None:
    with ztok.Pipeline.byte_id() as pipe:
        assert pipe.fingerprint().hex() == GOLDEN_BYTE_ID
