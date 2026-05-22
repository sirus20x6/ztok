"""PRNG-driven round-trip fuzz harness for the ztok Python binding.

Mirrors `fuzz/encode_decode.zig` in shape: a deterministic PRNG mutates
the byte input each iteration, encode-then-decode must round-trip
exactly. The pipeline is `byte_id` (identity normalizer + identity
pre-tok + byte_id model + concat decoder) — by construction each input
byte maps to one id and decoding concatenates them back, so for ANY
byte sequence we must have ``decode(encode(x)) == x``.

If libztok cannot be loaded we skip cleanly (mirrors the
fixture-missing skip pattern in test_loaders.py).
"""

from __future__ import annotations

import os
import random

import pytest

try:
    import ztok
    from ztok._lib import ZtokLibraryNotFoundError

    # Probe the loader once up-front so missing-lib environments skip
    # the whole module instead of erroring during fixture setup.
    ztok._get_lib()
except ZtokLibraryNotFoundError as exc:  # pragma: no cover - env-dependent
    pytest.skip(f"libztok not available: {exc}", allow_module_level=True)


# Deterministic seed: same value as the Ruby/Node harnesses so failures
# at a given iteration are cross-language reproducible. The fixed seed
# may be overridden per-run via FUZZ_SEED (hex like "0xdeadbeef" or
# decimal). ZTOK_FUZZ_ITERS likewise scales the iteration count for
# nightly fuzz workflows.
_DEFAULT_SEED = 0xFEEDB0B
_DEFAULT_ITERATIONS = 1000
_MAX_LEN = 256


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw, 0)
    except ValueError:
        return default


_SEED = _env_int("FUZZ_SEED", _DEFAULT_SEED)
_ITERATIONS = _env_int("ZTOK_FUZZ_ITERS", _DEFAULT_ITERATIONS)


def _random_bytes(rng: random.Random, max_len: int) -> bytes:
    """Pick a length in [0, max_len] and fill with uniform-random bytes."""

    n = rng.randint(0, max_len)
    if n == 0:
        return b""
    # random.Random.randbytes is deterministic given the seeded RNG.
    return rng.randbytes(n)


def test_byte_id_roundtrip_fuzz_1000_iterations() -> None:
    """PRNG-mutated round-trips against the byte_id pipeline.

    Default 1000 iterations (overridable via ZTOK_FUZZ_ITERS env, with
    FUZZ_SEED overriding the deterministic seed). Every iteration:
    random length 0..256, random byte content, encode then decode,
    assert decoded == original. Asserts no native handle leaks by
    reusing one pipeline for the full run and closing it on exit.
    """

    rng = random.Random(_SEED)
    failures: list[tuple[int, bytes, bytes]] = []

    with ztok.Pipeline.byte_id() as pipe:
        for i in range(_ITERATIONS):
            data = _random_bytes(rng, _MAX_LEN)
            ids = pipe.encode(data)
            # byte_id maps 1:1, so the id count must match the input length.
            assert len(ids) == len(data), (
                f"iter {i}: byte_id produced {len(ids)} ids for {len(data)} bytes"
            )
            roundtrip = pipe.decode_bytes(ids)
            if roundtrip != data:
                failures.append((i, data, roundtrip))
                if len(failures) >= 5:
                    # Don't flood the report — first 5 mismatches are enough
                    # to root-cause.
                    break

    if failures:
        msgs = [
            f"  iter {i}: in={inp!r} out={out!r}" for i, inp, out in failures
        ]
        pytest.fail(
            "byte_id round-trip mismatches:\n" + "\n".join(msgs)
        )
