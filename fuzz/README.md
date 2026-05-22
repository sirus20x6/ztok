# ztok fuzz harnesses

Round-trip fuzz harnesses for the encode/decode hot path.

## Running

```sh
# Default: 60-second budget, corpus-driven mutation loop.
zig build fuzz

# Custom budget + seed (reproducible).
ZTOK_FUZZ_BUDGET_SECS=600 ZTOK_FUZZ_SEED=0xC0FFEE zig build fuzz

# libFuzzer-driven (when the toolchain ships the runtime — Zig 0.16
# wires this up via `--fuzz` on the test step).
zig build test --fuzz
```

## What's checked

Two round-trip invariants run on each fuzz input:

1. **byte_id**: identity normalizer + identity pre-tok + byte_id model
   + concat decoder. `decode(encode(input)) == input` must hold for
   any byte sequence.
2. **cl100k BPE**: synthetic tiktoken vocab covering all 256 bytes +
   merges, cl100k pre-tokenizer, concat decoder. Same round-trip
   invariant — the vocab is byte-complete, so any divergence is a bug
   in the regex pre-tok / BPE merge / SoA scratch reuse.

## Corpus

Seeds live under `fuzz/corpus/`:
- `01-hello.txt`         — printable ASCII baseline
- `02-empty.txt`         — zero-length input
- `03-multibyte.txt`     — Japanese + Greek + math + emoji codepoints
- `04-code.txt`          — Zig-shaped tokens (braces, dots, semicolons)
- `05-whitespace.txt`    — whitespace-only (multiple consecutive newlines)
- `06-binary.bin`        — control bytes + high bytes
- `07-malformed-utf8.bin` — truncated multi-byte sequences

The harness loads every seed at startup, then mutates them per
iteration via bit-flips, truncation, and random-tail extension.
Inputs are capped at 8 KiB.

## Add a new seed

Drop a file under `fuzz/corpus/` (≤64 KiB). The corpus is read-only —
the harness MUST NOT touch any vocab under `bench/vocabs/`.

## Reporting a crash

When `zig build fuzz` panics, the harness prints the seed value to
stderr:

```
ztok fuzz: byte_id mismatch iter=12345 input_len=137 err=RoundTripMismatch
```

Re-run with the same `ZTOK_FUZZ_SEED` to reproduce. File a bug with
the seed + iteration number; the harness output is fully deterministic
given a seed.
