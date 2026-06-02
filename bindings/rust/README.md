# ztok — Rust bindings

Idiomatic safe Rust wrapper around **libztok**, a fast multithreaded
tokenizer library written in Zig. Mirrors the C ABI declared in
[`include/ztok.h`](../../include/ztok.h) and the surface of the existing
Python / Node / Ruby / Go bindings.

Status: **1.22.0** — every C ABI surface declared in `include/ztok.h`
for this release is wrapped (see [Wrapped surface](#wrapped-surface)).

## Installing libztok

This crate links against `libztok.{so,dylib}` (or `libztok.a` with
`--features link-static`). Build the library first:

```bash
cd ../..                            # repo root
zig build -Doptimize=ReleaseFast -p $HOME/.local
```

Then make sure cargo and the runtime loader can find it:

```bash
# Build-time: pkg-config discovery (the default)
export PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig:$PKG_CONFIG_PATH

# Runtime: dynamic loader search path
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH   # Linux
export DYLD_LIBRARY_PATH=$HOME/.local/lib:$DYLD_LIBRARY_PATH  # macOS
```

Alternatives:

- `ZTOK_LIB_DIR=/path/to/lib cargo build` — explicit override, skips
  pkg-config.
- `cargo build --features link-static` — link `libztok.a` instead of the
  shared object. The crate emits the typical Zig-stdlib C deps (`m`,
  `pthread`, `dl`) on Linux automatically.

## Quickstart

```rust
use ztok::Pipeline;

// Auto-detect format (.tiktoken / tokenizer.json / .model / .ztm).
let pipe = Pipeline::open("cl100k_base.tiktoken")?;
let ids = pipe.encode("hello world")?;
let text = pipe.decode(&ids)?;
assert_eq!(text, "hello world");
# Ok::<(), ztok::Error>(())
```

## Multi-threaded batching

```rust
use ztok::{BatchPool, Pipeline};

let pipe = Pipeline::open("cl100k_base.tiktoken")?;
let pool = BatchPool::new(8)?;  // 0 = auto-detect cpu count
let results = pool.encode_batch(&pipe, &["foo", "bar", "baz"])?;
assert_eq!(results.len(), 3);
# Ok::<(), ztok::Error>(())
```

`BatchPool` is persistent — create one, reuse across many batches.
Each pool owns its own arenas and worker threads, so spawning one per
batch wastes setup work.

## Streaming

Iterator-style:

```rust
use ztok::Pipeline;

let pipe = Pipeline::open("cl100k_base.tiktoken")?;
for batch in pipe.encode_stream_default(b"hello world")? {
    let ids = batch?;
    println!("{} ids", ids.len());
}
# Ok::<(), ztok::Error>(())
```

Or sink-style, matching the Node `koffi` and Python ctypes shape:

```rust
use ztok::{Pipeline, StreamEncoder};

let pipe = Pipeline::open("cl100k_base.tiktoken")?;
let mut enc = StreamEncoder::new(&pipe)?;
let mut ids: Vec<u32> = Vec::new();
for chunk in source_bytes().chunks(8192) {
    ids.extend(enc.feed(chunk)?);
}
ids.extend(enc.finish()?);   // drain remaining carry
# fn source_bytes() -> Vec<u8> { vec![] }
# Ok::<(), ztok::Error>(())
```

## Fingerprint (new in 1.22)

Deterministic 32-byte SHA-256 digest over the pipeline's encoding
behavior on a fixed canonical input set. Two pipelines that return the
same 32 bytes produce bit-identical id streams for any input — use it
as a cache key, KV-store discriminator, or training-pipeline guard.

```rust
use ztok::Pipeline;

let pipe = Pipeline::open("cl100k_base.tiktoken")?;
let fp: [u8; 32] = pipe.fingerprint()?;
println!("{}", hex::encode(fp));
# Ok::<(), ztok::Error>(())
```

## Wrapped surface

Everything in `include/ztok.h` for libztok 1.22:

| C ABI                                    | Rust                                          |
|------------------------------------------|-----------------------------------------------|
| `ztok_pipeline_new`                      | `Pipeline::byte_id`                           |
| `ztok_pipeline_new_bpe_from_tiktoken`    | `Pipeline::from_tiktoken`                     |
| `ztok_pipeline_new_bpe_from_hf_json`     | `Pipeline::from_hf_json`                      |
| `ztok_pipeline_new_wordpiece_from_hf_json` | `Pipeline::from_wordpiece`                  |
| `ztok_pipeline_new_unigram_from_sp_model` | `Pipeline::from_sentencepiece`               |
| `ztok_pipeline_new_monster_from_file`    | `Pipeline::from_monster`                      |
| `ztok_auto_detect`                       | `detect_format`, `Pipeline::open`             |
| `ztok_encode` / `ztok_decode`            | `Pipeline::encode{,_bytes}` / `decode{,_bytes}` |
| `ztok_batch_pool_*` / `ztok_encode_batch_pooled` | `BatchPool::new` / `encode_batch{,_bytes}` |
| `ztok_stream_*`                          | `StreamEncoder`, `Pipeline::encode_stream`    |
| `ztok_fingerprint`                       | `Pipeline::fingerprint`                       |
| `ztok_version`                           | `version`                                     |

Not yet exposed: the per-call `ztok_encode_batch` (the one that spawns
a fresh worker pool every invocation). Use `BatchPool` instead — it
reuses arenas + worker threads across calls and is the surface every
other binding promotes as primary.

## New in 1.28 — token-window chunking, n-gram hashing, RWKV

**Token-window chunking** for late-chunking embedding pipelines:

```rust
use ztok::{ChunkBoundary, Pipeline};

let pipe = Pipeline::byte_id()?;
for ch in pipe.chunk("the quick brown fox", 8, 2, ChunkBoundary::Token)? {
    println!("ids={:?} bytes={}..{} toks={}..{}",
        ch.ids, ch.byte_start, ch.byte_end, ch.token_start, ch.token_end);
}
# Ok::<(), ztok::Error>(())
```

**Engram n-gram hashing** — deterministic multi-head token-n-gram hashes
(row-major `[position][head]`, raw u64; mask to your table width):

```rust
let ids: Vec<u32> = vec![10, 20, 30, 40, 50];
let hashes = ztok::hash_ngrams(&ids, 3, 4)?;
// hashes.len() == (ids.len() - n + 1) * heads == 12
# Ok::<(), ztok::Error>(())
```

**RWKV "World" tokenizer** — greedy longest-match byte trie (no normalizer
or pre-tokenizer; byte-lossless):

```rust
let pipe = Pipeline::from_rwkv("rwkv_vocab_v20230424.txt")?;
let ids = pipe.encode("hello world")?;
# Ok::<(), ztok::Error>(())
```

## MSRV

**Rust 1.74** (current stable minus one minor). Conservative — the crate
uses only `core::ffi`, `std::ffi`, `OnceLock` (1.70+), and rust-2021
syntax. No proc-macros, no nightly features.

## Safety

All `unsafe` lives in the `sys` module (raw `extern "C"` declarations)
plus small unsafe blocks in `pipeline.rs`, `batch.rs`, and `stream.rs`
that translate the FFI surface to safe Rust. The public API contains no
`unsafe`. Handles are released in `Drop` impls — no manual `close()`
required.

## Testing

```bash
# With libztok available
ZTOK_LIB_DIR=$HOME/.local/lib \
LD_LIBRARY_PATH=$HOME/.local/lib \
PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig \
cargo test

# Static linking
cargo test --features link-static --no-default-features
```

`tests/smoke.rs` covers the basic surface (round-trip, batch, stream,
fingerprint, error mapping); `tests/fuzz.rs` does 1000 PRNG-mutated
round-trips against the byte_id vocab (seed = `0xFEEDB0B`, matching the
other bindings' fuzz harnesses for cross-language failure reproduction).

If `cargo test` reports a link error like `cannot find -lztok`, libztok
isn't on the path — see [Installing libztok](#installing-libztok)
above.

## License

AGPL-3.0-only (matches the parent project). See the top-level `LICENSE`.
