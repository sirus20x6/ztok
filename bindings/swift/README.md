# ztok-swift — Swift bindings for libztok

Idiomatic Swift wrapper around the C ABI shipped with **ztok 1.24+**.
Swift Package Manager, no third-party dependencies, Swift **5.9+**.
Targets macOS 11+ and Linux (any distro with a Swift 5.9 toolchain).

## Install

Build the shared library first, then add this package to your
`Package.swift`:

```sh
cd /path/to/ztok
zig build -p zig-out                      # produces zig-out/{lib,include,bin}
```

```swift
// Your Package.swift
dependencies: [
    .package(path: "/path/to/ztok/bindings/swift"),
]
```

Make sure libztok is on the loader's path at run time:

| Platform | Variable                |
|----------|-------------------------|
| Linux    | `LD_LIBRARY_PATH`       |
| macOS    | `DYLD_LIBRARY_PATH`     |

```sh
export LD_LIBRARY_PATH=/path/to/ztok/zig-out/lib   # Linux
export DYLD_LIBRARY_PATH=/path/to/ztok/zig-out/lib # macOS
```

For build-time discovery, either drop `zig-out/lib/pkgconfig/ztok.pc`
on `PKG_CONFIG_PATH` (pulled in automatically via `pkgConfig: "ztok"`
in `Package.swift`) or pass the paths explicitly:

```sh
swift build \
  -Xcc -I/path/to/ztok/zig-out/include \
  -Xlinker -L/path/to/ztok/zig-out/lib \
  -Xlinker -lztok
```

## Quickstart

```swift
import Ztok

let pipe = try Pipeline.open(path: "tokenizer.json")  // auto-detect
defer { pipe.close() }

let ids = try pipe.encode("hello world")
let text = try pipe.decode(ids)
print("\(ids.count) ids -> \(text)")
```

`Pipeline.open(path:)` sniffs the file format via the C ABI's
`ztok_auto_detect` and dispatches to the right loader:

| File suffix / magic        | Loader                                         |
|----------------------------|------------------------------------------------|
| `.tiktoken`                | `Pipeline.fromTiktoken(path:)`                 |
| `tokenizer.json` (HF JSON) | `Pipeline.fromHfJson(path:)`                   |
| `.model` (SentencePiece)   | `Pipeline.fromSentencePiece(path:unkId:)`      |
| `.ztm` (TokenMonster)      | `Pipeline.fromMonster(path:)`                  |
| `tekken.json` (Mistral)    | `Pipeline.fromTekken(path:)`                   |

For WordPiece (which lives inside `tokenizer.json` but needs a specific
unknown-token id) call `Pipeline.fromWordPiece(path:unkId:)` directly —
`open` defaults `tokenizer.json` to the BPE loader.

## Multithreaded batches

```swift
let pipe = try Pipeline.fromTiktoken(path: "cl100k_base.tiktoken")
defer { pipe.close() }

let pool = try BatchPool(workers: 8)   // 0 = auto-detect cpu count
defer { pool.close() }

let results = try pool.encodeBatch(
    pipeline: pipe, inputs: ["foo", "bar", "baz"])
// results: [[UInt32]]
```

Reuse one `BatchPool` across many `encodeBatch` calls — each pool owns
its own worker threads and arenas. Creating one per batch wastes setup
work.

## Streaming encode

Three ergonomic shapes are available, pick whichever fits the call site:

```swift
// 1. Sink-style: drive feed/finish by hand.
let enc = try StreamEncoder(pipeline: pipe)
defer { enc.close() }
var ids: [UInt32] = []
for chunk in chunks {
    ids.append(contentsOf: try enc.feed(chunk))
}
ids.append(contentsOf: try enc.finish())

// 2. Synchronous Sequence over a fixed input buffer.
for chunk in try pipe.encodeStream(text: largeText, chunkSize: 4096) {
    process(chunk)   // chunk: [UInt32]
}

// 3. AsyncSequence<UInt32> — one id at a time.
for try await id in try pipe.asyncEncodeStream(text: largeText) {
    await sink.write(id)
}
```

Internally all three wrap the C ABI's `ztok_stream_*` family. The
encoder defers a trailing partial UTF-8 codepoint / pre-tokenizer span
up to a 1 MiB soft cap (see `src/stream.zig`); past that it force-cuts
at the nearest codepoint boundary.

## Fingerprint

```swift
let fp = try pipe.fingerprint()
print(fp.hexString)        // 64-char lowercase hex
let key = fp.bytes         // Data, 32 bytes — cache key / KV discriminator
```

Two pipelines that produce equal `Fingerprint` values will emit
bit-identical id streams for any input.

## API surface

| Symbol                                          | Purpose                                                       |
|-------------------------------------------------|---------------------------------------------------------------|
| `Pipeline.version()`                            | libztok version string.                                       |
| `Pipeline.detectFormat(path:)`                  | Sniff a tokenizer file's format. Returns `Format`.            |
| `Pipeline.byteId(config:)`                      | Baseline byte_id pipeline.                                    |
| `Pipeline.open(path:)` / `Pipeline.open(url:)`  | Auto-detect format + dispatch to the right loader.            |
| `Pipeline.fromTiktoken(path:config:)`           | Byte-level BPE from `.tiktoken`.                              |
| `Pipeline.fromHfJson(path:config:)`             | BPE from HuggingFace `tokenizer.json`.                        |
| `Pipeline.fromWordPiece(path:unkId:config:)`    | WordPiece from `tokenizer.json` (requires `unkId`).           |
| `Pipeline.fromSentencePiece(path:unkId:config:)`| Unigram from SentencePiece `.model`.                          |
| `Pipeline.fromMonster(path:config:)`            | TokenMonster from `.ztm`.                                     |
| `Pipeline.fromTekken(path:config:)`             | Mistral Tekken from `tekken.json`.                           |
| `pipeline.encode(_:)` / `.encodeBytes(_:)`      | Encode a string / bytes.                                      |
| `pipeline.decode(_:)` / `.decodeBytes(_:)`      | Decode ids to string / raw bytes.                             |
| `pipeline.fingerprint()`                        | 32-byte deterministic tokenizer fingerprint.                  |
| `pipeline.streamEncoder(chunkSize:)`            | Open a `StreamEncoder` against this pipeline.                 |
| `pipeline.encodeStream(text:chunkSize:)`        | Synchronous `Sequence<[UInt32]>` over a fixed buffer.         |
| `pipeline.asyncEncodeStream(text:chunkSize:)`   | `AsyncSequence<UInt32>` over a fixed buffer.                  |
| `pipeline.close()` / `pipeline.isClosed`        | Release the underlying C handle. Idempotent.                  |
| `BatchPool(workers:)`                           | Persistent worker pool (`workers=0` = auto).                  |
| `pool.workerCount` / `.close()`                 | Inspect worker count / release.                               |
| `pipe.encodeBatch(pool:inputs:)` via `BatchPool`| Multithreaded batch encode through `BatchPool.encodeBatch`.   |

Configuration enums (raw `UInt32` matching the C ABI):
`Normalizer.identity`, `.nfc`, `.nfd`, `.nfkc`, `.nfkd`, `.byteLevel`,
`PreTokenizer.identity`, `.cl100k`,
`Decoder.concat`, `.wordPiece`, `.byteLevel`.

### Errors

All C-status failures throw `ZtokError`, a typed enum mirroring the
status codes:

```swift
do {
    let pipe = try Pipeline.fromTiktoken(path: "bad.tiktoken")
} catch ZtokError.invalidInput {
    // bad/malformed vocab
} catch ZtokError.internalError(let status, let op) {
    print("ztok call \(op) failed with status \(status)")
} catch {
    print("other error: \(error)")
}
```

| C status                      | Swift case                            |
|-------------------------------|---------------------------------------|
| `ZTOK_ERR_OUT_OF_MEMORY`      | `.outOfMemory`                        |
| `ZTOK_ERR_INVALID_INPUT`      | `.invalidInput`                       |
| `ZTOK_ERR_BUFFER_TOO_SMALL`   | `.bufferTooSmall`                     |
| `ZTOK_ERR_INTERNAL` / unknown | `.internalError(status:op:)`          |
| (NULL handle)                 | `.nullHandle(op:)`                    |
| auto-detect failure           | `.unknownFormat(path:)`               |
| post-`close()` call           | `.closed(op:)`                        |
| non-UTF-8 decoded bytes       | `.invalidUtf8`                        |

## New in 1.28 — token-window chunking, n-gram hashing, RWKV

**Token-window chunking** for late-chunking embedding pipelines:

```swift
let pipe = try Pipeline.byteId()
defer { pipe.close() }
for ch in try pipe.chunk("the quick brown fox", maxTokens: 8, overlap: 2) {
    print("\(ch.ids) bytes=\(ch.byteStart)..\(ch.byteEnd) toks=\(ch.tokenStart)..\(ch.tokenEnd)")
}
```

**Engram n-gram hashing** — deterministic multi-head token-n-gram hashes
(row-major `[position][head]`, raw u64; mask to your table width):

```swift
let ids: [UInt32] = [10, 20, 30, 40, 50]
let hashes = try Engram.hashNGrams(ids, n: 3, heads: 4)
// hashes.count == (ids.count - n + 1) * heads == 12
```

**RWKV "World" tokenizer** — greedy longest-match byte trie (no normalizer
or pre-tokenizer; byte-lossless):

```swift
let pipe = try Pipeline.fromRwkv(path: "rwkv_vocab_v20230424.txt")
defer { pipe.close() }
let ids = try pipe.encode("hello world")
```

## Wrapped C ABI surface

Every function declared in `include/ztok.h` for libztok 1.24 is wrapped:

- **lifecycle**: `ztok_pipeline_new` (byte_id), `*_from_tiktoken`,
  `*_from_hf_json`, `*_wordpiece_from_hf_json`,
  `*_unigram_from_sp_model`, `*_monster_from_file`,
  plus `Pipeline.open` (auto-detect via `ztok_auto_detect`)
- **encode / decode**: `ztok_encode`, `ztok_decode`
- **persistent batch**: `BatchPool` (`ztok_batch_pool_new`,
  `ztok_batch_pool_free`, `ztok_batch_pool_worker_count`,
  `ztok_encode_batch_pooled`)
- **streaming**: `StreamEncoder` sink-style API + `Sequence` /
  `AsyncSequence` adapters (`ztok_stream_new`, `ztok_stream_feed`,
  `ztok_stream_finish`, `ztok_stream_free`)
- **fingerprint**: `Pipeline.fingerprint()` (`ztok_fingerprint`)
- **utility**: `Pipeline.version` (`ztok_version`),
  `Pipeline.detectFormat` (`ztok_auto_detect`)
- **buffer freeing**: every libztok-owned id buffer is materialized
  into a Swift `[UInt32]` then released via `ztok_ids_free` — no
  foreign pointers leak into safe Swift.

Not yet exposed: `ztok_encode_batch` (the per-call worker pool variant).
Use `BatchPool` instead — it reuses arenas + worker threads across
calls and is what every other binding exposes as the primary batch
surface.

## Run the tests

```sh
cd /path/to/ztok/bindings/swift
PKG_CONFIG_PATH=../../zig-out/lib/pkgconfig \
LD_LIBRARY_PATH=../../zig-out/lib \
swift test
```

The fixture writes a synthetic 256-byte + handful-of-merges
`.tiktoken` vocab to a temp dir, matching the Python/Go/Rust binding
fixtures. The smoke suite covers version, encode/decode round-trip,
batch (4 inputs), streaming sync + async, fingerprint, error mapping,
auto-detect, and `close()` idempotence. The fuzz suite drives 1000
SplitMix64-mutated round-trips against the byte_id pipeline with the
canonical `0xFEEDB0B` seed (override via `FUZZ_SEED` /
`ZTOK_FUZZ_ITERS`).

Tests skip cleanly via `XCTSkipUnless` when libztok can't be loaded,
so the target stays green on dev machines without a built library.

## Notes

- **Thread safety**: a `Pipeline` is safe to share across actors /
  tasks for `encode`/`decode`/`encodeBatch`/`fingerprint` (the
  underlying Zig pipeline is `const` after construction). Streaming
  encode is *not* — each task should own its own `StreamEncoder`.
- **Handle lifetime**: `deinit` releases the C handle so dropping a
  `Pipeline` / `BatchPool` / `StreamEncoder` without `close()` still
  frees its resources. Prefer explicit `close()` (or
  `defer x.close()`) for determinism — Swift's ARC fires deinit when
  the last strong reference goes out of scope, but with refcounting
  delays that can be later than you'd like.
- **Encode** grows its output buffer on `BUFFER_TOO_SMALL` (the
  per-span `maxTokensFor` bound is conservative). Bound at 8 grow
  attempts — past that it throws `.internalError`.

## Status

- **libztok target**: 1.24 (compatible with all 1.x C ABI versions,
  additive enum values are forward-compatible).
- **Swift MSV**: 5.9. We use no Swift 5.10/6 features; bump only if a
  later compiler ergonomics win justifies dropping older toolchains.
- **Platforms**: tested on Linux (Swift 5.9 toolchain) and macOS 11+.
  iOS / tvOS / watchOS untested — libztok ships as a system shared
  library and Apple's mobile platforms don't permit user-installed
  `.dylib`s, so those targets would need a different deployment story
  (XCFramework with libztok statically linked).
