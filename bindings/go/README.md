# ztok-go — Go bindings for libztok

Thin cgo wrapper around the C ABI shipped with ztok 1.20+. Standard
library only (no third-party deps), Go 1.21+, uses `pkg-config` when
available and falls back to `-lztok`.

## Install

Build the shared library first, then `go get` (or vendor) the binding:

```sh
zig build -p zig-out                      # produces zig-out/{lib,include,bin}
go get github.com/sirus20x6/ztok-go      # placeholder import path
```

The cgo build needs:

- `ztok.h` on the C preprocessor include path (`CGO_CFLAGS` or
  `C_INCLUDE_PATH`).
- `libztok.{so,dylib}` on the linker path (`CGO_LDFLAGS` or
  `LD_LIBRARY_PATH`).

If you ran `zig build -p prefix` and dropped `prefix/lib/pkgconfig/ztok.pc`
on `PKG_CONFIG_PATH`, that's the whole config — the default build wires
`#cgo pkg-config: ztok` and picks up the rest. For environments without
pkg-config (e.g. minimal Docker), build with `-tags nopkgconfig` and
set the flags yourself:

```sh
go build -tags nopkgconfig ...
export CGO_CFLAGS="-I/path/to/ztok/include"
export CGO_LDFLAGS="-L/path/to/ztok/lib"
export LD_LIBRARY_PATH="/path/to/ztok/lib"
```

For in-tree development from this repo:

```sh
cd bindings/go
PKG_CONFIG_PATH=../../zig-out/lib/pkgconfig \
LD_LIBRARY_PATH=../../zig-out/lib \
CGO_CFLAGS=-I../../zig-out/include \
go test ./...
```

## Quickstart

```go
package main

import (
    "fmt"
    "log"

    ztok "github.com/sirus20x6/ztok-go"
)

func main() {
    pipe, err := ztok.Open("tokenizer.json")  // auto-detects format
    if err != nil { log.Fatal(err) }
    defer pipe.Close()

    ids, _ := pipe.Encode("hello world")
    text, _ := pipe.Decode(ids)
    fmt.Printf("%d ids -> %q\n", len(ids), text)
}
```

`ztok.Open` sniffs the file format via the C ABI's `ztok_auto_detect`
(post-1.18 agent C) and dispatches to the right loader:

| File suffix / magic        | Loader                                           |
|----------------------------|--------------------------------------------------|
| `.tiktoken`                | `OpenTiktoken(path, &Config{CL100K: true})`      |
| `tokenizer.json` (HF JSON) | `OpenHFJSON(path, nil)`                          |
| `.model` (SentencePiece)   | `OpenSentencePiece(path, &Config{UnkID: 0})`     |
| `.ztm` (TokenMonster)      | `OpenMonster(path, nil)`                         |

For WordPiece (which lives inside `tokenizer.json` but needs a specific
unknown-token id) call `OpenWordPiece(path, &Config{UnkID: ...})`
directly — `Open` defaults `tokenizer.json` to the BPE loader.

## Multithreaded batches

```go
pipe, _ := ztok.OpenTiktoken("cl100k_base.tiktoken", nil)
defer pipe.Close()

pool, _ := ztok.NewBatchPool(&ztok.BatchPoolOptions{Workers: 8})
defer pool.Close()

results, err := pipe.EncodeBatch(pool, []string{"foo", "bar", "baz"})
// results: [][]uint32
```

Reuse one `BatchPool` across many `EncodeBatch` calls — each pool owns
its own worker threads and arenas. Creating one per batch wastes setup
work.

## Streaming encode

```go
pipe, _ := ztok.Open("cl100k.tiktoken")
defer pipe.Close()

err := pipe.EncodeStream(largeText, func(ids []uint32) error {
    // process this chunk's ids
    return nil
}, &ztok.StreamOptions{ChunkSize: 4096})
```

Internally wraps the C ABI `ztok_stream_*` family. The encoder defers a
trailing partial UTF-8 codepoint / pre-tokenizer span up to a 1 MiB soft
cap (see `src/stream.zig`); past that it force-cuts at the nearest
codepoint boundary.

## cgo overhead

A cgo invocation has a measurable per-call cost (~50 ns on a modern
x86_64 — `Version()` clocks 48.8 ns/call locally, benchmarked at 1M
invocations). For latency-sensitive workloads this matters in two ways:

- **Encode single strings sparingly.** A million 1-byte `Encode()` calls
  burns ~50 ms in cgo alone, before any tokenization work. If you're
  feeding a serving loop with short requests, expect a few-µs floor per
  request.
- **Batch when you can.** `EncodeBatch` amortizes the cgo crossing
  across the whole batch — one round-trip instead of N. With 1000
  inputs you pay one ~50 ns crossing instead of 50 µs.

Streaming has the same shape: each `ztok_stream_feed` is one crossing
regardless of how many ids it emits.

## API surface

| Symbol                                          | Purpose                                                 |
|-------------------------------------------------|---------------------------------------------------------|
| `ztok.Version()`                                | libztok version string.                                 |
| `ztok.DetectFormat(path)`                       | Sniff a tokenizer file's format. Returns `Format`.      |
| `ztok.NewByteID(cfg)`                           | Baseline byte_id pipeline.                              |
| `ztok.Open(path)`                               | Auto-detect format + dispatch to the right loader.      |
| `ztok.OpenTiktoken(path, cfg)`                  | Byte-level BPE from `.tiktoken`.                        |
| `ztok.OpenHFJSON(path, cfg)`                    | BPE from HuggingFace `tokenizer.json`.                  |
| `ztok.OpenWordPiece(path, cfg)`                 | WordPiece from `tokenizer.json` (requires `cfg.UnkID`). |
| `ztok.OpenSentencePiece(path, cfg)`             | Unigram from SentencePiece `.model`.                    |
| `ztok.OpenMonster(path, cfg)`                   | TokenMonster from `.ztm`.                               |
| `Pipeline.Encode(text)` / `.EncodeBytes(b)`     | Encode a string / bytes.                                |
| `Pipeline.Decode(ids)` / `.DecodeBytes(ids)`    | Decode ids to string / raw bytes.                       |
| `Pipeline.EncodeBatch(pool, inputs)`            | Multithreaded batch encode via a persistent pool.       |
| `Pipeline.EncodeStream(text, cb, opts)`         | Stream-encode with per-chunk callback.                  |
| `Pipeline.EncodeStreamBytes(b, cb, opts)`       | Same, raw bytes.                                        |
| `Pipeline.Close()`                              | Release the underlying C handle. Idempotent.            |
| `ztok.NewBatchPool(opts)`                       | Persistent worker pool (`Workers=0` = auto).            |
| `BatchPool.Workers()` / `.Close()`              | Inspect worker count / release.                         |

Configuration constants (uint32):
`NormalizerIdentity`, `NormalizerNFC`, `NormalizerNFD`, `NormalizerNFKC`,
`NormalizerNFKD`, `NormalizerByteLevel`,
`PretokIdentity`, `PretokCL100K`,
`DecoderConcat`, `DecoderWordPiece`, `DecoderByteLevel`.

### Errors

All C-status failures land as typed errors satisfying the standard
`error` interface. Match them with `errors.As`:

```go
var iie *ztok.InvalidInputError
if errors.As(err, &iie) {
    log.Printf("bad input at %s: status=%d", iie.Op, iie.Status)
}
```

| C status                      | Go type                  |
|-------------------------------|--------------------------|
| `ZTOK_ERR_OUT_OF_MEMORY`      | `*OutOfMemoryError`      |
| `ZTOK_ERR_INVALID_INPUT`      | `*InvalidInputError`     |
| `ZTOK_ERR_BUFFER_TOO_SMALL`   | `*BufferTooSmallError`   |
| `ZTOK_ERR_INTERNAL` / unknown | `*InternalError`         |

`ErrClosed` is the sentinel returned by methods on a closed
Pipeline/BatchPool — match with `errors.Is`.

## Notes on the C ABI

- This is a *thin* wrapper. The C library owns every id buffer and
  vocab table; we never allocate id arrays on the Go side. Per-input id
  buffers returned by `ztok_encode_batch_pooled` carry a length-prefix
  header (see `src/c_api.zig::allocIdBuf`) and **must** be freed via
  `ztok_ids_free` — the binding does that for you after copying each
  buffer into a Go `[]uint32`.
- Handle lifetime is managed with `runtime.SetFinalizer`, so dropping a
  `Pipeline` or `BatchPool` without `Close()` still releases the C
  resource on GC. Prefer explicit `Close()` (or `defer pipe.Close()`)
  for determinism — finalizers fire on GC schedule, not when you'd like.
- `Encode` grows its output buffer on `BUFFER_TOO_SMALL` (the per-span
  `maxTokensFor` bound is conservative). Bound at 8 grow attempts.
- `Pipeline` is safe to share across goroutines for `Encode`/`Decode`
  (the underlying Zig pipeline is `const` after construction); the
  pool/stream APIs are not — each goroutine should own its own
  `BatchPool` / stream session.

## Run the tests

```sh
cd bindings/go
PKG_CONFIG_PATH=../../zig-out/lib/pkgconfig \
LD_LIBRARY_PATH=../../zig-out/lib \
CGO_CFLAGS=-I../../zig-out/include \
go test ./...
```

The fixture writes a synthetic 256-byte + handful-of-merges
`.tiktoken` vocab to a temp dir, matching the Python/Node test
fixtures. Covers version, encode/decode round-trip, auto-detect,
batch encode (1000 inputs, no leaks), streaming vs single-shot
equivalence, `Close()` idempotence, typed error mapping, finalizer
release, and concurrent encode safety.
