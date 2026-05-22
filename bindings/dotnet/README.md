# ztok — .NET bindings

Idiomatic .NET binding for **libztok**, a fast multithreaded tokenizer
library written in Zig. Wraps the stable C ABI declared in
[`include/ztok.h`](../../include/ztok.h) via P/Invoke and mirrors the
surface of the existing Python / Node / Ruby / Go / Rust bindings.

Status: **1.23** — every C ABI surface declared in `include/ztok.h` for
this release is wrapped (see [Wrapped surface](#wrapped-surface)).

## Requirements

- .NET 8 SDK (LTS) or newer.
- A built `libztok.so` / `libztok.dylib` / `ztok.dll` somewhere the
  loader can find it (see below).

## Installing libztok

Build the shared library first:

```bash
cd ../..                                # repo root
zig build -Doptimize=ReleaseFast
```

That produces `zig-out/lib/libztok.so` (on Linux) along with the C
header. The .NET binding looks for it in this order:

1. `ZTOK_LIB_PATH` environment variable (explicit absolute path).
2. Next to the consuming assembly (NuGet package layout).
3. `runtimes/{rid}/native/libztok.{so,dylib,dll}` (RID-folder layout).
4. The repo's `zig-out/lib/` directory (in-tree development).
5. `/usr/local/lib`, `/usr/lib`, `/usr/lib64` (Linux);
   `/usr/local/lib`, `/opt/homebrew/lib` (macOS).
6. The platform's normal dynamic loader search
   (`LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH` / `PATH`).

For the common in-tree case:

```bash
export LD_LIBRARY_PATH="$PWD/../../zig-out/lib:$LD_LIBRARY_PATH"
# or, explicit:
export ZTOK_LIB_PATH="$PWD/../../zig-out/lib/libztok.so"
```

## Quickstart

```csharp
using Ztok;

// Auto-detect file format (.tiktoken / tokenizer.json / .model / .ztm)
using var pipe = Pipeline.Open("cl100k_base.tiktoken");

uint[] ids = pipe.Encode("hello world");
string text = pipe.Decode(ids);

// Persistent worker pool for parallel batch encode.
using var pool = BatchPool.Create(workers: 8);
uint[][] results = pool.EncodeBatch(pipe, new[] { "foo", "bar", "baz" });

// Streaming.
await foreach (uint[] batch in pipe.EncodeStreamAsync("the quick brown fox"))
    Console.WriteLine($"emitted {batch.Length} ids");

// Tokenizer fingerprint as cache key.
Fingerprint fp = pipe.Fingerprint();
Console.WriteLine(fp.ToHexString());
```

## Building & testing

```bash
cd bindings/dotnet
dotnet build
ZTOK_LIB_PATH=$PWD/../../zig-out/lib/libztok.so dotnet test
```

The test suite (`Ztok.Tests/`) covers:

- `SmokeTests.cs` — version probe, byte_id round-trip, batch pool,
  streaming (sync + async), fingerprint determinism, format detection,
  typed-exception mapping, dispose idempotency.
- `FuzzTests.cs` — 1000-iteration PRNG round-trip against the byte_id
  vocab (seed `0xFEEDB0B`, matches the other bindings; override via
  `FUZZ_SEED` / `ZTOK_FUZZ_ITERS`).

Tests skip gracefully (return early with a diagnostic) when libztok
cannot be loaded.

## API surface

| Type | Purpose |
|------|---------|
| `Pipeline` | Loaded tokenizer (`Open` auto-detect, `FromTiktoken` / `FromHfJson` / `FromWordPiece` / `FromSentencePiece` / `FromMonster` / `ByteId`). `Encode`, `Decode`, `EncodeBytes`, `DecodeBytes`, `Fingerprint`. `IDisposable` + `SafeHandle` finalizer. Thread-safe for encode/decode/fingerprint. |
| `BatchPool` | Persistent multithreaded worker pool. `EncodeBatch` / `EncodeBatchBytes`. `IDisposable`. |
| `StreamEncoder` | Sink-style streaming encoder. `Feed`, `Finish`. `IDisposable`. |
| `Pipeline.EncodeStream` / `Pipeline.EncodeStreamAsync` | Iterator-style streaming, `IEnumerable<uint[]>` + `IAsyncEnumerable<uint[]>`. |
| `Fingerprint` | 32-byte value type with `ToHexString`, structural equality. |
| `ZtokLibrary.Version`, `ZtokLibrary.DetectFormat` | Top-level helpers. |
| `Format`, `Normalizer`, `PreTokenizer`, `Decoder`, `PipelineConfig` | Enum mirrors of the C ABI configuration types. |
| `ZtokException` + `ZtokOutOfMemoryException` / `ZtokInvalidInputException` / `ZtokBufferTooSmallException` | Typed errors mapping `ztok_status`. |

## Wrapped surface

Every function declared in `include/ztok.h` for libztok 1.23 is wrapped:

- **lifecycle**: `ztok_pipeline_new`, `ztok_pipeline_new_bpe_from_tiktoken`,
  `ztok_pipeline_new_bpe_from_hf_json`,
  `ztok_pipeline_new_wordpiece_from_hf_json`,
  `ztok_pipeline_new_unigram_from_sp_model`,
  `ztok_pipeline_new_monster_from_file`, `ztok_pipeline_free`
- **encode / decode**: `ztok_encode`, `ztok_decode`
- **batch**: `ztok_batch_pool_new`, `ztok_batch_pool_free`,
  `ztok_batch_pool_worker_count`, `ztok_encode_batch_pooled`,
  `ztok_ids_free`
- **streaming**: `ztok_stream_new`, `ztok_stream_feed`,
  `ztok_stream_finish`, `ztok_stream_free`
- **introspection**: `ztok_version`, `ztok_auto_detect`,
  `ztok_fingerprint`

Not currently exposed: `ztok_encode_batch` (the per-call worker-pool
variant). Use `BatchPool` instead — it reuses arenas + worker threads
across calls and is what every other binding offers as the primary
batch surface.

## Threading

- `Pipeline.Encode`, `Pipeline.Decode`, and `Pipeline.Fingerprint` are
  safe to call from multiple threads concurrently against the same
  instance (libztok's pipeline state is read-only after load).
- `Pipeline.Dispose` must **not** race with other calls on the same
  instance.
- `BatchPool` owns its own worker threads internally; one pool can be
  shared across threads, but the cheapest pattern is one pool per
  long-lived encoder.
- `StreamEncoder` is **not** thread-safe — one encoder per concurrent
  stream.

## Native interop notes

All `unsafe` code is confined to `Ztok/Native.cs`. The higher-level
wrappers (`Pipeline`, `BatchPool`, `StreamEncoder`) call into safe-ish
helpers in `Native.cs` that take `Span<T>` / `IReadOnlyList<byte[]>` and
own every pointer dereference. Public callers never see `unsafe`.

Native handles are wrapped in `SafeHandle` subclasses so finalization
happens even if `Dispose` is missed — useful for short-lived
test/benchmark code that doesn't reliably reach a `using` scope.

Id buffers returned by `ztok_encode_batch_pooled`, `ztok_stream_feed`,
and `ztok_stream_finish` carry an opaque length-prefix header (see
`src/c_api.zig`); the only safe free is `ztok_ids_free`, which the
binding always uses. Never pass these pointers to
`Marshal.FreeHGlobal` / `NativeMemory.Free`.
