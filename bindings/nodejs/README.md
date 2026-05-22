# ztok — Node.js bindings

Thin [koffi](https://koffi.dev/) wrapper around `libztok` (the C ABI shipped
with ztok 1.18+). One runtime dep, no native compile step, works on Node 16+.

## Install

Build the shared library first, then install the binding:

```sh
zig build                              # produces zig-out/lib/libztok.so
cd bindings/nodejs && npm install      # pulls in koffi (~600 KB)
```

At import time the package looks for `libztok` in this order:

1. `$ZTOK_LIB_PATH` (explicit override).
2. `<repo>/zig-out/lib/libztok.{so,dylib,dll}` (in-tree dev).
3. Package-relative `<pkg>/libztok.{so,dylib,dll}` (npm install layout).
4. `/usr/local/lib`, `/usr/lib`, `/usr/lib64` (Linux);
   `/usr/local/lib`, `/opt/homebrew/lib` (macOS).
5. Bare-name `dlopen` (resolves via `LD_LIBRARY_PATH` etc.).

If nothing works you get a `ZtokLibraryNotFoundError` listing every path tried.
For local development, set `ZTOK_LIB_PATH=../../zig-out/lib/libztok.so`.

## Quickstart

```javascript
const ztok = require('ztok');

const pipe = ztok.Pipeline.fromPath('tokenizer.json');   // auto-detects format
const ids  = pipe.encode('hello world');                 // -> Uint32Array
const text = pipe.decode(ids);                           // -> string
pipe.close();
```

`Pipeline.fromPath` sniffs the file format via the C ABI's `ztok_auto_detect`
(post-1.18 agent C) and dispatches to the right loader:

| File suffix / magic        | Loader                                       |
|----------------------------|----------------------------------------------|
| `.tiktoken`                | `Pipeline.fromTiktoken(..., { cl100k: true })` |
| `tokenizer.json` (HF JSON) | `Pipeline.fromHfJson(...)`                   |
| `.model` (SentencePiece)   | `Pipeline.fromSentencePiece(..., { unkId })` |
| `.ztm` (TokenMonster)      | `Pipeline.fromMonster(...)`                  |

You can also call the loaders directly when you want to pin the
pre-tokenizer / decoder, or set an `unkId` for SP / WordPiece models.

## Multithreaded batches

```javascript
const pipe = ztok.Pipeline.fromTiktoken('cl100k_base.tiktoken');
const pool = new ztok.BatchPool({ workers: 8 });
try {
    const results = pipe.encodeBatch(pool, ['foo', 'bar', 'baz']);
    // results: Uint32Array[]
} finally {
    pool.close();
    pipe.close();
}
```

Reuse one `BatchPool` across many `encodeBatch` calls — each pool owns its own
worker threads and arenas. Creating one per batch wastes setup work.

## Streaming encode

```javascript
const pipe = ztok.Pipeline.fromPath('cl100k.tiktoken');
for (const idsChunk of pipe.encodeStream(largeText, { chunkSize: 4096 })) {
    process.stdout.write(`${idsChunk.length} ids\n`);
}
pipe.close();
```

Internally wraps the post-1.18 C ABI `ztok_stream_*` family. The encoder
defers a trailing partial UTF-8 codepoint / pre-tokenizer span up to a 1 MiB
soft cap (see `src/stream.zig`); past that it force-cuts at the nearest
codepoint boundary.

## TypeScript

`index.d.ts` ships full declarations for every public method, including
precise types (`Uint32Array` for ids, `Buffer` for raw decoded bytes,
`string | Buffer | Uint8Array` for inputs). No `@types/ztok` package needed.

## API surface

| Symbol                                       | Purpose                                                       |
|----------------------------------------------|---------------------------------------------------------------|
| `ztok.version()`                             | Returns the libztok version string.                           |
| `ztok.Pipeline.byteId(opts?)`                | Baseline byte_id pipeline (one id per input byte).            |
| `ztok.Pipeline.fromPath(path, opts?)`        | Auto-detect format and dispatch to the right loader.          |
| `ztok.Pipeline.fromTiktoken(path, opts?)`    | Byte-level BPE from a `.tiktoken` file.                       |
| `ztok.Pipeline.fromHfJson(path, opts?)`      | BPE from a HuggingFace `tokenizer.json`.                      |
| `ztok.Pipeline.fromWordPiece(path, { unkId })` | WordPiece from a HuggingFace `tokenizer.json`.              |
| `ztok.Pipeline.fromSentencePiece(path, opts?)` | Unigram from a SentencePiece `.model`.                      |
| `ztok.Pipeline.fromMonster(path, opts?)`     | TokenMonster from a `.ztm` file.                              |
| `Pipeline.encode(text)` / `.decode(ids)`     | Encode / decode a single string (UTF-8).                      |
| `Pipeline.decodeBytes(ids)`                  | Decode to raw `Buffer` (no UTF-8 round-trip).                 |
| `Pipeline.encodeBatch(pool, inputs)`         | Multithreaded batch encode via a persistent pool.             |
| `Pipeline.encodeStream(text, opts?)`         | Stream-encode generator yielding `Uint32Array` chunks.        |
| `Pipeline.close()`                           | Release the underlying C handle. Idempotent.                  |
| `new BatchPool({ workers })`                 | Persistent worker pool (`workers=0` = auto).                  |
| `BatchPool.workers` / `.close()`             | Inspect worker count / release.                               |
| `ztok.detectFormat(path)`                    | Return `'tiktoken' \| 'hf_json' \| 'sentencepiece' \| 'ztm' \| 'unknown'`. |

Configuration constants (numeric):
`NORMALIZER_{IDENTITY,NFC,NFD,NFKC,NFKD,BYTE_LEVEL}`,
`PRETOK_{IDENTITY,CL100K}`,
`DECODER_{CONCAT,WORDPIECE,BYTE_LEVEL}`.

### Exceptions

All errors derive from `ZtokError`. Status codes map to subclasses:

| C status                      | JS class                            |
|-------------------------------|-------------------------------------|
| `ZTOK_ERR_OUT_OF_MEMORY`      | `ZtokOutOfMemoryError`              |
| `ZTOK_ERR_INVALID_INPUT`      | `ZtokInvalidInputError`             |
| `ZTOK_ERR_BUFFER_TOO_SMALL`   | `ZtokBufferTooSmallError`           |
| `ZTOK_ERR_INTERNAL` / unknown | `ZtokInternalError`                 |

Library-load failures throw `ZtokLibraryNotFoundError` (distinct because they
happen at module import, before any pipeline exists).

## Notes on the C ABI

- This is a *thin* wrapper. The C library owns every id buffer and vocab
  table; we never allocate id arrays on the JS side. Per-input id buffers
  returned by `ztok_encode_batch_pooled` carry a length-prefix header
  (see `src/c_api.zig::allocIdBuf`) and **must** be freed via `ztok_ids_free`
  — `Pipeline.encodeBatch` calls that under the hood after materializing
  the ids into `Uint32Array`s.
- Handle lifetime is managed with `FinalizationRegistry`, so dropping a
  `Pipeline` or `BatchPool` without `close()` still releases the C resource
  on GC. Prefer explicit `close()` (or `try/finally`) for determinism —
  finalizers fire on GC schedule, not when you'd like.
- The C ABI's `ztok_encode` reports the required buffer size on
  `BUFFER_TOO_SMALL`, but the per-span `maxTokensFor` bound is conservative,
  so the JS wrapper allocates a generous capacity upfront (input bytes +
  headroom) and grows on demand.
- koffi's pointer-in/pointer-out idioms use **single-element arrays**
  (e.g. `const status = [0]; fn(..., status); console.log(status[0]);`).
  We use that pattern everywhere for `_Out_` parameters.

## Run the tests

```sh
cd bindings/nodejs
ZTOK_LIB_PATH=../../zig-out/lib/libztok.so node --test 'test/*.test.js'
```

Tests use Node's built-in `node:test` runner (no test-framework dep). Covers
version, encode/decode round-trip, 100-line stress, batch encode (1000 inputs
without leaks), context-style `close()`, typed error mapping, `fromPath`
auto-dispatch across all four formats, streaming encode against single-shot,
and incremental streaming yields.
