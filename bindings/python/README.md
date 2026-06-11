# ztok — Python bindings

Thin `ctypes` wrapper around `libztok` (the C ABI shipped with ztok 1.16). Zero
non-stdlib runtime deps; works on any CPython 3.10+.

## Install

Build the shared library first, then `pip install` this package:

```sh
zig build                       # produces zig-out/lib/libztok.so
pip install ./bindings/python   # editable: `pip install -e ./bindings/python`
```

At import time the package looks for `libztok` in this order:

1. `$ZTOK_LIB_PATH` (explicit override).
2. Next to the `ztok/` Python package (wheel install).
3. `zig-out/lib/libztok.so` relative to the repo root (in-tree dev).
4. `ctypes.util.find_library("ztok")`.
5. `/usr/local/lib/libztok.so`, `/usr/lib/libztok.so`, `/usr/lib64/libztok.so`.

If none work you get a `ZtokLibraryNotFoundError` with the list of paths
tried. For local development, set `ZTOK_LIB_PATH=zig-out/lib/libztok.so`.

## Quickstart

```python
import ztok

with ztok.Pipeline.from_path("tokenizer.json") as pipe:
    ids = pipe.encode("hello world")
    text = pipe.decode(ids)
    print(ids, text)
```

`Pipeline.from_path` sniffs the file format and dispatches to the matching
loader:

| File suffix / magic        | Loader                                     |
|----------------------------|--------------------------------------------|
| `.tiktoken`                | `Pipeline.from_tiktoken(..., cl100k=True)` |
| `tokenizer.json` (HF JSON) | `Pipeline.from_hf_json(...)`               |
| `.model` (SentencePiece)   | `Pipeline.from_sentencepiece(..., unk_id)` |
| `.ztm` (TokenMonster)      | `Pipeline.from_monster(...)`               |

You can also call the loaders directly when you want to pin pre-tokenizer /
decoder choices, or to set `unk_id` for SP/WordPiece models.

## Multithreaded batches

```python
with ztok.Pipeline.from_tiktoken("cl100k_base.tiktoken") as pipe, \
     ztok.BatchPool(workers=8) as pool:
    results = pipe.encode_batch(pool, ["foo", "bar", "baz"])
    # results: list[list[int]]
```

A single `BatchPool` should be reused across many `encode_batch` calls — the
pool owns persistent worker threads + arenas.

## API surface

| Symbol                                  | Purpose                                                     |
|-----------------------------------------|-------------------------------------------------------------|
| `ztok.version()`                        | Returns the libztok version string (e.g. `"1.28.0"`).       |
| `ztok.Pipeline.byte_id(...)`            | Baseline byte_id pipeline (one id per input byte).          |
| `ztok.Pipeline.from_path(path)`         | Auto-detect format and dispatch to the right loader.        |
| `ztok.Pipeline.from_tiktoken(path, cl100k=True)` | Byte-level BPE from a `.tiktoken` file.            |
| `ztok.Pipeline.from_hf_json(path)`      | BPE from a HuggingFace `tokenizer.json`.                    |
| `ztok.Pipeline.from_wordpiece(path, unk_id)` | WordPiece from a HuggingFace `tokenizer.json`.         |
| `ztok.Pipeline.from_sentencepiece(path, unk_id)` | Unigram from a SentencePiece `.model`.             |
| `ztok.Pipeline.from_monster(path)`      | TokenMonster from a `.ztm` file.                            |
| `Pipeline.encode(text)` / `.decode(ids)` | Encode / decode a single string (UTF-8).                   |
| `Pipeline.encode_batch(pool, inputs)`   | Multithreaded batch encode via a persistent pool.           |
| `Pipeline.close()` / `with Pipeline ...`| Release the underlying C handle.                            |
| `BatchPool(workers=N)`                  | Persistent worker pool (`workers=0` = auto).                |
| `BatchPool.workers` / `.close()`        | Inspect worker count / release.                             |

Configuration constants: `NORMALIZER_{IDENTITY,NFC,NFD,NFKC,NFKD,BYTE_LEVEL}`,
`PRETOK_{IDENTITY,CL100K}`, `DECODER_{CONCAT,WORDPIECE,BYTE_LEVEL}`.

### Exceptions

All errors derive from `ZtokError`. Status codes map to subclasses:

| C status                      | Python                            |
|-------------------------------|-----------------------------------|
| `ZTOK_ERR_OUT_OF_MEMORY`      | `ZtokOutOfMemoryError` (also `MemoryError`) |
| `ZTOK_ERR_INVALID_INPUT`      | `ZtokInvalidInputError`           |
| `ZTOK_ERR_BUFFER_TOO_SMALL`   | `ZtokBufferTooSmallError`         |
| `ZTOK_ERR_INTERNAL` / unknown | `ZtokInternalError`               |

## New in 1.28 — token-window chunking, n-gram hashing, RWKV

**Token-window chunking** for late-chunking embedding pipelines:

```python
with ztok.Pipeline.byte_id() as pipe:
    for ch in pipe.chunk("the quick brown fox", max_tokens=8, overlap=2):
        print(ch.ids, ch.byte_start, ch.byte_end, ch.token_start, ch.token_end)
```

**Engram n-gram hashing** — deterministic multi-head token-n-gram hashes
(row-major `[position][head]`, raw u64; mask to your table width):

```python
ids = [10, 20, 30, 40, 50]
hashes = ztok.ngram_hash(ids, n=3, heads=4)
# len(hashes) == (len(ids) - n + 1) * heads == 12
```

**RWKV "World" tokenizer** — greedy longest-match byte trie (no normalizer
or pre-tokenizer; byte-lossless):

```python
with ztok.Pipeline.from_rwkv("rwkv_vocab_v20230424.txt") as pipe:
    ids = pipe.encode("hello world")
```

## Notes on the C ABI

- This is a *thin* wrapper. The C library owns every id buffer and vocab
  table; we never copy or re-allocate id arrays on the Python side. The
  per-input id buffers returned by `ztok_encode_batch_pooled` are
  prefixed with a length header (see `src/c_api.zig::allocIdBuf`) and
  **must** be freed via `ztok_ids_free` — `Pipeline.encode_batch` calls
  that under the hood after materializing the ids into Python lists.
- Handle lifetime is managed with `weakref.finalize`, so dropping a
  `Pipeline` or `BatchPool` without closing it still releases the C
  resource on GC. Prefer the context-manager pattern for determinism.
- The C ABI's `ztok_encode` reports the required buffer size on
  `BUFFER_TOO_SMALL`, but the per-span `maxTokensFor` bound is
  conservative, so the Python wrapper allocates a generous capacity
  upfront (input bytes + headroom) and grows on demand.

## Run the tests

```sh
cd bindings/python
ZTOK_LIB_PATH=../../zig-out/lib/libztok.so python3 -m pytest tests/ -v
```

There are 26 tests covering: version, encode/decode round-trip, batch
encode (1000 inputs without leaks), context-manager close, typed error
mapping, and `from_path` dispatch for tiktoken / HF / SP / ztm.
