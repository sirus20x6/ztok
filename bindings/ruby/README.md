# ztok — Ruby bindings

Thin [`ffi`](https://github.com/ffi/ffi)-gem wrapper around `libztok` (Zig 0.16).
Mirrors the Python and Node.js bindings — same `Pipeline` + `BatchPool` +
`StreamEncoder` shape, same auto-detect dispatch.

## Quickstart

```ruby
require "ztok"

pipe = Ztok::Pipeline.from_path("tokenizer.json") # auto-detects format
ids  = pipe.encode("hello world")                  # => Array<Integer>
text = pipe.decode(ids)                            # => "hello world"
pipe.close

puts Ztok.version # => "1.20.0"
```

Block-form auto-closes (recommended):

```ruby
Ztok::Pipeline.from_path("cl100k_base.tiktoken") do |pipe|
  Ztok::BatchPool.open(workers: 8) do |pool|
    results = pipe.encode_batch(pool, ["hello world", "foo bar"])
    # => [[15339, 1917], [8134, 3703]]
  end
end
```

Streaming (id-as-it-arrives, 64 KiB chunks by default):

```ruby
File.open("large_doc.txt") do |f|
  pipe.encode_stream(f.read) do |chunk_ids|
    process(chunk_ids)
  end
end
```

Low-level `StreamEncoder` for arbitrary feed sizes (e.g. chunked HTTP body):

```ruby
encoder = Ztok::StreamEncoder.new(pipe)
begin
  until reader.eof?
    ids = encoder.feed(reader.read(64 * 1024))
    handle_ids(ids) unless ids.empty?
  end
  final = encoder.finish
  handle_ids(final) unless final.empty?
ensure
  encoder.close
end
```

## Install

1. Build `libztok` from the repo root:

   ```sh
   zig build
   ```

2. Install the gem (in-tree):

   ```sh
   gem install ./bindings/ruby
   ```

   Or, for development without a global install:

   ```sh
   cd bindings/ruby
   bundle install
   ```

3. Point the loader at `libztok`. Resolution order:
   1. `ENV["ZTOK_LIB_PATH"]` (explicit override).
   2. `<gem>/ext/libztok.{so,dylib,dll}` (native gem install layout).
   3. `<gem>/../../../../zig-out/lib/libztok.{so,dylib,dll}` (in-tree dev).
   4. `/usr/local/lib`, `/usr/lib`, `/usr/lib64`, `/opt/homebrew/lib`.

## API reference

### `Ztok::Pipeline`

Constructors (all accept an optional block; when a block is passed the
pipeline is yielded and closed on the way out):

- `Ztok::Pipeline.byte_id(normalizer:, pre_tokenizer:, decoder:)` —
  baseline byte_id pipeline (each input byte maps to its own id).
- `Ztok::Pipeline.from_tiktoken(path, cl100k: true, ...)` — .tiktoken vocab into a byte-level BPE pipeline.
- `Ztok::Pipeline.from_hf_json(path, cl100k: false, ...)` — HuggingFace `tokenizer.json` BPE.
- `Ztok::Pipeline.from_wordpiece(path, unk_id:, ...)` — WordPiece from HF `tokenizer.json`.
- `Ztok::Pipeline.from_sentencepiece(path, unk_id: 0, ...)` — SentencePiece `.model` (Unigram).
- `Ztok::Pipeline.from_monster(path, ...)` — ztok `.ztm` TokenMonster vocab.
- `Ztok::Pipeline.from_tekken(path, ...)` — Mistral Tekken `tekken.json` vocab (Nemo / Pixtral / Devstral).
- `Ztok::Pipeline.from_path(path, unk_id: 0, ...)` — auto-detect via the
  C ABI's `ztok_auto_detect` and dispatch to the right loader.

Instance methods:

- `#encode(text)` → `Array<Integer>`. Accepts a UTF-8 `String` or any
  bytes-like object.
- `#decode(ids)` → `String` (UTF-8, scrubbed if invalid).
- `#decode_bytes(ids)` → `String` (ASCII-8BIT, no UTF-8 round-trip).
- `#encode_batch(pool, inputs)` → `Array<Array<Integer>>`.
- `#encode_stream(text, chunk_size: 64 * 1024) { |ids| ... }` — yields
  arrays of newly-emitted ids per feed; returns an `Enumerator` if no
  block is given.
- `#close` — release the underlying C handle. Idempotent.
- `#closed?` — query.

### `Ztok::BatchPool`

- `Ztok::BatchPool.new(workers: 0)` — `0` means auto-detect CPU count.
- `Ztok::BatchPool.open(workers:) { |pool| ... }` — block form.
- `#workers` — actual worker count (resolves `0`).
- `#close` / `#closed?`.

### `Ztok::StreamEncoder`

- `Ztok::StreamEncoder.new(pipe)`
- `#feed(bytes)` → `Array<Integer>` of newly-emitted ids (possibly empty).
- `#finish` → trailing ids after final flush. Idempotent.
- `#close` / `#closed?`.

### Errors

- `Ztok::Error` — base class.
  - `Ztok::LibraryNotFoundError` — `libztok` couldn't be located.
  - `Ztok::OutOfMemoryError` — `ZTOK_ERR_OUT_OF_MEMORY` (status 1).
  - `Ztok::InvalidInputError` — `ZTOK_ERR_INVALID_INPUT` (status 2).
  - `Ztok::BufferTooSmallError` — `ZTOK_ERR_BUFFER_TOO_SMALL` (status 3).
  - `Ztok::InternalError` — `ZTOK_ERR_INTERNAL` (status 99) + unknown codes.

## Tests

```sh
cd bindings/ruby
ZTOK_LIB_PATH=../../zig-out/lib/libztok.so bundle exec rake test
# or without bundler:
ZTOK_LIB_PATH=../../zig-out/lib/libztok.so ruby -Ilib -Itest test/test_basic.rb
```

## Design notes

This is a thin `ffi` wrapper on top of the C ABI declared in
`include/ztok.h`. The binding never allocates an id array Ruby-side —
all id buffers are owned by libztok and freed via `ztok_ids_free` once
materialized into a Ruby `Array<Integer>`.

C handle lifetimes use `ObjectSpace.define_finalizer` (Ruby's analog
to Python's `weakref.finalize` / Node's `FinalizationRegistry`).
The finalizer proc is built by a class method that captures only the
raw pointer address (an `Integer`), NOT `self` — passing an instance
method or a block that closes over `self` to `define_finalizer` would
pin the receiver and silently disable the finalizer. Explicit
`close` / `BatchPool.open { |p| ... }` / `Pipeline.from_path { |p| ... }`
remain the recommended pattern; the finalizer is the safety net.
