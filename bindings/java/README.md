# ztok — Java binding

Java 21+ binding for [libztok](../../README.md). Uses the
**Foreign Function & Memory API** (JEP 442) — no JNI, no native glue
JAR. The whole binding is plain Java that talks to `libztok` via
`java.lang.foreign.Linker` downcall handles.

Status: **1.28.0**. Wraps the full stable C ABI surface from
`include/ztok.h`:

- `ztok_pipeline_new`, `ztok_pipeline_free`
- `ztok_pipeline_new_bpe_from_tiktoken`
- `ztok_pipeline_new_bpe_from_hf_json`
- `ztok_pipeline_new_wordpiece_from_hf_json`
- `ztok_pipeline_new_unigram_from_sp_model`
- `ztok_pipeline_new_monster_from_file`
- `ztok_encode`, `ztok_decode`
- `ztok_encode_batch`, `ztok_encode_batch_pooled`,
  `ztok_batch_pool_{new,free,worker_count}`, `ztok_ids_free`
- `ztok_auto_detect`
- `ztok_stream_{new,free,feed,finish}`
- `ztok_fingerprint`, `ztok_version`

## Install

```xml
<dependency>
  <groupId>com.anthropic</groupId>
  <artifactId>ztok</artifactId>
  <version>1.28.0</version>
</dependency>
```

The binding does **not** ship a bundled `libztok` — locate one of:

1. Set `ZTOK_LIB_PATH=/absolute/path/to/libztok.so` (preferred).
2. Install system-wide: `cp zig-out/lib/libztok.so /usr/local/lib/`.
3. Place it on `java.library.path` or `LD_LIBRARY_PATH`.
4. Run from `bindings/java/` for in-tree dev — the loader resolves
   `../../zig-out/lib/libztok.{so,dylib,dll}` relative to `cwd`.

Build libztok in-tree with:

```bash
zig build
```

## JVM flags

FFM was finalized in Java 22 (JEP 454) and shipped as preview in
Java 21 (JEP 442). The binding compiles and runs against both.

| JDK    | Required flags                                           |
|--------|-----------------------------------------------------------|
| 21 LTS | `--enable-native-access=ALL-UNNAMED --enable-preview`     |
| 22+    | `--enable-native-access=ALL-UNNAMED`                      |

Without `--enable-native-access`, the first downcall throws
`IllegalCallerException`. The Maven Surefire config in `pom.xml`
already passes both flags so `mvn test` works on Java 21 out of the box.

## Quickstart

```java
import com.anthropic.ztok.*;

public class Demo {
    public static void main(String[] args) {
        System.out.println("libztok " + Ztok.version());
        try (Pipeline pipe = Pipeline.byteId()) {
            int[] ids = pipe.encode("hello world");
            System.out.println("ids: " + ids.length);
            System.out.println("round-trip: " + pipe.decode(ids));
            System.out.println("fingerprint: " + pipe.fingerprint().hex());
        }
    }
}
```

Auto-detecting loader (`.tiktoken` / `tokenizer.json` / `.model` / `.ztm`):

```java
try (Pipeline pipe = Pipeline.open(java.nio.file.Path.of("cl100k_base.tiktoken"))) {
    int[] ids = pipe.encode("hello");
}
```

Parallel batch encoding:

```java
try (Pipeline pipe = Pipeline.byteId();
     BatchPool pool = BatchPool.create()) {  // 0 = auto-detect CPU count
    int[][] ids = pipe.encodeBatch(pool, java.util.List.of("foo", "bar", "baz"));
}
```

Streaming encode:

```java
try (Pipeline pipe = Pipeline.byteId();
     StreamEncoder enc = pipe.encodeStream("a long-lived input...")) {
    enc.asStream().forEach(chunk -> System.out.println("emit " + chunk.length));
}
```

## New in 1.28 — token-window chunking, n-gram hashing, RWKV

**Token-window chunking** for late-chunking embedding pipelines:

```java
try (Pipeline pipe = Pipeline.byteId()) {
    for (Chunk ch : pipe.chunk("the quick brown fox", 8, 2)) {
        System.out.printf("ids=%d bytes=%d..%d toks=%d..%d%n",
            ch.ids.length, ch.byteStart, ch.byteEnd, ch.tokenStart, ch.tokenEnd);
    }
}
```

**Engram n-gram hashing** — deterministic multi-head token-n-gram hashes
(row-major `[position][head]`, raw u64 in `long`s; mask to your table width):

```java
int[] ids = {10, 20, 30, 40, 50};
long[] hashes = Engram.hashNGrams(ids, 3, 4);  // unsigned bits
// hashes.length == (ids.length - n + 1) * heads == 12
```

**RWKV "World" tokenizer** — greedy longest-match byte trie (no normalizer
or pre-tokenizer; byte-lossless):

```java
try (Pipeline pipe = Pipeline.fromRwkv(java.nio.file.Path.of("rwkv_vocab_v20230424.txt"))) {
    int[] ids = pipe.encode("hello world");
}
```

## Threading

`Pipeline` is **thread-safe** for `encode` / `decode` / `encodeBatch` /
`fingerprint`. A `BatchPool` is thread-safe for `encodeBatch`. A
`StreamEncoder` is **not** thread-safe — drive it from a single thread.

## Memory model

Every encode/decode allocates a per-call `Arena.ofConfined()` that is
freed as soon as the call returns. The opaque libztok handles are
registered with `java.lang.ref.Cleaner` so a `Pipeline` / `BatchPool` /
`StreamEncoder` released without `close()` still has its native memory
reclaimed on the next GC — but explicit `close()` via
`try-with-resources` remains the strongly preferred path.

The C ABI's id buffers (`ztok_encode_batch_pooled`, `ztok_stream_*`)
carry a length-prefix header. They are **only** safe to free via
`ztok_ids_free`; this binding does that internally — callers never see a
raw pointer.

## Build / test locally

```bash
cd bindings/java
mvn package -DskipTests
mvn test            # skips fuzz + smoke if libztok is not loadable
```

Override fuzz iterations with `-Dztok.fuzz.iters=10000`.
