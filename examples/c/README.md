# ztok C consumer examples

`hello.c` loads a `.tiktoken` BPE vocab, encodes one string, prints the
ids, and round-trips them back to the original text via `ztok_decode`.

`overlays.c` encodes one string with `ztok_encode_with_overlays` and
prints the per-token annotation channels (byte span, boundary bitset,
provenance) aligned 1:1 with the ids — the substrate for semantic-overlay
experiments.

Both build paths (CMake and pkg-config) assume you've installed ztok to
a prefix. Build that prefix from the repo root with:

```sh
cd /thearray/git/ztok
mkdir -p build_install && cd build_install
zig build -p $PWD --build-file ../build.zig
```

That produces:
- `<prefix>/lib/libztok.so` + `libztok.a`
- `<prefix>/include/ztok.h`
- `<prefix>/lib/cmake/ztok/ztokConfig.cmake` + `ztokTargets.cmake` +
  `ztokConfigVersion.cmake`
- `<prefix>/lib/pkgconfig/ztok.pc`

## Build with CMake

```sh
cd examples/c
mkdir -p build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/install ..
cmake --build .
./hello /path/to/cl100k_base.tiktoken "the quick brown fox"
./overlays /path/to/cl100k_base.tiktoken "the quick brown fox"
```

## Build with pkg-config

```sh
cd examples/c
PKG_CONFIG_PATH=/path/to/install/lib/pkgconfig make
./hello /path/to/cl100k_base.tiktoken "the quick brown fox"
./overlays /path/to/cl100k_base.tiktoken "the quick brown fox"
```

Either path produces a binary that links to `libztok.so` with the right
include path and embeds an RPATH so the dynamic loader finds the
library without `LD_LIBRARY_PATH=` plumbing.
