# Agent instructions — ztok

Fast, multithreaded tokenizer toolkit in **Zig 0.16** (byte-level BPE,
Unigram, WordPiece, TokenMonster) with a stable C ABI and 8 language
bindings (`bindings/{go,python,nodejs,ruby,rust,dotnet,java,swift}`).
Output is byte-identical with tiktoken / HuggingFace / SentencePiece —
that equivalence is the product; treat any parity regression as a
release blocker. AGPL-3.0-only.

## Build / test

```bash
zig build                      # static lib + ztok.h into zig-out/
zig build test                 # full unit suite (~1100 tests)
zig build test-install         # verify the install tree is consumer-complete
zig build bench                # tokenizer benchmark
zig build -p prefix -Doptimize=ReleaseFast   # refresh the local install prefix
```

`prefix/`, `zig-out/`, and stray artifacts (`libroot.a`, `vgcore.*`) are
gitignored — `prefix/{include,lib,bin}` is the **local install tree that
Adamaton consumers link against**; rebuild it (last command above) after
changing the C ABI surface (`include/ztok.h`, exported symbols).

## How Adamaton consumes ztok

- The Adamaton umbrella vendors this repo as its 8th submodule, but the
  canonical umbrella checkout usually does NOT have `./ztok` initialized.
  `r2g` and `plugin-host` ship-builds expect **ztok at the umbrella
  root** — create a temp symlink there when building those images.
- Local cgo builds of `knowledge/r2g` (chunking) and
  `platform/plugin-host` need **absolute** flag paths:
  `CGO_CFLAGS=-I/thearray/git/ztok/prefix/include`
  `CGO_LDFLAGS=-L/thearray/git/ztok/prefix/lib`.
  Relative paths break inside `bin/adam claim` worktrees.
- Docker images (`platform/worker`, `knowledge/reindex`, `knowledge/r2g`)
  build libztok in a dedicated **Zig build stage** and link it statically;
  this is what forced those images from alpine/CGO=0 to
  debian/CGO_ENABLED=1 on 2026-05-24. Only glibc is dynamic in the result.
- The Go binding (`bindings/go`) replaced `tiktoken-go` in r2g/chunking:
  byte-identical cl100k, ~5.7× faster. If you touch BPE merge logic, run
  the parity gates before calling it done.
- Verification gotcha: never check a cgo build with `go build ... | head`
  — the pipe masks the exit code; run the build bare and check `$?`.

## Conventions

- This repo is NOT governed by `bin/adam` claim/locks — standard flow:
  feature branch → PR → merge on GitHub (`sirus20x6/ztok`).
- Same commit hygiene as the umbrella: no `Co-Authored-By:` trailers, no
  `@`-mentions in commit bodies.
- `CHANGELOG.md` carries the per-release story (perf numbers +
  equivalence deltas); update it with any user-visible change. Releases
  are `release: ztok X.Y.Z` commits on main.
