# Changelog

All notable changes to ztok land here. Format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/) — each
release lists what landed under `Added` / `Changed` / `Fixed` /
`Performance` / `Equivalence` headings as relevant. ztok uses
semantic-ish versioning: minor bumps for new pipeline stages,
loaders, training kinds, or large perf waves; patch bumps for
focused fixes inside an existing surface.

Headline numbers throughout this changelog come from `bench/RESULTS.md`
(AMD EPYC 7473X reference box: 24 physical / 48 SMT cores, ReleaseFast,
10 MB mixed-text corpus, unless noted otherwise).

## [1.27.0] — 2026-05-21

**Headline: the first public release.** Completes overlays across all 8
language bindings, exposes the overlay domain through the C ABI, brings
Mistral Tekken to 100% equivalence, finishes streaming-over-TLS, and
makes per-script tokenization (and the `eval --fairness` metric) real.
Now bit-identical with every SentencePiece / HuggingFace / Tekken
reference cell except one float-determinism Viterbi tie-break
(`t5 × code`, 99.87%). **1099/1100 tests** (1 CI-only skip).

### Added

- **Overlay channels now in all 8 bindings** — added Go, Rust, Ruby,
  .NET, Java, and Swift `encode_with_overlays` / `EncodeWithOverlays`
  (Python + Node shipped in 1.26). Each follows the two-pass C ABI
  sizing protocol. (Swift written + reviewed but not compile-tested —
  no `swift` toolchain in the dev env, matching the original binding.)
- **C ABI `ztok_pipeline_set_overlay_domain(p, domain)`** + the
  `ztok_overlay_domain` enum — lets FFI consumers switch on the x86-64
  `opcode_class`/`operand_class` domain overlay channels (previously
  Zig-only).
- **`ztok serve` streaming routes now work over TLS** — `/encode_stream`
  (NDJSON), `/encode_chunked`, and `/encode_ws` (WebSocket) previously
  returned 501 on the TLS path; they now frame through MbedTLS via
  `std.Io` reader/writer adapters (build with `-Dtls=mbedtls`).

### Fixed

- **Mistral Tekken parity: 57–94% → 100%** across all corpora. Two bugs:
  (1) Tekken needs its own pre-tokenization regex (case-aware word
  split, single-codepoint digits, `/`-terminated punctuation) — added
  `tekken_pretok.zig` instead of reusing the cl100k splitter; (2) the
  vocab must be capped to `default_vocab_size − num_special_tokens`
  mergeable ranks, else high-rank tokens form that the reference never
  produces. Confined to Tekken-only code paths; SP/HF parity untouched.

### Changed

- **`eval` per-script token counts are now real** — computed by encoding
  maximal same-script runs, replacing the v1 proportional-by-codepoint-
  share approximation. This makes `eval --fairness` meaningful: on the
  multilingual corpus it now surfaces the real disparity (Gini ≈ 0.16,
  max/min fertility ratio ≈ 3.04 — CJK pays ~3× the token premium of
  ASCII) instead of collapsing to ≈ 0.

## [1.26.0] — 2026-05-21

**Headline: "beyond bit-identical" wave + a parallel feature wave.** ztok
already matches tiktoken / HF / SentencePiece bit-for-bit; 1.26 adds
capabilities none of the four reference libraries offer — optimal
(minimum-token) encoding, PathPiece vocab training, per-token overlay
channels, an x86-64 normalized-instruction classifier feeding those
channels, multimodal audio, transcoding, a tokenization debugger, and a
multilingual-fairness metric. Two multi-agent waves shipped under the
disjoint-file guardrails (no `main.zig`/`c_api.zig`/VERSION contention),
each closed with a cold-cache rebuild. **1084/1085 tests pass (1
CI-only skip)**, integrated-verified green. 13/13 SP+HF pairs remain at
100/100 on the 100-line gate.

### Added

- **Optimal (minimum-token) encoding** — `EncodeMode.optimal`: a DP over
  the vocab lattice emits the provably fewest tokens for a vocab.
  `ztok encode --optimal` and `ztok roundtrip --optimal`.
- **PathPiece vocabulary learner** — `ztok train --kind pathpiece`:
  top-down, Corpus-Token-Count-minimizing vocab construction (Schmidt et
  al., EMNLP 2024), the training-time complement to optimal segmentation.
  Emits a `.tiktoken` vocab; load as bpe + run `--optimal`. Baseline and
  marginal-cost DP loops parallelize over an optional `BatchPool`,
  bit-identical to the serial path.
- **Overlay channels** — `encodeWithOverlays` (Zig) and
  `ztok_encode_with_overlays` (C ABI): per-token annotation channels
  (byte span, boundary bitset, provenance, + pluggable domain channels)
  aligned 1:1 with the id stream, without altering tokenization. Exposed
  in the Python and Node.js bindings (`encode_with_overlays` /
  `encodeWithOverlays`); `examples/c/overlays.c` consumer.
- **x86-64 normalized-instruction classifier** (`asm_normalizer.zig`) and
  `Pipeline.overlay_domain = .x86_64`, which populates the
  `opcode_class`/`operand_class` overlay channels by classifying input
  machine code (immediates/displacements masked).
- **Grammar-constrained tokenization** (`constrained.zig` + `hf_regex`
  prefix-status NFA), **token healing** (`token_healing.zig`),
  **cross-tokenizer transcoding** (`ztok transcode`), and the
  **`ztok explain`** tokenization debugger (per-token why-chosen).
- **Multimodal audio encode** — `encodeAudio` / `encodeMultimodal` over
  Tekken `AudioConfig`; `ztok encode-multimodal` for text + image + audio.
- **Multilingual-fairness metric** (`fairness.zig`) + `ztok eval
  --fairness`: per-script fertility, premium (×vs best), max/min ratio,
  and Gini. (Note: per-script token counts are still eval's v1
  proportional approximation — true per-script tokenization is a
  follow-up before the numbers are authoritative.)

### Changed

- `ztok roundtrip` now honors `--optimal`, matching `encode --optimal`
  (it previously always validated under greedy merge segmentation).

### Performance

- **SIMD byte scanners widened to the target-native vector width** —
  `simd_bytes.zig` byte-class scanners + `copyPrintableAscii` pick width
  via `std.simd.suggestVectorLength` (32-lane AVX2 / 64 AVX-512 / 16
  SSE2·NEON·wasm128), cascading widest→16→scalar. NEON/SSE2/wasm keep the
  prior 16-lane path exactly; AVX2 hosts process twice the bytes per
  iteration on the dominant ASCII-run scan.

## [1.25.0] — 2026-05-19

**Headline: 12-agent Wave 5 ships in parallel — projected ≥6 additional
stress cells lift (integrated sweep deferred while user is gaming);
8th language binding (Swift); Tekken multimodal encode; MbedTLS
HTTP-over-TLS framing finished; persistent prefix cache wired into
serve; vocab-merge tool; negative training (security-aware vocab
training); Monster encoder +8.6-15% wall-clock; WASM SIMD scanners;
chunk dict-based word segmenter for Thai/Lao/zh/ja.** 13/13 SP+HF
pairs remain at 100/100 on the 100-line gate. Test count agent-reported
~988 (1 CI-only skip); integrated re-run pending.

### Added

- **Swift binding** (`bindings/swift/`, 8th language): Swift 5.9+, SPM,
  `pkgConfig: "ztok"` auto-picks up `zig-out/lib/pkgconfig/ztok.pc`.
  Pipeline/BatchPool/StreamEncoder/Fingerprint safe wrappers,
  `AsyncSequence<UInt32>` streaming, typed `ZtokError` enum,
  withMemoryRebound for UInt8↔CChar interchange. macOS 11+/Linux
  first-class (iOS/tvOS noted as needing XCFramework + static link).
  13 XCTest smoke + 1 fuzz test (1000-iter PRNG round-trip).
- **Tekken multimodal encode**: `pub fn encodeImage(allocator, model,
  image_dims) ![]TokenId` materializes Pixtral-style placeholder
  sequences via existing `placeholderImageTokens`. `pub fn
  encodeMultimodal(allocator, model, content: []const ContentPart)`
  interleaves text encodes with image-placeholder encodes for mixed-
  content messages.
- **MbedTLS HTTP-over-TLS framing** (`src/cli_serve.zig` TLS path):
  finishes 4D's handshake-only stub. Hand-parses HTTP/1.1 request line
  + headers via `mbedtls.Conn.read`, dispatches to existing
  `handleRequest`, writes response via `mbedtls.Conn.write`. `GET
  /health /version /metrics` + `POST /encode /decode /eval` flow
  end-to-end. Auth (bearer + OIDC), rate limiting, metrics, status
  accounting all flow through. Streaming routes (`/encode_stream`,
  `/encode_chunked`, `/encode_ws`) explicitly return 501 on TLS path —
  Wave 6.
- **`ztok serve --prefix-cache-dir PATH`**: wires 3H's persistent
  `prefix_cache.openPersistent` into the serve encode handler. First
  256 bytes of input key the cache; on hit, encode the suffix only
  and splice. Cache survives restart.
- **`ztok merge-vocab`** (`src/vocab_merge.zig`): merges two same-kind
  vocabs into one with unioned tokens. Preserves vocab A ids
  bit-identical, appends B (or B-with-prefix). `--on-conflict
  keep-a|keep-b|error`. `--prefix-b STR` for unrelated vocabs. Emits
  `<output>.merge-map.json` sidecar with `b_to_merged` map for
  embedding-table re-indexing. Cross-kind merges (BPE × Unigram)
  return `Error.IncompatibleModelKind`. mergeBpe / mergeUnigram /
  mergeWordPiece variants.
- **`ztok train --avoid PATTERNS_FILE`** + `--avoid-mode penalize|exclude`
  (`src/negative_train.zig` + train_bpe/unigram/monster.zig). Security/
  safety negative training: BPE merge scorer / Unigram seed vocab /
  Monster distillation all consult an `AvoidList` and demote (or
  exclude) candidates whose merged bytes contain any avoid-pattern
  substring. PENALTY = -1_000_000 pushes penalize-mode tokens to the
  bottom of every prune sort. Byte pieces exempted in Unigram (byte_fallback
  contract). Multi-pattern substring matcher; Aho-Corasick deferred as
  follow-up if avoid-lists grow into the thousands.
- **`Boundary.word_dict` + `Locale.th` / `Locale.lo`** in
  `src/chunk.zig`: dict-based word segmenter for scriptio-continua
  languages. Bundled tiny dictionaries: Thai (~465 entries), Lao
  (~465), Chinese (~888 HSK 1-3), Japanese (~855 hiragana/common kanji/
  katakana loans). Total ~2673 entries / ~25.7 KB raw UTF-8. Linked-in
  cost ~74 KB for Zig consumers; 0 KB for C ABI consumers via DCE.
  Longest-prefix-match with single-codepoint fallback. Documented as
  approximation-grade for "good enough" RAG chunking.
- **`src/simd_bytes.zig`**: new module. `@Vector(16, u8)` range/
  equality compare → packed bitmask → `@ctz(~bits)` first-mismatch lane.
  Exports `scanAsciiWs`/`Letter`/`Digit`/`Punct`/`WsNotNl` +
  `copyPrintableAscii`. Lowers to `i8x16.bitmask` + `i8x16.lt_u` +
  `i8x16.all_true` etc. on WASM SIMD128.
- **`unicode_props.isPunct` + `cat_P` range table** (198 ranges from
  UCD 16.0 DerivedGeneralCategory): full Pc|Pd|Pe|Pf|Pi|Po|Ps support.
  Used by `hf_bytelevel_pretok.isPunct` to fix Falcon (and any HF
  ByteLevel chain that uses Punctuation pretok).

### Changed

- **`src/normalizer.zig` SP normalizer**: skips standalone NFKC pass
  when `cfg.charsmap != null`. SP's `nmt_nfkc`/`nfkc`/`nfkc_cf`
  charsmaps already bake the full NFKC composition table into their
  Darts trie; pre-composing first ran sequences the trie was built to
  keep decomposed. Both `spNormalize` and `spNormalizeWithOrigin`
  gated. Doesn't regress charsmap-free models (gating preserves their
  existing NFKC path). Fixes T5 × unicode_stress 96.5→100%.
- **`src/hf_bytelevel_pretok.isPunct`**: replaced coarse block-range
  check (`0x2000-0x206F`, `0x3000-0x303F`, `0xFF00-0xFFEF`) with
  `unicode_props.isPunct`. Cf bidi isolates and other non-P*
  codepoints no longer mis-classified as punctuation. Fixes Falcon ×
  unicode_stress 92→100% + × code 99.99→100%.
- **`bench/bench_cross.zig` SP-BPE arm** builds an
  `added_tokens.Scanner` from SP `.user_defined` pieces (NOT `.control`
  pieces — SP splits those char-by-char). Lifts gemma/yi6b code 99.92/
  99.55→100/100. Also removed an over-applied scanner on the
  `.hf_unigram` arm that was causing llmjp3 code 99.84% by resolving
  `<unk>` substring against added_tokens when Python reference uses
  the lattice path.
- **`src/unigram.zig` + `src/sp_bridge.zig`**: `Builder.excludeFromTrie`
  API + `buildTrieExcluding`; `unigramFromSP` flags `.unknown`/`.byte`/
  `.control`/`.unused` pieces for exclusion. Literal `<unk>` in source
  no longer picks id 2 over the `<`+`unk`+`>` lattice path. Lifts t5
  × code 99.75→99.87%.
- **`src/tekken.zig` version resolution**: accepts top-level `version`
  OR nested `config.version` (`"v3"` form or bare integer). Unblocks
  real Mistral-Nemo `tekken.json` load.
- **`src/monster.zig` perf**: new `root_child_table: [256]u32` field +
  fast-path in `findChild` for `node_idx == 0` — one indexed load
  instead of 80-200-child binary search at the trie root. 1 KiB fixed
  memory per Monster. Wall-clock **8.6% faster on english, 15% on
  code**. (Score-eval state was already SoA from a prior pass — the
  candidate `cand_lens[]`/`cand_ids[]`/`br_ids[3]`/`br_lens[3]` arrays
  and SoA trie `child_bytes[]`/`child_nodes[]` were established
  earlier; the actual hot bottleneck was the trie root child search.)
- **`src/cl100k.zig` + `src/hf_bytelevel_pretok.zig` + `src/byte_level.zig`**:
  routine ASCII-prefix matchers (`matchWord`, `matchPunct`,
  `matchWsNewline`, `matchTrailingWs`, `matchWs`, `matchDigits`,
  `encodeBytes`) now SIMD-scan via `simd_bytes` then drop to scalar
  for non-ASCII codepoints. Native cl100k single-thread **~5-10%
  faster** (BPE encode dominates the full pipeline so the raw pretok
  lift is partially masked). WASM browser binary +4.9 KB, +59 SIMD
  opcodes.

### Fixed

- **t5 × unicode_stress 96.5→100%** (NFKC-before-charsmap was
  pre-composing sequences the Darts trie was built to keep decomposed).
- **falcon × unicode_stress 92→100% + × code 99.99→100%** (Cf bidi
  isolates U+2068/U+2069 mis-classified as punctuation broke
  byte-level merges on RTL-isolated Arabic/Hebrew).
- **gemma/yi6b × code 99.92/99.55→100%** (SP `.user_defined`
  pre-matching wasn't wired in `bench_cross`).
- **llmjp3_hf × code 99.84→100%** (HF Unigram added_tokens scanner
  was over-applied in `bench_cross`).
- **t5_unigram × code 99.75→99.87%** (literal `<unk>` no longer
  picks id 2; 13 residual long-dash Viterbi tie-breaks deferred).

### Equivalence (projected — pending integrated sweep)

| fixture            | 1.24 cells 100/100 | 1.25 projected      |
|--------------------|:------------------:|:-------------------:|
| LLaMA-2 SP-BPE     | 5/5                | 5/5                 |
| T5 SP-Unigram      | 3/5                | **5/5** (lift × unicode_stress + bonus) — 99.87% × code deferred |
| Gemma SP-BPE       | 4/5                | **5/5** (× code 99.92→100) |
| GPT-2 HF BPE       | 5/5                | 5/5                 |
| Mistral-7B SP-BPE  | 5/5                | 5/5                 |
| Yi-6B SP-BPE       | 4/5                | **5/5** (× code 99.55→100) |
| DeepSeek-V2-Lite   | 5/5                | 5/5                 |
| llm-jp-3 HF Uni    | 4/5                | **5/5** (× code 99.84→100) |
| bert-base-uncased  | 5/5                | 5/5                 |
| Falcon-7B HF BPE   | 3/5                | **5/5** (× unicode_stress + × code) |
| Qwen2-7B HF BPE    | 5/5                | 5/5                 |
| Llama-3-8B HF BPE  | 5/5                | 5/5                 |
| Phi-3-mini HF BPE  | 5/5                | 5/5                 |
| **TOTAL**          | **58/65**          | **~64-65/65 projected** |

Projection caveat: separate agents claimed overlapping fixes (e.g. 5B
+ 5C both touched t5 × unicode_stress paths; 5B + 5D both touched
falcon × unicode_stress). The integrated state after all changes
co-exist may differ from each agent's individually-measured snapshot.
Integrated re-run deferred — user gaming.

### Performance

- **Monster encoder**: english.txt 69.68→63.68 ms median (-8.6%); code.txt
  56.35→47.88 ms median (-15.0%). 5-run median, side-by-side same machine.
- **Native cl100k single-thread**: ~5-10% faster on the full pipeline
  via SIMD prefix scanners. BPE encode still dominates wall-clock.
- **WASM SIMD opcodes**: 151 → 210 (+59) in the browser binary;
  binary size 398,749 → 403,634 bytes (+1.2%).
- **AVX-512 audit**: reference box is **Zen 3** (EPYC 7473X family 25
  model 1), NOT Zen 4 as prior README claimed. Default ReleaseFast
  doesn't emit ZMM. With `-Dcpu=znver4` LLVM emits 17K ZMM uses but
  binary SIGILLs on Zen 3. No code changes shipped — current SIMD
  dispatch is correct, and the `simd_min.scanMinWide` path beats
  narrow at len≥4096 by 14-21% even on AVX-2 baseline. Full audit
  appended to `bench/RESULTS.md`.

### Notes

- **Integrated stress sweep + clean rebuild + perf bench deferred** —
  user is gaming. Run after they signal done:
  ```sh
  rm -rf .zig-cache && zig build test --summary all
  bash bench/equivalence_stress_sweep.sh --first-diff-only \
       > bench/_results/stress_1.25.ndjson
  ./zig-out/bin/bench_ztok --quick
  ```
- **Several agents fixed pre-existing parallel-WIP breakage** in
  `src/vocab_merge.zig` (trailing doc-comment), `src/main.zig` (duplicate
  `cmdMergeVocab` stub), `src/cli_serve.zig` (duplicate `handleTlsRequest`),
  and `src/chunk.zig` (duplicate `snapLeftWordDict`) — these were
  cross-agent integration issues that resolved by the time everyone
  finished.
- **README header had wrong CPU family** for the reference box. EPYC
  7473X is **Zen 3** (Milan-X), not Zen 4. AVX-512 unavailable. Fixed
  in this entry.

## [1.24.0] — 2026-05-19

**Headline: 12-agent Wave 4 ships in parallel — stress-sweep 56→58/65
cells at 100/100 (+2; +12 vs 1.21 baseline), 11 of 13 fixtures now
hold 5/5 across all 5 corpora; .NET + Java bindings (now 7 languages);
VS Code extension; chat template inversion; `ztok visualize` + `ztok
adapt-vocab`; .ztm format v2 with alias section lifts TM nocapcode
78→85/100 + full-capcode 72→77/100; OIDC RS256 + JWKS auto-fetch;
MbedTLS handshake (opt-in via `-Dtls=mbedtls`); demo site +
GitHub Pages workflow.** Test count 872 → 924 (+52, 1 CI-only skip).
13/13 SP+HF pairs remain at 100/100 on the 100-line gate.

### Added

- **.NET binding** (`bindings/dotnet/`, 6th language): net8.0,
  `RollForward=LatestMajor` so the test suite runs on machines with
  any post-net8 LTS installed. `[ModuleInitializer]` calls
  `NativeLibrary.SetDllImportResolver` to honor `ZTOK_LIB_PATH`.
  `Pipeline : IDisposable` with `SafeHandle` subclass + `Cleaner`
  finalizer fallback. `IAsyncEnumerable<uint[]>` streaming.
  `unsafe` confined to `Native.cs`. 12 xUnit smoke + 1 fuzz test.
  All 22 exported C ABI symbols wrapped except `ztok_encode_batch`
  (per-call variant; pooled variant promoted).
- **Java binding** (`bindings/java/`, 7th language): Java 21+ FFM API
  JEP 442 — **no JNI**. `java.lang.foreign.{Linker, MethodHandle,
  MemorySegment, Arena, SymbolLookup}`. `AutoCloseable` + `Cleaner`.
  Maven build. 10 JUnit 5 + 1 fuzz test (1000 PRNG round-trips). Requires
  `--enable-preview` + `--enable-native-access=ALL-UNNAMED` on Java 21;
  `--enable-preview` not needed on 22+.
- **VS Code extension** (`tools/vscode-ztok/`): live token highlighting
  (8-color rotating palette, dark+light theme adaptive via
  `vscode.ThemeColor`), status-bar token count, hover token info.
  `ZTok: Choose tokenizer...` + `ZTok: Toggle token highlighting`
  commands. Vocab resolution: workspace config → in-tree bench/vocabs →
  `~/.cache/vscode-ztok/` → download cl100k_base on first use. 5/5
  vscode-test-electron tests pass.
- **Chat template inversion** (`src/chat_template.zig`):
  `pub fn invert(allocator, template_kind, rendered_text) ![]ParsedMessage`.
  Parses ChatML, Mistral, Llama-2, Gemma rendered templates back into
  `[]ParsedMessage` (`{role, content, partial}`). Roundtrip property
  verified on all 4. Partial trailing turns marked `partial: true`.
  22 new tests.
- **`ztok visualize <vocab>`** (`src/vocab_viz.zig`): self-contained HTML
  report (~70 KB for typical 32K vocab, no external deps). Sections:
  top-K frequency grid (cells colored by log frequency, clickable
  inspector panel), 256-cell byte-distribution heatmap (16x16 table),
  7-bucket length histogram (CSS-flex bars), first-50 BPE merge
  dendrogram (inline SVG), added/special token table. CLI:
  `ztok visualize <vocab> [--corpus PATH] [--out report.html] [--top-k 200]`.
- **`ztok adapt-vocab`** (`src/vocab_continued_pretrain.zig`):
  continued-pretraining vocab adaptation. Preserves ALL base ids
  bit-identical (pretrained embedding tables stay valid), appends N new
  tokens trained on the worst-compressed sub-corpus. JSON report.
  Identity old↔new map. Supports tiktoken / HF BPE / SP BPE; returns
  `error.UnsupportedFormat` for Monster/Tekken/Unigram/WordPiece.
- **Public demo site** (`demo/` + `.github/workflows/deploy-demo.yml`):
  52 KB HTML+CSS+JS bundle, plus WASM + 2 vendored vocabs
  (cl100k_base.tiktoken + gpt2.json). Tokenizer picker, custom upload,
  live highlight, comparison mode, cost estimator. GitHub Pages
  workflow triggers on `push: main` to demo/** or `src/wasm_browser_root.zig`,
  builds both SIMD128 + scalar WASM, publishes via `actions/deploy-pages@v4`.
- **Tekken cross-bench wiring** (`bench/bench_cross.zig` +
  `bench/equivalence_check.py`): new `--kind tekken` arm uses
  `mistral_common.Tekkenizer` as primary reference, `tiktoken`-from-JSON
  fallback. 8 wave-3L fixtures wired into smoke + stress sweep scripts.
- **`Bpe.ignore_merges`** (`src/bpe.zig`): new bool field. When true,
  `encodeChunkScratch` (+ offsets + 2 trace variants) probes the entire
  pretok chunk against the vocab BEFORE running the merge loop and
  emits the single matching id on a hit. Required for Llama-3 multilingual.
- **Cf range table + `isCf`** (`src/unicode_props.zig`): 21-range UCD
  16.0 Format-category table used by the BertNormalizer cleanup.

### Changed

- **`.ztm` Monster format bumped to v2** (`src/monster_io.zig`,
  `src/monster.zig`, `bench/convert_tm_to_ztm.py`): MAGIC
  `ZTM\x01` → `ZTM\x02`. Alias section appended past v1 EOF:
  `[u32 alias_count, {u32 id, u32 alt_byte_len, u8[alt_byte_len]}]`.
  Preserves TM-Go's twin entries (`train` + `\x7F train` sharing
  `alt_id`) — ~5,891 collisions in the 32K nocapcode vocab, ~1,932
  in the capcode vocab. v1 readers strict-error on v2 magic; back-compat
  preserved via inline v1-magic load test. `Monster.Builder.addAlias`,
  alias-aware `buildTrie` + `finalizeWithCapcode`, encoder uses
  `bare_flags`/`bare_nwords` for first/second positions when matched
  length equals alias byte length.
- **`auth_oidc.Validator.validateBearer`** (`src/auth_oidc.zig`): now
  dispatches on `alg in {HS256, RS256}`. RS256 verifies RSASSA-PKCS1-v1_5
  + SHA256 via `std.crypto.Certificate.rsa.PublicKey.fromBytes(e, n)`.
  Modulus lengths 128/256/384/512 bytes supported. New
  `Validator.initFromDiscovery(allocator, fetch_fn, http_ctx, issuer, audience)`
  GETs `<issuer>/.well-known/openid-configuration`, parses `jwks_uri`,
  GETs JWKS, builds validator. 10-min JWKS TTL with `refreshJwks()` on
  kid miss. Production wires `std.http.Client` via
  `httpFetchWithStdClient`; tests stub the fetch function pointer.
- **`build.zig` + `src/mbedtls.zig` (NEW) + `src/cli_serve.zig`**:
  added `-Dtls=mbedtls|none` build option (default `none` so the regular
  `zig build` works without mbedtls installed). When `mbedtls`,
  `linkSystemLibrary("mbedtls"/"mbedx509"/"mbedcrypto")`. New
  `src/mbedtls.zig` wraps `mbedtls_ssl_init/setup/handshake/read/write`
  against custom send/recv BIO callbacks using `std.c.send`/`std.c.recv`.
  `cli_serve.zig` does TLS handshake on each accepted socket when
  enabled. **HTTP-over-TLS framing is partial** — handshake succeeds,
  cert+key parse succeed, but the request reader/writer is not yet
  wrapped through `mbedtls.Conn.read/write`; currently emits 503 after
  successful handshake so curl gets a real response instead of hanging.
  Wave 5 finishes the stream-level integration.
- **`src/normalizer.zig` BertNormalizer**: `isBertControlCp` now
  delegates to `unicode_props.isCf` (full UCD 16.0 Cf set including
  bidi isolates U+2066-U+2069 + TAG block U+E0001/U+E0020-U+E007F),
  not the hand-rolled subset. `unicodeLowerMap` (Bert arm only — not
  the standalone Lowercase normalizer) inlines simple-case-fold for
  U+24B6-U+24CF (circled Latin caps, +0x1A) and U+2160-U+216F (Roman
  numerals, +0x10).
- **`src/hf_bridge.zig` `bpeFromHF`**: wires `hf.ignore_merges` into
  the new `Bpe.ignore_merges` field on both the SP-reshelled and
  legacy GPT-2/byte-level construction paths.

### Fixed

- **llama3 × multilingual 95→100%**: byte-level-encoded ` Федерации`
  was producing 3 ztok tokens vs 1 HF token because Llama-3's
  `tokenizer.json` sets `model.ignore_merges: true`, which HF's BPE
  encoder honors but ztok was ignoring. Whole-chunk lookup short-circuit
  now matches HF.
- **bert × unicode_stress 88.7→100%**: 96/113 diffs were unmodeled Cf
  bidi/TAG codepoints; 17/113 were missing simple-case-folds. Both fixed.

### Equivalence

| fixture            | 1.23 cells 100/100 | 1.24 cells 100/100 | delta                    |
|--------------------|:------------------:|:------------------:|--------------------------|
| LLaMA-2 SP-BPE     | 5/5                | 5/5                |                          |
| T5 SP-Unigram      | 3/5                | 3/5                | unicode_stress 96.5%, code 99.75% |
| Gemma SP-BPE       | 4/5                | 4/5                | code 99.92%              |
| GPT-2 HF BPE       | 5/5                | 5/5                |                          |
| Mistral-7B SP-BPE  | 5/5                | 5/5                |                          |
| Yi-6B SP-BPE       | 4/5                | 4/5                | code 99.55%              |
| DeepSeek-V2-Lite   | 5/5                | 5/5                |                          |
| llm-jp-3 HF Uni    | 4/5                | 4/5                | code 99.84%              |
| bert-base-uncased  | 4/5                | **5/5**            | **+1** (unicode_stress 88.7→100) |
| Falcon-7B HF BPE   | 3/5                | 3/5                | unicode_stress 92, code 99.99 |
| Qwen2-7B HF BPE    | 5/5                | 5/5                |                          |
| Llama-3-8B HF BPE  | 4/5                | **5/5**            | **+1** (multilingual 95→100 via `ignore_merges`) |
| Phi-3-mini HF BPE  | 5/5                | 5/5                |                          |
| **TOTAL**          | **56/65**          | **58/65 (+2)**     | **11 of 13 fixtures at 5/5** |

100-line gate: 13/13 SP+HF pairs remain at 100/100.

**TM Monster (TokenMonster-Go reference)**: nocapcode 78→85/100 (+7),
full-capcode 72→77/100 (+5) at 100 lines; 89.3%/89.6% at 1000 lines.
Format v2 alias section preserves 5,891 + 1,932 twin entries that v1
collapsed.

### Tests

- BPE `ignore_merges` regression tests (3): whole-chunk short-circuit
  via synthetic vocab, non-ASCII multi-codepoint ` Федерации`, offsets
  variant
- BertNormalizer Cf + case-fold fix tests (4): bidi-isolate strip,
  TAG-codepoint strip, circled-letter lowercase, Roman-numeral lowercase
- .ztm v2 + alias-trie tests (5): roundtrip with aliases, v1 back-compat,
  alias trie lookup, auto-detect v2 magic, alias byte-tally for begin_byte
- OIDC RS256 + discovery tests (5): valid RS256, bad signature,
  unknown kid, initFromDiscovery happy path, refreshJwks re-pulls
- MbedTLS tests (2): build-option boundary, cert+key parse + ssl_setup smoke
- Chat template inversion (22): roundtrip property for ChatML × 7 +
  Gemma × 5 + Mistral × 5 + Llama-2 × 5
- Vocab visualization (6): well-formedness, top-K, heatmap cell count,
  dendrogram presence, mini-corpus, added-token table
- Continued pretraining (6): base ids preserved, ids appended, fewer
  tokens after adapt, identity map, JSON report shape, format rejection
- Total: 872 → 924 (+52, 1 CI-only skip).

### Notes

- **TM `.ztm` v2 is NOT backward-compatible to v1 readers**. v1 readers
  strict-error on v2 magic. This is intentional — silently reading
  half the data would be worse. Regenerate `.ztm` files via
  `bench/convert_tm_to_ztm.py` to upgrade.
- **MbedTLS HTTP-over-TLS framing finish is Wave 5 work**. Handshake +
  cert/key parsing are production-ready; the stream-level integration
  (wrapping the request reader/writer through `mbedtls.Conn.read/write`)
  is the documented gap.
- **Tekken cross-bench can't yet run end-to-end on the real
  Mistral-Nemo `tekken.json`**: that file has no top-level `version`
  field (only `config.version="v3"`), which `src/tekken.zig`'s strict
  version check rejects. One-line fix candidate for Wave 5.

## [1.23.0] — 2026-05-19

**Headline: 12 features shipped in a single 12-agent parallel wave —
stress-sweep 51→56/65 cells at 100/100 (+5; +10 vs 1.21 baseline);
TM Monster ungreedy port lifts nocapcode 63→78/100 (+15) and
full-capcode 60→72/100 (+12); 5 serve hardening features (Prometheus,
WebSocket, OIDC HS256, JSON logging, TLS stub); gRPC-Web server; Rust
binding; Helm chart; nightly fuzz cron; persistent prefix cache; Tekken
multimodal.** Test count 808 → 872 (+64). 13/13 SP+HF pairs remain at
100/100 on the 100-line gate.

### Added

- **Mistral Tekken multimodal** (`src/tekken.zig`): `ImageConfig`,
  `AudioConfig`, `AudioSpectrogramConfig` parsing. `specialImageIds()`
  locates `[IMG]`/`[IMG_BREAK]`/`[IMG_END]`. `imageTokenGrid(dims)` +
  `placeholderImageTokens(allocator, dims)` build the
  `([IMG]*w + [IMG_BREAK]) * h` sequence per Pixtral spec with last
  `[IMG_BREAK]` rewritten to `[IMG_END]`. Pre-v11 `multimodal` key
  accepted alongside v11+ `image`.
- **gRPC-Web server** — new `ztok grpc-serve` subcommand on default
  port 7891. Routes `POST /ztok.Tokenizer/{Encode,Decode,Eval}` with
  `Content-Type: application/grpc-web+proto`. New `src/proto_min.zig`
  hand-rolled minimal protobuf codec (varint + length-delimited + 6
  message types + gRPC-Web frame helpers). New `src/cli_grpc.zig`
  HTTP/1.1 server (gRPC-Web carries framed proto bodies over HTTP/1.1,
  no HTTP/2 needed; Zig 0.16 stdlib has no HTTP/2 server). Errors
  travel in trailers with `grpc-status` per gRPC convention.
- **`ztok serve --auth-oidc-issuer URL --auth-oidc-audience NAME`** —
  OAuth2/OIDC validator (`src/auth_oidc.zig`). HS256 JWT signature
  verification via `std.crypto.timing_safe.eql`, `iss`/`aud`
  (string OR array)/`exp` checks. RS256 + discovery doc auto-fetch
  deferred (see Notes).
- **`ztok serve --metrics`** — Prometheus `/metrics` endpoint
  (`src/metrics.zig`). Counters: `ztok_requests_total{method,path,status}`,
  `ztok_request_bytes_in_total{path}`, `ztok_request_bytes_out_total{path}`,
  `ztok_encode_tokens_total`. Histogram:
  `ztok_request_duration_seconds{path}` (buckets 0.001, 0.01, 0.1, 1, 10).
  Gauge: `ztok_active_requests`. Bounded-cardinality label enums + atomic
  counters + Prometheus 0.0.4 text exposition.
- **`ztok serve /encode_ws`** — WebSocket streaming endpoint
  (`src/websocket.zig`). Hand-rolled RFC 6455: handshake via SHA-1 +
  magic UUID + base64 (`computeAcceptKey`), frame codec with 7/16/64-bit
  length + masking + opcode dispatch (`readClientFrame`/`writeServerFrame`).
  Server frames are unmasked per RFC 6455 §5.1; client frames must be
  masked. Encoded ids stream as binary frames.
- **`ztok serve --log-format json`** — structured stderr logs. RFC 3339
  timestamp, method, path, status, latency_ms, bytes_in/out, client_ip.
- **`ztok serve --tls-cert PATH --tls-key PATH`** — stubbed (errors at
  startup with `error.TLSServerNotAvailable`). Zig 0.16 stdlib
  `std/crypto/tls` ships only client; server-side is a Wave 4 candidate
  (MbedTLS link or stdlib support).
- **Rust binding** — `bindings/rust/`. MSRV 1.74, hand-rolled pkg-config
  probe in `build.rs` (zero build deps), default `pkg-config` feature +
  opt-in `link-static`. `Pipeline`/`BatchPool`/`StreamEncoder`/`Fingerprint`
  safe wrappers, `unsafe` confined to `sys.rs` + 1-line FFI blocks, public
  API zero-unsafe. `cargo clippy --all-targets` clean. 12 smoke tests + 1
  fuzz test (1000-iter PRNG round-trip, seed `0xFEEDB0B` matching the
  other bindings). All 22 exported C ABI symbols wrapped except
  `ztok_encode_batch` (the per-call variant; `BatchPool`-based path is
  wrapped and promoted as primary).
- **Helm chart** — `helm/ztok/`. Bitnami-style `_helpers.tpl`,
  ServiceAccount/Service/Deployment + optional HPA/PDB/ServiceMonitor/
  Ingress/PVC. `runAsNonRoot 65532` + `readOnlyRootFilesystem` + drop-ALL
  caps + RuntimeDefault seccomp. Auth token via `--auth-token-file`
  mounted from Secret (never `--auth-token TOKEN` which would leak via
  `ps`/audit logs). Vocab mount supports three modes (ConfigMap subPath /
  PVC / baked-in image). `helm lint` clean both defaults + all-features.
  helm test hook hits `/health`.
- **Nightly fuzz cron** — `.github/workflows/fuzz-nightly.yml`. Cron
  `37 4 * * *` UTC (off-spike) + `workflow_dispatch`. Seed job derives
  one 32-bit hex seed (`(run_id ^ epoch) & 0xffffffff`), exposes as
  output, 5 fuzz jobs (zig/python/node/ruby/go) consume via `FUZZ_SEED`
  env, run in parallel with 30-min budgets. On failure: `fuzz-issue.js`
  helper files idempotent GitHub Issues tagged `fuzz-failure` keyed on
  `[fuzz-failure] <job> (seed <hex>)` — appends comment on match, opens
  new on miss. Python/Node/Ruby fuzz harnesses extended with
  `FUZZ_SEED`/`ZTOK_FUZZ_ITERS` env overrides.
- **Persistent prefix cache** — `openPersistent(cfg)` + `closePersistent`
  + `compactNow` + `sync` + `persistentStats` on `PrefixCache`. Single
  append-only log file at `cfg.path` with 64-byte header (`ZTOKPCv1`
  magic + version + entry_count + total_bytes), variable records
  `[hash u64][key_len u32][ids_len u32][key bytes][ids u32 LE][CRC64-Ecma182 u64]`,
  all little-endian. Backend = `pread`/`pwrite` (not mmap — variable
  record sizes make mremap unprofitable; kernel page cache does the
  buffering). `flock(LOCK_EX)` per-process serialization. Configurable
  sync cadence (every N writes or M bytes). Compaction at stale-fraction
  threshold via tmp-file + atomic rename. CRC corruption skipped + file
  truncated to last-good offset on open.
- **8 new tokenizer fixtures** in `bench/fetch_vocabs.py --extended`
  (fetch-on-demand, not vendored): mistral_nemo_tekken (14.8 MB Tekken
  v3), mistral_small_v3 + codestral_v3 (SP .model), phi35_mini + phi4,
  qwen25 + qwen3, deepseek_v3. Plus `bench/vocabs/AVAILABILITY.md`
  documenting which models are public vs auth-gated (Llama-3.1/3.3/4
  still gated under meta-llama).

### Changed

- **`src/normalizer.zig` SP `isSpWhitespace`**: reverted 1.22's
  U+2007/U+2028 widening — that was an over-fix. SP-python (reference)
  only recognizes ASCII `' '` in collapse/escape stages; Unicode-space
  cps reach those stages as ASCII only when the model's
  `precompiled_charsmap` already rewrote them (e.g. T5's `nmt_nfkc`).
  Hardcoding 2007/2028 over-collapsed for 4 of 5 SP fixtures. Tests
  updated to match the corrected behavior.
- **`src/unigram.zig` Viterbi DP** promoted cumulative `best_score`
  and candidate score from f32 → f64 across `encodeChunk`,
  `encodeChunkWithOffsets`, `encodeChunkTrace`. HF tokenizers (Rust)
  uses f64; f32 rounding flipped tie-breaks on equal-score paths
  (specifically `4|44` vs `44|4`). Piece scores stay f32 on disk and
  widen at the add site. **Lifts llm-jp-3 × multilingual 99.99→100/100.**
- **`src/monster.zig` TM ungreedy port** — 6 faithful TM-Go behaviors
  ported (refs in source):
  - Phantom-second via path-(b) (TM `tokenmonster.go:1068`): when plain
    lookahead trie misses but path-(b) synthesis succeeds at a letter
    position, use synthesized id/length as scoring substitute. Emitted
    ids unchanged.
  - Bare-form flags + nWords precompute (TM `:3490-3593`):
    `bare_flags`/`bare_nwords` arrays for tokens starting with `\x7F ` /
    `D ` prefix; bare values reflect the substring after the 2-byte
    prefix. Read by phantom-second score block.
  - begin_byte double-tally for `\x7F X` / `X` collisions (TM `:3522`,
    `:3779`): when tallying a `marker+space+X` token, also tally `X`'s
    first byte. Restores letter classifications the vocab-convert step
    dropped.
  - score-b uses `plain_second` not phantom-second (TM `:1088`):
    separate `plain_second_id`/`plain_second_len` from phantom-substituted
    `second_id` for the score-b gate.
  - score-b `split_word` formula uses ungated variant (TM `:1102`,
    `:1152`, `:1204`): `int(first.flag & 1) * 103` ungated by
    `drop_begin_space_bonus`.
  - `computeNwordsTm` also strips `D ` (TM `nWords==0` mid-word gate
    regardless of capcode marker style).

### Fixed

- **`bench/bench_competitors.py:191` line-splitter bug**: was using
  `str.splitlines()` which splits on U+2028/U+2029/U+0085 in addition
  to `\n\r`. ztok's harness splits on `\n` only. Misalignment after the
  first U+2028 in `unicode_stress.txt` cascaded into 5 SP fixtures
  bottoming out at ~4.7% match rate even when ztok and SP-python
  actually agreed per-line. Switched to `data.split("\n")`. **Lifts
  LLaMA-2/Mistral-7B/Gemma/Yi-6B unicode_stress 4.7→100/100.**

### Equivalence

| fixture            | 1.22 cells 100/100 | 1.23 cells 100/100 | delta                                 |
|--------------------|:------------------:|:------------------:|---------------------------------------|
| LLaMA-2 SP-BPE     | 4/5                | **5/5**            | **+1** (unicode_stress 4.7→100)       |
| T5 SP-Unigram      | 3/5                | 3/5                | unicode_stress 6.8→96.5 (residual)    |
| Gemma SP-BPE       | 3/5                | **4/5**            | **+1** (unicode_stress 4.7→100)       |
| GPT-2 HF BPE       | 5/5                | 5/5                |                                       |
| Mistral-7B SP-BPE  | 4/5                | **5/5**            | **+1** (unicode_stress 4.7→100)       |
| Yi-6B SP-BPE       | 3/5                | **4/5**            | **+1** (unicode_stress 4.7→100)       |
| DeepSeek-V2-Lite   | 5/5                | 5/5                |                                       |
| llm-jp-3 HF Uni    | 3/5                | **4/5**            | **+1** (multilingual 99.99→100 via f64 Viterbi) |
| bert-base-uncased  | 4/5                | 4/5                | unicode_stress 88.70 (rare combining) |
| Falcon-7B HF BPE   | 3/5                | 3/5                | unicode_stress 92, code 99.99         |
| Qwen2-7B HF BPE    | 5/5                | 5/5                |                                       |
| Llama-3-8B HF BPE  | 4/5                | 4/5                | multilingual 95 (BPE merge bug, NOT regex) |
| Phi-3-mini HF BPE  | 5/5                | 5/5                |                                       |
| **TOTAL**          | **51/65**          | **56/65 (+5)**     |                                       |

100-line gate: 13/13 SP+HF pairs remain at 100/100.

**TM Monster (TokenMonster-Go reference)**: nocapcode 63→78/100 (+15),
full-capcode 60→72/100 (+12). 1000-line spot-check: 83.4% both. Residual
mostly traces to the vocab-convert script collapsing TM-Go's twin
trie entries (`train` + `\x7F train` sharing `alt_id`) into a single
key — needs an extended `.ztm` format for next wave.

### Tests

- TM ungreedy regression tests (2): vocab-collapse fix + score-b gate-b
- OIDC validator tests (5): HS256 happy/wrong-key/expired/RS256-rejected/aud-array
- Prometheus metrics tests (3): render shape, bucket cumulative, label mapping
- WebSocket frame tests (6): handshake, 7-bit/16-bit length, masked round-trip, unmasked-rejection, server-to-client
- gRPC tests (9): /health, Encode (text+raw), Decode round-trip, Eval (tokens/bytes/fertility + max_lines truncation), unknown route 404, malformed frame trailer, content-type
- proto_min tests (17): varint round-trip, tag encode/decode, 6 messages, frame helpers, trailer payload
- cli_serve serve tests (7): JSON envelope + escaping + text mode, /metrics endpoint + auth bypass + 404, OIDC missing bearer → 401, /encode_ws missing Upgrade → 400
- Tekken multimodal tests (6): image+audio config parse, legacy `multimodal` key, placeholderImageTokens grid math, v6 regression, missing required field
- Persistent prefix cache tests (6): empty→100→reopen, corruption truncation, cross-process flock, compaction shrink, flock contention, cold-entry reload
- Llama-3 regex regression tests (2): `\p{N}{1,3}` clamping + chain digit-group spans
- TOTAL: 808 → 872 (+64).

### Notes

- **RS256 OIDC + JWKS auto-fetch deferred**: RS256 plumbing through
  `std.crypto.Certificate.rsa` is tightly coupled to the cert parser in
  Zig 0.16 — deferred with `error.UnsupportedAlg` rejection so
  misconfigured deployments fail loudly. Auto-fetching
  `<issuer>/.well-known/openid-configuration` + JWKS via
  `std.http.Client` also deferred for complexity-budget reasons. HS256
  + static JWKS shipped. Wave 4 candidate.
- **TLS server termination deferred**: Zig 0.16 stdlib has client-only
  TLS. Stubbed cleanly. Wave 4 candidate (link MbedTLS as opt-in, or
  wait for stdlib server support).
- **Llama-3 multilingual 95% root-cause clarified**: Wave 3B confirmed
  the `\p{N}{1,3}` regex is loaded verbatim from `bench/vocabs/llama3.json`
  and applied correctly by `src/hf_regex.zig`. The 5% gap is a separate
  **non-ASCII multi-codepoint BPE merge bug** in `src/bpe.zig` — diverging
  tokens decode to identical Cyrillic/Arabic strings (e.g. ` Федерации`
  is 3 ztok tokens vs 1 HF token). Filed for next wave.

## [1.22.0] — 2026-05-19

**Headline: stress-sweep equivalence 46→51/65 cells at 100/100 (+5
cells); 8 user-facing features ship — Mistral Tekken loader,
tokenizer fingerprint, `encode --trace`, `eval --price`, `diff --report`,
per-binding fuzz harnesses (Python/Node/Ruby/Go), VERSION single-source-of-truth,
SP/Bert normalizer gaps closed.** 13/13 SP+HF cross-tokenizer pairs
remain at 100/100 on the 100-line gate. Test count 780 → 808 (+28).

### Added

- **Mistral Tekken loader** (`src/tekken.zig`): tiktoken-style BPE with
  base64-encoded `vocab[]`, first-class `special_tokens[]`, byte-fallback
  populated. Auto-detected via `looksLikeTekkenJson` and `tekken.json`
  filename override; new `.tekken` enum on auto-detect, new
  `ZTOK_FORMAT_TEKKEN` (=5) on the C ABI `ztok_format` enum.
- **Tokenizer fingerprint** (`src/fingerprint.zig`): SHA-256 over
  `(model_kind || vocab_size || encode(canonical_inputs))` — two pipelines
  with the same fingerprint produce bit-identical encodes for any input.
  New `ztok fingerprint VOCAB` subcommand prints `ztok:<64-hex>`; new C ABI
  `ztok_fingerprint(handle, out_32) -> ZTOK_OK` for binding consumers.
- **`ztok encode --trace`** (`src/trace.zig` + BPE/Unigram/Monster `*Trace`
  variants): per-step encoder decisions streamed to stderr while ids still
  go to stdout. Format: `bpe merge pos=<i> rank=<r> left="..." right="..."`,
  `unigram pos=<i> piece_id=<id> piece="..." score=<s>`, `monster pos=<i>
  piece_id=<id> len=<l> score=<s>`.
- **`ztok eval --price`** with flags `--price`, `--price-input`,
  `--price-output`, `--per 1k|1M`, `--prices-file prices.toml`,
  `--model-name`. Adds `cost_input` / `cost_output` / `cost_total` lines
  to both text and JSON eval output. Minimal hand-rolled TOML parser
  for the prices file.
- **`ztok diff --report`**: writes a Markdown comparison report with
  fertility, compression, divergence histogram (exact / small / medium /
  large via bounded Levenshtein on id-streams, capped at 50), and top
  differing tokens.
- **Per-binding fuzz harnesses** — all four bindings now ship a
  round-trip fuzz target against the byte_id vocab to validate FFI
  safety:
  - Python: `bindings/python/tests/test_fuzz.py` (pytest, 1000 iter,
    seeded `random.Random(0xFEEDB0B)`)
  - Node.js: `bindings/nodejs/test/fuzz.test.js` (node:test, 1000 iter,
    inlined xorshift32)
  - Ruby: `bindings/ruby/test/test_fuzz.rb` (Minitest, 1000 iter)
  - Go: `bindings/go/fuzz_test.go` (native `testing.F` — **7.77 M execs
    in 10 s, 0 crashes**)

### Changed

- **VERSION single source of truth**: `build.zig.zon` parsed once by
  `build.zig::projectVersion`, exposed as `@import("build_options").version`,
  re-exported as `pub const ztok.VERSION`. Both `c_api.zig` and `main.zig`
  read from this — the multi-file VERSION drift that recurred across
  the 1.18-1.21 multi-agent waves is structurally prevented.
- `src/normalizer.zig` SP normalizer (`spNormalize` + `spNormalizeWithOrigin`):
  stages 2 (`remove_extra_whitespaces`) and 4 (`escape_whitespaces`)
  pivoted from byte-level to codepoint-walking. New `isSpWhitespace`
  helper recognizes ASCII space + U+2007 FIGURE SPACE + U+2028 LINE
  SEPARATOR as whitespace, matching SentencePiece reference behavior.
- `bench/bench_cross.zig` hf-bpe and hf-unigram loader arms build an
  `added_tokens.Scanner` from `hf_json.HFTokenizer.added_tokens` and
  pass it to the pipeline. SP / Monster / WordPiece paths unchanged.
  Yi-6B, Qwen2-7B, Phi-3-mini now emit `<|im_start|>` / `<|im_end|>` /
  `<|endoftext|>` as single ids on code corpora instead of byte-splitting
  them into 10+ tokens each.

### Fixed

- **BertNormalizer `stripAccentsMap`** dropped all combining marks
  (Mn+Mc+Me); HF tokenizers only drops Mn (nonspacing marks). Mc and Me
  carry semantic information in Devanagari and other scripts. One-line
  fix: `isMark(cp)` → `isMn(cp)`. **Lifts bert × multilingual 8750→10000**
  and bert × unicode_stress 847→887.

### Equivalence

| fixture            | 1.21 stress cells 100/100 | 1.22 stress cells 100/100 |
|--------------------|:-------------------------:|:-------------------------:|
| LLaMA-2 SP-BPE     | 4/5                       | 4/5                       |
| T5 SP-Unigram      | 3/5                       | 3/5                       |
| Gemma SP-BPE       | 3/5                       | 3/5                       |
| GPT-2 HF BPE       | 5/5                       | 5/5                       |
| Mistral-7B SP-BPE  | 4/5                       | 4/5                       |
| Yi-6B SP-BPE       | 3/5                       | 3/5                       |
| DeepSeek-V2-Lite   | 4/5                       | **5/5**                   |
| llm-jp-3 HF Uni    | 3/5                       | 3/5                       |
| bert-base-uncased  | 3/5                       | **4/5**                   |
| Falcon-7B HF BPE   | 3/5                       | 3/5                       |
| Qwen2-7B HF BPE    | 3/5                       | **5/5**                   |
| Llama-3-8B HF BPE  | 4/5                       | 4/5                       |
| Phi-3-mini HF BPE  | 4/5                       | **5/5**                   |
| **TOTAL**          | **46/65**                 | **51/65 (+5)**            |

100-line gate: 13/13 SP+HF pairs remain at 100/100 bit-identical.

Residual 14/65 sub-100% cells: SP fixtures × unicode_stress at ~4.7%
share a single normalizer signature (25-line offset failure across
LLaMA-2/T5/Gemma/Mistral-7B/Yi-6B). Bert × unicode_stress 88.70%,
Falcon × unicode_stress 92.00%, code-corpus residuals at 99-99.99%
single-digit line counts. Tracked in `bench/RESULTS.md` under `## 1.22
10K-line stress equivalence`.

### Tests

- +5 fingerprint tests
- +3 trace tests
- +7 tekken loader tests
- +4 cost-estimator tests + 4 prices.toml parser tests
- +3 diff-report tests
- +3 SP whitespace tests
- Per-binding fuzz tests (1 file per binding)
- Total: 780 → 808 (+28).

### Notes

CHANGELOG entries for 1.17 through 1.21 were never written into this
file during the per-version waves; the running narrative lives in
`README.md` (which tracks every version's headline) and
`bench/RESULTS.md`. This 1.22 entry resumes the per-release record
without back-filling.

## [1.16.0] — 2026-05-18

**Headline: cl100k single-thread back to 26.1 MB/s (+24% vs 1.15);
batch ×48 + pinning lands 377 MB/s (+7% vs 1.15); LLaMA-2 SP-BPE
equivalence locked at 100/100 after IEEE-754 signed-zero fix.**

### Added

- `Bpe.LoadOptions { hot_table: bool }` + four new constructor
  variants (`loadTiktokenFileWithOptions`,
  `loadTiktokenBytesWithOptions`, `bpeFromHFWithOptions`,
  `bpeFromHFBytesWithOptions`). Bare constructors keep their old
  signatures.
- `MarkerStyle = { .ztok, .tm_printable }` on capcode so TM
  full-capcode vocabs encode byte-identical with TM-Go's printable
  `C`/`W`/`D` markers.
- `piece_ranks` SP-aware BPE merge path — codepoint-level BPE using
  `|SP score|` as merge rank.
- `encode_mode = { bpe_merge, longest_match }` knob on BPE.
- `--hot-table` flag on `bench_ztok` (defaults to off, mirroring the
  new bare-loader policy).

### Changed

- Bare `Bpe.loadTiktokenFile` / `loadTiktokenBytes` now default to
  `hot_table = off` — the single-shot constructors bias toward
  single-thread workloads where the 1.15 hot table is net negative.
- `hf_bridge.bpeFromHF` keeps `hot_table = on` — library/serving
  deployments are batch-encode dominated and want the +43% pinned
  win.
- `capcode.encodeStyled` / `decodeStyled` / `NoCapcode.encode` /
  `NoCapcode.decode` use `initCapacity` (exact pre-size) instead of
  `.empty + ensureTotalCapacity` (power-of-two growth).
- `vocab_extend`, `vocab_prune`, and `doctor` still populate the hot
  table directly so their encode profile matches the production
  pipeline.

### Fixed

- IEEE-754 signed-zero on SP rank conversion: piece_ranks treated
  `-0.0` and `+0.0` as distinct ranks for cl100k edge-cases. Closes
  the last LLaMA-2 SP-BPE miscompare (Gemma SP-BPE simultaneously
  jumped to 100/100).
- TM full-capcode `lilbuf` gate now relaxes for the path-b synthetic
  boundary; precomputed alts populate correctly for capcode-marker
  runs.

### Performance

| config                                  | 1.15  | 1.16  | delta |
|-----------------------------------------|------:|------:|------:|
| cl100k single-thread (hot off)          | 17.9  | 26.1  | +46%  |
| cl100k batch ×48 + `--pin-physical`     | 352.9 | 377.0 |  +7%  |
| TM Monster nocapcode 10 MB single-thread | 10.1  | 10.7  |  +6%  |

### Equivalence

- LLaMA-2 SP-BPE: 31/100 → **100/100** vs sp-python.
- Gemma SP-BPE: → **100/100** vs sp-python.
- TokenMonster-Go nocapcode: → **63/100** (was 34/100).
- TokenMonster-Go full-capcode: → **34/100** (was 0/100).
- HF Unigram llm-jp-3-1.8b stays **100/100** bit-identical to
  sp-python (T5 Unigram likewise locked at 100/100).

### Tests

- `+5` BPE hot-table opt-out tests (`loadTiktokenBytes` defaults to
  off, `WithOptions(.{ .hot_table = true })` populates,
  encode-bit-identical with vs without, 1 MB encodeStyled
  determinism + envelope check, 1-byte input pre-size hint bounded).
- Test count: 625 → **630**.

## [1.15.0] — 2026-05-17

**Headline: persistent worker pool + CPU affinity push batch ×48
to 246 MB/s unpinned / 353 MB/s pinned (1.10 baseline: 205 MB/s).
Two-level BPE merge-rank cache (64 KB direct-mapped front, +43%
when pinned). Char-class fold collapses 4-5 per-codepoint range
lookups into one (~8 KB merged table). Targets cleared on every
front.**

### Added

- Persistent `BatchPool` workers — `std.Thread.spawn`/`join` per
  `runBatch` replaced by a futex-driven generation/wakeup loop. No
  per-batch thread setup; the small-batch / N=1 fast path is
  unchanged.
- `BatchPool.Options{ .pin_to_physical_cores }` +
  `BatchPool.initWithOptions`. Linux topology walks
  `/sys/devices/system/cpu/cpuN/topology/thread_siblings_list` to
  build a pin order with physical primaries first, SMT siblings
  second. `pin_diagnostic` reports realized pin state. Fallback paths
  for masked `/sys`, locked-down containers, non-Linux hosts.
- `--pin-physical` flag on `bench_ztok`.
- `Bpe` 64 KB hot table (4096 × 16 B `extern struct HotEntry`,
  `hotHash` packs 7 bytes + length, 1 multiply + xor fold). Slot 0
  is empty sentinel. Long keys (`> 7 B`) live exclusively on the
  cold map.
- `unicode_props.CharClass` packed-byte (letter/number/mark/space)
  + `classifyCp(cp)` returning all four bits in one binary search.
  Merged comptime range table coalesces ~2,649 per-class ranges into
  **1,152** disjoint slabs (~8 KB). ASCII goes through a 128-byte
  precomputed lookup.

### Changed

- `capcode.encodeStyled` / `NoCapcode.encode` call `classifyCp` once
  per codepoint and branch on the bitfield (was 4-5 separate range
  lookups).
- All `Bpe` constructors populate the hot table after building
  `by_bytes`; SP-derived `bpeFromSP` skips (its encode path is
  `encodeSpBpe`).
- `bpe_heap.encodeWithFallback` plumbs a `hot_table` parameter; both
  `pairRank` and the emit loop go through the same hot-first
  dispatch.

### Performance

| config                                | 1.14 baseline | 1.15  | delta |
|---------------------------------------|--------------:|------:|------:|
| cl100k batch ×48 (no pin)             |         204.8 | 245.7 | +20%  |
| cl100k batch ×48 + `--pin-physical`   |         247.1 | 352.9 | +43%  |
| cl100k single-thread (hot on)         |          21.3 |  17.9 | -16%  |
| TM nocapcode 10 MB single-thread      |           9.9 |  10.2 |  +3%  |
| TM capcode 10 MB single-thread        |          13.5 |  13.5 |   0%  |

(Single-thread regression is the cost of an unused hot table on an
L2-warm workload — addressed in 1.16 by flipping the bare-loader
default to off.)

### Variance

- At N=24 pinned: 204.0 / 208.1 / 210.1 MB/s (±1.5%) across 3 runs
  vs unpinned 167.3 / 170.5 / 184.8 (±5%).

### Equivalence

- TokenMonster-Go nocapcode: 63/100 (lilbuf gate relax + precomputed
  alts).
- TokenMonster-Go full-capcode: 34/100 (lilbuf path-b landed).

### Tests

- `+6` thread-pool tests (init/deinit-without-runBatch, 100 sequential
  batches, concurrency check, pin diagnostic on Linux,
  `discoverPinOrder`, `parseFirstCpu`).
- `+7` unicode-props/capcode tests (ASCII fast path equivalence,
  1000-cp deterministic sweep, merged-table size guard,
  `@bitCast` round-trip, two golden-hash regression guards).
- `+6` BPE hot-table tests.
- Test count: 549 → **625**.

## [1.14.0] — 2026-05-16

**Headline: TM Monster encode regression fix — 8.4 → 11.5 MB/s
nocapcode, 10.2 → 13.7 MB/s capcode. RobertaProcessing + ByteLevel
post-processors land. `prefix_cache` watermark-triggered arena
compaction (10× smaller footprint under churn).**

### Added

- `RobertaProcessing` post-processor (`[CLS] A [SEP]` /
  `[CLS] A [SEP] [SEP] B [SEP]`).
- `ByteLevel` post-processor (records flags, identity in id-space).
- `Sequence` post-processor (delegates to first inner framer).
- `prefix_cache` watermark-triggered arena compaction.
- `vocab_extend` weighted-similar K-NN init plan + q-gram pre-filter
  index (5× speedup at V=100K).

### Changed

- `Normalizer.normalize` is now a standalone origin-free path —
  `.identity` / `.byte_level` / NFC/NFD/NFKC/NFKD /
  `.sp_precompiled` / `.capcode` / `.nocapcode` dispatch directly
  without allocating a per-byte origin map. `normalizeWithOrigin`
  unchanged for span-aware callers.
- `unicode_norm.normalize` streams NF-stable runs as memcpy; only
  round-trips through the codepoint buffer on non-zero combining
  class, known decomposition, or Hangul jamo (~80% of typical
  corpus codepoints are stable).
- `capcode.NoCapcode.encode` and `capcode.encodeStyled` cache
  `rlast` classifications across iterations + take an ASCII fast
  path; ~50 Unicode table ops/cp drop to ~5 on the common path.

### Fixed

- `Pipeline.encodeText` callers no longer pay the ~40 MB origin-map
  alloc/write/free that 1.13 inadvertently introduced.

### Performance

| build                                | MB/s   | ids/run    |
|--------------------------------------|-------:|-----------:|
| pre-fix TM nocapcode                 |    8.4 |  8,234,263 |
| **post-fix TM nocapcode**            | **11.5** |  8,234,263 |
| pre-fix TM capcode                   |   10.2 |  8,733,279 |
| **post-fix TM capcode**              | **13.7** |  8,733,279 |
| post-fix batch ×8 TM nocapcode       |   74.8 |          — |

ids/run is bit-identical pre/post via `normalize ==
normalizeWithOrigin.bytes` regression test.

### Tests

- `normalize == normalizeWithOrigin.bytes` across 15 inputs × 15
  normalizer configs.
- `normalize does not leak: round-trip free with testing.allocator`.
- `TM-style capcode normalize of 1 MB synthetic input completes
  promptly` (regression guard against allocator regression).
- Test count: ~510 → **549**.

## [1.13.0] — 2026-05-15

**Headline: LLaMA-2 SP-BPE bit-identical to sp-python (31/100 →
100/100). T5 Unigram 100/100. `ztok validate` lands as a CLI.**

### Added

- `ztok validate` CLI subcommand — runs the 7 `doctor` lint checks
  on any loaded vocab. `--checks`, `--fixtures`, `--format
  text|json` flags. `cli_validate.zig` library helper.
- `ztok roundtrip` CLI subcommand — per-line encode+decode against
  a corpus, reports first divergent offset + token id. `--summary`
  for aggregate.
- `cli_validate.LoadedModel` union + `runValidateAny` — routes to
  the appropriate per-kind doctor (BPE / Unigram / WordPiece /
  Monster) based on auto-detected format.
- `sp_bridge` byte_fallback `<0xNN>` token population when
  `trainer_spec.byte_fallback` is set.
- `subword regularization` knob on Unigram train + encode time
  (noise sampling).
- `eval` per-script fertility breakdown + Top-K/Bottom-K most-
  frequent ids + KV-cache byte estimate.

### Changed

- `sp_precompiled` normalizer (NFKC + add_dummy_prefix +
  escape_whitespaces + remove_extra_whitespaces) — was a stub in
  1.12.

### Equivalence

- LLaMA-2 32K SP-BPE vs sp-python: 31/100 → **100/100**.
- T5 32K Unigram vs sp-python: → **100/100**.
- HF Unigram llm-jp-3-1.8b: → **100/100**.

### Tests

- `+12` validate / roundtrip / cli_validate coverage including the
  multi-format dispatch + HF Unigram autodetect path.

## [1.12.0] — 2026-05-14

**Headline: per-token byte spans through any normalizer
(`encodeWithOffsets`). Capcode + nocapcode TM-compat normalizers.
RAG chunking land with sentence boundary detection.**

### Added

- `Pipeline.encodeWithOffsets` — returns per-token byte spans in
  **original-input** coordinates. Normalizers emit a
  `normalized→original` byte map that the pipeline uses to translate
  spans back through NFC/NFD/NFKC/NFKD/byte_level.
- `Normalizer.normalizeWithOrigin` — every non-identity normalizer
  exposes byte-accurate offset round-tripping.
- `capcode` and `nocapcode` TM-compat normalizers (wraps capcode
  with `nfd` + `tm_compat_space` flags for TokenMonster byte-parity).
- `ztok.chunk.chunkText` — token-cap windows with overlap stride;
  boundary modes `{ token, codepoint, word, sentence, paragraph }`;
  per-chunk byte ranges back to original input.
- `.sentence` segmenter handles ellipsis, decimals, URLs, files,
  quoted dialog, 8 sentence-terminator scripts (Latin, CJK, Arabic,
  Armenian, Tibetan, Ethiopic, Devanagari, Mongolian, Burmese,
  Khmer), multi-language abbreviations (`Locale = { en, fr, de, es,
  it, hi, vi, pl, multilang_union }`).
- `.codepoint` snapping backs off UTF-8 continuation bytes.
- `ztok chunk` CLI subcommand: `--max-tokens`, `--overlap`,
  `--boundary`, `--format jsonl|text`.

### Changed

- `.tiktoken` / HF / SP loaders now report `origin_capable` so
  pipelines can opt into offset-tracking without a separate flag.

## [1.11.0] — 2026-05-13

**Headline: BPE → 86 → 148 MB/s at batch ×8; cross-tokenizer
benchmarks land (ztok beats TM-Go 2.0×, beats SentencePiece-Python
3.0×). C ABI hot path closes the C-vs-Zig gap to 0.99×-1.03×.**

### Added

- AVX-512 wide `scanMin` path (`@Vector(32, u32)`) — comptime-gated
  on `std.Target.x86.featureSetHas(..., .avx512f)`. AVX2 / aarch64 /
  WASM stay on the 16-lane narrow path.
- `bench_cross.zig` — ztok loading a TM `.ztm` or SP `.model` and
  encoding the same corpus the reference Python harness encodes
  through the native tool.
- `bench_c_api.c` + `bench_zig_path.zig` — C-vs-Zig parity harness
  for `single_small` / `single_large` / `batch_pooled` scenarios.
- `--workers N` + `--per-worker` flags on `bench_ztok` (scaling
  sweep + per-worker time histogram).
- `c_api.runBatch` — workers allocate caller-visible C buffers
  directly, no intermediate Zig alloc + post-encode memcpy. The
  added-tokens path keeps the `runBatchLegacy` alloc-then-copy
  fallback.

### Changed

- `c_api` batch encode dispatches its own worker context through
  `BatchPool.runBatch` instead of going through
  `Pipeline.encodeBatch`.

### Performance

| scenario       | C before | Zig before | ratio | C after | Zig after | ratio |
|----------------|---------:|-----------:|------:|--------:|----------:|------:|
| `single_small` |  1.47 us |   1.50 us  |  0.98 | 1.44 us |  1.46 us  |  0.99 |
| `single_large` | 800.3 ms |  801.6 ms  |  1.00 | 796.8 ms|  802.9 ms |  0.99 |
| `batch_pooled` |  39.9 ms |   34.1 ms  |  1.17 | 33.2 ms |  32.2 ms  |  1.03 |

### Cross-tokenizer

| pair                              | single | batch ×8 | batch ×48 |
|-----------------------------------|-------:|---------:|----------:|
| ztok vs TM-Go (TM 32K)            |  2.00× |   2.13×  |    2.01×  |
| ztok vs SP Python (LLaMA-2)       |  3.00× |   3.68×  |    4.72×  |

### Tests

- `+200-trial random-input scanMin equivalence sweep` up to 8192
  elements between narrow and wide paths.

## [1.10.0] — 2026-05-12

**Headline: TokenMonster ungreedy encoder lands (25.2 MB/s single-
thread on TM 32K). Marginal-value scoring v3 (mask-based) is up to
317× faster than v2 (rebuild) on V=1000. SoA + SIMD + 4-ary heap
make BPE 1.6× faster than tiktoken at batch ×8.**

### Added

- `Monster` model — TokenMonster 6-branch ungreedy encoder + nWords
  scoring. Loads via `.ztm` (ztok native binary) and `.vocab`
  (TM-Go wire format → converter).
- `ztok .ztm` Monster vocab binary format (header carries capcode +
  normalizer flags from TM-Go; `normalizerForVocab` infers from
  name).
- `train_monster` distillation with **marginal-value scoring v3** —
  `monster.mask[P]=1` then re-encode, up to 317× faster than v2
  rebuild at V=1000. Per-piece alt counts bit-identical v2-vs-v3.
- `monster_io.writeFile` / `monster_io.readFile` / `readFileMeta`.
- `bench/convert_tm_to_ztm.py` — TM `.vocab` → `.ztm` converter.
- `bench/equivalence_check.py` — cross-tokenizer correctness signal.

### Performance

| Mode                    | tiktoken 0.12 | ztok 1.5.0 | ztok 1.10 | ratio |
|-------------------------|--------------:|-----------:|----------:|------:|
| cl100k single-thread    |          12.3 |       13.2 |      12.6 | 1.02× |
| cl100k batch ×8         |          52.9 |       86.3 |      88.9 | 1.68× |
| cl100k batch ×48        |             — |          — |     204.8 |     — |
| TM 32K single (ztok)    |             — |          — |    **25.2** | 2.00× vs TM-Go |
| LLaMA-2 SP-BPE single   |             — |          — |       9.6 | 3.00× vs SP-py |

### Equivalence

- ztok-TM vs TM-Go (TM 32K): 34/100 (residual is TM's NFD pre-norm
  + deleteToken handling not yet wired).
- ztok-SP-BPE vs sp-python (LLaMA-2): 31/100 (residual is SP
  `dummy_prefix` + `▁` substitution + `<0xNN>` byte fallback).

## [1.9.0] — 2026-05-11

**Headline: TokenMonster training distillation lands (v2 rebuild
path). Native LR-criterion WordPiece training. `ztok train --kind`
dispatches across all four model kinds.**

### Added

- `train_monster.zig` — TokenMonster distillation (rebuild-based
  marginal-value scoring v2).
- `train_wordpiece.zig` — native LR-criterion WordPiece trainer.
- `ztok train --kind bpe|unigram|wordpiece|monster` — each writes
  its native on-disk format (`.tiktoken`, SentencePiece `.model`,
  HF `tokenizer.json`, ztok `.ztm`).
- Unigram-only training flags: `--em-iterations`, `--shrink-rate`.
- Monster-only: `--branches` (reserved).

### Changed

- `train_unigram` exposes subword-regularization knobs at train time.

## [1.8.0] — 2026-05-10

**Headline: Chat templates (Jinja subset) land — ChatML, Mistral,
full Llama-2, simplified Gemma. Added-token scanner resolves
specials before pre-tokenization.**

### Added

- `chat_template.zig` — Jinja-subset renderer
  (`apply_chat_template`).
- ChatML / Mistral / Llama-2 (`%` modulo + `is defined` + `is none`)
  / simplified Gemma template support.
- `added_tokens.Scanner` — special-token trie, longest-match,
  lstrip/rstrip honor full Unicode whitespace.
- `tokenizer_config.zig` — sibling `tokenizer_config.json`
  bos/eos/pad/unk loading.

## [1.7.0] — 2026-05-09

**Headline: HF tokenizer.json writer + SentencePiece `.model`
protobuf writer. `BertProcessing` and `TemplateProcessing`
post-processors round-trip clean through `hf_json` ↔ `hf_writer`.**

### Added

- `hf_writer.zig` — writes BPE / WordPiece / Unigram `tokenizer.json`
  with full post_processor support.
- `sp_writer.zig` — minimal SentencePiece `.model` protobuf writer.
- `post_processor.zig` — `BertProcessing` and `TemplateProcessing`
  framers.
- `proto.zig` — minimal protobuf reader, vendored.

## [1.6.0] — 2026-05-08

**Headline: Unigram (Viterbi) and WordPiece (zero-alloc longest-
match) models land. SentencePiece `.model` and HF `tokenizer.json`
loaders. Format auto-detect.**

### Added

- `Unigram` model — Viterbi over a flat trie + sampling for
  subword regularization.
- `WordPiece` model — longest-match, zero-alloc encode.
- `hf_json.zig` + `hf_bridge.zig` — `tokenizer.json` reader, BPE +
  WordPiece + Unigram.
- `sp_model.zig` + `sp_bridge.zig` — SentencePiece `.model` loader,
  BPE + Unigram.
- `auto_detect.zig` — byte/extension format sniffer (`.tiktoken` /
  HF JSON / SP `.model` / ztok `.ztm`).

### Changed

- `Pipeline.model` is now a tagged union of `{byte_id, bpe, unigram,
  wordpiece, monster}` — was BPE-only.

## [1.5.0] — 2026-05-07

**Headline: SoA + SIMD + 4-ary heap re-enabled. Single-thread
13.2 MB/s, batch ×8 86.3 MB/s — beats tiktoken 1.07× / 1.63×.
Bit-identical token ids to tiktoken on the full 10 MB corpus.**

### Added

- 4-ary heap encoder for BPE chunks > 64 bytes (`bpe_heap.zig`).
- `simd_min.zig` — `@Vector(16, u32)` min-index reduction; lowers
  to `vpminud` on AVX2/AVX-512 and `uminv` on NEON.

### Fixed

- 4-ary heap tiebreaking bug: rank-only ordering produced
  nondeterministic merges on equal-rank inputs (long runs of
  `'aaa…'`). Re-enabled the heap path after fix.

### Performance

| Mode          | ztok 1.3 | ztok 1.4 | ztok 1.5 | tiktoken 0.12 |
|---------------|---------:|---------:|---------:|--------------:|
| Single-thread |      4.6 |     12.5 |   **13.2** |          12.3 |
| Batch ×8      |     26.4 |     79.6 |   **86.3** |          52.9 |

- Bit-identical to tiktoken on the full 10 MB corpus (both emit
  3,396,054 tokens).
- Batch mode differs by 1 token out of 3.4M (chunk-boundary noise).

## [1.4.0] — 2026-05-06

**Headline: SoA + precomputed-and-maintained pair ranks lift BPE
single-thread from 4.6 → 12.5 MB/s — 2.7× speedup, 18× fewer
StringHashMap lookups per chunk (O(N²) → O(N)).**

### Changed

- `Bpe` switched to SoA layout (`bytes` + `offsets` + optional
  `ranks`); pair ranks are precomputed and maintained across the
  encode loop instead of recomputed per merge.

### Performance

- Single-thread cl100k: 4.6 → **12.5 MB/s** (2.7×).
- Bit-identical output: 3,396,054 ids on the 10 MB corpus, same as
  tiktoken's reference.

## [1.3.0] — 2026-05-05

**Headline: First apples-to-apples benchmark vs tiktoken. ztok at
4.6 MB/s single-thread / 26.4 MB/s batch ×8 — bit-identical token
ids, 2.6× slower per-call. Diagnoses the gap as the linear-scan
merge loop (next: SIMD min + heap).**

### Added

- `bench/bench_ztok.zig` — cl100k_base benchmark harness.
- `bench/bench_competitors.py` — tiktoken / HF tokenizers
  comparison.

### Fixed

- cl100k pattern-6 (`\s+(?!\S)`) lookahead bug: consecutive
  whitespace runs were greedily consumed instead of backing off
  one codepoint to satisfy the negative lookahead.

### Performance

- Single-thread: 4.6 MB/s (vs tiktoken 12.1 MB/s, 0.38×).
- Batch ×8 (48-worker pool): 26.4 MB/s (vs tiktoken 51.5 MB/s,
  0.51×).
- **Bit-identical ids** to tiktoken on the full 10 MB corpus:
  both emit 3,396,054 tokens.

## [1.2.0] — 2026-05-04

**Headline: NFC / NFD / NFKC / NFKD normalizers land with full
UCD 16.0 conformance (562 KB of baked decomposition + recomposition
+ combining-class tables). byte_level normalizer.**

### Added

- `unicode_norm.zig` — NFC / NFD / NFKC / NFKD with full UCD 16.0
  conformance.
- `byte_level` normalizer (GPT-2 byte_to_unicode 256-entry table,
  comptime-built).
- `Normalizer` tagged union with `.identity`, `.nfc`, `.nfd`,
  `.nfkc`, `.nfkd`, `.byte_level`.

## [1.1.0] — 2026-05-03

**Headline: cl100k_base pre-tokenizer lands with real Unicode
support — 2,633 sorted ranges for `\p{L}\p{N}\p{M}\s` from UCD 16.0.
HF GPT-2 ByteLevel pre-tokenizer. Multithreaded BPE training.**

### Added

- `cl100k.zig` — hand-coded cl100k_base scanner (real Unicode
  property tables).
- `unicode_props.zig` — 2,633 sorted ranges for `\p{L}`, `\p{N}`,
  `\p{M}`, `\s` (UCD 16.0).
- `hf_bytelevel_pretok.zig` — GPT-2 ByteLevel split-and-map in one
  pass.
- `train_bpe.zig` — multithreaded incremental BPE training.
- `BatchPool` — N per-worker `ArenaAllocator`s, atomic-cursor
  work-stealing.

### Changed

- `PreTokenizer` is now a tagged union of `{identity, cl100k,
  hf_byte_level}`.

## [1.0.0] — 2026-05-02

**Headline: initial release. Composable `Pipeline` (Normalizer →
PreTokenizer → Model → Decoder), byte-level BPE, Unigram (stub),
WordPiece (stub), `.tiktoken` loader, CLI for encode/decode/info.**

### Added

- `Pipeline` value composed of four tagged-union stages — no
  vtables, no per-token heap allocations on the hot path.
- `Bpe` byte-level BPE model + `.tiktoken` loader.
- `Vocab` SoA table (`bytes` + `offsets`).
- `byte_id` model stub (every byte → its own id, useful as a
  baseline / negative control).
- `Normalizer` `.identity`; `PreTokenizer` `.identity`; `Decoder`
  `.concat`.
- `Pipeline.encode` / `Pipeline.decode` single-threaded entry
  points.
- `Pipeline.encodeBatch` parallel encode over a `BatchPool`.
- `ztok encode` / `ztok decode` / `ztok info` CLI subcommands.
- `c_api.zig` + `include/ztok.h` — C ABI exports
  (`libztok.{a,so}`).
- Static + shared library build via `zig build`, header install
  into `zig-out/include/`.

### Tests

- Initial suite (~200 tests) covering pipeline composition, BPE
  loader, encode/decode round-trips, `BatchPool` basics.

[1.16.0]: https://github.com/sirus20x6/ztok/releases/tag/v1.16.0
[1.15.0]: https://github.com/sirus20x6/ztok/releases/tag/v1.15.0
[1.14.0]: https://github.com/sirus20x6/ztok/releases/tag/v1.14.0
[1.13.0]: https://github.com/sirus20x6/ztok/releases/tag/v1.13.0
[1.12.0]: https://github.com/sirus20x6/ztok/releases/tag/v1.12.0
[1.11.0]: https://github.com/sirus20x6/ztok/releases/tag/v1.11.0
[1.10.0]: https://github.com/sirus20x6/ztok/releases/tag/v1.10.0
[1.9.0]:  https://github.com/sirus20x6/ztok/releases/tag/v1.9.0
[1.8.0]:  https://github.com/sirus20x6/ztok/releases/tag/v1.8.0
[1.7.0]:  https://github.com/sirus20x6/ztok/releases/tag/v1.7.0
[1.6.0]:  https://github.com/sirus20x6/ztok/releases/tag/v1.6.0
[1.5.0]:  https://github.com/sirus20x6/ztok/releases/tag/v1.5.0
[1.4.0]:  https://github.com/sirus20x6/ztok/releases/tag/v1.4.0
[1.3.0]:  https://github.com/sirus20x6/ztok/releases/tag/v1.3.0
[1.2.0]:  https://github.com/sirus20x6/ztok/releases/tag/v1.2.0
[1.1.0]:  https://github.com/sirus20x6/ztok/releases/tag/v1.1.0
[1.0.0]:  https://github.com/sirus20x6/ztok/releases/tag/v1.0.0
