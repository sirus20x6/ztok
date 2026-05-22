# Tokenizer Feature Comparison

How ztok stacks up against the four libraries it learned from. Reference
repos are shallow-cloned under `refs/`.

| Feature | tiktoken (OpenAI) | tokenizers (HF) | SentencePiece (Google) | TokenMonster (Forsythe) | **ztok** |
|---|---|---|---|---|---|
| **Algorithms** | byte-level BPE | BPE, WordPiece, Unigram | BPE, Unigram | Ungreedy (6-branch lookahead) | **all of these** — byte-level BPE (3 encode modes), Unigram, WordPiece, TokenMonster ungreedy |
| **Training** | educational only | full | full (EM for Unigram) | yes (distillation) | BPE, Unigram EM, WordPiece, Monster distillation, **PathPiece (CTC-minimizing)** |
| **Pre-tokenization** | regex splitter | whitespace / regex / BERT / metaspace / byte-level | none (data-driven) | none (raw UTF-8) | identity, cl100k regex (UCD 16.0), HF byte-level |
| **Byte fallback** | yes | optional | yes | yes (UTF-8/16 aware) | yes |
| **Special tokens** | dict | `AddedToken` | yes | yes + single-byte tokens | `added_tokens.Scanner` (Unicode lstrip/rstrip) |
| **Normalization** | none | NFC/NFD/NFKC/strip | NFKC mandatory | NFD + Capcode | NFC/NFD/NFKC/NFKD, byte_level, capcode, SP charsmap, full HF chain |
| **Decoder roundtrip** | yes | yes (pluggable) | yes (whitespace restore) | yes (Capcode-aware) | yes (concat / wordpiece / byte_level / capcode) |
| **Multithreaded encode** | opt-in (Python pool) | conditional Rayon | single-threaded | single-threaded | **default** (per-thread arenas, persistent batch pools) |
| **SIMD** | none | none | none | branchless `pansearch` | **portable `@Vector`** — byte-class scanners + BPE min-scan, target-native width (AVX2 32-lane / AVX-512 64 / SSE2·NEON·wasm128 16) |
| **File format** | `.tiktoken` binary | HF `.json` | `.model` protobuf | `.yaml` | reads `.tiktoken` / HF `.json` / SP `.model` / `.ztm` / Tekken `.json`; writes `.tiktoken` / `.json` / `.model` / `.ztm` |
| **Bindings** | Py | Py / Node / Ruby | Py / C++ | Go / Py / JS | **8** — Py / Node / Ruby / Go / Rust / .NET / Java / Swift |
| **License** | MIT | Apache-2 | Apache-2 | MIT | **AGPL-3.0-only** |
| **Core LOC** | ~2.8K | ~23.8K | ~169K | ~4K | ~88K `src/` (full toolkit, not just inference) |

## Standout ideas worth absorbing (all now in ztok)
- **tiktoken**: thread-local regex caches, binary-heap merge inner loop, trivial `.tiktoken` rank→bytes file format. Best starting point for byte-level BPE inference.
- **HF**: modular `Normalizer → PreTok → Model → PostProc` pipeline with serde JSON. Best architectural template — ztok's `Pipeline` mirrors it (tagged unions, no vtables).
- **SentencePiece**: EM-trained Unigram, language-agnostic, subword regularization for training-time noise.
- **TokenMonster**: ungreedy multi-branch scoring (~37% fewer tokens at same vocab size), distillation-based vocab selection, Capcode uppercase encoding.

## What ztok adds beyond bit-identical parity

ztok is bit-identical with tiktoken / HF / SentencePiece (13/13 cross-tokenizer pairs at 100/100 on the 100-line gate) and close to TokenMonster-Go. These capabilities go past matching the reference libraries — none of the four offers them:

- **Optimal (minimum-token) encoding** — `EncodeMode.optimal` runs a DP over the vocab lattice to emit the provably fewest tokens for a given vocab (`encode --optimal`). A guarantee greedy BPE/WordPiece can't make.
- **PathPiece vocabulary learner** — top-down, CTC-minimizing vocab construction (`train --kind pathpiece`), the training-time complement to optimal segmentation. (See Schmidt et al., EMNLP 2024.)
- **Overlay channels** — `encodeWithOverlays` / `ztok_encode_with_overlays` return per-token annotation channels (byte span, boundary bitset, provenance, plus pluggable domain channels) aligned 1:1 with the id stream, without changing tokenization.
- **Grammar-constrained tokenization** — a token-prefix automaton (`constrained.zig`) yields the set of allowed next tokens for logit-masking in structured generation.
- **Token healing** — trims mid-token prompt tails so generation resumes on a natural boundary.
- **Cross-tokenizer transcoding** — `transcode` re-maps ids from vocab A to vocab B via a lossless text bridge.
- **Tokenization debugger** — `explain` reports per-token *why-chosen* (merge sequence / Viterbi / Monster branch) as text or JSON.
- **Multimodal encode** — Mistral Tekken text + image + audio in one unified stream (`encode-multimodal`).
- **Per-token byte offsets under any normalizer** — `encodeWithOffsets` translates spans back to original-input coordinates through NFC/NFD/NFKC/NFKD/byte_level via a normalizer origin map; powers RAG chunking with byte-accurate ranges.

## Design constraints for the Zig library
- **Multithreaded by default** — batch encode parallelism is the default path, single-thread is opt-in. Per-thread arenas/caches.
- **Data-oriented design** — struct-of-arrays for vocab tables, contiguous merge-rank arrays keyed by integer id, avoid pointer-chasing AST-style token trees, prefer indexed flat buffers.
- Pluggable algorithm/normalizer/pretok pipeline (HF-style modularity) but with DoD storage underneath.

## Build order — all phases complete
1. ✅ `.tiktoken` loader + byte-level BPE encode.
2. ✅ Decoder.
3. ✅ Regex pre-tokenizer (cl100k / GPT-4 pattern).
4. ✅ Multithreaded batch encode with per-thread arenas.
5. ✅ HF JSON loader (WordPiece + BPE + Unigram dispatch).
6. ✅ SentencePiece `.model` proto loader + Unigram Viterbi.
7. ✅ TokenMonster ungreedy scoring.
8. ✅ Training (BPE, Unigram EM, WordPiece, Monster distillation, PathPiece).
9. ✅ SIMD passes (portable `@Vector`, target-native width).
