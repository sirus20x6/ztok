# Equivalence stress corpora

Vendored corpus pack used by `bench/equivalence_check.py` to gate
cross-tokenizer parity at 10K lines per (model, corpus) pair. The
100-line `/tmp/corpus.txt` gate catches the dominant divergences
(normalizer rules, pretok regex, BPE merge ordering); these widen the
window so rare-codepoint and edge-case bugs surface too.

| file                | lines | source |
|---------------------|------:|--------|
| `english.txt`       | 10000 | Project Gutenberg public-domain prose, mixed register (Shakespeare #100 + Twain #76 + Darwin #1228) |
| `code.txt`          | 10000 | ztok repo's own code (MIT) + synthetic snippets covering Python / JS / Go / Rust / Zig / C idioms |
| `multilingual.txt`  | 10000 | Curated public-domain phrases across Spanish, French, German, Mandarin (Hans), Japanese, Russian, Arabic, Hindi |
| `chat.txt`          | 10000 | Synthetically-generated Q&A-shape conversational text covering contractions, code fences, URLs, smart quotes, mixed punctuation |
| `unicode_stress.txt`|  1000 | Adversarial Unicode: combining marks, ZWJ emoji families, bidi (RTL embedded in LTR + isolates), variation selectors, halfwidth/fullwidth, decomposed Hangul, mathematical alphanumerics, Zalgo, NBSP/ZWSP/MMSP whitespace, 4-byte astral planes |

Total vendored size: ~3 MB (well under the 20 MB cap).

## Regenerating

```sh
python3 bench/corpora/build_corpora.py            # idempotent, skips existing files
python3 bench/corpora/build_corpora.py --force    # rebuild everything
python3 bench/corpora/build_corpora.py --only unicode_stress.txt
```

Network-fetched sources (Gutenberg) are cached under
`bench/corpora/_cache/` (gitignored — fetch is cheap & idempotent).

## Running the equivalence sweep

```sh
# Single fixture × single corpus
python3 bench/equivalence_check.py sp-bpe bench/vocabs/llama2 \
    --corpus bench/corpora/english.txt --lines 10000

# All 13 × 5 — see the smoke harness
bench/equivalence_stress_sweep.sh
```

See `bench/RESULTS.md` `## 1.21 10K-line stress equivalence` for the
full table.
