# Tokenizer fixture availability matrix

This table tracks every large-model tokenizer ztok has investigated for
inclusion in the cross-tokenizer benchmark suite. "Vendored" means the
bytes live in this directory and are committed to the repo; "Fetched"
means `bench/fetch_vocabs.py --extended` downloads them on demand from
the URL listed.

Last updated: 2026-05-19. Re-validate URLs / sizes when adding new
entries — Hugging Face occasionally reshuffles repo paths, especially
for newly-released models.

## Vendored (committed to `bench/vocabs/`)

These are small enough (≤ ~10 MB each, ~43 MiB total) that we ship them
directly so the default `zig build test` / equivalence sweep works
offline.

| model              | tokenizer format          | url                                                              | auth | size    |
|--------------------|---------------------------|------------------------------------------------------------------|------|---------|
| LLaMA-2            | SentencePiece BPE         | huggingface.co/hf-internal-testing/llama-tokenizer               | no   | 500 KB  |
| LLaMA-3-8B         | HF byte-level BPE         | huggingface.co/unsloth/llama-3-8b                                | no   | 9.1 MB  |
| Mistral-7B v0.1    | SentencePiece BPE         | huggingface.co/mistralai/Mistral-7B-v0.1                         | no   | 493 KB  |
| Yi-6B              | SentencePiece BPE         | huggingface.co/01-ai/Yi-6B                                       | no   | 1.0 MB  |
| Gemma-2B           | SentencePiece BPE (nfkc_cf + byte_fallback) | huggingface.co/unsloth/gemma-2b              | no   | 4.2 MB  |
| T5-small           | SentencePiece Unigram     | huggingface.co/google-t5/t5-small                                | no   | 792 KB  |
| llm-jp-3-1.8B      | HF Unigram (byte_fallback)| huggingface.co/llm-jp/llm-jp-3-1.8b                              | no   | 6.4 MB  |
| GPT-2              | HF BPE (ByteLevel)        | huggingface.co/openai-community/gpt2                             | no   | 1.4 MB  |
| BERT base uncased  | HF WordPiece (BertNormalizer) | huggingface.co/google-bert/bert-base-uncased                 | no   | 466 KB  |
| Phi-3-mini-4k      | HF BPE                    | huggingface.co/microsoft/Phi-3-mini-4k-instruct                  | no   | 1.9 MB  |
| Falcon-7B          | HF BPE (ByteLevel)        | huggingface.co/tiiuae/falcon-7b                                  | no   | 2.7 MB  |
| DeepSeek-V2-Lite   | HF BPE                    | huggingface.co/deepseek-ai/DeepSeek-V2-Lite                      | no   | 4.6 MB  |
| Qwen2-7B           | HF BPE (ByteLevel)        | huggingface.co/Qwen/Qwen2-7B                                     | no   | 7.0 MB  |
| cl100k_base        | tiktoken BPE              | github.com/openai/tiktoken (bundled)                             | no   | 1.7 MB  |
| TokenMonster englishcode-32000 | TokenMonster (.vocab → .ztm) | pip `tokenmonster`                              | no   | ~900 KB |

## Fetched on demand (`fetch_vocabs.py --extended`)

These push the vendored-vocab directory past 50 MB or are added later
as the loader catches up. Each fetch is non-fatal — if HF rate-limits
or a mirror disappears, the bench skips that fixture.

| model                       | tokenizer format         | url                                                              | auth | size    | added |
|-----------------------------|--------------------------|------------------------------------------------------------------|------|---------|-------|
| **Mistral Nemo Base 2407**  | **Tekken v3**            | huggingface.co/mistralai/Mistral-Nemo-Base-2407 (`tekken.json`)  | no   | 14.8 MB | 1.22+ |
| Mistral-Small-Instruct-2409 | SentencePiece v3         | huggingface.co/mistralai/Mistral-Small-Instruct-2409 (`tokenizer.model.v3`) | no   | 587 KB  | 1.22+ |
| Codestral-22B v0.1          | SentencePiece v3         | huggingface.co/mistralai/Codestral-22B-v0.1 (`tokenizer.model.v3`) | no | 587 KB  | 1.22+ |
| Phi-3.5-mini                | HF BPE                   | huggingface.co/microsoft/Phi-3.5-mini-instruct                   | no   | 1.8 MB  | 1.22+ |
| Phi-4                       | HF BPE                   | huggingface.co/microsoft/phi-4                                   | no   | 4.3 MB  | 1.22+ |
| Qwen-2.5-7B                 | HF BPE (ByteLevel)       | huggingface.co/Qwen/Qwen2.5-7B                                   | no   | 7.0 MB  | 1.22+ |
| Qwen-3-8B                   | HF BPE (ByteLevel)       | huggingface.co/Qwen/Qwen3-8B                                     | no   | 11.4 MB | 1.22+ |
| DeepSeek-V3                 | HF BPE                   | huggingface.co/deepseek-ai/DeepSeek-V3                           | no   | 7.8 MB  | 1.22+ |

## Investigated, not yet added

Auth-free but skipped for size, loader-coverage, or licence reasons.
Worth picking up when the bandwidth or loader work lands.

| model                       | tokenizer format         | url                                                              | auth | size     | reason for skipping |
|-----------------------------|--------------------------|------------------------------------------------------------------|------|----------|---------------------|
| Pixtral-12B-2409            | Tekken + multimodal extras | huggingface.co/mistralai/Pixtral-12B-2409 (`tekken.json`)     | no   | 19.3 MB  | Multimodal-extension Tekken (image_config + audio_config); covered by Tekken loader but `src/tekken.zig` doesn't run the image pixel pipeline. Add when there's a multimodal bench. |
| Pixtral HF mirror           | HF BPE (`tokenizer.json`) | huggingface.co/mistralai/Pixtral-12B-2409 (`tokenizer.json`)    | no   | 17.1 MB  | Same vocab as Tekken file but in HF-tokenizers wire format. Redundant with the Tekken fetch. |
| Codestral-22B (HF mirror)   | HF BPE (`tokenizer.json`) | huggingface.co/mistralai/Codestral-22B-v0.1 (`tokenizer.json`)  | no   | 2.0 MB   | Same vocab as the .model.v3 fetch entry already listed. |
| Ministral-8B-Instruct-2410  | Tekken v3                | huggingface.co/mistralai/Ministral-8B-Instruct-2410 (`tekken.json`) | no | 14.8 MB | Tokenizer is byte-identical to Mistral Nemo's (same SHA on inspection). Redundant. |
| Mistral-Small-3.1-24B (HF)  | HF BPE (`tokenizer.json`) | huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503 (`tokenizer.json`) | no | 17.1 MB | HF-mirror of the Tekken vocab; mostly redundant with the Nemo Tekken file we already fetch. |
| OpenAI **GPT-OSS-20B**      | HF BPE (harmony/tiktoken-equivalent) | huggingface.co/openai/gpt-oss-20b (`tokenizer.json`)| no   | 27.9 MB  | Largest open-weight tokenizer found (o200k_harmony, ~200K vocab). Fits under the 32 MB fetch cap; can be added when a follow-up bench wants OpenAI's open release. |
| OpenAI **GPT-OSS-120B**     | HF BPE (harmony)         | huggingface.co/openai/gpt-oss-120b (`tokenizer.json`)             | no   | 33.4 MB  | Identical vocab to GPT-OSS-20B (same o200k_harmony export); skip in favour of the 20B file if GPT-OSS lands. |
| Gemma-2-9B                  | HF JSON                  | huggingface.co/google/gemma-2-9b (`tokenizer.json`)               | no   | 17.2 MB  | Same SP vocab as the existing `gemma.model` fixture but in HF wire format — only marginal coverage value. |
| Llama-3.1-8B (unsloth)      | HF byte-level BPE        | huggingface.co/unsloth/Meta-Llama-3.1-8B (`tokenizer.json`)       | no   | 17.0 MB  | Tokenizer is identical to Llama-3-8B's (vocab unchanged 3.0 → 3.1). Redundant with the already-fetched `llama3.json`. |

## Investigated, auth-gated (cannot fetch without HF token)

These all return HTTP 401 on a token-less `curl`. Documented here as
future work — if the project lands an authenticated HF mirror or a
licensed re-host appears, move the entry up into the fetched table.

| model                       | tokenizer format         | url                                                              | size      | notes |
|-----------------------------|--------------------------|------------------------------------------------------------------|-----------|-------|
| Llama-3.3-70B-Instruct      | HF byte-level BPE        | huggingface.co/meta-llama/Llama-3.3-70B-Instruct                 | unknown   | Meta-licence gate. Tokenizer believed identical to Llama-3.1 (no vocab bump shipped). Use the public Llama-3 fixture for now. |
| Llama-4-Scout-17B-16E       | unknown (likely HF BPE)  | huggingface.co/meta-llama/Llama-4-Scout-17B-16E                  | unknown   | Meta-licence gate; no public mirror seen yet. Document once an `unsloth/llama-4-*` mirror appears. |
| Gemma-3-12B-IT (unsloth)    | HF JSON                  | huggingface.co/unsloth/gemma-3-12b-it                             | unknown   | Mirror itself returned 401 in this sweep — re-check; if a true public mirror is available, vendor or fetch. |
| Meta-Llama-3.1-8B (Meta)    | HF byte-level BPE        | huggingface.co/meta-llama/Meta-Llama-3.1-8B                       | unknown   | Meta-licence gate. The `unsloth/Meta-Llama-3.1-8B` mirror above is the workaround. |
