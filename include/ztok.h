/* ztok C ABI — stable wrapper around the Zig Pipeline.
 *
 * Kind enums are numeric and open-ended: callers pass an integer, the Zig
 * side maps known values to variants and rejects everything else. Adding
 * new variants never breaks the ABI; older binaries just return
 * ZTOK_ERR_INVALID_INPUT for kinds they don't know.
 *
 * Backwards compatibility: every function, enum value, and struct present
 * in 1.1.0 is preserved verbatim. The Wave-C additions
 * (`ztok_batch_pool_*`, `ztok_encode_batch_pooled`, the file-based
 * constructors, and the extended enum values) are strictly additive.
 */

#ifndef ZTOK_H
#define ZTOK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t ztok_token_id;
typedef struct ztok_pipeline ztok_pipeline;     /* opaque */
typedef struct ztok_batch_pool ztok_batch_pool; /* opaque */
typedef struct ztok_stream ztok_stream;         /* opaque (streaming encode) */
typedef struct ztok_superposition_plan ztok_superposition_plan; /* experimental, opaque */

typedef enum {
    ZTOK_OK = 0,
    ZTOK_ERR_OUT_OF_MEMORY = 1,
    ZTOK_ERR_INVALID_INPUT = 2,
    ZTOK_ERR_BUFFER_TOO_SMALL = 3,
    ZTOK_ERR_INTERNAL = 99,
} ztok_status;

typedef enum {
    ZTOK_NORMALIZER_IDENTITY = 0,
    ZTOK_NORMALIZER_NFC = 1,
    ZTOK_NORMALIZER_NFD = 2,
    ZTOK_NORMALIZER_NFKC = 3,
    ZTOK_NORMALIZER_NFKD = 4,
    ZTOK_NORMALIZER_BYTE_LEVEL = 5,
} ztok_normalizer_kind;

typedef enum {
    ZTOK_PRETOK_IDENTITY = 0,
    ZTOK_PRETOK_CL100K = 1,
} ztok_pretok_kind;

typedef enum {
    ZTOK_MODEL_BYTE_ID = 0,
    /* Concrete model variants are picked by the constructor used (BPE,
     * Unigram, WordPiece, Monster), not via this enum. The byte_id stub
     * remains the only kind selectable from ztok_pipeline_new. */
} ztok_model_kind;

typedef enum {
    ZTOK_DECODER_CONCAT = 0,
    ZTOK_DECODER_WORDPIECE = 1,
    ZTOK_DECODER_BYTE_LEVEL = 2,
} ztok_decoder_kind;

/* Auto-detected on-disk vocab format. Returned by ztok_auto_detect.
 * Values are stable and additive. Unknown files (or any I/O error) map
 * to ZTOK_FORMAT_UNKNOWN — ztok_auto_detect is a best-effort sniffer,
 * never the place to surface a real error. */
typedef enum {
    ZTOK_FORMAT_UNKNOWN = 0,
    ZTOK_FORMAT_TIKTOKEN = 1,
    ZTOK_FORMAT_HF_JSON = 2,
    ZTOK_FORMAT_SP_MODEL = 3,
    ZTOK_FORMAT_ZTM = 4,
    /* Mistral Tekken (`tekken.json`): tiktoken-style BPE with base64
     * `token_bytes` entries, explicit `special_tokens` block, and a
     * top-level `config.pattern` regex. Distinguished from HF
     * tokenizer.json by the `"type": "Tekkenizer"` marker. */
    ZTOK_FORMAT_TEKKEN = 5,
    /* RWKV "World" vocab (`rwkv_vocab_v20230424.txt`): line-oriented
     * `<id> <python-repr> <byte-len>` entries driving a greedy
     * longest-match byte trie. Sniffed by the leading id + quoted/`b`-
     * quoted middle column + trailing length. */
    ZTOK_FORMAT_RWKV = 6,
} ztok_format;

typedef struct {
    ztok_normalizer_kind normalizer;
    ztok_pretok_kind pre_tokenizer;
    ztok_model_kind model;
    ztok_decoder_kind decoder;
} ztok_pipeline_config;

/* --- lifecycle ---------------------------------------------------- */

/* byte_id pipeline driven by the config enums. out_status may be NULL. */
ztok_pipeline* ztok_pipeline_new(const ztok_pipeline_config* cfg, ztok_status* out_status);
void ztok_pipeline_free(ztok_pipeline* p);

/* File-based constructors. cfg_or_null = NULL applies sensible defaults:
 *   { .normalizer = IDENTITY, .pre_tokenizer = IDENTITY,
 *     .model = (set by constructor), .decoder = CONCAT }
 * For BPE-from-tiktoken, default pre_tokenizer is CL100K instead. */
ztok_pipeline* ztok_pipeline_new_bpe_from_tiktoken(
    const char* tiktoken_path,
    const ztok_pipeline_config* cfg_or_null,
    ztok_status* out_status
);

ztok_pipeline* ztok_pipeline_new_bpe_from_hf_json(
    const char* tokenizer_json_path,
    const ztok_pipeline_config* cfg_or_null,
    ztok_status* out_status
);

ztok_pipeline* ztok_pipeline_new_wordpiece_from_hf_json(
    const char* tokenizer_json_path,
    uint32_t unk_id,
    const ztok_pipeline_config* cfg_or_null,
    ztok_status* out_status
);

ztok_pipeline* ztok_pipeline_new_unigram_from_sp_model(
    const char* sp_model_path,
    uint32_t unk_id,
    const ztok_pipeline_config* cfg_or_null,
    ztok_status* out_status
);

/* Monster loader. Reads the ztok-native .ztm binary format (magic
 * "ZTM\x01", little-endian); see src/monster_io.zig for the layout. */
ztok_pipeline* ztok_pipeline_new_monster_from_file(
    const char* path,
    const ztok_pipeline_config* cfg_or_null,
    ztok_status* out_status
);

/* RWKV "World" loader. Reads a `rwkv_vocab_v20230424.txt`-style file
 * (`<id> <python-repr> <byte-len>` per line) into a greedy longest-match
 * byte trie. The pipeline runs identity normalizer + identity
 * pre-tokenizer + concat decoder, matching the byte-lossless World
 * scheme (every byte 0..255 is a token, so encode never fails). */
ztok_pipeline* ztok_pipeline_new_rwkv_from_file(
    const char* path,
    const ztok_pipeline_config* cfg_or_null,
    ztok_status* out_status
);

/* Mistral Tekken loader. Reads a `tekken.json` file (Nemo / Pixtral /
 * Devstral / Magistral, etc.) and lowers its base64 byte vocab into a
 * BPE with the special tokens packed into the bottom of the id space.
 * The pipeline runs identity normalizer + the Tekken pre-tokenizer
 * (NOT cl100k) + concat decoder. */
ztok_pipeline* ztok_pipeline_new_tekken_from_file(
    const char* path,
    const ztok_pipeline_config* cfg_or_null,
    ztok_status* out_status
);

/* --- encode / decode --------------------------------------------- */

/* Encode: writes ids directly into the caller buffer. *out_len receives
 * the count written. If out is NULL or out_cap is too small, *out_len
 * is set to the required size and ZTOK_ERR_BUFFER_TOO_SMALL is
 * returned. */
ztok_status ztok_encode(
    const ztok_pipeline* p,
    const char* input, size_t input_len,
    ztok_token_id* out, size_t out_cap,
    size_t* out_len
);

/* Decode mirrors encode: *out_len receives bytes written. */
ztok_status ztok_decode(
    const ztok_pipeline* p,
    const ztok_token_id* ids, size_t ids_len,
    char* out, size_t out_cap,
    size_t* out_len
);

/* --- encode with overlay channels -------------------------------- */

/* Annotation channels aligned 1:1 with the emitted token stream:
 * channel[i] describes token ids[i]. Requesting overlays does NOT change
 * tokenization — the id stream is identical to ztok_encode.
 *
 * Cheap channels (BYTE_START/BYTE_END/BOUNDARY/PROVENANCE) are derived
 * from state the encoder already computes. Domain channels
 * (OPCODE/OPERAND/SYMBOL_REF/HUNK) and any USER_BASE+ kind are
 * zero-filled until a domain normalizer plugin populates them. */
typedef enum {
    ZTOK_OVERLAY_BYTE_START = 0, /* offsets[i].start (orig-input byte)   */
    ZTOK_OVERLAY_BYTE_END   = 1, /* offsets[i].end (exclusive)           */
    ZTOK_OVERLAY_BOUNDARY   = 2, /* bitset: 0x1 chunk-start, 0x2 cp-start*/
    ZTOK_OVERLAY_OPCODE     = 3, /* domain: normalized opcode class      */
    ZTOK_OVERLAY_OPERAND    = 4, /* domain: normalized operand class     */
    ZTOK_OVERLAY_SYMBOL_REF = 5, /* domain: symbol-table index, 0 = none */
    ZTOK_OVERLAY_HUNK       = 6, /* domain: diff-hunk id                 */
    ZTOK_OVERLAY_PROVENANCE = 7, /* 0 = model text, 1 = special token    */
    ZTOK_OVERLAY_USER_BASE  = 0x8000 /* domain plugins claim >= this     */
} ztok_overlay_kind;

/* One requested channel. `out` is a caller-owned buffer of `out_cap`
 * uint32_t entries, filled with one value per token. */
typedef struct {
    ztok_overlay_kind kind;
    uint32_t*         out;
    size_t            out_cap;
} ztok_overlay_channel;

/* Encode `input`, writing ids into `out_ids` and each channel's per-token
 * values into channels[i].out. All buffers share the token count reported
 * via *out_len.
 *
 * Sizing: pass out_ids = NULL to query the count (returns
 * ZTOK_ERR_BUFFER_TOO_SMALL, sets *out_len). When out_ids is non-NULL,
 * every listed channel's `out` must be non-NULL with capacity >= *out_len;
 * if any buffer is too small, nothing is copied, *out_len is set to the
 * required count, and ZTOK_ERR_BUFFER_TOO_SMALL is returned. */
ztok_status ztok_encode_with_overlays(
    const ztok_pipeline* p,
    const char* input, size_t input_len,
    ztok_token_id* out_ids, size_t out_ids_cap,
    ztok_overlay_channel* channels, size_t n_channels,
    size_t* out_len
);

/* Domain whose overlay channels (e.g. x86-64 opcode_class / operand_class)
 * ztok_encode_with_overlays should populate. Default is NONE, which
 * zero-fills the domain channels. */
typedef enum {
    ZTOK_OVERLAY_DOMAIN_NONE   = 0,
    ZTOK_OVERLAY_DOMAIN_X86_64 = 1
} ztok_overlay_domain;

/* Select the pipeline's overlay domain. An unrecognized `domain` value
 * leaves the pipeline unchanged and returns ZTOK_ERR_INVALID_INPUT. */
ztok_status ztok_pipeline_set_overlay_domain(ztok_pipeline* p, ztok_overlay_domain domain);

/* --- experimental fixed superposition plans -----------------------
 *
 * A plan is an owned descriptor over ordinary token IDs and offsets.
 * It never adds synthetic IDs to the vocabulary or changes encode().
 * All integer enum values and the JSON schema are versioned from v1. */

typedef enum {
    ZTOK_SUPERPOSITION_FUSION_MEAN = 0,
    ZTOK_SUPERPOSITION_FUSION_WEIGHTED_MEAN = 1,
    ZTOK_SUPERPOSITION_FUSION_NORM_PRESERVING_MEAN = 2
} ztok_superposition_fusion;

typedef enum {
    ZTOK_SUPERPOSITION_GROUP_FIXED_WINDOW = 0,
    ZTOK_SUPERPOSITION_GROUP_PARTIAL_WINDOW = 1,
    ZTOK_SUPERPOSITION_GROUP_PRESERVED_SPECIAL = 2,
    ZTOK_SUPERPOSITION_GROUP_PRESERVED_BOUNDARY = 3,
    ZTOK_SUPERPOSITION_GROUP_UNCOVERED_TAIL = 4
} ztok_superposition_group_kind;

typedef struct {
    uint16_t group_size;
    uint16_t stride; /* 0 = group_size */
    uint8_t fusion;
    uint8_t preserve_special_tokens;
    uint8_t preserve_boundary_tokens;
    uint8_t allow_partial_final_group;
} ztok_superposition_fixed_config;

/* Optional token-aligned arrays used by ztok_superposition_plan_build().
 * Boolean masks contain canonical bytes (0 or 1). NULL means absent.
 * source_weights must be finite and non-negative; each group's copied
 * weights are normalized to sum to one. */
typedef struct {
    const uint8_t* special_token_mask;
    const uint8_t* boundary_token_mask;
    const uint8_t* hard_boundary_before;
    const float* source_weights;
} ztok_superposition_metadata;

typedef struct {
    uint32_t token_index;
    ztok_token_id token_id;
    uint32_t byte_start;
    uint32_t byte_end;
    float weight;
} ztok_superposition_source;

/* source_start/source_count address a range in
 * ztok_superposition_plan_sources(). position_end and byte_end are
 * exclusive. */
typedef struct {
    uint32_t output_index;
    uint32_t source_start;
    uint32_t source_count;
    uint8_t fusion;
    uint8_t kind;
    uint16_t reserved;
    uint32_t position_start;
    uint32_t position_end;
    uint32_t byte_start;
    uint32_t byte_end;
    float center_position;
    float normalized_center;
} ztok_superposition_group;

/* Build from a caller-supplied ordinary encoding. Input arrays are copied.
 * config_or_null uses the v1 defaults (group size 4, norm-preserving mean).
 * metadata_or_null supplies optional protected-token/boundary/weight arrays. */
ztok_superposition_plan* ztok_superposition_plan_build(
    const ztok_token_id* ids,
    const uint32_t* byte_starts,
    const uint32_t* byte_ends,
    size_t token_count,
    const ztok_superposition_fixed_config* config_or_null,
    const ztok_superposition_metadata* metadata_or_null,
    ztok_status* out_status
);

/* Convenience path: ordinary encode-with-offsets plus fixed plan. Special
 * added tokens are automatically protected when requested by the config. */
ztok_superposition_plan* ztok_pipeline_encode_superposition(
    const ztok_pipeline* p,
    const char* input,
    size_t input_len,
    const ztok_superposition_fixed_config* config_or_null,
    ztok_status* out_status
);

void ztok_superposition_plan_free(ztok_superposition_plan* plan);
uint32_t ztok_superposition_schema_version(void);
size_t ztok_superposition_plan_original_token_count(const ztok_superposition_plan* plan);
size_t ztok_superposition_plan_output_token_count(const ztok_superposition_plan* plan);
size_t ztok_superposition_plan_source_count(const ztok_superposition_plan* plan);
const ztok_token_id* ztok_superposition_plan_original_ids(const ztok_superposition_plan* plan);
ztok_status ztok_superposition_plan_original_offset(
    const ztok_superposition_plan* plan,
    size_t index,
    uint32_t* out_start,
    uint32_t* out_end
);
const ztok_superposition_source* ztok_superposition_plan_sources(
    const ztok_superposition_plan* plan
);
const ztok_superposition_group* ztok_superposition_plan_groups(
    const ztok_superposition_plan* plan
);

/* Deterministic ztok.superposition.v1 JSON. Uses the ordinary sizing
 * protocol: out=NULL queries required bytes and returns BUFFER_TOO_SMALL. */
ztok_status ztok_superposition_plan_json(
    const ztok_superposition_plan* plan,
    char* out,
    size_t out_cap,
    size_t* out_len
);

/* Build a ztok.ccss.v1 plan from a ztok.semantic_spans.v1 JSON exchange
 * document. config_json may be NULL/0 for conservative defaults. The input
 * carries caller-generated contextual span embeddings; libztok performs no
 * model inference. Output follows the standard sizing protocol. */
ztok_status ztok_ccss_build_json(
    const char* semantic_json,
    size_t semantic_json_len,
    const char* config_json,
    size_t config_json_len,
    char* out,
    size_t out_cap,
    size_t* out_len
);
uint32_t ztok_ccss_schema_version(void);

/* --- batch encode ------------------------------------------------- */

/* Per-call thread pool. inputs/input_lens are parallel arrays of n
 * entries. out_ids[i] receives a heap-allocated array of length
 * out_lens[i]; caller must call ztok_ids_free(out_ids[i]) for each. On
 * any failure, partial allocations are freed internally and ZTOK_ERR_*
 * is returned. n_workers=0 means auto-detect cpu count.
 *
 * Spawns one worker pool per call. For hot loops, prefer the persistent
 * pool API below — `ztok_encode_batch_pooled` reuses arenas + worker
 * threads across calls. */
ztok_status ztok_encode_batch(
    const ztok_pipeline* p,
    const char* const* inputs, const size_t* input_lens, size_t n,
    ztok_token_id** out_ids, size_t* out_lens,
    uint32_t n_workers
);

/* Persistent BatchPool. Create once, reuse across many batches. */
ztok_batch_pool* ztok_batch_pool_new(uint32_t n_workers, ztok_status* out_status); /* 0 = auto */
void ztok_batch_pool_free(ztok_batch_pool* pool);
size_t ztok_batch_pool_worker_count(const ztok_batch_pool* pool);

/* Batch encode via a persistent pool. Same out-array contract as
 * ztok_encode_batch; each out_ids[i] must be freed with ztok_ids_free. */
ztok_status ztok_encode_batch_pooled(
    const ztok_pipeline* p,
    const ztok_batch_pool* pool,
    const char* const* inputs, const size_t* input_lens, size_t n,
    ztok_token_id** out_ids, size_t* out_lens
);

void ztok_ids_free(ztok_token_id* ids);

/* --- Engram n-gram hashing ---------------------------------------- */

/* Deterministic multi-head token-n-gram hashing for Engram-style
 * conditional-memory addressing. Operates on raw token ids — no
 * pipeline handle needed. Output is row-major [position][head]: the
 * `heads` hashes for window position 0 come first, then position 1, etc.
 * Raw u64 hashes are emitted; mask each to your table width
 * (hash & ((1<<bits)-1)). The number of window positions for an
 * `n_ids`-long stream is (n_ids - n + 1), or 0 if shorter than n.
 *
 * The two n-gram functions return their u64 hashes through buffers with
 * OPPOSITE ownership — read the per-function notes below carefully:
 *   - ztok_ngram_hash       writes into a CALLER-owned buffer (you alloc
 *                           and free it; ztok never owns it).
 *   - ztok_ngram_hash_batch returns ZTOK-owned buffers that MUST be
 *                           released with ztok_u64s_free (NOT free(),
 *                           NOT ztok_ids_free).
 * Both surface the same C type (uint64_t*), so the compiler cannot catch
 * a mis-free; the asymmetry is enforced only by this contract.
 *
 * Forward-compat note: the bare (struct-less) uint64_t* return shape of
 * these two functions is INTENTIONALLY FROZEN. Any future variant that
 * needs per-window or per-head metadata will ship as a new symbol (e.g.
 * a struct-returning ztok_ngram_hash_ex), never by mutating the shape or
 * ownership of the existing returns. Do not assume these grow fields. */

/* Hash every length-`n` window of `ids` under `heads` hash functions.
 * `out` is a CALLER-OWNED buffer of `out_cap` uint64_t entries — ztok
 * never allocates or owns it, so there is nothing to free on the ztok
 * side (free your own buffer however you allocated it). On success
 * writes positions*heads hashes and sets *out_len to that count.
 *
 * Sizing/NULL convention (note: this DIFFERS from ztok_encode):
 *   - If the required count is 0 (stream shorter than one window, or
 *     n==0 / heads==0), sets *out_len=0 and returns ZTOK_OK even when
 *     `out` is NULL — an empty result is a success, not a sizing error.
 *   - Otherwise, if `out` is NULL or `out_cap` is too small, sets
 *     *out_len to the required count and returns ZTOK_BUFFER_TOO_SMALL
 *     without writing.
 * This is deliberately unlike ztok_encode, where a NULL `out` ALWAYS
 * returns ZTOK_BUFFER_TOO_SMALL (its sizing query never returns OK).
 * Do not port the ztok_encode "NULL == always TOO_SMALL" assumption to
 * this function. */
ztok_status ztok_ngram_hash(
    const ztok_token_id* ids, size_t n_ids,
    uint32_t n, uint32_t heads,
    uint64_t* out, size_t out_cap, size_t* out_len
);

/* Hash `n_docs` id streams in parallel across `pool`.
 *
 * OWNERSHIP (read this): on a ZTOK_OK return each out_hashes[i] is set to
 * a ZTOK-OWNED, header-prefixed uint64_t buffer holding the row-major
 * hashes for doc i, and out_lens[i] to its u64 count. Each non-NULL
 * out_hashes[i] MUST be released with ztok_u64s_free — and ONLY with
 * ztok_u64s_free. These buffers carry a length header that ztok_u64s_free
 * recovers; passing one to plain free() or to ztok_ids_free (which expects
 * the unrelated TokenId header) is undefined behavior and silently
 * corrupts the heap. This is the key asymmetry with ztok_ngram_hash,
 * whose `out` is caller-owned and never freed through ztok. A doc stream
 * shorter than one window yields out_hashes[i]==NULL and out_lens[i]==0
 * (skip freeing the NULL slots — ztok_u64s_free(NULL) is also a no-op).
 *
 * On any NON-OK return, every out_hashes[i] is NULL and every out_lens[i]
 * is 0: ztok has already freed any buffers it allocated, so the caller
 * frees NOTHING (do not call ztok_u64s_free on a failed batch).
 *
 * If n_docs==0 the call returns ZTOK_OK and leaves the out_hashes /
 * out_lens arrays UNTOUCHED (it writes neither — nothing to free).
 * (Contrast ztok_ngram_hash, which always writes *out_len, including 0
 * for the empty case.) */
ztok_status ztok_ngram_hash_batch(
    ztok_batch_pool* pool,
    const ztok_token_id* const* id_arrays, const size_t* id_lens, size_t n_docs,
    uint32_t n, uint32_t heads,
    uint64_t** out_hashes, size_t* out_lens
);

/* Free a uint64_t buffer returned by ztok_ngram_hash_batch (and ONLY such
 * a buffer). NULL is a safe no-op. Never pass a buffer here that you
 * allocated yourself for ztok_ngram_hash, and never free a batch buffer
 * with free()/ztok_ids_free — the ownership/header mismatch is UB. */
void ztok_u64s_free(uint64_t* hashes);

/* --- chunking ----------------------------------------------------- */

/* One token-window chunk produced by ztok_chunk. `ids` points at a
 * ztok-allocated buffer of `ids_len` token ids (NULL when ids_len==0).
 * `byte_start`/`byte_end` are the half-open byte range this chunk covers
 * in the ORIGINAL input; `token_start`/`token_end` the half-open
 * token-index range in the full encoding. */
typedef struct {
    ztok_token_id* ids;
    size_t ids_len;
    uint32_t byte_start;
    uint32_t byte_end;
    uint32_t token_start;
    uint32_t token_end;
} ztok_chunk_rec;

/* Boundary mode for ztok_chunk (mirrors chunk.Boundary). */
typedef enum {
    ZTOK_CHUNK_BOUNDARY_TOKEN     = 0, /* pure token-count windows */
    ZTOK_CHUNK_BOUNDARY_CODEPOINT = 1, /* snap to UTF-8 codepoint boundary */
    ZTOK_CHUNK_BOUNDARY_WORD      = 2, /* snap to whitespace word boundary */
    ZTOK_CHUNK_BOUNDARY_WORD_DICT = 3, /* dict word boundary (CJK/Thai...) */
    ZTOK_CHUNK_BOUNDARY_SENTENCE  = 4, /* snap to sentence boundary */
    ZTOK_CHUNK_BOUNDARY_PARAGRAPH = 5, /* snap to \n\n */
} ztok_chunk_boundary;

/* Split `text` into windows of at most `max_tokens` tokens with `overlap`
 * tokens shared between neighbors (stride = max_tokens - overlap).
 * `out_chunks` is a caller-owned buffer of `out_cap` records. On success
 * writes one record per chunk and sets *out_len to the chunk count. If
 * `out_chunks` is NULL or too small, sets *out_len to the required count
 * and returns ZTOK_BUFFER_TOO_SMALL without writing (no ids allocated).
 * Empty input yields 0 chunks and ZTOK_OK. Returns ZTOK_ERR_INVALID_INPUT
 * if max_tokens==0, overlap>=max_tokens, or boundary is out of range.
 * Each written record's `ids` must be released with ztok_chunks_free. */
ztok_status ztok_chunk(
    const ztok_pipeline* pipeline,
    const char* text, size_t text_len,
    uint32_t max_tokens, uint32_t overlap, uint32_t boundary,
    ztok_chunk_rec* out_chunks, size_t out_cap, size_t* out_len
);

/* Free the `ids` buffers of `n` chunk records written by ztok_chunk. The
 * `chunks` array itself is caller-owned and is not freed. */
void ztok_chunks_free(ztok_chunk_rec* chunks, size_t n);

/* --- auto-detect -------------------------------------------------- */

/* Best-effort vocab-format sniffer. Reads the first 256 bytes of `path`
 * and matches against the four known magics/heuristics. Returns
 * ZTOK_FORMAT_UNKNOWN on any error (missing file, permission denied,
 * unreadable bytes) — language bindings can use this without first
 * statting the path. */
ztok_format ztok_auto_detect(const char* path);

/* --- streaming encode -------------------------------------------- */

/* Open a streaming encode session against pipeline `p`. The session
 * holds a small carry buffer (default soft cap 1 MiB) and re-encodes
 * from the last safe pre-tokenizer cut on each feed; if the carry ever
 * exceeds the cap, it force-cuts at the nearest UTF-8 codepoint
 * boundary, matching `encodeChunked`'s caveat. */
ztok_stream* ztok_stream_new(const ztok_pipeline* p, ztok_status* out_status);
void ztok_stream_free(ztok_stream* s);

/* Feed `n_bytes` of input. On success, *out_ids is set to a
 * length-prefixed buffer carrying every id newly emitted by THIS call,
 * with *out_n_ids holding the count (may be 0 when the encoder is
 * still buffering toward the next safe cut). Caller must free each
 * non-NULL buffer via ztok_ids_free. */
ztok_status ztok_stream_feed(
    ztok_stream* s,
    const char* bytes, size_t n_bytes,
    ztok_token_id** out_ids, size_t* out_n_ids
);

/* Flush any remaining carry as a final encode. Same allocation
 * contract as ztok_stream_feed. Idempotent: a second call is a no-op
 * (returns *out_ids=NULL, *out_n_ids=0). */
ztok_status ztok_stream_finish(
    ztok_stream* s,
    ztok_token_id** out_ids, size_t* out_n_ids
);

/* Version */
const char* ztok_version(void);

/* --- fingerprint -------------------------------------------------- */

/* Compute the tokenizer fingerprint: a deterministic 32-byte SHA-256
 * digest over the pipeline's encoding behavior on a fixed canonical
 * input set (empty string, every byte 0..255, a UTF-8 mix, and a
 * whitespace-heavy string) plus a model-kind tag and vocab size.
 *
 * Two pipelines that return the same 32 bytes here will produce
 * bit-identical id streams for ANY input — use this as a cache key,
 * KV-store discriminator, or training-pipeline guard.
 *
 * `handle` is an opaque `ztok_pipeline*` returned by any constructor.
 * On success, ZTOK_OK is returned and 32 raw bytes are written to
 * `out_32`. NULL inputs return ZTOK_ERR_INVALID_INPUT. */
ztok_status ztok_fingerprint(ztok_pipeline* handle, uint8_t out_32[32]);

#ifdef __cplusplus
}
#endif
#endif /* ZTOK_H */
