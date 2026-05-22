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
