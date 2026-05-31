'use strict';

// koffi bindings — mirrors include/ztok.h. Every exported function is
// declared here with full C signatures so calls fail loudly on signature
// drift. _Out_ markers tell koffi that JS will read the parameter back as
// a write-output (length-1 array or struct ref).
//
// Status / enum / format codes are mirrored directly from the header so
// callers can use them as named constants instead of magic numbers.

const koffi = require('koffi');
const { loadLibztok } = require('./lib');

// --- status codes (mirror enum ztok_status) ---
const ZTOK_OK = 0;
const ZTOK_ERR_OUT_OF_MEMORY = 1;
const ZTOK_ERR_INVALID_INPUT = 2;
const ZTOK_ERR_BUFFER_TOO_SMALL = 3;
const ZTOK_ERR_INTERNAL = 99;

// --- enum kinds (mirror ztok_*_kind enums) ---
const NORMALIZER_IDENTITY = 0;
const NORMALIZER_NFC = 1;
const NORMALIZER_NFD = 2;
const NORMALIZER_NFKC = 3;
const NORMALIZER_NFKD = 4;
const NORMALIZER_BYTE_LEVEL = 5;

const PRETOK_IDENTITY = 0;
const PRETOK_CL100K = 1;
const PRETOK_TEKKEN = 2;

const MODEL_BYTE_ID = 0;

const DECODER_CONCAT = 0;
const DECODER_WORDPIECE = 1;
const DECODER_BYTE_LEVEL = 2;

// --- auto-detect format codes (mirror enum ztok_format) ---
const FORMAT_UNKNOWN = 0;
const FORMAT_TIKTOKEN = 1;
const FORMAT_HF_JSON = 2;
const FORMAT_SP_MODEL = 3;
const FORMAT_ZTM = 4;
const FORMAT_TEKKEN = 5;
const FORMAT_RWKV = 6;

// --- chunk boundary modes (mirror enum ztok_chunk_boundary) ---
const CHUNK_BOUNDARY_TOKEN = 0;
const CHUNK_BOUNDARY_CODEPOINT = 1;
const CHUNK_BOUNDARY_WORD = 2;
const CHUNK_BOUNDARY_WORD_DICT = 3;
const CHUNK_BOUNDARY_SENTENCE = 4;
const CHUNK_BOUNDARY_PARAGRAPH = 5;

// --- overlay channel kinds (mirror enum ztok_overlay_kind) ---
const OVERLAY_BYTE_START = 0;
const OVERLAY_BYTE_END = 1;
const OVERLAY_BOUNDARY = 2;
const OVERLAY_OPCODE = 3;
const OVERLAY_OPERAND = 4;
const OVERLAY_SYMBOL_REF = 5;
const OVERLAY_HUNK = 6;
const OVERLAY_PROVENANCE = 7;
const OVERLAY_USER_BASE = 0x8000;

const FORMAT_CODE_TO_NAME = {
    [FORMAT_UNKNOWN]: 'unknown',
    [FORMAT_TIKTOKEN]: 'tiktoken',
    [FORMAT_HF_JSON]: 'hf_json',
    [FORMAT_SP_MODEL]: 'sentencepiece',
    [FORMAT_ZTM]: 'ztm',
    [FORMAT_TEKKEN]: 'tekken',
    [FORMAT_RWKV]: 'rwkv',
};

// Module-level singleton — koffi caches the dlopen handle, and binding
// the same library twice is wasteful. The first import wires everything.
let cached = null;

function getLib() {
    if (cached) return cached;

    const lib = loadLibztok(koffi);

    // Struct mirroring `ztok_pipeline_config`. All four fields are the
    // numeric enum values.
    const ZtokPipelineConfig = koffi.struct('ztok_pipeline_config', {
        normalizer: 'uint32_t',
        pre_tokenizer: 'uint32_t',
        model: 'uint32_t',
        decoder: 'uint32_t',
    });

    // --- lifecycle ---
    const ztok_pipeline_new = lib.func(
        'void* ztok_pipeline_new(ztok_pipeline_config* cfg, _Out_ int* status)'
    );
    const ztok_pipeline_free = lib.func('void ztok_pipeline_free(void* p)');

    const ztok_pipeline_new_bpe_from_tiktoken = lib.func(
        'void* ztok_pipeline_new_bpe_from_tiktoken(const char* path, ztok_pipeline_config* cfg, _Out_ int* status)'
    );
    const ztok_pipeline_new_bpe_from_hf_json = lib.func(
        'void* ztok_pipeline_new_bpe_from_hf_json(const char* path, ztok_pipeline_config* cfg, _Out_ int* status)'
    );
    const ztok_pipeline_new_wordpiece_from_hf_json = lib.func(
        'void* ztok_pipeline_new_wordpiece_from_hf_json(const char* path, uint32_t unk_id, ztok_pipeline_config* cfg, _Out_ int* status)'
    );
    const ztok_pipeline_new_unigram_from_sp_model = lib.func(
        'void* ztok_pipeline_new_unigram_from_sp_model(const char* path, uint32_t unk_id, ztok_pipeline_config* cfg, _Out_ int* status)'
    );
    const ztok_pipeline_new_monster_from_file = lib.func(
        'void* ztok_pipeline_new_monster_from_file(const char* path, ztok_pipeline_config* cfg, _Out_ int* status)'
    );
    const ztok_pipeline_new_rwkv_from_file = lib.func(
        'void* ztok_pipeline_new_rwkv_from_file(const char* path, ztok_pipeline_config* cfg, _Out_ int* status)'
    );
    const ztok_pipeline_new_tekken_from_file = lib.func(
        'void* ztok_pipeline_new_tekken_from_file(const char* path, ztok_pipeline_config* cfg, _Out_ int* status)'
    );

    // --- encode / decode ---
    const ztok_encode = lib.func(
        'int ztok_encode(void* p, const char* input, size_t input_len, _Out_ uint32_t* out, size_t out_cap, _Out_ size_t* out_len)'
    );
    const ztok_decode = lib.func(
        'int ztok_decode(void* p, uint32_t* ids, size_t ids_len, _Out_ uint8_t* out, size_t out_cap, _Out_ size_t* out_len)'
    );

    // --- encode with overlay channels ---
    // Struct mirroring `ztok_overlay_channel`. `out` is a caller-owned
    // uint32_t buffer of `out_cap` entries that the encoder fills with one
    // value per token. We pass it as a void* so JS can hand koffi either
    // NULL (sizing pass) or a typed-array buffer (fill pass).
    const ZtokOverlayChannel = koffi.struct('ztok_overlay_channel', {
        kind: 'uint32_t',
        out: 'void*',
        out_cap: 'size_t',
    });
    const ztok_encode_with_overlays = lib.func(
        'int ztok_encode_with_overlays(void* p, const char* input, size_t input_len, _Out_ uint32_t* out_ids, size_t out_ids_cap, ztok_overlay_channel* channels, size_t n_channels, _Out_ size_t* out_len)'
    );

    // --- batch ---
    const ztok_encode_batch = lib.func(
        'int ztok_encode_batch(void* p, const char** inputs, size_t* input_lens, size_t n, _Out_ void** out_ids, _Out_ size_t* out_lens, uint32_t n_workers)'
    );
    const ztok_batch_pool_new = lib.func(
        'void* ztok_batch_pool_new(uint32_t n_workers, _Out_ int* status)'
    );
    const ztok_batch_pool_free = lib.func('void ztok_batch_pool_free(void* pool)');
    const ztok_batch_pool_worker_count = lib.func(
        'size_t ztok_batch_pool_worker_count(void* pool)'
    );
    const ztok_encode_batch_pooled = lib.func(
        'int ztok_encode_batch_pooled(void* p, void* pool, const char** inputs, size_t* input_lens, size_t n, _Out_ void** out_ids, _Out_ size_t* out_lens)'
    );

    const ztok_ids_free = lib.func('void ztok_ids_free(void* ids)');

    // --- Engram n-gram hashing ---
    // Raw token ids in, row-major [position][head] uint64 hashes out. The
    // single-doc path fills a caller-owned uint64 buffer; the batch path
    // hands back one ztok-allocated buffer per doc (free with
    // ztok_u64s_free). We pass id arrays/hash buffers as void* so JS can
    // marshal typed arrays or NULL.
    const ztok_ngram_hash = lib.func(
        'int ztok_ngram_hash(uint32_t* ids, size_t n_ids, uint32_t n, uint32_t heads, _Out_ uint64_t* out, size_t out_cap, _Out_ size_t* out_len)'
    );
    const ztok_ngram_hash_batch = lib.func(
        'int ztok_ngram_hash_batch(void* pool, const uint32_t** id_arrays, size_t* id_lens, size_t n_docs, uint32_t n, uint32_t heads, _Out_ void** out_hashes, _Out_ size_t* out_lens)'
    );
    const ztok_u64s_free = lib.func('void ztok_u64s_free(void* hashes)');

    // --- chunking ---
    // Struct mirroring `ztok_chunk_rec`. `ids` is a ztok-allocated buffer
    // of `ids_len` token ids (NULL when ids_len==0), released via
    // ztok_chunks_free; byte_*/token_* are half-open ranges in the
    // original input.
    const ZtokChunkRec = koffi.struct('ztok_chunk_rec', {
        ids: 'void*',
        ids_len: 'size_t',
        byte_start: 'uint32_t',
        byte_end: 'uint32_t',
        token_start: 'uint32_t',
        token_end: 'uint32_t',
    });
    const ztok_chunk = lib.func(
        'int ztok_chunk(void* pipeline, const char* text, size_t text_len, uint32_t max_tokens, uint32_t overlap, uint32_t boundary, _Out_ ztok_chunk_rec* out_chunks, size_t out_cap, _Out_ size_t* out_len)'
    );
    const ztok_chunks_free = lib.func('void ztok_chunks_free(ztok_chunk_rec* chunks, size_t n)');

    // --- auto-detect ---
    const ztok_auto_detect = lib.func('uint32_t ztok_auto_detect(const char* path)');

    // --- streaming ---
    const ztok_stream_new = lib.func(
        'void* ztok_stream_new(void* p, _Out_ int* status)'
    );
    const ztok_stream_free = lib.func('void ztok_stream_free(void* s)');
    const ztok_stream_feed = lib.func(
        'int ztok_stream_feed(void* s, const char* bytes, size_t n_bytes, _Out_ void** out_ids, _Out_ size_t* out_n_ids)'
    );
    const ztok_stream_finish = lib.func(
        'int ztok_stream_finish(void* s, _Out_ void** out_ids, _Out_ size_t* out_n_ids)'
    );

    // --- version ---
    const ztok_version = lib.func('const char* ztok_version()');

    cached = {
        koffi,
        ZtokPipelineConfig,
        ZtokOverlayChannel,
        ZtokChunkRec,
        ztok_encode_with_overlays,
        ztok_pipeline_new,
        ztok_pipeline_free,
        ztok_pipeline_new_bpe_from_tiktoken,
        ztok_pipeline_new_bpe_from_hf_json,
        ztok_pipeline_new_wordpiece_from_hf_json,
        ztok_pipeline_new_unigram_from_sp_model,
        ztok_pipeline_new_monster_from_file,
        ztok_pipeline_new_rwkv_from_file,
        ztok_pipeline_new_tekken_from_file,
        ztok_ngram_hash,
        ztok_ngram_hash_batch,
        ztok_u64s_free,
        ztok_chunk,
        ztok_chunks_free,
        ztok_encode,
        ztok_decode,
        ztok_encode_batch,
        ztok_batch_pool_new,
        ztok_batch_pool_free,
        ztok_batch_pool_worker_count,
        ztok_encode_batch_pooled,
        ztok_ids_free,
        ztok_auto_detect,
        ztok_stream_new,
        ztok_stream_free,
        ztok_stream_feed,
        ztok_stream_finish,
        ztok_version,
    };
    return cached;
}

function detectFormat(libPath) {
    const lib = getLib();
    const code = lib.ztok_auto_detect(libPath);
    return FORMAT_CODE_TO_NAME[code] || 'unknown';
}

module.exports = {
    getLib,
    detectFormat,

    // Status codes
    ZTOK_OK,
    ZTOK_ERR_OUT_OF_MEMORY,
    ZTOK_ERR_INVALID_INPUT,
    ZTOK_ERR_BUFFER_TOO_SMALL,
    ZTOK_ERR_INTERNAL,

    // Enum kinds
    NORMALIZER_IDENTITY,
    NORMALIZER_NFC,
    NORMALIZER_NFD,
    NORMALIZER_NFKC,
    NORMALIZER_NFKD,
    NORMALIZER_BYTE_LEVEL,
    PRETOK_IDENTITY,
    PRETOK_CL100K,
    PRETOK_TEKKEN,
    MODEL_BYTE_ID,
    DECODER_CONCAT,
    DECODER_WORDPIECE,
    DECODER_BYTE_LEVEL,

    // Format codes
    FORMAT_UNKNOWN,
    FORMAT_TIKTOKEN,
    FORMAT_HF_JSON,
    FORMAT_SP_MODEL,
    FORMAT_ZTM,
    FORMAT_TEKKEN,
    FORMAT_RWKV,
    FORMAT_CODE_TO_NAME,

    // Chunk boundary modes
    CHUNK_BOUNDARY_TOKEN,
    CHUNK_BOUNDARY_CODEPOINT,
    CHUNK_BOUNDARY_WORD,
    CHUNK_BOUNDARY_WORD_DICT,
    CHUNK_BOUNDARY_SENTENCE,
    CHUNK_BOUNDARY_PARAGRAPH,

    // Overlay channel kinds
    OVERLAY_BYTE_START,
    OVERLAY_BYTE_END,
    OVERLAY_BOUNDARY,
    OVERLAY_OPCODE,
    OVERLAY_OPERAND,
    OVERLAY_SYMBOL_REF,
    OVERLAY_HUNK,
    OVERLAY_PROVENANCE,
    OVERLAY_USER_BASE,
};
