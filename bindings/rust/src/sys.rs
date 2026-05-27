//! Raw FFI declarations for libztok.
//!
//! Hand-written to mirror `include/ztok.h` exactly so we don't pull in
//! `bindgen` (and `libclang`) as a build dep. The wider Rust binding
//! confines `unsafe` to this module plus a handful of `unsafe { ... }`
//! blocks in `pipeline.rs`, `batch.rs`, and `stream.rs` — every public
//! API in the crate is safe.
//!
//! Type map mirrors the ctypes layout the Python binding documents in
//! `bindings/python/ztok/_ffi.py`:
//!
//! | C type             | Rust type             |
//! |--------------------|-----------------------|
//! | `ztok_token_id`    | `u32`                 |
//! | `ztok_status`      | `c_int`               |
//! | `ztok_pipeline*`   | `*mut ZtokPipeline` (opaque) |
//! | `ztok_batch_pool*` | `*mut ZtokBatchPool` (opaque) |
//! | `ztok_stream*`     | `*mut ZtokStream` (opaque) |
//! | `const char*`      | `*const c_char`       |
//! | `char*` (mutable)  | `*mut c_char`         |
//! | `size_t`           | `usize`               |
//!
//! Memory contract for id buffers returned by `ztok_encode_batch*` and
//! `ztok_stream_*`: each carries an opaque length-prefix header (see
//! `src/c_api.zig::allocIdBuf`). The ONLY safe free is `ztok_ids_free`.
//! Never call `libc::free` on these — the allocator state would be
//! corrupted on the next batch.

#![allow(non_camel_case_types)]
// Raw FFI surface: every item here is documented in include/ztok.h.
// The module-level docs above explain the type map; per-item rustdoc
// would just duplicate the C header.
#![allow(missing_docs)]

use core::ffi::{c_char, c_int};

// --- opaque handles ------------------------------------------------------

/// Opaque pipeline handle returned by every constructor.
#[repr(C)]
pub struct ZtokPipeline {
    _private: [u8; 0],
}

/// Opaque persistent batch-pool handle.
#[repr(C)]
pub struct ZtokBatchPool {
    _private: [u8; 0],
}

/// Opaque streaming-encoder handle.
#[repr(C)]
pub struct ZtokStream {
    _private: [u8; 0],
}

// --- status / enum codes -------------------------------------------------

pub const ZTOK_OK: c_int = 0;
pub const ZTOK_ERR_OUT_OF_MEMORY: c_int = 1;
pub const ZTOK_ERR_INVALID_INPUT: c_int = 2;
pub const ZTOK_ERR_BUFFER_TOO_SMALL: c_int = 3;
pub const ZTOK_ERR_INTERNAL: c_int = 99;

pub const ZTOK_NORMALIZER_IDENTITY: u32 = 0;
pub const ZTOK_NORMALIZER_NFC: u32 = 1;
pub const ZTOK_NORMALIZER_NFD: u32 = 2;
pub const ZTOK_NORMALIZER_NFKC: u32 = 3;
pub const ZTOK_NORMALIZER_NFKD: u32 = 4;
pub const ZTOK_NORMALIZER_BYTE_LEVEL: u32 = 5;

pub const ZTOK_PRETOK_IDENTITY: u32 = 0;
pub const ZTOK_PRETOK_CL100K: u32 = 1;

pub const ZTOK_MODEL_BYTE_ID: u32 = 0;

pub const ZTOK_DECODER_CONCAT: u32 = 0;
pub const ZTOK_DECODER_WORDPIECE: u32 = 1;
pub const ZTOK_DECODER_BYTE_LEVEL: u32 = 2;

pub const ZTOK_FORMAT_UNKNOWN: u32 = 0;
pub const ZTOK_FORMAT_TIKTOKEN: u32 = 1;
pub const ZTOK_FORMAT_HF_JSON: u32 = 2;
pub const ZTOK_FORMAT_SP_MODEL: u32 = 3;
pub const ZTOK_FORMAT_ZTM: u32 = 4;
pub const ZTOK_FORMAT_TEKKEN: u32 = 5;
pub const ZTOK_FORMAT_RWKV: u32 = 6;

// Chunk boundary modes (mirror `ztok_chunk_boundary`).
pub const ZTOK_CHUNK_BOUNDARY_TOKEN: u32 = 0;
pub const ZTOK_CHUNK_BOUNDARY_CODEPOINT: u32 = 1;
pub const ZTOK_CHUNK_BOUNDARY_WORD: u32 = 2;
pub const ZTOK_CHUNK_BOUNDARY_WORD_DICT: u32 = 3;
pub const ZTOK_CHUNK_BOUNDARY_SENTENCE: u32 = 4;
pub const ZTOK_CHUNK_BOUNDARY_PARAGRAPH: u32 = 5;

// Overlay channel kinds (mirror `ztok_overlay_kind`).
pub const ZTOK_OVERLAY_BYTE_START: u32 = 0;
pub const ZTOK_OVERLAY_BYTE_END: u32 = 1;
pub const ZTOK_OVERLAY_BOUNDARY: u32 = 2;
pub const ZTOK_OVERLAY_OPCODE: u32 = 3;
pub const ZTOK_OVERLAY_OPERAND: u32 = 4;
pub const ZTOK_OVERLAY_SYMBOL_REF: u32 = 5;
pub const ZTOK_OVERLAY_HUNK: u32 = 6;
pub const ZTOK_OVERLAY_PROVENANCE: u32 = 7;
pub const ZTOK_OVERLAY_USER_BASE: u32 = 0x8000;

/// Mirrors `struct ztok_pipeline_config` — four packed `u32` enum fields.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ZtokPipelineConfig {
    pub normalizer: u32,
    pub pre_tokenizer: u32,
    pub model: u32,
    pub decoder: u32,
}

/// Mirrors `struct ztok_overlay_channel`. `out` is a caller-owned buffer
/// of `out_cap` `u32` entries, filled with one value per emitted token.
#[repr(C)]
pub struct ZtokOverlayChannel {
    pub kind: u32,
    pub out: *mut u32,
    pub out_cap: usize,
}

/// Mirrors `struct ztok_chunk_rec`. `ids` points at a ztok-allocated
/// buffer of `ids_len` token ids (NULL when `ids_len == 0`), released via
/// `ztok_chunks_free`. `byte_*` is the half-open byte range this chunk
/// covers in the ORIGINAL input; `token_*` the half-open token-index
/// range in the full encoding.
#[repr(C)]
pub struct ZtokChunkRec {
    pub ids: *mut u32,
    pub ids_len: usize,
    pub byte_start: u32,
    pub byte_end: u32,
    pub token_start: u32,
    pub token_end: u32,
}

// --- extern fn declarations ---------------------------------------------

extern "C" {
    // Lifecycle
    pub fn ztok_pipeline_new(
        cfg: *const ZtokPipelineConfig,
        out_status: *mut c_int,
    ) -> *mut ZtokPipeline;

    pub fn ztok_pipeline_free(p: *mut ZtokPipeline);

    pub fn ztok_pipeline_new_bpe_from_tiktoken(
        tiktoken_path: *const c_char,
        cfg_or_null: *const ZtokPipelineConfig,
        out_status: *mut c_int,
    ) -> *mut ZtokPipeline;

    pub fn ztok_pipeline_new_bpe_from_hf_json(
        tokenizer_json_path: *const c_char,
        cfg_or_null: *const ZtokPipelineConfig,
        out_status: *mut c_int,
    ) -> *mut ZtokPipeline;

    pub fn ztok_pipeline_new_wordpiece_from_hf_json(
        tokenizer_json_path: *const c_char,
        unk_id: u32,
        cfg_or_null: *const ZtokPipelineConfig,
        out_status: *mut c_int,
    ) -> *mut ZtokPipeline;

    pub fn ztok_pipeline_new_unigram_from_sp_model(
        sp_model_path: *const c_char,
        unk_id: u32,
        cfg_or_null: *const ZtokPipelineConfig,
        out_status: *mut c_int,
    ) -> *mut ZtokPipeline;

    pub fn ztok_pipeline_new_monster_from_file(
        path: *const c_char,
        cfg_or_null: *const ZtokPipelineConfig,
        out_status: *mut c_int,
    ) -> *mut ZtokPipeline;

    pub fn ztok_pipeline_new_rwkv_from_file(
        path: *const c_char,
        cfg_or_null: *const ZtokPipelineConfig,
        out_status: *mut c_int,
    ) -> *mut ZtokPipeline;

    // Encode / decode
    pub fn ztok_encode(
        p: *const ZtokPipeline,
        input: *const c_char,
        input_len: usize,
        out: *mut u32,
        out_cap: usize,
        out_len: *mut usize,
    ) -> c_int;

    pub fn ztok_decode(
        p: *const ZtokPipeline,
        ids: *const u32,
        ids_len: usize,
        out: *mut c_char,
        out_cap: usize,
        out_len: *mut usize,
    ) -> c_int;

    // Encode with overlay channels
    pub fn ztok_encode_with_overlays(
        p: *const ZtokPipeline,
        input: *const c_char,
        input_len: usize,
        out_ids: *mut u32,
        out_ids_cap: usize,
        channels: *mut ZtokOverlayChannel,
        n_channels: usize,
        out_len: *mut usize,
    ) -> c_int;

    // Batch encode
    pub fn ztok_encode_batch(
        p: *const ZtokPipeline,
        inputs: *const *const c_char,
        input_lens: *const usize,
        n: usize,
        out_ids: *mut *mut u32,
        out_lens: *mut usize,
        n_workers: u32,
    ) -> c_int;

    pub fn ztok_batch_pool_new(n_workers: u32, out_status: *mut c_int) -> *mut ZtokBatchPool;
    pub fn ztok_batch_pool_free(pool: *mut ZtokBatchPool);
    pub fn ztok_batch_pool_worker_count(pool: *const ZtokBatchPool) -> usize;

    pub fn ztok_encode_batch_pooled(
        p: *const ZtokPipeline,
        pool: *const ZtokBatchPool,
        inputs: *const *const c_char,
        input_lens: *const usize,
        n: usize,
        out_ids: *mut *mut u32,
        out_lens: *mut usize,
    ) -> c_int;

    pub fn ztok_ids_free(ids: *mut u32);

    // Auto-detect
    pub fn ztok_auto_detect(path: *const c_char) -> u32;

    // Streaming encode
    pub fn ztok_stream_new(p: *const ZtokPipeline, out_status: *mut c_int) -> *mut ZtokStream;
    pub fn ztok_stream_free(s: *mut ZtokStream);

    pub fn ztok_stream_feed(
        s: *mut ZtokStream,
        bytes: *const c_char,
        n_bytes: usize,
        out_ids: *mut *mut u32,
        out_n_ids: *mut usize,
    ) -> c_int;

    pub fn ztok_stream_finish(
        s: *mut ZtokStream,
        out_ids: *mut *mut u32,
        out_n_ids: *mut usize,
    ) -> c_int;

    // Engram n-gram hashing
    pub fn ztok_ngram_hash(
        ids: *const u32,
        n_ids: usize,
        n: u32,
        heads: u32,
        out: *mut u64,
        out_cap: usize,
        out_len: *mut usize,
    ) -> c_int;

    pub fn ztok_ngram_hash_batch(
        pool: *mut ZtokBatchPool,
        id_arrays: *const *const u32,
        id_lens: *const usize,
        n_docs: usize,
        n: u32,
        heads: u32,
        out_hashes: *mut *mut u64,
        out_lens: *mut usize,
    ) -> c_int;

    pub fn ztok_u64s_free(hashes: *mut u64);

    // Chunking
    pub fn ztok_chunk(
        pipeline: *const ZtokPipeline,
        text: *const c_char,
        text_len: usize,
        max_tokens: u32,
        overlap: u32,
        boundary: u32,
        out_chunks: *mut ZtokChunkRec,
        out_cap: usize,
        out_len: *mut usize,
    ) -> c_int;

    pub fn ztok_chunks_free(chunks: *mut ZtokChunkRec, n: usize);

    pub fn ztok_version() -> *const c_char;

    // Fingerprint (new in 1.22)
    pub fn ztok_fingerprint(handle: *mut ZtokPipeline, out_32: *mut u8) -> c_int;
}
