"""ctypes mirror of include/ztok.h.

Every `extern fn` from the C ABI is bound here with proper `argtypes` and
`restype` annotations so calls fail loudly on signature drift.

Type map:
    ztok_token_id      -> c_uint32
    ztok_status        -> c_int
    ztok_pipeline*     -> c_void_p (opaque)
    ztok_batch_pool*   -> c_void_p (opaque)
    const char*        -> c_char_p
    char*              -> POINTER(c_char) (mutable; encode/decode write into it)
    size_t / size_t*   -> c_size_t / POINTER(c_size_t)
    uint32_t           -> c_uint32

All numeric enums (normalizer/pretok/model/decoder kinds) are passed as
plain c_uint inside the ZtokPipelineConfig struct.

The id-buffer header lifecycle:
    `ztok_encode_batch_pooled` returns each per-input id array as a
    ztok_token_id* whose bytes carry an opaque length header (see
    src/c_api.zig). The ONLY safe free is `ztok_ids_free`.
"""

from __future__ import annotations

import ctypes
import os
from ctypes import (
    POINTER,
    Structure,
    c_char,
    c_char_p,
    c_int,
    c_size_t,
    c_uint,
    c_uint32,
    c_uint64,
    c_void_p,
)


# Status codes (mirrors enum ztok_status in ztok.h).
ZTOK_OK = 0
ZTOK_ERR_OUT_OF_MEMORY = 1
ZTOK_ERR_INVALID_INPUT = 2
ZTOK_ERR_BUFFER_TOO_SMALL = 3
ZTOK_ERR_INTERNAL = 99

# Enum kinds (mirrors ztok_*_kind enums in ztok.h).
NORMALIZER_IDENTITY = 0
NORMALIZER_NFC = 1
NORMALIZER_NFD = 2
NORMALIZER_NFKC = 3
NORMALIZER_NFKD = 4
NORMALIZER_BYTE_LEVEL = 5

PRETOK_IDENTITY = 0
PRETOK_CL100K = 1

MODEL_BYTE_ID = 0

DECODER_CONCAT = 0
DECODER_WORDPIECE = 1
DECODER_BYTE_LEVEL = 2

# Auto-detect format codes (mirrors `ztok_format` in ztok.h).
FORMAT_UNKNOWN = 0
FORMAT_TIKTOKEN = 1
FORMAT_HF_JSON = 2
FORMAT_SP_MODEL = 3
FORMAT_ZTM = 4
FORMAT_TEKKEN = 5
FORMAT_RWKV = 6

# Chunk boundary modes (mirrors `ztok_chunk_boundary` in ztok.h).
CHUNK_BOUNDARY_TOKEN = 0
CHUNK_BOUNDARY_CODEPOINT = 1
CHUNK_BOUNDARY_WORD = 2
CHUNK_BOUNDARY_WORD_DICT = 3
CHUNK_BOUNDARY_SENTENCE = 4
CHUNK_BOUNDARY_PARAGRAPH = 5

# Overlay channel kinds (mirrors `ztok_overlay_kind` in ztok.h). Cheap
# channels are derived from encoder state; domain channels are zero-filled
# until a domain normalizer plugin populates them.
OVERLAY_BYTE_START = 0
OVERLAY_BYTE_END = 1
OVERLAY_BOUNDARY = 2
OVERLAY_OPCODE = 3
OVERLAY_OPERAND = 4
OVERLAY_SYMBOL_REF = 5
OVERLAY_HUNK = 6
OVERLAY_PROVENANCE = 7
OVERLAY_USER_BASE = 0x8000


# Typedefs.
TokenId = c_uint32
TokenIdPtr = POINTER(TokenId)


class ZtokPipelineConfig(Structure):
    """Mirrors `struct ztok_pipeline_config` (4 packed c_uint fields)."""

    _fields_ = [
        ("normalizer", c_uint),
        ("pre_tokenizer", c_uint),
        ("model", c_uint),
        ("decoder", c_uint),
    ]


class ZtokChunkRec(Structure):
    """Mirrors `struct ztok_chunk_rec` in ztok.h.

    `ids` is a ztok-allocated buffer of `ids_len` token ids (NULL when
    ids_len == 0), released via ztok_chunks_free. `byte_*` is the
    half-open byte range in the ORIGINAL input; `token_*` the half-open
    token-index range in the full encoding.
    """

    _fields_ = [
        ("ids", TokenIdPtr),
        ("ids_len", c_size_t),
        ("byte_start", c_uint32),
        ("byte_end", c_uint32),
        ("token_start", c_uint32),
        ("token_end", c_uint32),
    ]


class ZtokOverlayChannel(Structure):
    """Mirrors `struct ztok_overlay_channel` in ztok.h.

    `out` is a caller-owned uint32_t buffer of `out_cap` entries, filled
    with one value per emitted token.
    """

    _fields_ = [
        ("kind", c_uint),
        ("out", TokenIdPtr),
        ("out_cap", c_size_t),
    ]


def bind(lib: ctypes.CDLL) -> ctypes.CDLL:
    """Attach restype/argtypes to every exported function.

    Idempotent: safe to call twice on the same handle.
    """

    # --- lifecycle --------------------------------------------------------
    lib.ztok_pipeline_new.argtypes = [POINTER(ZtokPipelineConfig), POINTER(c_int)]
    lib.ztok_pipeline_new.restype = c_void_p

    lib.ztok_pipeline_free.argtypes = [c_void_p]
    lib.ztok_pipeline_free.restype = None

    # File-based constructors.
    for fname in (
        "ztok_pipeline_new_bpe_from_tiktoken",
        "ztok_pipeline_new_bpe_from_hf_json",
        "ztok_pipeline_new_monster_from_file",
        "ztok_pipeline_new_rwkv_from_file",
    ):
        fn = getattr(lib, fname)
        fn.argtypes = [c_char_p, POINTER(ZtokPipelineConfig), POINTER(c_int)]
        fn.restype = c_void_p

    for fname in (
        "ztok_pipeline_new_wordpiece_from_hf_json",
        "ztok_pipeline_new_unigram_from_sp_model",
    ):
        fn = getattr(lib, fname)
        fn.argtypes = [c_char_p, c_uint32, POINTER(ZtokPipelineConfig), POINTER(c_int)]
        fn.restype = c_void_p

    # --- encode / decode --------------------------------------------------
    lib.ztok_encode.argtypes = [
        c_void_p,           # pipeline*
        c_char_p,           # input
        c_size_t,           # input_len
        TokenIdPtr,         # out (nullable)
        c_size_t,           # out_cap
        POINTER(c_size_t),  # out_len
    ]
    lib.ztok_encode.restype = c_int

    lib.ztok_decode.argtypes = [
        c_void_p,           # pipeline*
        TokenIdPtr,         # ids
        c_size_t,           # ids_len
        POINTER(c_char),    # out (nullable for sizing)
        c_size_t,           # out_cap
        POINTER(c_size_t),  # out_len
    ]
    lib.ztok_decode.restype = c_int

    # --- encode with overlay channels ------------------------------------
    lib.ztok_encode_with_overlays.argtypes = [
        c_void_p,                       # pipeline*
        c_char_p,                       # input
        c_size_t,                       # input_len
        TokenIdPtr,                     # out_ids (nullable for sizing)
        c_size_t,                       # out_ids_cap
        POINTER(ZtokOverlayChannel),    # channels[]
        c_size_t,                       # n_channels
        POINTER(c_size_t),              # out_len
    ]
    lib.ztok_encode_with_overlays.restype = c_int

    # --- batch ------------------------------------------------------------
    lib.ztok_encode_batch.argtypes = [
        c_void_p,                   # pipeline*
        POINTER(c_char_p),          # inputs[]
        POINTER(c_size_t),          # input_lens[]
        c_size_t,                   # n
        POINTER(c_void_p),          # out_ids[] (each is ztok_token_id*)
        POINTER(c_size_t),          # out_lens[]
        c_uint32,                   # n_workers
    ]
    lib.ztok_encode_batch.restype = c_int

    lib.ztok_batch_pool_new.argtypes = [c_uint32, POINTER(c_int)]
    lib.ztok_batch_pool_new.restype = c_void_p

    lib.ztok_batch_pool_free.argtypes = [c_void_p]
    lib.ztok_batch_pool_free.restype = None

    lib.ztok_batch_pool_worker_count.argtypes = [c_void_p]
    lib.ztok_batch_pool_worker_count.restype = c_size_t

    lib.ztok_encode_batch_pooled.argtypes = [
        c_void_p,                   # pipeline*
        c_void_p,                   # batch_pool*
        POINTER(c_char_p),          # inputs[]
        POINTER(c_size_t),          # input_lens[]
        c_size_t,                   # n
        POINTER(c_void_p),          # out_ids[]
        POINTER(c_size_t),          # out_lens[]
    ]
    lib.ztok_encode_batch_pooled.restype = c_int

    lib.ztok_ids_free.argtypes = [c_void_p]
    lib.ztok_ids_free.restype = None

    lib.ztok_version.argtypes = []
    lib.ztok_version.restype = c_char_p

    # --- n-gram hashing (Engram) -----------------------------------------
    lib.ztok_ngram_hash.argtypes = [
        TokenIdPtr,             # ids
        c_size_t,               # n_ids
        c_uint32,               # n (window length)
        c_uint32,               # heads
        POINTER(c_uint64),      # out
        c_size_t,               # out_cap
        POINTER(c_size_t),      # out_len
    ]
    lib.ztok_ngram_hash.restype = c_int

    lib.ztok_ngram_hash_batch.argtypes = [
        c_void_p,                   # batch_pool*
        POINTER(TokenIdPtr),        # id_arrays[]
        POINTER(c_size_t),          # id_lens[]
        c_size_t,                   # n_docs
        c_uint32,                   # n
        c_uint32,                   # heads
        POINTER(POINTER(c_uint64)), # out_hashes[]
        POINTER(c_size_t),          # out_lens[]
    ]
    lib.ztok_ngram_hash_batch.restype = c_int

    lib.ztok_u64s_free.argtypes = [POINTER(c_uint64)]
    lib.ztok_u64s_free.restype = None

    # --- chunking ---------------------------------------------------------
    lib.ztok_chunk.argtypes = [
        c_void_p,                   # pipeline*
        c_char_p,                   # text
        c_size_t,                   # text_len
        c_uint32,                   # max_tokens
        c_uint32,                   # overlap
        c_uint32,                   # boundary
        POINTER(ZtokChunkRec),      # out_chunks (nullable for sizing)
        c_size_t,                   # out_cap
        POINTER(c_size_t),          # out_len
    ]
    lib.ztok_chunk.restype = c_int

    lib.ztok_chunks_free.argtypes = [POINTER(ZtokChunkRec), c_size_t]
    lib.ztok_chunks_free.restype = None

    # --- auto-detect ------------------------------------------------------
    lib.ztok_auto_detect.argtypes = [c_char_p]
    lib.ztok_auto_detect.restype = c_uint  # ztok_format enum (c_uint)

    # --- streaming encode ------------------------------------------------
    lib.ztok_stream_new.argtypes = [c_void_p, POINTER(c_int)]
    lib.ztok_stream_new.restype = c_void_p

    lib.ztok_stream_free.argtypes = [c_void_p]
    lib.ztok_stream_free.restype = None

    lib.ztok_stream_feed.argtypes = [
        c_void_p,            # stream*
        c_char_p,            # bytes
        c_size_t,            # n_bytes
        POINTER(c_void_p),   # out_ids (each is ztok_token_id*)
        POINTER(c_size_t),   # out_n_ids
    ]
    lib.ztok_stream_feed.restype = c_int

    lib.ztok_stream_finish.argtypes = [
        c_void_p,
        POINTER(c_void_p),
        POINTER(c_size_t),
    ]
    lib.ztok_stream_finish.restype = c_int

    return lib


# Format-code -> existing _detect_format string mapping. Keeps the
# back-compat callers (test_loaders.py) happy even though we now route
# detection through the C ABI.
_FORMAT_CODE_TO_NAME = {
    FORMAT_UNKNOWN: "unknown",
    FORMAT_TIKTOKEN: "tiktoken",
    FORMAT_HF_JSON: "hf_json",
    FORMAT_SP_MODEL: "sentencepiece",
    FORMAT_ZTM: "ztm",
    FORMAT_TEKKEN: "tekken",
    FORMAT_RWKV: "rwkv",
}


def ztok_auto_detect(lib: ctypes.CDLL, path: str) -> str:
    """Call ztok_auto_detect and translate the int code to the legacy
    string label used elsewhere in the binding."""

    code = int(lib.ztok_auto_detect(os.fsencode(path)))
    return _FORMAT_CODE_TO_NAME.get(code, "unknown")
