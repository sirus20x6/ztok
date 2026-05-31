# frozen_string_literal: true

require "ffi"

require_relative "errors"
require_relative "lib"

module Ztok
  # Mirrors include/ztok.h via the `ffi` gem. Every C function we touch
  # is declared with full `attach_function` signatures so calls fail
  # loudly on signature drift.
  #
  # Type map (FFI -> C):
  #   :uint32           -> ztok_token_id (uint32_t)
  #   :int              -> ztok_status   (enum, C int width)
  #   :pointer          -> ztok_pipeline* / ztok_batch_pool* / ztok_stream* (opaque)
  #   :string           -> const char*   (NUL-terminated)
  #   :pointer (bytes)  -> const char*   (when carrying possibly-NUL bytes)
  #   :size_t           -> size_t
  #
  # The id-buffer header lifecycle (CRITICAL):
  #   ztok_encode_batch_pooled and ztok_stream_{feed,finish} return each
  #   id buffer as a ztok_token_id* whose bytes carry an opaque length
  #   header (see src/c_api.zig::allocIdBuf). The ONLY safe free is
  #   ztok_ids_free. NEVER call ::FFI::MemoryPointer.free on these or
  #   pass them to libc free.
  module FFI
    extend ::FFI::Library

    # Enum values (mirror ztok_*_kind in ztok.h).
    NORMALIZER_IDENTITY   = 0
    NORMALIZER_NFC        = 1
    NORMALIZER_NFD        = 2
    NORMALIZER_NFKC       = 3
    NORMALIZER_NFKD       = 4
    NORMALIZER_BYTE_LEVEL = 5

    PRETOK_IDENTITY = 0
    PRETOK_CL100K   = 1
    PRETOK_TEKKEN   = 2

    MODEL_BYTE_ID = 0

    DECODER_CONCAT     = 0
    DECODER_WORDPIECE  = 1
    DECODER_BYTE_LEVEL = 2

    # Auto-detect format codes (mirror enum ztok_format in ztok.h).
    FORMAT_UNKNOWN  = 0
    FORMAT_TIKTOKEN = 1
    FORMAT_HF_JSON  = 2
    FORMAT_SP_MODEL = 3
    FORMAT_ZTM      = 4
    FORMAT_TEKKEN   = 5
    FORMAT_RWKV     = 6

    FORMAT_CODE_TO_NAME = {
      FORMAT_UNKNOWN  => :unknown,
      FORMAT_TIKTOKEN => :tiktoken,
      FORMAT_HF_JSON  => :hf_json,
      FORMAT_SP_MODEL => :sentencepiece,
      FORMAT_ZTM      => :ztm,
      FORMAT_TEKKEN   => :tekken,
      FORMAT_RWKV     => :rwkv,
    }.freeze

    # Chunk boundary modes (mirror ztok_chunk_boundary in ztok.h).
    CHUNK_BOUNDARY_TOKEN     = 0
    CHUNK_BOUNDARY_CODEPOINT = 1
    CHUNK_BOUNDARY_WORD      = 2
    CHUNK_BOUNDARY_WORD_DICT = 3
    CHUNK_BOUNDARY_SENTENCE  = 4
    CHUNK_BOUNDARY_PARAGRAPH = 5

    # Symbolic boundary names accepted by Pipeline#chunk's `boundary:` arg.
    CHUNK_BOUNDARY_NAME_TO_CODE = {
      token:     CHUNK_BOUNDARY_TOKEN,
      codepoint: CHUNK_BOUNDARY_CODEPOINT,
      word:      CHUNK_BOUNDARY_WORD,
      word_dict: CHUNK_BOUNDARY_WORD_DICT,
      sentence:  CHUNK_BOUNDARY_SENTENCE,
      paragraph: CHUNK_BOUNDARY_PARAGRAPH,
    }.freeze

    # Overlay channel kinds (mirror ztok_overlay_kind in ztok.h). Cheap
    # channels (BYTE_START/BYTE_END/BOUNDARY/PROVENANCE) carry
    # encoder-derived values; domain channels (OPCODE/OPERAND/SYMBOL_REF/
    # HUNK) come back zero-filled until a domain plugin populates them.
    OVERLAY_BYTE_START = 0
    OVERLAY_BYTE_END   = 1
    OVERLAY_BOUNDARY   = 2
    OVERLAY_OPCODE     = 3
    OVERLAY_OPERAND    = 4
    OVERLAY_SYMBOL_REF = 5
    OVERLAY_HUNK       = 6
    OVERLAY_PROVENANCE = 7
    OVERLAY_USER_BASE  = 0x8000

    # Struct mirroring `ztok_pipeline_config` (four packed uint32 fields).
    class PipelineConfig < ::FFI::Struct
      layout :normalizer,    :uint32,
             :pre_tokenizer, :uint32,
             :model,         :uint32,
             :decoder,       :uint32
    end

    # Struct mirroring `ztok_overlay_channel`. `out` is a caller-owned
    # uint32 buffer of `out_cap` entries, filled with one value per token.
    class OverlayChannel < ::FFI::Struct
      layout :kind,    :uint32,
             :out,     :pointer,
             :out_cap, :size_t
    end

    # Struct mirroring `ztok_chunk_rec` in ztok.h. `ids` is a
    # ztok-allocated buffer of `ids_len` token ids (NULL when ids_len ==
    # 0), released via ztok_chunks_free. `byte_*` is the half-open byte
    # range in the ORIGINAL input; `token_*` the half-open token-index
    # range in the full encoding.
    class ChunkRec < ::FFI::Struct
      layout :ids,         :pointer,
             :ids_len,     :size_t,
             :byte_start,  :uint32,
             :byte_end,    :uint32,
             :token_start, :uint32,
             :token_end,   :uint32
    end

    # Load the shared library once at require-time so signature errors
    # surface immediately. Lib.resolve raises Ztok::LibraryNotFoundError
    # with a descriptive message if the library can't be located.
    ffi_lib Ztok::Lib.resolve

    # --- lifecycle ----------------------------------------------------
    attach_function :ztok_pipeline_new,
                    [:pointer, :pointer], :pointer
    attach_function :ztok_pipeline_free, [:pointer], :void

    attach_function :ztok_pipeline_new_bpe_from_tiktoken,
                    [:string, :pointer, :pointer], :pointer
    attach_function :ztok_pipeline_new_bpe_from_hf_json,
                    [:string, :pointer, :pointer], :pointer
    attach_function :ztok_pipeline_new_wordpiece_from_hf_json,
                    [:string, :uint32, :pointer, :pointer], :pointer
    attach_function :ztok_pipeline_new_unigram_from_sp_model,
                    [:string, :uint32, :pointer, :pointer], :pointer
    attach_function :ztok_pipeline_new_monster_from_file,
                    [:string, :pointer, :pointer], :pointer
    attach_function :ztok_pipeline_new_rwkv_from_file,
                    [:string, :pointer, :pointer], :pointer
    attach_function :ztok_pipeline_new_tekken_from_file,
                    [:string, :pointer, :pointer], :pointer

    # --- encode / decode ---------------------------------------------
    attach_function :ztok_encode,
                    [:pointer, :pointer, :size_t, :pointer, :size_t, :pointer], :int
    attach_function :ztok_decode,
                    [:pointer, :pointer, :size_t, :pointer, :size_t, :pointer], :int

    # --- encode with overlay channels --------------------------------
    attach_function :ztok_encode_with_overlays,
                    [:pointer, :pointer, :size_t, :pointer, :size_t,
                     :pointer, :size_t, :pointer], :int

    # --- batch -------------------------------------------------------
    attach_function :ztok_encode_batch,
                    [:pointer, :pointer, :pointer, :size_t,
                     :pointer, :pointer, :uint32], :int

    attach_function :ztok_batch_pool_new,
                    [:uint32, :pointer], :pointer
    attach_function :ztok_batch_pool_free, [:pointer], :void
    attach_function :ztok_batch_pool_worker_count, [:pointer], :size_t

    attach_function :ztok_encode_batch_pooled,
                    [:pointer, :pointer, :pointer, :pointer, :size_t,
                     :pointer, :pointer], :int

    attach_function :ztok_ids_free, [:pointer], :void

    # --- n-gram hashing (Engram) -------------------------------------
    attach_function :ztok_ngram_hash,
                    [:pointer, :size_t, :uint32, :uint32,
                     :pointer, :size_t, :pointer], :int
    attach_function :ztok_ngram_hash_batch,
                    [:pointer, :pointer, :pointer, :size_t,
                     :uint32, :uint32, :pointer, :pointer], :int
    attach_function :ztok_u64s_free, [:pointer], :void

    # --- chunking ----------------------------------------------------
    attach_function :ztok_chunk,
                    [:pointer, :pointer, :size_t, :uint32, :uint32, :uint32,
                     :pointer, :size_t, :pointer], :int
    attach_function :ztok_chunks_free, [:pointer, :size_t], :void

    # --- fingerprint -------------------------------------------------
    # ztok_status ztok_fingerprint(ztok_pipeline*, uint8_t out_32[32]).
    # out_32 is a caller-owned 32-byte buffer ztok writes back into.
    attach_function :ztok_fingerprint, [:pointer, :pointer], :int

    # --- auto-detect -------------------------------------------------
    attach_function :ztok_auto_detect, [:string], :uint32

    # --- streaming ---------------------------------------------------
    attach_function :ztok_stream_new, [:pointer, :pointer], :pointer
    attach_function :ztok_stream_free, [:pointer], :void
    attach_function :ztok_stream_feed,
                    [:pointer, :pointer, :size_t, :pointer, :pointer], :int
    attach_function :ztok_stream_finish,
                    [:pointer, :pointer, :pointer], :int

    # --- version -----------------------------------------------------
    attach_function :ztok_version, [], :string

    # Translate a path through ztok_auto_detect and return the symbol
    # form ({:tiktoken, :hf_json, :sentencepiece, :ztm, :unknown}).
    def self.detect_format(path)
      code = ztok_auto_detect(path.to_s)
      FORMAT_CODE_TO_NAME[code] || :unknown
    end

    # Materialize a libztok-owned id buffer into a plain Ruby Array,
    # then call ztok_ids_free on it. Used by Pipeline#encode_batch and
    # the streaming path. The pointer carries a length-prefix header
    # (see src/c_api.zig::allocIdBuf) so the ONLY safe free is
    # ztok_ids_free — never wrap these in a finalizer that calls libc
    # free, and never deref them after this call returns.
    def self.materialize_and_free_ids(ptr, n)
      return [] if ptr.nil? || ptr.null? || n <= 0
      ids = ptr.read_array_of_uint32(n)
      ztok_ids_free(ptr)
      ids
    end
  end
end
