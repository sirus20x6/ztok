# frozen_string_literal: true

require_relative "ffi"
require_relative "errors"
require_relative "batch_pool"
require_relative "stream_encoder"

module Ztok
  # A loaded tokenizer pipeline. Construct via one of the `from_*`
  # class methods rather than `Pipeline.new` directly. Always close
  # (via `Pipeline.open` block form or an explicit `close`) when done;
  # the underlying library owns vocab tables that can be tens of MB.
  #
  # Usage:
  #
  #   pipe = Ztok::Pipeline.from_path("tokenizer.json")
  #   ids  = pipe.encode("hello world")
  #   text = pipe.decode(ids)
  #   pipe.close
  #
  # Or the block form:
  #
  #   Ztok::Pipeline.from_path("tokenizer.json") do |pipe|
  #     ids = pipe.encode("hello world")
  #   end
  # One token window produced by Pipeline#chunk. `ids` are the token ids
  # in this chunk. `byte_start`/`byte_end` is the half-open byte range the
  # chunk covers in the ORIGINAL input; `token_start`/`token_end` the
  # half-open token-index range in the full encoding.
  Chunk = Struct.new(:ids, :byte_start, :byte_end, :token_start, :token_end,
                     keyword_init: true)

  class Pipeline
    # Default chunk size for streaming encode (mirrors Python's 64 KiB).
    STREAM_FEED_SIZE = 64 * 1024

    # @api private — use the from_* class methods.
    def initialize(handle)
      if handle.nil? || handle.null?
        raise InternalError, "Pipeline constructed with NULL handle"
      end
      @handle = handle
      @closed = false
      ObjectSpace.define_finalizer(self, self.class.finalizer(handle.address))
    end

    # --- constructors ----------------------------------------------------

    # The byte_id baseline pipeline (each input byte maps to its own id).
    # @return [Ztok::Pipeline]
    def self.byte_id(normalizer: FFI::NORMALIZER_IDENTITY,
                     pre_tokenizer: FFI::PRETOK_IDENTITY,
                     decoder: FFI::DECODER_CONCAT,
                     &block)
      cfg = make_config(normalizer, pre_tokenizer, FFI::MODEL_BYTE_ID, decoder)
      status_buf = ::FFI::MemoryPointer.new(:int)
      handle = FFI.ztok_pipeline_new(cfg, status_buf)
      Ztok.raise_for_status(status_buf.read_int, "ztok_pipeline_new")
      wrap_or_yield(new(handle), &block)
    end

    # Load a .tiktoken vocab into a byte-level BPE pipeline. cl100k=true
    # (the default) is the right choice for OpenAI cl100k_base.
    def self.from_tiktoken(path,
                           cl100k: true,
                           normalizer: FFI::NORMALIZER_IDENTITY,
                           decoder: FFI::DECODER_CONCAT,
                           &block)
      pre = cl100k ? FFI::PRETOK_CL100K : FFI::PRETOK_IDENTITY
      pipe = from_file_with_cfg(:ztok_pipeline_new_bpe_from_tiktoken, path,
                                normalizer: normalizer,
                                pre_tokenizer: pre,
                                decoder: decoder)
      wrap_or_yield(pipe, &block)
    end

    # Load a HuggingFace tokenizer.json BPE model.
    def self.from_hf_json(path,
                          cl100k: false,
                          normalizer: FFI::NORMALIZER_IDENTITY,
                          decoder: FFI::DECODER_CONCAT,
                          &block)
      pre = cl100k ? FFI::PRETOK_CL100K : FFI::PRETOK_IDENTITY
      pipe = from_file_with_cfg(:ztok_pipeline_new_bpe_from_hf_json, path,
                                normalizer: normalizer,
                                pre_tokenizer: pre,
                                decoder: decoder)
      wrap_or_yield(pipe, &block)
    end

    # Load a HuggingFace WordPiece model from tokenizer.json. unk_id is
    # required (no sensible default — every WP vocab disagrees).
    def self.from_wordpiece(path, unk_id:,
                            normalizer: FFI::NORMALIZER_IDENTITY,
                            pre_tokenizer: FFI::PRETOK_IDENTITY,
                            decoder: FFI::DECODER_WORDPIECE,
                            &block)
      cfg = make_config(normalizer, pre_tokenizer, FFI::MODEL_BYTE_ID, decoder)
      status_buf = ::FFI::MemoryPointer.new(:int)
      handle = FFI.ztok_pipeline_new_wordpiece_from_hf_json(
        path.to_s, unk_id, cfg, status_buf
      )
      Ztok.raise_for_status(status_buf.read_int,
                            "ztok_pipeline_new_wordpiece_from_hf_json")
      wrap_or_yield(new(handle), &block)
    end

    # Load a SentencePiece .model (Unigram) file. unk_id defaults to 0
    # (the convention used by LLaMA-2, T5, Gemma, ...).
    def self.from_sentencepiece(path, unk_id: 0,
                                normalizer: FFI::NORMALIZER_IDENTITY,
                                pre_tokenizer: FFI::PRETOK_IDENTITY,
                                decoder: FFI::DECODER_CONCAT,
                                &block)
      cfg = make_config(normalizer, pre_tokenizer, FFI::MODEL_BYTE_ID, decoder)
      status_buf = ::FFI::MemoryPointer.new(:int)
      handle = FFI.ztok_pipeline_new_unigram_from_sp_model(
        path.to_s, unk_id, cfg, status_buf
      )
      Ztok.raise_for_status(status_buf.read_int,
                            "ztok_pipeline_new_unigram_from_sp_model")
      wrap_or_yield(new(handle), &block)
    end

    # Load a ztok TokenMonster .ztm vocab file.
    def self.from_monster(path,
                          normalizer: FFI::NORMALIZER_IDENTITY,
                          pre_tokenizer: FFI::PRETOK_IDENTITY,
                          decoder: FFI::DECODER_CONCAT,
                          &block)
      pipe = from_file_with_cfg(:ztok_pipeline_new_monster_from_file, path,
                                normalizer: normalizer,
                                pre_tokenizer: pre_tokenizer,
                                decoder: decoder)
      wrap_or_yield(pipe, &block)
    end

    # Load an RWKV "World" vocab (rwkv_vocab_v20230424.txt). The model is
    # a greedy longest-match byte trie that runs over the whole input as a
    # single span — there is no pre-tokenizer, so the pre_tokenizer slot is
    # fixed to identity.
    def self.from_rwkv(path,
                       normalizer: FFI::NORMALIZER_IDENTITY,
                       decoder: FFI::DECODER_CONCAT,
                       &block)
      pipe = from_file_with_cfg(:ztok_pipeline_new_rwkv_from_file, path,
                                normalizer: normalizer,
                                pre_tokenizer: FFI::PRETOK_IDENTITY,
                                decoder: decoder)
      wrap_or_yield(pipe, &block)
    end

    # Load a Mistral Tekken tekken.json vocab (Nemo / Pixtral / Devstral /
    # Magistral, etc.). The loader lowers Tekken's base64 byte vocab into a
    # BPE with the special tokens packed into the bottom of the id space.
    # The default pre-tokenizer is the Tekken pattern (PRETOK_TEKKEN) — NOT
    # cl100k — and the default decoder is concat (pieces are raw bytes).
    def self.from_tekken(path,
                         normalizer: FFI::NORMALIZER_IDENTITY,
                         pre_tokenizer: FFI::PRETOK_TEKKEN,
                         decoder: FFI::DECODER_CONCAT,
                         &block)
      pipe = from_file_with_cfg(:ztok_pipeline_new_tekken_from_file, path,
                                normalizer: normalizer,
                                pre_tokenizer: pre_tokenizer,
                                decoder: decoder)
      wrap_or_yield(pipe, &block)
    end

    # Auto-detect the file format and dispatch to the right loader:
    #
    #   .tiktoken            -> BPE + cl100k pre-tokenizer
    #   tokenizer.json (HF)  -> BPE
    #   .model (SentencePiece) -> Unigram (unk_id defaults to 0)
    #   .ztm  (TokenMonster) -> Monster
    #
    # Routes through the C ABI's `ztok_auto_detect` — no Ruby-side
    # magic-byte mirror.
    def self.from_path(path, unk_id: 0,
                       normalizer: FFI::NORMALIZER_IDENTITY,
                       decoder: nil,
                       &block)
      fmt = FFI.detect_format(path)
      pipe =
        case fmt
        when :tiktoken
          from_tiktoken(path, normalizer: normalizer,
                              decoder: decoder || FFI::DECODER_CONCAT)
        when :hf_json
          from_hf_json(path, normalizer: normalizer,
                             decoder: decoder || FFI::DECODER_CONCAT)
        when :sentencepiece
          from_sentencepiece(path, unk_id: unk_id, normalizer: normalizer,
                                   decoder: decoder || FFI::DECODER_CONCAT)
        when :ztm
          from_monster(path, normalizer: normalizer,
                             decoder: decoder || FFI::DECODER_CONCAT)
        when :rwkv
          from_rwkv(path, normalizer: normalizer,
                          decoder: decoder || FFI::DECODER_CONCAT)
        when :tekken
          from_tekken(path, normalizer: normalizer,
                            decoder: decoder || FFI::DECODER_CONCAT)
        else
          raise InvalidInputError,
                "could not auto-detect tokenizer format for #{path.inspect}; " \
                "use a specific from_* constructor instead"
        end
      wrap_or_yield(pipe, &block)
    end

    # --- lifecycle ------------------------------------------------------

    # Free the underlying pipeline. Safe to call multiple times.
    def close
      return if @closed
      @closed = true
      ObjectSpace.undefine_finalizer(self)
      FFI.ztok_pipeline_free(@handle)
      @handle = nil
    end

    def closed?
      @closed
    end

    def raw
      check_open!
      @handle
    end

    # --- encode / decode -----------------------------------------------

    # Encode a string (UTF-8) into an Array<Integer> of token ids.
    # Bytes-input is also accepted (anything responding to `to_str`).
    def encode(text)
      check_open!
      data = coerce_bytes(text)
      return [] if data.bytesize.zero?

      input_buf = ::FFI::MemoryPointer.new(:uint8, data.bytesize)
      input_buf.write_bytes(data)

      # The C ABI's per-span maxTokensFor bound is conservative, so a
      # buffer sized to the encoded length can still trip
      # BUFFER_TOO_SMALL mid-stream. Start generous, grow on demand.
      cap = [data.bytesize + 16, 64].max
      8.times do
        out_buf = ::FFI::MemoryPointer.new(:uint32, cap)
        out_len_buf = ::FFI::MemoryPointer.new(:size_t)
        rc = FFI.ztok_encode(@handle, input_buf, data.bytesize,
                             out_buf, cap, out_len_buf)
        if rc == STATUS_OK
          n = out_len_buf.read(:size_t)
          return out_buf.read_array_of_uint32(n)
        end
        if rc == STATUS_ERR_BUFFER_TOO_SMALL
          required = out_len_buf.read(:size_t)
          cap = [cap * 2, required + 16].max
          next
        end
        Ztok.raise_for_status(rc, "ztok_encode")
      end
      raise InternalError,
            "ztok_encode kept reporting BUFFER_TOO_SMALL after 8 grow attempts"
    end

    # Decode a sequence of token ids (Array<Integer> or anything
    # iterable) into a UTF-8 String. Invalid sequences are replaced
    # with U+FFFD.
    def decode(ids)
      bytes = decode_bytes(ids)
      bytes.force_encoding(Encoding::UTF_8)
      bytes.valid_encoding? ? bytes : bytes.scrub
    end

    # Decode token ids into raw bytes (ASCII-8BIT) without UTF-8
    # round-tripping. Useful if the vocab encodes arbitrary byte data.
    def decode_bytes(ids)
      check_open!
      arr = ids.respond_to?(:to_a) ? ids.to_a : ids
      n = arr.length
      return String.new(encoding: Encoding::ASCII_8BIT) if n.zero?

      id_buf = ::FFI::MemoryPointer.new(:uint32, n)
      id_buf.write_array_of_uint32(arr)

      # Sizing pass: pass nullptr + cap 0 and accept BUFFER_TOO_SMALL.
      out_len_buf = ::FFI::MemoryPointer.new(:size_t)
      rc = FFI.ztok_decode(@handle, id_buf, n,
                           ::FFI::Pointer::NULL, 0, out_len_buf)
      if rc != STATUS_OK && rc != STATUS_ERR_BUFFER_TOO_SMALL
        Ztok.raise_for_status(rc, "ztok_decode (sizing)")
      end
      nbytes = out_len_buf.read(:size_t)
      return String.new(encoding: Encoding::ASCII_8BIT) if nbytes.zero?

      out_buf = ::FFI::MemoryPointer.new(:uint8, nbytes)
      rc = FFI.ztok_decode(@handle, id_buf, n, out_buf, nbytes, out_len_buf)
      Ztok.raise_for_status(rc, "ztok_decode")
      written = out_len_buf.read(:size_t)
      out_buf.read_bytes(written)
    end

    # --- encode with overlays ------------------------------------------

    # Encode +text+ and return [ids, overlays] where +overlays+ is a Hash
    # mapping each requested overlay kind (the FFI::OVERLAY_* constants)
    # to an Array<Integer> of per-token values, one per id (so every
    # array has ids.length entries).
    #
    # Requesting overlays never changes tokenization — +ids+ is identical
    # to what #encode returns. Cheap channels (BYTE_START/BYTE_END/
    # BOUNDARY/PROVENANCE) carry encoder-derived values; domain channels
    # (OPCODE/OPERAND/SYMBOL_REF/HUNK) come back zero-filled until a
    # domain plugin populates them.
    #
    # Mirrors the C ABI sizing protocol: a first call with out_ids=NULL
    # queries the token count, then buffers are allocated and a second
    # call fills them.
    def encode_with_overlays(text, channels)
      check_open!
      kinds = channels.to_a
      if kinds.uniq.length != kinds.length
        raise InvalidInputError, "encode_with_overlays: duplicate overlay kinds requested"
      end

      data = coerce_bytes(text)
      if data.bytesize.zero?
        return [[], kinds.each_with_object({}) { |k, h| h[k] = [] }]
      end

      input_buf = ::FFI::MemoryPointer.new(:uint8, data.bytesize)
      input_buf.write_bytes(data)
      n_ch = kinds.length

      # Build the channels[] struct array. For the sizing pass `out` is
      # NULL / cap 0; for the fill pass each channel points at its own
      # uint32 buffer.
      build_channels = lambda do |bufs|
        next ::FFI::Pointer::NULL if n_ch.zero?

        arr = ::FFI::MemoryPointer.new(FFI::OverlayChannel, n_ch)
        kinds.each_with_index do |kind, i|
          ch = FFI::OverlayChannel.new(arr + i * FFI::OverlayChannel.size)
          ch[:kind] = kind
          if bufs.nil?
            ch[:out] = ::FFI::Pointer::NULL
            ch[:out_cap] = 0
          else
            ch[:out] = bufs[i]
            ch[:out_cap] = bufs[i].size / ::FFI.type_size(:uint32)
          end
        end
        arr
      end

      # Sizing pass: out_ids = NULL queries the token count.
      size_chans = build_channels.call(nil)
      out_len_buf = ::FFI::MemoryPointer.new(:size_t)
      rc = FFI.ztok_encode_with_overlays(@handle, input_buf, data.bytesize,
                                         ::FFI::Pointer::NULL, 0,
                                         size_chans, n_ch, out_len_buf)
      if rc != STATUS_OK && rc != STATUS_ERR_BUFFER_TOO_SMALL
        Ztok.raise_for_status(rc, "ztok_encode_with_overlays (sizing)")
      end

      count = out_len_buf.read(:size_t)
      if count.zero?
        return [[], kinds.each_with_object({}) { |k, h| h[k] = [] }]
      end

      # Fill pass: allocate the id buffer + one uint32 buffer per channel,
      # each sized to the exact token count from the sizing pass.
      id_buf = ::FFI::MemoryPointer.new(:uint32, count)
      chan_bufs = Array.new(n_ch) { ::FFI::MemoryPointer.new(:uint32, count) }
      fill_chans = build_channels.call(chan_bufs)
      out_len_buf = ::FFI::MemoryPointer.new(:size_t)
      rc = FFI.ztok_encode_with_overlays(@handle, input_buf, data.bytesize,
                                         id_buf, count,
                                         fill_chans, n_ch, out_len_buf)
      Ztok.raise_for_status(rc, "ztok_encode_with_overlays")

      n = out_len_buf.read(:size_t)
      ids = id_buf.read_array_of_uint32(n)
      overlays = {}
      kinds.each_with_index do |kind, i|
        overlays[kind] = chan_bufs[i].read_array_of_uint32(n)
      end
      [ids, overlays]
    end

    # --- batch ----------------------------------------------------------

    # Encode many strings in parallel via a persistent BatchPool. Each
    # per-input id array is materialized into a Ruby Array, then the
    # C-owned buffer is freed via ztok_ids_free (the only safe path).
    #
    # Returns an Array<Array<Integer>>.
    def encode_batch(pool, inputs)
      check_open!
      raise TypeError, "encode_batch: first arg must be a Ztok::BatchPool" unless pool.is_a?(BatchPool)

      arr = inputs.to_a
      n = arr.length
      return [] if n.zero?

      # Pre-encode each input and stash the bytes so we can both compute
      # exact lengths and keep the buffers alive across the C call.
      byte_inputs = arr.map { |s| coerce_bytes(s) }

      # Build a NUL-terminated MemoryPointer per input. We must hang on
      # to the input_ptrs array — if it goes out of scope before the
      # call finishes, Ruby's GC frees the C buffers underneath us.
      input_ptrs = byte_inputs.map do |b|
        mp = ::FFI::MemoryPointer.new(:uint8, b.bytesize + 1)
        mp.write_bytes(b) unless b.bytesize.zero?
        mp.put_uint8(b.bytesize, 0) # explicit NUL terminator
        mp
      end

      inputs_arr = ::FFI::MemoryPointer.new(:pointer, n)
      inputs_arr.write_array_of_pointer(input_ptrs)

      lens_arr = ::FFI::MemoryPointer.new(:size_t, n)
      byte_inputs.each_with_index { |b, i| lens_arr[i].write(:size_t, b.bytesize) }

      out_ids_arr  = ::FFI::MemoryPointer.new(:pointer, n)
      out_lens_arr = ::FFI::MemoryPointer.new(:size_t, n)

      rc = FFI.ztok_encode_batch_pooled(@handle, pool.raw,
                                        inputs_arr, lens_arr, n,
                                        out_ids_arr, out_lens_arr)

      begin
        Ztok.raise_for_status(rc, "ztok_encode_batch_pooled")

        out_ptrs = out_ids_arr.read_array_of_pointer(n)
        results = Array.new(n)
        n.times do |i|
          length = out_lens_arr[i].read(:size_t)
          ptr = out_ptrs[i]
          results[i] =
            if ptr.nil? || ptr.null? || length.zero?
              []
            else
              ptr.read_array_of_uint32(length)
            end
        end
        results
      ensure
        # Free every per-input id buffer through the C ABI. This is the
        # ONLY safe free path — the buffers carry a length-prefix header
        # (see src/c_api.zig::allocIdBuf).
        out_ptrs ||= out_ids_arr.read_array_of_pointer(n)
        out_ptrs.each do |p|
          FFI.ztok_ids_free(p) if p && !p.null?
        end
      end
    end

    # --- chunking -------------------------------------------------------

    # Split +text+ into overlapping token windows (late chunking). Each
    # window holds at most +max_tokens+ ids with +overlap+ ids shared
    # between neighbors (stride = max_tokens - overlap). +boundary+ is a
    # symbol from FFI::CHUNK_BOUNDARY_NAME_TO_CODE (:token, :codepoint,
    # :word, :word_dict, :sentence, :paragraph) or an integer boundary
    # code. Returns an empty Array for empty input. The C-owned id buffers
    # are copied into Ruby Chunk objects and freed before returning.
    def chunk(text, max_tokens:, overlap: 0, boundary: :token)
      check_open!

      boundary_code =
        if boundary.is_a?(Symbol)
          FFI::CHUNK_BOUNDARY_NAME_TO_CODE.fetch(boundary) do
            raise InvalidInputError, "chunk: unknown boundary #{boundary.inspect}"
          end
        else
          boundary.to_i
        end

      if max_tokens.to_i <= 0 || overlap.to_i >= max_tokens.to_i
        raise InvalidInputError,
              "chunk: max_tokens must be > 0 and overlap < max_tokens"
      end

      data = coerce_bytes(text)
      return [] if data.bytesize.zero?

      input_buf = ::FFI::MemoryPointer.new(:uint8, data.bytesize)
      input_buf.write_bytes(data)

      # Sizing pass: out_chunks = NULL -> *out_len = chunk count.
      out_len_buf = ::FFI::MemoryPointer.new(:size_t)
      rc = FFI.ztok_chunk(@handle, input_buf, data.bytesize,
                          max_tokens, overlap, boundary_code,
                          ::FFI::Pointer::NULL, 0, out_len_buf)
      if rc != STATUS_OK && rc != STATUS_ERR_BUFFER_TOO_SMALL
        Ztok.raise_for_status(rc, "ztok_chunk (sizing)")
      end
      count = out_len_buf.read(:size_t)
      return [] if count.zero?

      # Fill pass: caller-owned record array.
      recs = ::FFI::MemoryPointer.new(FFI::ChunkRec, count)
      out_len_buf = ::FFI::MemoryPointer.new(:size_t)
      rc = FFI.ztok_chunk(@handle, input_buf, data.bytesize,
                          max_tokens, overlap, boundary_code,
                          recs, count, out_len_buf)

      begin
        Ztok.raise_for_status(rc, "ztok_chunk")
        got = out_len_buf.read(:size_t)
        Array.new(got) do |i|
          r = FFI::ChunkRec.new(recs + i * FFI::ChunkRec.size)
          ids_ptr = r[:ids]
          ids_len = r[:ids_len]
          ids =
            if ids_ptr.nil? || ids_ptr.null? || ids_len.zero?
              []
            else
              ids_ptr.read_array_of_uint32(ids_len)
            end
          Chunk.new(ids: ids,
                    byte_start: r[:byte_start], byte_end: r[:byte_end],
                    token_start: r[:token_start], token_end: r[:token_end])
        end
      ensure
        # Release each record's ztok-allocated id buffer. The recs array
        # itself is Ruby-owned (an FFI::MemoryPointer).
        FFI.ztok_chunks_free(recs, count)
      end
    end

    # --- streaming ------------------------------------------------------

    # Stream-encode +text+, yielding an Array<Integer> of ids as each
    # feed produces new tokens. A trailing flush via ztok_stream_finish
    # drains the encoder's carry; empty yields are skipped so callers
    # only see non-empty chunks.
    #
    # Returns an Enumerator if no block is given (so callers can
    # `.to_a`, `.each_with_index`, etc.).
    def encode_stream(text, chunk_size: STREAM_FEED_SIZE)
      return enum_for(:encode_stream, text, chunk_size: chunk_size) unless block_given?

      check_open!
      raise ArgumentError, "chunk_size must be > 0" unless chunk_size.positive?

      data = coerce_bytes(text)

      status_buf = ::FFI::MemoryPointer.new(:int)
      stream_handle = FFI.ztok_stream_new(@handle, status_buf)
      Ztok.raise_for_status(status_buf.read_int, "ztok_stream_new")
      raise InternalError, "ztok_stream_new returned NULL" if stream_handle.null?

      begin
        offset = 0
        while offset < data.bytesize
          take = [chunk_size, data.bytesize - offset].min
          chunk = data.byteslice(offset, take)
          chunk_buf = ::FFI::MemoryPointer.new(:uint8, take)
          chunk_buf.write_bytes(chunk)

          out_ids_buf = ::FFI::MemoryPointer.new(:pointer)
          out_n_buf = ::FFI::MemoryPointer.new(:size_t)
          rc = FFI.ztok_stream_feed(stream_handle, chunk_buf, take,
                                    out_ids_buf, out_n_buf)
          Ztok.raise_for_status(rc, "ztok_stream_feed")

          ids = FFI.materialize_and_free_ids(out_ids_buf.read_pointer,
                                             out_n_buf.read(:size_t))
          yield ids unless ids.empty?
          offset += take
        end

        # Final flush.
        out_ids_buf = ::FFI::MemoryPointer.new(:pointer)
        out_n_buf = ::FFI::MemoryPointer.new(:size_t)
        rc = FFI.ztok_stream_finish(stream_handle, out_ids_buf, out_n_buf)
        Ztok.raise_for_status(rc, "ztok_stream_finish")
        ids = FFI.materialize_and_free_ids(out_ids_buf.read_pointer,
                                           out_n_buf.read(:size_t))
        yield ids unless ids.empty?
      ensure
        FFI.ztok_stream_free(stream_handle) unless stream_handle.null?
      end
    end

    # Build a finalizer proc that captures ONLY the raw pointer address
    # (an Integer), not `self`. Critical — see BatchPool.finalizer.
    def self.finalizer(address)
      proc do
        begin
          ptr = ::FFI::Pointer.new(:void, address)
          FFI.ztok_pipeline_free(ptr) unless ptr.null?
        rescue StandardError
          # Best-effort during GC.
        end
      end
    end

    # @api private
    def self.make_config(normalizer, pre_tokenizer, model, decoder)
      cfg = FFI::PipelineConfig.new
      cfg[:normalizer]    = normalizer
      cfg[:pre_tokenizer] = pre_tokenizer
      cfg[:model]         = model
      cfg[:decoder]       = decoder
      cfg
    end

    # @api private
    def self.from_file_with_cfg(fn_name, path,
                                normalizer:, pre_tokenizer:, decoder:)
      cfg = make_config(normalizer, pre_tokenizer, FFI::MODEL_BYTE_ID, decoder)
      status_buf = ::FFI::MemoryPointer.new(:int)
      handle = FFI.public_send(fn_name, path.to_s, cfg, status_buf)
      Ztok.raise_for_status(status_buf.read_int, fn_name.to_s)
      new(handle)
    end

    # @api private — block-form sugar: if the caller passed a block,
    # yield the pipeline and close on the way out. Otherwise return it.
    def self.wrap_or_yield(pipe)
      return pipe unless block_given?

      begin
        yield pipe
      ensure
        pipe.close
      end
    end

    private

    def check_open!
      raise Error, "Pipeline is closed" if @closed
    end

    # Convert anything stringy into a binary-encoded Ruby String. We
    # never need to allocate id arrays here — the C side owns them.
    def coerce_bytes(text)
      s = text.is_a?(String) ? text : text.to_s
      s.encoding == Encoding::BINARY ? s : s.dup.force_encoding(Encoding::UTF_8).b
    end
  end
end
