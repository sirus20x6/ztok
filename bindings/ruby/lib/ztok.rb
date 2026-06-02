# frozen_string_literal: true

# ztok — Ruby bindings for the ztok tokenizer library.
#
# Thin `ffi`-gem wrapper around libztok. Mirrors the C ABI declared in
# include/ztok.h. No Ruby-side allocation of id arrays — all id buffers
# remain owned by the C library and are freed via `ztok_ids_free` when
# the wrapper objects are garbage-collected (via
# `ObjectSpace.define_finalizer` with a class-method-built proc, the
# Ruby analog to Python's `weakref.finalize` / Node's
# `FinalizationRegistry`).
#
# Quickstart:
#
#   require "ztok"
#
#   pipe = Ztok::Pipeline.from_tiktoken("cl100k_base.tiktoken", cl100k: true)
#   ids  = pipe.encode("hello world")
#   text = pipe.decode(ids)
#   pipe.close
#
#   Ztok::BatchPool.open(workers: 8) do |pool|
#     results = pipe.encode_batch(pool, %w[foo bar baz])
#   end
#
#   pipe_auto = Ztok::Pipeline.from_path("tokenizer.json") # auto-detect

require_relative "ztok/version"
require_relative "ztok/errors"
require_relative "ztok/lib"
require_relative "ztok/ffi"
require_relative "ztok/batch_pool"
require_relative "ztok/stream_encoder"
require_relative "ztok/pipeline"

module Ztok
  # Returns the libztok version string (e.g. "1.28.0") as reported by
  # the loaded shared library — NOT the gem's bundled VERSION constant.
  # If they ever disagree, ztok.so was loaded from somewhere unexpected.
  def self.version
    raw = FFI.ztok_version
    raise InternalError, "ztok_version returned NULL" if raw.nil?
    raw
  end

  # --- Engram n-gram hashing ---------------------------------------------

  # Hash every length-+n+ window of +ids+ under +heads+ hash functions.
  #
  # Returns the row-major [position][head] uint64 hashes as a plain
  # Array<Integer> (positions = ids.length - n + 1, or 0 if the stream is
  # shorter than one window). Mask each hash to your table width
  # (hash & ((1 << bits) - 1)). Deterministic: identical ids always yield
  # identical hashes. Operates on raw ids — no Pipeline needed. Degenerate
  # args (n <= 0, heads <= 0, or a too-short stream) return [].
  def self.ngram_hash(ids, n:, heads:)
    return [] if n.to_i <= 0 || heads.to_i <= 0

    arr = ids.respond_to?(:to_a) ? ids.to_a : ids
    n_ids = arr.length
    return [] if n_ids < n.to_i
    positions = n_ids - n.to_i + 1
    want = positions * heads.to_i
    return [] if want <= 0

    id_buf = ::FFI::MemoryPointer.new(:uint32, n_ids)
    id_buf.write_array_of_uint32(arr)

    cap = want
    2.times do
      out_buf = ::FFI::MemoryPointer.new(:uint64, cap)
      out_len_buf = ::FFI::MemoryPointer.new(:size_t)
      rc = FFI.ztok_ngram_hash(id_buf, n_ids, n.to_i, heads.to_i,
                               out_buf, cap, out_len_buf)
      if rc == STATUS_OK
        return out_buf.read_array_of_uint64(out_len_buf.read(:size_t))
      end
      if rc == STATUS_ERR_BUFFER_TOO_SMALL
        cap = out_len_buf.read(:size_t)
        return [] if cap.zero?
        next
      end
      raise_for_status(rc, "ztok_ngram_hash")
    end
    raise InternalError, "ztok_ngram_hash reported BUFFER_TOO_SMALL twice"
  end

  # Hash many id streams in parallel across +pool+ (a Ztok::BatchPool).
  #
  # results[i] holds the row-major hashes for streams[i] (empty for a
  # stream shorter than one window). Equivalent to calling ngram_hash on
  # each stream, fanned out across the pool. Returns Array<Array<Integer>>.
  def self.ngram_hash_batch(pool, streams, n:, heads:)
    raise TypeError, "ngram_hash_batch: first arg must be a Ztok::BatchPool" unless pool.is_a?(BatchPool)

    docs = streams.map { |s| s.respond_to?(:to_a) ? s.to_a : s }
    n_docs = docs.length
    return [] if n_docs.zero?

    # Build a uint32 buffer per stream (NULL for empty streams). We must
    # hang on to id_bufs so the GC doesn't free them under the C call.
    id_bufs = docs.map do |s|
      next nil if s.empty?
      mp = ::FFI::MemoryPointer.new(:uint32, s.length)
      mp.write_array_of_uint32(s)
      mp
    end

    id_arrays = ::FFI::MemoryPointer.new(:pointer, n_docs)
    id_arrays.write_array_of_pointer(id_bufs.map { |b| b || ::FFI::Pointer::NULL })

    id_lens = ::FFI::MemoryPointer.new(:size_t, n_docs)
    docs.each_with_index { |s, i| id_lens[i].write(:size_t, s.length) }

    out_hashes = ::FFI::MemoryPointer.new(:pointer, n_docs)
    out_lens   = ::FFI::MemoryPointer.new(:size_t, n_docs)

    rc = FFI.ztok_ngram_hash_batch(pool.raw, id_arrays, id_lens, n_docs,
                                   n.to_i, heads.to_i, out_hashes, out_lens)

    out_ptrs = out_hashes.read_array_of_pointer(n_docs)
    begin
      results = Array.new(n_docs) do |i|
        ptr = out_ptrs[i]
        length = out_lens[i].read(:size_t)
        if ptr.nil? || ptr.null? || length.zero?
          []
        else
          ptr.read_array_of_uint64(length)
        end
      end
      raise_for_status(rc, "ztok_ngram_hash_batch")
      results
    ensure
      # Free each per-doc hash buffer through the C ABI.
      out_ptrs.each { |p| FFI.ztok_u64s_free(p) if p && !p.null? }
    end
  end
end
