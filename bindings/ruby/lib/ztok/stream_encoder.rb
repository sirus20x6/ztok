# frozen_string_literal: true

require_relative "ffi"
require_relative "errors"

module Ztok
  # Lower-level streaming encode handle.
  #
  # Most callers should just use `Pipeline#encode_stream`, which wraps
  # this class with a "chop the input into chunks + yield ids as they
  # arrive" loop. Use `StreamEncoder` directly when you need to feed
  # arbitrary-sized buffers as they become available (e.g. a chunked
  # HTTP body, a streaming socket) and you want id-as-it-arrives semantics
  # without your own outer loop.
  #
  #   encoder = Ztok::StreamEncoder.new(pipe)
  #   begin
  #     until reader.eof?
  #       ids = encoder.feed(reader.read(64 * 1024))
  #       handle_ids(ids) unless ids.empty?
  #     end
  #     final = encoder.finish
  #     handle_ids(final) unless final.empty?
  #   ensure
  #     encoder.close
  #   end
  #
  # The encoder defers a trailing partial UTF-8 codepoint or pre-
  # tokenizer span up to a soft cap of 1 MiB (see src/stream.zig); past
  # that it force-cuts at the nearest codepoint boundary.
  class StreamEncoder
    def initialize(pipeline)
      raise TypeError, "expected Ztok::Pipeline" unless pipeline.is_a?(Pipeline)

      status_buf = ::FFI::MemoryPointer.new(:int)
      handle = FFI.ztok_stream_new(pipeline.raw, status_buf)
      Ztok.raise_for_status(status_buf.read_int, "ztok_stream_new")
      raise InternalError, "ztok_stream_new returned NULL" if handle.null?

      @handle = handle
      @closed = false
      @finished = false
      ObjectSpace.define_finalizer(self, self.class.finalizer(handle.address))
    end

    # Feed +bytes+ into the encoder and return an Array<Integer> of
    # newly-emitted ids (may be empty when the encoder is still
    # buffering toward the next safe cut).
    def feed(bytes)
      check_open!
      data = bytes.is_a?(String) ? bytes : bytes.to_s
      data = data.encoding == Encoding::BINARY ? data : data.dup.force_encoding(Encoding::UTF_8).b
      return [] if data.bytesize.zero?

      buf = ::FFI::MemoryPointer.new(:uint8, data.bytesize)
      buf.write_bytes(data)

      out_ids_buf = ::FFI::MemoryPointer.new(:pointer)
      out_n_buf = ::FFI::MemoryPointer.new(:size_t)
      rc = FFI.ztok_stream_feed(@handle, buf, data.bytesize,
                                out_ids_buf, out_n_buf)
      Ztok.raise_for_status(rc, "ztok_stream_feed")
      FFI.materialize_and_free_ids(out_ids_buf.read_pointer,
                                   out_n_buf.read(:size_t))
    end

    # Flush any remaining carry. Returns the trailing ids (possibly
    # empty). Idempotent: a second call returns [].
    def finish
      check_open!
      return [] if @finished
      @finished = true

      out_ids_buf = ::FFI::MemoryPointer.new(:pointer)
      out_n_buf = ::FFI::MemoryPointer.new(:size_t)
      rc = FFI.ztok_stream_finish(@handle, out_ids_buf, out_n_buf)
      Ztok.raise_for_status(rc, "ztok_stream_finish")
      FFI.materialize_and_free_ids(out_ids_buf.read_pointer,
                                   out_n_buf.read(:size_t))
    end

    # Release the underlying C stream. Idempotent.
    def close
      return if @closed
      @closed = true
      ObjectSpace.undefine_finalizer(self)
      FFI.ztok_stream_free(@handle)
      @handle = nil
    end

    def closed?
      @closed
    end

    def self.finalizer(address)
      proc do
        begin
          ptr = ::FFI::Pointer.new(:void, address)
          FFI.ztok_stream_free(ptr) unless ptr.null?
        rescue StandardError
          # Best-effort during GC.
        end
      end
    end

    private

    def check_open!
      raise Error, "StreamEncoder is closed" if @closed
    end
  end
end
