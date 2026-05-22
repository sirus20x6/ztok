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
  # Returns the libztok version string (e.g. "1.20.0") as reported by
  # the loaded shared library — NOT the gem's bundled VERSION constant.
  # If they ever disagree, ztok.so was loaded from somewhere unexpected.
  def self.version
    raw = FFI.ztok_version
    raise InternalError, "ztok_version returned NULL" if raw.nil?
    raw
  end
end
