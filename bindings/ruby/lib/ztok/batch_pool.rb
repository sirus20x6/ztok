# frozen_string_literal: true

require_relative "ffi"
require_relative "errors"

module Ztok
  # Persistent multithreaded worker pool. Reuse one pool across many
  # `Pipeline#encode_batch` calls — each pool owns its own arenas + worker
  # threads, so creating one per batch wastes work.
  #
  # Usage:
  #
  #   pool = Ztok::BatchPool.new(workers: 8)
  #   begin
  #     ids = pipe.encode_batch(pool, %w[foo bar baz])
  #   ensure
  #     pool.close
  #   end
  #
  # Or via the block form (auto-close):
  #
  #   Ztok::BatchPool.open(workers: 8) do |pool|
  #     ids = pipe.encode_batch(pool, %w[foo bar baz])
  #   end
  #
  # ## Finalizer note
  #
  # `ObjectSpace.define_finalizer` is the Ruby analog to Python's
  # `weakref.finalize` and Node's `FinalizationRegistry`. Two pitfalls
  # make this trickier than the other two languages:
  #
  #   1. Passing an instance method (or a block that closes over `self`)
  #      pins the receiver and the finalizer never fires. We sidestep
  #      that by registering a `proc { ... }` built by a class method
  #      that captures ONLY the raw integer pointer address, never
  #      `self`.
  #   2. The C handle stored in the proc must outlive the wrapper, so
  #      we stash the address (an Integer), not the FFI::Pointer object.
  class BatchPool
    # @param workers [Integer] number of worker threads; 0 means auto
    def initialize(workers: 0)
      raise ArgumentError, "workers must be >= 0 (0 = auto)" if workers.negative?

      status_buf = ::FFI::MemoryPointer.new(:int)
      handle = Ztok::FFI.ztok_batch_pool_new(workers, status_buf)
      Ztok.raise_for_status(status_buf.read_int, "ztok_batch_pool_new")
      raise InternalError, "ztok_batch_pool_new returned NULL" if handle.null?

      @handle = handle
      @closed = false
      ObjectSpace.define_finalizer(self, self.class.finalizer(handle.address))
    end

    # Convenience: yield a pool to the block, close it on the way out
    # whether or not the block raised.
    def self.open(workers: 0)
      pool = new(workers: workers)
      begin
        yield pool
      ensure
        pool.close
      end
    end

    # Detected worker count (resolves workers=0 to the OS-reported CPU
    # count). Raises Ztok::Error if the pool was already closed.
    def workers
      check_open!
      Ztok::FFI.ztok_batch_pool_worker_count(@handle).to_i
    end

    # Release the pool. Safe to call repeatedly; second + later calls
    # are no-ops. After close, `workers` and `encode_batch` raise.
    def close
      return if @closed
      @closed = true
      ObjectSpace.undefine_finalizer(self)
      Ztok::FFI.ztok_batch_pool_free(@handle)
      @handle = nil
    end

    def closed?
      @closed
    end

    # Internal accessor — Pipeline reaches in for the raw handle.
    def raw
      check_open!
      @handle
    end

    # Build a finalizer proc that captures only the raw pointer address
    # (an Integer), NOT `self`. This is the critical workaround vs.
    # naïve `define_finalizer(self, method(:close))` — that would pin
    # self forever and silently disable the finalizer.
    def self.finalizer(address)
      proc do
        begin
          ptr = ::FFI::Pointer.new(:void, address)
          Ztok::FFI.ztok_batch_pool_free(ptr) unless ptr.null?
        rescue StandardError
          # Best-effort during GC: the library may already be unloaded.
        end
      end
    end

    private

    def check_open!
      raise Error, "BatchPool is closed" if @closed
    end
  end
end
