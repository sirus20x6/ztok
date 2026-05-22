/*
 * Persistent worker pool for parallel batch encode. Reuse one across
 * many batches — each pool owns its own arenas and worker threads, so
 * creating one per batch wastes work.
 *
 * Usage:
 *   try (BatchPool pool = BatchPool.create(8);
 *        Pipeline pipe = Pipeline.byteId()) {
 *       int[][] ids = pipe.encodeBatch(pool, List.of("foo", "bar"));
 *   }
 */
package com.anthropic.ztok;

import com.anthropic.ztok.internal.Native;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.ref.Cleaner;

public final class BatchPool implements AutoCloseable {
    private static final Cleaner CLEANER = Cleaner.create();

    /** Live-handle holder so the Cleaner can null-check without touching the wrapper. */
    private static final class State implements Runnable {
        volatile MemorySegment handle;

        State(MemorySegment handle) {
            this.handle = handle;
        }

        @Override
        public void run() {
            MemorySegment h = handle;
            if (h != null && h.address() != 0) {
                handle = null;
                try {
                    Native.ZTOK_BATCH_POOL_FREE.invoke(h);
                } catch (Throwable ignored) {
                    // best-effort during GC
                }
            }
        }
    }

    private final State state;
    private final Cleaner.Cleanable cleanable;
    private volatile boolean closed;

    private BatchPool(MemorySegment handle) {
        this.state = new State(handle);
        this.cleanable = CLEANER.register(this, state);
    }

    /** Build a pool. {@code workers=0} means auto-detect from cpu count. */
    public static BatchPool create(int workers) {
        if (workers < 0) {
            throw new IllegalArgumentException("workers must be >= 0 (0 = auto)");
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment status = arena.allocate(Native.INT);
            MemorySegment handle;
            try {
                handle = (MemorySegment) Native.ZTOK_BATCH_POOL_NEW
                    .invoke(workers, status);
            } catch (Throwable t) {
                throw new ZtokException.Internal(
                    "ztok_batch_pool_new threw: " + t.getMessage(),
                    Native.ZTOK_ERR_INTERNAL);
            }
            ZtokException.check(status.get(Native.INT, 0), "ztok_batch_pool_new");
            if (handle == null || handle.address() == 0) {
                throw new ZtokException.Internal(
                    "ztok_batch_pool_new returned NULL", Native.ZTOK_ERR_INTERNAL);
            }
            return new BatchPool(handle);
        }
    }

    /** Build a pool with auto-detected worker count. */
    public static BatchPool create() {
        return create(0);
    }

    /** Live worker count (resolves auto to the detected cpu count). */
    public int workers() {
        ensureOpen();
        try {
            long n = (long) Native.ZTOK_BATCH_POOL_WORKER_COUNT.invoke(state.handle);
            return (int) n;
        } catch (Throwable t) {
            throw new ZtokException.Internal(
                "ztok_batch_pool_worker_count threw: " + t.getMessage(),
                Native.ZTOK_ERR_INTERNAL);
        }
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        cleanable.clean();
    }

    /** Raw handle, for Pipeline.encodeBatch. */
    MemorySegment handle() {
        ensureOpen();
        return state.handle;
    }

    private void ensureOpen() {
        if (closed || state.handle == null) {
            throw new ZtokException.Internal(
                "BatchPool is closed", Native.ZTOK_ERR_INTERNAL);
        }
    }
}
