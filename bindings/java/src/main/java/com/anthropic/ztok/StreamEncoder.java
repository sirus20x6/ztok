/*
 * Streaming encoder — yields int[] chunks as ids are emitted by the
 * underlying ztok_stream_*. Implements Iterator<int[]> for explicit
 * drive and exposes asStream() for the fluent path.
 *
 * Lifetime: each StreamEncoder owns a ztok_stream handle. It is
 * AutoCloseable and must be closed exactly once; close() is idempotent.
 * If the caller drops the encoder without close(), the Cleaner releases
 * the native handle, but the encoder will have stopped emitting before
 * draining the final flush — explicit close() (or try-with-resources)
 * is strongly preferred.
 *
 * Feeds the underlying input in chunks of {@code chunkSize} bytes (64
 * KiB by default). Empty yields between chunks are skipped, so the
 * caller only sees non-empty arrays. The encoder defers a trailing
 * partial UTF-8 codepoint / pre-tokenizer span up to a 1 MiB soft cap
 * (see src/stream.zig); past that it force-cuts at the nearest
 * codepoint boundary.
 */
package com.anthropic.ztok;

import com.anthropic.ztok.internal.Native;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.ref.Cleaner;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

public final class StreamEncoder implements Iterator<int[]>, AutoCloseable {
    public static final int DEFAULT_CHUNK_SIZE = 64 * 1024;

    private static final Cleaner CLEANER = Cleaner.create();

    private static final class State implements Runnable {
        volatile MemorySegment handle;

        State(MemorySegment handle) { this.handle = handle; }

        @Override
        public void run() {
            MemorySegment h = handle;
            if (h != null && h.address() != 0) {
                handle = null;
                try { Native.ZTOK_STREAM_FREE.invoke(h); }
                catch (Throwable ignored) { /* best-effort */ }
            }
        }
    }

    private final byte[] input;
    private final int chunkSize;
    private final State state;
    private final Cleaner.Cleanable cleanable;

    private int cursor;       // next byte offset to feed
    private boolean finished; // ztok_stream_finish already called
    private int[] pending;    // next non-empty array to return (or null)
    private volatile boolean closed;

    StreamEncoder(MemorySegment streamHandle, byte[] input, int chunkSize) {
        if (chunkSize <= 0) {
            throw new IllegalArgumentException("chunkSize must be > 0");
        }
        this.input = input;
        this.chunkSize = chunkSize;
        this.state = new State(streamHandle);
        this.cleanable = CLEANER.register(this, state);
    }

    @Override
    public boolean hasNext() {
        if (pending != null) return true;
        if (closed) return false;
        advance();
        return pending != null;
    }

    @Override
    public int[] next() {
        if (!hasNext()) {
            throw new NoSuchElementException();
        }
        int[] out = pending;
        pending = null;
        return out;
    }

    public Stream<int[]> asStream() {
        return StreamSupport.stream(
            Spliterators.spliteratorUnknownSize(this, Spliterator.ORDERED | Spliterator.NONNULL),
            false);
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        cleanable.clean();
    }

    /** Pull the next non-empty id array, feeding more bytes as needed. */
    private void advance() {
        while (pending == null) {
            if (cursor < input.length) {
                int n = Math.min(chunkSize, input.length - cursor);
                int[] chunk = feedChunk(cursor, n);
                cursor += n;
                if (chunk != null && chunk.length > 0) {
                    pending = chunk;
                    return;
                }
                continue;
            }
            if (!finished) {
                finished = true;
                int[] tail = finishStream();
                if (tail != null && tail.length > 0) {
                    pending = tail;
                }
                return;
            }
            return;
        }
    }

    private int[] feedChunk(int offset, int len) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment bytes = arena.allocate(len == 0 ? 1 : len);
            if (len > 0) {
                MemorySegment.copy(input, offset, bytes, Native.BYTE, 0, len);
            }
            MemorySegment outIds = arena.allocate(Native.PTR);
            MemorySegment outN   = arena.allocate(Native.SIZE_T);
            int rc;
            try {
                rc = (int) Native.ZTOK_STREAM_FEED.invoke(
                    state.handle, bytes, (long) len, outIds, outN);
            } catch (Throwable t) {
                throw new ZtokException.Internal(
                    "ztok_stream_feed threw: " + t.getMessage(),
                    Native.ZTOK_ERR_INTERNAL);
            }
            ZtokException.check(rc, "ztok_stream_feed");
            return materializeAndFree(outIds, outN);
        }
    }

    private int[] finishStream() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment outIds = arena.allocate(Native.PTR);
            MemorySegment outN   = arena.allocate(Native.SIZE_T);
            int rc;
            try {
                rc = (int) Native.ZTOK_STREAM_FINISH.invoke(
                    state.handle, outIds, outN);
            } catch (Throwable t) {
                throw new ZtokException.Internal(
                    "ztok_stream_finish threw: " + t.getMessage(),
                    Native.ZTOK_ERR_INTERNAL);
            }
            ZtokException.check(rc, "ztok_stream_finish");
            return materializeAndFree(outIds, outN);
        }
    }

    /**
     * Copy a libztok-owned id buffer into a Java int[] then call
     * ztok_ids_free. This is the ONLY safe free path — the buffers
     * carry a length-prefix header (see src/c_api.zig::allocIdBuf).
     */
    private static int[] materializeAndFree(MemorySegment outIdsPtr, MemorySegment outNPtr) {
        long n = outNPtr.get(Native.SIZE_T, 0);
        MemorySegment ptr = outIdsPtr.get(Native.PTR, 0);
        if (n <= 0 || ptr.address() == 0) return new int[0];
        MemorySegment view = ptr.reinterpret(n * Native.TOKEN_ID.byteSize());
        int[] out = new int[(int) n];
        MemorySegment.copy(view, Native.TOKEN_ID, 0, out, 0, (int) n);
        try { Native.ZTOK_IDS_FREE.invoke(ptr); }
        catch (Throwable ignored) { /* best-effort */ }
        return out;
    }
}
