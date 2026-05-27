/*
 * Engram n-gram hashing — deterministic multi-head token-n-gram hashes
 * for conditional-memory addressing (see src/ngram.zig). These operate on
 * raw token ids and need no Pipeline; the output is row-major
 * [position][head] raw uint64 hashes which the caller masks to its own
 * table width.
 *
 * Java has no unsigned long, so each hash is returned as a {@code long}
 * holding the raw 64 bits — interpret it as unsigned (e.g. via
 * {@link Long#toUnsignedString} or {@link Long#compareUnsigned}). The sign
 * bit is meaningful, so do NOT assume hashes are >= 0.
 */
package com.anthropic.ztok;

import com.anthropic.ztok.internal.Native;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

public final class Engram {
    private Engram() {}

    /**
     * Hash every length-{@code n} window of {@code ids} under {@code heads}
     * independent hash functions, returning the row-major
     * {@code [position][head]} raw uint64 hashes (positions =
     * {@code ids.length - n + 1}, or 0 if the stream is shorter than one
     * window). Mask each hash to your table width
     * ({@code hash & ((1L << bits) - 1)}). Deterministic: the same ids
     * always yield the same hashes. Returns an empty array when there is
     * nothing to hash (n==0, heads==0, or a stream shorter than one window).
     * The raw bits are stored unsigned in each {@code long}.
     */
    public static long[] hashNGrams(int[] ids, int n, int heads) {
        if (n <= 0 || heads <= 0) return new long[0];
        int nIds = ids.length;
        if (nIds < n) return new long[0];
        long positions = (long) nIds - n + 1;
        long want = positions * heads;
        if (want == 0) return new long[0];

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment idsSeg = arena.allocateArray(Native.TOKEN_ID, nIds);
            MemorySegment.copy(ids, 0, idsSeg, Native.TOKEN_ID, 0, nIds);
            MemorySegment outLen = arena.allocate(Native.SIZE_T);

            long cap = want;
            for (int attempt = 0; attempt < 2; attempt++) {
                MemorySegment out = arena.allocateArray(Native.LONG, cap);
                int rc;
                try {
                    rc = (int) Native.ZTOK_NGRAM_HASH.invoke(
                        idsSeg, (long) nIds, n, heads, out, cap, outLen);
                } catch (Throwable t) {
                    throw rethrow("ztok_ngram_hash", t);
                }
                if (rc == Native.ZTOK_OK) {
                    long len = outLen.get(Native.SIZE_T, 0);
                    long[] hashes = new long[(int) len];
                    MemorySegment.copy(out, Native.LONG, 0, hashes, 0, (int) len);
                    return hashes;
                }
                if (rc == Native.ZTOK_ERR_BUFFER_TOO_SMALL) {
                    cap = outLen.get(Native.SIZE_T, 0);
                    if (cap == 0) return new long[0];
                    continue;
                }
                ZtokException.check(rc, "ztok_ngram_hash");
            }
            throw new ZtokException.Internal(
                "ztok_ngram_hash reported BUFFER_TOO_SMALL twice",
                Native.ZTOK_ERR_INTERNAL);
        }
    }

    /**
     * Hash many id streams in parallel across {@code pool}. {@code result[i]}
     * holds the row-major hashes for {@code streams[i]} (an empty array for a
     * stream shorter than one window). Equivalent to calling
     * {@link #hashNGrams} on each stream, but fanned out across the pool's
     * workers.
     */
    public static long[][] hashNGramsBatch(BatchPool pool, int[][] streams,
                                           int n, int heads) {
        MemorySegment poolHandle = pool.handle();
        int nDocs = streams.length;
        if (nDocs == 0) return new long[0][];

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment idArrays = arena.allocateArray(Native.PTR, nDocs);
            MemorySegment idLens   = arena.allocateArray(Native.SIZE_T, nDocs);
            for (int i = 0; i < nDocs; i++) {
                int[] s = streams[i];
                idLens.setAtIndex(Native.SIZE_T, i, (long) s.length);
                if (s.length == 0) {
                    idArrays.setAtIndex(Native.PTR, i, MemorySegment.NULL);
                    continue;
                }
                MemorySegment buf = arena.allocateArray(Native.TOKEN_ID, s.length);
                MemorySegment.copy(s, 0, buf, Native.TOKEN_ID, 0, s.length);
                idArrays.setAtIndex(Native.PTR, i, buf);
            }

            MemorySegment outHashes = arena.allocateArray(Native.PTR, nDocs);
            MemorySegment outLens   = arena.allocateArray(Native.SIZE_T, nDocs);

            int rc;
            try {
                rc = (int) Native.ZTOK_NGRAM_HASH_BATCH.invoke(
                    poolHandle, idArrays, idLens, (long) nDocs,
                    n, heads, outHashes, outLens);
            } catch (Throwable t) {
                throw rethrow("ztok_ngram_hash_batch", t);
            }

            long[][] results = new long[nDocs][];
            try {
                // Materialize every (possibly partial) buffer before raising,
                // so an error mid-batch can't leak the buffers already
                // allocated by the C side.
                for (int i = 0; i < nDocs; i++) {
                    long len = outLens.getAtIndex(Native.SIZE_T, i);
                    MemorySegment ptr = outHashes.getAtIndex(Native.PTR, i);
                    if (len == 0 || ptr.address() == 0) {
                        results[i] = new long[0];
                        continue;
                    }
                    MemorySegment view = ptr.reinterpret(len * Native.LONG.byteSize());
                    long[] hashes = new long[(int) len];
                    MemorySegment.copy(view, Native.LONG, 0, hashes, 0, (int) len);
                    results[i] = hashes;
                }
                ZtokException.check(rc, "ztok_ngram_hash_batch");
            } finally {
                // Each out_hashes[i] is a ztok-allocated u64 buffer; the only
                // safe free is ztok_u64s_free (length-prefix header).
                for (int i = 0; i < nDocs; i++) {
                    MemorySegment ptr = outHashes.getAtIndex(Native.PTR, i);
                    if (ptr.address() != 0) {
                        try { Native.ZTOK_U64S_FREE.invoke(ptr); }
                        catch (Throwable ignored) { /* best-effort */ }
                    }
                }
            }
            return results;
        }
    }

    private static ZtokException rethrow(String ctx, Throwable t) {
        if (t instanceof ZtokException ze) return ze;
        return new ZtokException.Internal(
            ctx + " threw: " + t.getClass().getSimpleName() + ": " + t.getMessage(),
            Native.ZTOK_ERR_INTERNAL);
    }
}
