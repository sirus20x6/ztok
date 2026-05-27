/*
 * NgramTest — Engram n-gram hashing (ztok_ngram_hash / _batch). Mirrors
 * bindings/python/tests/test_ngram.py: deterministic multi-head
 * token-n-gram hashes, row-major [position][head], positions =
 * ids.length - n + 1. Skips cleanly if libztok cannot be loaded.
 */
package com.anthropic.ztok;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class NgramTest {

    @BeforeAll
    static void requireLibztok() {
        assumeTrue(TestSupport.libztokAvailable(),
            "libztok not loadable (set ZTOK_LIB_PATH or build with `zig build`)");
    }

    @Test
    void lengthMath() {
        int[] ids = {1, 2, 3, 4, 5};
        // 5 ids, n=2 -> 4 positions; heads=3 -> 12 hashes.
        long[] out = Engram.hashNGrams(ids, 2, 3);
        assertEquals(4 * 3, out.length);
    }

    @Test
    void deterministic() {
        int[] ids = {7, 8, 9, 10, 11, 12};
        long[] a = Engram.hashNGrams(ids, 3, 4);
        long[] b = Engram.hashNGrams(ids, 3, 4);
        assertArrayEquals(a, b);
    }

    @Test
    void headIndependence() {
        // The heads of a single position should not all collide.
        long[] out = Engram.hashNGrams(new int[] {42, 43, 44}, 2, 4);
        long[] firstPosition = Arrays.copyOfRange(out, 0, 4);
        long distinct = Arrays.stream(firstPosition).distinct().count();
        assertTrue(distinct > 1, "the 4 heads of one position must not all collide");
    }

    @Test
    void shortAndBadArgs() {
        // Stream shorter than one window -> empty.
        assertEquals(0, Engram.hashNGrams(new int[] {1, 2}, 3, 2).length);
        // Degenerate args -> empty (no error).
        assertEquals(0, Engram.hashNGrams(new int[0], 1, 1).length);
        assertEquals(0, Engram.hashNGrams(new int[] {1, 2, 3}, 0, 1).length);
        assertEquals(0, Engram.hashNGrams(new int[] {1, 2, 3}, 2, 0).length);
    }

    @Test
    void batchMatchesSingle() {
        int[][] streams = {
            {1, 2, 3, 4},
            {},            // empty -> no hashes
            {9},           // shorter than window -> no hashes
            {5, 6, 7, 8, 9},
        };
        long[][] batched;
        try (BatchPool pool = BatchPool.create(2)) {
            batched = Engram.hashNGramsBatch(pool, streams, 2, 3);
        }
        assertEquals(streams.length, batched.length);
        for (int i = 0; i < streams.length; i++) {
            assertArrayEquals(Engram.hashNGrams(streams[i], 2, 3), batched[i],
                "batch row " + i + " must equal single-stream result");
        }
    }
}
