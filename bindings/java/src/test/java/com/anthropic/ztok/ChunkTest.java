/*
 * ChunkTest — token-window chunking (ztok_chunk). Mirrors
 * bindings/python/tests/test_chunk.py. Run over a byte_id pipeline (each
 * input byte = one token) so chunk boundaries are predictable:
 * "abcdefghij" is 10 tokens, one per byte. Skips cleanly if libztok
 * cannot be loaded.
 */
package com.anthropic.ztok;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class ChunkTest {

    @BeforeAll
    static void requireLibztok() {
        assumeTrue(TestSupport.libztokAvailable(),
            "libztok not loadable (set ZTOK_LIB_PATH or build with `zig build`)");
    }

    @Test
    void nonOverlapping() {
        try (Pipeline pipe = Pipeline.byteId()) {
            Chunk[] chunks = pipe.chunk("abcdefghij", 4, 0);
            // 10 tokens / window 4, stride 4 -> [0,4) [4,8) [8,10).
            assertEquals(3, chunks.length);
            int[][] want = {
                // {tokenStart, tokenEnd, byteStart, byteEnd, idsLen}
                {0, 4, 0, 4, 4},
                {4, 8, 4, 8, 4},
                {8, 10, 8, 10, 2},
            };
            for (int i = 0; i < want.length; i++) {
                Chunk c = chunks[i];
                assertEquals(want[i][0], c.tokenStart());
                assertEquals(want[i][1], c.tokenEnd());
                assertEquals(want[i][2], c.byteStart());
                assertEquals(want[i][3], c.byteEnd());
                assertEquals(want[i][4], c.ids().length);
            }
        }
    }

    @Test
    void overlap() {
        try (Pipeline pipe = Pipeline.byteId()) {
            Chunk[] chunks = pipe.chunk("abcdefghij", 4, 2);
            assertTrue(chunks.length >= 2);
            // stride = 2, so the last 2 ids of chunk[i] equal the first 2 of
            // chunk[i+1].
            for (int i = 0; i + 1 < chunks.length; i++) {
                int[] a = chunks[i].ids();
                int[] b = chunks[i + 1].ids();
                if (a.length >= 2 && b.length >= 2) {
                    assertArrayEquals(
                        Arrays.copyOfRange(a, a.length - 2, a.length),
                        Arrays.copyOfRange(b, 0, 2),
                        "overlap tail of chunk " + i + " must match head of next");
                }
            }
        }
    }

    @Test
    void emptyAndBadArgs() {
        try (Pipeline pipe = Pipeline.byteId()) {
            assertEquals(0, pipe.chunk("", 4).length);
            assertThrows(ZtokException.InvalidInput.class, () -> pipe.chunk("abc", 0));
            assertThrows(ZtokException.InvalidInput.class, () -> pipe.chunk("abc", 4, 4));
        }
    }

    @Test
    void boundaryOverloadDefaultsToToken() {
        try (Pipeline pipe = Pipeline.byteId()) {
            Chunk[] viaDefault = pipe.chunk("abcdefghij", 4);
            Chunk[] viaExplicit = pipe.chunk("abcdefghij", 4, 0, ChunkBoundary.TOKEN);
            assertEquals(viaDefault.length, viaExplicit.length);
            for (int i = 0; i < viaDefault.length; i++) {
                assertArrayEquals(viaDefault[i].ids(), viaExplicit[i].ids());
            }
        }
    }
}
