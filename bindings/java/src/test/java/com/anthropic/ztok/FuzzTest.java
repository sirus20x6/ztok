/*
 * FuzzTest — 1000 PRNG-driven round-trip iterations against the
 * byte_id pipeline. Mirrors fuzz/encode_decode.zig and the Python /
 * Ruby / Node fuzz harnesses: same seed (0xFEEDB0B) so a failing
 * iteration is cross-language reproducible.
 *
 * Skips cleanly when libztok is not loadable.
 */
package com.anthropic.ztok;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class FuzzTest {

    private static final long SEED = 0xFEEDB0BL;
    private static final int ITERATIONS = Integer.getInteger("ztok.fuzz.iters", 1000);
    private static final int MAX_LEN = 256;

    @BeforeAll
    static void requireLibztok() {
        assumeTrue(TestSupport.libztokAvailable(),
            "libztok not loadable (set ZTOK_LIB_PATH or build with `zig build`)");
    }

    @Test
    void byteId_roundTripFuzz() {
        Random rng = new Random(SEED);
        try (Pipeline pipe = Pipeline.byteId()) {
            for (int i = 0; i < ITERATIONS; i++) {
                int n = rng.nextInt(MAX_LEN + 1);
                byte[] data = new byte[n];
                rng.nextBytes(data);

                int[] ids = pipe.encodeBytes(data);
                assertEquals(n, ids.length,
                    "iter " + i + ": byte_id produced " + ids.length
                    + " ids for " + n + " bytes");

                byte[] roundTrip = pipe.decodeBytes(ids);
                assertArrayEquals(data, roundTrip,
                    "iter " + i + ": round-trip mismatch (len=" + n + ")");
            }
        }
    }
}
