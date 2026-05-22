/*
 * SmokeTest — basic round-trip and API surface checks against the
 * byte_id baseline pipeline. Skips cleanly if libztok cannot be loaded,
 * so this still passes on a CI runner without a built library.
 */
package com.anthropic.ztok;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class SmokeTest {

    @BeforeAll
    static void requireLibztok() {
        assumeTrue(TestSupport.libztokAvailable(),
            "libztok not loadable (set ZTOK_LIB_PATH or build with `zig build`)");
    }

    @Test
    void version_returnsNonEmpty() {
        String v = Ztok.version();
        assertNotNull(v);
        assertTrue(v.length() > 0, "version string must be non-empty");
    }

    @Test
    void byteId_roundTripsAscii() {
        try (Pipeline pipe = Pipeline.byteId()) {
            int[] ids = pipe.encode("hello world");
            assertEquals(11, ids.length, "byte_id is 1:1 with input bytes");
            assertEquals("hello world", pipe.decode(ids));
        }
    }

    @Test
    void byteId_roundTripsUtf8() {
        try (Pipeline pipe = Pipeline.byteId()) {
            String text = "héllo 😀";  // h + e-acute + grinning face emoji
            byte[] expectedBytes = text.getBytes(java.nio.charset.StandardCharsets.UTF_8);
            int[] ids = pipe.encode(text);
            assertEquals(expectedBytes.length, ids.length);
            assertArrayEquals(expectedBytes, pipe.decodeBytes(ids));
            assertEquals(text, pipe.decode(ids));
        }
    }

    @Test
    void byteId_emptyInputRoundTrips() {
        try (Pipeline pipe = Pipeline.byteId()) {
            assertArrayEquals(new int[0], pipe.encode(""));
            assertEquals("", pipe.decode(new int[0]));
        }
    }

    @Test
    void fingerprint_isDeterministicForSameConfig() {
        try (Pipeline a = Pipeline.byteId(); Pipeline b = Pipeline.byteId()) {
            Fingerprint fa = a.fingerprint();
            Fingerprint fb = b.fingerprint();
            assertEquals(fa, fb,
                "two byte_id pipelines must produce the same fingerprint");
            assertEquals(Fingerprint.LENGTH, fa.bytes().length);
        }
    }

    @Test
    void batchPool_encodesInParallel() {
        try (Pipeline pipe = Pipeline.byteId();
             BatchPool pool = BatchPool.create(2)) {
            assertTrue(pool.workers() >= 1, "auto-detect must resolve to >=1 worker");
            int[][] ids = pipe.encodeBatch(pool, List.of("foo", "bar", "baz", ""));
            assertEquals(4, ids.length);
            assertArrayEquals(new int[] {'f', 'o', 'o'}, ids[0]);
            assertArrayEquals(new int[] {'b', 'a', 'r'}, ids[1]);
            assertArrayEquals(new int[] {'b', 'a', 'z'}, ids[2]);
            assertArrayEquals(new int[0], ids[3]);
        }
    }

    @Test
    void streamEncoder_drainsAllChunks() {
        try (Pipeline pipe = Pipeline.byteId()) {
            String text = "abcdefghijklmnopqrstuvwxyz";
            byte[] data = text.getBytes(java.nio.charset.StandardCharsets.UTF_8);
            int total = 0;
            try (StreamEncoder enc = pipe.encodeStream(data, 4)) {
                while (enc.hasNext()) {
                    int[] chunk = enc.next();
                    assertTrue(chunk.length > 0);
                    total += chunk.length;
                }
            }
            assertEquals(text.length(), total);
        }
    }

    @Test
    void streamEncoder_asStreamMatchesIterator() {
        try (Pipeline pipe = Pipeline.byteId()) {
            String text = "stream me";
            try (StreamEncoder enc = pipe.encodeStream(text)) {
                long count = enc.asStream().mapToInt(arr -> arr.length).sum();
                assertEquals(text.length(), count);
            }
        }
    }

    @Test
    void closedPipeline_throwsOnUse() {
        Pipeline pipe = Pipeline.byteId();
        pipe.close();
        assertTrue(pipe.isClosed());
        ZtokException ex = assertThrows(ZtokException.class, () -> pipe.encode("x"));
        assertNotEquals(0, ex.status());
    }

    @Test
    void autoDetect_unknownFileReturnsUnknown() {
        // Use a path that almost certainly doesn't resolve to a known
        // tokenizer format — ztok_auto_detect is best-effort and never
        // surfaces an I/O error, so this just round-trips through the C.
        Format fmt = Ztok.autoDetect(java.nio.file.Path.of("/this/path/does/not/exist"));
        assertEquals(Format.UNKNOWN, fmt);
    }
}
