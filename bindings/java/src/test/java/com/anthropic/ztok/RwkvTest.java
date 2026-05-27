/*
 * RwkvTest — RWKV "World" tokenizer (ztok_pipeline_new_rwkv_from_file).
 * Mirrors bindings/python/tests/test_rwkv.py: loads the real
 * rwkv_vocab_v20230424.txt fixture (skipped when absent) and checks ztok
 * reproduces the canonical reference encodings. Skips cleanly if libztok
 * cannot be loaded.
 */
package com.anthropic.ztok;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class RwkvTest {

    // Golden id sequences captured from BlinkDL's canonical reference
    // tokenizer (see bench/rwkv_parity.py / src/rwkv_world.zig).
    private record Golden(String text, int[] ids) {}

    private static final Golden[] GOLDEN = {
        new Golden("Hello, world!", new int[] {33155, 45, 40213, 34}),
        new Golden("emoji 😀🚀✨ test",
            new int[] {34295, 33, 3319, 153, 129, 3319, 155, 129, 10059, 32223}),
        new Golden("0 1 2 10 99 100", new int[] {49, 284, 285, 3483, 3572, 3483, 49}),
    };

    @BeforeAll
    static void requireLibztok() {
        assumeTrue(TestSupport.libztokAvailable(),
            "libztok not loadable (set ZTOK_LIB_PATH or build with `zig build`)");
    }

    /** Locate bench/vocabs/rwkv_vocab_v20230424.txt by walking up from cwd. */
    private static Path locateVocab() {
        Path dir = Path.of("").toAbsolutePath();
        for (int i = 0; i < 8 && dir != null; i++) {
            Path candidate = dir.resolve("bench")
                                .resolve("vocabs")
                                .resolve("rwkv_vocab_v20230424.txt");
            if (Files.exists(candidate)) return candidate;
            dir = dir.getParent();
        }
        return null;
    }

    private static Pipeline openRwkv() {
        Path vocab = locateVocab();
        assumeTrue(vocab != null, "RWKV vocab fixture not present under bench/vocabs/");
        return Pipeline.fromRwkv(vocab);
    }

    @Test
    void matchesReference() {
        try (Pipeline pipe = openRwkv()) {
            for (Golden g : GOLDEN) {
                assertArrayEquals(g.ids(), pipe.encode(g.text()),
                    "encoding mismatch for: " + g.text());
            }
        }
    }

    @Test
    void roundTrips() {
        try (Pipeline pipe = openRwkv()) {
            for (Golden g : GOLDEN) {
                int[] ids = pipe.encode(g.text());
                assertEquals(g.text(), pipe.decode(ids),
                    "round-trip mismatch for: " + g.text());
            }
        }
    }
}
