/*
 * TekkenTest — Mistral Tekken tokenizer (ztok_pipeline_new_tekken_from_file).
 * Mirrors RwkvTest: loads the real mistral_nemo_tekken.json fixture
 * (skipped when absent) and asserts ztok reproduces the canonical
 * reference encodings, which were verified against mistral_common 1.8.6.
 * Also checks that auto-detect (Pipeline.open) routes Tekken vocabs to
 * fromTekken — guarding against the old bug where TEKKEN was routed
 * through the HF-JSON loader. Skips cleanly if libztok cannot be loaded.
 */
package com.anthropic.ztok;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class TekkenTest {

    // Golden id sequences verified against mistral_common 1.8.6 on
    // bench/vocabs/mistral_nemo_tekken.json.
    private record Golden(String text, int[] ids) {}

    private static final Golden[] GOLDEN = {
        new Golden("Hello, world!", new int[] {22177, 1044, 4304, 1033}),
        new Golden("The quick brown fox", new int[] {1784, 7586, 22980, 94137}),
        new Golden(" and the", new int[] {1321, 1278}),
    };

    @BeforeAll
    static void requireLibztok() {
        assumeTrue(TestSupport.libztokAvailable(),
            "libztok not loadable (set ZTOK_LIB_PATH or build with `zig build`)");
    }

    /** Locate bench/vocabs/mistral_nemo_tekken.json by walking up from cwd. */
    private static Path locateVocab() {
        Path dir = Path.of("").toAbsolutePath();
        for (int i = 0; i < 8 && dir != null; i++) {
            Path candidate = dir.resolve("bench")
                                .resolve("vocabs")
                                .resolve("mistral_nemo_tekken.json");
            if (Files.exists(candidate)) return candidate;
            dir = dir.getParent();
        }
        return null;
    }

    private static Path requireVocab() {
        Path vocab = locateVocab();
        assumeTrue(vocab != null,
            "Tekken vocab fixture not present under bench/vocabs/");
        return vocab;
    }

    @Test
    void matchesReference() {
        try (Pipeline pipe = Pipeline.fromTekken(requireVocab())) {
            for (Golden g : GOLDEN) {
                assertArrayEquals(g.ids(), pipe.encode(g.text()),
                    "encoding mismatch for: " + g.text());
            }
        }
    }

    @Test
    void roundTrips() {
        try (Pipeline pipe = Pipeline.fromTekken(requireVocab())) {
            for (Golden g : GOLDEN) {
                int[] ids = pipe.encode(g.text());
                assertEquals(g.text(), pipe.decode(ids),
                    "round-trip mismatch for: " + g.text());
            }
        }
    }

    @Test
    void autoDetectRoutesToTekken() {
        Path vocab = requireVocab();
        assertEquals(Format.TEKKEN, Pipeline.autoDetect(vocab),
            "autoDetect should sniff Tekken format");
        // Pipeline.open must dispatch to fromTekken (not the HF-JSON
        // loader) and reproduce the golden ids.
        try (Pipeline pipe = Pipeline.open(vocab)) {
            assertArrayEquals(GOLDEN[0].ids(), pipe.encode(GOLDEN[0].text()),
                "auto-detect open produced wrong ids — Tekken mis-routed?");
        }
    }
}
