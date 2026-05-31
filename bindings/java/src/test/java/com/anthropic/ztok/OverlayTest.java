/*
 * OverlayTest — coverage for Pipeline#encodeWithOverlays
 * (ztok_encode_with_overlays). Mirrors bindings/rust/tests/overlays.rs
 * and the other bindings' overlay suites. Skips cleanly if libztok
 * cannot be loaded.
 */
package com.anthropic.ztok;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class OverlayTest {

    @BeforeAll
    static void requireLibztok() {
        assumeTrue(TestSupport.libztokAvailable(),
            "libztok not loadable (set ZTOK_LIB_PATH or build with `zig build`)");
    }

    // Build a synthetic .tiktoken vocab: 256 single bytes + a handful of
    // merges. Same shape as the Rust/Swift/.NET fixtures.
    private static Path tiktokenFixture() throws IOException {
        Path dir = Files.createTempDirectory("ztok-java-ov");
        Path path = dir.resolve("synthetic_cl100k.tiktoken");
        StringBuilder sb = new StringBuilder();
        Base64.Encoder b64 = Base64.getEncoder();
        int rank = 0;
        for (int b = 0; b < 256; b++) {
            sb.append(b64.encodeToString(new byte[] {(byte) b}))
              .append(' ').append(rank++).append('\n');
        }
        String[] extras = {
            "he", "hel", "hell", "hello", " w", " wo", " wor", " worl", " world",
            "th", "the", " th", " the", "fo", "foo", "bar", "baz",
            " quick", " brown", " fox",
        };
        for (String extra : extras) {
            sb.append(b64.encodeToString(extra.getBytes(StandardCharsets.UTF_8)))
              .append(' ').append(rank++).append('\n');
        }
        Files.writeString(path, sb.toString());
        return path;
    }

    private static Pipeline bpePipeline() throws IOException {
        return Pipeline.fromTiktoken(tiktokenFixture(), true);
    }

    @Test
    void idsMatchPlainEncode() throws IOException {
        try (Pipeline pipe = bpePipeline()) {
            int[] plain = pipe.encode("hello world");
            Pipeline.OverlayResult res = pipe.encodeWithOverlays(
                "hello world", OverlayKind.BYTE_START, OverlayKind.BYTE_END);
            assertArrayEquals(plain, res.ids(), "overlays must not change tokenization");
            assertEquals(2, res.channels().size());
            assertTrue(res.channels().containsKey(OverlayKind.BYTE_START));
            assertTrue(res.channels().containsKey(OverlayKind.BYTE_END));
        }
    }

    @Test
    void channelLengthsEqualIds() throws IOException {
        try (Pipeline pipe = bpePipeline()) {
            Pipeline.OverlayResult res = pipe.encodeWithOverlays(
                "the quick brown fox",
                OverlayKind.BYTE_START, OverlayKind.BYTE_END,
                OverlayKind.BOUNDARY, OverlayKind.PROVENANCE);
            for (Map.Entry<OverlayKind, int[]> e : res.channels().entrySet()) {
                assertEquals(res.ids().length, e.getValue().length,
                    "channel " + e.getKey() + " length mismatch");
            }
        }
    }

    @Test
    void byteSpansAreSensible() throws IOException {
        try (Pipeline pipe = bpePipeline()) {
            String text = "hello world";
            Pipeline.OverlayResult res = pipe.encodeWithOverlays(
                text, OverlayKind.BYTE_START, OverlayKind.BYTE_END);
            int[] starts = res.channels().get(OverlayKind.BYTE_START);
            int[] ends = res.channels().get(OverlayKind.BYTE_END);
            long n = text.getBytes(StandardCharsets.UTF_8).length;
            assertTrue(starts.length > 0);
            for (int i = 0; i < starts.length; i++) {
                long s = Integer.toUnsignedLong(starts[i]);
                long e = Integer.toUnsignedLong(ends[i]);
                assertTrue(s < e && e <= n, "bad span (" + s + ", " + e + ") for " + n + " bytes");
            }
            assertEquals(0, starts[0]);
            assertEquals(n, Integer.toUnsignedLong(ends[ends.length - 1]));
            for (int i = 1; i < starts.length; i++) {
                assertEquals(ends[i - 1], starts[i], "spans must tile left-to-right");
            }
        }
    }

    @Test
    void byteIdSingleByteSpans() {
        try (Pipeline pipe = Pipeline.byteId()) {
            Pipeline.OverlayResult res = pipe.encodeWithOverlays(
                "hi", OverlayKind.BYTE_START, OverlayKind.BYTE_END);
            assertArrayEquals(new int[] {0x68, 0x69}, res.ids());
            assertArrayEquals(new int[] {0, 1}, res.channels().get(OverlayKind.BYTE_START));
            assertArrayEquals(new int[] {1, 2}, res.channels().get(OverlayKind.BYTE_END));
        }
    }

    @Test
    void opcodeDomainChannelIsAllZero() throws IOException {
        try (Pipeline pipe = bpePipeline()) {
            Pipeline.OverlayResult res = pipe.encodeWithOverlays(
                "hello world", OverlayKind.OPCODE);
            int[] opcode = res.channels().get(OverlayKind.OPCODE);
            assertEquals(res.ids().length, opcode.length);
            for (int v : opcode) {
                assertEquals(0, v, "OPCODE must be zero-filled without a domain plugin");
            }
        }
    }

    @Test
    void emptyInputReturnsEmptyChannels() {
        try (Pipeline pipe = Pipeline.byteId()) {
            Pipeline.OverlayResult res = pipe.encodeWithOverlays(
                "", OverlayKind.BYTE_START, OverlayKind.OPCODE);
            assertEquals(0, res.ids().length);
            assertArrayEquals(new int[0], res.channels().get(OverlayKind.BYTE_START));
            assertArrayEquals(new int[0], res.channels().get(OverlayKind.OPCODE));
        }
    }

    @Test
    void noChannelsReturnsJustIds() throws IOException {
        try (Pipeline pipe = bpePipeline()) {
            Pipeline.OverlayResult res = pipe.encodeWithOverlays("hello world");
            assertArrayEquals(pipe.encode("hello world"), res.ids());
            assertTrue(res.channels().isEmpty());
        }
    }

    @Test
    void duplicateKindsRejected() {
        try (Pipeline pipe = Pipeline.byteId()) {
            assertThrows(ZtokException.InvalidInput.class, () ->
                pipe.encodeWithOverlays("hi", OverlayKind.BYTE_START, OverlayKind.BYTE_START));
        }
    }

    // x86-64 machine code: 48 89 d8 (mov rax,rbx) / e8 00000000 (call rel32) /
    // c3 (ret). With byteId each byte is its own token, so the OPCODE channel
    // carries one class per byte.
    private static final byte[] X86_64_CODE = {
        0x48, (byte) 0x89, (byte) 0xd8,
        (byte) 0xe8, 0x00, 0x00, 0x00, 0x00,
        (byte) 0xc3,
    };

    @Test
    void setOverlayDomainX8664PopulatesOpcode() {
        try (Pipeline pipe = Pipeline.byteId()) {
            // Default domain (NONE): OPCODE is zero-filled.
            Pipeline.OverlayResult none =
                pipe.encodeBytesWithOverlays(X86_64_CODE, OverlayKind.OPCODE);
            int[] opcodeNone = none.channels().get(OverlayKind.OPCODE);
            assertEquals(X86_64_CODE.length, opcodeNone.length);
            for (int v : opcodeNone) {
                assertEquals(0, v, "OPCODE must be zero-filled with domain=NONE");
            }

            // After selecting x86-64 the OPCODE channel is populated.
            pipe.setOverlayDomain(OverlayDomain.X86_64);
            Pipeline.OverlayResult x86 =
                pipe.encodeBytesWithOverlays(X86_64_CODE, OverlayKind.OPCODE);
            int[] opcodeX86 = x86.channels().get(OverlayKind.OPCODE);

            assertArrayEquals(none.ids(), x86.ids(), "tokenization must be unchanged");
            assertFalse(java.util.Arrays.equals(opcodeNone, opcodeX86),
                "domain channel must differ from NONE");
            boolean any = false;
            for (int v : opcodeX86) {
                if (v != 0) { any = true; break; }
            }
            assertTrue(any, "OPCODE must be populated with domain=X86_64");
        }
    }

    @Test
    void setOverlayDomainNoneRoundTrips() {
        try (Pipeline pipe = Pipeline.byteId()) {
            pipe.setOverlayDomain(OverlayDomain.X86_64);
            pipe.setOverlayDomain(OverlayDomain.NONE);
            Pipeline.OverlayResult res =
                pipe.encodeBytesWithOverlays(X86_64_CODE, OverlayKind.OPCODE);
            for (int v : res.channels().get(OverlayKind.OPCODE)) {
                assertEquals(0, v, "OPCODE must be zero-filled after switching back to NONE");
            }
        }
    }

    @Test
    void setOverlayDomainInvalidRejected() {
        try (Pipeline pipe = Pipeline.byteId()) {
            // OverlayDomain.fromCode rejects unknown codes before the native call.
            assertThrows(IllegalArgumentException.class, () ->
                pipe.setOverlayDomain(OverlayDomain.fromCode(999)));
        }
    }
}
