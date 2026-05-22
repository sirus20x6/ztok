/*
 * 32-byte tokenizer fingerprint. See `ztok_fingerprint` in include/ztok.h.
 *
 * Two pipelines that produce equal Fingerprints will emit bit-identical
 * id streams for ANY input — use this as a cache key, KV-store
 * discriminator, or training-pipeline guard.
 */
package com.anthropic.ztok;

import java.util.Arrays;
import java.util.HexFormat;

public final class Fingerprint {
    public static final int LENGTH = 32;

    private final byte[] bytes;

    Fingerprint(byte[] bytes) {
        if (bytes == null || bytes.length != LENGTH) {
            throw new IllegalArgumentException(
                "Fingerprint must be exactly " + LENGTH + " bytes");
        }
        this.bytes = bytes.clone();
    }

    public byte[] bytes() {
        return bytes.clone();
    }

    public String hex() {
        return HexFormat.of().formatHex(bytes);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Fingerprint other)) return false;
        return Arrays.equals(bytes, other.bytes);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(bytes);
    }

    @Override
    public String toString() {
        return "Fingerprint(" + hex() + ")";
    }
}
