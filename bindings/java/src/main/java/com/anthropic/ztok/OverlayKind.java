/*
 * Enum mirror of ztok_overlay_kind from include/ztok.h. Pass a set of
 * these to Pipeline#encodeWithOverlays to request per-token annotation
 * channels aligned 1:1 with the id stream.
 *
 * Cheap channels (BYTE_START / BYTE_END / BOUNDARY / PROVENANCE) carry
 * encoder-derived values; domain channels (OPCODE / OPERAND / SYMBOL_REF
 * / HUNK) come back zero-filled until a domain plugin populates them.
 */
package com.anthropic.ztok;

public enum OverlayKind {
    /** Original-input byte offset where the token's span starts. */
    BYTE_START(0),
    /** Original-input byte offset where the token's span ends (exclusive). */
    BYTE_END(1),
    /** Boundary bitset: 0x1 chunk-start, 0x2 codepoint-start. */
    BOUNDARY(2),
    /** Domain: normalized opcode class (zero-filled without a plugin). */
    OPCODE(3),
    /** Domain: normalized operand class (zero-filled without a plugin). */
    OPERAND(4),
    /** Domain: symbol-table index, 0 = none (zero-filled without a plugin). */
    SYMBOL_REF(5),
    /** Domain: diff-hunk id (zero-filled without a plugin). */
    HUNK(6),
    /** Provenance: 0 = model text, 1 = special token. */
    PROVENANCE(7);

    private final int code;

    OverlayKind(int code) {
        this.code = code;
    }

    public int code() {
        return code;
    }

    /** Map a raw C code to the enum. Unknown codes throw. */
    public static OverlayKind fromCode(int code) {
        for (OverlayKind k : values()) {
            if (k.code == code) return k;
        }
        throw new IllegalArgumentException("unknown ztok_overlay_kind code: " + code);
    }
}
