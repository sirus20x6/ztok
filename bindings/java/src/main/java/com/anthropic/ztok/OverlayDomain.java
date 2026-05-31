/*
 * Enum mirror of ztok_overlay_domain from include/ztok.h. Pass to
 * Pipeline#setOverlayDomain to select which domain normalizer populates
 * the domain overlay channels (OPCODE / OPERAND / SYMBOL_REF / HUNK).
 *
 * NONE (the default) leaves those channels zero-filled; X86_64 decodes
 * the input as x86-64 machine code.
 */
package com.anthropic.ztok;

public enum OverlayDomain {
    /** No domain normalizer; domain channels stay zero-filled (default). */
    NONE(0),
    /** Decode the input as x86-64 machine code. */
    X86_64(1);

    private final int code;

    OverlayDomain(int code) {
        this.code = code;
    }

    public int code() {
        return code;
    }

    /** Map a raw C code to the enum. Unknown codes throw. */
    public static OverlayDomain fromCode(int code) {
        for (OverlayDomain d : values()) {
            if (d.code == code) return d;
        }
        throw new IllegalArgumentException("unknown ztok_overlay_domain code: " + code);
    }
}
