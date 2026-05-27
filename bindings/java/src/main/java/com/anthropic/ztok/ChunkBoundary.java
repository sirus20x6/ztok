/*
 * Enum mirror of ztok_chunk_boundary from include/ztok.h. Selects where
 * chunk edges are allowed to fall. Values match the C codes verbatim so
 * callers can pass either the enum or the int via code() / fromCode().
 */
package com.anthropic.ztok;

import com.anthropic.ztok.internal.Native;

public enum ChunkBoundary {
    /** Pure token-count windows (default). */
    TOKEN(Native.CHUNK_BOUNDARY_TOKEN),
    /** Snap to a UTF-8 codepoint boundary. */
    CODEPOINT(Native.CHUNK_BOUNDARY_CODEPOINT),
    /** Snap to a whitespace word boundary. */
    WORD(Native.CHUNK_BOUNDARY_WORD),
    /** Snap to a dictionary word boundary (CJK/Thai/...). */
    WORD_DICT(Native.CHUNK_BOUNDARY_WORD_DICT),
    /** Snap to a sentence boundary. */
    SENTENCE(Native.CHUNK_BOUNDARY_SENTENCE),
    /** Snap to a paragraph break (\n\n). */
    PARAGRAPH(Native.CHUNK_BOUNDARY_PARAGRAPH);

    private final int code;

    ChunkBoundary(int code) {
        this.code = code;
    }

    public int code() {
        return code;
    }

    /** Map a raw C code to the enum. Unknown codes collapse to TOKEN. */
    public static ChunkBoundary fromCode(int code) {
        for (ChunkBoundary b : values()) {
            if (b.code == code) return b;
        }
        return TOKEN;
    }
}
