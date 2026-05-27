/*
 * Enum mirror of ztok_format from include/ztok.h. Values match the C
 * codes verbatim so callers can pass either the enum or the int via
 * Format#code / Format#fromCode.
 */
package com.anthropic.ztok;

import com.anthropic.ztok.internal.Native;

public enum Format {
    UNKNOWN(Native.FORMAT_UNKNOWN),
    TIKTOKEN(Native.FORMAT_TIKTOKEN),
    HF_JSON(Native.FORMAT_HF_JSON),
    SENTENCEPIECE(Native.FORMAT_SP_MODEL),
    ZTM(Native.FORMAT_ZTM),
    TEKKEN(Native.FORMAT_TEKKEN),
    RWKV(Native.FORMAT_RWKV);

    private final int code;

    Format(int code) {
        this.code = code;
    }

    public int code() {
        return code;
    }

    /** Map a raw C code to the enum. Unknown codes collapse to UNKNOWN. */
    public static Format fromCode(int code) {
        for (Format f : values()) {
            if (f.code == code) return f;
        }
        return UNKNOWN;
    }
}
