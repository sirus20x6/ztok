/*
 * Runtime exception carrying the raw ztok_status int alongside a
 * message. Mirrors the Python ZtokError hierarchy: one base, four
 * subclass-style helpers exposed via static factories so callers can
 * still pattern-match on status() if they want.
 */
package com.anthropic.ztok;

import com.anthropic.ztok.internal.Native;

public class ZtokException extends RuntimeException {
    private final int status;

    public ZtokException(String message, int status) {
        super(message);
        this.status = status;
    }

    public int status() {
        return status;
    }

    /** Throw the appropriate ZtokException subclass for a non-OK status. */
    public static void check(int status, String ctx) {
        if (status == Native.ZTOK_OK) return;
        String msg = ctx + ": ztok status " + status;
        switch (status) {
            case Native.ZTOK_ERR_OUT_OF_MEMORY -> throw new OutOfMemory(msg, status);
            case Native.ZTOK_ERR_INVALID_INPUT -> throw new InvalidInput(msg, status);
            case Native.ZTOK_ERR_BUFFER_TOO_SMALL -> throw new BufferTooSmall(msg, status);
            case Native.ZTOK_ERR_INTERNAL -> throw new Internal(msg, status);
            default -> throw new Internal(msg, status);
        }
    }

    public static final class OutOfMemory extends ZtokException {
        public OutOfMemory(String message, int status) { super(message, status); }
    }

    public static final class InvalidInput extends ZtokException {
        public InvalidInput(String message, int status) { super(message, status); }
    }

    public static final class BufferTooSmall extends ZtokException {
        public BufferTooSmall(String message, int status) { super(message, status); }
    }

    public static final class Internal extends ZtokException {
        public Internal(String message, int status) { super(message, status); }
    }
}
