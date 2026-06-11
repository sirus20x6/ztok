/*
 * Top-level convenience entry point — version string + auto-detect.
 *
 * The real surface area lives on {@link Pipeline}, {@link BatchPool},
 * {@link StreamEncoder}, and {@link Fingerprint}; this class just
 * exposes the two stand-alone helpers that don't need a pipeline
 * handle.
 */
package com.anthropic.ztok;

import com.anthropic.ztok.internal.Native;

import java.lang.foreign.MemorySegment;
import java.nio.file.Path;

public final class Ztok {
    private Ztok() {}

    /** libztok version string (e.g. "1.28.0"). */
    public static String version() {
        try {
            MemorySegment ptr = (MemorySegment) Native.ZTOK_VERSION.invoke();
            String s = Native.cString(ptr);
            if (s == null) {
                throw new ZtokException.Internal(
                    "ztok_version returned NULL", Native.ZTOK_ERR_INTERNAL);
            }
            return s;
        } catch (Throwable t) {
            if (t instanceof ZtokException ze) throw ze;
            throw new ZtokException.Internal(
                "ztok_version threw: " + t.getMessage(),
                Native.ZTOK_ERR_INTERNAL);
        }
    }

    /** Best-effort format sniffer. Delegates to {@link Pipeline#autoDetect}. */
    public static Format autoDetect(Path path) {
        return Pipeline.autoDetect(path);
    }
}
