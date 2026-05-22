/*
 * Shared probe used by every test class to decide whether to skip on
 * environments without a built libztok. Touching Ztok.version() runs
 * the LibraryLoader exactly once — subsequent calls hit the JVM-lifetime
 * cache.
 */
package com.anthropic.ztok;

final class TestSupport {
    private TestSupport() {}

    private static volatile Boolean cached;

    static boolean libztokAvailable() {
        Boolean v = cached;
        if (v != null) return v;
        synchronized (TestSupport.class) {
            if (cached != null) return cached;
            try {
                Ztok.version();
                cached = Boolean.TRUE;
            } catch (Throwable t) {
                cached = Boolean.FALSE;
            }
            return cached;
        }
    }
}
