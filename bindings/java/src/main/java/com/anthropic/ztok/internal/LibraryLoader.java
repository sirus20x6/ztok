/*
 * libztok loader — mirrors bindings/python/ztok/_lib.py and
 * bindings/nodejs/lib.js.
 *
 * Resolution order:
 *   1. ZTOK_LIB_PATH env var (explicit absolute path).
 *   2. <cwd>/../../zig-out/lib/libztok.{so,dylib,dll} (in-tree dev path).
 *   3. Standard system paths: /usr/local/lib, /usr/lib, /usr/lib64,
 *      /opt/homebrew/lib on macOS.
 *   4. System.loadLibrary("ztok") — lets the platform dynamic loader
 *      walk java.library.path / LD_LIBRARY_PATH / etc.
 *
 * The C ABI's id buffers carry a length-prefix header (see
 * src/c_api.zig::allocIdBuf); the ONLY safe free is `ztok_ids_free`.
 */
package com.anthropic.ztok.internal;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.SymbolLookup;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public final class LibraryLoader {
    private LibraryLoader() {}

    /** Thrown when libztok cannot be located by any of the resolution strategies. */
    public static final class LibraryNotFoundException extends RuntimeException {
        public LibraryNotFoundException(String message) {
            super(message);
        }
    }

    /**
     * Locate and bind libztok. Returns a SymbolLookup scoped to the given
     * arena. Pass {@link Arena#global()} from a static initializer so the
     * library lives for the JVM lifetime (this matches the singleton model
     * other bindings use).
     */
    public static SymbolLookup locate(Arena arena) {
        // 1. Explicit env override.
        String override = System.getenv("ZTOK_LIB_PATH");
        if (override != null && !override.isEmpty()) {
            Path p = Paths.get(override);
            if (!Files.exists(p)) {
                throw new LibraryNotFoundException(
                    "ZTOK_LIB_PATH=" + override + " does not point to an existing file."
                );
            }
            return SymbolLookup.libraryLookup(p, arena);
        }

        // 2/3. Walk candidate paths.
        List<String> tried = new ArrayList<>();
        for (Path candidate : candidatePaths()) {
            tried.add(candidate.toString());
            if (Files.exists(candidate)) {
                try {
                    return SymbolLookup.libraryLookup(candidate, arena);
                } catch (IllegalArgumentException e) {
                    tried.set(tried.size() - 1,
                        candidate + " (dlopen failed: " + e.getMessage() + ")");
                }
            }
        }

        // 4. Bare-name fallback via System.loadLibrary; this throws on miss.
        tried.add("System.loadLibrary(\"ztok\")");
        try {
            System.loadLibrary("ztok");
            return SymbolLookup.loaderLookup();
        } catch (UnsatisfiedLinkError e) {
            tried.set(tried.size() - 1,
                "System.loadLibrary(\"ztok\") failed: " + e.getMessage());
        }

        throw new LibraryNotFoundException(
            "Could not locate libztok. Build it with `zig build` and either:\n"
            + "  - install it system-wide (e.g. cp zig-out/lib/libztok.so /usr/local/lib/),\n"
            + "  - place it on java.library.path, or\n"
            + "  - set ZTOK_LIB_PATH=/absolute/path/to/libztok.so.\n"
            + "Tried: " + String.join("; ", tried)
        );
    }

    /** Build the OS-appropriate candidate path list (most-specific first). */
    private static List<Path> candidatePaths() {
        String base = sharedLibBasename();
        String os = osName();
        List<Path> out = new ArrayList<>();

        // 2. In-tree dev: cwd -> repo root -> zig-out/lib
        // (relative-from-cwd lets tests run from bindings/java/ pick up
        // the freshly-built artifact without env tweaking).
        Path cwd = Paths.get("").toAbsolutePath();
        out.add(cwd.resolve("../../zig-out/lib").resolve(base).normalize());
        out.add(cwd.resolve("zig-out/lib").resolve(base).normalize());

        // 3. System paths.
        if (os.contains("linux")) {
            out.add(Paths.get("/usr/local/lib", base));
            out.add(Paths.get("/usr/lib", base));
            out.add(Paths.get("/usr/lib64", base));
        } else if (os.contains("mac") || os.contains("darwin")) {
            out.add(Paths.get("/usr/local/lib", base));
            out.add(Paths.get("/opt/homebrew/lib", base));
        }
        return out;
    }

    static String sharedLibBasename() {
        String os = osName();
        if (os.contains("mac") || os.contains("darwin")) return "libztok.dylib";
        if (os.contains("win")) return "ztok.dll";
        return "libztok.so";
    }

    private static String osName() {
        return System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
    }
}
