/*
 * Foreign Function & Memory API downcall handles for libztok.
 *
 * Every exported function from include/ztok.h is declared here as a
 * MethodHandle with the exact C signature. Calls go through these
 * handles directly — no JNI, no glue layer.
 *
 * The library handle is bound once into a global Arena (its lifetime
 * matches the JVM) and exposed as static final fields. Other packages
 * call e.g. `Native.ZTOK_ENCODE.invoke(...)` and let exceptions surface
 * as ZtokException via the wrappers in the parent package.
 */
package com.anthropic.ztok.internal;

import java.lang.foreign.AddressLayout;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;

public final class Native {
    private Native() {}

    // --- status / enum constants (mirror ztok.h) ---
    public static final int ZTOK_OK = 0;
    public static final int ZTOK_ERR_OUT_OF_MEMORY = 1;
    public static final int ZTOK_ERR_INVALID_INPUT = 2;
    public static final int ZTOK_ERR_BUFFER_TOO_SMALL = 3;
    public static final int ZTOK_ERR_INTERNAL = 99;

    public static final int NORMALIZER_IDENTITY = 0;
    public static final int NORMALIZER_NFC = 1;
    public static final int NORMALIZER_NFD = 2;
    public static final int NORMALIZER_NFKC = 3;
    public static final int NORMALIZER_NFKD = 4;
    public static final int NORMALIZER_BYTE_LEVEL = 5;

    public static final int PRETOK_IDENTITY = 0;
    public static final int PRETOK_CL100K = 1;

    public static final int MODEL_BYTE_ID = 0;

    public static final int DECODER_CONCAT = 0;
    public static final int DECODER_WORDPIECE = 1;
    public static final int DECODER_BYTE_LEVEL = 2;

    public static final int FORMAT_UNKNOWN = 0;
    public static final int FORMAT_TIKTOKEN = 1;
    public static final int FORMAT_HF_JSON = 2;
    public static final int FORMAT_SP_MODEL = 3;
    public static final int FORMAT_ZTM = 4;
    public static final int FORMAT_TEKKEN = 5;

    // Common layouts.
    public static final ValueLayout.OfInt   INT    = ValueLayout.JAVA_INT;
    public static final ValueLayout.OfLong  LONG   = ValueLayout.JAVA_LONG;
    public static final ValueLayout.OfByte  BYTE   = ValueLayout.JAVA_BYTE;
    public static final AddressLayout       PTR    = ValueLayout.ADDRESS;
    /** size_t — assume 64-bit pointers (Linux/macOS/Win64). */
    public static final ValueLayout.OfLong  SIZE_T = ValueLayout.JAVA_LONG;
    /** ztok_token_id is uint32_t; Java ints are signed but the bit pattern matches. */
    public static final ValueLayout.OfInt   TOKEN_ID = ValueLayout.JAVA_INT;

    // --- linker + library lookup (JVM-lifetime singleton) ---
    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup LIB =
        LibraryLoader.locate(Arena.global());

    private static MethodHandle dc(String symbol, FunctionDescriptor desc) {
        return LIB.find(symbol)
            .map(addr -> LINKER.downcallHandle(addr, desc))
            .orElseThrow(() ->
                new UnsatisfiedLinkError("libztok missing symbol: " + symbol));
    }

    // --- lifecycle ---
    public static final MethodHandle ZTOK_PIPELINE_NEW = dc(
        "ztok_pipeline_new",
        FunctionDescriptor.of(PTR, PTR, PTR));
    public static final MethodHandle ZTOK_PIPELINE_FREE = dc(
        "ztok_pipeline_free",
        FunctionDescriptor.ofVoid(PTR));
    public static final MethodHandle ZTOK_PIPELINE_NEW_BPE_FROM_TIKTOKEN = dc(
        "ztok_pipeline_new_bpe_from_tiktoken",
        FunctionDescriptor.of(PTR, PTR, PTR, PTR));
    public static final MethodHandle ZTOK_PIPELINE_NEW_BPE_FROM_HF_JSON = dc(
        "ztok_pipeline_new_bpe_from_hf_json",
        FunctionDescriptor.of(PTR, PTR, PTR, PTR));
    public static final MethodHandle ZTOK_PIPELINE_NEW_WORDPIECE_FROM_HF_JSON = dc(
        "ztok_pipeline_new_wordpiece_from_hf_json",
        FunctionDescriptor.of(PTR, PTR, INT, PTR, PTR));
    public static final MethodHandle ZTOK_PIPELINE_NEW_UNIGRAM_FROM_SP_MODEL = dc(
        "ztok_pipeline_new_unigram_from_sp_model",
        FunctionDescriptor.of(PTR, PTR, INT, PTR, PTR));
    public static final MethodHandle ZTOK_PIPELINE_NEW_MONSTER_FROM_FILE = dc(
        "ztok_pipeline_new_monster_from_file",
        FunctionDescriptor.of(PTR, PTR, PTR, PTR));

    // --- encode/decode ---
    public static final MethodHandle ZTOK_ENCODE = dc(
        "ztok_encode",
        FunctionDescriptor.of(INT, PTR, PTR, SIZE_T, PTR, SIZE_T, PTR));
    public static final MethodHandle ZTOK_DECODE = dc(
        "ztok_decode",
        FunctionDescriptor.of(INT, PTR, PTR, SIZE_T, PTR, SIZE_T, PTR));

    // ztok_encode_with_overlays(p, input, input_len, out_ids, out_ids_cap,
    //                           channels, n_channels, out_len)
    public static final MethodHandle ZTOK_ENCODE_WITH_OVERLAYS = dc(
        "ztok_encode_with_overlays",
        FunctionDescriptor.of(INT, PTR, PTR, SIZE_T, PTR, SIZE_T, PTR, SIZE_T, PTR));

    // --- batch / pool ---
    public static final MethodHandle ZTOK_ENCODE_BATCH = dc(
        "ztok_encode_batch",
        FunctionDescriptor.of(INT, PTR, PTR, PTR, SIZE_T, PTR, PTR, INT));
    public static final MethodHandle ZTOK_BATCH_POOL_NEW = dc(
        "ztok_batch_pool_new",
        FunctionDescriptor.of(PTR, INT, PTR));
    public static final MethodHandle ZTOK_BATCH_POOL_FREE = dc(
        "ztok_batch_pool_free",
        FunctionDescriptor.ofVoid(PTR));
    public static final MethodHandle ZTOK_BATCH_POOL_WORKER_COUNT = dc(
        "ztok_batch_pool_worker_count",
        FunctionDescriptor.of(SIZE_T, PTR));
    public static final MethodHandle ZTOK_ENCODE_BATCH_POOLED = dc(
        "ztok_encode_batch_pooled",
        FunctionDescriptor.of(INT, PTR, PTR, PTR, PTR, SIZE_T, PTR, PTR));
    public static final MethodHandle ZTOK_IDS_FREE = dc(
        "ztok_ids_free",
        FunctionDescriptor.ofVoid(PTR));

    // --- auto-detect ---
    public static final MethodHandle ZTOK_AUTO_DETECT = dc(
        "ztok_auto_detect",
        FunctionDescriptor.of(INT, PTR));

    // --- streaming ---
    public static final MethodHandle ZTOK_STREAM_NEW = dc(
        "ztok_stream_new",
        FunctionDescriptor.of(PTR, PTR, PTR));
    public static final MethodHandle ZTOK_STREAM_FREE = dc(
        "ztok_stream_free",
        FunctionDescriptor.ofVoid(PTR));
    public static final MethodHandle ZTOK_STREAM_FEED = dc(
        "ztok_stream_feed",
        FunctionDescriptor.of(INT, PTR, PTR, SIZE_T, PTR, PTR));
    public static final MethodHandle ZTOK_STREAM_FINISH = dc(
        "ztok_stream_finish",
        FunctionDescriptor.of(INT, PTR, PTR, PTR));

    // --- version / fingerprint ---
    public static final MethodHandle ZTOK_VERSION = dc(
        "ztok_version",
        FunctionDescriptor.of(PTR));
    public static final MethodHandle ZTOK_FINGERPRINT = dc(
        "ztok_fingerprint",
        FunctionDescriptor.of(INT, PTR, PTR));

    /** Reads a NUL-terminated C string from the given address into UTF-8. */
    public static String cString(MemorySegment ptr) {
        if (ptr == null || ptr.address() == 0) return null;
        // Reinterpret with unbounded size so getUtf8String walks until NUL.
        return ptr.reinterpret(Long.MAX_VALUE).getUtf8String(0);
    }
}
