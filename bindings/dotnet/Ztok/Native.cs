// Native.cs — raw P/Invoke surface for libztok.
//
// This is the only file in the binding that touches unsafe / extern code.
// Every higher-level wrapper (Pipeline, BatchPool, StreamEncoder) calls
// through here; the public API never exposes pointers.
//
// Mirrors include/ztok.h. Type map:
//
//   ztok_token_id      -> uint
//   ztok_status        -> int (see Status)
//   ztok_pipeline*     -> IntPtr (opaque)
//   ztok_batch_pool*   -> IntPtr (opaque)
//   ztok_stream*       -> IntPtr (opaque)
//   const char*        -> byte* (UTF-8) — we marshal manually to skip
//                         the default Ansi codepage round-trip
//   char*              -> byte*
//   size_t / size_t*   -> nuint / nuint*
//   uint32_t           -> uint
//
// Id-buffer header lifecycle (CRITICAL):
//   ztok_encode_batch_pooled / ztok_stream_feed / ztok_stream_finish
//   return per-input id arrays as `ztok_token_id*` whose bytes carry an
//   opaque length-prefix header (see src/c_api.zig::allocIdBuf). The
//   ONLY safe free for these buffers is ztok_ids_free — never call
//   Marshal.FreeHGlobal / NativeMemory.Free on them.

using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;

namespace Ztok;

/// <summary>
/// P/Invoke surface for libztok. Internal — public callers use
/// <see cref="Pipeline"/>, <see cref="BatchPool"/>, etc.
/// </summary>
internal static unsafe class Native
{
    // The DllImport name. The actual resolution (ZTOK_LIB_PATH env var,
    // candidate paths, system loader) happens in our DllImportResolver
    // installed by ModuleInit below.
    internal const string LibName = "ztok";

    // ----- module initializer ------------------------------------------------

    [System.Runtime.CompilerServices.ModuleInitializer]
    internal static void Init()
    {
        // Install once per process. SetDllImportResolver throws on a
        // duplicate registration, so guard with a flag.
        if (_resolverInstalled) return;
        _resolverInstalled = true;
        NativeLibrary.SetDllImportResolver(
            typeof(Native).Assembly,
            ResolveLibrary);
    }

    private static bool _resolverInstalled;

    private static IntPtr ResolveLibrary(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (!string.Equals(libraryName, LibName, StringComparison.Ordinal))
            return IntPtr.Zero;

        // 1. ZTOK_LIB_PATH explicit override.
        var overridePath = Environment.GetEnvironmentVariable("ZTOK_LIB_PATH");
        if (!string.IsNullOrEmpty(overridePath))
        {
            if (!File.Exists(overridePath))
                throw new ZtokException(
                    $"ZTOK_LIB_PATH=\"{overridePath}\" does not exist", Status.ErrInvalidInput);
            if (NativeLibrary.TryLoad(overridePath, out var handle))
                return handle;
            throw new ZtokException(
                $"ZTOK_LIB_PATH=\"{overridePath}\" exists but could not be loaded",
                Status.ErrInternal);
        }

        // 2. Candidate paths (assembly dir, repo zig-out, system dirs).
        foreach (var candidate in CandidatePaths())
        {
            if (!File.Exists(candidate)) continue;
            if (NativeLibrary.TryLoad(candidate, out var handle))
                return handle;
        }

        // 3. Default loader search (LD_LIBRARY_PATH, /usr/lib, rpath, etc.).
        //    Returning IntPtr.Zero hands resolution back to the runtime
        //    which will try the platform-conventional names (libztok.so,
        //    libztok.dylib, ztok.dll).
        return IntPtr.Zero;
    }

    private static IEnumerable<string> CandidatePaths()
    {
        var baseName = SharedLibBaseName();
        var asmDir = Path.GetDirectoryName(typeof(Native).Assembly.Location);
        if (!string.IsNullOrEmpty(asmDir))
        {
            yield return Path.Combine(asmDir, baseName);
            // RID-style: runtimes/{rid}/native/{baseName}
            yield return Path.Combine(asmDir, "runtimes", "linux-x64", "native", baseName);
            yield return Path.Combine(asmDir, "runtimes", "osx-x64", "native", baseName);
            yield return Path.Combine(asmDir, "runtimes", "osx-arm64", "native", baseName);
            yield return Path.Combine(asmDir, "runtimes", "win-x64", "native", baseName);

            // Walk up to repo root for in-tree dev:
            // bindings/dotnet/Ztok/bin/{Debug,Release}/net8.0 → repo root → zig-out/lib
            var repoZigOut = TryFindRepoZigOut(asmDir);
            if (repoZigOut is not null)
                yield return Path.Combine(repoZigOut, baseName);
        }

        if (OperatingSystem.IsLinux())
        {
            yield return Path.Combine("/usr/local/lib", baseName);
            yield return Path.Combine("/usr/lib", baseName);
            yield return Path.Combine("/usr/lib64", baseName);
        }
        else if (OperatingSystem.IsMacOS())
        {
            yield return Path.Combine("/usr/local/lib", baseName);
            yield return Path.Combine("/opt/homebrew/lib", baseName);
        }
    }

    private static string? TryFindRepoZigOut(string startDir)
    {
        var dir = new DirectoryInfo(startDir);
        for (int i = 0; i < 8 && dir is not null; i++)
        {
            var probe = Path.Combine(dir.FullName, "zig-out", "lib");
            if (Directory.Exists(probe)) return probe;
            dir = dir.Parent;
        }
        return null;
    }

    private static string SharedLibBaseName()
    {
        if (OperatingSystem.IsWindows()) return "ztok.dll";
        if (OperatingSystem.IsMacOS()) return "libztok.dylib";
        return "libztok.so";
    }

    // ----- enum constants (mirror include/ztok.h) ---------------------------

    internal static class Status
    {
        internal const int Ok = 0;
        internal const int ErrOutOfMemory = 1;
        internal const int ErrInvalidInput = 2;
        internal const int ErrBufferTooSmall = 3;
        internal const int ErrInternal = 99;
    }

    internal const uint NormalizerIdentity = 0;
    internal const uint NormalizerNfc = 1;
    internal const uint NormalizerNfd = 2;
    internal const uint NormalizerNfkc = 3;
    internal const uint NormalizerNfkd = 4;
    internal const uint NormalizerByteLevel = 5;

    internal const uint PretokIdentity = 0;
    internal const uint PretokCl100k = 1;
    internal const uint PretokTekken = 2;

    internal const uint ModelByteId = 0;

    internal const uint DecoderConcat = 0;
    internal const uint DecoderWordPiece = 1;
    internal const uint DecoderByteLevel = 2;

    internal const uint FormatUnknown = 0;
    internal const uint FormatTiktoken = 1;
    internal const uint FormatHfJson = 2;
    internal const uint FormatSpModel = 3;
    internal const uint FormatZtm = 4;
    internal const uint FormatTekken = 5;
    internal const uint FormatRwkv = 6;

    internal const uint ChunkBoundaryToken = 0;
    internal const uint ChunkBoundaryCodepoint = 1;
    internal const uint ChunkBoundaryWord = 2;
    internal const uint ChunkBoundaryWordDict = 3;
    internal const uint ChunkBoundarySentence = 4;
    internal const uint ChunkBoundaryParagraph = 5;

    // ----- struct mirrors ----------------------------------------------------

    [StructLayout(LayoutKind.Sequential)]
    internal struct ZtokPipelineConfig
    {
        internal uint Normalizer;
        internal uint PreTokenizer;
        internal uint Model;
        internal uint Decoder;
    }

    // ztok_overlay_channel: { ztok_overlay_kind kind; uint32_t* out; size_t out_cap; }
    // The C enum is int-sized; we mirror it as a uint here (the kind values
    // are small and non-negative). `Out` is a caller-owned uint32* buffer.
    [StructLayout(LayoutKind.Sequential)]
    internal struct ZtokOverlayChannel
    {
        internal uint Kind;
        internal uint* Out;
        internal nuint OutCap;
    }

    // ztok_chunk_rec: { ztok_token_id* ids; size_t ids_len; uint32_t
    // byte_start, byte_end, token_start, token_end; }. `Ids` points at a
    // ztok-allocated buffer of `IdsLen` token ids (NULL when IdsLen == 0),
    // released via ztok_chunks_free. `Byte*` is the half-open byte range
    // in the ORIGINAL input; `Token*` the half-open token-index range in
    // the full encoding.
    [StructLayout(LayoutKind.Sequential)]
    internal struct ZtokChunkRec
    {
        internal IntPtr Ids;
        internal nuint IdsLen;
        internal uint ByteStart;
        internal uint ByteEnd;
        internal uint TokenStart;
        internal uint TokenEnd;
    }

    // ----- lifecycle ---------------------------------------------------------

    [DllImport(LibName, EntryPoint = "ztok_pipeline_new")]
    internal static extern IntPtr PipelineNew(ZtokPipelineConfig* cfg, int* outStatus);

    [DllImport(LibName, EntryPoint = "ztok_pipeline_free")]
    internal static extern void PipelineFree(IntPtr p);

    [DllImport(LibName, EntryPoint = "ztok_pipeline_new_bpe_from_tiktoken")]
    internal static extern IntPtr PipelineNewBpeFromTiktoken(byte* path, ZtokPipelineConfig* cfg, int* outStatus);

    [DllImport(LibName, EntryPoint = "ztok_pipeline_new_bpe_from_hf_json")]
    internal static extern IntPtr PipelineNewBpeFromHfJson(byte* path, ZtokPipelineConfig* cfg, int* outStatus);

    [DllImport(LibName, EntryPoint = "ztok_pipeline_new_wordpiece_from_hf_json")]
    internal static extern IntPtr PipelineNewWordPieceFromHfJson(byte* path, uint unkId, ZtokPipelineConfig* cfg, int* outStatus);

    [DllImport(LibName, EntryPoint = "ztok_pipeline_new_unigram_from_sp_model")]
    internal static extern IntPtr PipelineNewUnigramFromSpModel(byte* path, uint unkId, ZtokPipelineConfig* cfg, int* outStatus);

    [DllImport(LibName, EntryPoint = "ztok_pipeline_new_monster_from_file")]
    internal static extern IntPtr PipelineNewMonsterFromFile(byte* path, ZtokPipelineConfig* cfg, int* outStatus);

    [DllImport(LibName, EntryPoint = "ztok_pipeline_new_rwkv_from_file")]
    internal static extern IntPtr PipelineNewRwkvFromFile(byte* path, ZtokPipelineConfig* cfg, int* outStatus);

    [DllImport(LibName, EntryPoint = "ztok_pipeline_new_tekken_from_file")]
    internal static extern IntPtr PipelineNewTekkenFromFile(byte* path, ZtokPipelineConfig* cfg, int* outStatus);

    // ----- encode / decode ---------------------------------------------------

    [DllImport(LibName, EntryPoint = "ztok_encode")]
    internal static extern int Encode(
        IntPtr p,
        byte* input, nuint inputLen,
        uint* outBuf, nuint outCap,
        nuint* outLen);

    [DllImport(LibName, EntryPoint = "ztok_decode")]
    internal static extern int Decode(
        IntPtr p,
        uint* ids, nuint idsLen,
        byte* outBuf, nuint outCap,
        nuint* outLen);

    [DllImport(LibName, EntryPoint = "ztok_encode_with_overlays")]
    internal static extern int EncodeWithOverlays(
        IntPtr p,
        byte* input, nuint inputLen,
        uint* outIds, nuint outIdsCap,
        ZtokOverlayChannel* channels, nuint nChannels,
        nuint* outLen);

    // ----- batch -------------------------------------------------------------

    [DllImport(LibName, EntryPoint = "ztok_encode_batch")]
    internal static extern int EncodeBatch(
        IntPtr p,
        byte** inputs, nuint* inputLens, nuint n,
        IntPtr* outIds, nuint* outLens,
        uint nWorkers);

    [DllImport(LibName, EntryPoint = "ztok_batch_pool_new")]
    internal static extern IntPtr BatchPoolNew(uint nWorkers, int* outStatus);

    [DllImport(LibName, EntryPoint = "ztok_batch_pool_free")]
    internal static extern void BatchPoolFree(IntPtr pool);

    [DllImport(LibName, EntryPoint = "ztok_batch_pool_worker_count")]
    internal static extern nuint BatchPoolWorkerCount(IntPtr pool);

    [DllImport(LibName, EntryPoint = "ztok_encode_batch_pooled")]
    internal static extern int EncodeBatchPooled(
        IntPtr p,
        IntPtr pool,
        byte** inputs, nuint* inputLens, nuint n,
        IntPtr* outIds, nuint* outLens);

    [DllImport(LibName, EntryPoint = "ztok_ids_free")]
    internal static extern void IdsFree(IntPtr ids);

    // ----- Engram n-gram hashing ---------------------------------------------

    [DllImport(LibName, EntryPoint = "ztok_ngram_hash")]
    internal static extern int NgramHash(
        uint* ids, nuint nIds,
        uint n, uint heads,
        ulong* outBuf, nuint outCap, nuint* outLen);

    [DllImport(LibName, EntryPoint = "ztok_ngram_hash_batch")]
    internal static extern int NgramHashBatch(
        IntPtr pool,
        uint** idArrays, nuint* idLens, nuint nDocs,
        uint n, uint heads,
        IntPtr* outHashes, nuint* outLens);

    [DllImport(LibName, EntryPoint = "ztok_u64s_free")]
    internal static extern void U64sFree(IntPtr hashes);

    // ----- chunking ----------------------------------------------------------

    [DllImport(LibName, EntryPoint = "ztok_chunk")]
    internal static extern int Chunk(
        IntPtr pipeline,
        byte* text, nuint textLen,
        uint maxTokens, uint overlap, uint boundary,
        ZtokChunkRec* outChunks, nuint outCap, nuint* outLen);

    [DllImport(LibName, EntryPoint = "ztok_chunks_free")]
    internal static extern void ChunksFree(ZtokChunkRec* chunks, nuint n);

    // ----- auto-detect -------------------------------------------------------

    [DllImport(LibName, EntryPoint = "ztok_auto_detect")]
    internal static extern uint AutoDetect(byte* path);

    // ----- streaming ---------------------------------------------------------

    [DllImport(LibName, EntryPoint = "ztok_stream_new")]
    internal static extern IntPtr StreamNew(IntPtr p, int* outStatus);

    [DllImport(LibName, EntryPoint = "ztok_stream_free")]
    internal static extern void StreamFree(IntPtr s);

    [DllImport(LibName, EntryPoint = "ztok_stream_feed")]
    internal static extern int StreamFeed(
        IntPtr s,
        byte* bytes, nuint nBytes,
        IntPtr* outIds, nuint* outNIds);

    [DllImport(LibName, EntryPoint = "ztok_stream_finish")]
    internal static extern int StreamFinish(
        IntPtr s,
        IntPtr* outIds, nuint* outNIds);

    // ----- version / fingerprint --------------------------------------------

    [DllImport(LibName, EntryPoint = "ztok_version")]
    internal static extern IntPtr Version();

    [DllImport(LibName, EntryPoint = "ztok_fingerprint")]
    internal static extern int Fingerprint(IntPtr handle, byte* out32);

    // ----- helpers (internal, still confined to Native.cs) -------------------

    /// <summary>UTF-8 encode a string into a NUL-terminated byte array.</summary>
    internal static byte[] CStringUtf8(string s)
    {
        var bytes = System.Text.Encoding.UTF8.GetBytes(s);
        var nul = new byte[bytes.Length + 1];
        Buffer.BlockCopy(bytes, 0, nul, 0, bytes.Length);
        nul[bytes.Length] = 0;
        return nul;
    }

    /// <summary>
    /// Copy an id buffer returned by libztok into a managed uint[] and
    /// free the native buffer via ztok_ids_free. ptr may be IntPtr.Zero
    /// (returns an empty array). The buffer at ptr has an opaque length
    /// header prefix; ztok_ids_free is the only safe free path.
    /// </summary>
    internal static uint[] MaterializeAndFreeIds(IntPtr ptr, nuint n)
    {
        if (ptr == IntPtr.Zero) return Array.Empty<uint>();
        if (n == 0)
        {
            IdsFree(ptr);
            return Array.Empty<uint>();
        }
        var managed = new uint[checked((int)n)];
        fixed (uint* dst = managed)
        {
            Buffer.MemoryCopy(
                (void*)ptr, dst,
                (long)n * sizeof(uint),
                (long)n * sizeof(uint));
        }
        IdsFree(ptr);
        return managed;
    }

    /// <summary>Decode a NUL-terminated UTF-8 C string returned by libztok.</summary>
    internal static string? PtrToStringUtf8(IntPtr ptr)
    {
        if (ptr == IntPtr.Zero) return null;
        return Marshal.PtrToStringUTF8(ptr);
    }

    /// <summary>
    /// Copy a u64 hash buffer returned by libztok into a managed ulong[]
    /// and free the native buffer via ztok_u64s_free. ptr may be
    /// IntPtr.Zero (returns an empty array). The buffer carries an opaque
    /// length-prefix header (see src/c_api.zig::allocU64Buf); ztok_u64s_free
    /// is the only safe free path.
    /// </summary>
    internal static ulong[] MaterializeAndFreeHashes(IntPtr ptr, nuint n)
    {
        if (ptr == IntPtr.Zero) return Array.Empty<ulong>();
        if (n == 0)
        {
            U64sFree(ptr);
            return Array.Empty<ulong>();
        }
        var managed = new ulong[checked((int)n)];
        fixed (ulong* dst = managed)
        {
            Buffer.MemoryCopy(
                (void*)ptr, dst,
                (long)n * sizeof(ulong),
                (long)n * sizeof(ulong));
        }
        U64sFree(ptr);
        return managed;
    }

    // ----- safe-ish call helpers --------------------------------------------
    //
    // These wrappers let the higher-level Pipeline / BatchPool / StreamEncoder
    // types stay free of `unsafe` blocks. Every pointer dereference lives
    // here in Native.cs. We use the raw P/Invokes' `byte*` / `ZtokPipelineConfig*`
    // surface internally; callers see only value/Span/array types.

    /// <summary>Which file-loading constructor to invoke. Keeps the
    /// caller-side surface free of pointer-typed delegates.</summary>
    internal enum PathCtorKind
    {
        BpeTiktoken,
        BpeHfJson,
        Monster,
        Rwkv,
        Tekken,
    }

    internal enum PathUnkCtorKind
    {
        WordPiece,
        Unigram,
    }

    internal static IntPtr CallPipelineNew(ZtokPipelineConfig cfg, string op)
    {
        int status = 0;
        IntPtr raw = PipelineNew(&cfg, &status);
        ZtokException.Check(status, op);
        if (raw == IntPtr.Zero)
            throw new ZtokException($"{op} returned NULL", Status.ErrInternal);
        return raw;
    }

    internal static IntPtr CallPathCfgCtor(string path, ZtokPipelineConfig cfg, PathCtorKind kind, string op)
    {
        var cpath = CStringUtf8(path);
        int status = 0;
        IntPtr raw;
        fixed (byte* p = cpath)
        {
            raw = kind switch
            {
                PathCtorKind.BpeTiktoken => PipelineNewBpeFromTiktoken(p, &cfg, &status),
                PathCtorKind.BpeHfJson => PipelineNewBpeFromHfJson(p, &cfg, &status),
                PathCtorKind.Monster => PipelineNewMonsterFromFile(p, &cfg, &status),
                PathCtorKind.Rwkv => PipelineNewRwkvFromFile(p, &cfg, &status),
                PathCtorKind.Tekken => PipelineNewTekkenFromFile(p, &cfg, &status),
                _ => throw new ArgumentOutOfRangeException(nameof(kind)),
            };
        }
        ZtokException.Check(status, op);
        if (raw == IntPtr.Zero)
            throw new ZtokException($"{op} returned NULL", Status.ErrInternal);
        return raw;
    }

    internal static IntPtr CallPathUnkCfgCtor(string path, uint unkId, ZtokPipelineConfig cfg, PathUnkCtorKind kind, string op)
    {
        var cpath = CStringUtf8(path);
        int status = 0;
        IntPtr raw;
        fixed (byte* p = cpath)
        {
            raw = kind switch
            {
                PathUnkCtorKind.WordPiece => PipelineNewWordPieceFromHfJson(p, unkId, &cfg, &status),
                PathUnkCtorKind.Unigram => PipelineNewUnigramFromSpModel(p, unkId, &cfg, &status),
                _ => throw new ArgumentOutOfRangeException(nameof(kind)),
            };
        }
        ZtokException.Check(status, op);
        if (raw == IntPtr.Zero)
            throw new ZtokException($"{op} returned NULL", Status.ErrInternal);
        return raw;
    }

    internal static uint AutoDetect(string path)
    {
        var cpath = CStringUtf8(path);
        fixed (byte* p = cpath)
        {
            return AutoDetect(p);
        }
    }

    /// <summary>Encode call wrapping the buffer-grow loop's single attempt.</summary>
    internal static int CallEncode(IntPtr pipeline, ReadOnlySpan<byte> input, Span<uint> outBuf, out nuint outLen)
    {
        nuint local = 0;
        int rc;
        fixed (byte* inputPtr = input)
        fixed (uint* outPtr = outBuf)
        {
            rc = Encode(
                pipeline,
                inputPtr, (nuint)input.Length,
                outPtr, (nuint)outBuf.Length,
                &local);
        }
        outLen = local;
        return rc;
    }

    /// <summary>Decode call. Pass an empty Span for the sizing pass.</summary>
    internal static int CallDecode(IntPtr pipeline, ReadOnlySpan<uint> ids, Span<byte> outBuf, out nuint outLen)
    {
        nuint local = 0;
        int rc;
        fixed (uint* idsPtr = ids)
        fixed (byte* outPtr = outBuf)
        {
            rc = Decode(
                pipeline,
                idsPtr, (nuint)ids.Length,
                outBuf.Length == 0 ? null : outPtr, (nuint)outBuf.Length,
                &local);
        }
        outLen = local;
        return rc;
    }

    /// <summary>
    /// Sizing pass for ztok_encode_with_overlays: pass out_ids = NULL so
    /// the C side reports the token count via out_len. The channel kinds
    /// are forwarded with NULL out buffers. Returns the raw status.
    /// </summary>
    internal static int CallEncodeWithOverlaysSize(
        IntPtr pipeline, ReadOnlySpan<byte> input, ReadOnlySpan<uint> kinds, out nuint outLen)
    {
        nuint local = 0;
        int rc;
        int nCh = kinds.Length;
        var chans = nCh == 0 ? null : new ZtokOverlayChannel[nCh];
        if (chans is not null)
        {
            for (int i = 0; i < nCh; i++)
                chans[i] = new ZtokOverlayChannel { Kind = kinds[i], Out = null, OutCap = 0 };
        }
        fixed (byte* inputPtr = input)
        fixed (ZtokOverlayChannel* chanPtr = chans)
        {
            rc = EncodeWithOverlays(
                pipeline,
                inputPtr, (nuint)input.Length,
                null, 0,
                chanPtr, (nuint)nCh,
                &local);
        }
        outLen = local;
        return rc;
    }

    /// <summary>
    /// Fill pass for ztok_encode_with_overlays. <paramref name="ids"/> and
    /// each row of <paramref name="channelBufs"/> must already be sized to
    /// the token count reported by the sizing pass. On return each channel
    /// buffer holds one value per token. Returns the raw status.
    /// </summary>
    internal static int CallEncodeWithOverlaysFill(
        IntPtr pipeline, ReadOnlySpan<byte> input,
        ReadOnlySpan<uint> kinds, uint[] ids, uint[][] channelBufs, out nuint outLen)
    {
        nuint local = 0;
        int rc;
        int nCh = kinds.Length;
        nuint count = (nuint)ids.Length;

        // Pin each channel buffer so the GC can't move it across the call,
        // then build the native channel descriptor array pointing at them.
        var pins = new GCHandle[nCh];
        try
        {
            var chans = nCh == 0 ? null : new ZtokOverlayChannel[nCh];
            for (int i = 0; i < nCh; i++)
            {
                pins[i] = GCHandle.Alloc(channelBufs[i], GCHandleType.Pinned);
                chans![i] = new ZtokOverlayChannel
                {
                    Kind = kinds[i],
                    Out = (uint*)pins[i].AddrOfPinnedObject(),
                    OutCap = count,
                };
            }
            fixed (byte* inputPtr = input)
            fixed (uint* idsPtr = ids)
            fixed (ZtokOverlayChannel* chanPtr = chans)
            {
                rc = EncodeWithOverlays(
                    pipeline,
                    inputPtr, (nuint)input.Length,
                    idsPtr, count,
                    chanPtr, (nuint)nCh,
                    &local);
            }
        }
        finally
        {
            for (int i = 0; i < nCh; i++)
                if (pins[i].IsAllocated) pins[i].Free();
        }
        outLen = local;
        return rc;
    }

    internal static int CallFingerprint(IntPtr pipeline, Span<byte> out32)
    {
        fixed (byte* p = out32)
        {
            return Fingerprint(pipeline, p);
        }
    }

    internal static IntPtr CallBatchPoolNew(uint workers)
    {
        int status = 0;
        IntPtr raw = BatchPoolNew(workers, &status);
        ZtokException.Check(status, "ztok_batch_pool_new");
        if (raw == IntPtr.Zero)
            throw new ZtokException("ztok_batch_pool_new returned NULL", Status.ErrInternal);
        return raw;
    }

    /// <summary>
    /// Persistent-pool batch encode. Takes already-encoded inputs as
    /// byte arrays. Always materializes every result slot (even on
    /// error) so partial allocations are freed via ztok_ids_free.
    /// </summary>
    internal static uint[][] CallEncodeBatchPooled(IntPtr pipeline, IntPtr pool, IReadOnlyList<byte[]> inputs)
    {
        int n = inputs.Count;
        if (n == 0) return Array.Empty<uint[]>();

        var pins = new GCHandle[n];
        var inputsPtrs = new IntPtr[n];
        var lensArr = new nuint[n];
        var outIdsArr = new IntPtr[n];
        var outLensArr = new nuint[n];
        try
        {
            for (int i = 0; i < n; i++)
            {
                var src = inputs[i] ?? Array.Empty<byte>();
                if (src.Length == 0)
                {
                    inputsPtrs[i] = IntPtr.Zero;
                    lensArr[i] = 0;
                }
                else
                {
                    pins[i] = GCHandle.Alloc(src, GCHandleType.Pinned);
                    inputsPtrs[i] = pins[i].AddrOfPinnedObject();
                    lensArr[i] = (nuint)src.Length;
                }
            }

            int rc;
            fixed (IntPtr* inputsPin = inputsPtrs)
            fixed (nuint* lensPin = lensArr)
            fixed (IntPtr* outIdsPin = outIdsArr)
            fixed (nuint* outLensPin = outLensArr)
            {
                rc = EncodeBatchPooled(
                    pipeline,
                    pool,
                    (byte**)inputsPin, lensPin, (nuint)n,
                    outIdsPin, outLensPin);
            }

            // Always materialize so we free everything, even on error.
            var results = new uint[n][];
            for (int i = 0; i < n; i++)
            {
                results[i] = MaterializeAndFreeIds(outIdsArr[i], outLensArr[i]);
            }
            ZtokException.Check(rc, "ztok_encode_batch_pooled");
            return results;
        }
        finally
        {
            for (int i = 0; i < n; i++)
            {
                if (pins[i].IsAllocated) pins[i].Free();
            }
        }
    }

    internal static IntPtr CallStreamNew(IntPtr pipeline)
    {
        int status = 0;
        IntPtr raw = StreamNew(pipeline, &status);
        ZtokException.Check(status, "ztok_stream_new");
        if (raw == IntPtr.Zero)
            throw new ZtokException("ztok_stream_new returned NULL", Status.ErrInternal);
        return raw;
    }

    /// <summary>Feed a chunk and return any newly-emitted ids (already freed).</summary>
    internal static uint[] CallStreamFeed(IntPtr stream, ReadOnlySpan<byte> chunk)
    {
        IntPtr outIds = IntPtr.Zero;
        nuint outN = 0;
        int rc;
        fixed (byte* p = chunk)
        {
            rc = StreamFeed(
                stream,
                chunk.Length == 0 ? null : p, (nuint)chunk.Length,
                &outIds, &outN);
        }
        if (rc != Status.Ok)
        {
            if (outIds != IntPtr.Zero) IdsFree(outIds);
            ZtokException.Check(rc, "ztok_stream_feed");
        }
        return MaterializeAndFreeIds(outIds, outN);
    }

    internal static uint[] CallStreamFinish(IntPtr stream)
    {
        IntPtr outIds = IntPtr.Zero;
        nuint outN = 0;
        int rc = StreamFinish(stream, &outIds, &outN);
        if (rc != Status.Ok)
        {
            if (outIds != IntPtr.Zero) IdsFree(outIds);
            ZtokException.Check(rc, "ztok_stream_finish");
        }
        return MaterializeAndFreeIds(outIds, outN);
    }

    // ----- n-gram hashing ----------------------------------------------------

    /// <summary>
    /// Hash every length-<paramref name="n"/> window of <paramref name="ids"/>
    /// under <paramref name="heads"/> hash functions into <paramref name="outBuf"/>.
    /// Returns the raw status; <paramref name="outLen"/> receives the count
    /// written (or required, on BUFFER_TOO_SMALL).
    /// </summary>
    internal static int CallNgramHash(
        ReadOnlySpan<uint> ids, uint n, uint heads, Span<ulong> outBuf, out nuint outLen)
    {
        nuint local = 0;
        int rc;
        fixed (uint* idsPtr = ids)
        fixed (ulong* outPtr = outBuf)
        {
            rc = NgramHash(
                idsPtr, (nuint)ids.Length,
                n, heads,
                outBuf.Length == 0 ? null : outPtr, (nuint)outBuf.Length,
                &local);
        }
        outLen = local;
        return rc;
    }

    /// <summary>
    /// Batch n-gram hash via a persistent pool. Each id stream is pinned
    /// for the duration of the call; every result slot is materialized
    /// (and freed via ztok_u64s_free) even on error so partial allocations
    /// can't leak.
    /// </summary>
    internal static ulong[][] CallNgramHashBatch(
        IntPtr pool, IReadOnlyList<uint[]> streams, uint n, uint heads)
    {
        int nDocs = streams.Count;
        if (nDocs == 0) return Array.Empty<ulong[]>();

        var pins = new GCHandle[nDocs];
        var idArrays = new IntPtr[nDocs];
        var idLens = new nuint[nDocs];
        var outHashes = new IntPtr[nDocs];
        var outLens = new nuint[nDocs];
        try
        {
            for (int i = 0; i < nDocs; i++)
            {
                var src = streams[i] ?? Array.Empty<uint>();
                idLens[i] = (nuint)src.Length;
                if (src.Length == 0)
                {
                    idArrays[i] = IntPtr.Zero;
                }
                else
                {
                    pins[i] = GCHandle.Alloc(src, GCHandleType.Pinned);
                    idArrays[i] = pins[i].AddrOfPinnedObject();
                }
            }

            int rc;
            fixed (IntPtr* idArraysPin = idArrays)
            fixed (nuint* idLensPin = idLens)
            fixed (IntPtr* outHashesPin = outHashes)
            fixed (nuint* outLensPin = outLens)
            {
                rc = NgramHashBatch(
                    pool,
                    (uint**)idArraysPin, idLensPin, (nuint)nDocs,
                    n, heads,
                    outHashesPin, outLensPin);
            }

            // Always materialize so we free everything, even on error.
            var results = new ulong[nDocs][];
            for (int i = 0; i < nDocs; i++)
                results[i] = MaterializeAndFreeHashes(outHashes[i], outLens[i]);
            ZtokException.Check(rc, "ztok_ngram_hash_batch");
            return results;
        }
        finally
        {
            for (int i = 0; i < nDocs; i++)
                if (pins[i].IsAllocated) pins[i].Free();
        }
    }

    // ----- chunking ----------------------------------------------------------

    /// <summary>One materialized chunk record (raw field values + copied ids).</summary>
    internal readonly struct ChunkData
    {
        internal readonly uint[] Ids;
        internal readonly uint ByteStart;
        internal readonly uint ByteEnd;
        internal readonly uint TokenStart;
        internal readonly uint TokenEnd;

        internal ChunkData(uint[] ids, uint byteStart, uint byteEnd, uint tokenStart, uint tokenEnd)
        {
            Ids = ids;
            ByteStart = byteStart;
            ByteEnd = byteEnd;
            TokenStart = tokenStart;
            TokenEnd = tokenEnd;
        }
    }

    /// <summary>
    /// Split <paramref name="text"/> into token-window chunks. Runs the C
    /// ABI's sizing pass (out_chunks = NULL → chunk count) then the fill
    /// pass, copies each record's ztok-allocated id buffer into managed
    /// memory, and frees the buffers via ztok_chunks_free before returning.
    /// </summary>
    internal static ChunkData[] CallChunk(
        IntPtr pipeline, ReadOnlySpan<byte> text,
        uint maxTokens, uint overlap, uint boundary)
    {
        if (text.IsEmpty) return Array.Empty<ChunkData>();

        // Sizing pass: out_chunks = NULL -> *out_len = chunk count.
        nuint need = 0;
        int rc;
        fixed (byte* textPtr = text)
        {
            rc = Chunk(
                pipeline, textPtr, (nuint)text.Length,
                maxTokens, overlap, boundary,
                null, 0, &need);
        }
        if (rc != Status.Ok && rc != Status.ErrBufferTooSmall)
            ZtokException.Check(rc, "ztok_chunk (sizing)");

        int count = (int)need;
        if (count == 0) return Array.Empty<ChunkData>();

        var recs = new ZtokChunkRec[count];
        nuint got = 0;
        fixed (byte* textPtr = text)
        fixed (ZtokChunkRec* recsPtr = recs)
        {
            rc = Chunk(
                pipeline, textPtr, (nuint)text.Length,
                maxTokens, overlap, boundary,
                recsPtr, (nuint)count, &got);
            if (rc != Status.Ok)
            {
                // Nothing allocated on a non-OK fill pass; report it.
                ZtokException.Check(rc, "ztok_chunk");
            }

            int n = (int)got;
            var result = new ChunkData[n];
            for (int i = 0; i < n; i++)
            {
                ref var r = ref recs[i];
                int idsLen = (int)r.IdsLen;
                uint[] ids;
                if (r.Ids != IntPtr.Zero && idsLen > 0)
                {
                    ids = new uint[idsLen];
                    fixed (uint* dst = ids)
                    {
                        Buffer.MemoryCopy(
                            (void*)r.Ids, dst,
                            (long)idsLen * sizeof(uint),
                            (long)idsLen * sizeof(uint));
                    }
                }
                else
                {
                    ids = Array.Empty<uint>();
                }
                result[i] = new ChunkData(ids, r.ByteStart, r.ByteEnd, r.TokenStart, r.TokenEnd);
            }
            // Release each record's ztok-allocated id buffer; the recs array
            // itself is managed (caller-owned).
            ChunksFree(recsPtr, got);
            return result;
        }
    }
}
