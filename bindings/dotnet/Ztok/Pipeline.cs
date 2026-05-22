// Pipeline.cs — managed, IDisposable wrapper around ztok_pipeline*.
//
// Construction goes through the static factory methods (Open,
// FromTiktoken, FromHfJson, ...). Disposal frees the C handle; the
// SafeHandle subclass acts as a finalizer-grade safety net if callers
// forget to dispose.
//
// Thread-safety: libztok's encode / decode / fingerprint paths take a
// `const ztok_pipeline*` and the underlying state is read-only after
// load (the worker pool lives on BatchPool, not here). It is therefore
// safe to call Encode / Decode / Fingerprint from multiple threads
// concurrently against the same Pipeline. Disposal must NOT race with
// other calls.
//
// No `unsafe` blocks live here — every pointer dereference is funneled
// through the safe-ish helper layer in Native.cs.

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Ztok;

/// <summary>
/// Optional pipeline configuration. The model kind is implicit (set by
/// the constructor used: BPE, WordPiece, Unigram, Monster, or byte_id).
/// </summary>
public sealed class PipelineConfig
{
    /// <summary>Unicode / byte-level normalization applied before pre-tokenization.</summary>
    public Normalizer Normalizer { get; init; } = Normalizer.Identity;
    /// <summary>Pre-tokenizer regex/strategy applied before the BPE merge loop.</summary>
    public PreTokenizer PreTokenizer { get; init; } = PreTokenizer.Identity;
    /// <summary>Decoder strategy applied when joining decoded token bytes.</summary>
    public Decoder Decoder { get; init; } = Decoder.Concat;

    internal Native.ZtokPipelineConfig ToCConfig() => new()
    {
        Normalizer = (uint)Normalizer,
        PreTokenizer = (uint)PreTokenizer,
        Model = Native.ModelByteId,
        Decoder = (uint)Decoder,
    };
}

/// <summary>
/// A loaded tokenizer pipeline. Construct via one of the static factory
/// methods (<see cref="Open"/>, <see cref="FromTiktoken"/>, etc.) and
/// dispose when done. The wrapped handle is freed by <see cref="Dispose"/>
/// (or the SafeHandle finalizer as a safety net).
///
/// <para>
/// Thread-safety: encode / decode / fingerprint are safe to call from
/// multiple threads concurrently against the same instance.
/// </para>
/// </summary>
public sealed class Pipeline : IDisposable
{
    // SafeHandle subclass that owns the ztok_pipeline*. SafeHandle gives
    // us stronger guarantees than a bare IntPtr: the runtime keeps the
    // handle alive across P/Invoke calls and serializes ReleaseHandle
    // against the finalizer.
    internal sealed class PipelineHandle : SafeHandle
    {
        public PipelineHandle() : base(IntPtr.Zero, ownsHandle: true) { }

        public override bool IsInvalid => handle == IntPtr.Zero;

        protected override bool ReleaseHandle()
        {
            if (handle != IntPtr.Zero)
            {
                Native.PipelineFree(handle);
                SetHandle(IntPtr.Zero);
            }
            return true;
        }

        internal void SetRaw(IntPtr raw) => SetHandle(raw);
        internal IntPtr Raw => handle;
    }

    private readonly PipelineHandle _handle;

    private Pipeline(PipelineHandle handle)
    {
        _handle = handle;
    }

    /// <summary>True after <see cref="Dispose"/> has run.</summary>
    public bool IsDisposed => _handle.IsClosed || _handle.IsInvalid;

    internal IntPtr Raw
    {
        get
        {
            ThrowIfDisposed();
            return _handle.Raw;
        }
    }

    private void ThrowIfDisposed()
    {
        if (IsDisposed) throw new ObjectDisposedException(nameof(Pipeline));
    }

    /// <inheritdoc />
    public void Dispose() => _handle.Dispose();

    // ----- constructors -----------------------------------------------------

    /// <summary>
    /// Build the byte_id baseline pipeline (each input byte maps to its
    /// own id). Useful for tests / fuzzing because the round-trip is
    /// guaranteed exact for any byte sequence.
    /// </summary>
    public static Pipeline ByteId(PipelineConfig? config = null)
    {
        var raw = Native.CallPipelineNew((config ?? new PipelineConfig()).ToCConfig(), "ztok_pipeline_new");
        return Wrap(raw);
    }

    /// <summary>
    /// Auto-detect <paramref name="path"/>'s vocab format and dispatch
    /// to the right loader. WordPiece (which lives inside
    /// <c>tokenizer.json</c> but needs a specific unk_id) is not
    /// covered — call <see cref="FromWordPiece"/> directly for that.
    /// </summary>
    public static Pipeline Open(string path)
    {
        ArgumentNullException.ThrowIfNull(path);
        var fmt = ZtokLibrary.DetectFormat(path);
        return fmt switch
        {
            Format.Tiktoken => FromTiktoken(path),
            Format.HfJson => FromHfJson(path),
            Format.SentencePiece => FromSentencePiece(path, unkId: 0),
            Format.Ztm => FromMonster(path),
            _ => throw new ZtokInvalidInputException(
                $"Open: could not auto-detect tokenizer format for '{path}'",
                Native.Status.ErrInvalidInput),
        };
    }

    /// <summary>
    /// Load a <c>.tiktoken</c> vocab into a byte-level BPE pipeline. By
    /// default the cl100k_base pre-tokenizer is used; pass a config with
    /// <see cref="PreTokenizer.Identity"/> to opt out.
    /// </summary>
    public static Pipeline FromTiktoken(string path, PipelineConfig? config = null)
    {
        ArgumentNullException.ThrowIfNull(path);
        var cfg = config ?? new PipelineConfig { PreTokenizer = PreTokenizer.Cl100k };
        var raw = Native.CallPathCfgCtor(path, cfg.ToCConfig(),
            Native.PathCtorKind.BpeTiktoken,
            "ztok_pipeline_new_bpe_from_tiktoken");
        return Wrap(raw);
    }

    /// <summary>Load a HuggingFace <c>tokenizer.json</c> BPE model.</summary>
    public static Pipeline FromHfJson(string path, PipelineConfig? config = null)
    {
        ArgumentNullException.ThrowIfNull(path);
        var cfg = config ?? new PipelineConfig();
        var raw = Native.CallPathCfgCtor(path, cfg.ToCConfig(),
            Native.PathCtorKind.BpeHfJson,
            "ztok_pipeline_new_bpe_from_hf_json");
        return Wrap(raw);
    }

    /// <summary>
    /// Load a HuggingFace WordPiece model from <c>tokenizer.json</c>.
    /// <paramref name="unkId"/> is required — there is no sensible default
    /// for an unknown-token id.
    /// </summary>
    public static Pipeline FromWordPiece(string path, uint unkId, PipelineConfig? config = null)
    {
        ArgumentNullException.ThrowIfNull(path);
        var cfg = config ?? new PipelineConfig { Decoder = Decoder.WordPiece };
        var raw = Native.CallPathUnkCfgCtor(path, unkId, cfg.ToCConfig(),
            Native.PathUnkCtorKind.WordPiece,
            "ztok_pipeline_new_wordpiece_from_hf_json");
        return Wrap(raw);
    }

    /// <summary>
    /// Load a SentencePiece <c>.model</c> (Unigram) file.
    /// <paramref name="unkId"/> defaults to 0 to match the Python binding.
    /// </summary>
    public static Pipeline FromSentencePiece(string path, uint unkId = 0, PipelineConfig? config = null)
    {
        ArgumentNullException.ThrowIfNull(path);
        var cfg = config ?? new PipelineConfig();
        var raw = Native.CallPathUnkCfgCtor(path, unkId, cfg.ToCConfig(),
            Native.PathUnkCtorKind.Unigram,
            "ztok_pipeline_new_unigram_from_sp_model");
        return Wrap(raw);
    }

    /// <summary>Load a ztok TokenMonster <c>.ztm</c> vocab file.</summary>
    public static Pipeline FromMonster(string path, PipelineConfig? config = null)
    {
        ArgumentNullException.ThrowIfNull(path);
        var cfg = config ?? new PipelineConfig();
        var raw = Native.CallPathCfgCtor(path, cfg.ToCConfig(),
            Native.PathCtorKind.Monster,
            "ztok_pipeline_new_monster_from_file");
        return Wrap(raw);
    }

    private static Pipeline Wrap(IntPtr raw)
    {
        var handle = new PipelineHandle();
        handle.SetRaw(raw);
        return new Pipeline(handle);
    }

    // ----- encode / decode --------------------------------------------------

    /// <summary>Encode <paramref name="text"/> (UTF-8) into a token id array.</summary>
    public uint[] Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (text.Length == 0) return Array.Empty<uint>();
        var bytes = Encoding.UTF8.GetBytes(text);
        return EncodeBytes(bytes);
    }

    /// <summary>
    /// Encode raw bytes into token ids. Use this for non-UTF-8 inputs;
    /// libztok treats input as a byte stream.
    /// </summary>
    public uint[] EncodeBytes(ReadOnlySpan<byte> data)
    {
        ThrowIfDisposed();
        if (data.IsEmpty) return Array.Empty<uint>();

        // libztok's per-span maxTokensFor upper bound is conservative,
        // so even a buffer sized to the true encoded length can trip
        // BUFFER_TOO_SMALL mid-stream. Start generous, double on retry,
        // cap at 8 attempts (matches the Python/Go/Rust bindings).
        int cap = Math.Max(data.Length + 16, 64);
        for (int attempt = 0; attempt < 8; attempt++)
        {
            var buf = new uint[cap];
            int rc = Native.CallEncode(_handle.Raw, data, buf, out var outLen);
            if (rc == Native.Status.Ok)
            {
                int n = (int)outLen;
                if (n == buf.Length) return buf;
                var result = new uint[n];
                Array.Copy(buf, result, n);
                return result;
            }
            if (rc == Native.Status.ErrBufferTooSmall)
            {
                cap = Math.Max(cap * 2, (int)outLen + 16);
                continue;
            }
            ZtokException.Check(rc, "ztok_encode");
        }
        throw new ZtokException(
            "ztok_encode: BUFFER_TOO_SMALL after 8 grow attempts",
            Native.Status.ErrBufferTooSmall);
    }

    /// <summary>
    /// Decode token ids back to a UTF-8 string. Invalid UTF-8 is
    /// preserved using the replacement character; use
    /// <see cref="DecodeBytes"/> for raw bytes.
    /// </summary>
    public string Decode(ReadOnlySpan<uint> ids)
    {
        var bytes = DecodeBytes(ids);
        return Encoding.UTF8.GetString(bytes);
    }

    /// <summary>Decode token ids to raw bytes (no UTF-8 round-tripping).</summary>
    public byte[] DecodeBytes(ReadOnlySpan<uint> ids)
    {
        ThrowIfDisposed();
        if (ids.IsEmpty) return Array.Empty<byte>();

        // Sizing pass: pass an empty span so libztok writes the required
        // size into out_len and returns BUFFER_TOO_SMALL.
        int rc = Native.CallDecode(_handle.Raw, ids, Span<byte>.Empty, out var sized);
        if (rc != Native.Status.Ok && rc != Native.Status.ErrBufferTooSmall)
            ZtokException.Check(rc, "ztok_decode (sizing)");
        if (sized == 0) return Array.Empty<byte>();

        var buf = new byte[(int)sized];
        rc = Native.CallDecode(_handle.Raw, ids, buf, out var written);
        ZtokException.Check(rc, "ztok_decode");
        int n = (int)written;
        if (n == buf.Length) return buf;
        var result = new byte[n];
        Array.Copy(buf, result, n);
        return result;
    }

    // ----- encode with overlays ---------------------------------------------

    /// <summary>
    /// Result of <see cref="EncodeWithOverlays"/>: the token id stream plus
    /// a map of requested <see cref="OverlayKind"/> to its per-token value
    /// array. Each channel array has the same length as <see cref="Ids"/>.
    /// </summary>
    public sealed class OverlayResult
    {
        /// <summary>The token ids — identical to what <see cref="Encode"/> returns.</summary>
        public uint[] Ids { get; }

        /// <summary>
        /// Per-token overlay channels keyed by kind. Each value array is
        /// aligned 1:1 with <see cref="Ids"/>.
        /// </summary>
        public IReadOnlyDictionary<OverlayKind, uint[]> Channels { get; }

        internal OverlayResult(uint[] ids, IReadOnlyDictionary<OverlayKind, uint[]> channels)
        {
            Ids = ids;
            Channels = channels;
        }
    }

    /// <summary>
    /// Encode <paramref name="text"/> (UTF-8) and return the ids plus a map
    /// of per-token overlay channels aligned 1:1 with the id stream.
    /// Requesting overlays never changes tokenization — the ids are
    /// identical to <see cref="Encode"/>. Cheap channels carry
    /// encoder-derived values; domain channels come back zero-filled until
    /// a domain plugin populates them. Duplicate kinds are rejected with
    /// <see cref="ZtokInvalidInputException"/>.
    /// </summary>
    public OverlayResult EncodeWithOverlays(string text, params OverlayKind[] channels)
    {
        ArgumentNullException.ThrowIfNull(text);
        var bytes = text.Length == 0 ? Array.Empty<byte>() : Encoding.UTF8.GetBytes(text);
        return EncodeBytesWithOverlays(bytes, channels);
    }

    /// <summary>
    /// Raw-bytes form of <see cref="EncodeWithOverlays"/>. Use this for
    /// non-UTF-8 inputs; libztok treats input as a byte stream.
    /// </summary>
    public OverlayResult EncodeBytesWithOverlays(ReadOnlySpan<byte> data, params OverlayKind[] channels)
    {
        ThrowIfDisposed();
        channels ??= Array.Empty<OverlayKind>();

        // Reject duplicate kinds — the result map would silently collapse
        // them, which is almost certainly a caller bug.
        for (int i = 0; i < channels.Length; i++)
        {
            for (int j = 0; j < i; j++)
            {
                if (channels[i] == channels[j])
                    throw new ZtokInvalidInputException(
                        $"EncodeWithOverlays: duplicate channel kind {channels[i]}",
                        Native.Status.ErrInvalidInput);
            }
        }

        var kinds = new uint[channels.Length];
        for (int i = 0; i < channels.Length; i++) kinds[i] = (uint)channels[i];

        OverlayResult Empty()
        {
            var map = new Dictionary<OverlayKind, uint[]>(channels.Length);
            foreach (var k in channels) map[k] = Array.Empty<uint>();
            return new OverlayResult(Array.Empty<uint>(), map);
        }

        if (data.IsEmpty) return Empty();

        // Sizing pass: out_ids = NULL queries the token count.
        int rc = Native.CallEncodeWithOverlaysSize(_handle.Raw, data, kinds, out var sized);
        if (rc != Native.Status.Ok && rc != Native.Status.ErrBufferTooSmall)
            ZtokException.Check(rc, "ztok_encode_with_overlays (sizing)");

        int count = (int)sized;
        if (count == 0) return Empty();

        // Fill pass: allocate the id buffer + one uint[] per channel,
        // each sized to the exact count.
        var ids = new uint[count];
        var channelBufs = new uint[channels.Length][];
        for (int i = 0; i < channels.Length; i++) channelBufs[i] = new uint[count];

        rc = Native.CallEncodeWithOverlaysFill(
            _handle.Raw, data, kinds, ids, channelBufs, out var written);
        ZtokException.Check(rc, "ztok_encode_with_overlays");

        int n = (int)written;
        uint[] Trim(uint[] buf) =>
            n == buf.Length ? buf : buf[..n];

        var result = new Dictionary<OverlayKind, uint[]>(channels.Length);
        for (int i = 0; i < channels.Length; i++)
            result[channels[i]] = Trim(channelBufs[i]);
        return new OverlayResult(Trim(ids), result);
    }

    // ----- fingerprint ------------------------------------------------------

    /// <summary>
    /// Compute the tokenizer fingerprint — a deterministic 32-byte
    /// SHA-256 digest over the pipeline's encoding behavior on a fixed
    /// canonical input set plus model-kind tag and vocab size. Two
    /// pipelines that return the same value will produce bit-identical
    /// id streams for any input. New in libztok 1.22.
    /// </summary>
    public Fingerprint Fingerprint()
    {
        ThrowIfDisposed();
        Span<byte> bytes = stackalloc byte[32]; // Fingerprint.Size
        int rc = Native.CallFingerprint(_handle.Raw, bytes);
        ZtokException.Check(rc, "ztok_fingerprint");
        return new Fingerprint(bytes);
    }
}
