// StreamEncoder.cs — streaming encode session.
//
// Wraps the C ABI `ztok_stream_*` family. Two surface variants are
// offered:
//
//   1. Sink-style ([StreamEncoder]): call Feed(bytes) repeatedly, then
//      Finish(). Each call returns the ids newly emitted. Useful when
//      you control the chunk cadence and want to slot the binding into
//      an existing stream pipeline.
//
//   2. Iterator-style ([Pipeline.EncodeStream] / [Pipeline.EncodeStreamAsync]):
//      pass the full bytes + a chunk size; the binding chops them up and
//      yields each non-empty id batch as it arrives. Synchronous
//      IEnumerable plus an IAsyncEnumerable variant for await foreach.
//
// No `unsafe` blocks here — Native.cs owns the pointer math.

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Ztok;

/// <summary>
/// Sink-style streaming encoder. Call <see cref="Feed(ReadOnlySpan{byte})"/>
/// repeatedly, then <see cref="Finish"/> to drain the encoder's carry.
/// Dispose to release the C-side stream handle.
/// </summary>
public sealed class StreamEncoder : IDisposable
{
    internal sealed class StreamHandle : SafeHandle
    {
        public StreamHandle() : base(IntPtr.Zero, ownsHandle: true) { }

        public override bool IsInvalid => handle == IntPtr.Zero;

        protected override bool ReleaseHandle()
        {
            if (handle != IntPtr.Zero)
            {
                Native.StreamFree(handle);
                SetHandle(IntPtr.Zero);
            }
            return true;
        }

        internal void SetRaw(IntPtr raw) => SetHandle(raw);
        internal IntPtr Raw => handle;
    }

    private readonly StreamHandle _handle;
    private bool _finished;

    private StreamEncoder(StreamHandle handle)
    {
        _handle = handle;
    }

    /// <summary>Open a new streaming encode session against <paramref name="pipeline"/>.</summary>
    public static StreamEncoder Open(Pipeline pipeline)
    {
        ArgumentNullException.ThrowIfNull(pipeline);
        var raw = Native.CallStreamNew(pipeline.Raw);
        var handle = new StreamHandle();
        handle.SetRaw(raw);
        return new StreamEncoder(handle);
    }

    /// <summary>True after <see cref="Dispose"/> has run.</summary>
    public bool IsDisposed => _handle.IsClosed || _handle.IsInvalid;

    private void ThrowIfDisposed()
    {
        if (IsDisposed) throw new ObjectDisposedException(nameof(StreamEncoder));
    }

    /// <summary>
    /// Feed a chunk of bytes. Returns the ids newly emitted by this call
    /// (possibly empty if the encoder is still buffering toward the next
    /// safe pre-tokenizer cut).
    /// </summary>
    public uint[] Feed(ReadOnlySpan<byte> chunk)
    {
        ThrowIfDisposed();
        return Native.CallStreamFeed(_handle.Raw, chunk);
    }

    /// <summary>
    /// Flush any remaining carry as a final encode. Idempotent: a second
    /// call returns an empty array.
    /// </summary>
    public uint[] Finish()
    {
        ThrowIfDisposed();
        if (_finished) return Array.Empty<uint>();
        var ids = Native.CallStreamFinish(_handle.Raw);
        _finished = true;
        return ids;
    }

    /// <inheritdoc />
    public void Dispose() => _handle.Dispose();
}

/// <summary>
/// Streaming encode extensions on <see cref="Pipeline"/>. Defined as
/// extensions rather than instance methods to keep the Pipeline file
/// focused on the load + encode/decode/fingerprint surface.
/// </summary>
public static class PipelineStreamExtensions
{
    /// <summary>Default per-feed chunk size in bytes (64 KiB).</summary>
    public const int DefaultChunkSize = 64 * 1024;

    /// <summary>
    /// Stream-encode <paramref name="text"/> (UTF-8) and yield each
    /// non-empty batch of ids as it's emitted. The synchronous
    /// counterpart to <see cref="EncodeStreamAsync(Pipeline, ReadOnlyMemory{byte}, int, CancellationToken)"/>.
    /// </summary>
    public static IEnumerable<uint[]> EncodeStream(this Pipeline pipeline, string text, int chunkSize = DefaultChunkSize)
    {
        ArgumentNullException.ThrowIfNull(pipeline);
        ArgumentNullException.ThrowIfNull(text);
        if (chunkSize <= 0) throw new ArgumentOutOfRangeException(nameof(chunkSize), "chunk size must be > 0");
        return EncodeStreamCore(pipeline, Encoding.UTF8.GetBytes(text), chunkSize);
    }

    /// <summary>Stream-encode raw bytes synchronously.</summary>
    public static IEnumerable<uint[]> EncodeStream(this Pipeline pipeline, byte[] data, int chunkSize = DefaultChunkSize)
    {
        ArgumentNullException.ThrowIfNull(pipeline);
        ArgumentNullException.ThrowIfNull(data);
        if (chunkSize <= 0) throw new ArgumentOutOfRangeException(nameof(chunkSize), "chunk size must be > 0");
        return EncodeStreamCore(pipeline, data, chunkSize);
    }

    private static IEnumerable<uint[]> EncodeStreamCore(Pipeline pipeline, byte[] data, int chunkSize)
    {
        using var enc = StreamEncoder.Open(pipeline);
        for (int i = 0; i < data.Length; i += chunkSize)
        {
            int end = Math.Min(i + chunkSize, data.Length);
            var ids = enc.Feed(new ReadOnlySpan<byte>(data, i, end - i));
            if (ids.Length > 0) yield return ids;
        }
        // Always run finish, even on empty input, to match the C
        // contract (and so an empty-input Feed/Finish sequence still
        // exercises the stream lifecycle).
        var final = enc.Finish();
        if (final.Length > 0) yield return final;
    }

    /// <summary>
    /// Async stream-encode. Yields each non-empty id batch as it's
    /// emitted; the work itself is synchronous (libztok is in-process),
    /// but the iterator can be awaited and cancelled from async code.
    /// </summary>
    public static async IAsyncEnumerable<uint[]> EncodeStreamAsync(
        this Pipeline pipeline,
        ReadOnlyMemory<byte> data,
        int chunkSize = DefaultChunkSize,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(pipeline);
        if (chunkSize <= 0) throw new ArgumentOutOfRangeException(nameof(chunkSize), "chunk size must be > 0");

        using var enc = StreamEncoder.Open(pipeline);
        for (int i = 0; i < data.Length; i += chunkSize)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int end = Math.Min(i + chunkSize, data.Length);
            var ids = enc.Feed(data.Span.Slice(i, end - i));
            if (ids.Length > 0) yield return ids;
            // Yield once per chunk so cooperative cancellation has a chance.
            await Task.Yield();
        }
        cancellationToken.ThrowIfCancellationRequested();
        var final = enc.Finish();
        if (final.Length > 0) yield return final;
    }

    /// <summary>Convenience: encode a UTF-8 string asynchronously.</summary>
    public static IAsyncEnumerable<uint[]> EncodeStreamAsync(
        this Pipeline pipeline,
        string text,
        int chunkSize = DefaultChunkSize,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(text);
        return EncodeStreamAsync(pipeline, Encoding.UTF8.GetBytes(text), chunkSize, cancellationToken);
    }
}
