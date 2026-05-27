// BatchPool.cs — persistent multithreaded worker pool.
//
// libztok exposes two batch entry points:
//   - ztok_encode_batch        : spawns a worker pool per call (wasteful)
//   - ztok_encode_batch_pooled : reuses arenas + worker threads
//
// Like every other binding (Python, Node, Ruby, Go, Rust), we only
// expose the pooled variant. Create one BatchPool, reuse it across many
// EncodeBatch calls.
//
// No `unsafe` blocks live here — the parallel-array pointer juggling
// lives in Native.CallEncodeBatchPooled.

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Ztok;

/// <summary>
/// Persistent multithreaded worker pool. Reuse one instance across many
/// <see cref="EncodeBatch(Pipeline, IEnumerable{string})"/> calls — each
/// pool owns its own arenas and worker threads, so creating one per
/// batch wastes setup work.
/// </summary>
public sealed class BatchPool : IDisposable
{
    internal sealed class BatchPoolHandle : SafeHandle
    {
        public BatchPoolHandle() : base(IntPtr.Zero, ownsHandle: true) { }

        public override bool IsInvalid => handle == IntPtr.Zero;

        protected override bool ReleaseHandle()
        {
            if (handle != IntPtr.Zero)
            {
                Native.BatchPoolFree(handle);
                SetHandle(IntPtr.Zero);
            }
            return true;
        }

        internal void SetRaw(IntPtr raw) => SetHandle(raw);
        internal IntPtr Raw => handle;
    }

    private readonly BatchPoolHandle _handle;

    private BatchPool(BatchPoolHandle handle)
    {
        _handle = handle;
    }

    /// <summary>
    /// Create a new pool with <paramref name="workers"/> worker threads.
    /// Pass 0 to auto-detect the cpu count.
    /// </summary>
    public static BatchPool Create(uint workers = 0)
    {
        var raw = Native.CallBatchPoolNew(workers);
        var handle = new BatchPoolHandle();
        handle.SetRaw(raw);
        return new BatchPool(handle);
    }

    /// <summary>True after <see cref="Dispose"/> has run.</summary>
    public bool IsDisposed => _handle.IsClosed || _handle.IsInvalid;

    /// <summary>
    /// Actual worker count (resolves the 0=auto request to the detected
    /// cpu count at creation time).
    /// </summary>
    public int Workers
    {
        get
        {
            ThrowIfDisposed();
            return checked((int)Native.BatchPoolWorkerCount(_handle.Raw));
        }
    }

    /// <inheritdoc />
    public void Dispose() => _handle.Dispose();

    /// <summary>Raw pool handle for internal callers (Engram batch path).</summary>
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
        if (IsDisposed) throw new ObjectDisposedException(nameof(BatchPool));
    }

    /// <summary>
    /// Encode many strings in parallel via this pool. Inputs are
    /// converted to UTF-8 before the call. Each per-input result is
    /// copied into a managed <c>uint[]</c>; the underlying C buffers are
    /// freed via <c>ztok_ids_free</c> before this method returns.
    /// </summary>
    public uint[][] EncodeBatch(Pipeline pipeline, IEnumerable<string> inputs)
    {
        ArgumentNullException.ThrowIfNull(pipeline);
        ArgumentNullException.ThrowIfNull(inputs);
        ThrowIfDisposed();

        var byteInputs = new List<byte[]>();
        foreach (var s in inputs)
        {
            byteInputs.Add(s is null ? Array.Empty<byte>() : Encoding.UTF8.GetBytes(s));
        }
        return EncodeBatchBytes(pipeline, byteInputs);
    }

    /// <summary>Encode many raw-byte inputs in parallel via this pool.</summary>
    public uint[][] EncodeBatchBytes(Pipeline pipeline, IReadOnlyList<byte[]> inputs)
    {
        ArgumentNullException.ThrowIfNull(pipeline);
        ArgumentNullException.ThrowIfNull(inputs);
        ThrowIfDisposed();
        return Native.CallEncodeBatchPooled(pipeline.Raw, _handle.Raw, inputs);
    }
}
