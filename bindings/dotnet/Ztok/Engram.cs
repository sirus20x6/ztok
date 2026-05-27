// Engram.cs — deterministic multi-head token-n-gram hashing.
//
// Wraps the C ABI's ztok_ngram_hash / ztok_ngram_hash_batch (see
// src/ngram.zig). These operate on raw token ids and need no Pipeline;
// the output is row-major [position][head] raw ulong hashes which the
// caller masks to its own table width (hash & ((1<<bits)-1)).
//
// No `unsafe` blocks live here — the pointer math lives in Native.cs.

using System;
using System.Collections.Generic;

namespace Ztok;

/// <summary>
/// Engram-style n-gram hashing for conditional-memory addressing.
/// Deterministic: identical ids always yield identical hashes. Operates
/// on raw token ids — no <see cref="Pipeline"/> needed.
/// </summary>
public static class Engram
{
    /// <summary>
    /// Hash every length-<paramref name="n"/> window of <paramref name="ids"/>
    /// under <paramref name="heads"/> independent hash functions, returning
    /// the row-major <c>[position][head]</c> ulong hashes
    /// (positions = <c>ids.Length - n + 1</c>, or 0 if the stream is shorter
    /// than one window). Mask each hash to your table width
    /// (<c>hash &amp; ((1 &lt;&lt; bits) - 1)</c>). Returns an empty array
    /// when there is nothing to hash (empty input, <c>n == 0</c>,
    /// <c>heads == 0</c>, or a stream shorter than <paramref name="n"/>).
    /// </summary>
    public static ulong[] HashNGrams(ReadOnlySpan<uint> ids, uint n, uint heads)
    {
        if (ids.IsEmpty || n == 0 || heads == 0 || ids.Length < n)
            return Array.Empty<ulong>();

        int positions = ids.Length - (int)n + 1;
        long want = (long)positions * heads;
        if (want == 0) return Array.Empty<ulong>();

        // We size exactly, so BUFFER_TOO_SMALL should never fire — but the
        // C contract permits it, so honor it with one grow attempt.
        int cap = checked((int)want);
        for (int attempt = 0; attempt < 2; attempt++)
        {
            var buf = new ulong[cap];
            int rc = Native.CallNgramHash(ids, n, heads, buf, out var outLen);
            if (rc == Native.Status.Ok)
            {
                int len = (int)outLen;
                if (len == buf.Length) return buf;
                var result = new ulong[len];
                Array.Copy(buf, result, len);
                return result;
            }
            if (rc == Native.Status.ErrBufferTooSmall)
            {
                cap = (int)outLen;
                if (cap == 0) return Array.Empty<ulong>();
                continue;
            }
            ZtokException.Check(rc, "ztok_ngram_hash");
        }
        throw new ZtokException(
            "ztok_ngram_hash: BUFFER_TOO_SMALL after 2 grow attempts",
            Native.Status.ErrBufferTooSmall);
    }

    /// <summary>
    /// Hash many id streams in parallel across <paramref name="pool"/>.
    /// <c>result[i]</c> holds the row-major hashes for <c>streams[i]</c>
    /// (empty for a stream shorter than one window). Equivalent to calling
    /// <see cref="HashNGrams"/> on each stream, fanned out across the pool's
    /// workers.
    /// </summary>
    public static ulong[][] HashNGramsBatch(BatchPool pool, IReadOnlyList<uint[]> streams, uint n, uint heads)
    {
        ArgumentNullException.ThrowIfNull(pool);
        ArgumentNullException.ThrowIfNull(streams);
        if (pool.IsDisposed) throw new ObjectDisposedException(nameof(BatchPool));
        if (streams.Count == 0) return Array.Empty<ulong[]>();
        return Native.CallNgramHashBatch(pool.Raw, streams, n, heads);
    }
}
