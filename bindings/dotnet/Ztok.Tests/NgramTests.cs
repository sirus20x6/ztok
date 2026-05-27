// NgramTests.cs — Engram n-gram hashing coverage for the .NET binding.
//
// Mirrors src/ngram.zig's contract and the Go/Python suites: deterministic
// multi-head token-n-gram hashes, row-major [position][head], with
// positions = ids.Length - n + 1.
//
// Skips cleanly (returns early) when libztok cannot be loaded, matching
// the skip pattern in SmokeTests.

using System;
using System.Linq;
using Xunit;
using Ztok;

namespace Ztok.Tests;

public class NgramTests
{
    private static bool LibAvailable() => SmokeTests.LibAvailable();

    [Fact]
    public void LengthMath()
    {
        if (!LibAvailable()) return;
        var ids = new uint[] { 1, 2, 3, 4, 5 };
        // 5 ids, n=2 -> 4 positions; heads=3 -> 12 hashes.
        var outHashes = Engram.HashNGrams(ids, n: 2, heads: 3);
        Assert.Equal(4 * 3, outHashes.Length);
    }

    [Fact]
    public void Deterministic()
    {
        if (!LibAvailable()) return;
        var ids = new uint[] { 7, 8, 9, 10, 11, 12 };
        var a = Engram.HashNGrams(ids, n: 3, heads: 4);
        var b = Engram.HashNGrams(ids, n: 3, heads: 4);
        Assert.Equal(a, b);
        Assert.NotEmpty(a);
    }

    [Fact]
    public void HeadIndependence()
    {
        if (!LibAvailable()) return;
        // The heads of a single position should not all collide.
        var outHashes = Engram.HashNGrams(new uint[] { 42, 43, 44 }, n: 2, heads: 4);
        var firstPosition = outHashes.Take(4).ToArray();
        Assert.True(firstPosition.Distinct().Count() > 1);
    }

    [Fact]
    public void ShortAndBadArgsReturnEmpty()
    {
        if (!LibAvailable()) return;
        // Stream shorter than one window -> empty.
        Assert.Empty(Engram.HashNGrams(new uint[] { 1, 2 }, n: 3, heads: 2));
        // Degenerate args -> empty (no error).
        Assert.Empty(Engram.HashNGrams(Array.Empty<uint>(), n: 1, heads: 1));
        Assert.Empty(Engram.HashNGrams(new uint[] { 1, 2, 3 }, n: 0, heads: 1));
        Assert.Empty(Engram.HashNGrams(new uint[] { 1, 2, 3 }, n: 2, heads: 0));
    }

    [Fact]
    public void BatchMatchesSingle()
    {
        if (!LibAvailable()) return;
        var streams = new[]
        {
            new uint[] { 1, 2, 3, 4 },
            Array.Empty<uint>(),       // empty -> no hashes
            new uint[] { 9 },          // shorter than window -> no hashes
            new uint[] { 5, 6, 7, 8, 9 },
        };
        using var pool = BatchPool.Create(2);
        var batched = Engram.HashNGramsBatch(pool, streams, n: 2, heads: 3);
        Assert.Equal(streams.Length, batched.Length);
        for (int i = 0; i < streams.Length; i++)
        {
            Assert.Equal(Engram.HashNGrams(streams[i], n: 2, heads: 3), batched[i]);
        }
    }
}
