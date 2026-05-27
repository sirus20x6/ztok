// ChunkTests.cs — token-window chunking coverage for the .NET binding.
//
// Run over a byte_id pipeline (each input byte = one token) so chunk
// boundaries are predictable: "abcdefghij" is 10 tokens, one per byte.
// Mirrors the Go/Python chunk suites.
//
// Skips cleanly (returns early) when libztok cannot be loaded, matching
// the skip pattern in SmokeTests.

using System;
using System.Linq;
using Xunit;
using Ztok;

namespace Ztok.Tests;

public class ChunkTests
{
    private static bool LibAvailable() => SmokeTests.LibAvailable();

    [Fact]
    public void NonOverlapping()
    {
        if (!LibAvailable()) return;
        using var pipe = Pipeline.ByteId();
        var chunks = pipe.Chunk("abcdefghij", maxTokens: 4, overlap: 0);
        // 10 tokens / window 4, stride 4 -> [0,4) [4,8) [8,10).
        Assert.Equal(3, chunks.Length);
        var want = new (uint ts, uint te, uint bs, uint be, int n)[]
        {
            (0, 4, 0, 4, 4),
            (4, 8, 4, 8, 4),
            (8, 10, 8, 10, 2),
        };
        for (int i = 0; i < want.Length; i++)
        {
            var c = chunks[i];
            Assert.Equal((want[i].ts, want[i].te), (c.TokenStart, c.TokenEnd));
            Assert.Equal((want[i].bs, want[i].be), (c.ByteStart, c.ByteEnd));
            Assert.Equal(want[i].n, c.Ids.Length);
        }
    }

    [Fact]
    public void Overlap()
    {
        if (!LibAvailable()) return;
        using var pipe = Pipeline.ByteId();
        var chunks = pipe.Chunk("abcdefghij", maxTokens: 4, overlap: 2);
        Assert.True(chunks.Length >= 2);
        // stride = 2, so the last 2 ids of chunk[i] equal the first 2 of
        // chunk[i+1].
        for (int i = 0; i + 1 < chunks.Length; i++)
        {
            var a = chunks[i];
            var b = chunks[i + 1];
            if (a.Ids.Length >= 2 && b.Ids.Length >= 2)
            {
                Assert.Equal(a.Ids[^2..], b.Ids[..2]);
            }
        }
    }

    [Fact]
    public void EmptyReturnsEmpty()
    {
        if (!LibAvailable()) return;
        using var pipe = Pipeline.ByteId();
        Assert.Empty(pipe.Chunk("", maxTokens: 4));
    }

    [Fact]
    public void BadArgsThrow()
    {
        if (!LibAvailable()) return;
        using var pipe = Pipeline.ByteId();
        Assert.Throws<ZtokInvalidInputException>(() => pipe.Chunk("abc", maxTokens: 0));
        Assert.Throws<ZtokInvalidInputException>(() => pipe.Chunk("abc", maxTokens: 4, overlap: 4));
    }
}
