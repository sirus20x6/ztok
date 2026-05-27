// RwkvTests.cs — RWKV "World" tokenizer coverage for the .NET binding.
//
// Loads the real rwkv_vocab_v20230424.txt fixture (skipped when absent)
// and checks ztok reproduces the canonical reference encodings, matching
// the in-tree gate in src/rwkv_world.zig and the Go/Python suites.
//
// Skips cleanly (returns early) when libztok cannot be loaded, matching
// the skip pattern in SmokeTests.

using System;
using System.IO;
using Xunit;
using Ztok;

namespace Ztok.Tests;

public class RwkvTests
{
    private static bool LibAvailable() => SmokeTests.LibAvailable();

    // Golden id sequences captured from BlinkDL's canonical reference
    // tokenizer (see bench/rwkv_parity.py / src/rwkv_world.zig).
    public static readonly (string Text, uint[] Want)[] Golden =
    {
        ("Hello, world!", new uint[] { 33155, 45, 40213, 34 }),
        ("emoji \U0001F600\U0001F680✨ test",
            new uint[] { 34295, 33, 3319, 153, 129, 3319, 155, 129, 10059, 32223 }),
        ("0 1 2 10 99 100", new uint[] { 49, 284, 285, 3483, 3572, 3483, 49 }),
    };

    // Walk up from the test assembly dir to the repo root and locate the
    // RWKV vocab fixture. Returns null if it can't be found.
    private static string? FindVocab()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        for (int i = 0; i < 10 && dir is not null; i++)
        {
            var probe = Path.Combine(dir.FullName, "bench", "vocabs", "rwkv_vocab_v20230424.txt");
            if (File.Exists(probe)) return probe;
            dir = dir.Parent;
        }
        return null;
    }

    private static Pipeline? OpenRwkv()
    {
        var vocab = FindVocab();
        if (vocab is null)
        {
            Console.Error.WriteLine("RWKV vocab fixture not present; skipping.");
            return null;
        }
        return Pipeline.FromRwkv(vocab);
    }

    [Fact]
    public void MatchesReference()
    {
        if (!LibAvailable()) return;
        using var pipe = OpenRwkv();
        if (pipe is null) return;
        foreach (var (text, want) in Golden)
        {
            Assert.Equal(want, pipe.Encode(text));
        }
    }

    [Fact]
    public void RoundTrips()
    {
        if (!LibAvailable()) return;
        using var pipe = OpenRwkv();
        if (pipe is null) return;
        foreach (var (text, _) in Golden)
        {
            var ids = pipe.Encode(text);
            Assert.Equal(text, pipe.Decode(ids));
        }
    }

    [Fact]
    public void AutoDetectIsRwkv()
    {
        if (!LibAvailable()) return;
        var vocab = FindVocab();
        if (vocab is null)
        {
            Console.Error.WriteLine("RWKV vocab fixture not present; skipping.");
            return;
        }
        Assert.Equal(Format.Rwkv, ZtokLibrary.DetectFormat(vocab));
        // Open should dispatch to FromRwkv via auto-detect.
        using var pipe = Pipeline.Open(vocab);
        Assert.Equal(new uint[] { 33155, 45, 40213, 34 }, pipe.Encode("Hello, world!"));
    }
}
