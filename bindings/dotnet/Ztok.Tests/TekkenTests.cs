// TekkenTests.cs — Mistral Tekken tokenizer coverage for the .NET binding.
//
// Loads the real mistral_nemo_tekken.json fixture (skipped when absent)
// and checks ztok reproduces the canonical reference encodings, verified
// against mistral_common 1.8.6. Mirrors the C constructor
// ztok_pipeline_new_tekken_from_file and the Python reference impl.
//
// Skips cleanly (returns early) when libztok cannot be loaded, matching
// the skip pattern in SmokeTests.

using System;
using System.IO;
using Xunit;
using Ztok;

namespace Ztok.Tests;

public class TekkenTests
{
    private static bool LibAvailable() => SmokeTests.LibAvailable();

    // Golden id sequences verified against mistral_common 1.8.6 on
    // bench/vocabs/mistral_nemo_tekken.json.
    public static readonly (string Text, uint[] Want)[] Golden =
    {
        ("Hello, world!", new uint[] { 22177, 1044, 4304, 1033 }),
        ("The quick brown fox", new uint[] { 1784, 7586, 22980, 94137 }),
        (" and the", new uint[] { 1321, 1278 }),
    };

    // Walk up from the test assembly dir to the repo root and locate the
    // Tekken vocab fixture. Returns null if it can't be found.
    private static string? FindVocab()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        for (int i = 0; i < 10 && dir is not null; i++)
        {
            var probe = Path.Combine(dir.FullName, "bench", "vocabs", "mistral_nemo_tekken.json");
            if (File.Exists(probe)) return probe;
            dir = dir.Parent;
        }
        return null;
    }

    private static Pipeline? OpenTekken()
    {
        var vocab = FindVocab();
        if (vocab is null)
        {
            Console.Error.WriteLine("Tekken vocab fixture not present; skipping.");
            return null;
        }
        return Pipeline.FromTekken(vocab);
    }

    [Fact]
    public void MatchesReference()
    {
        if (!LibAvailable()) return;
        using var pipe = OpenTekken();
        if (pipe is null) return;
        foreach (var (text, want) in Golden)
        {
            Assert.Equal(want, pipe.Encode(text));
        }
    }

    [Fact]
    public void AutoDetectIsTekken()
    {
        if (!LibAvailable()) return;
        var vocab = FindVocab();
        if (vocab is null)
        {
            Console.Error.WriteLine("Tekken vocab fixture not present; skipping.");
            return;
        }
        Assert.Equal(Format.Tekken, ZtokLibrary.DetectFormat(vocab));
        // Open should dispatch to FromTekken via auto-detect.
        using var pipe = Pipeline.Open(vocab);
        Assert.Equal(new uint[] { 22177, 1044, 4304, 1033 }, pipe.Encode("Hello, world!"));
    }
}
