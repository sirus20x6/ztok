// FuzzTests.cs — PRNG-driven round-trip fuzz harness.
//
// Mirrors fuzz/encode_decode.zig and the Python/Node/Ruby/Rust fuzz
// tests in shape: a deterministic PRNG mutates the byte input each
// iteration, encode-then-decode must round-trip exactly. The pipeline
// is byte_id (each input byte maps to one id), so for any byte sequence
// the invariant decode(encode(x)) == x must hold.
//
// Seed defaults to 0xFEEDB0B to match the Python/Ruby/Node/Rust
// harnesses — failures at a given iteration are cross-language
// reproducible. Override via FUZZ_SEED (hex or decimal) and
// ZTOK_FUZZ_ITERS for nightly runs.

using System;
using System.Collections.Generic;
using System.Globalization;
using Xunit;
using Ztok;

namespace Ztok.Tests;

public class FuzzTests
{
    // System.Random with a seeded int gives a deterministic stream. We
    // use the seeded int directly (no Random.Shared) so reproducibility
    // is unconditional.
    private const int DefaultSeed = unchecked((int)0x0FEEDB0B);
    private const int DefaultIterations = 1000;
    private const int MaxLen = 256;

    private static int EnvInt(string name, int defaultValue)
    {
        var raw = Environment.GetEnvironmentVariable(name);
        if (string.IsNullOrEmpty(raw)) return defaultValue;
        // Accept hex like "0xFEEDB0B" and decimal.
        if (raw.StartsWith("0x", StringComparison.OrdinalIgnoreCase) ||
            raw.StartsWith("0X", StringComparison.OrdinalIgnoreCase))
        {
            if (int.TryParse(raw.AsSpan(2), NumberStyles.HexNumber, CultureInfo.InvariantCulture, out var hex))
                return hex;
            return defaultValue;
        }
        return int.TryParse(raw, out var dec) ? dec : defaultValue;
    }

    [Fact]
    public void ByteIdRoundtripFuzz1000Iterations()
    {
        if (!SmokeTests.LibAvailable())
        {
            Console.Error.WriteLine("libztok not available; skipping fuzz harness.");
            return;
        }

        int seed = EnvInt("FUZZ_SEED", DefaultSeed);
        int iters = EnvInt("ZTOK_FUZZ_ITERS", DefaultIterations);
        var rng = new Random(seed);

        var failures = new List<(int Iter, byte[] Input, byte[] Output)>();

        using var pipe = Pipeline.ByteId();
        for (int i = 0; i < iters; i++)
        {
            int n = rng.Next(0, MaxLen + 1);
            var data = new byte[n];
            if (n > 0) rng.NextBytes(data);

            var ids = pipe.EncodeBytes(data);
            // byte_id maps 1:1: id count must equal input byte count.
            Assert.Equal(data.Length, ids.Length);

            var roundtrip = pipe.DecodeBytes(ids);
            if (!roundtrip.AsSpan().SequenceEqual(data))
            {
                failures.Add((i, data, roundtrip));
                if (failures.Count >= 5) break;
            }
        }

        if (failures.Count > 0)
        {
            var msg = "byte_id round-trip mismatches:\n";
            foreach (var (iter, inp, outp) in failures)
            {
                msg += $"  iter {iter}: in={Convert.ToHexString(inp)} out={Convert.ToHexString(outp)}\n";
            }
            Assert.Fail(msg);
        }
    }
}
