// SmokeTests.cs — basic coverage for the .NET ztok binding.
//
// Skips cleanly (returns early) when libztok cannot be loaded, mirroring
// the skip pattern in the other bindings. Set ZTOK_LIB_PATH (or
// LD_LIBRARY_PATH on Linux pointing to zig-out/lib) before `dotnet test`
// to enable the full suite.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;
using Ztok;

namespace Ztok.Tests;

public class SmokeTests
{
    // Returns true if libztok could be loaded. Caches the answer per
    // process so we only pay the dlopen cost once.
    private static bool? _libAvailable;
    internal static bool LibAvailable()
    {
        if (_libAvailable is not null) return _libAvailable.Value;
        try
        {
            var v = ZtokLibrary.Version;
            _libAvailable = !string.IsNullOrEmpty(v);
        }
        catch
        {
            _libAvailable = false;
        }
        return _libAvailable.Value;
    }

    private static void RequireLib()
    {
        if (!LibAvailable())
        {
            // xUnit doesn't ship Skip.If out of the box without an extra
            // package, so we follow the Rust/Go pattern: print a message
            // and return early. The test still counts as passed.
            Console.Error.WriteLine("libztok not available; skipping. Set ZTOK_LIB_PATH or LD_LIBRARY_PATH.");
        }
    }

    [Fact]
    public void VersionIsNonEmptyString()
    {
        if (!LibAvailable()) { RequireLib(); return; }
        var v = ZtokLibrary.Version;
        Assert.False(string.IsNullOrEmpty(v));
        Assert.Contains('.', v);
    }

    [Fact]
    public void ByteIdRoundtrip()
    {
        if (!LibAvailable()) { RequireLib(); return; }
        using var pipe = Pipeline.ByteId();
        var ids = pipe.Encode("hi");
        Assert.Equal(new uint[] { 0x68, 0x69 }, ids);
        Assert.Equal("hi", pipe.Decode(ids));
    }

    [Fact]
    public void EmptyInputReturnsEmpty()
    {
        if (!LibAvailable()) { RequireLib(); return; }
        using var pipe = Pipeline.ByteId();
        Assert.Empty(pipe.Encode(""));
        Assert.Equal("", pipe.Decode(Array.Empty<uint>()));
    }

    [Fact]
    public void DecodeBytesPreservesRawBytes()
    {
        if (!LibAvailable()) { RequireLib(); return; }
        using var pipe = Pipeline.ByteId();
        var ids = pipe.Encode("ab");
        Assert.Equal(new byte[] { (byte)'a', (byte)'b' }, pipe.DecodeBytes(ids));
    }

    [Fact]
    public void BatchPoolRoundtrip()
    {
        if (!LibAvailable()) { RequireLib(); return; }
        using var pipe = Pipeline.ByteId();
        using var pool = BatchPool.Create(2);
        Assert.True(pool.Workers >= 1);
        var inputs = new[] { "foo", "bar", "baz", "" };
        var results = pool.EncodeBatch(pipe, inputs);
        Assert.Equal(4, results.Length);
        Assert.Equal(new uint[] { (byte)'f', (byte)'o', (byte)'o' }, results[0]);
        Assert.Equal(new uint[] { (byte)'b', (byte)'a', (byte)'r' }, results[1]);
        Assert.Equal(new uint[] { (byte)'b', (byte)'a', (byte)'z' }, results[2]);
        Assert.Empty(results[3]);
    }

    [Fact]
    public void StreamEncodeRoundtrip()
    {
        if (!LibAvailable()) { RequireLib(); return; }
        using var pipe = Pipeline.ByteId();
        var collected = new List<uint>();
        foreach (var batch in pipe.EncodeStream("hello world", chunkSize: 4))
        {
            collected.AddRange(batch);
        }
        var expected = "hello world".Select(c => (uint)c).ToArray();
        Assert.Equal(expected, collected.ToArray());
        Assert.Equal("hello world", pipe.Decode(collected.ToArray()));
    }

    [Fact]
    public async Task StreamEncodeAsyncRoundtrip()
    {
        if (!LibAvailable()) { RequireLib(); return; }
        using var pipe = Pipeline.ByteId();
        var collected = new List<uint>();
        await foreach (var batch in pipe.EncodeStreamAsync("hello world", chunkSize: 3))
        {
            collected.AddRange(batch);
        }
        Assert.Equal("hello world".Select(c => (uint)c).ToArray(), collected.ToArray());
    }

    [Fact]
    public void StreamSinkApiRoundtrip()
    {
        if (!LibAvailable()) { RequireLib(); return; }
        using var pipe = Pipeline.ByteId();
        using var enc = StreamEncoder.Open(pipe);
        var collected = new List<uint>();
        byte[] data = Encoding.UTF8.GetBytes("hello world");
        for (int i = 0; i < data.Length; i += 3)
        {
            int end = Math.Min(i + 3, data.Length);
            collected.AddRange(enc.Feed(new ReadOnlySpan<byte>(data, i, end - i)));
        }
        collected.AddRange(enc.Finish());
        // Finish is idempotent.
        Assert.Empty(enc.Finish());
        Assert.Equal("hello world", pipe.Decode(collected.ToArray()));
    }

    [Fact]
    public void FingerprintIsDeterministic()
    {
        if (!LibAvailable()) { RequireLib(); return; }
        using var a = Pipeline.ByteId();
        using var b = Pipeline.ByteId();
        var fpA = a.Fingerprint();
        var fpB = b.Fingerprint();
        Assert.Equal(fpA, fpB);
        Assert.Equal(32, fpA.Bytes.Length);
        // Hex string is 64 lowercase chars.
        var hex = fpA.ToHexString();
        Assert.Equal(64, hex.Length);
        Assert.All(hex, c => Assert.True(char.IsDigit(c) || (c >= 'a' && c <= 'f')));
        // Not all zeros.
        Assert.Contains(fpA.Bytes, b => b != 0);
    }

    [Fact]
    public void DetectFormatUnknownForMissingFile()
    {
        if (!LibAvailable()) { RequireLib(); return; }
        var fmt = ZtokLibrary.DetectFormat("/definitely/does/not/exist.bin");
        Assert.Equal(Format.Unknown, fmt);
    }

    [Fact]
    public void InvalidNormalizerThrowsTypedException()
    {
        if (!LibAvailable()) { RequireLib(); return; }
        // The safe Normalizer enum can't hold an invalid value, so we
        // verify the error path through the C ABI by triggering an
        // unknown-format Open (which raises ZtokInvalidInputException).
        var ex = Assert.Throws<ZtokInvalidInputException>(() =>
            Pipeline.Open("/definitely/does/not/exist.bin"));
        Assert.Equal(ZtokStatus.InvalidInput, ex.Status);
    }

    [Fact]
    public void DisposeIsIdempotent()
    {
        if (!LibAvailable()) { RequireLib(); return; }
        var pipe = Pipeline.ByteId();
        pipe.Dispose();
        pipe.Dispose(); // Second call must not throw.
        Assert.True(pipe.IsDisposed);
        Assert.Throws<ObjectDisposedException>(() => pipe.Encode("x"));
    }
}
