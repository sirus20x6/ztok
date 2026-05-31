// OverlayTests.cs — coverage for Pipeline.EncodeWithOverlays
// (ztok_encode_with_overlays). Mirrors bindings/rust/tests/overlays.rs
// and the other bindings' overlay suites.
//
// Skips cleanly (returns early) when libztok cannot be loaded, matching
// the skip pattern in SmokeTests.

using System;
using System.IO;
using System.Linq;
using System.Text;
using Xunit;
using Ztok;

namespace Ztok.Tests;

public class OverlayTests
{
    private static bool LibAvailable() => SmokeTests.LibAvailable();

    // Build a synthetic .tiktoken vocab covering all 256 single bytes plus
    // a handful of merges — same shape as the Rust/Swift/Go fixtures.
    private static string? TiktokenFixture()
    {
        try
        {
            var dir = Path.Combine(
                Path.GetTempPath(),
                $"ztok-dotnet-ov-{Environment.ProcessId}");
            Directory.CreateDirectory(dir);
            var path = Path.Combine(dir, "synthetic_cl100k.tiktoken");
            if (File.Exists(path)) return path;

            var sb = new StringBuilder();
            uint rank = 0;
            for (int b = 0; b < 256; b++)
            {
                sb.Append(Convert.ToBase64String(new[] { (byte)b }));
                sb.Append(' ').Append(rank++).Append('\n');
            }
            foreach (var extra in new[]
            {
                "he", "hel", "hell", "hello", " w", " wo", " wor", " worl", " world",
                "th", "the", " th", " the", "fo", "foo", "bar", "baz",
                " quick", " brown", " fox",
            })
            {
                sb.Append(Convert.ToBase64String(Encoding.UTF8.GetBytes(extra)));
                sb.Append(' ').Append(rank++).Append('\n');
            }
            File.WriteAllText(path, sb.ToString());
            return path;
        }
        catch
        {
            return null;
        }
    }

    private static Pipeline? BpePipeline()
    {
        var fixture = TiktokenFixture();
        if (fixture is null) return null;
        return Pipeline.FromTiktoken(fixture,
            new PipelineConfig { PreTokenizer = PreTokenizer.Cl100k });
    }

    [Fact]
    public void IdsMatchPlainEncode()
    {
        if (!LibAvailable()) return;
        using var pipe = BpePipeline();
        if (pipe is null) return;
        var plain = pipe.Encode("hello world");
        var res = pipe.EncodeWithOverlays("hello world",
            OverlayKind.ByteStart, OverlayKind.ByteEnd);
        Assert.Equal(plain, res.Ids);
        Assert.Equal(2, res.Channels.Count);
        Assert.True(res.Channels.ContainsKey(OverlayKind.ByteStart));
        Assert.True(res.Channels.ContainsKey(OverlayKind.ByteEnd));
    }

    [Fact]
    public void ChannelLengthsEqualIds()
    {
        if (!LibAvailable()) return;
        using var pipe = BpePipeline();
        if (pipe is null) return;
        var res = pipe.EncodeWithOverlays("the quick brown fox",
            OverlayKind.ByteStart, OverlayKind.ByteEnd,
            OverlayKind.Boundary, OverlayKind.Provenance);
        foreach (var kv in res.Channels)
            Assert.Equal(res.Ids.Length, kv.Value.Length);
    }

    [Fact]
    public void ByteSpansAreSensible()
    {
        if (!LibAvailable()) return;
        using var pipe = BpePipeline();
        if (pipe is null) return;
        const string text = "hello world";
        var res = pipe.EncodeWithOverlays(text,
            OverlayKind.ByteStart, OverlayKind.ByteEnd);
        var starts = res.Channels[OverlayKind.ByteStart];
        var ends = res.Channels[OverlayKind.ByteEnd];
        uint n = (uint)Encoding.UTF8.GetByteCount(text);
        Assert.NotEmpty(starts);
        for (int i = 0; i < starts.Length; i++)
            Assert.True(starts[i] < ends[i] && ends[i] <= n,
                $"bad span ({starts[i]}, {ends[i]}) for {n} bytes");
        Assert.Equal(0u, starts[0]);
        Assert.Equal(n, ends[^1]);
        for (int i = 1; i < starts.Length; i++)
            Assert.Equal(ends[i - 1], starts[i]); // spans tile left-to-right
    }

    [Fact]
    public void ByteIdSingleByteSpans()
    {
        if (!LibAvailable()) return;
        using var pipe = Pipeline.ByteId();
        var res = pipe.EncodeWithOverlays("hi",
            OverlayKind.ByteStart, OverlayKind.ByteEnd);
        Assert.Equal(new uint[] { 0x68, 0x69 }, res.Ids);
        Assert.Equal(new uint[] { 0, 1 }, res.Channels[OverlayKind.ByteStart]);
        Assert.Equal(new uint[] { 1, 2 }, res.Channels[OverlayKind.ByteEnd]);
    }

    [Fact]
    public void OpcodeDomainChannelIsAllZero()
    {
        if (!LibAvailable()) return;
        using var pipe = BpePipeline();
        if (pipe is null) return;
        var res = pipe.EncodeWithOverlays("hello world", OverlayKind.Opcode);
        var opcode = res.Channels[OverlayKind.Opcode];
        Assert.Equal(res.Ids.Length, opcode.Length);
        Assert.All(opcode, v => Assert.Equal(0u, v));
    }

    [Fact]
    public void EmptyInputReturnsEmptyChannels()
    {
        if (!LibAvailable()) return;
        using var pipe = Pipeline.ByteId();
        var res = pipe.EncodeWithOverlays("",
            OverlayKind.ByteStart, OverlayKind.Opcode);
        Assert.Empty(res.Ids);
        Assert.Empty(res.Channels[OverlayKind.ByteStart]);
        Assert.Empty(res.Channels[OverlayKind.Opcode]);
    }

    [Fact]
    public void NoChannelsReturnsJustIds()
    {
        if (!LibAvailable()) return;
        using var pipe = BpePipeline();
        if (pipe is null) return;
        var res = pipe.EncodeWithOverlays("hello world");
        Assert.Equal(pipe.Encode("hello world"), res.Ids);
        Assert.Empty(res.Channels);
    }

    [Fact]
    public void DuplicateKindsRejected()
    {
        if (!LibAvailable()) return;
        using var pipe = Pipeline.ByteId();
        Assert.Throws<ZtokInvalidInputException>(() =>
            pipe.EncodeWithOverlays("hi", OverlayKind.ByteStart, OverlayKind.ByteStart));
    }

    // x86-64 machine code: 48 89 d8 (mov rax,rbx) / e8 00000000 (call rel32) /
    // c3 (ret). With ByteId each byte is its own token, so the OPCODE channel
    // carries one class per byte.
    private static readonly byte[] X86_64Code =
        { 0x48, 0x89, 0xd8, 0xe8, 0x00, 0x00, 0x00, 0x00, 0xc3 };

    [Fact]
    public void SetOverlayDomainX8664PopulatesOpcode()
    {
        if (!LibAvailable()) return;
        using var pipe = Pipeline.ByteId();

        // Default domain (None): OPCODE is zero-filled.
        var none = pipe.EncodeBytesWithOverlays(X86_64Code, OverlayKind.Opcode);
        var opcodeNone = none.Channels[OverlayKind.Opcode];
        Assert.Equal(X86_64Code.Length, opcodeNone.Length);
        Assert.All(opcodeNone, v => Assert.Equal(0u, v));

        // After selecting x86-64 the OPCODE channel is populated.
        pipe.SetOverlayDomain(OverlayDomain.X86_64);
        var x86 = pipe.EncodeBytesWithOverlays(X86_64Code, OverlayKind.Opcode);
        var opcodeX86 = x86.Channels[OverlayKind.Opcode];

        Assert.Equal(none.Ids, x86.Ids); // tokenization unchanged
        Assert.False(opcodeNone.SequenceEqual(opcodeX86), "domain channel must differ from None");
        Assert.Contains(opcodeX86, v => v != 0);
    }

    [Fact]
    public void SetOverlayDomainNoneRoundTrips()
    {
        if (!LibAvailable()) return;
        using var pipe = Pipeline.ByteId();
        pipe.SetOverlayDomain(OverlayDomain.X86_64);
        pipe.SetOverlayDomain(OverlayDomain.None);
        var res = pipe.EncodeBytesWithOverlays(X86_64Code, OverlayKind.Opcode);
        Assert.All(res.Channels[OverlayKind.Opcode], v => Assert.Equal(0u, v));
    }

    [Fact]
    public void SetOverlayDomainInvalidThrows()
    {
        if (!LibAvailable()) return;
        using var pipe = Pipeline.ByteId();
        Assert.Throws<ZtokInvalidInputException>(() =>
            pipe.SetOverlayDomain((OverlayDomain)999));
    }
}
