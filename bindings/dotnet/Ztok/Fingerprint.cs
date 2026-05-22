// Fingerprint.cs — typed wrapper around ztok_fingerprint.
//
// The fingerprint is a deterministic 32-byte SHA-256 digest over the
// pipeline's encoding behavior on a fixed canonical input set plus a
// model-kind tag and vocab size. Two pipelines that return the same 32
// bytes will produce bit-identical id streams for any input — use it as
// a cache key, KV-store discriminator, or training-pipeline guard.

using System;

namespace Ztok;

/// <summary>
/// 32-byte deterministic tokenizer fingerprint. Value type — cheap to
/// pass around, structural equality, fixed size buffer. Use
/// <see cref="ToHexString"/> for a printable form and the
/// <see cref="Bytes"/> span for raw access.
/// </summary>
public readonly struct Fingerprint : IEquatable<Fingerprint>
{
    /// <summary>Fixed digest size in bytes.</summary>
    public const int Size = 32;

    // Backing storage: two ulongs + two ulongs = 32 bytes. Using a fixed
    // 32-byte byte array would force heap allocation; an inline buffer
    // keeps Fingerprint a value type.
    private readonly ulong _b0, _b1, _b2, _b3;

    internal Fingerprint(ReadOnlySpan<byte> bytes)
    {
        if (bytes.Length != Size)
            throw new ArgumentException($"fingerprint must be exactly {Size} bytes, got {bytes.Length}", nameof(bytes));
        _b0 = BitConverter.ToUInt64(bytes.Slice(0, 8));
        _b1 = BitConverter.ToUInt64(bytes.Slice(8, 8));
        _b2 = BitConverter.ToUInt64(bytes.Slice(16, 8));
        _b3 = BitConverter.ToUInt64(bytes.Slice(24, 8));
    }

    /// <summary>Copy the 32 raw bytes into a fresh array.</summary>
    public byte[] ToArray()
    {
        var arr = new byte[Size];
        BitConverter.TryWriteBytes(arr.AsSpan(0, 8), _b0);
        BitConverter.TryWriteBytes(arr.AsSpan(8, 8), _b1);
        BitConverter.TryWriteBytes(arr.AsSpan(16, 8), _b2);
        BitConverter.TryWriteBytes(arr.AsSpan(24, 8), _b3);
        return arr;
    }

    /// <summary>Materialize the fingerprint bytes (allocates a 32-byte array).</summary>
    public byte[] Bytes => ToArray();

    /// <summary>Lowercase hexadecimal form (64 chars, no separators).</summary>
    public string ToHexString() => Convert.ToHexString(ToArray()).ToLowerInvariant();

    /// <inheritdoc />
    public override string ToString() => $"Fingerprint({ToHexString()})";

    /// <inheritdoc />
    public bool Equals(Fingerprint other) =>
        _b0 == other._b0 && _b1 == other._b1 && _b2 == other._b2 && _b3 == other._b3;

    /// <inheritdoc />
    public override bool Equals(object? obj) => obj is Fingerprint f && Equals(f);

    /// <inheritdoc />
    public override int GetHashCode() => HashCode.Combine(_b0, _b1, _b2, _b3);

    /// <summary>Structural equality.</summary>
    public static bool operator ==(Fingerprint a, Fingerprint b) => a.Equals(b);

    /// <summary>Structural inequality.</summary>
    public static bool operator !=(Fingerprint a, Fingerprint b) => !a.Equals(b);
}
