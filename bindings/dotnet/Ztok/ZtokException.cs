// ZtokException — typed errors mapping ztok_status codes.

using System;

namespace Ztok;

/// <summary>
/// Mirrors <c>ztok_status</c> values from <c>include/ztok.h</c>. Use this
/// when you need to discriminate on the underlying error category without
/// catching the more specific subclasses.
/// </summary>
public enum ZtokStatus
{
    /// <summary>Success (ZTOK_OK = 0).</summary>
    Ok = 0,
    /// <summary>ZTOK_ERR_OUT_OF_MEMORY = 1.</summary>
    OutOfMemory = 1,
    /// <summary>ZTOK_ERR_INVALID_INPUT = 2.</summary>
    InvalidInput = 2,
    /// <summary>ZTOK_ERR_BUFFER_TOO_SMALL = 3.</summary>
    BufferTooSmall = 3,
    /// <summary>ZTOK_ERR_INTERNAL = 99 (or any unknown non-zero status).</summary>
    Internal = 99,
}

/// <summary>
/// Base exception for every error returned by libztok. The
/// <see cref="StatusCode"/> property carries the raw ztok_status int and
/// <see cref="Status"/> the typed enum. Specific subclasses
/// (<see cref="ZtokOutOfMemoryException"/>, etc.) are thrown for the
/// well-known codes so callers can catch them directly.
/// </summary>
public class ZtokException : Exception
{
    /// <summary>Raw ztok_status integer.</summary>
    public int StatusCode { get; }
    /// <summary>Typed status enum (Unknown maps to <see cref="ZtokStatus.Internal"/>).</summary>
    public ZtokStatus Status { get; }

    /// <param name="message">Operation name + context (e.g. "ztok_encode").</param>
    /// <param name="statusCode">Raw status from the C ABI.</param>
    public ZtokException(string message, int statusCode)
        : base($"{message}: ztok status {statusCode}")
    {
        StatusCode = statusCode;
        Status = statusCode switch
        {
            (int)ZtokStatus.OutOfMemory => ZtokStatus.OutOfMemory,
            (int)ZtokStatus.InvalidInput => ZtokStatus.InvalidInput,
            (int)ZtokStatus.BufferTooSmall => ZtokStatus.BufferTooSmall,
            _ => ZtokStatus.Internal,
        };
    }

    /// <summary>
    /// Translate a ztok_status into the appropriate typed exception.
    /// Returns null on <c>ZTOK_OK</c>. The returned exception is not
    /// thrown — callers do <c>throw FromStatus(...) ?? null!</c> or guard
    /// with a null-check.
    /// </summary>
    internal static ZtokException? FromStatus(int status, string op)
    {
        if (status == Native.Status.Ok) return null;
        return status switch
        {
            Native.Status.ErrOutOfMemory => new ZtokOutOfMemoryException(op, status),
            Native.Status.ErrInvalidInput => new ZtokInvalidInputException(op, status),
            Native.Status.ErrBufferTooSmall => new ZtokBufferTooSmallException(op, status),
            _ => new ZtokException(op, status),
        };
    }

    /// <summary>Throw if status is non-OK.</summary>
    internal static void Check(int status, string op)
    {
        var ex = FromStatus(status, op);
        if (ex is not null) throw ex;
    }
}

/// <summary>Maps ZTOK_ERR_OUT_OF_MEMORY.</summary>
public sealed class ZtokOutOfMemoryException : ZtokException
{
    /// <inheritdoc />
    public ZtokOutOfMemoryException(string message, int statusCode) : base(message, statusCode) { }
}

/// <summary>Maps ZTOK_ERR_INVALID_INPUT.</summary>
public sealed class ZtokInvalidInputException : ZtokException
{
    /// <inheritdoc />
    public ZtokInvalidInputException(string message, int statusCode) : base(message, statusCode) { }
}

/// <summary>
/// Maps ZTOK_ERR_BUFFER_TOO_SMALL. Most callers should never see this —
/// the binding grows internal buffers automatically.
/// </summary>
public sealed class ZtokBufferTooSmallException : ZtokException
{
    /// <inheritdoc />
    public ZtokBufferTooSmallException(string message, int statusCode) : base(message, statusCode) { }
}
