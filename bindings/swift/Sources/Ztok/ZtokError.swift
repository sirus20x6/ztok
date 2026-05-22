// ZtokError — typed Swift error enum mapping ztok_status to idiomatic
// Swift. Every fallible API in this module throws ZtokError; the
// variants mirror the C status codes 1:1 with `.internalError` as the
// catch-all for unknown status integers.
//
// Mirrors the shape of the Rust binding's `Error` enum and the Java
// binding's `ZtokException.*` hierarchy.

import CZtok

/// Swift error type covering every failure surface in the ztok binding.
///
/// Variants are deliberately non-exhaustive-friendly: callers should
/// match the cases they care about and treat unknown status integers as
/// `.internalError`.
public enum ZtokError: Error, Equatable, CustomStringConvertible, Sendable {
    /// Maps `ZTOK_ERR_OUT_OF_MEMORY` (status 1).
    case outOfMemory

    /// Maps `ZTOK_ERR_INVALID_INPUT` (status 2). Covers bad arguments,
    /// unknown enum kinds, and (for `Pipeline.open`) format-detection
    /// failures.
    case invalidInput

    /// Maps `ZTOK_ERR_BUFFER_TOO_SMALL` (status 3). The wrapper grows
    /// buffers automatically, so callers should rarely see this — if
    /// they do, the grow loop has hit its hard cap (currently 8
    /// attempts).
    case bufferTooSmall

    /// Maps `ZTOK_ERR_INTERNAL` (status 99) or any unrecognized status
    /// code. Carries the raw status and the C call site for diagnostic
    /// logging.
    case internalError(status: Int32, op: String)

    /// A path argument contained a NUL byte and couldn't be passed to
    /// the C ABI as a NUL-terminated string.
    case invalidPath(String)

    /// A C constructor returned a NULL pointer without setting a status
    /// code. Distinct from `.internalError` so callers can route around
    /// it (it usually means libztok mis-built).
    case nullHandle(op: String)

    /// `Pipeline.open` (auto-detect) couldn't determine the file's
    /// vocab format. Use a specific `from*` constructor instead.
    case unknownFormat(path: String)

    /// libztok returned bytes that aren't valid UTF-8 and the caller
    /// asked for a `String`. Use the `*Bytes` decoder variant to keep
    /// raw bytes.
    case invalidUtf8

    /// Operation attempted on a closed Pipeline / BatchPool /
    /// StreamEncoder.
    case closed(op: String)

    public var description: String {
        switch self {
        case .outOfMemory:
            return "ztok: out of memory"
        case .invalidInput:
            return "ztok: invalid input"
        case .bufferTooSmall:
            return "ztok: buffer too small"
        case .internalError(let status, let op):
            return "ztok: internal error in \(op) (status \(status))"
        case .invalidPath(let p):
            return "ztok: invalid path: \(p)"
        case .nullHandle(let op):
            return "ztok: \(op) returned NULL"
        case .unknownFormat(let path):
            return "ztok: could not auto-detect tokenizer format for \(path); use a specific from* constructor"
        case .invalidUtf8:
            return "ztok: returned bytes were not valid UTF-8"
        case .closed(let op):
            return "ztok: \(op) called on a closed handle"
        }
    }
}

/// Translate a non-zero `ztok_status` to a `ZtokError`. Returns
/// without throwing when status is `ZTOK_OK`.
@inlinable
internal func checkStatus(_ status: Int32, op: String) throws {
    // ztok_status is a plain C enum. Compare against the raw integer
    // values exported by the CZtok header.
    switch status {
    case Int32(ZTOK_OK.rawValue):
        return
    case Int32(ZTOK_ERR_OUT_OF_MEMORY.rawValue):
        throw ZtokError.outOfMemory
    case Int32(ZTOK_ERR_INVALID_INPUT.rawValue):
        throw ZtokError.invalidInput
    case Int32(ZTOK_ERR_BUFFER_TOO_SMALL.rawValue):
        throw ZtokError.bufferTooSmall
    case Int32(ZTOK_ERR_INTERNAL.rawValue):
        throw ZtokError.internalError(status: status, op: op)
    default:
        throw ZtokError.internalError(status: status, op: op)
    }
}
