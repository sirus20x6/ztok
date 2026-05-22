//! Typed error enum mapping `ztok_status` to idiomatic Rust.
//!
//! Every fallible API in the crate returns `Result<T, Error>`. The
//! variants mirror the C enum 1:1 so `match` exhaustiveness catches
//! drift the moment libztok adds a new code (we map unknown codes to
//! `Internal`).

use core::ffi::c_int;
use std::fmt;

use crate::sys;

/// All error conditions surfaced by the safe wrapper.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// Maps `ZTOK_ERR_OUT_OF_MEMORY` (status 1).
    OutOfMemory,
    /// Maps `ZTOK_ERR_INVALID_INPUT` (status 2). Covers bad arguments,
    /// unknown enum kinds, and (for `Pipeline::open`) format-detection
    /// failures.
    InvalidInput,
    /// Maps `ZTOK_ERR_BUFFER_TOO_SMALL` (status 3). The safe wrapper
    /// grows buffers automatically, so callers should rarely see this —
    /// if they do, the grow loop has hit its hard cap (currently 8
    /// attempts).
    BufferTooSmall,
    /// Maps `ZTOK_ERR_INTERNAL` (status 99) or any unrecognized status
    /// code. The `op` and `status` fields carry the call site and raw
    /// integer for diagnostic logging.
    Internal {
        /// Raw `ztok_status` integer the C ABI returned.
        status: c_int,
        /// Name of the C call site (e.g. `"ztok_encode"`).
        op: &'static str,
    },
    /// A path argument contained a NUL byte and couldn't be passed to
    /// the C ABI as a NUL-terminated string.
    InvalidPath,
    /// A C constructor returned a NULL pointer without setting a status
    /// code. This shouldn't happen on a healthy libztok build but is
    /// distinct from `Internal` so callers can route around it.
    NullHandle {
        /// Name of the C constructor that returned NULL.
        op: &'static str,
    },
    /// `Pipeline::open` (auto-detect) couldn't determine the file's
    /// vocab format. Use a specific `from_*` constructor instead.
    UnknownFormat,
    /// A UTF-8 string returned by libztok couldn't be decoded.
    InvalidUtf8,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::OutOfMemory => write!(f, "ztok: out of memory"),
            Error::InvalidInput => write!(f, "ztok: invalid input"),
            Error::BufferTooSmall => write!(f, "ztok: buffer too small"),
            Error::Internal { status, op } => {
                write!(f, "ztok: internal error in {} (status {})", op, status)
            }
            Error::InvalidPath => write!(f, "ztok: path contains an interior NUL byte"),
            Error::NullHandle { op } => write!(f, "ztok: {} returned NULL", op),
            Error::UnknownFormat => write!(
                f,
                "ztok: could not auto-detect tokenizer format; use a specific from_* constructor"
            ),
            Error::InvalidUtf8 => write!(f, "ztok: returned bytes were not valid UTF-8"),
        }
    }
}

impl std::error::Error for Error {}

/// Translate a non-zero `ztok_status` to an `Error`. Returns `Ok(())`
/// when status is `ZTOK_OK`.
pub(crate) fn check_status(status: c_int, op: &'static str) -> core::result::Result<(), Error> {
    match status {
        sys::ZTOK_OK => Ok(()),
        sys::ZTOK_ERR_OUT_OF_MEMORY => Err(Error::OutOfMemory),
        sys::ZTOK_ERR_INVALID_INPUT => Err(Error::InvalidInput),
        sys::ZTOK_ERR_BUFFER_TOO_SMALL => Err(Error::BufferTooSmall),
        sys::ZTOK_ERR_INTERNAL => Err(Error::Internal { status, op }),
        _ => Err(Error::Internal { status, op }),
    }
}

/// Convenience alias for the binding's `Result` shape.
pub type Result<T> = core::result::Result<T, Error>;
