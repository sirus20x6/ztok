//! Safe `StreamEncoder` wrapper around `ztok_stream*`.
//!
//! Two surfaces:
//!
//! 1. `StreamEncoder::feed` / `finish` — sink-style API matching the
//!    Node `koffi` and Python ctypes shape. Returns each batch of
//!    newly-emitted ids as a `Vec<u32>`.
//!
//! 2. `Pipeline::encode_stream` + `StreamIter` — an `Iterator` adapter
//!    that yields one `Vec<u32>` per non-empty emit, including the
//!    final flush. Idiomatic Rust shape, useful in `for` loops.
//!
//! The encoder defers a trailing partial UTF-8 codepoint / pre-tokenizer
//! span up to a soft 1 MiB cap (see `src/stream.zig`); past that it
//! force-cuts at the nearest codepoint boundary.

use core::ffi::c_int;

use crate::batch::materialize_and_free;
use crate::error::{check_status, Error, Result};
use crate::pipeline::Pipeline;
use crate::sys;

/// Default feed chunk size (64 KiB) — matches the Python/Go bindings.
pub const DEFAULT_CHUNK_SIZE: usize = 64 * 1024;

/// A streaming-encode session against a [`Pipeline`].
///
/// Construct via [`StreamEncoder::new`] or [`Pipeline::encode_stream`].
/// Feed bytes via [`StreamEncoder::feed`] and call [`StreamEncoder::finish`]
/// to drain the final carry.
pub struct StreamEncoder<'a> {
    handle: *mut sys::ZtokStream,
    // Keep the pipeline alive for the encoder's lifetime. The stream
    // borrows from the pipeline (the C ABI documents that the stream
    // holds a pointer to the pipeline internally, see src/stream.zig),
    // so dropping the pipeline first would dangle.
    _pipeline: &'a Pipeline,
    finished: bool,
}

impl<'a> StreamEncoder<'a> {
    /// Open a new streaming session against `pipeline`.
    pub fn new(pipeline: &'a Pipeline) -> Result<Self> {
        let mut status: c_int = sys::ZTOK_OK;
        let h = unsafe { sys::ztok_stream_new(pipeline.raw(), &mut status) };
        check_status(status, "ztok_stream_new")?;
        if h.is_null() {
            return Err(Error::NullHandle {
                op: "ztok_stream_new",
            });
        }
        Ok(Self {
            handle: h,
            _pipeline: pipeline,
            finished: false,
        })
    }

    /// Feed `bytes` to the encoder. Returns any token ids newly emitted
    /// by this call (may be empty when the encoder is still buffering
    /// toward the next safe cut).
    pub fn feed(&mut self, bytes: &[u8]) -> Result<Vec<u32>> {
        if self.finished {
            return Err(Error::Internal {
                status: sys::ZTOK_ERR_INVALID_INPUT,
                op: "ztok_stream_feed: stream already finished",
            });
        }
        let mut out_ids: *mut u32 = core::ptr::null_mut();
        let mut out_n: usize = 0;
        let rc = unsafe {
            let (ptr, len) = if bytes.is_empty() {
                (core::ptr::null(), 0)
            } else {
                (bytes.as_ptr() as *const _, bytes.len())
            };
            sys::ztok_stream_feed(self.handle, ptr, len, &mut out_ids, &mut out_n)
        };
        if let Err(e) = check_status(rc, "ztok_stream_feed") {
            // Even on error, the C ABI may have allocated a partial
            // buffer — free it before propagating.
            if !out_ids.is_null() {
                unsafe { sys::ztok_ids_free(out_ids) };
            }
            return Err(e);
        }
        Ok(materialize_and_free(out_ids, out_n))
    }

    /// Flush any remaining carry as a final encode. Idempotent: a
    /// second call returns an empty `Vec`.
    pub fn finish(&mut self) -> Result<Vec<u32>> {
        if self.finished {
            return Ok(Vec::new());
        }
        self.finished = true;
        let mut out_ids: *mut u32 = core::ptr::null_mut();
        let mut out_n: usize = 0;
        let rc =
            unsafe { sys::ztok_stream_finish(self.handle, &mut out_ids, &mut out_n) };
        if let Err(e) = check_status(rc, "ztok_stream_finish") {
            if !out_ids.is_null() {
                unsafe { sys::ztok_ids_free(out_ids) };
            }
            return Err(e);
        }
        Ok(materialize_and_free(out_ids, out_n))
    }
}

impl<'a> Drop for StreamEncoder<'a> {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { sys::ztok_stream_free(self.handle) };
            self.handle = core::ptr::null_mut();
        }
    }
}

/// `Iterator` adapter over a [`StreamEncoder`] driven by a fixed input
/// buffer. Yields one non-empty `Vec<u32>` per emit; the final flush is
/// emitted last.
///
/// Returned by [`Pipeline::encode_stream`].
pub struct StreamIter<'a> {
    encoder: StreamEncoder<'a>,
    data: &'a [u8],
    cursor: usize,
    chunk_size: usize,
    drained: bool,
}

impl<'a> StreamIter<'a> {
    pub(crate) fn new(pipeline: &'a Pipeline, data: &'a [u8], chunk_size: usize) -> Result<Self> {
        Ok(Self {
            encoder: StreamEncoder::new(pipeline)?,
            data,
            cursor: 0,
            chunk_size: chunk_size.max(1),
            drained: false,
        })
    }
}

impl<'a> Iterator for StreamIter<'a> {
    type Item = Result<Vec<u32>>;

    fn next(&mut self) -> Option<Self::Item> {
        // Loop until we either produce a non-empty batch, hit the final
        // flush, or exhaust everything — the C ABI happily returns
        // empty emit lists between cuts, but exposing those would be
        // noise.
        loop {
            if self.cursor < self.data.len() {
                let end = (self.cursor + self.chunk_size).min(self.data.len());
                let chunk = &self.data[self.cursor..end];
                self.cursor = end;
                match self.encoder.feed(chunk) {
                    Ok(ids) if ids.is_empty() => continue,
                    Ok(ids) => return Some(Ok(ids)),
                    Err(e) => return Some(Err(e)),
                }
            }
            if !self.drained {
                self.drained = true;
                match self.encoder.finish() {
                    Ok(ids) if ids.is_empty() => return None,
                    Ok(ids) => return Some(Ok(ids)),
                    Err(e) => return Some(Err(e)),
                }
            }
            return None;
        }
    }
}

impl Pipeline {
    /// Stream-encode `data` and produce one `Vec<u32>` per non-empty
    /// emit. The final flush is yielded last. `chunk_size` controls the
    /// per-feed byte count; pass `0` (or use [`encode_stream_default`])
    /// to get the standard 64 KiB.
    ///
    /// [`encode_stream_default`]: Pipeline::encode_stream_default
    pub fn encode_stream<'a>(
        &'a self,
        data: &'a [u8],
        chunk_size: usize,
    ) -> Result<StreamIter<'a>> {
        StreamIter::new(self, data, chunk_size)
    }

    /// Convenience wrapper that uses [`DEFAULT_CHUNK_SIZE`].
    pub fn encode_stream_default<'a>(&'a self, data: &'a [u8]) -> Result<StreamIter<'a>> {
        self.encode_stream(data, DEFAULT_CHUNK_SIZE)
    }
}
