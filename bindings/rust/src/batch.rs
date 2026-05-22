//! Safe `BatchPool` wrapper around `ztok_batch_pool*`.
//!
//! A persistent worker pool reused across many batched encodes. Each
//! pool owns its own arenas and worker threads, so creating one per
//! batch wastes setup work.
//!
//! Output handling: libztok returns each per-input id buffer with an
//! opaque length-prefix header (see `src/c_api.zig::allocIdBuf`). The
//! ONLY safe free is `ztok_ids_free`. We always materialize the buffer
//! into a Rust `Vec<u32>` and free it through `ztok_ids_free` before
//! returning — no foreign pointers leak into safe Rust.

use core::ffi::{c_char, c_int};

use crate::error::{check_status, Error, Result};
use crate::pipeline::Pipeline;
use crate::sys;

/// A persistent multi-threaded encode worker pool.
pub struct BatchPool {
    handle: *mut sys::ZtokBatchPool,
}

// libztok's batch pool is internally synchronized — multiple threads
// can drive concurrent batches against the same pool. Mirrors the Go
// binding's lack of guarding around BatchPool calls.
unsafe impl Send for BatchPool {}
unsafe impl Sync for BatchPool {}

impl BatchPool {
    /// Build a persistent worker pool. `workers = 0` auto-detects the
    /// CPU count.
    pub fn new(workers: u32) -> Result<Self> {
        let mut status: c_int = sys::ZTOK_OK;
        let h = unsafe { sys::ztok_batch_pool_new(workers, &mut status) };
        check_status(status, "ztok_batch_pool_new")?;
        if h.is_null() {
            return Err(Error::NullHandle {
                op: "ztok_batch_pool_new",
            });
        }
        Ok(Self { handle: h })
    }

    /// Actual worker count (resolves `workers = 0` to the detected CPU
    /// count).
    pub fn workers(&self) -> usize {
        unsafe { sys::ztok_batch_pool_worker_count(self.handle) }
    }

    /// Encode many inputs in parallel. Returns one `Vec<u32>` per input,
    /// in the same order. Empty inputs produce empty vectors.
    pub fn encode_batch(&self, pipeline: &Pipeline, inputs: &[&str]) -> Result<Vec<Vec<u32>>> {
        let byte_inputs: Vec<&[u8]> = inputs.iter().map(|s| s.as_bytes()).collect();
        self.encode_batch_bytes(pipeline, &byte_inputs)
    }

    /// Encode many byte slices in parallel. Use this for non-UTF-8
    /// inputs.
    pub fn encode_batch_bytes(
        &self,
        pipeline: &Pipeline,
        inputs: &[&[u8]],
    ) -> Result<Vec<Vec<u32>>> {
        let n = inputs.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        // The C ABI expects NUL-terminated `const char*` arrays. We
        // materialize each input into a CString so the bytes survive
        // the call. Note: NUL bytes in input are *not* permitted by
        // CString — but ztok's encoder treats `input_len` as the
        // authoritative length, so we could in principle skip the
        // CString and pass a plain buffer. We use CString anyway for
        // belt-and-braces: the ABI documents `const char*` and we
        // shouldn't smuggle interior NULs into a path libztok believes
        // is C-string-shaped.
        //
        // Actually — the ABI takes (ptr, len), and inputs may legitimately
        // contain NULs (think binary tokenization). Use raw byte
        // pointers with explicit length instead.
        let storage: Vec<Vec<u8>> = inputs.iter().map(|b| b.to_vec()).collect();
        let c_inputs: Vec<*const c_char> = storage
            .iter()
            .map(|v| v.as_ptr() as *const c_char)
            .collect();
        let lens: Vec<usize> = storage.iter().map(|v| v.len()).collect();
        let mut out_ids: Vec<*mut u32> = vec![core::ptr::null_mut(); n];
        let mut out_lens: Vec<usize> = vec![0; n];

        let rc = unsafe {
            sys::ztok_encode_batch_pooled(
                pipeline.raw(),
                self.handle,
                c_inputs.as_ptr(),
                lens.as_ptr(),
                n,
                out_ids.as_mut_ptr(),
                out_lens.as_mut_ptr(),
            )
        };

        // Always materialize + free, even on error, so we never leak C
        // buffers from a partially-populated batch.
        let mut results: Vec<Vec<u32>> = Vec::with_capacity(n);
        for i in 0..n {
            let len = out_lens[i];
            let ptr = out_ids[i];
            results.push(materialize_and_free(ptr, len));
        }

        check_status(rc, "ztok_encode_batch_pooled")?;
        // Storage and lens must outlive the C call.
        drop(storage);
        drop(c_inputs);
        drop(lens);
        Ok(results)
    }
}

impl Drop for BatchPool {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { sys::ztok_batch_pool_free(self.handle) };
            self.handle = core::ptr::null_mut();
        }
    }
}

/// Copy a libztok-owned id buffer into a `Vec<u32>`, then free it via
/// `ztok_ids_free`. The C buffer carries an opaque length-prefix header
/// (see `src/c_api.zig::allocIdBuf`) so this is the only safe free
/// path — never call `libc::free`.
pub(crate) fn materialize_and_free(ptr: *mut u32, n: usize) -> Vec<u32> {
    if ptr.is_null() {
        return Vec::new();
    }
    if n == 0 {
        // Even for an empty buffer the C side hands us a (possibly
        // non-null) header to free.
        unsafe { sys::ztok_ids_free(ptr) };
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n);
    unsafe {
        // Copy into Rust-owned memory before freeing the C buffer.
        let slice = core::slice::from_raw_parts(ptr, n);
        out.extend_from_slice(slice);
        sys::ztok_ids_free(ptr);
    }
    out
}
