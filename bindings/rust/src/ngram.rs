//! Engram n-gram hashing — deterministic multi-head token-n-gram hashes
//! for conditional-memory addressing (see `src/ngram.zig`).
//!
//! These operate on raw token ids and need no [`Pipeline`]. The output
//! is row-major `[position][head]` raw `u64` hashes which the caller
//! masks to its own table width (`hash & ((1 << bits) - 1)`).
//!
//! Memory contract: the batch path returns each per-doc hash buffer with
//! an opaque length-prefix header (see `src/c_api.zig::allocU64Buf`). The
//! ONLY safe free is `ztok_u64s_free`. We always copy into a Rust
//! `Vec<u64>` and free through `ztok_u64s_free` before returning — no
//! foreign pointer leaks into safe Rust.
//!
//! [`Pipeline`]: crate::Pipeline

use crate::batch::BatchPool;
use crate::error::{check_status, Error, Result};
use crate::sys;

/// Hash every length-`n` window of `ids` under `heads` independent hash
/// functions, returning the row-major `[position][head]` `u64` hashes
/// (positions = `ids.len() - n + 1`, or `0` if the stream is shorter
/// than one window). Mask each hash to your table width
/// (`hash & ((1 << bits) - 1)`).
///
/// Deterministic: identical ids always yield identical hashes. Operates
/// on raw ids — no [`Pipeline`](crate::Pipeline) needed. Returns an empty
/// `Vec` when there is nothing to hash (`ids` shorter than `n`, or
/// `n == 0` / `heads == 0`).
pub fn hash_ngrams(ids: &[u32], n: u32, heads: u32) -> Result<Vec<u64>> {
    if ids.is_empty() || n == 0 || heads == 0 || ids.len() < n as usize {
        return Ok(Vec::new());
    }
    let positions = ids.len() - n as usize + 1;
    let want = positions * heads as usize;
    if want == 0 {
        return Ok(Vec::new());
    }

    // We size the buffer exactly, so BUFFER_TOO_SMALL should never fire;
    // honor the contract anyway by retrying once at the reported size.
    let mut cap = want;
    for _ in 0..2 {
        let mut out: Vec<u64> = vec![0; cap];
        let mut out_len: usize = 0;
        let rc = unsafe {
            sys::ztok_ngram_hash(
                ids.as_ptr(),
                ids.len(),
                n,
                heads,
                out.as_mut_ptr(),
                cap,
                &mut out_len,
            )
        };
        match rc {
            sys::ZTOK_OK => {
                out.truncate(out_len);
                return Ok(out);
            }
            sys::ZTOK_ERR_BUFFER_TOO_SMALL => {
                if out_len == 0 {
                    return Ok(Vec::new());
                }
                cap = out_len;
                continue;
            }
            _ => return Err(check_status(rc, "ztok_ngram_hash").unwrap_err()),
        }
    }
    Err(Error::Internal {
        status: sys::ZTOK_ERR_BUFFER_TOO_SMALL,
        op: "ztok_ngram_hash: BUFFER_TOO_SMALL reported twice",
    })
}

/// Hash many id streams in parallel across `pool`. `results[i]` holds the
/// row-major hashes for `streams[i]` (empty for a stream shorter than one
/// window). Equivalent to calling [`hash_ngrams`] on each stream, fanned
/// out across the pool's workers.
///
/// The C-owned hash buffers are materialized into Rust `Vec`s and freed
/// before returning, so the result is fully owned by Rust.
pub fn hash_ngrams_batch(
    pool: &BatchPool,
    streams: &[&[u32]],
    n: u32,
    heads: u32,
) -> Result<Vec<Vec<u64>>> {
    let n_docs = streams.len();
    if n_docs == 0 {
        return Ok(Vec::new());
    }

    // The C ABI takes `const ztok_token_id* const*` — an array of
    // pointers into per-doc id buffers. We borrow directly from the
    // caller's slices (they outlive the call) and pass NULL for empty
    // streams, matching the other bindings.
    let id_arrays: Vec<*const u32> = streams
        .iter()
        .map(|s| {
            if s.is_empty() {
                core::ptr::null()
            } else {
                s.as_ptr()
            }
        })
        .collect();
    let id_lens: Vec<usize> = streams.iter().map(|s| s.len()).collect();
    let mut out_hashes: Vec<*mut u64> = vec![core::ptr::null_mut(); n_docs];
    let mut out_lens: Vec<usize> = vec![0; n_docs];

    let rc = unsafe {
        sys::ztok_ngram_hash_batch(
            pool.raw(),
            id_arrays.as_ptr(),
            id_lens.as_ptr(),
            n_docs,
            n,
            heads,
            out_hashes.as_mut_ptr(),
            out_lens.as_mut_ptr(),
        )
    };

    // Always materialize + free, even on error, so a partially-populated
    // batch never leaks C buffers.
    let mut results: Vec<Vec<u64>> = Vec::with_capacity(n_docs);
    for i in 0..n_docs {
        results.push(materialize_u64s_and_free(out_hashes[i], out_lens[i]));
    }

    check_status(rc, "ztok_ngram_hash_batch")?;
    Ok(results)
}

/// Copy a libztok-owned `u64` hash buffer into a `Vec<u64>`, then free it
/// via `ztok_u64s_free`. The C buffer carries an opaque length-prefix
/// header (see `src/c_api.zig::allocU64Buf`) so this is the only safe
/// free path — never call `libc::free`.
fn materialize_u64s_and_free(ptr: *mut u64, n: usize) -> Vec<u64> {
    if ptr.is_null() {
        return Vec::new();
    }
    if n == 0 {
        // Even an empty buffer hands us a (possibly non-null) header to
        // free.
        unsafe { sys::ztok_u64s_free(ptr) };
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n);
    unsafe {
        // Copy into Rust-owned memory before freeing the C buffer.
        let slice = core::slice::from_raw_parts(ptr, n);
        out.extend_from_slice(slice);
        sys::ztok_u64s_free(ptr);
    }
    out
}
