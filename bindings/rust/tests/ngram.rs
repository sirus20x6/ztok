//! Engram n-gram hashing tests for the ztok Rust binding.
//!
//! Mirrors bindings/go/ngram_test.go and
//! bindings/python/tests/test_ngram.py: determinism, output length,
//! per-head independence, and batch-matches-single. n-gram hashing
//! operates on raw ids and needs no on-disk fixture, so there is nothing
//! to skip here beyond libztok itself being linkable.

use ztok::{hash_ngrams, hash_ngrams_batch, version, BatchPool};

/// Probe libztok by calling `version()`. The test binary won't load at
/// all if the shared object is missing, so an Err here means the symbol
/// is present but misbehaving — same idiom as smoke.rs.
fn require_libztok() -> bool {
    version().is_ok()
}

#[test]
fn hash_ngrams_deterministic() {
    if !require_libztok() {
        return;
    }
    let ids = [7u32, 42, 1000, 3, 99, 7, 42];
    let a = hash_ngrams(&ids, 3, 2).unwrap();
    // positions = 7 - 3 + 1 = 5, heads = 2 -> 10 hashes.
    assert_eq!(a.len(), 10);
    let b = hash_ngrams(&ids, 3, 2).unwrap();
    assert_eq!(a, b, "hashes must be deterministic across calls");
}

#[test]
fn hash_ngrams_short_input_is_empty() {
    if !require_libztok() {
        return;
    }
    let out = hash_ngrams(&[1, 2], 3, 4).unwrap();
    assert!(
        out.is_empty(),
        "stream shorter than window must yield no hashes"
    );
}

#[test]
fn hash_ngrams_zero_n_or_heads_is_empty() {
    if !require_libztok() {
        return;
    }
    assert!(hash_ngrams(&[1, 2, 3], 0, 2).unwrap().is_empty());
    assert!(hash_ngrams(&[1, 2, 3], 2, 0).unwrap().is_empty());
}

#[test]
fn hash_ngrams_heads_independent() {
    if !require_libztok() {
        return;
    }
    let out = hash_ngrams(&[5, 6, 7, 8], 3, 3).unwrap();
    // 2 positions x 3 heads = 6.
    assert_eq!(out.len(), 6);
    // The 3 heads for position 0 must differ from each other.
    assert!(
        out[0] != out[1] && out[1] != out[2] && out[0] != out[2],
        "heads not independent: {:?}",
        &out[..3]
    );
}

#[test]
fn hash_ngrams_batch_matches_single() {
    if !require_libztok() {
        return;
    }
    let pool = BatchPool::new(2).unwrap();
    let s0: &[u32] = &[1, 2, 3, 4, 5];
    let s1: &[u32] = &[9, 8, 7];
    let s2: &[u32] = &[1, 1]; // shorter than n -> empty
    let streams: [&[u32]; 3] = [s0, s1, s2];

    let got = hash_ngrams_batch(&pool, &streams, 3, 2).unwrap();
    assert_eq!(got.len(), streams.len());
    for (i, s) in streams.iter().enumerate() {
        let want = hash_ngrams(s, 3, 2).unwrap();
        assert_eq!(got[i], want, "stream {i}: batch != single");
    }
    // The short stream comes back empty.
    assert!(got[2].is_empty());
}

#[test]
fn hash_ngrams_batch_empty_streams() {
    if !require_libztok() {
        return;
    }
    let pool = BatchPool::new(2).unwrap();
    let got = hash_ngrams_batch(&pool, &[], 3, 2).unwrap();
    assert!(got.is_empty());
}
