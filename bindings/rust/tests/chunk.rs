//! Token-window chunking tests for the ztok Rust binding.
//!
//! Mirrors bindings/go/chunk_test.go: chunking over a byte_id pipeline
//! (each input byte = one token) so the windows are predictable —
//! "abcdefghij" is 10 tokens, one per ASCII byte.

use ztok::{version, ChunkBoundary, Pipeline};

fn require_libztok() -> bool {
    version().is_ok()
}

#[test]
fn chunk_non_overlapping() {
    if !require_libztok() {
        return;
    }
    let pipe = Pipeline::byte_id(None).expect("byte_id pipeline");
    let chunks = pipe
        .chunk("abcdefghij", 4, 0, ChunkBoundary::Token)
        .expect("chunk");
    // 10 tokens / window 4, stride 4 -> [0,4) [4,8) [8,10).
    assert_eq!(chunks.len(), 3, "got {} chunks, want 3", chunks.len());

    let want: [(u32, u32, u32, u32, usize); 3] = [
        (0, 4, 0, 4, 4),
        (4, 8, 4, 8, 4),
        (8, 10, 8, 10, 2),
    ];
    for (i, &(ts, te, bs, be, n)) in want.iter().enumerate() {
        let c = &chunks[i];
        assert_eq!(
            (c.token_start, c.token_end),
            (ts, te),
            "chunk {i} token range"
        );
        assert_eq!((c.byte_start, c.byte_end), (bs, be), "chunk {i} byte range");
        assert_eq!(c.ids.len(), n, "chunk {i} id count");
    }
}

#[test]
fn chunk_overlap() {
    if !require_libztok() {
        return;
    }
    let pipe = Pipeline::byte_id(None).unwrap();
    let chunks = pipe
        .chunk("abcdefghij", 4, 2, ChunkBoundary::Token)
        .expect("chunk");
    assert!(chunks.len() >= 2, "got {} chunks, want >= 2", chunks.len());
    // stride = 2, so chunk[i+1] starts 2 tokens after chunk[i]; the last
    // 2 ids of chunk[i] equal the first 2 ids of chunk[i+1].
    for w in chunks.windows(2) {
        let (a, b) = (&w[0], &w[1]);
        if a.ids.len() < 2 || b.ids.len() < 2 {
            continue;
        }
        assert_eq!(a.ids[a.ids.len() - 2], b.ids[0], "overlap[0] mismatch");
        assert_eq!(a.ids[a.ids.len() - 1], b.ids[1], "overlap[1] mismatch");
    }
}

#[test]
fn chunk_empty_input_no_chunks() {
    if !require_libztok() {
        return;
    }
    let pipe = Pipeline::byte_id(None).unwrap();
    let chunks = pipe.chunk("", 4, 0, ChunkBoundary::Token).expect("chunk");
    assert!(chunks.is_empty(), "empty input must give no chunks");
}

#[test]
fn chunk_bad_args_error() {
    if !require_libztok() {
        return;
    }
    let pipe = Pipeline::byte_id(None).unwrap();
    // max_tokens == 0 is invalid.
    assert_eq!(
        pipe.chunk("abc", 0, 0, ChunkBoundary::Token),
        Err(ztok::Error::InvalidInput)
    );
    // overlap >= max_tokens is invalid.
    assert_eq!(
        pipe.chunk("abc", 4, 4, ChunkBoundary::Token),
        Err(ztok::Error::InvalidInput)
    );
}

#[test]
fn chunk_default_boundary_is_token() {
    // The Default impl on ChunkBoundary mirrors the C zero value.
    assert_eq!(ChunkBoundary::default(), ChunkBoundary::Token);
}
