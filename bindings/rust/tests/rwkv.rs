//! RWKV "World" tokenizer tests for the ztok Rust binding.
//!
//! Loads the real `rwkv_vocab_v20230424.txt` fixture (skipped when
//! absent) and checks ztok reproduces the canonical reference encodings,
//! mirroring bindings/python/tests/test_rwkv.py and the in-tree gate in
//! src/rwkv_world.zig. Also covers auto-detect dispatch through
//! `Pipeline::open`.

use std::path::PathBuf;

use ztok::{version, Format, Pipeline};

fn require_libztok() -> bool {
    version().is_ok()
}

/// bindings/rust/ -> repo root -> bench/vocabs/rwkv_vocab_v20230424.txt.
fn vocab_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("bench")
        .join("vocabs")
        .join("rwkv_vocab_v20230424.txt")
}

/// Golden id sequences captured from BlinkDL's canonical reference
/// tokenizer (see bench/rwkv_parity.py / src/rwkv_world.zig).
const GOLDEN: &[(&str, &[u32])] = &[
    ("Hello, world!", &[33155, 45, 40213, 34]),
    (
        "emoji 😀🚀✨ test",
        &[34295, 33, 3319, 153, 129, 3319, 155, 129, 10059, 32223],
    ),
    ("0 1 2 10 99 100", &[49, 284, 285, 3483, 3572, 3483, 49]),
];

#[test]
fn rwkv_matches_reference() {
    if !require_libztok() {
        return;
    }
    let path = vocab_path();
    if !path.exists() {
        eprintln!("RWKV vocab fixture not present at {path:?}; skipping");
        return;
    }
    let pipe = Pipeline::from_rwkv(&path, None).expect("from_rwkv");
    for &(text, want) in GOLDEN {
        let ids = pipe.encode(text).unwrap();
        assert_eq!(ids, want, "encode mismatch for {text:?}");
    }
}

#[test]
fn rwkv_round_trips() {
    if !require_libztok() {
        return;
    }
    let path = vocab_path();
    if !path.exists() {
        eprintln!("RWKV vocab fixture not present at {path:?}; skipping");
        return;
    }
    let pipe = Pipeline::from_rwkv(&path, None).expect("from_rwkv");
    for &(text, _) in GOLDEN {
        let ids = pipe.encode(text).unwrap();
        assert_eq!(pipe.decode(&ids).unwrap(), text, "round-trip {text:?}");
    }
}

#[test]
fn rwkv_auto_detect_and_open() {
    if !require_libztok() {
        return;
    }
    let path = vocab_path();
    if !path.exists() {
        eprintln!("RWKV vocab fixture not present at {path:?}; skipping");
        return;
    }
    assert_eq!(ztok::detect_format(&path).unwrap(), Format::Rwkv);
    // Open should dispatch to from_rwkv via auto-detect.
    let pipe = Pipeline::open(&path).expect("open auto-detect");
    let ids = pipe.encode("Hello, world!").unwrap();
    assert_eq!(ids, &[33155, 45, 40213, 34]);
}
