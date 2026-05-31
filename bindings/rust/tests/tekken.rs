//! Mistral Tekken tokenizer tests for the ztok Rust binding.
//!
//! Loads the real `mistral_nemo_tekken.json` fixture (skipped when
//! absent) and checks ztok reproduces the canonical reference encodings,
//! mirroring bindings/python/tests and the C constructor
//! `ztok_pipeline_new_tekken_from_file`. Also covers auto-detect dispatch
//! through `Pipeline::open`.

use std::path::PathBuf;

use ztok::{version, Format, Pipeline};

fn require_libztok() -> bool {
    version().is_ok()
}

/// bindings/rust/ -> repo root -> bench/vocabs/mistral_nemo_tekken.json.
fn vocab_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("bench")
        .join("vocabs")
        .join("mistral_nemo_tekken.json")
}

/// Golden id sequences verified against mistral_common 1.8.6 on
/// bench/vocabs/mistral_nemo_tekken.json.
const GOLDEN: &[(&str, &[u32])] = &[
    ("Hello, world!", &[22177, 1044, 4304, 1033]),
    ("The quick brown fox", &[1784, 7586, 22980, 94137]),
    (" and the", &[1321, 1278]),
];

#[test]
fn tekken_matches_reference() {
    if !require_libztok() {
        return;
    }
    let path = vocab_path();
    if !path.exists() {
        eprintln!("Tekken vocab fixture not present at {path:?}; skipping");
        return;
    }
    let pipe = Pipeline::from_tekken(&path, None).expect("from_tekken");
    for &(text, want) in GOLDEN {
        let ids = pipe.encode(text).unwrap();
        assert_eq!(ids, want, "encode mismatch for {text:?}");
    }
}

#[test]
fn tekken_auto_detect_and_open() {
    if !require_libztok() {
        return;
    }
    let path = vocab_path();
    if !path.exists() {
        eprintln!("Tekken vocab fixture not present at {path:?}; skipping");
        return;
    }
    assert_eq!(ztok::detect_format(&path).unwrap(), Format::Tekken);
    // Open should dispatch to from_tekken via auto-detect.
    let pipe = Pipeline::open(&path).expect("open auto-detect");
    let ids = pipe.encode("Hello, world!").unwrap();
    assert_eq!(ids, &[22177, 1044, 4304, 1033]);
}
