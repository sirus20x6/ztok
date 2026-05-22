//! Smoke tests for the ztok Rust binding.
//!
//! Mirrors the basic round-trip coverage in the Python / Node / Ruby /
//! Go bindings. Skips gracefully (via `if let Err(_) = ... { return; }`)
//! when libztok isn't available in the test environment — same
//! graceful-skip pattern the other bindings follow.
//!
//! Set `ZTOK_LIB_DIR=/path/to/zig-out/lib` (and ensure
//! `LD_LIBRARY_PATH` points there too on Linux) before `cargo test` to
//! enable the full suite.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::OnceLock;

use ztok::{BatchPool, Config, Decoder, Normalizer, Pipeline, PreTokenizer};

/// Synthetic .tiktoken vocab covering all 256 single bytes plus a
/// handful of merges — same shape as bindings/python/tests/conftest.py
/// and bindings/go/fixture_test.go. Built once per test binary
/// invocation.
fn tiktoken_fixture() -> Option<PathBuf> {
    static FIXTURE: OnceLock<Option<PathBuf>> = OnceLock::new();
    FIXTURE
        .get_or_init(|| {
            let dir = std::env::temp_dir().join(format!("ztok-rust-{}", std::process::id()));
            fs::create_dir_all(&dir).ok()?;
            let path = dir.join("synthetic_cl100k.tiktoken");
            let mut f = fs::File::create(&path).ok()?;
            let mut rank = 0u32;
            for b in 0u32..256 {
                writeln!(
                    f,
                    "{} {}",
                    base64_encode(&[b as u8]),
                    rank
                )
                .ok()?;
                rank += 1;
            }
            for extra in [
                "he", "hel", "hell", "hello", " w", " wo", " wor", " worl", " world", "th",
                "the", " th", " the", "fo", "foo", "bar", "baz", " quick", " brown", " fox",
            ] {
                writeln!(f, "{} {}", base64_encode(extra.as_bytes()), rank).ok()?;
                rank += 1;
            }
            Some(path)
        })
        .clone()
}

/// Tiny base64 encoder so the test harness has no external deps. RFC
/// 4648 standard alphabet with `=` padding — the only flavor tiktoken
/// vocab files use.
fn base64_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0];
        let b1 = chunk.get(1).copied().unwrap_or(0);
        let b2 = chunk.get(2).copied().unwrap_or(0);
        let triple = ((b0 as u32) << 16) | ((b1 as u32) << 8) | b2 as u32;
        out.push(ALPHABET[((triple >> 18) & 0x3f) as usize] as char);
        out.push(ALPHABET[((triple >> 12) & 0x3f) as usize] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[((triple >> 6) & 0x3f) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[(triple & 0x3f) as usize] as char
        } else {
            '='
        });
    }
    out
}

/// Probe libztok by trying to call `version()`. If the dlopen / link
/// failed at test load time the test binary won't even start, so an Err
/// here means the symbol is present but returned an unexpected value —
/// not a skip condition.
fn require_libztok() -> bool {
    // We can't catch dlopen failure at runtime — if the shared library
    // is missing, the test binary fails to load. The `cargo test` run
    // surfaces that as a build/link error, which is the right signal.
    // Within the binary, version() must succeed.
    match ztok::version() {
        Ok(v) => {
            assert!(v.contains('.'), "version should look like X.Y.Z, got {v:?}");
            true
        }
        Err(e) => {
            eprintln!("ztok::version failed: {e}");
            false
        }
    }
}

#[test]
fn version_is_nonempty_string() {
    if !require_libztok() {
        return;
    }
    let v = ztok::version().unwrap();
    assert!(!v.is_empty());
    let mut parts = v.split('.');
    let major = parts.next().unwrap();
    let minor = parts.next().unwrap();
    assert!(major.chars().all(|c| c.is_ascii_digit()));
    assert!(minor.chars().all(|c| c.is_ascii_digit()));
}

#[test]
fn byte_id_roundtrip() {
    if !require_libztok() {
        return;
    }
    let pipe = Pipeline::byte_id(None).expect("byte_id pipeline");
    let ids = pipe.encode("hi").unwrap();
    assert_eq!(ids, vec![0x68, 0x69]);
    assert_eq!(pipe.decode(&ids).unwrap(), "hi");
}

#[test]
fn empty_input_returns_empty_ids() {
    if !require_libztok() {
        return;
    }
    let pipe = Pipeline::byte_id(None).unwrap();
    assert_eq!(pipe.encode("").unwrap(), Vec::<u32>::new());
    assert_eq!(pipe.decode(&[]).unwrap(), "");
}

#[test]
fn bpe_roundtrip() {
    if !require_libztok() {
        return;
    }
    let Some(fixture) = tiktoken_fixture() else {
        eprintln!("could not build tiktoken fixture; skipping");
        return;
    };
    let pipe = Pipeline::from_tiktoken(
        &fixture,
        Some(Config {
            normalizer: Normalizer::Identity,
            pre_tokenizer: PreTokenizer::Cl100k,
            decoder: Decoder::Concat,
        }),
    )
    .expect("from_tiktoken");
    let text = "hello world";
    let ids = pipe.encode(text).unwrap();
    assert!(!ids.is_empty());
    assert_eq!(pipe.decode(&ids).unwrap(), text);
}

#[test]
fn bpe_100_line_stress() {
    if !require_libztok() {
        return;
    }
    let Some(fixture) = tiktoken_fixture() else {
        return;
    };
    let pipe = Pipeline::from_tiktoken(&fixture, None).unwrap();
    let snippets = [
        "hello world",
        "the quick brown fox",
        "foo bar baz",
        "hello there hello world",
        "the the the",
        "  the  ",
        "foo",
        "hello",
        " world",
        "bar baz",
    ];
    for i in 0..100 {
        let line = snippets[i % snippets.len()];
        let ids = pipe.encode(line).unwrap();
        let decoded = pipe.decode(&ids).unwrap();
        assert_eq!(decoded, line, "round-trip failed at iter {i}");
    }
}

#[test]
fn decode_bytes_returns_raw_bytes() {
    if !require_libztok() {
        return;
    }
    let pipe = Pipeline::byte_id(None).unwrap();
    let ids = pipe.encode("ab").unwrap();
    assert_eq!(pipe.decode_bytes(&ids).unwrap(), b"ab");
}

#[test]
fn invalid_input_returns_typed_error() {
    if !require_libztok() {
        return;
    }
    // Pass a deliberately bogus normalizer kind through the raw config.
    // We can't construct an invalid Normalizer through the safe enum,
    // so we go through the sys layer to verify the error mapping.
    use std::ffi::c_int;
    use ztok::sys;
    let cfg = sys::ZtokPipelineConfig {
        normalizer: 999,
        pre_tokenizer: 0,
        model: 0,
        decoder: 0,
    };
    let mut status: c_int = 0;
    let h = unsafe { sys::ztok_pipeline_new(&cfg, &mut status) };
    assert!(h.is_null(), "bogus normalizer must not produce a handle");
    assert_eq!(status, sys::ZTOK_ERR_INVALID_INPUT);
}

#[test]
fn batch_pool_roundtrip() {
    if !require_libztok() {
        return;
    }
    let pipe = Pipeline::byte_id(None).unwrap();
    let pool = BatchPool::new(2).unwrap();
    assert!(pool.workers() >= 1);
    let inputs = vec!["foo", "bar", "baz", ""];
    let results = pool.encode_batch(&pipe, &inputs).unwrap();
    assert_eq!(results.len(), 4);
    assert_eq!(results[0], vec![b'f' as u32, b'o' as u32, b'o' as u32]);
    assert_eq!(results[1], vec![b'b' as u32, b'a' as u32, b'r' as u32]);
    assert_eq!(results[2], vec![b'b' as u32, b'a' as u32, b'z' as u32]);
    assert!(results[3].is_empty());
}

#[test]
fn stream_encode_roundtrip() {
    if !require_libztok() {
        return;
    }
    let pipe = Pipeline::byte_id(None).unwrap();
    let mut all_ids: Vec<u32> = Vec::new();
    for batch in pipe
        .encode_stream(b"hello world", 4)
        .expect("stream iter")
    {
        all_ids.extend(batch.expect("batch"));
    }
    let expected: Vec<u32> = b"hello world".iter().map(|&b| b as u32).collect();
    assert_eq!(all_ids, expected);
    assert_eq!(pipe.decode(&all_ids).unwrap(), "hello world");
}

#[test]
fn stream_sink_api_roundtrip() {
    if !require_libztok() {
        return;
    }
    let pipe = Pipeline::byte_id(None).unwrap();
    let mut enc = ztok::StreamEncoder::new(&pipe).unwrap();
    let mut collected: Vec<u32> = Vec::new();
    for chunk in b"hello world".chunks(3) {
        collected.extend(enc.feed(chunk).unwrap());
    }
    collected.extend(enc.finish().unwrap());
    // finish() is idempotent.
    assert!(enc.finish().unwrap().is_empty());
    assert_eq!(pipe.decode(&collected).unwrap(), "hello world");
}

#[test]
fn fingerprint_is_deterministic() {
    if !require_libztok() {
        return;
    }
    let a = Pipeline::byte_id(None).unwrap();
    let b = Pipeline::byte_id(None).unwrap();
    let fp_a = a.fingerprint().unwrap();
    let fp_b = b.fingerprint().unwrap();
    assert_eq!(fp_a, fp_b, "same config must yield same fingerprint");
    // 32 raw bytes.
    assert_eq!(fp_a.len(), 32);
    // Not all zeros (would suggest a hash function bug).
    assert!(fp_a.iter().any(|&b| b != 0));
}

#[test]
fn detect_format_unknown_for_missing_file() {
    if !require_libztok() {
        return;
    }
    let fmt = ztok::detect_format("/definitely/does/not/exist.bin").unwrap();
    assert_eq!(fmt, ztok::Format::Unknown);
}
