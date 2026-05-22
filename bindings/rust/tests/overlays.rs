//! Tests for `Pipeline::encode_with_overlays` (ztok_encode_with_overlays).
//!
//! Mirrors bindings/python/tests/test_overlays.py. Skips gracefully when
//! libztok isn't available — same pattern as smoke.rs.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::OnceLock;

use ztok::{Config, Decoder, Normalizer, OverlayKind, Pipeline, PreTokenizer};

fn require_libztok() -> bool {
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

/// Synthetic .tiktoken vocab — same shape as smoke.rs / the other bindings.
fn tiktoken_fixture() -> Option<PathBuf> {
    static FIXTURE: OnceLock<Option<PathBuf>> = OnceLock::new();
    FIXTURE
        .get_or_init(|| {
            let dir = std::env::temp_dir().join(format!("ztok-rust-ov-{}", std::process::id()));
            fs::create_dir_all(&dir).ok()?;
            let path = dir.join("synthetic_cl100k.tiktoken");
            let mut f = fs::File::create(&path).ok()?;
            let mut rank = 0u32;
            for b in 0u32..256 {
                writeln!(f, "{} {}", base64_encode(&[b as u8]), rank).ok()?;
                rank += 1;
            }
            for extra in [
                "he", "hel", "hell", "hello", " w", " wo", " wor", " worl", " world", "th", "the",
                " th", " the", "fo", "foo", "bar", "baz", " quick", " brown", " fox",
            ] {
                writeln!(f, "{} {}", base64_encode(extra.as_bytes()), rank).ok()?;
                rank += 1;
            }
            Some(path)
        })
        .clone()
}

fn base64_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
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

fn bpe_pipeline() -> Option<Pipeline> {
    let fixture = tiktoken_fixture()?;
    Pipeline::from_tiktoken(
        &fixture,
        Some(Config {
            normalizer: Normalizer::Identity,
            pre_tokenizer: PreTokenizer::Cl100k,
            decoder: Decoder::Concat,
        }),
    )
    .ok()
}

#[test]
fn ids_match_plain_encode() {
    if !require_libztok() {
        return;
    }
    let Some(pipe) = bpe_pipeline() else { return };
    let text = "hello world";
    let plain = pipe.encode(text).unwrap();
    let (ids, overlays) = pipe
        .encode_with_overlays(text, &[OverlayKind::ByteStart, OverlayKind::ByteEnd])
        .unwrap();
    assert_eq!(ids, plain, "overlays must not change tokenization");
    assert_eq!(overlays.len(), 2);
    assert!(overlays.contains_key(&OverlayKind::ByteStart));
    assert!(overlays.contains_key(&OverlayKind::ByteEnd));
}

#[test]
fn channel_lengths_equal_ids() {
    if !require_libztok() {
        return;
    }
    let Some(pipe) = bpe_pipeline() else { return };
    let (ids, overlays) = pipe
        .encode_with_overlays(
            "the quick brown fox",
            &[
                OverlayKind::ByteStart,
                OverlayKind::ByteEnd,
                OverlayKind::Boundary,
                OverlayKind::Provenance,
            ],
        )
        .unwrap();
    for (kind, values) in &overlays {
        assert_eq!(values.len(), ids.len(), "channel {kind:?} length mismatch");
    }
}

#[test]
fn byte_spans_are_sensible() {
    if !require_libztok() {
        return;
    }
    let Some(pipe) = bpe_pipeline() else { return };
    let text = "hello world";
    let (_ids, overlays) = pipe
        .encode_with_overlays(text, &[OverlayKind::ByteStart, OverlayKind::ByteEnd])
        .unwrap();
    let starts = &overlays[&OverlayKind::ByteStart];
    let ends = &overlays[&OverlayKind::ByteEnd];
    let n = text.len() as u32;
    assert!(!starts.is_empty());
    for (&s, &e) in starts.iter().zip(ends.iter()) {
        assert!(s < e && e <= n, "bad span ({s}, {e}) for {n} bytes");
    }
    assert_eq!(starts[0], 0);
    assert_eq!(*ends.last().unwrap(), n);
    for i in 1..starts.len() {
        assert_eq!(starts[i], ends[i - 1], "spans must tile left-to-right");
    }
}

#[test]
fn byte_id_single_byte_spans() {
    if !require_libztok() {
        return;
    }
    let pipe = Pipeline::byte_id(None).unwrap();
    let (ids, overlays) = pipe
        .encode_with_overlays("hi", &[OverlayKind::ByteStart, OverlayKind::ByteEnd])
        .unwrap();
    assert_eq!(ids, vec![0x68, 0x69]);
    assert_eq!(overlays[&OverlayKind::ByteStart], vec![0, 1]);
    assert_eq!(overlays[&OverlayKind::ByteEnd], vec![1, 2]);
}

#[test]
fn opcode_domain_channel_is_all_zero() {
    if !require_libztok() {
        return;
    }
    let Some(pipe) = bpe_pipeline() else { return };
    let (ids, overlays) = pipe
        .encode_with_overlays("hello world", &[OverlayKind::Opcode])
        .unwrap();
    let opcode = &overlays[&OverlayKind::Opcode];
    assert_eq!(opcode.len(), ids.len());
    assert!(
        opcode.iter().all(|&v| v == 0),
        "OPCODE must be zero-filled without a domain plugin"
    );
}

#[test]
fn empty_input_returns_empty_channels() {
    if !require_libztok() {
        return;
    }
    let pipe = Pipeline::byte_id(None).unwrap();
    let (ids, overlays) = pipe
        .encode_with_overlays("", &[OverlayKind::ByteStart, OverlayKind::Opcode])
        .unwrap();
    assert!(ids.is_empty());
    assert_eq!(overlays[&OverlayKind::ByteStart], Vec::<u32>::new());
    assert_eq!(overlays[&OverlayKind::Opcode], Vec::<u32>::new());
}

#[test]
fn no_channels_returns_just_ids() {
    if !require_libztok() {
        return;
    }
    let Some(pipe) = bpe_pipeline() else { return };
    let (ids, overlays) = pipe.encode_with_overlays("hello world", &[]).unwrap();
    assert_eq!(ids, pipe.encode("hello world").unwrap());
    assert!(overlays.is_empty());
}

#[test]
fn duplicate_kinds_rejected() {
    if !require_libztok() {
        return;
    }
    let pipe = Pipeline::byte_id(None).unwrap();
    let err = pipe
        .encode_with_overlays("hi", &[OverlayKind::ByteStart, OverlayKind::ByteStart])
        .unwrap_err();
    assert_eq!(err, ztok::Error::InvalidInput);
}
