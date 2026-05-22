//! PRNG-driven round-trip fuzz harness for the ztok Rust binding.
//!
//! Mirrors `fuzz/encode_decode.zig` and the Python/Node/Ruby fuzz tests
//! in shape: a deterministic PRNG mutates the byte input each
//! iteration, encode-then-decode must round-trip exactly.
//!
//! The pipeline is `byte_id` (identity normalizer + identity pre-tok +
//! byte_id model + concat decoder); by construction each input byte
//! maps to one id and decoding concatenates them back, so for any byte
//! sequence the invariant `decode(encode(x)) == x` must hold.
//!
//! Seed defaults to `0xFEEDB0B` to match the Python/Ruby/Node harnesses
//! — failures at a given iteration are cross-language reproducible.
//! Override via `FUZZ_SEED` (decimal or `0x...` hex) and
//! `ZTOK_FUZZ_ITERS` env vars for nightly runs.

use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

use ztok::Pipeline;

#[allow(clippy::unusual_byte_groupings)] // Matches Python/Ruby/Node fuzz seed.
const DEFAULT_SEED: u64 = 0xFEEDB0B;
const DEFAULT_ITERATIONS: usize = 1000;
const MAX_LEN: usize = 256;

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|raw| {
            if let Some(hex) = raw.strip_prefix("0x").or_else(|| raw.strip_prefix("0X")) {
                u64::from_str_radix(hex, 16).ok()
            } else {
                raw.parse::<u64>().ok()
            }
        })
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(default)
}

#[test]
fn byte_id_roundtrip_fuzz() {
    // Skip if libztok isn't usable (mirrors smoke.rs's pattern). We
    // can't catch a dlopen failure here — if the shared library is
    // missing the binary fails to load before this function runs.
    let Ok(_) = ztok::version() else {
        eprintln!("ztok::version failed; skipping fuzz harness");
        return;
    };

    let seed = env_u64("FUZZ_SEED", DEFAULT_SEED);
    let iters = env_usize("ZTOK_FUZZ_ITERS", DEFAULT_ITERATIONS);
    let mut rng = StdRng::seed_from_u64(seed);

    let pipe = Pipeline::byte_id(None).expect("byte_id pipeline");
    let mut failures: Vec<(usize, Vec<u8>, Vec<u8>)> = Vec::new();

    for i in 0..iters {
        let len = (rng.next_u32() as usize) % (MAX_LEN + 1);
        let mut data = vec![0u8; len];
        rng.fill_bytes(&mut data);

        let ids = pipe.encode_bytes(&data).expect("encode_bytes");
        assert_eq!(
            ids.len(),
            data.len(),
            "iter {i}: byte_id produced {} ids for {} bytes",
            ids.len(),
            data.len()
        );
        let roundtrip = pipe.decode_bytes(&ids).expect("decode_bytes");
        if roundtrip != data {
            failures.push((i, data, roundtrip));
            if failures.len() >= 5 {
                // First 5 mismatches are enough to root-cause.
                break;
            }
        }
    }

    if !failures.is_empty() {
        let mut msg = String::from("byte_id round-trip mismatches:\n");
        for (i, inp, out) in &failures {
            msg.push_str(&format!("  iter {i}: in={inp:?} out={out:?}\n"));
        }
        panic!("{msg}");
    }
}
