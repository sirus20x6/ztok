'use strict';

// PRNG-driven round-trip fuzz harness for the ztok Node.js binding.
//
// Mirrors fuzz/encode_decode.zig in shape: a deterministic xorshift32
// PRNG mutates the byte input each iteration; encode-then-decode must
// round-trip exactly. The pipeline is byteId (identity normalizer +
// identity pre-tok + byte_id model + concat decoder) — by construction
// each input byte maps to one id and decoding concatenates them back,
// so for ANY byte sequence we must have decodeBytes(encode(x)) == x.
//
// Skips cleanly if libztok cannot be loaded (mirrors the
// fixture-missing skip pattern in loaders.test.js).

const test = require('node:test');
const assert = require('node:assert/strict');

let ztok;
let loadFailure = null;
try {
    ztok = require('..');
    // Touch the lib loader so missing-shared-lib environments fail at
    // module init rather than mid-test.
    ztok.version();
} catch (e) {
    loadFailure = e;
}

// Deterministic seed: same value as the Python/Ruby harnesses so
// failures at a given iteration are cross-language reproducible. The
// fixed seed may be overridden per-run via FUZZ_SEED env (hex like
// "0xdeadbeef" or decimal); ZTOK_FUZZ_ITERS scales the iteration count
// for nightly fuzz workflows.
const DEFAULT_SEED = 0xfeedb0b;
const DEFAULT_ITERATIONS = 1000;
const MAX_LEN = 256;

function envInt(name, fallback) {
    const raw = process.env[name];
    if (!raw) return fallback;
    // Number() handles "0x..." hex and decimal alike; NaN falls back.
    const parsed = Number(raw);
    return Number.isFinite(parsed) ? (parsed >>> 0) : fallback;
}

const SEED = envInt('FUZZ_SEED', DEFAULT_SEED);
const ITERATIONS = envInt('ZTOK_FUZZ_ITERS', DEFAULT_ITERATIONS);

// Tiny xorshift32 — Node has no seedable PRNG in the stdlib, so we
// inline one. Period 2^32 - 1, fine for fuzzing a 1000-iteration loop.
function makeXorshift32(seed) {
    // xorshift32 has a degenerate state at 0; coerce away from it.
    let state = (seed >>> 0) || 0xdeadbeef;
    return function next() {
        state ^= state << 13;
        state ^= state >>> 17;
        state ^= state << 5;
        // Mask to keep the result in the unsigned 32-bit range.
        state >>>= 0;
        return state;
    };
}

function randomBytes(next, maxLen) {
    const n = next() % (maxLen + 1);
    if (n === 0) return Buffer.alloc(0);
    const out = Buffer.alloc(n);
    for (let i = 0; i < n; i++) {
        out[i] = next() & 0xff;
    }
    return out;
}

test('byteId round-trip fuzz: PRNG-mutated iterations', (t) => {
    if (loadFailure) {
        t.skip(`libztok not available: ${loadFailure.message}`);
        return;
    }

    const next = makeXorshift32(SEED);
    const failures = [];

    const pipe = ztok.Pipeline.byteId();
    try {
        for (let i = 0; i < ITERATIONS; i++) {
            const data = randomBytes(next, MAX_LEN);
            const ids = pipe.encode(data);
            // byte_id maps 1:1 — id count must equal input byte length.
            assert.equal(
                ids.length,
                data.length,
                `iter ${i}: byteId produced ${ids.length} ids for ${data.length} bytes`
            );
            const roundtrip = pipe.decodeBytes(ids);
            if (!roundtrip.equals(data)) {
                failures.push({ i, input: data.toString('hex'), out: roundtrip.toString('hex') });
                // Don't flood — first 5 are enough to root-cause.
                if (failures.length >= 5) break;
            }
        }
    } finally {
        // Release the native handle deterministically; the
        // FinalizationRegistry would eventually do it on GC but explicit
        // close keeps the test isolated.
        pipe.close();
    }

    if (failures.length > 0) {
        const msg = failures
            .map((f) => `  iter ${f.i}: in=${f.input} out=${f.out}`)
            .join('\n');
        assert.fail(`byteId round-trip mismatches:\n${msg}`);
    }
});
