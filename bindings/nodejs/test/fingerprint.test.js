'use strict';

// Tokenizer-fingerprint tests for the Node.js binding. Mirror the
// rust/dotnet/java fingerprint tests: 32-byte length + determinism,
// plus a cross-binding golden value for the byte_id pipeline.

const test = require('node:test');
const assert = require('node:assert/strict');

const ztok = require('..');

// Golden fingerprint for the default byte_id pipeline. Computed directly
// from libztok's ztok_fingerprint and shared across the
// rust/dotnet/java/python/ruby/go bindings to confirm agreement.
const GOLDEN_BYTE_ID =
    '201ecf86554b5a970471e0189d7e78dc2c3df24519f7d5ebb0caacc86701e77c';

test('fingerprint is 32 bytes and non-zero', () => {
    const pipe = ztok.Pipeline.byteId();
    try {
        const fp = pipe.fingerprint();
        assert.ok(fp instanceof ztok.Fingerprint);
        assert.equal(fp.bytes.length, 32);
        assert.ok(fp.bytes.some((b) => b !== 0));
        assert.equal(fp.hex().length, 64);
    } finally {
        pipe.close();
    }
});

test('fingerprint is deterministic for same config', () => {
    const a = ztok.Pipeline.byteId();
    const b = ztok.Pipeline.byteId();
    try {
        const fa = a.fingerprint();
        const fb = b.fingerprint();
        assert.equal(fa.hex(), fb.hex());
        assert.ok(fa.equals(fb));
    } finally {
        a.close();
        b.close();
    }
});

test('fingerprint matches cross-binding golden value', () => {
    const pipe = ztok.Pipeline.byteId();
    try {
        assert.equal(pipe.fingerprint().hex(), GOLDEN_BYTE_ID);
    } finally {
        pipe.close();
    }
});
