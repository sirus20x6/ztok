'use strict';

// Engram n-gram hashing tests for the Node.js binding. Mirrors
// src/ngram.zig's contract (and bindings/python/tests/test_ngram.py):
// deterministic multi-head token-n-gram hashes, row-major
// [position][head], with positions = ids.length - n + 1. Hashes are
// raw uint64, returned as a BigUint64Array.

const test = require('node:test');
const assert = require('node:assert/strict');

const ztok = require('..');

test('hashNgrams length math', () => {
    const ids = [1, 2, 3, 4, 5];
    // 5 ids, n=2 -> 4 positions; heads=3 -> 12 hashes.
    const out = ztok.hashNgrams(ids, 2, 3);
    assert.ok(out instanceof BigUint64Array);
    assert.equal(out.length, 4 * 3);
});

test('hashNgrams deterministic', () => {
    const ids = [7, 8, 9, 10, 11, 12];
    const a = ztok.hashNgrams(ids, 3, 4);
    const b = ztok.hashNgrams(ids, 3, 4);
    assert.deepEqual(Array.from(a), Array.from(b));
    for (const h of a) {
        assert.equal(typeof h, 'bigint');
        assert.ok(h >= 0n);
    }
});

test('hashNgrams head independence', () => {
    // The heads of a single position should not all collide.
    const out = ztok.hashNgrams([42, 43, 44], 2, 4);
    const firstPosition = Array.from(out.slice(0, 4));
    assert.ok(new Set(firstPosition.map(String)).size > 1);
});

test('hashNgrams short stream and degenerate args -> empty', () => {
    // Stream shorter than one window -> empty.
    assert.equal(ztok.hashNgrams([1, 2], 3, 2).length, 0);
    // Degenerate args -> empty (no error).
    assert.equal(ztok.hashNgrams([], 1, 1).length, 0);
    assert.equal(ztok.hashNgrams([1, 2, 3], 0, 1).length, 0);
    assert.equal(ztok.hashNgrams([1, 2, 3], 2, 0).length, 0);
});

test('hashNgrams accepts Uint32Array input', () => {
    const a = ztok.hashNgrams(Uint32Array.from([1, 2, 3, 4]), 2, 2);
    const b = ztok.hashNgrams([1, 2, 3, 4], 2, 2);
    assert.deepEqual(Array.from(a), Array.from(b));
});

test('hashNgramsBatch matches single', () => {
    const streams = [
        [1, 2, 3, 4],
        [],            // empty -> no hashes
        [9],           // shorter than window -> no hashes
        [5, 6, 7, 8, 9],
    ];
    const pool = new ztok.BatchPool({ workers: 2 });
    try {
        const batched = ztok.hashNgramsBatch(pool, streams, 2, 3);
        assert.equal(batched.length, streams.length);
        for (let i = 0; i < streams.length; i++) {
            const single = ztok.hashNgrams(streams[i], 2, 3);
            assert.deepEqual(Array.from(batched[i]), Array.from(single));
        }
    } finally {
        pool.close();
    }
});

test('hashNgramsBatch with empty stream list returns []', () => {
    const pool = new ztok.BatchPool({ workers: 2 });
    try {
        assert.deepEqual(ztok.hashNgramsBatch(pool, [], 2, 3), []);
    } finally {
        pool.close();
    }
});
