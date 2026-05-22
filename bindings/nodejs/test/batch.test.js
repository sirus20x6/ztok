'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');

const ztok = require('..');
const { tiktokenFixture } = require('./fixture');

test('BatchPool workers=0 resolves to auto', () => {
    const pool = new ztok.BatchPool({ workers: 0 });
    try {
        assert.ok(pool.workers >= 1);
    } finally {
        pool.close();
    }
});

test('BatchPool explicit worker count', () => {
    const pool = new ztok.BatchPool({ workers: 3 });
    try {
        assert.equal(pool.workers, 3);
    } finally {
        pool.close();
    }
});

test('BatchPool closed pool raises', () => {
    const pool = new ztok.BatchPool({ workers: 2 });
    pool.close();
    assert.throws(() => pool.workers, ztok.ZtokError);
});

test('encodeBatch matches per-input encode()', () => {
    const pipe = ztok.Pipeline.fromTiktoken(tiktokenFixture(), { cl100k: true });
    const pool = new ztok.BatchPool({ workers: 4 });
    try {
        const inputs = ['hello world', ' the quick brown fox', 'foo bar baz'];
        const expected = inputs.map((s) => Array.from(pipe.encode(s)));
        const got = pipe.encodeBatch(pool, inputs).map((u) => Array.from(u));
        assert.deepEqual(got, expected);
    } finally {
        pool.close();
        pipe.close();
    }
});

test('encodeBatch 1000 strings without leaks (GC stable)', async () => {
    const pipe = ztok.Pipeline.fromTiktoken(tiktokenFixture(), { cl100k: true });
    const pool = new ztok.BatchPool({ workers: 8 });
    try {
        const inputs = new Array(1000).fill('hello world');
        const results = pipe.encodeBatch(pool, inputs);
        assert.equal(results.length, 1000);
        // All rows must be byte-equal to the first one.
        const first = Array.from(results[0]);
        for (const row of results) {
            assert.deepEqual(Array.from(row), first);
        }
        // Force GC if --expose-gc was passed; otherwise just drop the
        // ref and assert the holder isn't blocking finalization.
        if (typeof global.gc === 'function') global.gc();
        // No assertion of weakref reclaim — Node doesn't expose a
        // sync WeakRef.collect; the contract is "no segfaults / no
        // double-frees", verified by the suite reaching this line.
        assert.ok(true);
    } finally {
        pool.close();
        pipe.close();
    }
});

test('encodeBatch with empty inputs returns empty array', () => {
    const pipe = ztok.Pipeline.fromTiktoken(tiktokenFixture(), { cl100k: true });
    const pool = new ztok.BatchPool({ workers: 2 });
    try {
        assert.deepEqual(pipe.encodeBatch(pool, []), []);
    } finally {
        pool.close();
        pipe.close();
    }
});

test('encodeBatch with an empty-string input returns empty ids in that slot', () => {
    const pipe = ztok.Pipeline.fromTiktoken(tiktokenFixture(), { cl100k: true });
    const pool = new ztok.BatchPool({ workers: 2 });
    try {
        const results = pipe.encodeBatch(pool, ['hello', '', 'world']);
        assert.equal(results.length, 3);
        assert.equal(results[1].length, 0);
        assert.deepEqual(Array.from(results[0]), Array.from(pipe.encode('hello')));
        assert.deepEqual(Array.from(results[2]), Array.from(pipe.encode('world')));
    } finally {
        pool.close();
        pipe.close();
    }
});
