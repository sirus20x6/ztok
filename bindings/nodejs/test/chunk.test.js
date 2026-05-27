'use strict';

// Token-window chunking tests for the Node.js binding. Run over a
// byte_id pipeline (each input byte = one token) so chunk boundaries are
// predictable: "abcdefghij" is 10 tokens, one per byte. Mirrors
// bindings/python/tests/test_chunk.py.

const test = require('node:test');
const assert = require('node:assert/strict');

const ztok = require('..');

test('chunk non-overlapping', () => {
    const pipe = ztok.Pipeline.byteId();
    try {
        const chunks = pipe.chunk('abcdefghij', 4, { overlap: 0 });
        // 10 tokens / window 4, stride 4 -> [0,4) [4,8) [8,10).
        assert.equal(chunks.length, 3);
        const want = [
            { tokenStart: 0, tokenEnd: 4, byteStart: 0, byteEnd: 4, n: 4 },
            { tokenStart: 4, tokenEnd: 8, byteStart: 4, byteEnd: 8, n: 4 },
            { tokenStart: 8, tokenEnd: 10, byteStart: 8, byteEnd: 10, n: 2 },
        ];
        for (let i = 0; i < want.length; i++) {
            const c = chunks[i];
            assert.equal(c.tokenStart, want[i].tokenStart);
            assert.equal(c.tokenEnd, want[i].tokenEnd);
            assert.equal(c.byteStart, want[i].byteStart);
            assert.equal(c.byteEnd, want[i].byteEnd);
            assert.ok(c.ids instanceof Uint32Array);
            assert.equal(c.ids.length, want[i].n);
        }
    } finally {
        pipe.close();
    }
});

test('chunk overlap shares trailing/leading ids', () => {
    const pipe = ztok.Pipeline.byteId();
    try {
        const chunks = pipe.chunk('abcdefghij', 4, { overlap: 2 });
        assert.ok(chunks.length >= 2);
        // stride = 2, so the last 2 ids of chunk[i] equal the first 2 of
        // chunk[i+1].
        for (let i = 0; i < chunks.length - 1; i++) {
            const a = chunks[i].ids;
            const b = chunks[i + 1].ids;
            if (a.length >= 2 && b.length >= 2) {
                assert.deepEqual(
                    Array.from(a.slice(a.length - 2)),
                    Array.from(b.slice(0, 2))
                );
            }
        }
    } finally {
        pipe.close();
    }
});

test('chunk empty input returns []', () => {
    const pipe = ztok.Pipeline.byteId();
    try {
        assert.deepEqual(pipe.chunk('', 4), []);
    } finally {
        pipe.close();
    }
});

test('chunk bad args throw ZtokInvalidInputError', () => {
    const pipe = ztok.Pipeline.byteId();
    try {
        assert.throws(() => pipe.chunk('abc', 0), ztok.ZtokInvalidInputError);
        assert.throws(
            () => pipe.chunk('abc', 4, { overlap: 4 }),
            ztok.ZtokInvalidInputError
        );
    } finally {
        pipe.close();
    }
});

test('chunk boundary constant is wired up', () => {
    const pipe = ztok.Pipeline.byteId();
    try {
        // CHUNK_BOUNDARY_TOKEN is the default; an explicit pass must agree.
        const a = pipe.chunk('abcdefghij', 4);
        const b = pipe.chunk('abcdefghij', 4, {
            boundary: ztok.CHUNK_BOUNDARY_TOKEN,
        });
        assert.equal(a.length, b.length);
        for (let i = 0; i < a.length; i++) {
            assert.deepEqual(Array.from(a[i].ids), Array.from(b[i].ids));
        }
    } finally {
        pipe.close();
    }
});
