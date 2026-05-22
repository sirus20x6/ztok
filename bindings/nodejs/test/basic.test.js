'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');

const ztok = require('..');
const { tiktokenFixture } = require('./fixture');

test('version returns a non-empty string', () => {
    const v = ztok.version();
    assert.equal(typeof v, 'string');
    assert.ok(v.includes('.'));
    const parts = v.split('.');
    assert.ok(/^\d+$/.test(parts[0]));
    assert.ok(/^\d+$/.test(parts[1]));
});

test('byteId encode/decode roundtrip', () => {
    const pipe = ztok.Pipeline.byteId();
    try {
        const ids = pipe.encode('hi');
        assert.deepEqual(Array.from(ids), [0x68, 0x69]);
        assert.equal(pipe.decode(ids), 'hi');
        // decodeBytes path
        assert.ok(Buffer.from('hi').equals(pipe.decodeBytes(ids)));
    } finally {
        pipe.close();
    }
});

test('encode/decode roundtrip on a real BPE pipeline', () => {
    const pipe = ztok.Pipeline.fromTiktoken(tiktokenFixture(), { cl100k: true });
    try {
        const text = 'hello world';
        const ids = pipe.encode(text);
        assert.ok(ids.length > 0);
        assert.equal(pipe.decode(ids), text);
    } finally {
        pipe.close();
    }
});

test('100-line round-trip stress', () => {
    const pipe = ztok.Pipeline.fromTiktoken(tiktokenFixture(), { cl100k: true });
    try {
        const snippets = [
            'hello world',
            'the quick brown fox',
            'foo bar baz',
            'hello there hello world',
            'the the the',
            '  the  ',
            'foo',
            'hello',
            ' world',
            'bar baz',
        ];
        for (let i = 0; i < 100; i++) {
            const line = snippets[i % snippets.length];
            const ids = pipe.encode(line);
            assert.equal(pipe.decode(ids), line,
                `round-trip failed for ${JSON.stringify(line)}`);
        }
    } finally {
        pipe.close();
    }
});

test('empty input returns empty ids', () => {
    const pipe = ztok.Pipeline.byteId();
    try {
        assert.equal(pipe.encode('').length, 0);
        assert.equal(pipe.decode(new Uint32Array(0)), '');
    } finally {
        pipe.close();
    }
});

test('close() is idempotent and post-close ops throw', () => {
    const pipe = ztok.Pipeline.byteId();
    pipe.close();
    pipe.close(); // no-op
    assert.throws(() => pipe.encode('z'), ztok.ZtokError);
});

test('invalid input raises ZtokInvalidInputError', () => {
    // Passing an unknown normalizer kind through the raw config trips
    // ZTOK_ERR_INVALID_INPUT -> ZtokInvalidInputError.
    assert.throws(
        () => ztok.Pipeline.byteId({ normalizer: 999 }),
        ztok.ZtokInvalidInputError
    );
});
