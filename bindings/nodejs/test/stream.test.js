'use strict';

// Streaming encode tests. Wraps the post-1.18 C ABI's `ztok_stream_*`
// family. The streamed output must equal `encode()` byte-for-byte when
// the pre-tokenizer is real (cl100k splits on whitespace, giving safe
// cuts on every span).

const test = require('node:test');
const assert = require('node:assert/strict');

const ztok = require('..');
const { tiktokenFixture } = require('./fixture');

function collect(generator) {
    const out = [];
    for (const chunk of generator) {
        assert.ok(chunk instanceof Uint32Array, 'chunk must be a Uint32Array');
        for (const id of chunk) out.push(id);
    }
    return out;
}

test('stream single chunk matches encode()', () => {
    const pipe = ztok.Pipeline.fromTiktoken(tiktokenFixture(), { cl100k: true });
    try {
        const text = 'hello world the quick brown fox';
        const want = Array.from(pipe.encode(text));
        const got = collect(pipe.encodeStream(text));
        assert.deepEqual(got, want);
    } finally {
        pipe.close();
    }
});

test('stream many tiny chunks matches encode()', () => {
    const pipe = ztok.Pipeline.fromTiktoken(tiktokenFixture(), { cl100k: true });
    try {
        const text = 'hello world the quick brown fox hello world the quick brown fox';
        const want = Array.from(pipe.encode(text));
        const got = collect(pipe.encodeStream(text, { chunkSize: 4 }));
        assert.deepEqual(got, want);
    } finally {
        pipe.close();
    }
});

test('stream defers a mid-UTF8 codepoint cut (byteId)', () => {
    const pipe = ztok.Pipeline.byteId();
    try {
        const text = 'héllo'; // 'h' + 0xC3 0xA9 + 'llo' = 6 bytes
        const want = Array.from(pipe.encode(text));
        // chunkSize=2 cuts mid-é (after 0xC3); the stream must defer the
        // trailing byte to the next feed and never emit a garbage id.
        const got = collect(pipe.encodeStream(text, { chunkSize: 2 }));
        assert.deepEqual(got, want);
    } finally {
        pipe.close();
    }
});

test('stream empty input yields nothing', () => {
    const pipe = ztok.Pipeline.byteId();
    try {
        const got = [...pipe.encodeStream('')];
        assert.deepEqual(got, []);
    } finally {
        pipe.close();
    }
});

test('stream emits ids incrementally (more than one yield)', () => {
    const pipe = ztok.Pipeline.fromTiktoken(tiktokenFixture(), { cl100k: true });
    try {
        // The stream encoder flushes per-feed (after the safe pre-tok
        // cut inside that feed). With cl100k splitting on whitespace, a
        // ~240 KB input chunked at 4 KiB yields ~one batch per feed —
        // i.e. multiple yields, not one giant flush at the end.
        const text = ('hello world '.repeat(1000) + '\n').repeat(20);
        let yields = 0;
        let totalIds = 0;
        for (const chunk of pipe.encodeStream(text, { chunkSize: 4096 })) {
            yields++;
            totalIds += chunk.length;
        }
        assert.ok(yields > 1, `expected multiple yields, got ${yields}`);
        assert.equal(totalIds, pipe.encode(text).length);
    } finally {
        pipe.close();
    }
});

test('stream multiline matches encode()', () => {
    const pipe = ztok.Pipeline.fromTiktoken(tiktokenFixture(), { cl100k: true });
    try {
        const text = 'hello world\nthe quick brown fox\nfoo bar baz\n';
        const want = Array.from(pipe.encode(text));
        const got = collect(pipe.encodeStream(text, { chunkSize: 8 }));
        assert.deepEqual(got, want);
    } finally {
        pipe.close();
    }
});
