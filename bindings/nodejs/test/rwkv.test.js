'use strict';

// RWKV "World" tokenizer tests for the Node.js binding. Loads the real
// rwkv_vocab_v20230424.txt fixture (skipped when absent) and checks ztok
// reproduces the canonical reference encodings, matching the in-tree
// gate in src/rwkv_world.zig and bindings/python/tests/test_rwkv.py.

const fs = require('fs');
const path = require('path');
const test = require('node:test');
const assert = require('node:assert/strict');

const ztok = require('..');

// bindings/nodejs/test/ -> repo root -> bench/vocabs/...
const VOCAB = path.resolve(
    __dirname, '..', '..', '..', 'bench', 'vocabs', 'rwkv_vocab_v20230424.txt'
);

// Golden id sequences captured from BlinkDL's canonical reference
// tokenizer (see bench/rwkv_parity.py / src/rwkv_world.zig).
const GOLDEN = [
    ['Hello, world!', [33155, 45, 40213, 34]],
    ['emoji 😀🚀✨ test', [34295, 33, 3319, 153, 129, 3319, 155, 129, 10059, 32223]],
    ['0 1 2 10 99 100', [49, 284, 285, 3483, 3572, 3483, 49]],
];

test('RWKV matches reference golden encodings', (t) => {
    if (!fs.existsSync(VOCAB)) {
        t.skip(`RWKV vocab fixture not present at ${VOCAB}`);
        return;
    }
    const pipe = ztok.Pipeline.fromRWKV(VOCAB);
    try {
        for (const [text, want] of GOLDEN) {
            assert.deepEqual(Array.from(pipe.encode(text)), want);
        }
    } finally {
        pipe.close();
    }
});

test('RWKV round-trips', (t) => {
    if (!fs.existsSync(VOCAB)) {
        t.skip(`RWKV vocab fixture not present at ${VOCAB}`);
        return;
    }
    const pipe = ztok.Pipeline.fromRWKV(VOCAB);
    try {
        for (const [text] of GOLDEN) {
            const ids = pipe.encode(text);
            assert.equal(pipe.decode(ids), text);
        }
    } finally {
        pipe.close();
    }
});

test('RWKV auto-detects via detectFormat + fromPath', (t) => {
    if (!fs.existsSync(VOCAB)) {
        t.skip(`RWKV vocab fixture not present at ${VOCAB}`);
        return;
    }
    assert.equal(ztok.detectFormat(VOCAB), 'rwkv');
    const pipe = ztok.Pipeline.fromPath(VOCAB);
    try {
        assert.deepEqual(Array.from(pipe.encode('Hello, world!')), [33155, 45, 40213, 34]);
    } finally {
        pipe.close();
    }
});
