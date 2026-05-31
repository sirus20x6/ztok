'use strict';

// Mistral Tekken tokenizer tests for the Node.js binding. Loads the real
// mistral_nemo_tekken.json fixture (skipped when absent) and checks ztok
// reproduces the golden ids verified against mistral_common 1.8.6.

const fs = require('fs');
const path = require('path');
const test = require('node:test');
const assert = require('node:assert/strict');

const ztok = require('..');

// bindings/nodejs/test/ -> repo root -> bench/vocabs/...
const VOCAB = path.resolve(
    __dirname, '..', '..', '..', 'bench', 'vocabs', 'mistral_nemo_tekken.json'
);

// Golden id sequences verified against mistral_common 1.8.6 on
// bench/vocabs/mistral_nemo_tekken.json.
const GOLDEN = [
    ['Hello, world!', [22177, 1044, 4304, 1033]],
    ['The quick brown fox', [1784, 7586, 22980, 94137]],
    [' and the', [1321, 1278]],
];

test('Tekken matches reference golden encodings', (t) => {
    if (!fs.existsSync(VOCAB)) {
        t.skip(`Tekken vocab fixture not present at ${VOCAB}`);
        return;
    }
    const pipe = ztok.Pipeline.fromTekken(VOCAB);
    try {
        for (const [text, want] of GOLDEN) {
            assert.deepEqual(Array.from(pipe.encode(text)), want, `encode mismatch for ${text}`);
        }
    } finally {
        pipe.close();
    }
});

test('Tekken auto-detects via detectFormat + fromPath', (t) => {
    if (!fs.existsSync(VOCAB)) {
        t.skip(`Tekken vocab fixture not present at ${VOCAB}`);
        return;
    }
    assert.equal(ztok.detectFormat(VOCAB), 'tekken');
    const pipe = ztok.Pipeline.fromPath(VOCAB);
    try {
        assert.deepEqual(Array.from(pipe.encode('Hello, world!')), [22177, 1044, 4304, 1033]);
    } finally {
        pipe.close();
    }
});
