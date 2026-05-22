'use strict';

// Shared test fixtures — mirrors bindings/python/tests/conftest.py.
// Writes a tiny synthetic .tiktoken vocab covering all 256 single bytes
// plus a handful of merges, into a per-process temp dir. The same file
// fixture is shared across every test in the directory.

const fs = require('fs');
const os = require('os');
const path = require('path');

const EXTRAS = [
    'he', 'hel', 'hell', 'hello',
    ' w', ' wo', ' wor', ' worl', ' world',
    'th', 'the', ' th', ' the',
    'fo', 'foo', 'bar', 'baz',
    ' quick', ' brown', ' fox',
];

let cachedDir = null;

function tempDir() {
    if (cachedDir) return cachedDir;
    cachedDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ztok-node-'));
    return cachedDir;
}

function writeTiktokenVocab(filePath) {
    const lines = [];
    let rank = 0;
    for (let b = 0; b < 256; b++) {
        const token = Buffer.from([b]).toString('base64');
        lines.push(`${token} ${rank++}`);
    }
    for (const extra of EXTRAS) {
        const token = Buffer.from(extra, 'utf-8').toString('base64');
        lines.push(`${token} ${rank++}`);
    }
    fs.writeFileSync(filePath, lines.join('\n') + '\n');
}

function tiktokenFixture() {
    const p = path.join(tempDir(), 'synthetic_cl100k.tiktoken');
    if (!fs.existsSync(p)) writeTiktokenVocab(p);
    return p;
}

module.exports = { tempDir, tiktokenFixture };
