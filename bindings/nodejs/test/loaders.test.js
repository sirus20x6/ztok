'use strict';

// Auto-detect + Pipeline.fromPath dispatch tests. Routes through the C
// ABI's `ztok_auto_detect` (post-1.18 agent C) so signature drift in the
// C enum surfaces immediately.

const fs = require('fs');
const os = require('os');
const path = require('path');
const test = require('node:test');
const assert = require('node:assert/strict');

const ztok = require('..');
const { tempDir, tiktokenFixture } = require('./fixture');

test('detectFormat: tiktoken', () => {
    assert.equal(ztok.detectFormat(tiktokenFixture()), 'tiktoken');
});

test('detectFormat: hf_json', () => {
    const p = path.join(tempDir(), 'tokenizer.json');
    fs.writeFileSync(p, '{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]}}');
    assert.equal(ztok.detectFormat(p), 'hf_json');
});

test('detectFormat: ztm', () => {
    const p = path.join(tempDir(), 'v.ztm');
    fs.writeFileSync(p, Buffer.concat([Buffer.from('ZTM\x01'), Buffer.alloc(60)]));
    assert.equal(ztok.detectFormat(p), 'ztm');
});

test('detectFormat: sentencepiece (skipped if fixture missing)', (t) => {
    const sp = '/thearray/git/ztok/bench/vocabs/llama2.model';
    if (!fs.existsSync(sp)) {
        t.skip('llama2.model fixture not present');
        return;
    }
    assert.equal(ztok.detectFormat(sp), 'sentencepiece');
});

test('detectFormat: unknown for missing file', () => {
    const missing = path.join(tempDir(), 'no_such_file.bin');
    assert.equal(ztok.detectFormat(missing), 'unknown');
});

test('Pipeline.fromPath loads .tiktoken via auto-detect', () => {
    const pipe = ztok.Pipeline.fromPath(tiktokenFixture());
    try {
        const ids = pipe.encode('hello world');
        assert.equal(pipe.decode(ids), 'hello world');
    } finally {
        pipe.close();
    }
});

test('Pipeline.fromPath loads SentencePiece .model (skipped if missing)', (t) => {
    const sp = '/thearray/git/ztok/bench/vocabs/llama2.model';
    if (!fs.existsSync(sp)) {
        t.skip('llama2.model fixture not present');
        return;
    }
    const pipe = ztok.Pipeline.fromPath(sp, { unkId: 0 });
    try {
        const ids = pipe.encode('hello world');
        assert.ok(ids.length > 0);
    } finally {
        pipe.close();
    }
});

test('Pipeline.fromPath loads .ztm (skipped if missing)', (t) => {
    const zm = '/thearray/git/ztok/bench/vocabs/tm_englishcode_32k.ztm';
    if (!fs.existsSync(zm)) {
        t.skip('tm_englishcode_32k.ztm fixture not present');
        return;
    }
    const pipe = ztok.Pipeline.fromPath(zm);
    try {
        const ids = pipe.encode('hello world');
        assert.ok(ids.length > 0);
    } finally {
        pipe.close();
    }
});

test('Pipeline.fromPath rejects an unknown format', () => {
    const p = path.join(tempDir(), 'mystery.bin');
    fs.writeFileSync(p, Buffer.from([0xff, 0xfe, 0xfd, 0xfc, 0x20, 0x6e, 0x6f]));
    assert.throws(() => ztok.Pipeline.fromPath(p), ztok.ZtokInvalidInputError);
});
