'use strict';

// Tests for Pipeline.encodeWithOverlays (ztok_encode_with_overlays).

const test = require('node:test');
const assert = require('node:assert/strict');

const ztok = require('..');
const { tiktokenFixture } = require('./fixture');

test('ids match plain encode', () => {
    const pipe = ztok.Pipeline.fromTiktoken(tiktokenFixture(), { cl100k: true });
    try {
        const text = 'hello world';
        const plain = pipe.encode(text);
        const { ids, overlays } = pipe.encodeWithOverlays(
            text, [ztok.OVERLAY_BYTE_START, ztok.OVERLAY_BYTE_END]
        );
        assert.deepEqual(Array.from(ids), Array.from(plain));
        assert.deepEqual(
            new Set(overlays.keys()),
            new Set([ztok.OVERLAY_BYTE_START, ztok.OVERLAY_BYTE_END])
        );
    } finally {
        pipe.close();
    }
});

test('channel arrays have the same length as ids', () => {
    const pipe = ztok.Pipeline.fromTiktoken(tiktokenFixture(), { cl100k: true });
    try {
        const { ids, overlays } = pipe.encodeWithOverlays(
            'the quick brown fox',
            [
                ztok.OVERLAY_BYTE_START,
                ztok.OVERLAY_BYTE_END,
                ztok.OVERLAY_BOUNDARY,
                ztok.OVERLAY_PROVENANCE,
            ]
        );
        for (const [kind, vals] of overlays) {
            assert.equal(vals.length, ids.length, `channel ${kind} length mismatch`);
        }
    } finally {
        pipe.close();
    }
});

test('BYTE_START/BYTE_END give sensible spans', () => {
    const pipe = ztok.Pipeline.fromTiktoken(tiktokenFixture(), { cl100k: true });
    try {
        const text = 'hello world';
        const { ids, overlays } = pipe.encodeWithOverlays(
            text, [ztok.OVERLAY_BYTE_START, ztok.OVERLAY_BYTE_END]
        );
        const starts = overlays.get(ztok.OVERLAY_BYTE_START);
        const ends = overlays.get(ztok.OVERLAY_BYTE_END);
        const n = Buffer.byteLength(text, 'utf-8');
        for (let i = 0; i < ids.length; i++) {
            assert.ok(starts[i] < ends[i], `span ${i} not increasing`);
            assert.ok(ends[i] <= n, `span ${i} end out of bounds`);
        }
        // Spans tile the input left-to-right.
        assert.equal(starts[0], 0);
        assert.equal(ends[ends.length - 1], n);
        for (let i = 1; i < ids.length; i++) {
            assert.equal(starts[i], ends[i - 1], `gap before token ${i}`);
        }
    } finally {
        pipe.close();
    }
});

test('OPCODE domain channel is all zeros (no plugin)', () => {
    const pipe = ztok.Pipeline.fromTiktoken(tiktokenFixture(), { cl100k: true });
    try {
        const { ids, overlays } = pipe.encodeWithOverlays(
            'hello world', [ztok.OVERLAY_OPCODE]
        );
        const opcode = overlays.get(ztok.OVERLAY_OPCODE);
        assert.equal(opcode.length, ids.length);
        assert.ok(opcode.every((v) => v === 0), 'OPCODE channel not all zero');
    } finally {
        pipe.close();
    }
});

test('byteId single-byte spans', () => {
    const pipe = ztok.Pipeline.byteId();
    try {
        const { ids, overlays } = pipe.encodeWithOverlays(
            'hi', [ztok.OVERLAY_BYTE_START, ztok.OVERLAY_BYTE_END]
        );
        assert.deepEqual(Array.from(ids), [0x68, 0x69]);
        assert.deepEqual(Array.from(overlays.get(ztok.OVERLAY_BYTE_START)), [0, 1]);
        assert.deepEqual(Array.from(overlays.get(ztok.OVERLAY_BYTE_END)), [1, 2]);
    } finally {
        pipe.close();
    }
});

test('empty input returns empty channels', () => {
    const pipe = ztok.Pipeline.byteId();
    try {
        const { ids, overlays } = pipe.encodeWithOverlays(
            '', [ztok.OVERLAY_BYTE_START, ztok.OVERLAY_OPCODE]
        );
        assert.equal(ids.length, 0);
        assert.equal(overlays.get(ztok.OVERLAY_BYTE_START).length, 0);
        assert.equal(overlays.get(ztok.OVERLAY_OPCODE).length, 0);
    } finally {
        pipe.close();
    }
});

test('no channels returns just ids', () => {
    const pipe = ztok.Pipeline.fromTiktoken(tiktokenFixture(), { cl100k: true });
    try {
        const { ids, overlays } = pipe.encodeWithOverlays('hello world', []);
        assert.deepEqual(Array.from(ids), Array.from(pipe.encode('hello world')));
        assert.equal(overlays.size, 0);
    } finally {
        pipe.close();
    }
});
