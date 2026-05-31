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

// x86-64 machine code: `48 89 d8` mov rax,rbx / `e8 00000000` call rel32 /
// `c3` ret. With the byteId pipeline each byte is its own token.
const X86_64_CODE = Buffer.from([0x48, 0x89, 0xd8, 0xe8, 0x00, 0x00, 0x00, 0x00, 0xc3]);

test('setOverlayDomain(X86_64) populates the OPCODE channel', () => {
    const pipe = ztok.Pipeline.byteId();
    try {
        // Default domain (NONE): OPCODE is zero-filled.
        const none = pipe.encodeWithOverlays(X86_64_CODE, [ztok.OVERLAY_OPCODE]);
        const opcodeNone = Array.from(none.overlays.get(ztok.OVERLAY_OPCODE));
        assert.equal(opcodeNone.length, X86_64_CODE.length);
        assert.ok(opcodeNone.every((v) => v === 0));

        // After selecting x86-64 the OPCODE channel is populated.
        pipe.setOverlayDomain(ztok.OVERLAY_DOMAIN_X86_64);
        const x86 = pipe.encodeWithOverlays(X86_64_CODE, [ztok.OVERLAY_OPCODE]);
        const opcodeX86 = Array.from(x86.overlays.get(ztok.OVERLAY_OPCODE));
        // Tokenization unchanged; only the domain channel differs.
        assert.deepEqual(Array.from(x86.ids), Array.from(none.ids));
        assert.notDeepEqual(opcodeX86, opcodeNone);
        assert.ok(opcodeX86.some((v) => v !== 0));
    } finally {
        pipe.close();
    }
});

test('setOverlayDomain(NONE) round-trips back to zero-fill', () => {
    const pipe = ztok.Pipeline.byteId();
    try {
        pipe.setOverlayDomain(ztok.OVERLAY_DOMAIN_X86_64);
        pipe.setOverlayDomain(ztok.OVERLAY_DOMAIN_NONE);
        const { overlays } = pipe.encodeWithOverlays(X86_64_CODE, [ztok.OVERLAY_OPCODE]);
        const opcode = Array.from(overlays.get(ztok.OVERLAY_OPCODE));
        assert.ok(opcode.every((v) => v === 0));
    } finally {
        pipe.close();
    }
});

test('setOverlayDomain rejects an unknown domain', () => {
    const pipe = ztok.Pipeline.byteId();
    try {
        assert.throws(() => pipe.setOverlayDomain(999));
    } finally {
        pipe.close();
    }
});
