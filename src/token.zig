//! Core integer types. Tokens are u32 throughout — large enough for any
//! vocab in the wild (TokenMonster's largest are ~64k, HF's bigger models
//! ~256k, but 32-bit gives headroom forever and stays cache-friendly).

const std = @import("std");

pub const TokenId = u32;

/// Half-open byte range `[start, end)` into some owning buffer. Used by the
/// pre-tokenizer to hand chunks to the model without copying.
pub const Span = struct {
    start: u32,
    end: u32,

    pub fn len(self: Span) u32 {
        return self.end - self.start;
    }

    pub fn slice(self: Span, buf: []const u8) []const u8 {
        return buf[self.start..self.end];
    }
};

/// A typed annotation channel aligned 1:1 with an emitted token stream:
/// for an encoding, `channel[i]` describes the token `ids[i]`. Overlays
/// never alter tokenization or split tokens — they only describe it, so
/// requesting overlays leaves the id stream byte-identical to a plain
/// encode.
///
/// The cheap channels (`byte_start`, `byte_end`, `boundary`,
/// `provenance`) are derivable from pipeline state the encoder already
/// computes. The domain channels (`opcode_class`, `operand_class`,
/// `symbol_ref`, `hunk`) require a domain-specific normalizer plugin to
/// populate them; absent that plugin they are zero-filled.
///
/// Non-exhaustive: domain plugins may define their own channel kinds at
/// values >= `user_base` (0x8000) without colliding with future
/// first-party kinds.
pub const OverlayKind = enum(u16) {
    /// `offsets[i].start` — byte offset of the token in the original input.
    byte_start = 0,
    /// `offsets[i].end` — exclusive end byte offset in the original input.
    byte_end = 1,
    /// Bitset of `Boundary` flags describing what the token starts.
    boundary = 2,
    /// Domain (asm/binary): normalized opcode class. 0 without a plugin.
    opcode_class = 3,
    /// Domain (asm/binary): normalized operand class. 0 without a plugin.
    operand_class = 4,
    /// Domain: index into a side symbol table; 0 = none. 0 without a plugin.
    symbol_ref = 5,
    /// Domain: diff-hunk id. 0 without a plugin.
    hunk = 6,
    /// `Provenance` value: which input source produced the token.
    provenance = 7,
    /// Domain plugins claim channel kinds at or above this value.
    user_base = 0x8000,
    _,
};

/// Bit flags for the `boundary` overlay channel (OR-combined).
pub const Boundary = struct {
    /// Token is the first emitted for its pre-tokenizer chunk.
    pub const chunk_start: u32 = 0x1;
    /// Token's `byte_start` lands on a UTF-8 leading byte in the
    /// original input (i.e. not mid-codepoint).
    pub const codepoint_start: u32 = 0x2;
};

/// Values for the `provenance` overlay channel.
pub const Provenance = struct {
    /// Token was produced by the model from input text.
    pub const model_text: u32 = 0;
    /// Token was injected by the added-token scanner (special token).
    pub const special: u32 = 1;
};

test "Span len/slice" {
    const buf = "hello world";
    const s: Span = .{ .start = 6, .end = 11 };
    try std.testing.expectEqual(@as(u32, 5), s.len());
    try std.testing.expectEqualStrings("world", s.slice(buf));
}
