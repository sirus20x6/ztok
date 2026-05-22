//! Pre-tokenizers split normalized input into byte spans the model
//! encodes independently.
//!
//! Some pre-tokenizers (HF ByteLevel) also TRANSFORM the bytes — they
//! map raw bytes through `byte_to_unicode` per chunk, producing a
//! separate buffer that the spans index into. The `Result` struct
//! captures both cases: `.data` is either a borrowed slice of the
//! caller's input or an owned-allocated mapped buffer, indicated by
//! `.owned_data`.

const std = @import("std");
const Span = @import("token.zig").Span;
const cl100k = @import("cl100k.zig");
const cl100k_split = cl100k.split;
const hf_bytelevel = @import("hf_bytelevel_pretok.zig");

pub const Result = struct {
    /// Bytes the spans index into. May be the caller's input (borrowed)
    /// or an owned buffer the pre-tokenizer produced.
    data: []const u8,
    /// True if `data` was allocated by the pre-tokenizer and the caller
    /// must free it through `deinit`.
    owned_data: bool,
    /// Flat span array. Always owned.
    spans: []Span,

    pub fn deinit(self: *Result, allocator: std.mem.Allocator) void {
        if (self.owned_data) allocator.free(@constCast(self.data));
        allocator.free(self.spans);
    }
};

pub const PreTokenizer = union(enum) {
    /// Single span covering the entire input.
    identity,
    /// tiktoken's cl100k_base regex.
    cl100k,
    /// HF GPT-2 ByteLevel: GPT-2 regex split + byte_to_unicode mapping
    /// in one pass. Returns the MAPPED bytes as the data buffer.
    hf_byte_level,
    /// HF Sequence pretok: a chain of small ops (Split/ByteLevel/Digits/
    /// Punctuation/Whitespace/Metaspace) parsed from a tokenizer.json
    /// `pre_tokenizer.Sequence`. The chain object is heap-owned by the
    /// caller (typically the `HFTokenizer` that parsed the JSON); this
    /// variant carries a `*const` pointer so the union stays small.
    /// See `hf_bytelevel_pretok.Chain` for the chain representation.
    chain: *const hf_bytelevel.Chain,

    /// Find a position near `desired` where `input` may be cut into two
    /// halves whose independently pre-tokenized outputs concatenate to
    /// the same span list as running pre-tokenization on `input` whole.
    ///
    /// Used by `Pipeline.encodeChunked` to split a single input across
    /// BatchPool workers without altering the token sequence. Returns
    /// null if no safe cut exists in `[desired - window, desired + window]`.
    ///
    /// For `identity`, every codepoint boundary is safe (the model sees
    /// each whole half as a single span anyway). For `cl100k`, see
    /// `cl100k.findSafeCut` for the boundary rule. For `hf_byte_level`,
    /// the byte→unicode remap is per-byte so any regex-match boundary in
    /// the raw input is also safe at the remap stage; see
    /// `hf_bytelevel.findSafeCut` for the boundary rule. For `chain`,
    /// we conservatively use the same `\n`-before-non-ws rule as
    /// `hf_byte_level` — newlines are safe boundary points for nearly
    /// every HF pretok variant because nothing in the supported op set
    /// re-flows across `\n`.
    pub fn findSafeCut(
        self: PreTokenizer,
        input: []const u8,
        desired: usize,
        window: usize,
    ) ?usize {
        return switch (self) {
            .identity => findCodepointSafeCut(input, desired, window),
            .cl100k => cl100k.findSafeCut(input, desired, window),
            .hf_byte_level => hf_bytelevel.findSafeCut(input, desired, window),
            .chain => hf_bytelevel.findSafeCut(input, desired, window),
        };
    }

    /// Worst-case byte expansion factor: `output_bytes <= input_bytes * factor`.
    /// `hf_byte_level` maps each input byte to a codepoint in U+0021..U+0142;
    /// those in U+0100..U+0142 take 2 UTF-8 bytes, so the factor is 2.
    pub fn maxByteExpansion(self: PreTokenizer) usize {
        return switch (self) {
            .identity, .cl100k => 1,
            .hf_byte_level => 2,
            .chain => |c| c.maxByteExpansion(),
        };
    }

    pub fn split(
        self: PreTokenizer,
        allocator: std.mem.Allocator,
        input: []const u8,
    ) !Result {
        switch (self) {
            .identity => {
                const spans = try allocator.alloc(Span, 1);
                spans[0] = .{ .start = 0, .end = @intCast(input.len) };
                return .{ .data = input, .owned_data = false, .spans = spans };
            },
            .cl100k => {
                const spans = try cl100k_split(allocator, input);
                return .{ .data = input, .owned_data = false, .spans = spans };
            },
            .hf_byte_level => {
                const r = try hf_bytelevel.splitAndMap(allocator, input);
                // Re-pack r's Span type into our Span type (same shape but
                // distinct nominal type via re-export path).
                const out_spans = try allocator.alloc(Span, r.spans.len);
                for (r.spans, 0..) |s, i| out_spans[i] = .{ .start = s.start, .end = s.end };
                allocator.free(r.spans);
                return .{ .data = r.mapped, .owned_data = true, .spans = out_spans };
            },
            .chain => |c| {
                const r = try hf_bytelevel.runChain(allocator, c, input);
                const out_spans = try allocator.alloc(Span, r.spans.len);
                for (r.spans, 0..) |s, i| out_spans[i] = .{ .start = s.start, .end = s.end };
                allocator.free(r.spans);
                return .{ .data = r.mapped, .owned_data = true, .spans = out_spans };
            },
        }
    }
};

/// Snap `desired` to the nearest UTF-8 codepoint boundary within
/// `window` bytes. Bytes 0x00..0x7F and 0xC0..0xFF are leading; 0x80..0xBF
/// are continuation bytes (mid-codepoint). Position 0 and `input.len`
/// are always boundaries.
fn findCodepointSafeCut(input: []const u8, desired: usize, window: usize) ?usize {
    if (desired == 0 or desired == input.len) return desired;
    if (desired > input.len) return null;

    const isLead = struct {
        fn f(b: u8) bool {
            return b < 0x80 or b >= 0xC0;
        }
    }.f;

    if (isLead(input[desired])) return desired;

    // Scan backwards then forwards for the nearest leading byte.
    var back: usize = 1;
    while (back <= window and back <= desired) : (back += 1) {
        if (isLead(input[desired - back])) return desired - back;
    }
    var fwd: usize = 1;
    while (fwd <= window and desired + fwd <= input.len) : (fwd += 1) {
        if (desired + fwd == input.len or isLead(input[desired + fwd])) return desired + fwd;
    }
    return null;
}

test "identity pre-tokenizer yields one span" {
    var r = try (PreTokenizer{ .identity = {} }).split(std.testing.allocator, "hello world");
    defer r.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 1), r.spans.len);
    try std.testing.expectEqual(@as(u32, 11), r.spans[0].len());
    try std.testing.expect(!r.owned_data);
}

test "cl100k pre-tokenizer dispatch" {
    var r = try (PreTokenizer{ .cl100k = {} }).split(std.testing.allocator, "hello world");
    defer r.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 2), r.spans.len);
    try std.testing.expectEqualStrings("hello", r.spans[0].slice(r.data));
    try std.testing.expectEqualStrings(" world", r.spans[1].slice(r.data));
}

test "hf_byte_level pre-tokenizer maps and splits" {
    var r = try (PreTokenizer{ .hf_byte_level = {} }).split(std.testing.allocator, " hi");
    defer r.deinit(std.testing.allocator);
    try std.testing.expect(r.owned_data);
    try std.testing.expectEqual(@as(usize, 1), r.spans.len);
    // " hi" maps to "Ġhi" — the space becomes U+0120 (UTF-8 0xC4 0xA0).
    try std.testing.expectEqualStrings("\xC4\xA0hi", r.spans[0].slice(r.data));
}
