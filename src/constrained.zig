//! Grammar-constrained tokenization — a token-prefix automaton.
//!
//! Given a grammar and the bytes generated so far, compute the set of vocab
//! token ids whose bytes keep the running output a valid PREFIX of some
//! string the grammar accepts. This is the core primitive behind constrained
//! LLM decoding (logit masking à la outlines / guidance): each generation
//! step, mask out every token whose id is not in `allowedNextTokens`.
//!
//! Acceptance rule: a token `t` is allowed after `generated_bytes` iff
//!   `generated_bytes ++ t.bytes`
//! is still a prefix of some grammar-matching string (i.e. not `.dead`).
//! We additionally report whether the *current* `generated_bytes` is itself
//! an accepting state (`.full`), meaning generation may legally stop.
//!
//! Backends
//! --------
//!   - Regex grammar: wraps `hf_regex` (literals, classes incl `\p{L}\p{N}`,
//!     quantifiers, groups, alternation, anchors, lookahead). We do NOT
//!     reimplement regex matching here — we lean on the engine's Thompson
//!     NFA prefix walker via the small `matchPrefix` helper below, which is
//!     the only regex-specific glue this module adds.
//!   - Char-set grammar: the trivial "any sequence over an allowed byte set"
//!     grammar — every token whose bytes are all drawn from the set.
//!
//! Performance
//! -----------
//! v1 iterates the whole vocab once per step: O(vocab * |token bytes| *
//! |program|). For a 100k-token vocab this is fine for interactive decode
//! but redundant — most tokens share byte prefixes. The principled
//! optimization is a *token trie* keyed by token bytes walked in lockstep
//! with the regex NFA: descend the trie, carrying the NFA thread-set, and
//! prune whole subtrees the instant the NFA goes dead. That collapses the
//! per-step cost to O(distinct live trie nodes) and is the standard
//! automaton-intersection trick (outlines' "index"). Left as a follow-up;
//! the O(n) loop here is the correct, simple baseline it would replace.

const std = @import("std");
const hf_regex = @import("hf_regex.zig");
const vocab_mod = @import("vocab.zig");
const TokenId = @import("token.zig").TokenId;

pub const ConstraintError = error{
    EmptyCharSet,
} || hf_regex.CompileError || std.mem.Allocator.Error;

/// Prefix-acceptance verdict for a candidate byte string. Mirrors
/// `hf_regex.PrefixStatus` but is the module's public vocabulary so callers
/// need not import the regex engine.
pub const PrefixStatus = enum {
    /// No grammar string starts with these bytes — reject.
    dead,
    /// A valid prefix, but not yet a complete match — keep going.
    partial,
    /// A complete grammar match — valid, and generation may stop here.
    full,
};

const Kind = enum { regex, char_set };

pub const Constraint = struct {
    allocator: std.mem.Allocator,
    vocab: *const vocab_mod.Vocab,
    kind: Kind,

    // regex backend
    re: ?hf_regex.Regex = null,
    // char_set backend: membership bitset over 256 byte values
    allowed_bytes: [256]bool = [_]bool{false} ** 256,

    /// Build a constraint from a regex grammar. The accepted language is the
    /// set of strings the (anchored-at-0) pattern matches in full.
    pub fn initRegex(
        allocator: std.mem.Allocator,
        vocab: *const vocab_mod.Vocab,
        pattern: []const u8,
    ) ConstraintError!Constraint {
        const re = try hf_regex.compile(allocator, pattern);
        return .{
            .allocator = allocator,
            .vocab = vocab,
            .kind = .regex,
            .re = re,
        };
    }

    /// Build a constraint whose language is any sequence (incl. empty) over
    /// the byte set `allowed` — equivalent to the regex `[<allowed>]*`.
    pub fn initCharSet(
        allocator: std.mem.Allocator,
        vocab: *const vocab_mod.Vocab,
        allowed: []const u8,
    ) ConstraintError!Constraint {
        if (allowed.len == 0) return error.EmptyCharSet;
        var c: Constraint = .{
            .allocator = allocator,
            .vocab = vocab,
            .kind = .char_set,
        };
        for (allowed) |b| c.allowed_bytes[b] = true;
        return c;
    }

    pub fn deinit(self: *Constraint) void {
        if (self.re) |*re| re.deinit();
        self.* = undefined;
    }

    /// Classify a candidate byte string under the grammar: is it dead, a
    /// live partial, or a complete match?
    pub fn matchPrefix(self: *const Constraint, text: []const u8) PrefixStatus {
        switch (self.kind) {
            .regex => {
                // The regex engine answers prefix acceptance directly via its
                // Thompson NFA thread-set walk (see hf_regex.prefixStatus).
                return switch (self.re.?.prefixStatus(text)) {
                    .dead => .dead,
                    .partial => .partial,
                    .full => .full,
                };
            },
            .char_set => {
                for (text) |b| {
                    if (!self.allowed_bytes[b]) return .dead;
                }
                // Any all-allowed string is a complete match of `[set]*`
                // (which also accepts the empty string), so it is `.full`:
                // every prefix is itself accepting.
                return .full;
            },
        }
    }

    /// True if `generated_bytes` is an accepting state — generation may stop.
    pub fn isAccepting(self: *const Constraint, generated_bytes: []const u8) bool {
        return self.matchPrefix(generated_bytes) == .full;
    }

    /// Fill `out` with the ids of every vocab token whose bytes, appended to
    /// `generated_bytes`, keep the output a valid grammar prefix (not dead).
    ///
    /// `out` must be sized to at least `vocab.count` bits; it is reset to all
    /// zero first, then the allowed ids are set.
    pub fn allowedNextTokens(
        self: *const Constraint,
        generated_bytes: []const u8,
        out: *std.DynamicBitSet,
    ) !void {
        std.debug.assert(out.capacity() >= self.vocab.count);
        out.setRangeValue(.{ .start = 0, .end = out.capacity() }, false);

        // Scratch buffer reused across tokens: generated_bytes ++ token.bytes.
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(self.allocator);
        try buf.appendSlice(self.allocator, generated_bytes);
        const base_len = buf.items.len;

        // v1: linear scan over the vocab. See module docs for the trie
        // optimization that would replace this loop.
        var id: TokenId = 0;
        while (id < self.vocab.count) : (id += 1) {
            const tb = self.vocab.tokenBytes(id);
            // Skip empty-byte tokens: appending nothing never changes state,
            // and admitting them would let decoding stall forever.
            if (tb.len == 0) continue;
            buf.items.len = base_len;
            try buf.appendSlice(self.allocator, tb);
            if (self.matchPrefix(buf.items) != .dead) out.set(id);
        }
    }
};

// =====================================================================
// Tests

const testing = std.testing;

/// Build a throwaway SoA vocab from a list of token byte strings.
fn buildVocab(allocator: std.mem.Allocator, toks: []const []const u8) !vocab_mod.Vocab {
    var total: usize = 0;
    for (toks) |t| total += t.len;
    const bytes = try allocator.alloc(u8, total);
    const offsets = try allocator.alloc(u32, toks.len + 1);
    var off: u32 = 0;
    for (toks, 0..) |t, i| {
        offsets[i] = off;
        @memcpy(bytes[off .. off + t.len], t);
        off += @intCast(t.len);
    }
    offsets[toks.len] = off;
    return .{
        .allocator = allocator,
        .bytes = bytes,
        .offsets = offsets,
        .ranks = null,
        .count = @intCast(toks.len),
    };
}

test "regex [0-9]+ allows only digit-bytes tokens" {
    const toks = [_][]const u8{ "0", "1", "12", "3", "a", "1a", "" };
    var v = try buildVocab(testing.allocator, &toks);
    defer v.deinit();

    var c = try Constraint.initRegex(testing.allocator, &v, "[0-9]+");
    defer c.deinit();

    var bs = try std.DynamicBitSet.initEmpty(testing.allocator, v.count);
    defer bs.deinit();

    // From empty: digit tokens & all-digit multi-byte tokens allowed;
    // "a", "1a" and the empty token are not.
    try c.allowedNextTokens("", &bs);
    try testing.expect(bs.isSet(0)); // "0"
    try testing.expect(bs.isSet(1)); // "1"
    try testing.expect(bs.isSet(2)); // "12"
    try testing.expect(bs.isSet(3)); // "3"
    try testing.expect(!bs.isSet(4)); // "a"
    try testing.expect(!bs.isSet(5)); // "1a"
    try testing.expect(!bs.isSet(6)); // ""

    // After "12": "3" yes, "a" no.
    try c.allowedNextTokens("12", &bs);
    try testing.expect(bs.isSet(3)); // "3"
    try testing.expect(!bs.isSet(4)); // "a"

    // Accepting after "1" (a complete [0-9]+ match), so generation may stop.
    try testing.expect(c.isAccepting("1"));
    // Empty string is NOT accepting for [0-9]+ (needs >= 1 digit) but is a
    // live partial.
    try testing.expectEqual(PrefixStatus.partial, c.matchPrefix(""));
    // "1a" is dead.
    try testing.expectEqual(PrefixStatus.dead, c.matchPrefix("1a"));
}

test "regex (true|false) prefix steering" {
    const toks = [_][]const u8{ "true", "false", "tr", "ue", "x", "fa" };
    var v = try buildVocab(testing.allocator, &toks);
    defer v.deinit();

    var c = try Constraint.initRegex(testing.allocator, &v, "(true|false)");
    defer c.deinit();

    var bs = try std.DynamicBitSet.initEmpty(testing.allocator, v.count);
    defer bs.deinit();

    // After "tr": only continuations toward "true" survive. "ue" completes
    // it; "x"/"fa"/"false"/"true" do not extend the "tr" prefix.
    try c.allowedNextTokens("tr", &bs);
    try testing.expect(bs.isSet(1) == false); // "false"
    try testing.expect(bs.isSet(3)); // "ue"  -> "true"
    try testing.expect(!bs.isSet(2)); // "tr" -> "trtr" dead
    try testing.expect(!bs.isSet(4)); // "x"
    try testing.expect(!bs.isSet(5)); // "fa"

    // "tr" is a live partial, not accepting; "true" is accepting.
    try testing.expectEqual(PrefixStatus.partial, c.matchPrefix("tr"));
    try testing.expect(c.isAccepting("true"));
    try testing.expect(c.isAccepting("false"));
    try testing.expectEqual(PrefixStatus.dead, c.matchPrefix("truex"));

    // From empty, both whole words and viable prefixes are allowed.
    try c.allowedNextTokens("", &bs);
    try testing.expect(bs.isSet(0)); // "true"
    try testing.expect(bs.isSet(1)); // "false"
    try testing.expect(bs.isSet(2)); // "tr"
    try testing.expect(bs.isSet(5)); // "fa"
    try testing.expect(!bs.isSet(3)); // "ue" doesn't start a word
    try testing.expect(!bs.isSet(4)); // "x"
}

test "char set {a,b,c} admits only those bytes" {
    const toks = [_][]const u8{ "a", "b", "c", "abc", "ab", "ad", "d", "" };
    var v = try buildVocab(testing.allocator, &toks);
    defer v.deinit();

    var c = try Constraint.initCharSet(testing.allocator, &v, "abc");
    defer c.deinit();

    var bs = try std.DynamicBitSet.initEmpty(testing.allocator, v.count);
    defer bs.deinit();

    try c.allowedNextTokens("", &bs);
    try testing.expect(bs.isSet(0)); // "a"
    try testing.expect(bs.isSet(1)); // "b"
    try testing.expect(bs.isSet(2)); // "c"
    try testing.expect(bs.isSet(3)); // "abc"
    try testing.expect(bs.isSet(4)); // "ab"
    try testing.expect(!bs.isSet(5)); // "ad" contains 'd'
    try testing.expect(!bs.isSet(6)); // "d"
    try testing.expect(!bs.isSet(7)); // "" empty token skipped

    // Mid-stream the rule is unchanged.
    try c.allowedNextTokens("aba", &bs);
    try testing.expect(bs.isSet(0));
    try testing.expect(!bs.isSet(6)); // "d"

    // A char-set state is always accepting (the language includes every
    // all-allowed string).
    try testing.expect(c.isAccepting("abcabc"));
    try testing.expectEqual(PrefixStatus.dead, c.matchPrefix("abd"));
    try testing.expectError(error.EmptyCharSet, Constraint.initCharSet(testing.allocator, &v, ""));
}

test "regex digits with explicit anchors" {
    const toks = [_][]const u8{ "1", "2", "a" };
    var v = try buildVocab(testing.allocator, &toks);
    defer v.deinit();

    // Anchored on both ends should behave identically for the prefix walk.
    var c = try Constraint.initRegex(testing.allocator, &v, "^[0-9]+$");
    defer c.deinit();

    try testing.expectEqual(PrefixStatus.partial, c.matchPrefix(""));
    try testing.expectEqual(PrefixStatus.full, c.matchPrefix("12"));
    try testing.expectEqual(PrefixStatus.dead, c.matchPrefix("1a"));
}
