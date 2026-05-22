//! Mistral Tekken pre-tokenization regex, implemented by hand.
//!
//! Tekken ships its own tiktoken-style pattern in `config.pattern`. It is
//! NOT the cl100k_base pattern — feeding cl100k splits to the Tekken BPE
//! diverges from the reference (`mistral_common`) on case boundaries,
//! digit grouping, combining marks, and the `/`-terminated punctuation
//! arm. The pattern (for Mistral-Nemo / Pixtral / Codestral, config v3+):
//!
//! ```
//! [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+
//! |[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*
//! |\p{N}
//! | ?[^\s\p{L}\p{N}]+[\r\n/]*
//! |\s*[\r\n]+
//! |\s+(?!\S)
//! |\s+
//! ```
//!
//! Two things make this a hand-rolled splitter rather than a reuse of the
//! shared `hf_regex` engine:
//!
//!   1. `hf_regex` collapses the `\p{Lu}` / `\p{Lt}` / `\p{Lm}` / `\p{Lo}`
//!      / `\p{Ll}` subcategories to their parent `\p{L}` (documented limit
//!      in `hf_regex.zig`). That collapse turns arms 1 and 2 into a single
//!      `\p{L}+` arm and destroys Tekken's case split
//!      (`CamelCase` -> `Camel`, `Case`).
//!   2. We must NOT byte-map the spans (Tekken's BPE eats raw bytes, like
//!      tiktoken — there is no GPT-2 byte_to_unicode remap), so the HF
//!      ByteLevel chain op is wrong here too.
//!
//! `unicode_props` exposes the exact subcategory predicates we need
//! (`isLu`/`isLt`/`isLm`/`isLo`/`isLl`/`isMark`), so we mirror the cl100k
//! splitter's structure with Tekken's arm definitions. Read-only use of
//! the shared `unicode_props` tables — no shared code is modified.

const std = @import("std");
const Span = @import("token.zig").Span;
const unicode_props = @import("unicode_props.zig");

const Decoded = struct { cp: u21, len: usize };

fn decodeAt(s: []const u8, i: usize) ?Decoded {
    if (i >= s.len) return null;
    const b0 = s[i];
    if (b0 < 0x80) return .{ .cp = b0, .len = 1 };
    const seq_len = std.unicode.utf8ByteSequenceLength(b0) catch return .{ .cp = b0, .len = 1 };
    if (i + seq_len > s.len) return .{ .cp = b0, .len = 1 };
    const cp = std.unicode.utf8Decode(s[i .. i + seq_len]) catch return .{ .cp = b0, .len = 1 };
    return .{ .cp = cp, .len = seq_len };
}

fn isLetter(cp: u21) bool {
    if (cp < 0x80) return (cp >= 'A' and cp <= 'Z') or (cp >= 'a' and cp <= 'z');
    return unicode_props.isLetter(cp);
}

fn isNumber(cp: u21) bool {
    if (cp < 0x80) return cp >= '0' and cp <= '9';
    return unicode_props.isNumber(cp);
}

fn isWhitespace(cp: u21) bool {
    if (cp < 0x80) return cp == ' ' or cp == '\t' or cp == '\n' or cp == '\r' or cp == 0x0B or cp == 0x0C;
    return unicode_props.isWhitespace(cp);
}

/// `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]` — the "uppercase-ish" class used by
/// the first half of both word arms. (Lu/Lt are case; Lm/Lo are caseless
/// letters; M is combining marks.)
fn isUpperClass(cp: u21) bool {
    if (cp < 0x80) return cp >= 'A' and cp <= 'Z';
    return unicode_props.isLu(cp) or unicode_props.isLt(cp) or
        unicode_props.isLm(cp) or unicode_props.isLo(cp) or unicode_props.isMark(cp);
}

/// `[\p{Ll}\p{Lm}\p{Lo}\p{M}]` — the "lowercase-ish" class used by the
/// second half of both word arms.
fn isLowerClass(cp: u21) bool {
    if (cp < 0x80) return cp >= 'a' and cp <= 'z';
    return unicode_props.isLl(cp) or unicode_props.isLm(cp) or
        unicode_props.isLo(cp) or unicode_props.isMark(cp);
}

/// True if `cp` may serve as the optional `[^\r\n\p{L}\p{N}]?` prefix
/// codepoint that opens both word arms.
fn isWordPrefix(cp: u21) bool {
    return cp != '\r' and cp != '\n' and !isLetter(cp) and !isNumber(cp);
}

/// Split input bytes into Tekken pattern spans. Caller owns the slice.
pub fn split(allocator: std.mem.Allocator, input: []const u8) ![]Span {
    var out = try allocator.alloc(Span, input.len + 1);
    var n: usize = 0;
    var i: usize = 0;
    while (i < input.len) {
        const took = matchOne(input, i);
        const len = if (took == 0) 1 else took;
        out[n] = .{ .start = @intCast(i), .end = @intCast(i + len) };
        n += 1;
        i += len;
    }
    return allocator.realloc(out, n);
}

// Try the 7 Tekken alternatives in priority order at position `i`.
// Returns the match length in bytes (0 only on a degenerate no-match,
// which the caller treats as a single-byte advance).
fn matchOne(s: []const u8, i: usize) usize {
    // Arms 1 & 2 are the word arms. The regex engine tries arm 1 first,
    // then arm 2; the longer/earlier-listed alternative wins at the same
    // start. We compute both and pick per regex-alternation semantics
    // (first arm that matches a non-empty string wins; on a tie of which
    // matched, arm 1 is listed first).
    if (matchWord(s, i)) |len| return len;
    // Arm 3: \p{N} — exactly ONE numeral codepoint.
    if (matchOneDigit(s, i)) |len| return len;
    // Arm 4:  ?[^\s\p{L}\p{N}]+[\r\n/]*
    if (matchPunct(s, i)) |len| return len;
    // Arm 5: \s*[\r\n]+
    if (matchWsNewline(s, i)) |len| return len;
    // Arm 6: \s+(?!\S)
    if (matchTrailingWs(s, i)) |len| return len;
    // Arm 7: \s+
    if (matchWs(s, i)) |len| return len;
    return 0;
}

/// Arms 1 and 2 combined. Both share the optional prefix; they differ in
/// where the upper-run / lower-run split falls:
///   arm 1: prefix? UPPER* LOWER+   (requires >=1 lower-ish at the end)
///   arm 2: prefix? UPPER+ LOWER*   (requires >=1 upper-ish in the middle)
/// A POSIX-style alternation tries arm 1 first; if it matches a non-empty
/// span, it wins. Otherwise arm 2 is tried. We replicate that ordering.
fn matchWord(s: []const u8, i: usize) ?usize {
    // Resolve the optional prefix: take one codepoint if it is a
    // word-prefix char AND it is followed by something that lets the arm
    // continue (an upper-ish or lower-ish letter). The regex `?` is
    // greedy but will give the prefix back if doing so is the only way to
    // match; here, if the prefix char is itself consumed, the run that
    // follows must be a letter class. If not, we skip the prefix and try
    // a bare letter run starting at `i`.
    if (matchWordFrom(s, i, true)) |len| return len;
    return matchWordFrom(s, i, false);
}

/// `allow_prefix` controls whether we may consume the leading optional
/// `[^\r\n\p{L}\p{N}]?` codepoint. Returns the total byte length of a
/// non-empty match, or null.
fn matchWordFrom(s: []const u8, i: usize, allow_prefix: bool) ?usize {
    var p = i;
    if (allow_prefix) {
        // The prefix attempt only fires when the leading codepoint is a
        // word-prefix char followed by a letter class. Otherwise it has
        // nothing to contribute over the no-prefix attempt, so bail and
        // let the no-prefix attempt handle a bare letter run.
        const d = decodeAt(s, p) orelse return null;
        if (!isWordPrefix(d.cp)) return null;
        const d2 = decodeAt(s, p + d.len) orelse return null;
        if (!(isUpperClass(d2.cp) or isLowerClass(d2.cp))) return null;
        p += d.len;
    }

    // Arm 1: UPPER* LOWER+
    if (matchArm1(s, p)) |body_len| {
        if (body_len > 0) return (p - i) + body_len;
    }
    // Arm 2: UPPER+ LOWER*
    if (matchArm2(s, p)) |body_len| {
        if (body_len > 0) return (p - i) + body_len;
    }
    return null;
}

/// `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+`
/// Greedy upper-ish run, then a required (>=1) lower-ish run. Because the
/// classes overlap (Lm/Lo/M are in both), the greedy `UPPER*` can swallow
/// codepoints the trailing `LOWER+` needs; regex backtracks to leave at
/// least one lower-ish codepoint. We mirror that with backtracking on the
/// upper run.
fn matchArm1(s: []const u8, start: usize) ?usize {
    // Greedily consume the upper-ish run, then require a LOWER+ run,
    // backing off the upper run a codepoint at a time when necessary.
    var p = start;
    var upper_end = start;
    while (decodeAt(s, upper_end)) |d| {
        if (!isUpperClass(d.cp)) break;
        upper_end += d.len;
    }
    // Now require LOWER+ after some prefix of [start, upper_end]. Try the
    // greediest upper run first, backing off one codepoint at a time until
    // a lower-ish codepoint is available to start the LOWER+ run.
    p = upper_end;
    while (true) {
        // Try to match LOWER+ at p.
        var q = p;
        var lower_count: usize = 0;
        while (decodeAt(s, q)) |d| {
            if (!isLowerClass(d.cp)) break;
            q += d.len;
            lower_count += 1;
        }
        if (lower_count > 0) return q - start;
        // No lower run at p. Back off one codepoint of the upper run (if
        // any) so that codepoint can seed LOWER+ (valid only if it is
        // itself lower-ish).
        if (p == start) return null;
        // Find the previous codepoint boundary < p within [start, p).
        const prev = prevBoundary(s, start, p);
        const d = decodeAt(s, prev) orelse return null;
        // The backed-off codepoint must be lower-ish to start LOWER+.
        if (!isLowerClass(d.cp)) return null;
        p = prev;
    }
}

/// `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*`
/// Required (>=1) upper-ish run, then a greedy lower-ish run.
fn matchArm2(s: []const u8, start: usize) ?usize {
    var p = start;
    var upper_count: usize = 0;
    while (decodeAt(s, p)) |d| {
        if (!isUpperClass(d.cp)) break;
        p += d.len;
        upper_count += 1;
    }
    if (upper_count == 0) return null;
    // Greedy LOWER* — note Lm/Lo/M are in both classes, so the lower run
    // can extend past where the upper run stopped only on Ll/Lm/Lo/M; the
    // regex's LOWER* simply continues consuming lower-ish codepoints.
    while (decodeAt(s, p)) |d| {
        if (!isLowerClass(d.cp)) break;
        p += d.len;
    }
    return p - start;
}

/// Previous UTF-8 codepoint boundary strictly less than `pos`, not below
/// `floor`.
fn prevBoundary(s: []const u8, floor: usize, pos: usize) usize {
    var q = pos;
    if (q <= floor) return floor;
    q -= 1;
    while (q > floor and (s[q] & 0xC0) == 0x80) q -= 1;
    return q;
}

/// Arm 3: `\p{N}` — exactly one numeral codepoint.
fn matchOneDigit(s: []const u8, i: usize) ?usize {
    const d = decodeAt(s, i) orelse return null;
    if (!isNumber(d.cp)) return null;
    return d.len;
}

/// Arm 4: ` ?[^\s\p{L}\p{N}]+[\r\n/]*`
/// Optional single leading ASCII space, then >=1 non-ws/letter/digit
/// codepoints, then a greedy run of `[\r\n/]`.
fn matchPunct(s: []const u8, i: usize) ?usize {
    var p = i;
    if (p < s.len and s[p] == ' ') p += 1;
    const body_start = p;
    while (decodeAt(s, p)) |d| {
        if (isWhitespace(d.cp) or isLetter(d.cp) or isNumber(d.cp)) break;
        p += d.len;
    }
    if (p == body_start) return null;
    // Trailing [\r\n/]* — ASCII only.
    while (p < s.len and (s[p] == '\r' or s[p] == '\n' or s[p] == '/')) p += 1;
    return p - i;
}

/// Arm 5: `\s*[\r\n]+`
fn matchWsNewline(s: []const u8, i: usize) ?usize {
    // Greedy \s*, then require a non-empty [\r\n]+ to remain. Mirror the
    // cl100k approach: walk \s* greedily, then shrink back to the last
    // position whose following char is \r or \n.
    var p = i;
    while (decodeAt(s, p)) |d| {
        if (!isWhitespace(d.cp)) break;
        p += d.len;
    }
    var nl_end = p;
    while (nl_end > i and (nl_end >= s.len or (s[nl_end] != '\r' and s[nl_end] != '\n'))) {
        nl_end -= 1;
    }
    if (nl_end >= s.len or (s[nl_end] != '\r' and s[nl_end] != '\n')) return null;
    var q = nl_end;
    while (q < s.len and (s[q] == '\r' or s[q] == '\n')) q += 1;
    return q - i;
}

/// Arm 6: `\s+(?!\S)` — a whitespace run that is not followed by a
/// non-space (i.e. trailing whitespace before EOF or before more space).
fn matchTrailingWs(s: []const u8, i: usize) ?usize {
    var p = i;
    var last_w: usize = 0;
    while (decodeAt(s, p)) |d| {
        if (!isWhitespace(d.cp)) break;
        last_w = d.len;
        p += d.len;
    }
    if (p == i) return null;
    if (p == s.len) return p - i; // run reaches EOF — lookahead satisfied
    // Next char is non-ws (\S); back off one codepoint so a ws sits right
    // after the match, satisfying (?!\S).
    if (p - last_w == i) return null;
    return (p - last_w) - i;
}

/// Arm 7: `\s+`
fn matchWs(s: []const u8, i: usize) ?usize {
    var p = i;
    while (decodeAt(s, p)) |d| {
        if (!isWhitespace(d.cp)) break;
        p += d.len;
    }
    if (p == i) return null;
    return p - i;
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

fn expectSplit(input: []const u8, expected: []const []const u8) !void {
    const spans = try split(std.testing.allocator, input);
    defer std.testing.allocator.free(spans);
    try std.testing.expectEqual(expected.len, spans.len);
    for (spans, expected) |sp, want| {
        try std.testing.expectEqualStrings(want, sp.slice(input));
    }
}

test "empty input" {
    const spans = try split(std.testing.allocator, "");
    defer std.testing.allocator.free(spans);
    try std.testing.expectEqual(@as(usize, 0), spans.len);
}

test "hello world" {
    try expectSplit("hello world", &.{ "hello", " world" });
}

test "camel case splits on uppercase" {
    try expectSplit("CamelCaseWord", &.{ "Camel", "Case", "Word" });
}

test "all caps stays together" {
    try expectSplit("HELLOworld", &.{"HELLOworld"});
    try expectSplit("ABC", &.{"ABC"});
}

test "digits split one per token" {
    try expectSplit("12345", &.{ "1", "2", "3", "4", "5" });
}

test "letters then digits then letters" {
    try expectSplit("hello123world", &.{ "hello", "1", "2", "3", "world" });
}

test "punct arm captures slash and newline" {
    try expectSplit("abc/def//", &.{ "abc", "/def", "//" });
}

test "unicode letters group as one word" {
    try expectSplit("résumé café", &.{ "résumé", " café" });
}

test "cjk run stays together" {
    try expectSplit("日本語テキスト", &.{"日本語テキスト"});
}

test "leading whitespace then newline then word" {
    try expectSplit("  \n  word", &.{ "  \n", " ", " word" });
}

test "trailing whitespace at EOF" {
    try expectSplit("hi   ", &.{ "hi", "   " });
}
