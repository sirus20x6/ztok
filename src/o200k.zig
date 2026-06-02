// o200k_base / o200k_harmony (GPT-OSS) pre-tokenizer splitter.
//
// Implements the o200k_base pat_str (also used by o200k_harmony):
//
//   [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//  |[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//  |\p{N}{1,3}
//  | ?[^\s\p{L}\p{N}]+[\r\n/]*
//  |\s*[\r\n]+
//  |\s+(?!\S)
//  |\s+
//
// Differences vs cl100k (see src/cl100k.zig):
//   * Letter runs are CASE-AWARE: two alternatives split an uppercase run
//     from a following lowercase run differently than cl100k's plain \p{L}+.
//     \p{M} (combining marks), \p{Lm} and \p{Lo} count as BOTH "upper-ish"
//     and "lower-ish" letters here.
//   * Contractions ('s 't 're 've 'm 'll 'd, case-insensitive) attach as a
//     SUFFIX of a letter run rather than as a standalone leading alternative.
//   * Punctuation run trailing class is [\r\n/]* (the '/' is added) vs
//     cl100k's [\r\n]*.
//   * \p{N}{1,3}, \s*[\r\n]+, \s+(?!\S), \s+ are identical to cl100k.
//
// Structure intentionally mirrors cl100k.zig (Decoded iterator, the per-
// alternative matchOne dispatch, the same whitespace eaters).

const std = @import("std");
const Span = @import("token.zig").Span;
const unicode_props = @import("unicode_props.zig");

/// Split input bytes into o200k spans. Caller owns the returned slice.
pub fn split(allocator: std.mem.Allocator, input: []const u8) ![]Span {
    var out = try allocator.alloc(Span, input.len);
    var n: usize = 0;
    var i: usize = 0;
    while (i < input.len) {
        const consumed = matchOne(input, i);
        // Defensive: matchOne always returns >=1 on non-empty input because
        // the punct/whitespace alternatives cover every byte.
        const took = if (consumed == 0) 1 else consumed;
        out[n] = .{ .start = @intCast(i), .end = @intCast(i + took) };
        n += 1;
        i += took;
    }
    return allocator.realloc(out, n);
}

// Try the o200k alternatives in order at position `i`. Return match length
// in bytes (0 means no alternative matched — caller advances by 1).
fn matchOne(s: []const u8, i: usize) usize {
    // Alts 1 & 2 are the two case-aware letter runs (with optional leading
    // non-letter and trailing contraction). matchLetters tries them in the
    // regex's documented order and returns the first non-empty match.
    if (matchLetters(s, i)) |len| return len;
    // 3: \p{N}{1,3}
    if (matchDigits(s, i)) |len| return len;
    // 4:  ?[^\s\p{L}\p{N}]+[\r\n/]*
    if (matchPunct(s, i)) |len| return len;
    // 5: \s*[\r\n]+
    if (matchWsNewline(s, i)) |len| return len;
    // 6: \s+(?!\S)
    if (matchTrailingWs(s, i)) |len| return len;
    // 7: \s+
    if (matchWs(s, i)) |len| return len;
    return 0;
}

// --- character classes for the two letter alternatives ---

// "Upper-ish" set: [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}].
fn isUpperClass(cp: u21) bool {
    if (cp < 0x80) return cp >= 'A' and cp <= 'Z';
    return unicode_props.isLu(cp) or unicode_props.isLt(cp) or
        unicode_props.isLm(cp) or unicode_props.isLo(cp) or
        unicode_props.isMark(cp);
}

// "Lower-ish" set: [\p{Ll}\p{Lm}\p{Lo}\p{M}].
fn isLowerClass(cp: u21) bool {
    if (cp < 0x80) return cp >= 'a' and cp <= 'z';
    return unicode_props.isLl(cp) or unicode_props.isLm(cp) or
        unicode_props.isLo(cp) or unicode_props.isMark(cp);
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

// Match the optional leading [^\r\n\p{L}\p{N}] at `p`. Returns the byte
// length consumed (0 if the optional is skipped). The optional is taken
// only when the codepoint is NOT \r, \n, a letter, or a digit AND there
// is a following letter to anchor the letter run (matching how a backtracking
// regex behaves: if the optional were consumed but no letter follows, the
// whole letter alternative fails, so the engine skips the optional).
//
// `require_first` is the predicate the first mandatory letter must satisfy
// (isLowerClass for alt 1's L+, isUpperClass for alt 2's U+); for the
// U*-then-L+ shape the optional is valid as long as SOME letter run can
// follow, which we approximate by "next codepoint is a letter-class member
// of the relevant set". We keep this simple and correct by only consuming
// the optional when the following codepoint is in the broad letter/mark set
// that both alternatives start their run with.
fn matchOptionalPrefix(s: []const u8, p: usize) usize {
    const d = decodeAt(s, p) orelse return 0;
    if (d.cp == '\r' or d.cp == '\n' or isLetter(d.cp) or isNumber(d.cp)) return 0;
    // Only take the optional if a letter-class codepoint follows (else the
    // letter run would be empty and the alternative cannot match).
    const d2 = decodeAt(s, p + d.len) orelse return 0;
    if (isUpperClass(d2.cp) or isLowerClass(d2.cp)) return d.len;
    return 0;
}

// Alts 1 & 2: the two case-aware letter runs.
//
// Alt 1: pfx? U* L+ contr?   (run must contain >=1 lower-ish letter)
// Alt 2: pfx? U+ L* contr?   (run must contain >=1 upper-ish letter)
//
// The regex tries alt 1 first. Because U* is greedy but backtracks to let
// L+ match, alt 1 effectively matches: optional prefix, then a maximal run
// of letters/marks that ENDS at a lower-ish letter (the last consumed
// codepoint must be lower-ish so L+ is satisfied). If no lower-ish letter
// exists in the run, alt 1 fails and alt 2 takes over: pfx? U+ L* — a
// maximal run of letters/marks starting with an upper-ish letter.
//
// We implement this directly:
//   * Consume the optional prefix.
//   * Scan the maximal contiguous run of codepoints that are in
//     (U ∪ L) = letters+marks. (U and L only differ on Lu/Lt vs Ll; their
//     union is exactly \p{L}\p{M}.) Record the end position of the LAST
//     lower-ish codepoint in that run.
//   * Alt 1 applies iff the run contains at least one lower-ish codepoint
//     AND the run, truncated to end at the last lower-ish codepoint, has
//     a non-empty L+ tail with the preceding part being U*. Since any
//     codepoint in the run is in U∪L and U*L+ can partition any such run
//     that ends in a lower-ish letter (every codepoint before the final
//     lower-ish one is in U if it's upper-ish, but a lower-ish-only run is
//     also fine because Ll∈? no — Ll is NOT in U). See note below.
//
// IMPORTANT subtlety: a lowercase letter 'a' (Ll) is in L but NOT in U.
// So for alt 1 `U* L+`, the U* part can ONLY contain upper-ish codepoints
// (Lu/Lt/Lm/Lo/M), and the L+ part contains lower-ish codepoints
// (Ll/Lm/Lo/M). A run like "aB" (Ll then Lu): U* matches empty, L+ must
// match "a" — but then "B" (Lu, not in L) stops L+. So "aB" matches alt 1
// as just "a", leaving "B" for the next iteration. A run like "Ba"
// (Lu then Ll): U* eats "B", L+ eats "a" → "Ba". A run "AB" (Lu Lu): no
// lower-ish letter → alt 1 fails; alt 2 `U+ L*` eats "AB".
//
// Therefore the correct algorithm is a left-to-right partition, NOT
// "maximal letter run then look for last lower-ish". We must find the
// LONGEST prefix of the letter run matching `U* L+` (for alt 1) where the
// boundary between U* and L+ is the first lower-ish (and-not-upper-ish)
// codepoint, and L+ continues only over codepoints in L. See matchAlt1 /
// matchAlt2 for the exact greedy/backtracking semantics.
fn matchLetters(s: []const u8, i: usize) ?usize {
    if (matchAlt1(s, i)) |len| return len;
    if (matchAlt2(s, i)) |len| return len;
    return null;
}

// Alt 1: pfx? U* L+ contr?
//
// U* is greedy-with-backtrack, L+ greedy. The leftmost-longest PCRE match is:
//   1. Consume the optional prefix.
//   2. Consume the MAXIMAL run of U-class codepoints (greedy U*).
//   3. The first codepoint NOT in U-class is, if it is a letter/mark at all,
//      necessarily \p{Ll} (the only letter class in L but not in U). If that
//      codepoint is L-class, it begins the mandatory L+; otherwise alt 1
//      cannot match (no L+ start) and we fall through to alt 2.
//   4. Consume the maximal run of L-class codepoints (greedy L+), which stops
//      at the first \p{Lu}/\p{Lt} or non-letter — i.e. the run is CUT right
//      before the next uppercase letter. This is the camelCase split.
//   5. Optional contraction.
//
// We deliberately do NOT model U*'s backtracking over ambiguous trailing
// codepoints (Lm/Lo/M, which are in both U and L). When the U-run ends in
// such an ambiguous codepoint and no pure-Ll follows, alt 1 would backtrack
// to expose an L+ start, producing the SAME total span that alt 2's `U+ L*`
// produces — so deferring to alt 2 yields an identical split. The only case
// where alt 1 vs alt 2 produce DIFFERENT spans is a pure-\p{Ll} boundary,
// which step 3 handles exactly.
fn matchAlt1(s: []const u8, i: usize) ?usize {
    var p = i;
    p += matchOptionalPrefix(s, p);

    // Step 2: maximal U-class run (greedy U*).
    while (decodeAt(s, p)) |d| {
        if (!isUpperClass(d.cp)) break;
        p += d.len;
    }

    // Step 3: the boundary codepoint must be L-class to start L+.
    const first = decodeAt(s, p) orelse return null;
    if (!isLowerClass(first.cp)) return null; // no L+ start → alt 1 fails

    // Step 4: maximal L-class run (greedy L+, >=1 by step 3).
    while (decodeAt(s, p)) |d| {
        if (!isLowerClass(d.cp)) break;
        p += d.len;
    }

    // Step 5: contr?
    p += matchContraction(s, p);
    return p - i;
}

// Alt 2: pfx? U+ L* contr?
//
// Requires at least one upper-ish codepoint, then a maximal lower-ish run.
// Greedy U+ then greedy L*: consume the maximal U-class prefix (>=1), then
// the maximal L-class run, then contr?.
fn matchAlt2(s: []const u8, i: usize) ?usize {
    var p = i;
    p += matchOptionalPrefix(s, p);

    // U+ : at least one upper-ish codepoint, greedily.
    var saw_upper = false;
    while (decodeAt(s, p)) |d| {
        if (!isUpperClass(d.cp)) break;
        p += d.len;
        saw_upper = true;
    }
    if (!saw_upper) return null;

    // L* : maximal lower-ish run.
    while (decodeAt(s, p)) |d| {
        if (!isLowerClass(d.cp)) break;
        p += d.len;
    }

    // contr?
    p += matchContraction(s, p);
    return p - i;
}

// (?i:'s|'t|'re|'ve|'m|'ll|'d) — optional, case-insensitive. Returns the
// byte length matched (0 if no contraction at `p`).
fn matchContraction(s: []const u8, p: usize) usize {
    if (p >= s.len or s[p] != '\'') return 0;
    if (p + 1 >= s.len) return 0;
    const c1 = s[p + 1] | 0x20; // ASCII tolower
    // two-letter forms first: 're, 've, 'll
    if (p + 2 < s.len) {
        const c2 = s[p + 2] | 0x20;
        if (c1 == 'r' and c2 == 'e') return 3;
        if (c1 == 'v' and c2 == 'e') return 3;
        if (c1 == 'l' and c2 == 'l') return 3;
    }
    if (c1 == 's' or c1 == 't' or c1 == 'm' or c1 == 'd') return 2;
    return 0;
}

// 3: \p{N}{1,3}
fn matchDigits(s: []const u8, i: usize) ?usize {
    var p = i;
    var count: usize = 0;
    while (count < 3) : (count += 1) {
        const d = decodeAt(s, p) orelse break;
        if (!isNumber(d.cp)) break;
        p += d.len;
    }
    if (p == i) return null;
    return p - i;
}

// 4:  ?[^\s\p{L}\p{N}]+[\r\n/]*
//
// Same as cl100k's matchPunct except the trailing class is [\r\n/]* (adds
// '/').
fn matchPunct(s: []const u8, i: usize) ?usize {
    var p = i;
    // optional leading single ASCII space
    const had_space = p < s.len and s[p] == ' ';
    if (had_space) p += 1;
    const body_start = p;
    while (decodeAt(s, p)) |d| {
        if (isWhitespace(d.cp) or isLetter(d.cp) or isNumber(d.cp)) break;
        p += d.len;
    }
    if (p == body_start) {
        // No body matched. If we consumed a leading space, this alternative
        // didn't actually match (the ` ?` requires the [^\s...]+ to follow);
        // back off so the space is handled by a whitespace alternative.
        return null;
    }
    // optional trailing [\r\n/]*
    while (p < s.len and (s[p] == '\r' or s[p] == '\n' or s[p] == '/')) p += 1;
    return p - i;
}

// 5: \s*[\r\n]+
fn matchWsNewline(s: []const u8, i: usize) ?usize {
    var p = i;
    // \s* greedy
    while (decodeAt(s, p)) |d| {
        if (!isWhitespace(d.cp)) break;
        p += d.len;
    }
    // require at least one [\r\n] in the run; back off so a \r\n remains.
    var nl_end = p;
    while (nl_end > i and (nl_end >= s.len or (s[nl_end] != '\r' and s[nl_end] != '\n'))) {
        nl_end -= 1;
    }
    if (nl_end >= s.len or (s[nl_end] != '\r' and s[nl_end] != '\n')) return null;
    var q = nl_end;
    while (q < s.len and (s[q] == '\r' or s[q] == '\n')) q += 1;
    return q - i;
}

// 6: \s+(?!\S)
fn matchTrailingWs(s: []const u8, i: usize) ?usize {
    var p = i;
    var last_w: usize = 0;
    while (decodeAt(s, p)) |d| {
        if (!isWhitespace(d.cp)) break;
        last_w = d.len;
        p += d.len;
    }
    if (p == i) return null;
    if (p == s.len) return p - i;
    if (p - last_w == i) return null; // backing off would empty the match
    return (p - last_w) - i;
}

// 7: \s+
fn matchWs(s: []const u8, i: usize) ?usize {
    var p = i;
    while (decodeAt(s, p)) |d| {
        if (!isWhitespace(d.cp)) break;
        p += d.len;
    }
    if (p == i) return null;
    return p - i;
}

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

// --- tests ---

fn expectSplit(input: []const u8, expected: []const []const u8) !void {
    const spans = try split(std.testing.allocator, input);
    defer std.testing.allocator.free(spans);
    try std.testing.expectEqual(expected.len, spans.len);
    for (spans, expected) |sp, want| {
        try std.testing.expectEqualStrings(want, sp.slice(input));
    }
}

test "o200k empty input" {
    const spans = try split(std.testing.allocator, "");
    defer std.testing.allocator.free(spans);
    try std.testing.expectEqual(@as(usize, 0), spans.len);
}

test "o200k hello world" {
    try expectSplit("hello world", &.{ "hello", " world" });
}

test "o200k case-aware: leading caps glued to lowercase" {
    // "Hello" — alt 1: U* eats "H", L+ eats "ello" → "Hello" stays together.
    try expectSplit("Hello", &.{"Hello"});
    // "HELLO" — no lower-ish letter → alt 2 U+ eats all → "HELLO".
    try expectSplit("HELLO", &.{"HELLO"});
    // "HELLOworld" — alt 1: U* greedily eats "HELLOworl"? no: U* eats the
    // uppercase prefix "HELLO", then L+ eats "world". Single token.
    try expectSplit("HELLOworld", &.{"HELLOworld"});
}

test "o200k case-aware: lowercase then uppercase splits" {
    // "aB": alt 1 U*(empty) L+("a"); "B" left over → ["a","B"].
    try expectSplit("aB", &.{ "a", "B" });
    // "fooBar": alt1 eats "foo" (U* empty, L+ "foo"); "B" starts next:
    // alt1 U*="B" L+="ar" → "Bar". So ["foo","Bar"].
    try expectSplit("fooBar", &.{ "foo", "Bar" });
}

test "o200k contractions attach as suffix" {
    try expectSplit("It's", &.{"It's"});
    try expectSplit("they'll", &.{"they'll"});
    // case-insensitive
    try expectSplit("IT'S", &.{"IT'S"});
}

test "o200k three digits chunked" {
    try expectSplit("123456789", &.{ "123", "456", "789" });
}

test "o200k punct slash tail" {
    // " ?[^\s\p{L}\p{N}]+[\r\n/]*" — '/' attaches as tail.
    try expectSplit("a//b", &.{ "a", "//", "b" });
}

test "o200k unicode letters: café and naïve" {
    try expectSplit("café", &.{"café"});
    // "naïve" — all lowercase letters (ï is Ll) → one token.
    try expectSplit("naïve", &.{"naïve"});
}

test "o200k leading non-letter optional prefix" {
    // " the" — alt1: pfx? eats " " (space, non-letter), then L+ "the".
    try expectSplit(" the", &.{" the"});
}
