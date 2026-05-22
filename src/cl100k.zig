// Real Unicode \p{L}\p{N}\s via unicode_props. ASCII fast-path inline
// before the binary-search lookup since the cl100k hot loop is dominated
// by ASCII codepoints.

const std = @import("std");
const Span = @import("token.zig").Span;
const simd_bytes = @import("simd_bytes.zig");

/// True if cutting `input` immediately before position `pos` produces two
/// halves whose concatenated cl100k pre-tokenization is bit-identical to
/// running cl100k on `input` in one shot.
///
/// All cl100k patterns are anchored prefix matchers with no lookbehind,
/// so the only divergence risk is a single-shot match that SPANS `pos`.
/// The only patterns that can span a position are the whitespace eaters
/// (5, 6, 7) and pattern 4's trailing `[\r\n]*`. Both terminate exactly
/// at the end of a `[\r\n]` run.
///
/// Therefore `pos` is a safe cut iff:
///   * `pos == 0` (trivially), or
///   * the byte at `pos - 1` is `\n`, AND
///   * either `pos == input.len`, OR the byte at `pos` is NOT whitespace.
///
/// The "non-whitespace next" guard prevents cutting inside a whitespace
/// run that pattern 5 (`\s*[\r\n]+`) would have absorbed across the
/// boundary in single-shot mode (e.g. `foo\n   \nbar`).
pub fn isSafeCut(input: []const u8, pos: usize) bool {
    if (pos == 0 or pos == input.len) return true;
    if (pos > input.len) return false;
    if (input[pos - 1] != '\n') return false;
    const c = input[pos];
    // ASCII whitespace check first — covers \r, \t, ' ', \n, \v, \f.
    if (c == '\r' or c == '\n' or c == ' ' or c == '\t' or c == 0x0B or c == 0x0C) return false;
    // Non-ASCII: a leading 0x80..0xBF byte means we're inside a multibyte
    // sequence (not a codepoint boundary). Reject. For valid UTF-8 leading
    // bytes, decode and check Unicode whitespace.
    if (c >= 0x80) {
        if (c < 0xC0) return false; // continuation byte — mid-codepoint
        const d = decodeAt(input, pos) orelse return false;
        if (isWhitespace(d.cp)) return false;
    }
    return true;
}

/// Search for a safe cl100k cut position near `desired`. Scans backwards
/// first (up to `window` bytes) for the nearest safe position at or
/// before `desired`; if none is found, scans forward. Returns null only
/// if neither direction yields a safe cut within `window` bytes.
///
/// `desired` is the byte position the caller WANTS to cut at; the result
/// is the position to actually cut at (always within
/// `[desired - window, desired + window]`).
pub fn findSafeCut(input: []const u8, desired: usize, window: usize) ?usize {
    if (isSafeCut(input, desired)) return desired;

    // Scan backwards: prefer cuts earlier than `desired` so the left
    // half gets at most the requested size.
    var back: usize = 1;
    while (back <= window and back <= desired) : (back += 1) {
        const p = desired - back;
        if (isSafeCut(input, p)) return p;
    }

    // Fall back to scanning forward.
    var fwd: usize = 1;
    while (fwd <= window and desired + fwd <= input.len) : (fwd += 1) {
        const p = desired + fwd;
        if (isSafeCut(input, p)) return p;
    }

    return null;
}

/// Split input bytes into cl100k_base spans. Caller owns the returned slice.
pub fn split(allocator: std.mem.Allocator, input: []const u8) ![]Span {
    var out = try allocator.alloc(Span, input.len);
    var n: usize = 0;
    var i: usize = 0;
    while (i < input.len) {
        const consumed = matchOne(input, i);
        // Defensive: matchOne always returns >=1 on non-empty input because
        // pattern 4 (and patterns 5..7 for whitespace) cover every byte.
        const took = if (consumed == 0) 1 else consumed;
        out[n] = .{ .start = @intCast(i), .end = @intCast(i + took) };
        n += 1;
        i += took;
    }
    return allocator.realloc(out, n);
}

// Try the 7 cl100k alternatives in order at position `i`. Return match length in bytes.
fn matchOne(s: []const u8, i: usize) usize {
    // 1: contractions (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if (matchContraction(s, i)) |len| return len;
    // 2: [^\r\n\p{L}\p{N}]? \p{L}+
    if (matchWord(s, i)) |len| return len;
    // 3: \p{N}{1,3}
    if (matchDigits(s, i)) |len| return len;
    // 4:  ?[^\s\p{L}\p{N}]+[\r\n]*
    if (matchPunct(s, i)) |len| return len;
    // 5: \s*[\r\n]+
    if (matchWsNewline(s, i)) |len| return len;
    // 6: \s+(?!\S)
    if (matchTrailingWs(s, i)) |len| return len;
    // 7: \s+
    if (matchWs(s, i)) |len| return len;
    return 0;
}

fn matchContraction(s: []const u8, i: usize) ?usize {
    if (i >= s.len or s[i] != '\'') return null;
    if (i + 1 >= s.len) return null;
    const c1 = s[i + 1] | 0x20; // ASCII tolower
    // two-letter forms first: 're, 've, 'll
    if (i + 2 < s.len) {
        const c2 = s[i + 2] | 0x20;
        if (c1 == 'r' and c2 == 'e') return 3;
        if (c1 == 'v' and c2 == 'e') return 3;
        if (c1 == 'l' and c2 == 'l') return 3;
    }
    if (c1 == 's' or c1 == 't' or c1 == 'm' or c1 == 'd') return 2;
    return null;
}

fn matchWord(s: []const u8, i: usize) ?usize {
    var p = i;
    // optional one codepoint that is not \r, \n, letter, or digit
    if (decodeAt(s, p)) |d| {
        if (d.cp != '\r' and d.cp != '\n' and !isLetter(d.cp) and !isNumber(d.cp)) {
            // only take it if the following codepoint is a letter (else the
            // optional should be skipped and the letter run starts here)
            if (decodeAt(s, p + d.len)) |d2| {
                if (isLetter(d2.cp)) p += d.len;
            }
        }
    }
    const letters_start = p;
    // SIMD fast-path: vectorise the ASCII letter run. The scanner
    // stops at the first non-ASCII-letter byte (which may be a real
    // boundary OR a UTF-8 leading byte we need the scalar decoder
    // for).
    while (true) {
        const ascii_n = simd_bytes.scanAsciiLetter(s[p..]);
        p += ascii_n;
        // After the ASCII run, either we ran off the end or we hit a
        // non-ASCII / non-letter byte. Try one scalar codepoint:
        // if it's a Unicode letter, consume it and re-enter SIMD;
        // otherwise we're done.
        const d = decodeAt(s, p) orelse break;
        if (d.len == 1) break; // ASCII non-letter — terminal
        if (!isLetter(d.cp)) break;
        p += d.len;
    }
    if (p == letters_start) return null;
    return p - i;
}

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

fn matchPunct(s: []const u8, i: usize) ?usize {
    var p = i;
    // optional leading single ASCII space
    const had_space = p < s.len and s[p] == ' ';
    if (had_space) p += 1;
    const body_start = p;
    // SIMD fast-path: scan a run of ASCII non-ws/letter/digit bytes.
    // On hitting any non-ASCII byte, drop into the scalar decoder to
    // check whether the codepoint is body-class under Unicode rules.
    while (true) {
        const ascii_n = simd_bytes.scanAsciiPunct(s[p..]);
        p += ascii_n;
        const d = decodeAt(s, p) orelse break;
        if (d.len == 1) break; // ASCII byte that failed the punct test — terminal
        if (isWhitespace(d.cp) or isLetter(d.cp) or isNumber(d.cp)) break;
        p += d.len;
    }
    if (p == body_start) return null;
    // optional trailing [\r\n]*
    while (p < s.len and (s[p] == '\r' or s[p] == '\n')) p += 1;
    return p - i;
}

fn matchWsNewline(s: []const u8, i: usize) ?usize {
    var p = i;
    // \s* greedy — SIMD-fast over ASCII ws prefix, scalar for non-ASCII tail.
    while (true) {
        p += simd_bytes.scanAsciiWs(s[p..]);
        const d = decodeAt(s, p) orelse break;
        if (d.len == 1) break;
        if (!isWhitespace(d.cp)) break;
        p += d.len;
    }
    // require at least one [\r\n] in the run; back off if needed so a \r\n
    // remains in the suffix
    var nl_end = p;
    // walk backwards to find the last position where the next char is \r or \n
    // simpler: scan from i, greedy \s* then require [\r\n]+
    // Implement by: find the rightmost prefix of [i..p] whose suffix is all
    // whitespace and the next char (at the boundary) is \r or \n.
    // Practical impl: shrink p back while the byte just past us isn't \r\n,
    // until we find one.
    while (nl_end > i and (nl_end >= s.len or (s[nl_end] != '\r' and s[nl_end] != '\n'))) {
        nl_end -= 1;
    }
    if (nl_end >= s.len or (s[nl_end] != '\r' and s[nl_end] != '\n')) return null;
    var q = nl_end;
    while (q < s.len and (s[q] == '\r' or s[q] == '\n')) q += 1;
    return q - i;
}

fn matchTrailingWs(s: []const u8, i: usize) ?usize {
    // Greedily match \s+, recording the last codepoint width so we can
    // back off if needed to satisfy (?!\S). SIMD-fast over ASCII ws,
    // scalar for non-ASCII transitions.
    var p = i;
    var last_w: usize = 0;
    while (true) {
        const ascii_n = simd_bytes.scanAsciiWs(s[p..]);
        if (ascii_n > 0) {
            last_w = 1;
            p += ascii_n;
        }
        const d = decodeAt(s, p) orelse break;
        if (d.len == 1) break;
        if (!isWhitespace(d.cp)) break;
        last_w = d.len;
        p += d.len;
    }
    if (p == i) return null;
    // (?!\S) — the char right after the match must NOT be \S. That means
    // either end-of-input, or another whitespace. Since the greedy loop
    // already ate every whitespace, if we're NOT at EOF we're at \S — so
    // back off one codepoint, leaving the previous whitespace right after
    // the match (which satisfies the lookahead).
    if (p == s.len) return p - i;
    if (p - last_w == i) return null; // backing off would empty the match
    return (p - last_w) - i;
}

fn matchWs(s: []const u8, i: usize) ?usize {
    var p = i;
    while (true) {
        p += simd_bytes.scanAsciiWs(s[p..]);
        const d = decodeAt(s, p) orelse break;
        if (d.len == 1) break;
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

const unicode_props = @import("unicode_props.zig");

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

// --- tests ---

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

test "contractions" {
    try expectSplit("It's a test", &.{ "It", "'s", " a", " test" });
}

test "three digits" {
    try expectSplit("123", &.{"123"});
}

test "digits chunked by 3" {
    try expectSplit("12345", &.{ "123", "45" });
}

test "leading whitespace then word" {
    // Pattern 6 `\s+(?!\S)` backs off one codepoint to satisfy the
    // negative lookahead, so 3 spaces split as ["  ", " trailing"]:
    // alt 6 takes 2 chars (last char is ws → lookahead OK), then alt 2's
    // ` ?\p{L}+` eats the remaining space + word.
    try expectSplit("   trailing", &.{ "  ", " trailing" });
}

test "double newline between words" {
    try expectSplit("hello\n\nworld", &.{ "hello", "\n\n", "world" });
}

test "contractions case-insensitive" {
    try expectSplit("IT'S", &.{ "IT", "'S" });
    try expectSplit("they'LL", &.{ "they", "'LL" });
}

test "punctuation with leading space" {
    try expectSplit("hi !!", &.{ "hi", " !!" });
}

test "trailing whitespace at EOF" {
    try expectSplit("hi   ", &.{ "hi", "   " });
}

test "ws-then-newline-then-ws-then-word" {
    // pattern 5 backtracks \s* so the trailing \r/\n run is non-empty;
    // remaining "  word": alt 6 \s+(?!\S) backs off one codepoint to
    // leave a trailing ws, then alt 2 takes ` ?\p{L}+`.
    try expectSplit("  \n  word", &.{ "  \n", " ", " word" });
}

test "single non-ascii byte falls into pattern 4" {
    // "é" is now a real Unicode letter (cat Ll), so pattern 2 matches it
    // as a one-codepoint word. Pre-Unicode-tables this went through
    // pattern 4 (punct catch-all) and produced the same single-span
    // output by coincidence — keeping the test pins both behaviors.
    try expectSplit("é", &.{"é"});
}

test "isSafeCut edges and obvious cases" {
    const s = "hello\nworld";
    try std.testing.expect(isSafeCut(s, 0));
    try std.testing.expect(isSafeCut(s, s.len));
    // pos 6 = right after `\n`, before `w`. Safe.
    try std.testing.expect(isSafeCut(s, 6));
    // mid-word: previous char isn't `\n` → unsafe.
    try std.testing.expect(!isSafeCut(s, 3));
    // The `\n` itself is at pos 5; cutting at 5 means previous char is `o` → unsafe.
    try std.testing.expect(!isSafeCut(s, 5));
}

test "isSafeCut rejects mid-whitespace runs after newline" {
    // `foo\n   bar` — cutting at pos 4 (after \n, before space) would be
    // unsafe because single-shot pattern 5 might absorb whitespace
    // following the newline across the boundary.
    const s = "foo\n   bar";
    try std.testing.expect(!isSafeCut(s, 4));
    // pos 7 = after the 3 spaces, before `b`. Previous char ' ' not `\n` → unsafe.
    try std.testing.expect(!isSafeCut(s, 7));
}

test "isSafeCut rejects between consecutive newlines" {
    const s = "a\n\nb";
    // pos 2: prev=\n, cur=\n → unsafe (would split the newline run).
    try std.testing.expect(!isSafeCut(s, 2));
    // pos 3: prev=\n, cur=`b` → safe.
    try std.testing.expect(isSafeCut(s, 3));
}

test "isSafeCut handles UTF-8 continuation bytes" {
    // "a\nĉ" — `ĉ` is U+0109 = 0xC4 0x89 (two bytes).
    const s = "a\n\xC4\x89";
    // pos 2 = after `\n`, before `\xC4` (leading byte of `ĉ`). Safe.
    try std.testing.expect(isSafeCut(s, 2));
    // pos 3 = inside the multibyte sequence. Unsafe.
    try std.testing.expect(!isSafeCut(s, 3));
}

test "findSafeCut scans backwards then forwards" {
    const s = "abc\ndef\nghi";
    // Desired position 6 (mid-`def`). Backwards search hits pos 4 (after first `\n`).
    try std.testing.expectEqual(@as(?usize, 4), findSafeCut(s, 6, 16));
    // Desired position 0 is always safe.
    try std.testing.expectEqual(@as(?usize, 0), findSafeCut(s, 0, 16));
    // Tiny window may force forward scan.
    try std.testing.expectEqual(@as(?usize, 4), findSafeCut(s, 5, 1));
}

test "findSafeCut returns null when no safe boundary in window" {
    const s = "abcdefghij";
    // No newlines at all → only safe cuts are 0 and len. With desired=5 and
    // window=2, neither is reachable.
    try std.testing.expectEqual(@as(?usize, null), findSafeCut(s, 5, 2));
}

test "Unicode letters group as one word (real-tables distinguisher)" {
    // ASCII-only classifiers would split "café" into ["caf", "é"] since
    // they treat 'é' as a non-letter, but real Unicode classifiers see
    // 'é' as Ll and group the whole word under pattern 2.
    try expectSplit("café", &.{"café"});
    // Mixed-script Greek/Latin word stays together too.
    try expectSplit("αβγ", &.{"αβγ"});
    // Arabic-Indic digits cluster under pattern 3 (\p{N}{1,3}).
    try expectSplit("٠١٢", &.{"٠١٢"});
}
