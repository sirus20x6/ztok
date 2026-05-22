//! Avoid-pattern training support — refuses to mint vocab tokens whose
//! byte sequences match (contain) any avoid-pattern.
//!
//! Use case: security/safety. When training on potentially-injectable
//! inputs (SQL fragments, shell metacharacters, HTML tag opens) you may
//! want to prevent the trainer from creating tokens that shadow those
//! sequences. A single-token shadow of `<script>` can let a downstream
//! decoder emit the literal bytes via a benign-looking token id.
//!
//! Modes:
//!   * `.penalize` — subtract a large constant from candidate scores so
//!     the merge/seed never wins. The bytes are still representable via
//!     multiple smaller tokens (typically byte-fallback or shorter
//!     subwords); the trainer just won't pack them into one piece.
//!   * `.exclude` — outright skip the candidate. The bytes can still be
//!     produced by concatenating smaller tokens at encode time, so this
//!     is not a "ban these bytes from the output" guarantee — only "no
//!     single token in this vocab equals or contains the pattern".
//!
//! Matcher: a simple multi-pattern substring scanner (per-pattern
//! `std.mem.indexOfPos`). Aho-Corasick would be asymptotically better
//! for many patterns, but our candidate strings are short (BPE: ≤2 *
//! prior-merge-length; Unigram/Monster: ≤ max_piece_length, typically
//! 16-24) and avoid-lists are typically a handful of entries. The
//! O(P * L) per check dominates the lookup cost only when both P and L
//! grow large, which the spec doesn't anticipate. Worth revisiting if
//! avoid-lists ever grow into the thousands.
//!
//! Pattern format (v1):
//!   * One pattern per line.
//!   * Lines are interpreted as literal byte sequences. No escape
//!     sequences, no regex — those are explicit follow-ups.
//!   * Trailing `\r` (CRLF) is stripped so files saved on Windows work.
//!   * Blank lines are skipped.
//!   * Lines beginning with `#` are treated as comments and skipped, so
//!     authors can annotate avoid-lists without inventing a sidecar
//!     manifest. If you genuinely need a pattern starting with `#`,
//!     prefix it with a space (whitespace is otherwise significant).

const std = @import("std");

/// Score penalty applied to matching candidates in `.penalize` mode.
/// Tuned to drown out any realistic merge-frequency or marginal-value
/// score for vocab sizes up to ~10^6 and corpora up to ~10^9 tokens.
pub const PENALTY: i64 = -1_000_000;

pub const Mode = enum {
    /// Subtract `PENALTY` from the candidate's score. Still lets the
    /// bytes appear as a token in the rare case nothing else competes.
    penalize,
    /// Hard-skip the candidate. Guarantees no single token in the
    /// final vocab equals or contains the pattern.
    exclude,
};

/// In-memory avoid-pattern list. Owns its backing buffer.
pub const AvoidList = struct {
    allocator: std.mem.Allocator,
    /// Concatenated patterns; `slices` point into this buffer.
    buf: []u8,
    /// Borrowed views into `buf`. Stable for the lifetime of the list.
    slices: [][]const u8,

    pub fn deinit(self: *AvoidList) void {
        self.allocator.free(self.slices);
        self.allocator.free(self.buf);
        self.* = undefined;
    }

    /// Number of patterns.
    pub fn len(self: *const AvoidList) usize {
        return self.slices.len;
    }

    /// True if any pattern is a substring of `bytes`. Empty avoid lists
    /// always return false (no patterns to match against).
    pub fn matches(self: *const AvoidList, bytes: []const u8) bool {
        for (self.slices) |pat| {
            if (pat.len == 0) continue;
            if (pat.len > bytes.len) continue;
            if (std.mem.indexOf(u8, bytes, pat) != null) return true;
        }
        return false;
    }

    /// True if any pattern equals `bytes` exactly.
    /// Useful for excluding only single-token shadows while leaving
    /// longer-than-pattern compositions alone. Not used by the trainers
    /// today — they call `matches` — but exposed for future tuning.
    pub fn matchesExact(self: *const AvoidList, bytes: []const u8) bool {
        for (self.slices) |pat| {
            if (std.mem.eql(u8, pat, bytes)) return true;
        }
        return false;
    }
};

/// Parse an avoid-list text buffer. The buffer is copied into a fresh
/// allocation owned by the returned `AvoidList`, so the caller is free
/// to drop the input afterwards.
pub fn parse(allocator: std.mem.Allocator, text: []const u8) !AvoidList {
    // First pass: count patterns so we can size the slices array once.
    var pat_count: usize = 0;
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |raw_line| {
        const line = stripPattern(raw_line);
        if (line.len == 0) continue;
        if (line[0] == '#') continue;
        pat_count += 1;
    }

    // Second pass: copy bytes into a contiguous buffer and record slices.
    // Concatenating without separators is fine — `slices` carry exact
    // bounds.
    var total_bytes: usize = 0;
    it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |raw_line| {
        const line = stripPattern(raw_line);
        if (line.len == 0) continue;
        if (line[0] == '#') continue;
        total_bytes += line.len;
    }

    const buf = try allocator.alloc(u8, total_bytes);
    errdefer allocator.free(buf);
    const slices = try allocator.alloc([]const u8, pat_count);
    errdefer allocator.free(slices);

    var cursor: usize = 0;
    var idx: usize = 0;
    it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |raw_line| {
        const line = stripPattern(raw_line);
        if (line.len == 0) continue;
        if (line[0] == '#') continue;
        @memcpy(buf[cursor .. cursor + line.len], line);
        slices[idx] = buf[cursor .. cursor + line.len];
        cursor += line.len;
        idx += 1;
    }

    return .{
        .allocator = allocator,
        .buf = buf,
        .slices = slices,
    };
}

/// Load an avoid-list from disk via `std.fs`. The CLI uses the newer
/// `std.Io.Dir.cwd().readFileAlloc` path; this helper is here so unit
/// tests (and callers without an `std.Io` in hand) can load from a
/// path without ceremony.
pub fn loadFromPath(allocator: std.mem.Allocator, path: []const u8) !AvoidList {
    var f = try std.fs.cwd().openFile(path, .{});
    defer f.close();
    const stat = try f.stat();
    if (stat.size > 1 << 24) return error.AvoidFileTooLarge; // 16 MiB sanity cap
    const buf = try allocator.alloc(u8, @intCast(stat.size));
    defer allocator.free(buf);
    const n = try f.readAll(buf);
    return parse(allocator, buf[0..n]);
}

/// Build an `AvoidList` directly from an in-memory slice of patterns.
/// Convenience for callers that already have the patterns in hand
/// (tests, programmatic users). Empty patterns are skipped.
pub fn fromPatterns(allocator: std.mem.Allocator, patterns: []const []const u8) !AvoidList {
    var total_bytes: usize = 0;
    var keep: usize = 0;
    for (patterns) |p| {
        if (p.len == 0) continue;
        total_bytes += p.len;
        keep += 1;
    }
    const buf = try allocator.alloc(u8, total_bytes);
    errdefer allocator.free(buf);
    const slices = try allocator.alloc([]const u8, keep);
    errdefer allocator.free(slices);
    var cursor: usize = 0;
    var idx: usize = 0;
    for (patterns) |p| {
        if (p.len == 0) continue;
        @memcpy(buf[cursor .. cursor + p.len], p);
        slices[idx] = buf[cursor .. cursor + p.len];
        cursor += p.len;
        idx += 1;
    }
    return .{
        .allocator = allocator,
        .buf = buf,
        .slices = slices,
    };
}

/// Strip a trailing '\r' (CRLF tolerance) and surrounding nothing else.
/// We deliberately do NOT trim whitespace — leading/trailing spaces in
/// a pattern are byte-significant.
inline fn stripPattern(line: []const u8) []const u8 {
    if (line.len > 0 and line[line.len - 1] == '\r') return line[0 .. line.len - 1];
    return line;
}

// --- tests ----------------------------------------------------------------

const testing = std.testing;

test "parses lines, ignores comments and blanks, handles CRLF" {
    const text =
        "DROP TABLE\r\n" ++
        "<script>\n" ++
        "\n" ++
        "# a comment\n" ++
        "  leading space matters\n" ++
        "trailing space matters \n";

    var al = try parse(testing.allocator, text);
    defer al.deinit();

    try testing.expectEqual(@as(usize, 4), al.len());
    try testing.expectEqualStrings("DROP TABLE", al.slices[0]);
    try testing.expectEqualStrings("<script>", al.slices[1]);
    try testing.expectEqualStrings("  leading space matters", al.slices[2]);
    try testing.expectEqualStrings("trailing space matters ", al.slices[3]);
}

test "matcher: substring vs exact on hand-rolled fixture" {
    const patterns = [_][]const u8{ "DROP TABLE", "<script>", "rm -rf" };
    var al = try fromPatterns(testing.allocator, &patterns);
    defer al.deinit();

    // Exact and substring positives.
    try testing.expect(al.matches("DROP TABLE"));
    try testing.expect(al.matches("xxDROP TABLEyy"));
    try testing.expect(al.matches("<script>"));
    try testing.expect(al.matches("hello <script>world"));
    try testing.expect(al.matches("rm -rf /"));

    // Negatives — must not match unrelated strings.
    try testing.expect(!al.matches("DROP"));
    try testing.expect(!al.matches("DROP_TABLE")); // underscore differs
    try testing.expect(!al.matches("<scrip"));
    try testing.expect(!al.matches("rm-rf"));
    try testing.expect(!al.matches(""));

    // matchesExact is strictly equality.
    try testing.expect(al.matchesExact("DROP TABLE"));
    try testing.expect(!al.matchesExact("xxDROP TABLEyy"));
    try testing.expect(!al.matchesExact("DROP"));
}

test "empty avoid list never matches" {
    var al = try fromPatterns(testing.allocator, &.{});
    defer al.deinit();
    try testing.expectEqual(@as(usize, 0), al.len());
    try testing.expect(!al.matches("anything"));
    try testing.expect(!al.matches(""));
}

test "BPE trainer respects exclude mode: vocab has no single-token <script>" {
    const allocator = testing.allocator;
    const train_bpe = @import("train_bpe.zig");

    // Corpus designed to make `<script>` a tempting merge: the literal
    // byte sequence appears repeatedly with high frequency, alongside
    // some unrelated padding so the trainer has other choices too.
    const words = [_][]const u8{ "<script>", "<script>hello", "world", "<scriptx>" };
    const counts = [_]u32{ 50, 30, 20, 10 };

    var al = try fromPatterns(allocator, &.{"<script>"});
    defer al.deinit();

    var bpe = try train_bpe.train(allocator, .{
        .words = &words,
        .counts = &counts,
    }, .{
        .vocab_size = 320,
        .avoid = &al,
        .avoid_mode = .exclude,
    });
    defer bpe.deinit();

    // No single token equals `<script>` exactly.
    var id: u32 = 0;
    while (id < bpe.count) : (id += 1) {
        const piece = bpe.idBytes(id);
        try testing.expect(!std.mem.eql(u8, piece, "<script>"));
        // And no token contains it as a substring either — exclude mode
        // forbids any super-pattern token too.
        try testing.expect(std.mem.indexOf(u8, piece, "<script>") == null);
    }

    // Sanity: the bytes are still representable (each byte is in the
    // base vocab as a single-byte token).
    var byte: u32 = 0;
    while (byte < 256) : (byte += 1) {
        const c: u8 = @intCast(byte);
        const buf = [_]u8{c};
        const piece = bpe.idBytes(byte);
        try testing.expectEqualSlices(u8, &buf, piece);
    }
}

test "BPE trainer respects penalize mode: same guarantee, demoted score" {
    const allocator = testing.allocator;
    const train_bpe = @import("train_bpe.zig");

    const words = [_][]const u8{ "DROP TABLE", "DROP TABLE users", "SELECT", "FROM" };
    const counts = [_]u32{ 100, 80, 40, 30 };

    var al = try fromPatterns(allocator, &.{"DROP TABLE"});
    defer al.deinit();

    var bpe = try train_bpe.train(allocator, .{
        .words = &words,
        .counts = &counts,
    }, .{
        .vocab_size = 320,
        .avoid = &al,
        .avoid_mode = .penalize,
    });
    defer bpe.deinit();

    // Penalize is a strong demotion; even with very high frequency the
    // candidate cannot win against any unrelated pair with non-zero
    // count. The vocab MUST NOT contain `DROP TABLE` as a single token.
    var id: u32 = 0;
    var found_exact: bool = false;
    while (id < bpe.count) : (id += 1) {
        if (std.mem.eql(u8, bpe.idBytes(id), "DROP TABLE")) found_exact = true;
    }
    try testing.expect(!found_exact);
}
