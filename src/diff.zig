//! Tokenization diff: encode the same corpus with two pipelines and
//! report where they disagree. Useful for evaluating whether a vocab
//! swap actually changes the output, and by how much.
//!
//! Algorithm is line-oriented and O(corpus.bytes) with two encoder
//! passes. v1 is sequential; batching across the two pipelines could
//! help in v2 but is out of scope here.

const std = @import("std");

const TokenId = @import("token.zig").TokenId;
const Pipeline = @import("pipeline.zig").Pipeline;

pub const Side = enum { a, b };

pub const Summary = struct {
    corpus_bytes: u64,
    tokens_a: u64,
    tokens_b: u64,
    bytes_per_token_a: f64,
    bytes_per_token_b: f64,
    divergent_lines: u32,
    total_lines: u32,
    reduction_pct: f64,
};

pub const LineDiff = struct {
    line_number: u32,
    line: []const u8,
    tokens_a: []TokenId,
    tokens_b: []TokenId,

    pub fn deinit(self: *LineDiff, allocator: std.mem.Allocator) void {
        allocator.free(self.tokens_a);
        allocator.free(self.tokens_b);
        self.tokens_a = &.{};
        self.tokens_b = &.{};
    }
};

pub const DiffOptions = struct {
    max_detailed_diffs: u32 = 100,
    min_line_bytes: u32 = 1,
};

pub const DiffReport = struct {
    allocator: std.mem.Allocator,
    summary: Summary,
    diffs: []LineDiff,

    pub fn deinit(self: *DiffReport) void {
        for (self.diffs) |*d| d.deinit(self.allocator);
        if (self.diffs.len > 0) self.allocator.free(self.diffs);
        self.diffs = &.{};
    }
};

/// Split `corpus` by '\n'. The final segment is included even when no
/// trailing newline is present, matching the intuition that a corpus
/// like "a\nb" has two lines.
pub fn diff(
    allocator: std.mem.Allocator,
    pipeline_a: *const Pipeline,
    pipeline_b: *const Pipeline,
    corpus: []const u8,
    opts: DiffOptions,
) !DiffReport {
    var diffs: std.ArrayList(LineDiff) = .empty;
    errdefer {
        for (diffs.items) |*d| d.deinit(allocator);
        diffs.deinit(allocator);
    }

    var tokens_a: u64 = 0;
    var tokens_b: u64 = 0;
    var divergent: u32 = 0;
    var total_lines: u32 = 0;

    var line_no: u32 = 0;
    var it = std.mem.splitScalar(u8, corpus, '\n');
    while (it.next()) |line| {
        line_no += 1;
        total_lines += 1;
        if (line.len < opts.min_line_bytes) continue;

        const ids_a = try pipeline_a.encode(allocator, line);
        var keep_a = false;
        defer if (!keep_a) allocator.free(ids_a);

        const ids_b = try pipeline_b.encode(allocator, line);
        var keep_b = false;
        defer if (!keep_b) allocator.free(ids_b);

        tokens_a += ids_a.len;
        tokens_b += ids_b.len;

        const same = std.mem.eql(TokenId, ids_a, ids_b);
        if (!same) {
            divergent += 1;
            if (diffs.items.len < opts.max_detailed_diffs) {
                try diffs.append(allocator, .{
                    .line_number = line_no,
                    .line = line,
                    .tokens_a = ids_a,
                    .tokens_b = ids_b,
                });
                keep_a = true;
                keep_b = true;
            }
        }
    }

    const bpt_a: f64 = if (tokens_a == 0) 0 else @as(f64, @floatFromInt(corpus.len)) / @as(f64, @floatFromInt(tokens_a));
    const bpt_b: f64 = if (tokens_b == 0) 0 else @as(f64, @floatFromInt(corpus.len)) / @as(f64, @floatFromInt(tokens_b));
    const reduction: f64 = if (tokens_a == 0)
        0
    else
        (@as(f64, @floatFromInt(tokens_a)) - @as(f64, @floatFromInt(tokens_b))) * 100.0 / @as(f64, @floatFromInt(tokens_a));

    const owned = try diffs.toOwnedSlice(allocator);

    return .{
        .allocator = allocator,
        .summary = .{
            .corpus_bytes = corpus.len,
            .tokens_a = tokens_a,
            .tokens_b = tokens_b,
            .bytes_per_token_a = bpt_a,
            .bytes_per_token_b = bpt_b,
            .divergent_lines = divergent,
            .total_lines = total_lines,
            .reduction_pct = reduction,
        },
        .diffs = owned,
    };
}

pub fn formatText(allocator: std.mem.Allocator, report: *const DiffReport) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);

    const s = report.summary;
    try buf.appendSlice(allocator, "ztok diff summary\n");
    try buf.appendSlice(allocator, "=================\n");
    try buf.print(allocator, "corpus:           {d} bytes / {d} lines\n", .{ s.corpus_bytes, s.total_lines });
    try buf.print(allocator, "tokens A:         {d}  (bytes/token {d:.2})\n", .{ s.tokens_a, s.bytes_per_token_a });
    try buf.print(allocator, "tokens B:         {d}  (bytes/token {d:.2})\n", .{ s.tokens_b, s.bytes_per_token_b });
    try buf.print(allocator, "reduction:        {d:.2}% (A -> B)\n", .{s.reduction_pct});

    const denom: f64 = if (s.total_lines == 0) 1 else @floatFromInt(s.total_lines);
    const div_pct: f64 = @as(f64, @floatFromInt(s.divergent_lines)) * 100.0 / denom;
    try buf.print(allocator, "divergent lines:  {d} / {d} ({d:.2}%)\n", .{ s.divergent_lines, s.total_lines, div_pct });

    if (report.diffs.len == 0) {
        try buf.appendSlice(allocator, "\nno divergent lines captured.\n");
        return buf.toOwnedSlice(allocator);
    }

    try buf.print(allocator, "\nfirst {d} divergent lines:\n", .{report.diffs.len});
    try buf.appendSlice(allocator, "-----------------------\n");

    for (report.diffs) |d| {
        try buf.print(allocator, "line {d:>4}: \"", .{d.line_number});
        try appendEscaped(allocator, &buf, d.line);
        try buf.appendSlice(allocator, "\"\n");

        try buf.appendSlice(allocator, "  A: [");
        try appendIds(allocator, &buf, d.tokens_a);
        try buf.print(allocator, "]   ({d} tokens)\n", .{d.tokens_a.len});

        try buf.appendSlice(allocator, "  B: [");
        try appendIds(allocator, &buf, d.tokens_b);
        try buf.print(allocator, "]   ({d} tokens)\n", .{d.tokens_b.len});
    }

    return buf.toOwnedSlice(allocator);
}

fn appendIds(allocator: std.mem.Allocator, buf: *std.ArrayList(u8), ids: []const TokenId) !void {
    for (ids, 0..) |id, i| {
        if (i != 0) try buf.appendSlice(allocator, ", ");
        try buf.print(allocator, "{d}", .{id});
    }
}

fn appendEscaped(allocator: std.mem.Allocator, buf: *std.ArrayList(u8), line: []const u8) !void {
    for (line) |c| switch (c) {
        '\n' => try buf.appendSlice(allocator, "\\n"),
        '\r' => try buf.appendSlice(allocator, "\\r"),
        '\t' => try buf.appendSlice(allocator, "\\t"),
        '"' => try buf.appendSlice(allocator, "\\\""),
        '\\' => try buf.appendSlice(allocator, "\\\\"),
        else => try buf.append(allocator, c),
    };
}

// --- markdown report ----------------------------------------------------

/// Maximum Levenshtein distance we'll compute precisely. Anything larger
/// is bucketed as "large" without running the full DP. Keeps a noisy diff
/// line from blowing up to O(n*m) on huge id streams.
pub const LEVENSHTEIN_CAP: usize = 50;

/// Bounded Levenshtein distance on token-id streams.
///
/// Returns the true distance when it's `<= cap`; returns `cap + 1` as a
/// sentinel ("greater than cap") otherwise. If either side already
/// exceeds `cap` purely by length, we short-circuit without allocating.
pub fn levenshteinIds(
    allocator: std.mem.Allocator,
    a: []const TokenId,
    b: []const TokenId,
    cap: usize,
) !usize {
    // Fast cases.
    if (a.len == 0) return @min(b.len, cap + 1);
    if (b.len == 0) return @min(a.len, cap + 1);
    // Length difference is a lower bound on edit distance.
    const len_diff: usize = if (a.len > b.len) a.len - b.len else b.len - a.len;
    if (len_diff > cap) return cap + 1;

    // Make `b` the shorter side so the DP row is the smaller one.
    const x = if (a.len <= b.len) a else b;
    const y = if (a.len <= b.len) b else a;

    // Two-row DP: prev[j] = D(i-1, j), curr[j] = D(i, j). x along columns,
    // y along rows. Length: x.len + 1.
    var prev = try allocator.alloc(usize, x.len + 1);
    defer allocator.free(prev);
    var curr = try allocator.alloc(usize, x.len + 1);
    defer allocator.free(curr);

    var j: usize = 0;
    while (j <= x.len) : (j += 1) prev[j] = j;

    var i: usize = 1;
    while (i <= y.len) : (i += 1) {
        curr[0] = i;
        var row_min: usize = curr[0];
        var jj: usize = 1;
        while (jj <= x.len) : (jj += 1) {
            const cost: usize = if (y[i - 1] == x[jj - 1]) 0 else 1;
            const del = prev[jj] + 1;
            const ins = curr[jj - 1] + 1;
            const sub = prev[jj - 1] + cost;
            var best = del;
            if (ins < best) best = ins;
            if (sub < best) best = sub;
            curr[jj] = best;
            if (best < row_min) row_min = best;
        }
        // Early-exit: if every cell of this row is already > cap, the
        // final distance can only grow from here.
        if (row_min > cap) return cap + 1;
        const swap = prev;
        prev = curr;
        curr = swap;
    }
    const dist = prev[x.len];
    return if (dist > cap) cap + 1 else dist;
}

pub const DivergenceBuckets = struct {
    exact_match: u32 = 0,
    small: u32 = 0, // 1..3
    medium: u32 = 0, // 4..10
    large: u32 = 0, // >10 (includes capped-out)
};

pub const ModelKind = enum { bpe, unigram, wordpiece, monster, byte_id };

pub const VocabMeta = struct {
    path: []const u8,
    kind: ModelKind,
    vocab_size: u32,
};

pub const ReportOptions = struct {
    /// Cap on top-differing-tokens lists. Defaults to 20 per spec.
    top_tokens: u32 = 20,
    /// Stop bothering with full Levenshtein once the streams are this
    /// far apart in length; just bucket as `large`. Defaults to the
    /// module-level cap.
    levenshtein_cap: usize = LEVENSHTEIN_CAP,
};

pub const ReportData = struct {
    allocator: std.mem.Allocator,
    corpus_bytes: u64,
    total_lines: u32,
    equal_lines: u32,
    word_count: u64,
    tokens_a: u64,
    tokens_b: u64,
    buckets: DivergenceBuckets,
    // Top-differing token strings (owned slices).
    top_a_only: []TokenCount,
    top_b_only: []TokenCount,

    pub const TokenCount = struct {
        token: []u8,
        count: u32,
    };

    pub fn deinit(self: *ReportData) void {
        for (self.top_a_only) |*t| self.allocator.free(t.token);
        for (self.top_b_only) |*t| self.allocator.free(t.token);
        if (self.top_a_only.len > 0) self.allocator.free(self.top_a_only);
        if (self.top_b_only.len > 0) self.allocator.free(self.top_b_only);
        self.top_a_only = &.{};
        self.top_b_only = &.{};
    }

    pub fn fertilityA(self: ReportData) f64 {
        return if (self.word_count == 0) 0 else @as(f64, @floatFromInt(self.tokens_a)) / @as(f64, @floatFromInt(self.word_count));
    }
    pub fn fertilityB(self: ReportData) f64 {
        return if (self.word_count == 0) 0 else @as(f64, @floatFromInt(self.tokens_b)) / @as(f64, @floatFromInt(self.word_count));
    }
    pub fn bytesPerTokenA(self: ReportData) f64 {
        return if (self.tokens_a == 0) 0 else @as(f64, @floatFromInt(self.corpus_bytes)) / @as(f64, @floatFromInt(self.tokens_a));
    }
    pub fn bytesPerTokenB(self: ReportData) f64 {
        return if (self.tokens_b == 0) 0 else @as(f64, @floatFromInt(self.corpus_bytes)) / @as(f64, @floatFromInt(self.tokens_b));
    }
};

/// Whitespace-split word count for fertility. Matches `wc -w` semantics:
/// runs of non-whitespace separated by ASCII whitespace.
pub fn countWords(corpus: []const u8) u64 {
    var n: u64 = 0;
    var in_word = false;
    for (corpus) |c| {
        const is_ws = c == ' ' or c == '\t' or c == '\n' or c == '\r' or c == 11 or c == 12;
        if (is_ws) {
            in_word = false;
        } else if (!in_word) {
            in_word = true;
            n += 1;
        }
    }
    return n;
}

fn bucketDistance(d: usize, buckets: *DivergenceBuckets) void {
    if (d == 0) {
        buckets.exact_match += 1;
    } else if (d <= 3) {
        buckets.small += 1;
    } else if (d <= 10) {
        buckets.medium += 1;
    } else {
        buckets.large += 1;
    }
}

const Vocab_for_report = @import("vocab.zig").Vocab;
const Model_for_report = @import("model.zig").Model;

/// Resolve a single id back to its byte string via the pipeline's model.
/// Falls back to "<id N>" for ids the model can't decode (e.g. specials
/// it has no slot for).
fn idToBytes(
    allocator: std.mem.Allocator,
    model: Model_for_report,
    vocab: *const Vocab_for_report,
    id: TokenId,
) ![]u8 {
    var scratch: [1]u8 = .{0};
    // Some models will assert on out-of-range ids; guard by size.
    const size: u32 = switch (model) {
        .byte_id => 256,
        .bpe => |b| b.count,
        .unigram => |u| u.count,
        .wordpiece => |w| w.count,
        .monster => |m| m.count,
    };
    if (id >= size) return std.fmt.allocPrint(allocator, "<id {d}>", .{id});
    const bytes = model.idBytes(id, vocab, &scratch);
    return allocator.dupe(u8, bytes);
}

/// Walk `corpus` line-by-line, encoding through both pipelines, and
/// collect every metric the Markdown report needs in a single pass.
/// Caller owns the returned `ReportData` and must call `deinit`.
pub fn computeReport(
    allocator: std.mem.Allocator,
    pipeline_a: *const Pipeline,
    pipeline_b: *const Pipeline,
    corpus: []const u8,
    opts: ReportOptions,
) !ReportData {
    var tokens_a: u64 = 0;
    var tokens_b: u64 = 0;
    var total_lines: u32 = 0;
    var equal_lines: u32 = 0;
    var buckets: DivergenceBuckets = .{};

    // Diff token strings: token-string -> count.
    var a_only: std.StringHashMap(u32) = .init(allocator);
    defer {
        var it = a_only.iterator();
        while (it.next()) |e| allocator.free(e.key_ptr.*);
        a_only.deinit();
    }
    var b_only: std.StringHashMap(u32) = .init(allocator);
    defer {
        var it = b_only.iterator();
        while (it.next()) |e| allocator.free(e.key_ptr.*);
        b_only.deinit();
    }

    var line_it = std.mem.splitScalar(u8, corpus, '\n');
    while (line_it.next()) |line| {
        total_lines += 1;

        const ids_a = try pipeline_a.encode(allocator, line);
        defer allocator.free(ids_a);
        const ids_b = try pipeline_b.encode(allocator, line);
        defer allocator.free(ids_b);

        tokens_a += ids_a.len;
        tokens_b += ids_b.len;

        const same = std.mem.eql(TokenId, ids_a, ids_b);
        if (same) {
            equal_lines += 1;
            buckets.exact_match += 1;
            continue;
        }

        const d = try levenshteinIds(allocator, ids_a, ids_b, opts.levenshtein_cap);
        bucketDistance(d, &buckets);

        // Symmetric-difference per-line on token *strings* — multiset
        // semantics would over-count repeats inside one line, so we
        // de-duplicate within each side's id set first.
        var seen_a: std.AutoHashMap(TokenId, void) = .init(allocator);
        defer seen_a.deinit();
        var seen_b: std.AutoHashMap(TokenId, void) = .init(allocator);
        defer seen_b.deinit();
        for (ids_a) |id| try seen_a.put(id, {});
        for (ids_b) |id| try seen_b.put(id, {});

        var it_a = seen_a.keyIterator();
        while (it_a.next()) |idp| {
            if (seen_b.contains(idp.*)) continue;
            const bytes = try idToBytes(allocator, pipeline_a.model, pipeline_a.vocab, idp.*);
            const gop = try a_only.getOrPut(bytes);
            if (gop.found_existing) {
                allocator.free(bytes);
                gop.value_ptr.* += 1;
            } else {
                gop.value_ptr.* = 1;
            }
        }
        var it_b = seen_b.keyIterator();
        while (it_b.next()) |idp| {
            if (seen_a.contains(idp.*)) continue;
            const bytes = try idToBytes(allocator, pipeline_b.model, pipeline_b.vocab, idp.*);
            const gop = try b_only.getOrPut(bytes);
            if (gop.found_existing) {
                allocator.free(bytes);
                gop.value_ptr.* += 1;
            } else {
                gop.value_ptr.* = 1;
            }
        }
    }

    const top_a = try topTokens(allocator, &a_only, opts.top_tokens);
    errdefer freeTokenCounts(allocator, top_a);
    const top_b = try topTokens(allocator, &b_only, opts.top_tokens);

    return .{
        .allocator = allocator,
        .corpus_bytes = corpus.len,
        .total_lines = total_lines,
        .equal_lines = equal_lines,
        .word_count = countWords(corpus),
        .tokens_a = tokens_a,
        .tokens_b = tokens_b,
        .buckets = buckets,
        .top_a_only = top_a,
        .top_b_only = top_b,
    };
}

fn freeTokenCounts(allocator: std.mem.Allocator, slice: []ReportData.TokenCount) void {
    for (slice) |*t| allocator.free(t.token);
    if (slice.len > 0) allocator.free(slice);
}

/// Sort the diff-token map and return the top-N entries by descending
/// count, with alphabetical tiebreak. Always returns a freshly-allocated
/// slice with freshly-duplicated keys; the caller owns it.
fn topTokens(
    allocator: std.mem.Allocator,
    map: *std.StringHashMap(u32),
    top_n: u32,
) ![]ReportData.TokenCount {
    const n = map.count();
    if (n == 0) return &[_]ReportData.TokenCount{};

    var entries = try allocator.alloc(ReportData.TokenCount, n);
    defer allocator.free(entries);

    var it = map.iterator();
    var i: usize = 0;
    while (it.next()) |e| : (i += 1) {
        // Borrow the key bytes for sorting — we'll dupe the survivors.
        entries[i] = .{ .token = @constCast(e.key_ptr.*), .count = e.value_ptr.* };
    }

    const Ctx = struct {
        fn lessThan(_: void, a: ReportData.TokenCount, b: ReportData.TokenCount) bool {
            if (a.count != b.count) return a.count > b.count;
            return std.mem.lessThan(u8, a.token, b.token);
        }
    };
    std.mem.sort(ReportData.TokenCount, entries, {}, Ctx.lessThan);

    const keep = @min(@as(usize, top_n), n);
    if (keep == 0) return &[_]ReportData.TokenCount{};
    var out = try allocator.alloc(ReportData.TokenCount, keep);
    var k: usize = 0;
    while (k < keep) : (k += 1) {
        out[k] = .{
            .token = try allocator.dupe(u8, entries[k].token),
            .count = entries[k].count,
        };
    }
    return out;
}

fn modelKindName(k: ModelKind) []const u8 {
    return switch (k) {
        .bpe => "bpe",
        .unigram => "unigram",
        .wordpiece => "wordpiece",
        .monster => "monster",
        .byte_id => "byte_id",
    };
}

fn appendMdEscaped(allocator: std.mem.Allocator, buf: *std.ArrayList(u8), token: []const u8) !void {
    // Backslash-escape backticks and backslashes, and replace control
    // bytes with their `\xNN` form so the rendered cell stays one line.
    try buf.append(allocator, '`');
    for (token) |c| {
        if (c == '`' or c == '\\') {
            try buf.append(allocator, '\\');
            try buf.append(allocator, c);
        } else if (c == '|') {
            // Pipe would break a Markdown table cell.
            try buf.appendSlice(allocator, "\\|");
        } else if (c < 0x20 or c == 0x7f) {
            try buf.print(allocator, "\\x{x:0>2}", .{c});
        } else {
            try buf.append(allocator, c);
        }
    }
    try buf.append(allocator, '`');
}

/// Render the Markdown report with the five sections the spec mandates.
/// Pure function over the already-collected `ReportData` plus the two
/// vocab metadata records.
pub fn formatMarkdownReport(
    allocator: std.mem.Allocator,
    data: *const ReportData,
    meta_a: VocabMeta,
    meta_b: VocabMeta,
) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);

    // --- 1. Header ------------------------------------------------------
    try buf.appendSlice(allocator, "# ztok diff report\n\n");
    try buf.appendSlice(allocator, "| Side | Path | Kind | Vocab size |\n");
    try buf.appendSlice(allocator, "|------|------|------|------------|\n");
    try buf.print(allocator, "| A | `{s}` | {s} | {d} |\n", .{ meta_a.path, modelKindName(meta_a.kind), meta_a.vocab_size });
    try buf.print(allocator, "| B | `{s}` | {s} | {d} |\n", .{ meta_b.path, modelKindName(meta_b.kind), meta_b.vocab_size });
    try buf.print(allocator, "\nCorpus: {d} bytes, {d} lines, {d} whitespace-words.\n\n", .{
        data.corpus_bytes, data.total_lines, data.word_count,
    });

    // --- 2. Fertility ---------------------------------------------------
    try buf.appendSlice(allocator, "## Fertility (tokens per word)\n\n");
    try buf.appendSlice(allocator, "| Side | Tokens | Words | Tokens/word |\n");
    try buf.appendSlice(allocator, "|------|-------:|------:|------------:|\n");
    try buf.print(allocator, "| A | {d} | {d} | {d:.4} |\n", .{ data.tokens_a, data.word_count, data.fertilityA() });
    try buf.print(allocator, "| B | {d} | {d} | {d:.4} |\n\n", .{ data.tokens_b, data.word_count, data.fertilityB() });

    // --- 3. Compression -------------------------------------------------
    try buf.appendSlice(allocator, "## Compression (bytes per token)\n\n");
    try buf.appendSlice(allocator, "| Side | Bytes | Tokens | Bytes/token |\n");
    try buf.appendSlice(allocator, "|------|------:|-------:|------------:|\n");
    try buf.print(allocator, "| A | {d} | {d} | {d:.4} |\n", .{ data.corpus_bytes, data.tokens_a, data.bytesPerTokenA() });
    try buf.print(allocator, "| B | {d} | {d} | {d:.4} |\n\n", .{ data.corpus_bytes, data.tokens_b, data.bytesPerTokenB() });

    // --- 4. Divergence per line -----------------------------------------
    try buf.appendSlice(allocator, "## Divergence per line\n\n");
    const denom: f64 = if (data.total_lines == 0) 1 else @floatFromInt(data.total_lines);
    const eq_pct: f64 = @as(f64, @floatFromInt(data.equal_lines)) * 100.0 / denom;
    try buf.print(allocator, "Identical lines: {d} / {d} ({d:.2}%).\n\n", .{
        data.equal_lines, data.total_lines, eq_pct,
    });
    try buf.appendSlice(allocator, "Edit-distance histogram on id streams (Levenshtein, ");
    try buf.print(allocator, "cap={d}):\n\n", .{LEVENSHTEIN_CAP});
    try buf.appendSlice(allocator, "| Bucket | Lines |\n");
    try buf.appendSlice(allocator, "|--------|------:|\n");
    try buf.print(allocator, "| exact_match (d=0) | {d} |\n", .{data.buckets.exact_match});
    try buf.print(allocator, "| small (1..3)      | {d} |\n", .{data.buckets.small});
    try buf.print(allocator, "| medium (4..10)    | {d} |\n", .{data.buckets.medium});
    try buf.print(allocator, "| large (>10)       | {d} |\n\n", .{data.buckets.large});

    // --- 5. Top differing tokens ----------------------------------------
    try buf.appendSlice(allocator, "## Top differing tokens\n\n");
    try buf.appendSlice(allocator, "Counted on non-matching lines; counts the lines on which a token\n");
    try buf.appendSlice(allocator, "appears in one side's output but not the other.\n\n");

    try buf.appendSlice(allocator, "### In A but not B\n\n");
    if (data.top_a_only.len == 0) {
        try buf.appendSlice(allocator, "_(none)_\n\n");
    } else {
        try buf.appendSlice(allocator, "| Rank | Token | Lines |\n");
        try buf.appendSlice(allocator, "|-----:|-------|------:|\n");
        for (data.top_a_only, 0..) |t, idx| {
            try buf.print(allocator, "| {d} | ", .{idx + 1});
            try appendMdEscaped(allocator, &buf, t.token);
            try buf.print(allocator, " | {d} |\n", .{t.count});
        }
        try buf.append(allocator, '\n');
    }

    try buf.appendSlice(allocator, "### In B but not A\n\n");
    if (data.top_b_only.len == 0) {
        try buf.appendSlice(allocator, "_(none)_\n");
    } else {
        try buf.appendSlice(allocator, "| Rank | Token | Lines |\n");
        try buf.appendSlice(allocator, "|-----:|-------|------:|\n");
        for (data.top_b_only, 0..) |t, idx| {
            try buf.print(allocator, "| {d} | ", .{idx + 1});
            try appendMdEscaped(allocator, &buf, t.token);
            try buf.print(allocator, " | {d} |\n", .{t.count});
        }
    }

    return buf.toOwnedSlice(allocator);
}

// --- tests ---

const Vocab = @import("vocab.zig").Vocab;
const Bpe = @import("bpe.zig").Bpe;
const added_tokens_mod = @import("added_tokens.zig");

test "identical pipelines produce zero divergent lines" {
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();

    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };

    const corpus = "alpha\nbeta\ngamma\ndelta\nepsilon";
    var report = try diff(std.testing.allocator, &pipe, &pipe, corpus, .{});
    defer report.deinit();

    try std.testing.expectEqual(@as(u32, 0), report.summary.divergent_lines);
    try std.testing.expectEqual(@as(u32, 5), report.summary.total_lines);
    try std.testing.expectEqual(report.summary.tokens_a, report.summary.tokens_b);
    try std.testing.expectEqual(@as(usize, 0), report.diffs.len);
}

// Build a Bpe with 256 single-byte tokens + extras (ranks 256+).
fn buildByteBpeForTest(allocator: std.mem.Allocator) !Bpe {
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);
    const enc = std.base64.standard.Encoder;
    var enc_buf: [16]u8 = undefined;

    var b: u32 = 0;
    while (b < 256) : (b += 1) {
        const byte: [1]u8 = .{@intCast(b)};
        const out = enc.encode(&enc_buf, &byte);
        try buf.print(allocator, "{s} {d}\n", .{ out, b });
    }
    const extras = [_][]const u8{ "he", "hel", "hell", "hello" };
    var rank: u32 = 256;
    for (extras) |bytes| {
        const out = enc.encode(&enc_buf, bytes);
        try buf.print(allocator, "{s} {d}\n", .{ out, rank });
        rank += 1;
    }
    return Bpe.loadTiktokenBytes(allocator, buf.items);
}

test "different pipelines diverge" {
    const a = std.testing.allocator;

    var v = Vocab.empty(a);
    defer v.deinit();

    var bpe = try buildByteBpeForTest(a);
    defer bpe.deinit();

    const pipe_a: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };
    const pipe_b: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .{ .bpe = &bpe },
        .decoder = .concat,
        .vocab = &v,
    };

    // Lines that exercise the "hello" merges plus a plain one.
    const corpus = "hello\nworld\nhello world\nhe\nzzz";
    var report = try diff(a, &pipe_a, &pipe_b, corpus, .{});
    defer report.deinit();

    try std.testing.expect(report.summary.divergent_lines > 0);
    try std.testing.expect(report.summary.tokens_b < report.summary.tokens_a);
    try std.testing.expect(report.diffs.len == report.summary.divergent_lines);
}

test "reduction_pct is computed correctly" {
    const a = std.testing.allocator;

    var v = Vocab.empty(a);
    defer v.deinit();

    var bpe = try buildByteBpeForTest(a);
    defer bpe.deinit();

    const pipe_a: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };
    const pipe_b: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .{ .bpe = &bpe },
        .decoder = .concat,
        .vocab = &v,
    };

    // "hello" by byte_id = 5 tokens; by bpe with full merges = 1 token.
    const corpus = "hello";
    var report = try diff(a, &pipe_a, &pipe_b, corpus, .{});
    defer report.deinit();

    try std.testing.expectEqual(@as(u64, 5), report.summary.tokens_a);
    try std.testing.expectEqual(@as(u64, 1), report.summary.tokens_b);
    // (5 - 1) / 5 = 80%
    try std.testing.expectApproxEqAbs(@as(f64, 80.0), report.summary.reduction_pct, 1e-9);

    // Reverse direction -> negative reduction.
    var rev = try diff(a, &pipe_b, &pipe_a, corpus, .{});
    defer rev.deinit();
    try std.testing.expect(rev.summary.reduction_pct < 0);
    // (1 - 5) / 1 = -400%
    try std.testing.expectApproxEqAbs(@as(f64, -400.0), rev.summary.reduction_pct, 1e-9);
}

test "max_detailed_diffs caps the diffs slice" {
    const a = std.testing.allocator;

    var v = Vocab.empty(a);
    defer v.deinit();

    var bpe = try buildByteBpeForTest(a);
    defer bpe.deinit();

    const pipe_a: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };
    const pipe_b: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .{ .bpe = &bpe },
        .decoder = .concat,
        .vocab = &v,
    };

    // 10 lines, each "hello" -> all 10 diverge under byte_id vs bpe.
    const corpus = "hello\nhello\nhello\nhello\nhello\nhello\nhello\nhello\nhello\nhello";
    var report = try diff(a, &pipe_a, &pipe_b, corpus, .{ .max_detailed_diffs = 3 });
    defer report.deinit();

    try std.testing.expectEqual(@as(u32, 10), report.summary.divergent_lines);
    try std.testing.expectEqual(@as(usize, 3), report.diffs.len);
}

test "min_line_bytes skips short lines" {
    const a = std.testing.allocator;

    var v = Vocab.empty(a);
    defer v.deinit();

    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };

    var bpe = try buildByteBpeForTest(a);
    defer bpe.deinit();

    const pipe_b: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .{ .bpe = &bpe },
        .decoder = .concat,
        .vocab = &v,
    };

    // Mix of empty + non-empty lines. With min_line_bytes=1, the empty
    // lines (length 0) are skipped, contributing nothing to token
    // counts and never appearing in diffs.
    const corpus = "\nhello\n\n\nhello\n";
    var report = try diff(a, &pipe, &pipe_b, corpus, .{ .min_line_bytes = 1 });
    defer report.deinit();

    // 6 segments from splitScalar (trailing "\n" yields a final empty).
    try std.testing.expectEqual(@as(u32, 6), report.summary.total_lines);
    // Only the two "hello" lines should be encoded; both diverge.
    try std.testing.expectEqual(@as(u32, 2), report.summary.divergent_lines);
    // tokens_a = 2 * 5 = 10, tokens_b = 2 * 1 = 2
    try std.testing.expectEqual(@as(u64, 10), report.summary.tokens_a);
    try std.testing.expectEqual(@as(u64, 2), report.summary.tokens_b);

    // None of the recorded diffs should have an empty line.
    for (report.diffs) |d| try std.testing.expect(d.line.len > 0);
}

test "formatText produces non-empty output" {
    const a = std.testing.allocator;

    var v = Vocab.empty(a);
    defer v.deinit();

    var bpe = try buildByteBpeForTest(a);
    defer bpe.deinit();

    const pipe_a: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };
    const pipe_b: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .{ .bpe = &bpe },
        .decoder = .concat,
        .vocab = &v,
    };

    const corpus = "hello\nworld";
    var report = try diff(a, &pipe_a, &pipe_b, corpus, .{});
    defer report.deinit();

    const text = try formatText(a, &report);
    defer a.free(text);

    try std.testing.expect(text.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, text, "ztok diff summary") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "tokens A:") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "tokens B:") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "reduction:") != null);
}

// --- report tests -------------------------------------------------------

test "report: fertility ratio matches known token counts" {
    const a = std.testing.allocator;

    var v = Vocab.empty(a);
    defer v.deinit();
    var bpe = try buildByteBpeForTest(a);
    defer bpe.deinit();

    // byte_id encodes "hello world" -> 11 tokens; bpe (with hello merges)
    // -> 1 ("hello") + 6 single bytes = 7 tokens. Whitespace word count
    // is 2.
    const pipe_a: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };
    const pipe_b: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .{ .bpe = &bpe },
        .decoder = .concat,
        .vocab = &v,
    };

    const corpus = "hello world";
    var data = try computeReport(a, &pipe_a, &pipe_b, corpus, .{});
    defer data.deinit();

    try std.testing.expectEqual(@as(u64, 2), data.word_count);
    try std.testing.expectEqual(@as(u64, 11), data.tokens_a);
    try std.testing.expectEqual(@as(u64, 7), data.tokens_b);
    // fertility A = 11/2 = 5.5, fertility B = 7/2 = 3.5
    try std.testing.expectApproxEqAbs(@as(f64, 5.5), data.fertilityA(), 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 3.5), data.fertilityB(), 1e-9);
}

test "report: divergence histogram buckets lines correctly" {
    const a = std.testing.allocator;

    // Hand-rolled scenario: we'll skip pipelines and exercise the
    // bucketing logic directly via `levenshteinIds` + `bucketDistance`,
    // matching the spec's "4 lines where 2 match, 1 differs by 1 id, 1
    // differs by 20 ids" requirement.
    const id_seq_a = [_]TokenId{ 1, 2, 3, 4, 5 };
    const id_seq_b_eq = [_]TokenId{ 1, 2, 3, 4, 5 };
    const id_seq_b_one = [_]TokenId{ 1, 2, 99, 4, 5 }; // one substitution
    var seq_b_twenty: [25]TokenId = undefined;
    var k: usize = 0;
    while (k < 25) : (k += 1) seq_b_twenty[k] = @intCast(1000 + k);

    var buckets: DivergenceBuckets = .{};
    // line 1: equal
    const d1 = try levenshteinIds(a, &id_seq_a, &id_seq_b_eq, LEVENSHTEIN_CAP);
    bucketDistance(d1, &buckets);
    // line 2: equal
    const d2 = try levenshteinIds(a, &id_seq_a, &id_seq_b_eq, LEVENSHTEIN_CAP);
    bucketDistance(d2, &buckets);
    // line 3: differs by 1 id (substitution -> distance 1)
    const d3 = try levenshteinIds(a, &id_seq_a, &id_seq_b_one, LEVENSHTEIN_CAP);
    bucketDistance(d3, &buckets);
    // line 4: 5 ids vs 25 entirely-different ids — distance is 25
    // (5 substitutions + 20 insertions = 25, which is > 10 -> large).
    const d4 = try levenshteinIds(a, &id_seq_a, &seq_b_twenty, LEVENSHTEIN_CAP);
    bucketDistance(d4, &buckets);

    try std.testing.expectEqual(@as(u32, 2), buckets.exact_match);
    try std.testing.expectEqual(@as(u32, 1), buckets.small);
    try std.testing.expectEqual(@as(u32, 0), buckets.medium);
    try std.testing.expectEqual(@as(u32, 1), buckets.large);
}

test "report: top differing tokens is deterministic with alphabetical tiebreak" {
    const a = std.testing.allocator;

    // Build a map with deliberate count ties so the alphabetical
    // tiebreak is exercised.
    var m: std.StringHashMap(u32) = .init(a);
    // Defers run in reverse declaration order — free keys first, then
    // deinit the map, so the iterator still sees live key pointers.
    defer m.deinit();
    defer {
        var it = m.iterator();
        while (it.next()) |e| a.free(e.key_ptr.*);
    }
    // counts: zebra=3, alpha=2, mango=2, bravo=1, kiwi=1.
    try m.put(try a.dupe(u8, "zebra"), 3);
    try m.put(try a.dupe(u8, "alpha"), 2);
    try m.put(try a.dupe(u8, "mango"), 2);
    try m.put(try a.dupe(u8, "bravo"), 1);
    try m.put(try a.dupe(u8, "kiwi"), 1);

    const top = try topTokens(a, &m, 20);
    defer {
        for (top) |*t| a.free(t.token);
        a.free(top);
    }

    try std.testing.expectEqual(@as(usize, 5), top.len);
    // Highest count first: zebra (3).
    try std.testing.expectEqualStrings("zebra", top[0].token);
    try std.testing.expectEqual(@as(u32, 3), top[0].count);
    // Then count=2 entries alphabetically: alpha, mango.
    try std.testing.expectEqualStrings("alpha", top[1].token);
    try std.testing.expectEqualStrings("mango", top[2].token);
    // Then count=1 entries alphabetically: bravo, kiwi.
    try std.testing.expectEqualStrings("bravo", top[3].token);
    try std.testing.expectEqualStrings("kiwi", top[4].token);

    // Truncation to top-N preserves the same order.
    const top2 = try topTokens(a, &m, 3);
    defer {
        for (top2) |*t| a.free(t.token);
        a.free(top2);
    }
    try std.testing.expectEqual(@as(usize, 3), top2.len);
    try std.testing.expectEqualStrings("zebra", top2[0].token);
    try std.testing.expectEqualStrings("alpha", top2[1].token);
    try std.testing.expectEqualStrings("mango", top2[2].token);
}
