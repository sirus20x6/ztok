//! Library-level helpers backing the `ztok diff` CLI subcommand. Kept out
//! of `src/main.zig` so the formatting + summary logic can be unit-tested
//! without spawning a subprocess.
//!
//! Inputs: two already-loaded `*const Pipeline` values, the input text,
//! and a `*std.Io.Writer`. Output: either a text report or newline-
//! delimited JSON (NDJSON) — one JSON object per input line plus a final
//! "summary" object. The JSON schema is documented at the bottom of this
//! file and is stable across releases.

const std = @import("std");
const TokenId = @import("token.zig").TokenId;
const Pipeline = @import("pipeline.zig").Pipeline;

pub const Format = enum { text, json };

pub const DiffOptions = struct {
    format: Format = .text,
};

pub const DiffResult = struct {
    total_lines: u32,
    matching_lines: u32,
    diverging_lines: u32,
    total_ids_a: u64,
    total_ids_b: u64,
    total_bytes: u64,

    pub fn allMatch(self: DiffResult) bool {
        return self.diverging_lines == 0;
    }
    pub fn exitCode(self: DiffResult) u8 {
        return if (self.allMatch()) 0 else 1;
    }
};

/// Encode each line of `input_text` through both pipelines and emit per-
/// line diff records to `out`. A trailing newline is treated as a line
/// terminator, not a separate empty line — matching the convention used
/// by `runRoundtrip`. If the input contains no newlines, the whole buffer
/// is processed as one line.
pub fn runDiff(
    allocator: std.mem.Allocator,
    pipeline_a: *const Pipeline,
    pipeline_b: *const Pipeline,
    input_text: []const u8,
    opts: DiffOptions,
    out: *std.Io.Writer,
) !DiffResult {
    var result: DiffResult = .{
        .total_lines = 0,
        .matching_lines = 0,
        .diverging_lines = 0,
        .total_ids_a = 0,
        .total_ids_b = 0,
        .total_bytes = 0,
    };

    var line_no: u32 = 0;
    var iter = std.mem.splitScalar(u8, input_text, '\n');
    const last_was_newline = input_text.len > 0 and input_text[input_text.len - 1] == '\n';

    if (opts.format == .text) {
        try out.writeAll("ztok diff\n");
    }

    while (iter.next()) |raw| {
        if (iter.peek() == null and raw.len == 0 and last_was_newline) break;

        line_no += 1;
        result.total_lines += 1;
        result.total_bytes += raw.len;

        const ids_a = try pipeline_a.encode(allocator, raw);
        defer allocator.free(ids_a);
        const ids_b = try pipeline_b.encode(allocator, raw);
        defer allocator.free(ids_b);

        result.total_ids_a += ids_a.len;
        result.total_ids_b += ids_b.len;

        const match = std.mem.eql(TokenId, ids_a, ids_b);
        if (match) {
            result.matching_lines += 1;
        } else {
            result.diverging_lines += 1;
        }

        // First divergence position (token index). -1 when fully matching.
        // For mismatched-length encodings, the first divergence is at the
        // first position the two arrays disagree, including the position
        // where the shorter one ends.
        const first_div: i64 = blk: {
            const min_len = @min(ids_a.len, ids_b.len);
            var k: usize = 0;
            while (k < min_len) : (k += 1) {
                if (ids_a[k] != ids_b[k]) break :blk @intCast(k);
            }
            if (ids_a.len != ids_b.len) break :blk @intCast(min_len);
            break :blk -1;
        };

        const bpt_a: f64 = if (ids_a.len == 0) 0 else @as(f64, @floatFromInt(raw.len)) / @as(f64, @floatFromInt(ids_a.len));
        const bpt_b: f64 = if (ids_b.len == 0) 0 else @as(f64, @floatFromInt(raw.len)) / @as(f64, @floatFromInt(ids_b.len));

        switch (opts.format) {
            .text => {
                const marker: []const u8 = if (match) "=" else "!";
                try out.print(
                    "  {s} line {d:>4}: a={d} ids ({d:.2} b/tok)  b={d} ids ({d:.2} b/tok)  first_div={d}\n",
                    .{ marker, line_no, ids_a.len, bpt_a, ids_b.len, bpt_b, first_div },
                );
            },
            .json => {
                try out.writeAll("{\"type\":\"line\",\"line\":");
                try out.print("{d}", .{line_no});
                try out.writeAll(",\"bytes\":");
                try out.print("{d}", .{raw.len});
                try out.writeAll(",\"ids_a\":");
                try out.print("{d}", .{ids_a.len});
                try out.writeAll(",\"ids_b\":");
                try out.print("{d}", .{ids_b.len});
                try out.writeAll(",\"bytes_per_token_a\":");
                try writeJsonFloat(out, bpt_a);
                try out.writeAll(",\"bytes_per_token_b\":");
                try writeJsonFloat(out, bpt_b);
                try out.writeAll(",\"match\":");
                try out.writeAll(if (match) "true" else "false");
                try out.writeAll(",\"first_divergence\":");
                try out.print("{d}", .{first_div});
                try out.writeAll("}\n");
            },
        }
    }

    // Summary.
    const corpus_bpt_a: f64 = if (result.total_ids_a == 0) 0 else @as(f64, @floatFromInt(result.total_bytes)) / @as(f64, @floatFromInt(result.total_ids_a));
    const corpus_bpt_b: f64 = if (result.total_ids_b == 0) 0 else @as(f64, @floatFromInt(result.total_bytes)) / @as(f64, @floatFromInt(result.total_ids_b));
    const reduction_pct: f64 = if (result.total_ids_a == 0) 0 else (@as(f64, @floatFromInt(result.total_ids_a)) - @as(f64, @floatFromInt(result.total_ids_b))) * 100.0 / @as(f64, @floatFromInt(result.total_ids_a));

    switch (opts.format) {
        .text => {
            try out.print(
                "\nsummary: {d} lines, {d} matching, {d} diverging, {d} bytes\n",
                .{ result.total_lines, result.matching_lines, result.diverging_lines, result.total_bytes },
            );
            try out.print(
                "         a: {d} ids ({d:.3} b/tok)  b: {d} ids ({d:.3} b/tok)  reduction: {d:.2}% (a -> b)\n",
                .{ result.total_ids_a, corpus_bpt_a, result.total_ids_b, corpus_bpt_b, reduction_pct },
            );
        },
        .json => {
            try out.writeAll("{\"type\":\"summary\",\"total_lines\":");
            try out.print("{d}", .{result.total_lines});
            try out.writeAll(",\"matching_lines\":");
            try out.print("{d}", .{result.matching_lines});
            try out.writeAll(",\"diverging_lines\":");
            try out.print("{d}", .{result.diverging_lines});
            try out.writeAll(",\"total_bytes\":");
            try out.print("{d}", .{result.total_bytes});
            try out.writeAll(",\"total_ids_a\":");
            try out.print("{d}", .{result.total_ids_a});
            try out.writeAll(",\"total_ids_b\":");
            try out.print("{d}", .{result.total_ids_b});
            try out.writeAll(",\"bytes_per_token_a\":");
            try writeJsonFloat(out, corpus_bpt_a);
            try out.writeAll(",\"bytes_per_token_b\":");
            try writeJsonFloat(out, corpus_bpt_b);
            try out.writeAll(",\"reduction_pct\":");
            try writeJsonFloat(out, reduction_pct);
            try out.writeAll("}\n");
        },
    }

    return result;
}

fn writeJsonFloat(out: *std.Io.Writer, v: f64) !void {
    // Avoid emitting `nan`/`inf` since JSON has no representation for them.
    if (std.math.isNan(v) or std.math.isInf(v)) {
        try out.writeAll("null");
        return;
    }
    try out.print("{d:.6}", .{v});
}

// --- markdown report driver ---------------------------------------------

const diff_mod = @import("diff.zig");

pub const ReportOptions = struct {
    /// Maximum number of top differing tokens to render per side.
    top_tokens: u32 = 20,
};

/// Compute the report data via `diff_mod.computeReport` and stream the
/// rendered Markdown to `out`. Caller passes the two vocab metas (path
/// + kind + size) so the header can name them precisely — the pipeline
/// objects themselves don't carry their source path.
pub fn runReport(
    allocator: std.mem.Allocator,
    pipeline_a: *const Pipeline,
    pipeline_b: *const Pipeline,
    meta_a: diff_mod.VocabMeta,
    meta_b: diff_mod.VocabMeta,
    corpus: []const u8,
    opts: ReportOptions,
    out: *std.Io.Writer,
) !void {
    var data = try diff_mod.computeReport(allocator, pipeline_a, pipeline_b, corpus, .{
        .top_tokens = opts.top_tokens,
    });
    defer data.deinit();

    const md = try diff_mod.formatMarkdownReport(allocator, &data, meta_a, meta_b);
    defer allocator.free(md);

    try out.writeAll(md);
}

// --- tests ---------------------------------------------------------------

const testing = std.testing;
const Vocab = @import("vocab.zig").Vocab;
const Bpe = @import("bpe.zig").Bpe;

fn captureOutput(comptime cap: usize) type {
    return struct {
        buf: [cap]u8 = undefined,
        writer: std.Io.Writer = undefined,

        const Self = @This();
        fn init(self: *Self) void {
            self.writer = .fixed(&self.buf);
        }
        fn slice(self: *Self) []const u8 {
            return self.writer.buffered();
        }
    };
}

fn buildByteBpe(alloc: std.mem.Allocator) !Bpe {
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(alloc);
    const enc = std.base64.standard.Encoder;
    var enc_buf: [16]u8 = undefined;
    var b: u32 = 0;
    while (b < 256) : (b += 1) {
        const byte: [1]u8 = .{@intCast(b)};
        const out = enc.encode(&enc_buf, &byte);
        try buf.print(alloc, "{s} {d}\n", .{ out, b });
    }
    const extras = [_][]const u8{ "he", "hel", "hell", "hello" };
    var rank: u32 = 256;
    for (extras) |bytes| {
        const out = enc.encode(&enc_buf, bytes);
        try buf.print(alloc, "{s} {d}\n", .{ out, rank });
        rank += 1;
    }
    return Bpe.loadTiktokenBytes(alloc, buf.items);
}

test "runDiff: identical pipelines produce all matching lines" {
    var v = Vocab.empty(testing.allocator);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };

    const input = "hello\nworld\nfoo bar\n";
    var cap: captureOutput(4096) = .{};
    cap.init();
    const r = try runDiff(testing.allocator, &pipe, &pipe, input, .{}, &cap.writer);
    try testing.expectEqual(@as(u32, 3), r.total_lines);
    try testing.expectEqual(@as(u32, 3), r.matching_lines);
    try testing.expectEqual(@as(u32, 0), r.diverging_lines);
    try testing.expect(r.allMatch());
    try testing.expectEqual(@as(u8, 0), r.exitCode());

    const s = cap.slice();
    try testing.expect(std.mem.indexOf(u8, s, "ztok diff") != null);
    try testing.expect(std.mem.indexOf(u8, s, "summary:") != null);
    try testing.expect(std.mem.indexOf(u8, s, "diverging") != null);
    // All three lines should carry the `=` marker.
    try testing.expectEqual(@as(usize, 3), std.mem.count(u8, s, "= line"));
}

test "runDiff: JSON output is parseable NDJSON with a final summary" {
    var v = Vocab.empty(testing.allocator);
    defer v.deinit();
    var bpe = try buildByteBpe(testing.allocator);
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

    const input = "hello\nworld\n";
    var cap: captureOutput(8192) = .{};
    cap.init();
    const r = try runDiff(testing.allocator, &pipe_a, &pipe_b, input, .{ .format = .json }, &cap.writer);

    // "hello" diverges (byte_id => 5 ids, bpe => 1 id); "world" matches
    // (each byte is its own id under both encoders since no "world"
    // merges exist in the test vocab).
    try testing.expectEqual(@as(u32, 2), r.total_lines);
    try testing.expectEqual(@as(u32, 1), r.diverging_lines);
    try testing.expectEqual(@as(u8, 1), r.exitCode());

    // Parse every newline-delimited JSON object.
    const s = cap.slice();
    var lines = std.mem.splitScalar(u8, s, '\n');
    var line_count: u32 = 0;
    var saw_summary = false;
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, line, .{});
        defer parsed.deinit();
        const obj = parsed.value.object;
        const typ = obj.get("type").?.string;
        if (std.mem.eql(u8, typ, "summary")) {
            saw_summary = true;
            try testing.expect(obj.get("total_lines") != null);
            try testing.expect(obj.get("matching_lines") != null);
            try testing.expect(obj.get("diverging_lines") != null);
            try testing.expect(obj.get("reduction_pct") != null);
        } else if (std.mem.eql(u8, typ, "line")) {
            line_count += 1;
            try testing.expect(obj.get("ids_a") != null);
            try testing.expect(obj.get("ids_b") != null);
            try testing.expect(obj.get("first_divergence") != null);
            try testing.expect(obj.get("match") != null);
        } else {
            try testing.expect(false); // unknown record type
        }
    }
    try testing.expectEqual(@as(u32, 2), line_count);
    try testing.expect(saw_summary);
}
