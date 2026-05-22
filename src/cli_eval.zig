//! Library-level helper backing the `ztok eval` CLI subcommand. The
//! `evaluate` function in `src/eval.zig` already computes every metric
//! we want; this module just wraps it in a CLI-friendly entry point that
//! formats the report as either human-readable text or a single JSON
//! object. The JSON schema is documented at the bottom of this file.

const std = @import("std");
const Pipeline = @import("pipeline.zig").Pipeline;
const eval = @import("eval.zig");

pub const Format = enum { text, json };

pub const EvalOptions = struct {
    format: Format = .text,
    /// Top-K most frequent token ids to report.
    top_k: u32 = 10,
    /// Bottom-K least frequent token ids to report.
    bottom_k: u32 = 10,
    /// Hidden dim for the KV cache estimate. When null, KV cache is skipped.
    hidden_dim: ?u32 = null,
    /// Layer count for KV cache estimate (defaults to 32 when hidden_dim is set).
    num_layers: u32 = 32,
    /// Dtype size in bytes for KV cache estimate (default 2 = fp16/bf16).
    dtype_bytes: u32 = 2,
    /// Per-token input price. When set, eval prints `cost_input` /
    /// `cost_total`. Treat all tokens as input by default.
    price_input: ?f64 = null,
    /// Optional separate per-token output price. When both input/output
    /// prices are set we still treat the corpus as input-only (it's a
    /// corpus, not a conversation transcript) but we surface
    /// `cost_output` = 0 and `cost_total` = `cost_input` so the user
    /// can sanity-check their price-file wiring.
    price_output: ?f64 = null,
    /// Unit the price is denominated in (per 1K or per 1M tokens).
    price_unit: eval.PriceUnit = .per_1m,
};

pub fn runEval(
    allocator: std.mem.Allocator,
    pipeline: *const Pipeline,
    input_text: []const u8,
    opts: EvalOptions,
    out: *std.Io.Writer,
) !void {
    // We want both top-K and bottom-K. The underlying `evaluate` returns
    // the same number for both arms (`opts.top_k`), so request the larger
    // of the two and slice down in the renderer.
    const k_req: u32 = @max(opts.top_k, opts.bottom_k);

    var report = try eval.evaluate(allocator, pipeline, input_text, .{
        .top_k = k_req,
        .model_hidden_size = opts.hidden_dim,
        .model_num_layers = if (opts.hidden_dim != null) opts.num_layers else null,
        .model_dtype_bytes = opts.dtype_bytes,
    });
    defer report.deinit();

    switch (opts.format) {
        .text => try writeText(out, &report, opts),
        .json => try writeJson(out, &report, opts),
    }
}

fn writeText(out: *std.Io.Writer, r: *const eval.Report, opts: EvalOptions) !void {
    try out.writeAll("ztok eval\n");
    try out.print("  corpus_bytes:       {d}\n", .{r.corpus_bytes});
    try out.print("  corpus_codepoints:  {d}\n", .{r.corpus_codepoints});
    try out.print("  corpus_words:       {d}\n", .{r.corpus_words});
    try out.print("  corpus_lines:       {d}\n", .{r.corpus_lines});
    try out.print("  total_tokens:       {d}\n", .{r.total_tokens});
    try out.print("  vocab_size:         {d}\n", .{r.vocab_size});
    try out.print("  unique_tokens_used: {d}\n", .{r.unique_tokens_used});
    try out.print("  bytes_per_token:    {d:.4}\n", .{r.bytes_per_token});
    try out.print("  chars_per_token:    {d:.4}\n", .{r.chars_per_token});
    try out.print("  tokens_per_word:    {d:.4}\n", .{r.tokens_per_word});
    try out.print("  tokens_per_line:    {d:.4}\n", .{r.tokens_per_line});
    try out.print("  fallback_rate:      {d:.4}\n", .{r.fallback_rate});
    if (r.kv_cache_bytes) |kv| {
        try out.print("  kv_cache_bytes:     {d}\n", .{kv});
    } else {
        try out.writeAll("  kv_cache_bytes:     (skipped — pass --hidden-dim)\n");
    }

    try out.writeAll("\n  fertility per script:\n");
    for (r.script_metrics) |sm| {
        try out.print("    {s:<32}  cp={d:>8}  tokens={d:>8}  fertility={d:.4}\n", .{
            sm.script_name, sm.codepoints, sm.tokens, sm.fertility,
        });
    }

    const top_n = @min(@as(usize, opts.top_k), r.top_k_ids.len);
    try out.print("\n  top-{d} most frequent tokens:\n", .{top_n});
    var i: usize = 0;
    while (i < top_n) : (i += 1) {
        try out.print("    id={d:>8}  count={d}\n", .{ r.top_k_ids[i], r.top_k_counts[i] });
    }
    const bot_n = @min(@as(usize, opts.bottom_k), r.bottom_k_ids.len);
    try out.print("\n  bottom-{d} least frequent tokens:\n", .{bot_n});
    i = 0;
    while (i < bot_n) : (i += 1) {
        try out.print("    id={d:>8}  count={d}\n", .{ r.bottom_k_ids[i], r.bottom_k_counts[i] });
    }

    // Cost projection — only printed when a price was supplied so the
    // existing eval output stays byte-identical for flag-free runs.
    if (opts.price_input) |pin| {
        const cin = eval.computeCost(r.total_tokens, pin, opts.price_unit);
        try out.print("  cost_input:  ${d:.2}\n", .{cin});
        if (opts.price_output) |pout| {
            // Corpus has no "output" tokens; surface 0 so users can
            // confirm both prices were parsed.
            const cout = eval.computeCost(0, pout, opts.price_unit);
            try out.print("  cost_output: ${d:.2}\n", .{cout});
            try out.print("  cost_total:  ${d:.2}\n", .{cin + cout});
        } else {
            try out.print("  cost_total:  ${d:.2}\n", .{cin});
        }
    }
}

fn writeJsonString(out: *std.Io.Writer, s: []const u8) !void {
    try out.writeByte('"');
    for (s) |c| switch (c) {
        '"' => try out.writeAll("\\\""),
        '\\' => try out.writeAll("\\\\"),
        '\n' => try out.writeAll("\\n"),
        '\r' => try out.writeAll("\\r"),
        '\t' => try out.writeAll("\\t"),
        0x08 => try out.writeAll("\\b"),
        0x0C => try out.writeAll("\\f"),
        0x00...0x07, 0x0B, 0x0E...0x1F => try out.print("\\u{x:0>4}", .{c}),
        else => try out.writeByte(c),
    };
    try out.writeByte('"');
}

fn writeJsonFloat(out: *std.Io.Writer, v: f64) !void {
    if (std.math.isNan(v) or std.math.isInf(v)) {
        try out.writeAll("null");
        return;
    }
    try out.print("{d:.6}", .{v});
}

fn writeJson(out: *std.Io.Writer, r: *const eval.Report, opts: EvalOptions) !void {
    try out.writeAll("{");
    try out.print("\"corpus_bytes\":{d},", .{r.corpus_bytes});
    try out.print("\"corpus_codepoints\":{d},", .{r.corpus_codepoints});
    try out.print("\"corpus_words\":{d},", .{r.corpus_words});
    try out.print("\"corpus_lines\":{d},", .{r.corpus_lines});
    try out.print("\"total_tokens\":{d},", .{r.total_tokens});
    try out.print("\"vocab_size\":{d},", .{r.vocab_size});
    try out.print("\"unique_tokens_used\":{d},", .{r.unique_tokens_used});
    try out.writeAll("\"bytes_per_token\":");
    try writeJsonFloat(out, r.bytes_per_token);
    try out.writeAll(",\"chars_per_token\":");
    try writeJsonFloat(out, r.chars_per_token);
    try out.writeAll(",\"tokens_per_word\":");
    try writeJsonFloat(out, r.tokens_per_word);
    try out.writeAll(",\"tokens_per_line\":");
    try writeJsonFloat(out, r.tokens_per_line);
    try out.writeAll(",\"fallback_rate\":");
    try writeJsonFloat(out, r.fallback_rate);
    try out.writeAll(",\"kv_cache_bytes\":");
    if (r.kv_cache_bytes) |kv| {
        try out.print("{d}", .{kv});
    } else {
        try out.writeAll("null");
    }

    try out.writeAll(",\"scripts\":[");
    for (r.script_metrics, 0..) |sm, i| {
        if (i > 0) try out.writeAll(",");
        try out.writeAll("{\"name\":");
        try writeJsonString(out, sm.script_name);
        try out.print(",\"codepoints\":{d},\"tokens\":{d},\"fertility\":", .{ sm.codepoints, sm.tokens });
        try writeJsonFloat(out, sm.fertility);
        try out.writeAll("}");
    }
    try out.writeAll("]");

    const top_n = @min(@as(usize, opts.top_k), r.top_k_ids.len);
    try out.writeAll(",\"top_k\":[");
    var i: usize = 0;
    while (i < top_n) : (i += 1) {
        if (i > 0) try out.writeAll(",");
        try out.print("{{\"id\":{d},\"count\":{d}}}", .{ r.top_k_ids[i], r.top_k_counts[i] });
    }
    try out.writeAll("]");

    const bot_n = @min(@as(usize, opts.bottom_k), r.bottom_k_ids.len);
    try out.writeAll(",\"bottom_k\":[");
    i = 0;
    while (i < bot_n) : (i += 1) {
        if (i > 0) try out.writeAll(",");
        try out.print("{{\"id\":{d},\"count\":{d}}}", .{ r.bottom_k_ids[i], r.bottom_k_counts[i] });
    }
    try out.writeAll("]");

    if (opts.price_input) |pin| {
        const cin = eval.computeCost(r.total_tokens, pin, opts.price_unit);
        try out.writeAll(",\"cost_input\":");
        try writeJsonFloat(out, cin);
        if (opts.price_output) |pout| {
            const cout = eval.computeCost(0, pout, opts.price_unit);
            try out.writeAll(",\"cost_output\":");
            try writeJsonFloat(out, cout);
            try out.writeAll(",\"cost_total\":");
            try writeJsonFloat(out, cin + cout);
        } else {
            try out.writeAll(",\"cost_total\":");
            try writeJsonFloat(out, cin);
        }
    }

    try out.writeAll("}\n");
}

// --- tests ----------------------------------------------------------------

const testing = std.testing;
const Vocab = @import("vocab.zig").Vocab;

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

test "runEval text output reports the documented metrics" {
    var v = Vocab.empty(testing.allocator);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };

    var cap: captureOutput(16384) = .{};
    cap.init();
    try runEval(testing.allocator, &pipe, "hello world hello world", .{}, &cap.writer);
    const s = cap.slice();

    // Every documented metric must show up in the text report.
    inline for ([_][]const u8{
        "corpus_bytes",
        "corpus_codepoints",
        "corpus_words",
        "corpus_lines",
        "total_tokens",
        "vocab_size",
        "unique_tokens_used",
        "bytes_per_token",
        "chars_per_token",
        "tokens_per_word",
        "tokens_per_line",
        "fallback_rate",
        "fertility per script",
        "most frequent tokens",
        "least frequent tokens",
        "kv_cache_bytes",
    }) |needle| {
        try testing.expect(std.mem.indexOf(u8, s, needle) != null);
    }
}

test "runEval --top-k 3 returns exactly 3 entries in top section" {
    var v = Vocab.empty(testing.allocator);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };

    var cap: captureOutput(16384) = .{};
    cap.init();
    try runEval(testing.allocator, &pipe, "hello world hello world hello", .{
        .top_k = 3,
        .bottom_k = 3,
        .format = .json,
    }, &cap.writer);
    const s = cap.slice();

    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, s, .{});
    defer parsed.deinit();
    const root = parsed.value.object;

    const top = root.get("top_k").?.array;
    try testing.expectEqual(@as(usize, 3), top.items.len);
    const bot = root.get("bottom_k").?.array;
    try testing.expectEqual(@as(usize, 3), bot.items.len);

    // Sanity: top-K count is monotonically non-increasing.
    var prev: i64 = std.math.maxInt(i64);
    for (top.items) |entry| {
        const c = entry.object.get("count").?.integer;
        try testing.expect(c <= prev);
        prev = c;
    }
}

// --- train kind smoke tests --------------------------------------------
//
// These exercise the writer paths the new `ztok train --kind ...` CLI
// uses end-to-end: train a tiny vocab, write it through the matching
// writer, then load the written bytes through the matching loader and
// confirm the output is parseable.

const train_unigram = @import("train_unigram.zig");
const train_wordpiece = @import("train_wordpiece.zig");
const train_monster = @import("train_monster.zig");
const sp_writer = @import("sp_writer.zig");
const sp_model = @import("sp_model.zig");
const hf_writer = @import("hf_writer.zig");
const hf_json = @import("hf_json.zig");
const monster_io = @import("monster_io.zig");
const Span = @import("token.zig").Span;

fn identitySplitForTest(allocator: std.mem.Allocator, input: []const u8) anyerror![]Span {
    const out = try allocator.alloc(Span, 1);
    out[0] = .{ .start = 0, .end = @intCast(input.len) };
    return out;
}

test "ztok train --kind unigram writes a parseable SentencePiece .model" {
    const a = testing.allocator;
    const words = [_][]const u8{"ababab"};
    const counts = [_]u32{100};
    var u = try train_unigram.train(a, .{
        .words = &words,
        .counts = &counts,
    }, .{
        .vocab_size = 270,
        .max_piece_length = 6,
        .em_iters_per_round = 2,
    });
    defer u.deinit();

    const bytes = try sp_writer.writeUnigram(a, &u, .{ .unk_id = u.unk_id });
    defer a.free(bytes);

    var sp = try sp_model.loadFromBytes(a, bytes);
    defer sp.deinit();
    try testing.expectEqual(sp_model.ModelKind.unigram, sp.model_kind);
    try testing.expectEqual(u.count, sp.count);
}

test "ztok train --kind wordpiece writes a parseable HF tokenizer.json" {
    const a = testing.allocator;
    const raw = "the quick brown fox jumps over the lazy dog the quick brown fox";
    var wp = try train_wordpiece.trainFromBytes(a, raw, identitySplitForTest, .{ .vocab_size = 290 });
    defer wp.deinit();

    const bytes = try hf_writer.writeWordPiece(a, &wp, .{});
    defer a.free(bytes);

    var hf = try hf_json.loadFromBytes(a, bytes);
    defer hf.deinit();
    try testing.expectEqual(hf_json.ModelKind.wordpiece, hf.model_kind);
    // The HF reader populates a vocab; confirm it covers what the writer
    // emitted (count is the public vocab size on both sides).
    try testing.expect(hf.vocab.count == wp.count);
}

test "ztok train --kind monster writes a parseable .ztm" {
    const a = testing.allocator;

    // Tiny synthetic Monster — go through the public Builder rather than
    // a real training run to keep the test fast. The writer/reader pair
    // is what we're verifying for the CLI integration.
    const Monster = @import("monster.zig").Monster;
    var b = Monster.Builder.init(a);
    defer b.deinit();
    _ = try b.addToken("a");
    _ = try b.addToken("b");
    _ = try b.addToken("c");
    _ = try b.addToken("ab");
    _ = try b.addToken("abc");
    const unk = try b.addToken("<unk>");
    var m = try b.finalize(unk);
    defer m.deinit();

    const bytes = try monster_io.writeBytes(a, &m);
    defer a.free(bytes);

    var loaded = try monster_io.readBytes(a, bytes);
    defer loaded.deinit();
    try testing.expectEqual(m.count, loaded.count);
    try testing.expectEqual(m.unk_id, loaded.unk_id);
}

test "runEval JSON output: kv_cache_bytes is set when hidden_dim is given" {
    var v = Vocab.empty(testing.allocator);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };

    var cap: captureOutput(16384) = .{};
    cap.init();
    try runEval(testing.allocator, &pipe, "abc", .{
        .format = .json,
        .hidden_dim = 4096,
        .num_layers = 32,
        .dtype_bytes = 2,
    }, &cap.writer);

    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, cap.slice(), .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    const kv = root.get("kv_cache_bytes").?;
    try testing.expect(kv == .integer);
    // 2 * H * L * tokens * D = 2 * 4096 * 32 * 3 * 2 = 1_572_864
    try testing.expectEqual(@as(i64, 1_572_864), kv.integer);
}
