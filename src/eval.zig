const std = @import("std");
const TokenId = @import("token.zig").TokenId;
const Pipeline = @import("pipeline.zig").Pipeline;
const ScratchArena = @import("pipeline.zig").ScratchArena;
const Model = @import("model.zig").Model;
const BatchPool = @import("thread_pool.zig").BatchPool;
const cl100k_split = @import("cl100k.zig").split;
const unicode_props = @import("unicode_props.zig");
const fairness = @import("fairness.zig");

/// Unit a per-token price is denominated in. `per_1k` = price refers to
/// one thousand tokens (e.g. legacy OpenAI billing); `per_1m` = price
/// refers to one million tokens (modern OpenAI/Anthropic style).
pub const PriceUnit = enum { per_1k, per_1m };

/// Estimate the dollar cost of `tokens` at `price_per_unit` dollars per
/// `unit` (1K or 1M tokens). Computed in f64 to avoid integer overflow
/// at trillion-token scales; the division-by-unit happens first so
/// `tokens * price` never has to fit in a u64.
pub fn computeCost(tokens: u64, price_per_unit: f64, unit: PriceUnit) f64 {
    const divisor: f64 = switch (unit) {
        .per_1k => 1_000.0,
        .per_1m => 1_000_000.0,
    };
    const t: f64 = @floatFromInt(tokens);
    return (t / divisor) * price_per_unit;
}

pub const ScriptMetric = struct {
    script_name: []const u8,
    codepoints: u64,
    tokens: u64,
    fertility: f64,
};

pub const Report = struct {
    allocator: std.mem.Allocator,
    corpus_bytes: u64,
    corpus_codepoints: u64,
    corpus_words: u64,
    corpus_lines: u64,
    total_tokens: u64,
    vocab_size: u32,
    unique_tokens_used: u32,
    bytes_per_token: f64,
    chars_per_token: f64,
    tokens_per_word: f64,
    tokens_per_line: f64,
    fallback_rate: f64,
    top_k_ids: []const u32,
    top_k_counts: []const u64,
    bottom_k_ids: []const u32,
    bottom_k_counts: []const u64,
    script_metrics: []ScriptMetric,
    kv_cache_bytes: ?u64 = null,

    pub fn deinit(self: *Report) void {
        if (self.top_k_ids.len > 0) self.allocator.free(self.top_k_ids);
        if (self.top_k_counts.len > 0) self.allocator.free(self.top_k_counts);
        if (self.bottom_k_ids.len > 0) self.allocator.free(self.bottom_k_ids);
        if (self.bottom_k_counts.len > 0) self.allocator.free(self.bottom_k_counts);
        if (self.script_metrics.len > 0) self.allocator.free(self.script_metrics);
        self.* = undefined;
    }
};

pub const Options = struct {
    scripts: bool = true,
    top_k: u32 = 20,
    model_hidden_size: ?u32 = null,
    model_num_layers: ?u32 = null,
    model_dtype_bytes: u32 = 2,
    pool: ?*BatchPool = null,
    fallback_threshold: u32 = 256,
};

// NOTE: scriptOf is a v1 RANGE-BASED APPROXIMATION. The real
// Unicode Script_Extensions table would be a follow-up. Mixed-script
// blocks (e.g. U+0080..U+03FF) collapse into coarse buckets. Treat
// per-script fertility as indicative, not authoritative.
fn scriptOf(cp: u21) []const u8 {
    if (cp < 0x0080) return "Latin (ASCII)";
    if (cp < 0x0100) return "Latin (extended)";
    if (cp < 0x0400) return "Latin/Greek/Coptic/Cyrillic";
    if (cp < 0x0500) return "Cyrillic";
    if (cp < 0x0600) return "Armenian/Hebrew";
    if (cp < 0x0700) return "Arabic";
    if (cp < 0x0900) return "Other";
    if (cp < 0x0A00) return "Devanagari/Bengali";
    if (cp < 0x1000) return "Indic";
    if (cp < 0x2000) return "Other";
    if (cp < 0x2E80) return "Symbols/Punctuation";
    if (cp < 0x3000) return "CJK Radicals";
    if (cp < 0x3040) return "CJK Symbols";
    if (cp < 0x30A0) return "Hiragana";
    if (cp < 0x3100) return "Katakana";
    if (cp < 0x4E00) return "Bopomofo/Other";
    if (cp < 0xA000) return "CJK Unified Ideographs";
    if (cp < 0xAC00) return "Yi/Other";
    if (cp < 0xD800) return "Hangul";
    if (cp < 0xE000) return "Surrogates";
    if (cp < 0xF900) return "Private Use";
    return "Other";
}

fn modelVocabSize(model: Model) u32 {
    return switch (model) {
        .byte_id => 256,
        .bpe => |b| b.count,
        .unigram => |u| u.count,
        .wordpiece => |w| w.count,
        .monster => |m| m.count,
    };
}

const PreScan = struct {
    bytes: u64,
    codepoints: u64,
    lines: u64,
    words: u64,
};

fn preScan(corpus: []const u8, allocator: std.mem.Allocator) !PreScan {
    const bytes: u64 = @intCast(corpus.len);
    var codepoints: u64 = 0;
    var lines: u64 = 0;
    var i: usize = 0;
    while (i < corpus.len) {
        const b = corpus[i];
        if (b == '\n') lines += 1;
        const cp_len: usize = std.unicode.utf8ByteSequenceLength(b) catch 1;
        codepoints += 1;
        i += @min(cp_len, corpus.len - i);
    }
    // Spec: a corpus with no trailing '\n' still has one final line.
    if (bytes > 0 and corpus[corpus.len - 1] != '\n') lines += 1;
    if (bytes == 0) lines = 0;

    var words: u64 = 0;
    if (bytes > 0) {
        const spans = cl100k_split(allocator, corpus) catch |e| switch (e) {
            error.OutOfMemory => return e,
        };
        defer allocator.free(spans);
        for (spans) |s| {
            const sl = s.slice(corpus);
            // Skip leading ASCII/Unicode whitespace, then check first
            // codepoint — counts a span as a word if it contains a
            // letter or digit after leading whitespace.
            var p: usize = 0;
            while (p < sl.len) {
                const cl: usize = std.unicode.utf8ByteSequenceLength(sl[p]) catch 1;
                if (p + cl > sl.len) break;
                const cp = std.unicode.utf8Decode(sl[p .. p + cl]) catch {
                    p += 1;
                    continue;
                };
                if (!unicode_props.isWhitespace(cp)) {
                    if (unicode_props.isLetter(cp) or unicode_props.isNumber(cp)) {
                        words += 1;
                    }
                    break;
                }
                p += cl;
            }
        }
    }

    return .{ .bytes = bytes, .codepoints = codepoints, .lines = lines, .words = words };
}

const Pair = struct { id: u32, count: u64 };
fn cmpDesc(_: void, a: Pair, b: Pair) bool {
    if (a.count != b.count) return a.count > b.count;
    return a.id < b.id;
}
fn cmpAsc(_: void, a: Pair, b: Pair) bool {
    if (a.count != b.count) return a.count < b.count;
    return a.id < b.id;
}

pub fn evaluate(
    allocator: std.mem.Allocator,
    pipeline: *const Pipeline,
    corpus: []const u8,
    opts: Options,
) !Report {
    const scan = try preScan(corpus, allocator);

    // Encode corpus. Single-shot today; BatchPool kept in API for future
    // chunked workflow (would require carving the corpus on line/safe
    // boundaries before fan-out — out of scope for v1).
    _ = opts.pool;
    const ids = try pipeline.encode(allocator, corpus);
    defer allocator.free(ids);

    var counts: std.AutoHashMap(u32, u64) = .init(allocator);
    defer counts.deinit();
    var fallback_count: u64 = 0;
    for (ids) |id| {
        if (id < opts.fallback_threshold) fallback_count += 1;
        const gop = try counts.getOrPut(id);
        if (gop.found_existing) gop.value_ptr.* += 1 else gop.value_ptr.* = 1;
    }

    const total_tokens: u64 = @intCast(ids.len);
    const unique_tokens_used: u32 = @intCast(counts.count());

    // Histograms.
    var pairs: std.ArrayList(Pair) = .empty;
    defer pairs.deinit(allocator);
    try pairs.ensureTotalCapacity(allocator, counts.count());
    var it = counts.iterator();
    while (it.next()) |e| {
        pairs.appendAssumeCapacity(.{ .id = e.key_ptr.*, .count = e.value_ptr.* });
    }

    const k_desired: usize = opts.top_k;
    const k_actual: usize = @min(k_desired, pairs.items.len);

    std.mem.sort(Pair, pairs.items, {}, cmpDesc);
    const top_ids = try allocator.alloc(u32, k_actual);
    errdefer allocator.free(top_ids);
    const top_counts = try allocator.alloc(u64, k_actual);
    errdefer allocator.free(top_counts);
    for (pairs.items[0..k_actual], 0..) |p, i| {
        top_ids[i] = p.id;
        top_counts[i] = p.count;
    }

    std.mem.sort(Pair, pairs.items, {}, cmpAsc);
    const bot_ids = try allocator.alloc(u32, k_actual);
    errdefer allocator.free(bot_ids);
    const bot_counts = try allocator.alloc(u64, k_actual);
    errdefer allocator.free(bot_counts);
    for (pairs.items[0..k_actual], 0..) |p, i| {
        bot_ids[i] = p.id;
        bot_counts[i] = p.count;
    }

    // Scripts — REAL per-script token attribution (v2). We segment the
    // corpus into maximal same-script runs and encode each run through the
    // SAME pipeline `eval` uses for the corpus, then attribute that run's
    // token count to its script. This replaces the v1 proportional split
    // (tokens allocated by codepoint share), under which every script's
    // fertility collapsed to the corpus-wide tokens/codepoint ratio and the
    // fairness metric carried no signal.
    //
    // Segmentation rule
    // -----------------
    // We walk the corpus codepoint by codepoint. Each codepoint has a
    // "script" via `scriptOf`. We split runs only on a STRONG (letter)
    // codepoint whose script differs from the current run's script. Neutral
    // codepoints — whitespace, punctuation, numbers, marks, and any
    // `scriptOf` bucket that is not letter-bearing at that position — ATTACH
    // to the current run. This keeps a "Hello, мир!" style fragment from
    // shattering on the comma/space and matches how a tokenizer actually
    // sees surrounding glue. A run that begins with leading neutrals (before
    // any letter is seen) is keyed by the first codepoint's `scriptOf` so no
    // text is dropped.
    //
    // Caveat (the one honest one)
    // ---------------------------
    // Merges that straddle a script boundary are attributed to the run they
    // are encoded WITHIN, because each run is encoded independently. A merge
    // that would have spanned e.g. a Latin word and the following Han glyph
    // is instead split at the run boundary, so a few tokens shift between
    // adjacent scripts versus a single whole-corpus encode. This is a
    // boundary effect on the order of one token per script transition —
    // acceptable for a fairness proxy and far more faithful than allocating
    // tokens by raw codepoint share.
    const ScriptBucket = struct { codepoints: u64 = 0, tokens: u64 = 0 };
    var script_buckets: std.StringHashMap(ScriptBucket) = .init(allocator);
    defer script_buckets.deinit();

    if (opts.scripts and corpus.len > 0) {
        // Reuse one scratch arena across all run encodes (hot path).
        var scratch = ScratchArena.init(allocator);
        defer scratch.deinit();

        var run_start: usize = 0;
        var run_script: []const u8 = undefined;
        var run_cps: u64 = 0;
        var have_run: bool = false;

        const flush = struct {
            fn call(
                buckets: *std.StringHashMap(ScriptBucket),
                pl: *const Pipeline,
                alloc: std.mem.Allocator,
                arena: *ScratchArena,
                slice: []const u8,
                name: []const u8,
                cps: u64,
            ) !void {
                const run_ids = try pl.encodeWithScratch(alloc, slice, arena);
                defer alloc.free(run_ids);
                const gop = try buckets.getOrPut(name);
                if (!gop.found_existing) gop.value_ptr.* = .{};
                gop.value_ptr.codepoints += cps;
                gop.value_ptr.tokens += @intCast(run_ids.len);
            }
        }.call;

        var p: usize = 0;
        while (p < corpus.len) {
            const cl: usize = std.unicode.utf8ByteSequenceLength(corpus[p]) catch 1;
            const end = @min(p + cl, corpus.len);
            const cp: u21 = @intCast((std.unicode.utf8Decode(corpus[p..end]) catch 0xFFFD) & 0x10FFFF);
            const name = scriptOf(cp);
            const is_strong = unicode_props.isLetter(cp);

            if (!have_run) {
                run_start = p;
                run_script = name;
                run_cps = 1;
                have_run = true;
            } else if (is_strong and !std.mem.eql(u8, name, run_script)) {
                // Boundary: flush the accumulated run, start a fresh one.
                try flush(&script_buckets, pipeline, allocator, &scratch, corpus[run_start..p], run_script, run_cps);
                run_start = p;
                run_script = name;
                run_cps = 1;
            } else {
                run_cps += 1;
            }
            p = end;
        }
        if (have_run) {
            try flush(&script_buckets, pipeline, allocator, &scratch, corpus[run_start..corpus.len], run_script, run_cps);
        }
    }

    const script_metrics = try allocator.alloc(ScriptMetric, script_buckets.count());
    errdefer allocator.free(script_metrics);
    {
        var idx: usize = 0;
        var sit = script_buckets.iterator();
        while (sit.next()) |e| : (idx += 1) {
            const cp_count = e.value_ptr.codepoints;
            const tok = e.value_ptr.tokens;
            const fert: f64 = if (cp_count == 0) 0.0 else @as(f64, @floatFromInt(tok)) / @as(f64, @floatFromInt(cp_count));
            script_metrics[idx] = .{
                .script_name = e.key_ptr.*,
                .codepoints = cp_count,
                .tokens = tok,
                .fertility = fert,
            };
        }
    }

    const bpt: f64 = if (total_tokens == 0) 0.0 else @as(f64, @floatFromInt(scan.bytes)) / @as(f64, @floatFromInt(total_tokens));
    const cpt: f64 = if (total_tokens == 0) 0.0 else @as(f64, @floatFromInt(scan.codepoints)) / @as(f64, @floatFromInt(total_tokens));
    const tpw: f64 = if (scan.words == 0) 0.0 else @as(f64, @floatFromInt(total_tokens)) / @as(f64, @floatFromInt(scan.words));
    const tpl: f64 = if (scan.lines == 0) 0.0 else @as(f64, @floatFromInt(total_tokens)) / @as(f64, @floatFromInt(scan.lines));
    const fbr: f64 = if (total_tokens == 0) 0.0 else @as(f64, @floatFromInt(fallback_count)) / @as(f64, @floatFromInt(total_tokens));

    var kv: ?u64 = null;
    if (opts.model_hidden_size != null and opts.model_num_layers != null) {
        const H: u64 = @intCast(opts.model_hidden_size.?);
        const L: u64 = @intCast(opts.model_num_layers.?);
        const D: u64 = @intCast(opts.model_dtype_bytes);
        kv = 2 * H * L * total_tokens * D;
    }

    const vsize = modelVocabSize(pipeline.model);

    return .{
        .allocator = allocator,
        .corpus_bytes = scan.bytes,
        .corpus_codepoints = scan.codepoints,
        .corpus_words = scan.words,
        .corpus_lines = scan.lines,
        .total_tokens = total_tokens,
        .vocab_size = vsize,
        .unique_tokens_used = unique_tokens_used,
        .bytes_per_token = bpt,
        .chars_per_token = cpt,
        .tokens_per_word = tpw,
        .tokens_per_line = tpl,
        .fallback_rate = fbr,
        .top_k_ids = top_ids,
        .top_k_counts = top_counts,
        .bottom_k_ids = bot_ids,
        .bottom_k_counts = bot_counts,
        .script_metrics = script_metrics,
        .kv_cache_bytes = kv,
    };
}

// ---------------------------------------------------------------------------
// Multilingual fairness (per-script proxy)
// ---------------------------------------------------------------------------
//
// The fairness metric in `fairness.zig` consumes pre-counted per-language
// (tokens, units) tallies. We do not have true per-language attribution, but
// `Report.script_metrics` already carries per-*script* token and codepoint
// counts (Latin/Han/Cyrillic/…). Treating each script as a "language" gives a
// reasonable multilingual fairness proxy: a tokenizer that shreds Han or
// Devanagari into many tokens while compressing Latin shows up as a high Gini
// / max-min ratio. Note this is per-script, not per-true-language.

/// Build the `[]fairness.LangStat` view over a report's per-script metrics.
/// Scripts with zero codepoints are skipped (they carry no information and
/// `fairness.analyze` would drop them anyway). The returned slice borrows the
/// script-name strings from `report`, so it stays valid only as long as the
/// report does; free the slice itself with `allocator.free`.
pub fn scriptLangStats(allocator: std.mem.Allocator, report: *const Report) ![]fairness.LangStat {
    var list: std.ArrayList(fairness.LangStat) = .empty;
    errdefer list.deinit(allocator);
    for (report.script_metrics) |sm| {
        if (sm.codepoints == 0) continue;
        try list.append(allocator, .{
            .name = sm.script_name,
            .tokens = sm.tokens,
            .units = sm.codepoints,
        });
    }
    return list.toOwnedSlice(allocator);
}

/// Render a human-readable per-script fairness block for `report` to `out`.
/// Allocates a temporary `LangStat` slice and a `fairness.Report`; both are
/// freed before returning. When no script carries codepoints, prints a short
/// notice instead of a table.
pub fn writeFairnessText(
    allocator: std.mem.Allocator,
    out: *std.Io.Writer,
    report: *const Report,
) !void {
    const stats = try scriptLangStats(allocator, report);
    defer allocator.free(stats);

    try out.writeAll("\n  fairness (per-script multilingual proxy):\n");

    var fr = fairness.analyze(allocator, stats) catch |err| switch (err) {
        error.NoData => {
            try out.writeAll("    (no per-script data — corpus had no countable codepoints)\n");
            return;
        },
        else => return err,
    };
    defer fr.deinit(allocator);

    try out.print("    languages (scripts): {d}\n", .{fr.count()});
    try out.print("    gini:                {d:.4}  (0 = perfectly fair)\n", .{fr.gini});
    try out.print("    max_min_ratio:       {d:.4}  (worst/best fertility)\n", .{fr.max_min_ratio});
    try out.print("    mean_fertility:      {d:.4}\n", .{fr.mean_fertility});
    try out.print("    best:  {s:<28} fertility={d:.4}\n", .{ fr.best().name, fr.best_fertility });
    try out.print("    worst: {s:<28} fertility={d:.4}\n", .{ fr.worst().name, fr.worst_fertility });
    try out.writeAll("\n    per-script fertility / premium (premium = x vs best):\n");
    for (fr.langs) |l| {
        try out.print("      {s:<32}  fertility={d:.4}  premium={d:.2}x\n", .{
            l.name, l.fertility, l.premium,
        });
    }
}

/// Render the fairness block as a single standalone JSON object terminated by
/// a newline. (The main `eval --format json` object is emitted by the CLI
/// layer and is already closed, so the fairness payload is emitted as its own
/// object line rather than nested inside it.)
pub fn writeFairnessJson(
    allocator: std.mem.Allocator,
    out: *std.Io.Writer,
    report: *const Report,
) !void {
    const stats = try scriptLangStats(allocator, report);
    defer allocator.free(stats);

    var fr = fairness.analyze(allocator, stats) catch |err| switch (err) {
        error.NoData => {
            try out.writeAll("{\"fairness\":null}\n");
            return;
        },
        else => return err,
    };
    defer fr.deinit(allocator);

    try out.writeAll("{\"fairness\":{\"basis\":\"per-script\",");
    try out.print("\"languages\":{d},", .{fr.count()});
    try out.print("\"gini\":{d:.6},", .{fr.gini});
    try out.print("\"max_min_ratio\":{d:.6},", .{fr.max_min_ratio});
    try out.print("\"mean_fertility\":{d:.6},", .{fr.mean_fertility});
    try out.print("\"best_fertility\":{d:.6},", .{fr.best_fertility});
    try out.print("\"worst_fertility\":{d:.6},", .{fr.worst_fertility});
    try out.writeAll("\"scripts\":[");
    for (fr.langs, 0..) |l, i| {
        if (i > 0) try out.writeAll(",");
        try out.writeAll("{\"name\":");
        try writeJsonStringEscaped(out, l.name);
        try out.print(",\"tokens\":{d},\"units\":{d},\"fertility\":{d:.6},\"premium\":{d:.6}}}", .{
            l.tokens, l.units, l.fertility, l.premium,
        });
    }
    try out.writeAll("]}}\n");
}

fn writeJsonStringEscaped(out: *std.Io.Writer, s: []const u8) !void {
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

const Vocab = @import("vocab.zig").Vocab;

test "evaluate a tiny ASCII corpus" {
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };
    const corpus = "hello world hello world";
    var r = try evaluate(std.testing.allocator, &pipe, corpus, .{});
    defer r.deinit();
    try std.testing.expectEqual(@as(u64, corpus.len), r.total_tokens);
    try std.testing.expectEqual(@as(f64, 1.0), r.bytes_per_token);
}

test "vocab_size is reported" {
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };
    var r = try evaluate(std.testing.allocator, &pipe, "abc", .{});
    defer r.deinit();
    try std.testing.expectEqual(@as(u32, 256), r.vocab_size);
}

test "tokens_per_word is reasonable" {
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };
    const corpus = "hello world hello world";
    var r = try evaluate(std.testing.allocator, &pipe, corpus, .{});
    defer r.deinit();
    try std.testing.expectEqual(@as(u64, 4), r.corpus_words);
    try std.testing.expectApproxEqAbs(@as(f64, 5.75), r.tokens_per_word, 0.001);
}

test "script_metrics buckets ASCII correctly" {
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };
    var r = try evaluate(std.testing.allocator, &pipe, "hello world hello world", .{});
    defer r.deinit();
    var found = false;
    for (r.script_metrics) |sm| {
        if (std.mem.eql(u8, sm.script_name, "Latin (ASCII)")) {
            found = true;
            try std.testing.expect(sm.codepoints > 0);
        }
    }
    try std.testing.expect(found);
}

test "per-script tokens reflect real per-run encoding, not proportional split" {
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };
    // Latin "hi" (2 cp / 2 bytes), Cyrillic "мир" (3 cp / 6 bytes),
    // Han "字符" (2 cp / 6 bytes). Spaces are neutral and attach to the
    // current (preceding) run.
    const corpus = "hi мир 字符";
    var r = try evaluate(std.testing.allocator, &pipe, corpus, .{});
    defer r.deinit();

    var latin: ?ScriptMetric = null;
    var cyr: ?ScriptMetric = null;
    var han: ?ScriptMetric = null;
    for (r.script_metrics) |sm| {
        if (std.mem.eql(u8, sm.script_name, "Latin (ASCII)")) latin = sm;
        if (std.mem.eql(u8, sm.script_name, "Cyrillic")) cyr = sm;
        if (std.mem.eql(u8, sm.script_name, "CJK Unified Ideographs")) han = sm;
    }
    try std.testing.expect(latin != null);
    try std.testing.expect(cyr != null);
    try std.testing.expect(han != null);

    // byte_id => one token per byte of each run. The Latin run swallows the
    // trailing space ("hi " = 3 bytes), Cyrillic swallows its trailing space
    // ("мир " = 7 bytes), Han is the tail ("字符" = 6 bytes).
    try std.testing.expectEqual(@as(u64, 3), latin.?.tokens);
    try std.testing.expectEqual(@as(u64, 7), cyr.?.tokens);
    try std.testing.expectEqual(@as(u64, 6), han.?.tokens);

    // Codepoints (incl. swallowed spaces): Latin 3, Cyrillic 4, Han 2.
    try std.testing.expectEqual(@as(u64, 3), latin.?.codepoints);
    try std.testing.expectEqual(@as(u64, 4), cyr.?.codepoints);
    try std.testing.expectEqual(@as(u64, 2), han.?.codepoints);

    // The PROPORTIONAL v1 split would have assigned tokens by codepoint
    // share of total_tokens (16). Han's proportional share would be
    // 16 * 2/9 ≈ 3.6 -> 4 tokens; the real per-run count is 6. Assert the
    // real value diverges from that placeholder for Han.
    const han_proportional: u64 = @intFromFloat(@round(
        @as(f64, @floatFromInt(r.total_tokens)) *
            @as(f64, @floatFromInt(han.?.codepoints)) /
            @as(f64, @floatFromInt(r.corpus_codepoints)),
    ));
    try std.testing.expect(han.?.tokens != han_proportional);

    // Fertility ordering: multibyte scripts (1 token per byte, many bytes
    // per cp) must be more "fertile" (higher tokens/cp) than ASCII Latin.
    try std.testing.expect(han.?.fertility > latin.?.fertility);
    try std.testing.expect(cyr.?.fertility > latin.?.fertility);
    // Han packs 3 bytes/cp with no swallowed space -> highest fertility.
    try std.testing.expect(han.?.fertility >= cyr.?.fertility);
}

test "computeCost: 1M tokens at $0.50 per 1M = $0.50" {
    const c = computeCost(1_000_000, 0.50, .per_1m);
    try std.testing.expectApproxEqAbs(@as(f64, 0.50), c, 1e-9);
}

test "computeCost: 5M tokens at $1.00 per 1M = $5.00" {
    const c = computeCost(5_000_000, 1.00, .per_1m);
    try std.testing.expectApproxEqAbs(@as(f64, 5.00), c, 1e-9);
}

test "computeCost: 1K tokens at $0.001 per 1K = $0.001" {
    const c = computeCost(1_000, 0.001, .per_1k);
    try std.testing.expectApproxEqAbs(@as(f64, 0.001), c, 1e-12);
}

test "computeCost: integer overflow safety at 100B tokens" {
    // 100B tokens at $1 per 1M = $100_000.00. The intermediate
    // tokens*price would overflow u64 dollars*tokens but we work in f64.
    const c = computeCost(100_000_000_000, 1.00, .per_1m);
    try std.testing.expectApproxEqAbs(@as(f64, 100_000.0), c, 1e-3);
}

test "kv_cache_bytes computed when dims provided" {
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };
    const corpus = "hello world hello world";
    var r = try evaluate(std.testing.allocator, &pipe, corpus, .{
        .model_hidden_size = 4096,
        .model_num_layers = 32,
        .model_dtype_bytes = 2,
    });
    defer r.deinit();
    const expected: u64 = 2 * 4096 * 32 * @as(u64, corpus.len) * 2;
    try std.testing.expectEqual(expected, r.kv_cache_bytes.?);
}

test "scriptLangStats extracts per-script tokens/units and skips zero-cp scripts" {
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };
    // Mixed Latin + Hiragana so at least two scripts appear.
    var r = try evaluate(std.testing.allocator, &pipe, "hello \xe3\x81\x82\xe3\x81\x84 world", .{});
    defer r.deinit();

    const stats = try scriptLangStats(std.testing.allocator, &r);
    defer std.testing.allocator.free(stats);

    // Every emitted stat must mirror a non-zero-codepoint script_metric.
    try std.testing.expect(stats.len > 0);
    try std.testing.expect(stats.len <= r.script_metrics.len);
    for (stats) |s| {
        try std.testing.expect(s.units > 0);
        var matched = false;
        for (r.script_metrics) |sm| {
            if (std.mem.eql(u8, sm.script_name, s.name)) {
                try std.testing.expectEqual(sm.codepoints, s.units);
                try std.testing.expectEqual(sm.tokens, s.tokens);
                matched = true;
            }
        }
        try std.testing.expect(matched);
    }
}

test "fairness integration: report builds and is internally consistent" {
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };
    var r = try evaluate(std.testing.allocator, &pipe, "hello \xe3\x81\x82\xe3\x81\x84 world", .{});
    defer r.deinit();

    const stats = try scriptLangStats(std.testing.allocator, &r);
    defer std.testing.allocator.free(stats);

    var fr = try fairness.analyze(std.testing.allocator, stats);
    defer fr.deinit(std.testing.allocator);

    try std.testing.expectEqual(stats.len, fr.count());
    try std.testing.expect(fr.gini >= 0.0);
    try std.testing.expect(fr.max_min_ratio >= 1.0 - 1e-9);
    try std.testing.expect(fr.best_fertility <= fr.worst_fertility + 1e-9);
    // The baseline language always reads as premium 1.0.
    try std.testing.expect(@abs(fr.best().premium - 1.0) <= 1e-9);
}

test "writeFairnessText emits the expected headers" {
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };
    var r = try evaluate(std.testing.allocator, &pipe, "hello \xe3\x81\x82 world", .{});
    defer r.deinit();

    var buf: [8192]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try writeFairnessText(std.testing.allocator, &w, &r);
    const s = w.buffered();

    inline for ([_][]const u8{
        "fairness (per-script multilingual proxy)",
        "gini:",
        "max_min_ratio:",
        "premium=",
    }) |needle| {
        try std.testing.expect(std.mem.indexOf(u8, s, needle) != null);
    }
}

test "writeFairnessJson emits a parseable fairness object" {
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };
    var r = try evaluate(std.testing.allocator, &pipe, "hello \xe3\x81\x82 world", .{});
    defer r.deinit();

    var buf: [8192]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try writeFairnessJson(std.testing.allocator, &w, &r);

    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, w.buffered(), .{});
    defer parsed.deinit();
    const obj = parsed.value.object.get("fairness").?.object;
    try std.testing.expect(obj.get("gini").? == .float or obj.get("gini").? == .integer);
    try std.testing.expectEqualStrings("per-script", obj.get("basis").?.string);
    const scripts = obj.get("scripts").?.array;
    try std.testing.expect(scripts.items.len > 0);
}
