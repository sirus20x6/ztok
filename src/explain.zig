//! Tokenization debugger: `ztok explain`.
//!
//! For a given input + vocab, produce a per-token breakdown answering
//! three questions:
//!
//!   1. WHAT each token is — its id, the bytes/text it decodes to, the
//!      half-open byte span `[start, end)` in the ORIGINAL input it
//!      covers (via `Pipeline.encodeWithOffsets`), and its byte length.
//!
//!   2. WHY it was chosen — the encoder's per-step decision record. We
//!      reuse the existing `trace.zig` sink rather than re-deriving the
//!      logic: BPE emits its merge sequence, Unigram emits the chosen
//!      Viterbi piece + path score, Monster emits the winning branch +
//!      score (the same record shape the `ZTOK_MONSTER_TRACE=1` tracer
//!      prints). The pipeline is run a second time with a `Trace` sink
//!      pointed at an in-memory buffer, so `explain` carries no encoder
//!      knowledge of its own — it just captures and presents.
//!
//!   3. What it COSTS — total tokens, bytes-per-token, and the least
//!      byte-efficient spans (fewest input bytes per token), sorted so
//!      the worst offenders surface first.
//!
//! Output is human-readable by default; `--format json` emits a
//! structured object that round-trips through `std.json`.

const std = @import("std");

const pipeline = @import("pipeline.zig");
const model_mod = @import("model.zig");
const trace_mod = @import("trace.zig");
const token = @import("token.zig");

const Pipeline = pipeline.Pipeline;
const Model = model_mod.Model;
const Trace = trace_mod.Trace;
const TokenId = token.TokenId;
const Span = token.Span;

/// Which encoder produced this explanation. Determines how the "why"
/// section is labelled and how the trace records are interpreted.
pub const ModelKind = enum {
    bpe,
    unigram,
    wordpiece,
    monster,

    pub fn name(self: ModelKind) []const u8 {
        return switch (self) {
            .bpe => "bpe",
            .unigram => "unigram",
            .wordpiece => "wordpiece",
            .monster => "monster",
        };
    }

    /// One-line description of the decision strategy, shown in the "why"
    /// header so the reader knows what the trace records mean.
    pub fn strategy(self: ModelKind) []const u8 {
        return switch (self) {
            .bpe => "byte-level BPE: greedily fuse the highest-priority adjacent pair until none remain",
            .unigram => "unigram LM: Viterbi over piece scores picks the max-likelihood segmentation",
            .wordpiece => "wordpiece: longest-match-first over the vocab, falling back to [UNK]",
            .monster => "TokenMonster: 2-branch ungreedy lookahead picks the branch with the better score",
        };
    }
};

/// Derive the `ModelKind` tag from a `Model` union value.
pub fn modelKindOf(m: Model) ModelKind {
    return switch (m) {
        .byte_id => .bpe, // byte_id is a degenerate BPE (no merges)
        .bpe => .bpe,
        .unigram => .unigram,
        .wordpiece => .wordpiece,
        .monster => .monster,
    };
}

/// A single token's WHAT view.
pub const TokenRow = struct {
    /// Position in the output id stream.
    index: usize,
    /// Vocab id.
    id: TokenId,
    /// Decoded bytes for this id (owned by the Explanation arena).
    text: []const u8,
    /// Byte span in the ORIGINAL input.
    span: Span,

    /// Input bytes this token covers (span width). May differ from
    /// `text.len` for byte-fallback / lilbuf pieces; the cost view uses
    /// THIS (input bytes), since that is what the user paid for.
    pub fn byteLen(self: TokenRow) u32 {
        return self.span.len();
    }
};

/// Cost view: aggregate efficiency plus the least byte-efficient spans.
pub const CostView = struct {
    total_tokens: usize,
    total_bytes: usize,
    /// bytes / tokens — higher is more efficient (more input compressed
    /// per emitted token). Zero tokens → 0.
    bytes_per_token: f64,
    /// Indices into `Explanation.tokens`, sorted least-efficient first
    /// (smallest input-byte span, ties broken by earlier index). Owned
    /// by the Explanation arena.
    least_efficient: []const usize,
};

/// The full explanation. Owns an arena holding all token text, the
/// trace buffer, and the cost-view index list. Call `deinit`.
pub const Explanation = struct {
    arena: std.heap.ArenaAllocator,
    input: []const u8,
    kind: ModelKind,
    tokens: []TokenRow,
    /// Raw per-step decision records captured from the `Trace` sink, in
    /// encoder-native line format (see `trace.zig`). This is the "why".
    why: []const u8,
    cost: CostView,

    pub fn deinit(self: *Explanation) void {
        self.arena.deinit();
    }
};

/// Build an `Explanation` for `input` under `pipe`.
///
/// Runs the pipeline twice: once via `encodeWithOffsets` for the id +
/// span breakdown, once via `encode` with a `Trace` sink for the "why"
/// records. Both runs are deterministic on the same input, so the two
/// views describe the same encoding.
pub fn explain(
    gpa: std.mem.Allocator,
    pipe: *const Pipeline,
    input: []const u8,
) !Explanation {
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();

    const owned_input = try a.dupe(u8, input);

    // --- WHAT: ids + original-input byte spans ---------------------
    var enc = try pipe.encodeWithOffsets(gpa, input);
    defer enc.deinit(gpa);

    var rows = try a.alloc(TokenRow, enc.ids.len);
    var byte_scratch: [1]u8 = undefined;
    for (enc.ids, enc.offsets, 0..) |id, span, i| {
        const decoded = pipe.model.idBytes(id, pipe.vocab, &byte_scratch);
        rows[i] = .{
            .index = i,
            .id = id,
            .text = try a.dupe(u8, decoded),
            .span = span,
        };
    }

    // --- WHY: capture encoder decision records via the Trace sink ---
    var why_buf: std.Io.Writer.Allocating = .init(a);
    var trace_sink: Trace = .{ .writer = &why_buf.writer };

    var traced = pipe.*;
    traced.trace = &trace_sink;
    const ids2 = try traced.encode(gpa, input);
    gpa.free(ids2);
    const why = why_buf.written();

    // --- COST: bytes/token + least-efficient spans -----------------
    var total_bytes: usize = 0;
    for (rows) |r| total_bytes += r.byteLen();

    const order = try a.alloc(usize, rows.len);
    for (order, 0..) |*o, i| o.* = i;
    std.sort.pdq(usize, order, rows, lessEfficient);

    const bpt: f64 = if (rows.len == 0)
        0
    else
        @as(f64, @floatFromInt(total_bytes)) / @as(f64, @floatFromInt(rows.len));

    return .{
        .arena = arena,
        .input = owned_input,
        .kind = modelKindOf(pipe.model),
        .tokens = rows,
        .why = why,
        .cost = .{
            .total_tokens = rows.len,
            .total_bytes = total_bytes,
            .bytes_per_token = bpt,
            .least_efficient = order,
        },
    };
}

/// Sort predicate: smaller input-byte span = less efficient = sorts
/// first. Ties broken by earlier index for stable, reproducible output.
fn lessEfficient(rows: []const TokenRow, lhs: usize, rhs: usize) bool {
    const a = rows[lhs].byteLen();
    const b = rows[rhs].byteLen();
    if (a != b) return a < b;
    return lhs < rhs;
}

pub const Format = enum { text, json };

/// Render `exp` to `out` in the requested format.
pub fn render(exp: *const Explanation, fmt: Format, out: *std.Io.Writer) !void {
    switch (fmt) {
        .text => try renderText(exp, out),
        .json => try renderJson(exp, out),
    }
}

fn renderText(exp: *const Explanation, out: *std.Io.Writer) !void {
    try out.print("== ztok explain ({s}) ==\n", .{exp.kind.name()});
    try out.print("input: ", .{});
    try writeQuoted(out, exp.input);
    try out.print(" ({d} bytes)\n\n", .{exp.input.len});

    // 1. Token breakdown.
    try out.writeAll("[tokens]\n");
    try out.writeAll("  #   id        span        len  text\n");
    for (exp.tokens) |r| {
        try out.print(
            "  {d:<3} {d:<9} {d:>4}..{d:<4} {d:>4}  ",
            .{ r.index, r.id, r.span.start, r.span.end, r.byteLen() },
        );
        try writeQuoted(out, r.text);
        try out.writeAll("\n");
    }
    try out.writeAll("\n");

    // 2. Why chosen.
    try out.print("[why] {s}\n", .{exp.kind.strategy()});
    if (exp.why.len == 0) {
        try out.writeAll("  (no decision records — encoder emitted none for this input)\n");
    } else {
        var it = std.mem.splitScalar(u8, exp.why, '\n');
        while (it.next()) |line| {
            if (line.len == 0) continue;
            try out.print("  {s}\n", .{line});
        }
    }
    try out.writeAll("\n");

    // 3. Cost view.
    try out.writeAll("[cost]\n");
    try out.print("  total tokens:    {d}\n", .{exp.cost.total_tokens});
    try out.print("  total bytes:     {d}\n", .{exp.cost.total_bytes});
    try out.print("  bytes/token:     {d:.3}\n", .{exp.cost.bytes_per_token});
    try out.writeAll("  least efficient (input bytes per token, worst first):\n");
    const show = @min(exp.cost.least_efficient.len, 5);
    for (exp.cost.least_efficient[0..show]) |idx| {
        const r = exp.tokens[idx];
        try out.print("    #{d:<3} {d:>2} byte(s)  ", .{ r.index, r.byteLen() });
        try writeQuoted(out, r.text);
        try out.writeAll("\n");
    }
}

fn renderJson(exp: *const Explanation, out: *std.Io.Writer) !void {
    try out.writeAll("{");
    try out.print("\"model\":\"{s}\",", .{exp.kind.name()});
    try out.writeAll("\"strategy\":");
    try writeJsonString(out, exp.kind.strategy());
    try out.writeAll(",\"input\":");
    try writeJsonString(out, exp.input);
    try out.print(",\"input_bytes\":{d},", .{exp.input.len});

    // tokens
    try out.writeAll("\"tokens\":[");
    for (exp.tokens, 0..) |r, i| {
        if (i > 0) try out.writeAll(",");
        try out.print(
            "{{\"index\":{d},\"id\":{d},\"span\":[{d},{d}],\"byte_len\":{d},\"text\":",
            .{ r.index, r.id, r.span.start, r.span.end, r.byteLen() },
        );
        try writeJsonString(out, r.text);
        try out.writeAll("}");
    }
    try out.writeAll("],");

    // why — array of trace record lines (verbatim, encoder-native).
    try out.writeAll("\"why\":[");
    {
        var it = std.mem.splitScalar(u8, exp.why, '\n');
        var first = true;
        while (it.next()) |line| {
            if (line.len == 0) continue;
            if (!first) try out.writeAll(",");
            first = false;
            try writeJsonString(out, line);
        }
    }
    try out.writeAll("],");

    // cost
    try out.print(
        "\"cost\":{{\"total_tokens\":{d},\"total_bytes\":{d},\"bytes_per_token\":{d:.6},\"least_efficient\":[",
        .{ exp.cost.total_tokens, exp.cost.total_bytes, exp.cost.bytes_per_token },
    );
    for (exp.cost.least_efficient, 0..) |idx, i| {
        if (i > 0) try out.writeAll(",");
        const r = exp.tokens[idx];
        try out.print("{{\"index\":{d},\"byte_len\":{d}}}", .{ r.index, r.byteLen() });
    }
    try out.writeAll("]}}");
}

/// Quote bytes for the text report (printable passthrough, escapes for
/// control / non-printable). Matches `trace.zig`'s convention.
fn writeQuoted(w: *std.Io.Writer, bytes: []const u8) !void {
    try w.writeAll("\"");
    for (bytes) |b| {
        switch (b) {
            '\\' => try w.writeAll("\\\\"),
            '"' => try w.writeAll("\\\""),
            '\n' => try w.writeAll("\\n"),
            '\r' => try w.writeAll("\\r"),
            '\t' => try w.writeAll("\\t"),
            0x20...0x21, 0x23...0x5B, 0x5D...0x7E => try w.writeByte(b),
            else => try w.print("\\x{x:0>2}", .{b}),
        }
    }
    try w.writeAll("\"");
}

/// Strict JSON string emitter (RFC 8259): escapes control chars as
/// \u00NN, escapes quote/backslash, passes other bytes through (UTF-8
/// stays valid; lone bytes are rare but tolerated by most parsers — we
/// keep them raw rather than risk malformed \u for non-codepoint bytes).
fn writeJsonString(w: *std.Io.Writer, bytes: []const u8) !void {
    try w.writeAll("\"");
    for (bytes) |b| {
        switch (b) {
            '\\' => try w.writeAll("\\\\"),
            '"' => try w.writeAll("\\\""),
            '\n' => try w.writeAll("\\n"),
            '\r' => try w.writeAll("\\r"),
            '\t' => try w.writeAll("\\t"),
            0x08 => try w.writeAll("\\b"),
            0x0c => try w.writeAll("\\f"),
            0x00...0x07, 0x0b, 0x0e...0x1f => try w.print("\\u{x:0>4}", .{b}),
            else => try w.writeByte(b),
        }
    }
    try w.writeAll("\"");
}

// ---------------------------------------------------------------- tests

const root = @import("root.zig");

/// Build a tiny tiktoken-format BPE over `h e l o` as single bytes plus
/// the `he`, `ll`, `llo` merges, so "hello" exercises real merges.
fn buildHelloBpe(gpa: std.mem.Allocator) !root.Bpe {
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(gpa);
    const enc = std.base64.standard.Encoder;
    var b64: [32]u8 = undefined;
    inline for (.{ "h", "e", "l", "o" }, 0..) |s, id| {
        const e = enc.encode(&b64, s);
        try buf.print(gpa, "{s} {d}\n", .{ e, id });
    }
    inline for (.{ "he", "ll", "llo" }, 4..) |s, id| {
        const e = enc.encode(&b64, s);
        try buf.print(gpa, "{s} {d}\n", .{ e, id });
    }
    return root.Bpe.loadTiktokenBytes(gpa, buf.items);
}

fn helloPipeline(bpe: *const root.Bpe, v: *const root.Vocab) Pipeline {
    return .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .{ .bpe = bpe },
        .decoder = .concat,
        .vocab = v,
    };
}

test "explain: BPE hello — ids, spans, lengths sum to input length" {
    const gpa = std.testing.allocator;
    var bpe = try buildHelloBpe(gpa);
    defer bpe.deinit();
    var v = root.Vocab.empty(gpa);
    defer v.deinit();
    const pipe = helloPipeline(&bpe, &v);

    var exp = try explain(gpa, &pipe, "hello");
    defer exp.deinit();

    try std.testing.expect(exp.kind == .bpe);
    try std.testing.expect(exp.tokens.len > 0);

    // Span byte-accuracy: spans tile the input contiguously and the
    // total byte length equals the input length.
    var total: u32 = 0;
    var cursor: u32 = 0;
    for (exp.tokens) |r| {
        try std.testing.expectEqual(cursor, r.span.start);
        try std.testing.expect(r.span.end >= r.span.start);
        cursor = r.span.end;
        total += r.byteLen();
        try std.testing.expect(r.id < bpe.count);
    }
    try std.testing.expectEqual(@as(u32, 5), total);
    try std.testing.expectEqual(@as(u32, 5), cursor);
    try std.testing.expectEqual(@as(usize, 5), exp.cost.total_bytes);

    // The "why" section should contain BPE merge records (we asked for
    // merges via the trace sink, and "hello" forces at least one).
    try std.testing.expect(std.mem.indexOf(u8, exp.why, "bpe merge") != null);
}

test "explain: text render is well-formed (has all three sections)" {
    const gpa = std.testing.allocator;
    var bpe = try buildHelloBpe(gpa);
    defer bpe.deinit();
    var v = root.Vocab.empty(gpa);
    defer v.deinit();
    const pipe = helloPipeline(&bpe, &v);

    var exp = try explain(gpa, &pipe, "hello");
    defer exp.deinit();

    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try render(&exp, .text, &aw.writer);
    const s = aw.written();

    try std.testing.expect(std.mem.indexOf(u8, s, "[tokens]") != null);
    try std.testing.expect(std.mem.indexOf(u8, s, "[why]") != null);
    try std.testing.expect(std.mem.indexOf(u8, s, "[cost]") != null);
    try std.testing.expect(std.mem.indexOf(u8, s, "bytes/token:") != null);
}

test "explain: json render parses and round-trips key fields" {
    const gpa = std.testing.allocator;
    var bpe = try buildHelloBpe(gpa);
    defer bpe.deinit();
    var v = root.Vocab.empty(gpa);
    defer v.deinit();
    const pipe = helloPipeline(&bpe, &v);

    var exp = try explain(gpa, &pipe, "hello");
    defer exp.deinit();

    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try render(&exp, .json, &aw.writer);
    const s = aw.written();

    var parsed = try std.json.parseFromSlice(std.json.Value, gpa, s, .{});
    defer parsed.deinit();
    const obj = parsed.value.object;

    try std.testing.expectEqualStrings("bpe", obj.get("model").?.string);
    try std.testing.expectEqual(@as(i64, 5), obj.get("input_bytes").?.integer);

    const toks = obj.get("tokens").?.array;
    try std.testing.expectEqual(exp.tokens.len, toks.items.len);

    const cost = obj.get("cost").?.object;
    try std.testing.expectEqual(
        @as(i64, @intCast(exp.tokens.len)),
        cost.get("total_tokens").?.integer,
    );
    try std.testing.expectEqual(@as(i64, 5), cost.get("total_bytes").?.integer);

    // "why" must be a JSON array (the captured trace records).
    try std.testing.expect(obj.get("why").? == .array);
}

test "explain: cost view flags the least byte-efficient token" {
    const gpa = std.testing.allocator;
    // Craft a vocab where "ab" is a 2-byte merge but "c" stays a lone
    // single-byte token, so on input "abc" the "c" token is the least
    // byte-efficient (1 input byte) and must sort first.
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(gpa);
    const b64enc = std.base64.standard.Encoder;
    var b64: [32]u8 = undefined;
    inline for (.{ "a", "b", "c" }, 0..) |s, id| {
        const e = b64enc.encode(&b64, s);
        try buf.print(gpa, "{s} {d}\n", .{ e, id });
    }
    const e = b64enc.encode(&b64, "ab");
    try buf.print(gpa, "{s} {d}\n", .{ e, 3 });

    var bpe = try root.Bpe.loadTiktokenBytes(gpa, buf.items);
    defer bpe.deinit();
    var v = root.Vocab.empty(gpa);
    defer v.deinit();
    const pipe = helloPipeline(&bpe, &v);

    var exp = try explain(gpa, &pipe, "abc");
    defer exp.deinit();

    // Expect two tokens: "ab" (2 bytes) and "c" (1 byte).
    try std.testing.expectEqual(@as(usize, 2), exp.tokens.len);
    try std.testing.expectEqual(@as(usize, 3), exp.cost.total_bytes);

    // Least-efficient first must be the 1-byte "c" token.
    const worst = exp.tokens[exp.cost.least_efficient[0]];
    try std.testing.expectEqual(@as(u32, 1), worst.byteLen());
    try std.testing.expectEqualStrings("c", worst.text);
}
