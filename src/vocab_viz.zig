//! Self-contained HTML visualization for a tokenizer vocab.
//!
//! Used by `ztok visualize <vocab>` to produce a single HTML file
//! showing:
//!   1. Top-K token grid (frequency from a corpus, log-scaled).
//!   2. 256-cell byte-distribution heatmap (per-byte token coverage).
//!   3. Token-length histogram (bucketed 1/2/3/4/5-8/9-16/17+).
//!   4. BPE merge dendrogram (first 50 merges, inferred from id order).
//!   5. Added/special token table (HF tokenizer.json only).
//!
//! Output is a single HTML file with inline CSS/JS — no network deps,
//! no React/Vue, target < 200 KB for typical vocabs. The renderer is
//! a pure-Zig string builder; data arrays are serialized to inline
//! JSON for vanilla JS to consume.

const std = @import("std");
const TokenId = @import("token.zig").TokenId;
const Bpe = @import("bpe.zig").Bpe;
const Unigram = @import("unigram.zig").Unigram;
const WordPiece = @import("wordpiece.zig").WordPiece;
const Monster = @import("monster.zig").Monster;
const RwkvWorld = @import("rwkv_world.zig").RwkvWorld;
const Pipeline = @import("pipeline.zig").Pipeline;
const Vocab = @import("vocab.zig").Vocab;
const added_tokens_mod = @import("added_tokens.zig");
const hf_json = @import("hf_json.zig");

/// Discriminated reference to a loaded model. Mirrors
/// `cli_validate.LoadedModel` so the CLI can pass either one through.
pub const ModelRef = union(enum) {
    bpe: *const Bpe,
    unigram: *const Unigram,
    wordpiece: *const WordPiece,
    monster: *const Monster,
    rwkv_world: *const RwkvWorld,

    pub fn count(self: ModelRef) u32 {
        return switch (self) {
            .bpe => |b| b.count,
            .unigram => |u| u.count,
            .wordpiece => |w| w.count,
            .monster => |m| m.count,
            .rwkv_world => |r| r.count,
        };
    }

    pub fn tokenBytes(self: ModelRef, id: u32) []const u8 {
        return switch (self) {
            .bpe => |b| b.bytes[b.offsets[id]..b.offsets[id + 1]],
            .unigram => |u| u.bytes[u.offsets[id]..u.offsets[id + 1]],
            .wordpiece => |w| w.bytes[w.offsets[id]..w.offsets[id + 1]],
            .monster => |m| m.bytes[m.offsets[id]..m.offsets[id + 1]],
            .rwkv_world => |r| r.bytes[r.offsets[id]..r.offsets[id + 1]],
        };
    }

    pub fn kindName(self: ModelRef) []const u8 {
        return switch (self) {
            .bpe => "bpe",
            .unigram => "unigram",
            .wordpiece => "wordpiece",
            .monster => "monster",
            .rwkv_world => "rwkv_world",
        };
    }
};

pub const Options = struct {
    /// Tokenizer file path, displayed in the page title.
    title: []const u8,
    /// Top-K cap for the frequency grid. Capped at vocab size.
    top_k: u32 = 200,
    /// Optional corpus bytes to encode for the frequency grid. When
    /// `null`, the bundled mini-corpus (`MINI_CORPUS`) is used.
    corpus: ?[]const u8 = null,
    /// Maximum number of BPE merges to render in the dendrogram (one
    /// SVG per merge; bigger numbers blow up the file size).
    max_merges: u32 = 50,
    /// Optional HF added-token list (id + content + lstrip/rstrip
    /// flags). Populated by the CLI when the source vocab is an HF
    /// tokenizer.json; left null for tiktoken / .ztm / SentencePiece.
    added_tokens: []const AddedTokenView = &.{},
};

pub const AddedTokenView = struct {
    id: u32,
    content: []const u8,
    lstrip: bool = false,
    rstrip: bool = false,
    single_word: bool = false,
};

/// Small English+code mini-corpus used when the caller doesn't
/// provide one. Enough variety to give the top-K grid a realistic
/// shape on byte-level BPE and SP vocabs.
pub const MINI_CORPUS: []const u8 =
    "The quick brown fox jumps over the lazy dog.\n" ++
    "She sells seashells by the seashore.\n" ++
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?\n" ++
    "To be, or not to be: that is the question.\n" ++
    "It was the best of times, it was the worst of times.\n" ++
    "All happy families are alike; each unhappy family is unhappy in its own way.\n" ++
    "In the beginning was the Word, and the Word was with God.\n" ++
    "Call me Ishmael. Some years ago—never mind how long precisely—\n" ++
    "It is a truth universally acknowledged, that a single man in possession\n" ++
    "of a good fortune, must be in want of a wife.\n" ++
    "fn main() void { std.debug.print(\"hello, world!\\n\", .{}); }\n" ++
    "const x: i32 = 42; var arr = [_]u8{1,2,3,4,5};\n" ++
    "def fibonacci(n: int) -> int:\n    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)\n" ++
    "class Foo:\n    def __init__(self, x): self.x = x\n" ++
    "SELECT id, name, COUNT(*) FROM users WHERE active = 1 GROUP BY country;\n" ++
    "<html><body><h1>Hello</h1><p>Paragraph of text.</p></body></html>\n" ++
    "{\"key\": \"value\", \"list\": [1, 2, 3], \"nested\": {\"a\": true}}\n" ++
    "# Markdown heading\n## Sub heading\n- bullet one\n- bullet two\n" ++
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod\n" ++
    "tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam,\n" ++
    "quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo.\n";

// ---------------------------------------------------------------------
// Public entry point.

/// Render the full HTML page to `writer`. Caller is responsible for
/// flushing. The page is fully self-contained: no external CSS/JS, no
/// fonts, no images. SVG is inlined.
pub fn render(
    allocator: std.mem.Allocator,
    writer: *std.Io.Writer,
    model: ModelRef,
    pipeline: *const Pipeline,
    opts: Options,
) !void {
    // Compute everything up front so the HTML body can stream in order.
    const corpus_bytes = opts.corpus orelse MINI_CORPUS;
    const freqs = try computeTokenFrequencies(allocator, pipeline, corpus_bytes);
    defer allocator.free(freqs);

    const top_k_eff = @min(opts.top_k, model.count());
    const top_k = try topKByFreq(allocator, freqs, top_k_eff);
    defer allocator.free(top_k);

    const byte_dist = try computeByteDistribution(allocator, model);
    defer allocator.free(byte_dist);

    const length_hist = computeLengthHistogram(model);

    // BPE-only: infer dendrogram from id order.
    var dendro_nodes: []DendrogramNode = &.{};
    defer if (dendro_nodes.len > 0) allocator.free(dendro_nodes);
    if (model == .bpe) {
        dendro_nodes = try inferBpeDendrogram(allocator, model.bpe, opts.max_merges);
    }

    try writeHeader(writer, opts.title);
    try writeStyles(writer);
    try writeBody(writer, model, opts, freqs, top_k, byte_dist, length_hist, dendro_nodes);
    try writeScripts(writer);
    try writeFooter(writer);
}

// ---------------------------------------------------------------------
// Data computation.

/// Encode the corpus through the pipeline and tally a per-id frequency
/// table. Returns a `count`-length slice owned by the caller.
pub fn computeTokenFrequencies(
    allocator: std.mem.Allocator,
    pipeline: *const Pipeline,
    corpus_bytes: []const u8,
) ![]u32 {
    const n = pipeline.vocab.count;
    const freqs = try allocator.alloc(u32, if (n == 0) modelCount(pipeline) else n);
    @memset(freqs, 0);

    if (corpus_bytes.len == 0) return freqs;

    const ids = try pipeline.encode(allocator, corpus_bytes);
    defer allocator.free(ids);

    for (ids) |id| {
        if (id < freqs.len) freqs[id] +|= 1;
    }
    return freqs;
}

fn modelCount(pipeline: *const Pipeline) u32 {
    return switch (pipeline.model) {
        .byte_id => 256,
        .bpe => |b| b.count,
        .unigram => |u| u.count,
        .wordpiece => |w| w.count,
        .monster => |m| m.count,
        .rwkv_world => |r| r.count,
    };
}

pub const TopKEntry = struct { id: u32, freq: u32 };

/// Return the top-K ids by frequency, descending. Stable for equal
/// frequencies (ids appearing earlier win).
pub fn topKByFreq(
    allocator: std.mem.Allocator,
    freqs: []const u32,
    k: u32,
) ![]TopKEntry {
    const cap = @min(k, @as(u32, @intCast(freqs.len)));
    const entries = try allocator.alloc(TopKEntry, freqs.len);
    defer allocator.free(entries);
    for (freqs, 0..) |f, i| entries[i] = .{ .id = @intCast(i), .freq = f };

    // Simple insertion-sorted top-K: O(N·K). For K=200 and N<=100K
    // this is ~20M ops — fine for a one-shot CLI command, and avoids
    // dragging in a heap implementation just for this.
    const out = try allocator.alloc(TopKEntry, cap);
    @memset(out, .{ .id = 0, .freq = 0 });
    var filled: u32 = 0;
    for (entries) |e| {
        if (e.freq == 0 and filled >= cap) continue;
        // Find insertion point (descending by freq, id ascending on tie).
        var i: u32 = 0;
        while (i < filled and out[i].freq > e.freq) : (i += 1) {}
        while (i < filled and out[i].freq == e.freq and out[i].id < e.id) : (i += 1) {}
        if (i >= cap) continue;
        // Shift down.
        var j: u32 = if (filled >= cap) cap - 1 else filled;
        while (j > i) : (j -= 1) out[j] = out[j - 1];
        out[i] = e;
        if (filled < cap) filled += 1;
    }
    return out;
}

/// Count tokens that contain each byte 0x00..0xFF. Returns a 256-long
/// slice owned by the caller (allocated rather than stack-returned so
/// the JSON serializer can hand it straight to `std.json.stringify`).
pub fn computeByteDistribution(allocator: std.mem.Allocator, model: ModelRef) ![]u32 {
    const out = try allocator.alloc(u32, 256);
    @memset(out, 0);
    var id: u32 = 0;
    const n = model.count();
    while (id < n) : (id += 1) {
        const bytes = model.tokenBytes(id);
        var seen: [256]bool = [_]bool{false} ** 256;
        for (bytes) |b| {
            if (!seen[b]) {
                seen[b] = true;
                out[b] +|= 1;
            }
        }
    }
    return out;
}

/// Bucket counts: indexes are 1, 2, 3, 4, [5-8], [9-16], [17+].
pub const LengthHistogram = struct {
    /// Buckets, indexed [0..7) for labels "1", "2", "3", "4", "5-8",
    /// "9-16", "17+".
    buckets: [7]u32 = [_]u32{0} ** 7,

    pub const LABELS = [_][]const u8{ "1", "2", "3", "4", "5-8", "9-16", "17+" };
};

pub fn computeLengthHistogram(model: ModelRef) LengthHistogram {
    var h: LengthHistogram = .{};
    var id: u32 = 0;
    const n = model.count();
    while (id < n) : (id += 1) {
        const len = model.tokenBytes(id).len;
        const bucket: u32 = switch (len) {
            0, 1 => 0,
            2 => 1,
            3 => 2,
            4 => 3,
            5...8 => 4,
            9...16 => 5,
            else => 6,
        };
        h.buckets[bucket] +|= 1;
    }
    return h;
}

pub const DendrogramNode = struct {
    /// Token id of the merged result.
    id: u32,
    /// Left child id (smaller-id token of the inferred pair).
    left: u32,
    /// Right child id.
    right: u32,
    /// Byte length of the merged token (so the SVG renderer can size
    /// the label).
    len: u32,
};

/// Infer the first `max_merges` merge events for a BPE vocab from id
/// order. For tiktoken / HF BPE, token id == merge rank — the
/// lowest-id multi-byte tokens are the first merges learned. For each
/// such token we find the (left, right) split whose two pieces both
/// already exist in the vocab and have IDs strictly less than this
/// token's. This matches BPE's "merges build on prior pieces" rule.
///
/// Skips single-byte ids and ids whose bytes don't decompose cleanly
/// into two prior pieces — those are usually `added_tokens` or
/// out-of-band byte-fallback markers.
pub fn inferBpeDendrogram(
    allocator: std.mem.Allocator,
    bpe: *const Bpe,
    max_merges: u32,
) ![]DendrogramNode {
    var out: std.ArrayList(DendrogramNode) = .empty;
    errdefer out.deinit(allocator);

    var id: u32 = 0;
    while (id < bpe.count and out.items.len < max_merges) : (id += 1) {
        const bytes = bpe.bytes[bpe.offsets[id]..bpe.offsets[id + 1]];
        if (bytes.len < 2) continue;

        // Try every split point; pick the first valid (left,right) pair
        // whose ids are both strictly less than `id`.
        var split: u32 = 1;
        var found: ?DendrogramNode = null;
        while (split < bytes.len) : (split += 1) {
            const left_bytes = bytes[0..split];
            const right_bytes = bytes[split..];
            const left_id_opt = bpe.by_bytes.get(left_bytes);
            const right_id_opt = bpe.by_bytes.get(right_bytes);
            if (left_id_opt) |li| if (right_id_opt) |ri| {
                if (li < id and ri < id) {
                    found = .{
                        .id = id,
                        .left = li,
                        .right = ri,
                        .len = @intCast(bytes.len),
                    };
                    break;
                }
            };
        }
        if (found) |node| try out.append(allocator, node);
    }
    return out.toOwnedSlice(allocator);
}

// ---------------------------------------------------------------------
// HTML emission.

fn writeHeader(w: *std.Io.Writer, title: []const u8) !void {
    try w.writeAll("<!DOCTYPE html>\n<html lang=\"en\"><head>\n<meta charset=\"utf-8\">\n");
    try w.writeAll("<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n");
    try w.writeAll("<title>ztok visualize: ");
    try writeHtmlEscaped(w, title);
    try w.writeAll("</title>\n");
}

fn writeStyles(w: *std.Io.Writer) !void {
    try w.writeAll(
        \\<style>
        \\:root { --bg:#0e0f12; --fg:#e8e8ea; --muted:#888; --accent:#7cc; --warn:#e63; --grid:#222; }
        \\* { box-sizing: border-box; }
        \\body { margin: 0; padding: 16px; font: 13px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; background: var(--bg); color: var(--fg); }
        \\h1 { font-size: 18px; margin: 0 0 8px 0; }
        \\h2 { font-size: 14px; margin: 24px 0 8px 0; color: var(--accent); border-bottom: 1px solid var(--grid); padding-bottom: 4px; }
        \\code, .mono { font: 11px/1.3 "SF Mono", Consolas, monospace; }
        \\.meta { color: var(--muted); font-size: 12px; margin-bottom: 8px; }
        \\.section { margin-bottom: 32px; }
        \\.legend { font-size: 11px; color: var(--muted); }
        \\
        \\/* Token grid */
        \\.tokgrid { display: grid; grid-template-columns: repeat(auto-fill, minmax(56px, 1fr)); gap: 2px; }
        \\.tokcell { padding: 6px 4px; text-align: center; border-radius: 3px; cursor: pointer; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; font: 11px/1.2 "SF Mono", Consolas, monospace; color: #fff; }
        \\.tokcell:hover { outline: 1px solid var(--accent); }
        \\
        \\/* Byte heatmap */
        \\.byteheat { border-collapse: collapse; }
        \\.byteheat td { width: 18px; height: 18px; padding: 0; text-align: center; font: 9px/1 "SF Mono", monospace; color: rgba(255,255,255,0.65); cursor: default; }
        \\.byteheat td:hover { outline: 1px solid var(--accent); }
        \\
        \\/* Length histogram */
        \\.histbar { display: flex; align-items: flex-end; gap: 6px; height: 160px; }
        \\.histcol { flex: 1; background: var(--accent); border-radius: 2px 2px 0 0; position: relative; min-height: 2px; }
        \\.histcol .label { position: absolute; top: -16px; left: 50%; transform: translateX(-50%); color: var(--muted); font-size: 10px; }
        \\.histcol .axis { position: absolute; bottom: -14px; left: 50%; transform: translateX(-50%); color: var(--muted); font-size: 10px; }
        \\
        \\/* Dendrogram */
        \\.dendro { background: #14161a; border: 1px solid var(--grid); border-radius: 3px; padding: 8px; overflow-x: auto; }
        \\.dendro svg text { fill: var(--fg); font: 10px "SF Mono", monospace; }
        \\.dendro svg line { stroke: var(--accent); stroke-width: 1.2; }
        \\
        \\/* Tables */
        \\table.added { border-collapse: collapse; width: 100%; }
        \\table.added th, table.added td { padding: 4px 8px; border-bottom: 1px solid var(--grid); text-align: left; font-size: 12px; }
        \\table.added th { color: var(--accent); font-weight: 600; }
        \\table.added td.mono { font: 11px/1.3 "SF Mono", monospace; }
        \\
        \\/* Inspector panel */
        \\#inspector { position: fixed; right: 16px; bottom: 16px; background: #1c1d22; border: 1px solid var(--grid); border-radius: 4px; padding: 10px 14px; font-size: 12px; max-width: 360px; display: none; box-shadow: 0 2px 8px rgba(0,0,0,0.4); }
        \\#inspector .close { float: right; cursor: pointer; color: var(--muted); margin-left: 8px; }
        \\#inspector .id { color: var(--accent); }
        \\#inspector .bytes { font: 11px/1.4 "SF Mono", monospace; word-break: break-all; }
        \\</style>
        \\
    );
}

fn writeBody(
    w: *std.Io.Writer,
    model: ModelRef,
    opts: Options,
    freqs: []const u32,
    top_k: []const TopKEntry,
    byte_dist: []const u32,
    length_hist: LengthHistogram,
    dendro: []const DendrogramNode,
) !void {
    try w.writeAll("</head>\n<body>\n");
    try w.writeAll("<h1>ztok visualize: ");
    try writeHtmlEscaped(w, opts.title);
    try w.writeAll("</h1>\n");
    try w.print(
        "<div class=\"meta\">model: <strong>{s}</strong> &middot; vocab_size: <strong>{d}</strong> &middot; top-k: {d}</div>\n",
        .{ model.kindName(), model.count(), top_k.len },
    );

    // --- Top-K grid ---------------------------------------------------
    try w.writeAll("<div class=\"section\"><h2>Top-K most-frequent tokens</h2>\n");
    try w.writeAll("<div class=\"legend\">Color intensity = log frequency. Click a cell for details.</div>\n");
    try writeTopKGrid(w, model, top_k);
    try w.writeAll("</div>\n");

    // --- Byte heatmap -------------------------------------------------
    try w.writeAll("<div class=\"section\"><h2>Byte-distribution heatmap (0x00..0xFF)</h2>\n");
    try w.writeAll("<div class=\"legend\">Number of vocab tokens containing each byte. Brighter = more.</div>\n");
    try writeByteHeatmap(w, byte_dist);
    try w.writeAll("</div>\n");

    // --- Length histogram --------------------------------------------
    try w.writeAll("<div class=\"section\"><h2>Token length histogram (bytes)</h2>\n");
    try writeLengthHistogram(w, length_hist);
    try w.writeAll("</div>\n");

    // --- Dendrogram (BPE only) ---------------------------------------
    if (model == .bpe) {
        try w.writeAll("<div class=\"section\"><h2>BPE merge dendrogram (first ");
        try w.print("{d})</h2>\n", .{dendro.len});
        try w.writeAll("<div class=\"legend\">Each node = a merge. Inferred from token-id order. Children below are earlier merges.</div>\n");
        try writeDendrogram(w, model.bpe, dendro);
        try w.writeAll("</div>\n");
    }

    // --- Added tokens table -------------------------------------------
    if (opts.added_tokens.len > 0) {
        try w.writeAll("<div class=\"section\"><h2>Added / special tokens (");
        try w.print("{d})</h2>\n", .{opts.added_tokens.len});
        try writeAddedTokensTable(w, opts.added_tokens);
        try w.writeAll("</div>\n");
    }

    // --- Inspector panel + embedded data for JS ----------------------
    try w.writeAll("<div id=\"inspector\"><span class=\"close\" onclick=\"this.parentNode.style.display='none'\">x</span><div id=\"insp-body\"></div></div>\n");

    // Embed the per-token data the inspector needs: id, freq, bytes
    // (as a hex-encoded array — keeps the JSON small + lets the JS
    // render both the hex and the UTF-8 preview).
    try w.writeAll("<script>\nwindow.ZTOK_TOKENS = [");
    var first: bool = true;
    for (top_k) |entry| {
        if (!first) try w.writeAll(",");
        first = false;
        const bytes = model.tokenBytes(entry.id);
        try w.print("{{\"id\":{d},\"freq\":{d},\"hex\":\"", .{ entry.id, entry.freq });
        for (bytes) |b| try w.print("{x:0>2}", .{b});
        try w.writeAll("\",\"utf8\":\"");
        try writeJsString(w, bytes);
        try w.writeAll("\"}");
    }
    try w.writeAll("];\n");
    _ = freqs; // freqs is fully reflected in TOP_K_ENTRIES + TOKENS arrays
    try w.writeAll("</script>\n");
}

fn writeTopKGrid(w: *std.Io.Writer, model: ModelRef, top_k: []const TopKEntry) !void {
    // Find max frequency for color scaling.
    var max_freq: u32 = 1;
    for (top_k) |e| {
        if (e.freq > max_freq) max_freq = e.freq;
    }
    const log_max: f64 = std.math.log2(@as(f64, @floatFromInt(max_freq)) + 1.0);

    try w.writeAll("<div class=\"tokgrid\">");
    for (top_k) |e| {
        const log_f = std.math.log2(@as(f64, @floatFromInt(e.freq)) + 1.0);
        const intensity: u8 = @intFromFloat(@max(0.0, @min(255.0, (log_f / log_max) * 255.0)));
        const r = intensity;
        const g: u8 = @intFromFloat(@as(f64, @floatFromInt(intensity)) * 0.5);
        const b: u8 = @intFromFloat(@as(f64, @floatFromInt(255 - intensity)) * 0.4);

        const bytes = model.tokenBytes(e.id);
        try w.print(
            "<div class=\"tokcell\" style=\"background:rgb({d},{d},{d})\" onclick=\"ztokInspect({d})\" title=\"id={d} freq={d}\">",
            .{ r, g, b, e.id, e.id, e.freq },
        );
        try writeHtmlEscapedPreview(w, bytes, 8);
        try w.writeAll("</div>");
    }
    try w.writeAll("</div>\n");
}

fn writeByteHeatmap(w: *std.Io.Writer, dist: []const u32) !void {
    // 16x16 grid.
    var max_count: u32 = 1;
    for (dist) |c| {
        if (c > max_count) max_count = c;
    }
    const log_max: f64 = std.math.log2(@as(f64, @floatFromInt(max_count)) + 1.0);

    try w.writeAll("<table class=\"byteheat\"><tbody>\n");
    var row: u32 = 0;
    while (row < 16) : (row += 1) {
        try w.writeAll("<tr>");
        var col: u32 = 0;
        while (col < 16) : (col += 1) {
            const idx: u32 = row * 16 + col;
            const c = dist[idx];
            const log_c = std.math.log2(@as(f64, @floatFromInt(c)) + 1.0);
            const intensity: u8 = @intFromFloat(@max(0.0, @min(255.0, (log_c / log_max) * 255.0)));
            try w.print(
                "<td style=\"background:rgb({d},{d},{d})\" title=\"byte 0x{x:0>2} ({d} tokens)\">{x:0>2}</td>",
                .{ intensity, @as(u8, @intFromFloat(@as(f64, @floatFromInt(intensity)) * 0.8)), @as(u8, 30), idx, c, idx },
            );
        }
        try w.writeAll("</tr>\n");
    }
    try w.writeAll("</tbody></table>\n");
}

fn writeLengthHistogram(w: *std.Io.Writer, h: LengthHistogram) !void {
    var max_count: u32 = 1;
    for (h.buckets) |c| {
        if (c > max_count) max_count = c;
    }
    try w.writeAll("<div class=\"histbar\">\n");
    for (h.buckets, 0..) |c, i| {
        const pct: u32 = @intFromFloat(@as(f64, @floatFromInt(c)) / @as(f64, @floatFromInt(max_count)) * 100.0);
        try w.print(
            "<div class=\"histcol\" style=\"height:{d}%\"><span class=\"label\">{d}</span><span class=\"axis\">{s}</span></div>",
            .{ pct, c, LengthHistogram.LABELS[i] },
        );
    }
    try w.writeAll("</div>\n<div style=\"height:18px\"></div>\n");
}

fn writeDendrogram(
    w: *std.Io.Writer,
    bpe: *const Bpe,
    dendro: []const DendrogramNode,
) !void {
    // Layout: simple horizontal tree. Each merge is a row at y = i * 24.
    // Children sit below on a "child row" indexed by the order we first
    // see each unique child id. SVG width scales with the deepest x
    // position we need.
    const row_h: u32 = 24;
    const col_w: u32 = 110;
    const margin: u32 = 8;

    const height: u32 = @intCast(margin * 2 + (dendro.len + 1) * row_h);
    // Two columns are enough for the visual: parent label on the left,
    // left/right child labels under it. We compute a fixed column width
    // and just stack rows.
    const width: u32 = margin * 2 + col_w * 3;

    try w.writeAll("<div class=\"dendro\"><svg xmlns=\"http://www.w3.org/2000/svg\" ");
    try w.print("width=\"{d}\" height=\"{d}\">\n", .{ width, height });

    for (dendro, 0..) |node, i| {
        const y_parent: u32 = @intCast(margin + i * row_h + row_h / 2);
        const y_child: u32 = @intCast(y_parent + row_h);
        const x_parent: u32 = margin + col_w / 2;
        const x_left: u32 = margin + col_w + col_w / 4;
        const x_right: u32 = margin + col_w * 2 + col_w / 4;

        // Lines from parent to each child.
        try w.print(
            "<line x1=\"{d}\" y1=\"{d}\" x2=\"{d}\" y2=\"{d}\"/>\n",
            .{ x_parent, y_parent, x_left, y_child },
        );
        try w.print(
            "<line x1=\"{d}\" y1=\"{d}\" x2=\"{d}\" y2=\"{d}\"/>\n",
            .{ x_parent, y_parent, x_right, y_child },
        );

        // Parent label.
        try w.print("<text x=\"{d}\" y=\"{d}\" text-anchor=\"middle\">", .{ x_parent, y_parent - 4 });
        try writeSvgIdLabel(w, bpe, node.id);
        try w.writeAll("</text>\n");

        // Child labels.
        try w.print("<text x=\"{d}\" y=\"{d}\" text-anchor=\"middle\">", .{ x_left, y_child + 4 });
        try writeSvgIdLabel(w, bpe, node.left);
        try w.writeAll("</text>\n");
        try w.print("<text x=\"{d}\" y=\"{d}\" text-anchor=\"middle\">", .{ x_right, y_child + 4 });
        try writeSvgIdLabel(w, bpe, node.right);
        try w.writeAll("</text>\n");
    }

    try w.writeAll("</svg></div>\n");
}

fn writeSvgIdLabel(w: *std.Io.Writer, bpe: *const Bpe, id: u32) !void {
    const bytes = bpe.bytes[bpe.offsets[id]..bpe.offsets[id + 1]];
    try w.print("[{d}] ", .{id});
    try writeHtmlEscapedPreview(w, bytes, 10);
}

fn writeAddedTokensTable(w: *std.Io.Writer, tokens: []const AddedTokenView) !void {
    try w.writeAll(
        \\<table class="added"><thead><tr>
        \\<th>id</th><th>content</th><th>lstrip</th><th>rstrip</th><th>single_word</th>
        \\</tr></thead><tbody>
        \\
    );
    for (tokens) |t| {
        try w.print("<tr><td class=\"mono\">{d}</td><td class=\"mono\">", .{t.id});
        try writeHtmlEscaped(w, t.content);
        try w.print("</td><td>{s}</td><td>{s}</td><td>{s}</td></tr>\n", .{
            yesno(t.lstrip),
            yesno(t.rstrip),
            yesno(t.single_word),
        });
    }
    try w.writeAll("</tbody></table>\n");
}

fn yesno(b: bool) []const u8 {
    return if (b) "yes" else "no";
}

fn writeScripts(w: *std.Io.Writer) !void {
    try w.writeAll(
        \\<script>
        \\function ztokInspect(id) {
        \\  const t = (window.ZTOK_TOKENS || []).find(x => x.id === id);
        \\  if (!t) return;
        \\  const el = document.getElementById('inspector');
        \\  const body = document.getElementById('insp-body');
        \\  body.innerHTML =
        \\    '<div><span class="id">id ' + t.id + '</span> &middot; freq ' + t.freq + '</div>' +
        \\    '<div class="bytes">hex: ' + t.hex + '</div>' +
        \\    '<div class="bytes">utf8: ' + escapeHtml(t.utf8) + '</div>';
        \\  el.style.display = 'block';
        \\}
        \\function escapeHtml(s) {
        \\  return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
        \\}
        \\</script>
        \\
    );
}

fn writeFooter(w: *std.Io.Writer) !void {
    try w.writeAll("</body></html>\n");
}

// ---------------------------------------------------------------------
// String escaping helpers.

fn writeHtmlEscaped(w: *std.Io.Writer, s: []const u8) !void {
    for (s) |c| switch (c) {
        '&' => try w.writeAll("&amp;"),
        '<' => try w.writeAll("&lt;"),
        '>' => try w.writeAll("&gt;"),
        '"' => try w.writeAll("&quot;"),
        '\'' => try w.writeAll("&#39;"),
        else => try w.writeByte(c),
    };
}

/// Like `writeHtmlEscaped` but caps the rendered glyph count at
/// `max_chars`, falls back to a hex placeholder for non-printable
/// bytes, and elides anything beyond the cap with a single `…` (UTF-8).
fn writeHtmlEscapedPreview(w: *std.Io.Writer, bytes: []const u8, max_chars: usize) !void {
    var written: usize = 0;
    for (bytes) |b| {
        if (written >= max_chars) {
            try w.writeAll("…");
            return;
        }
        if (b >= 0x20 and b < 0x7f) {
            try escapeOneHtml(w, b);
            written += 1;
        } else if (b == ' ') {
            // explicit visible space marker for byte-level vocabs.
            try w.writeAll("·");
            written += 1;
        } else {
            try w.print("\\x{x:0>2}", .{b});
            written += 4;
        }
    }
}

fn escapeOneHtml(w: *std.Io.Writer, c: u8) !void {
    switch (c) {
        '&' => try w.writeAll("&amp;"),
        '<' => try w.writeAll("&lt;"),
        '>' => try w.writeAll("&gt;"),
        '"' => try w.writeAll("&quot;"),
        '\'' => try w.writeAll("&#39;"),
        else => try w.writeByte(c),
    }
}

fn writeJsString(w: *std.Io.Writer, bytes: []const u8) !void {
    for (bytes) |b| switch (b) {
        '\\' => try w.writeAll("\\\\"),
        '"' => try w.writeAll("\\\""),
        '\n' => try w.writeAll("\\n"),
        '\r' => try w.writeAll("\\r"),
        '\t' => try w.writeAll("\\t"),
        '<' => try w.writeAll("\\u003c"),
        '>' => try w.writeAll("\\u003e"),
        '&' => try w.writeAll("\\u0026"),
        else => if (b < 0x20 or b == 0x7f) {
            try w.print("\\u{x:0>4}", .{b});
        } else {
            try w.writeByte(b);
        },
    };
}

// ---------------------------------------------------------------------
// Tests.

const testing = std.testing;

test "MINI_CORPUS is non-empty" {
    try testing.expect(MINI_CORPUS.len > 100);
}

test "byte distribution length is 256 for any vocab" {
    var bpe = try buildTinyBpe(testing.allocator);
    defer bpe.deinit();
    const ref: ModelRef = .{ .bpe = &bpe };
    const dist = try computeByteDistribution(testing.allocator, ref);
    defer testing.allocator.free(dist);
    try testing.expectEqual(@as(usize, 256), dist.len);
}

test "length histogram buckets sum to vocab count" {
    var bpe = try buildTinyBpe(testing.allocator);
    defer bpe.deinit();
    const ref: ModelRef = .{ .bpe = &bpe };
    const h = computeLengthHistogram(ref);
    var sum: u32 = 0;
    for (h.buckets) |c| sum += c;
    try testing.expectEqual(bpe.count, sum);
}

test "topKByFreq returns descending freqs" {
    const freqs = [_]u32{ 3, 1, 7, 0, 5 };
    const out = try topKByFreq(testing.allocator, &freqs, 3);
    defer testing.allocator.free(out);
    try testing.expectEqual(@as(usize, 3), out.len);
    try testing.expectEqual(@as(u32, 7), out[0].freq);
    try testing.expectEqual(@as(u32, 2), out[0].id);
    try testing.expectEqual(@as(u32, 5), out[1].freq);
    try testing.expectEqual(@as(u32, 3), out[2].freq);
}

test "render produces well-formed HTML with DOCTYPE and balanced tags" {
    var bpe = try buildTinyBpe(testing.allocator);
    defer bpe.deinit();
    const ref: ModelRef = .{ .bpe = &bpe };
    var vocab = Vocab.empty(testing.allocator);
    defer vocab.deinit();
    const pipeline: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .{ .bpe = &bpe },
        .decoder = .concat,
        .vocab = &vocab,
    };

    var buf: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buf.deinit();

    try render(testing.allocator, &buf.writer, ref, &pipeline, .{
        .title = "tiny",
        .top_k = 16,
        .corpus = "abababab abc abcabcabc",
    });

    const html = buf.written();
    try testing.expect(std.mem.startsWith(u8, html, "<!DOCTYPE html>"));
    try testing.expect(std.mem.endsWith(u8, std.mem.trimEnd(u8, html, "\n"), "</html>"));
    try testing.expect(std.mem.indexOf(u8, html, "<head>") != null);
    try testing.expect(std.mem.indexOf(u8, html, "</head>") != null);
    try testing.expect(std.mem.indexOf(u8, html, "<body>") != null);
    try testing.expect(std.mem.indexOf(u8, html, "</body>") != null);
    // Per spec: byte heatmap renders a 256-cell <table>.
    var td_count: usize = 0;
    var i: usize = 0;
    while (std.mem.indexOfPos(u8, html, i, "<td")) |p| : (i = p + 1) td_count += 1;
    try testing.expect(td_count >= 256);
    // Dendrogram for BPE: at least one <line> + one <text> SVG element.
    try testing.expect(std.mem.indexOf(u8, html, "<svg ") != null);
    try testing.expect(std.mem.indexOf(u8, html, "<line ") != null);
    try testing.expect(std.mem.indexOf(u8, html, "<text ") != null);
}

test "BPE dendrogram inference picks valid merges" {
    var bpe = try buildTinyBpe(testing.allocator);
    defer bpe.deinit();
    const nodes = try inferBpeDendrogram(testing.allocator, &bpe, 50);
    defer testing.allocator.free(nodes);
    // Every parent id must be > both children ids (the rule we follow).
    for (nodes) |n| {
        try testing.expect(n.left < n.id);
        try testing.expect(n.right < n.id);
    }
}

/// Build a deterministic tiny BPE vocab for unit tests: bytes 'a','b','c'
/// (ids 0..2) plus "ab" (3), "bc" (4), "abc" (5). Mirrors the on-disk
/// tiktoken format: id == rank.
fn buildTinyBpe(allocator: std.mem.Allocator) !Bpe {
    // Concatenated bytes: a|b|c|ab|bc|abc -> "a"+"b"+"c"+"ab"+"bc"+"abc"
    const concat = "abcabbcabc";
    const bytes = try allocator.dupe(u8, concat);
    errdefer allocator.free(bytes);
    const offsets = try allocator.alloc(u32, 7);
    errdefer allocator.free(offsets);
    offsets[0] = 0; // "a"
    offsets[1] = 1; // "b"
    offsets[2] = 2; // "c"
    offsets[3] = 3; // "ab"
    offsets[4] = 5; // "bc"
    offsets[5] = 7; // "abc"
    offsets[6] = 10;

    var by_bytes: std.StringHashMap(TokenId) = .init(allocator);
    errdefer by_bytes.deinit();
    var id: u32 = 0;
    while (id < 6) : (id += 1) {
        const piece = bytes[offsets[id]..offsets[id + 1]];
        try by_bytes.put(piece, id);
    }

    return .{
        .allocator = allocator,
        .bytes = bytes,
        .offsets = offsets,
        .count = 6,
        .by_bytes = by_bytes,
        .max_piece_len = 3,
    };
}

// ---------------------------------------------------------------------
// Convenience: build an AddedTokenView slice from an HF tokenizer's
// added_tokens. The CLI calls this when the loaded vocab is HF-format.

pub fn addedTokensFromHF(
    allocator: std.mem.Allocator,
    hf: *const hf_json.HFTokenizer,
) ![]AddedTokenView {
    const out = try allocator.alloc(AddedTokenView, hf.added_tokens.len);
    for (hf.added_tokens, 0..) |t, i| {
        out[i] = .{
            .id = t.id,
            .content = t.content,
            .lstrip = t.lstrip,
            .rstrip = t.rstrip,
            .single_word = t.single_word,
        };
    }
    return out;
}

// Silence unused-decl warnings for cross-module references that we only
// touch through the `added_tokens_mod`-typed `AddedToken` field above.
comptime {
    _ = added_tokens_mod;
}
