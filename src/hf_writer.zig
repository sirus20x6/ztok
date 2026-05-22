const std = @import("std");
const TokenId = @import("token.zig").TokenId;
const Bpe = @import("bpe.zig").Bpe;
const WordPiece = @import("wordpiece.zig").WordPiece;
const Unigram = @import("unigram.zig").Unigram;
const AddedToken = @import("hf_json.zig").AddedToken;
const hf_json = @import("hf_json.zig");

const Stringify = std.json.Stringify;
const Writer = std.Io.Writer;

pub const WriteOptions = struct {
    added_tokens: []const AddedToken = &.{},
    indent: u8 = 2,
};

fn buildOptions(indent: u8) Stringify.Options {
    return .{
        .whitespace = switch (indent) {
            0 => .minified,
            1 => .indent_1,
            2 => .indent_2,
            3 => .indent_3,
            4 => .indent_4,
            5, 6, 7 => .indent_4,
            else => .indent_8,
        },
    };
}

// Emit `bytes` as a JSON string body (no surrounding quotes). Handles
// raw byte vocabs: ASCII printables verbatim, control chars and bytes
// >= 0x80 as `\u00XX` so the output is always valid JSON. The parser
// will decode `\u00XX` back into 1- or 2-byte UTF-8 depending on the
// codepoint.
fn writeStringBody(w: *Writer, bytes: []const u8) !void {
    for (bytes) |b| {
        switch (b) {
            '"' => try w.writeAll("\\\""),
            '\\' => try w.writeAll("\\\\"),
            0x08 => try w.writeAll("\\b"),
            0x09 => try w.writeAll("\\t"),
            0x0A => try w.writeAll("\\n"),
            0x0C => try w.writeAll("\\f"),
            0x0D => try w.writeAll("\\r"),
            0x20...0x21, 0x23...0x5B, 0x5D...0x7E => try w.writeByte(b),
            else => {
                try w.writeAll("\\u00");
                const hex = "0123456789abcdef";
                try w.writeByte(hex[b >> 4]);
                try w.writeByte(hex[b & 0x0F]);
            },
        }
    }
}

fn writeRawString(s: *Stringify, bytes: []const u8) !void {
    try s.beginWriteRaw();
    try s.writer.writeByte('"');
    try writeStringBody(s.writer, bytes);
    try s.writer.writeByte('"');
    s.endWriteRaw();
}

fn writeRawField(s: *Stringify, key: []const u8) !void {
    try s.beginObjectFieldRaw();
    try s.writer.writeByte('"');
    try writeStringBody(s.writer, key);
    try s.writer.writeByte('"');
    s.endObjectFieldRaw();
}

fn writeAddedTokens(s: *Stringify, added: []const AddedToken) !void {
    try s.objectField("added_tokens");
    try s.beginArray();
    for (added) |t| {
        try s.beginObject();
        try s.objectField("id");
        try s.write(@as(i64, @intCast(t.id)));
        try s.objectField("content");
        try writeRawString(s, t.content);
        try s.objectField("single_word");
        try s.write(t.single_word);
        try s.objectField("lstrip");
        try s.write(t.lstrip);
        try s.objectField("rstrip");
        try s.write(t.rstrip);
        try s.objectField("normalized");
        try s.write(t.normalized);
        try s.objectField("special");
        try s.write(t.special);
        try s.endObject();
    }
    try s.endArray();
}

fn writeNullField(s: *Stringify, key: []const u8) !void {
    try s.objectField(key);
    try s.write(null);
}

fn writeBpeVocab(s: *Stringify, bpe: *const Bpe) !void {
    try s.objectField("vocab");
    try s.beginObject();
    var i: u32 = 0;
    while (i < bpe.count) : (i += 1) {
        const key = bpe.idBytes(i);
        try writeRawField(s, key);
        try s.write(@as(i64, @intCast(i)));
    }
    try s.endObject();
}

// Find a (left, right) split where both pieces already exist with smaller ids.
// Pick the leftmost-shortest split (smallest split_pos).
fn findMergeSplit(bpe: *const Bpe, id: TokenId) ?struct { left: []const u8, right: []const u8 } {
    const piece = bpe.idBytes(id);
    if (piece.len < 2) return null;
    var split_pos: usize = 1;
    while (split_pos < piece.len) : (split_pos += 1) {
        const left = piece[0..split_pos];
        const right = piece[split_pos..];
        const lid = bpe.by_bytes.get(left) orelse continue;
        const rid = bpe.by_bytes.get(right) orelse continue;
        if (lid < id and rid < id) {
            return .{ .left = left, .right = right };
        }
    }
    return null;
}

fn writeBpeMerges(s: *Stringify, bpe: *const Bpe) !void {
    try s.objectField("merges");
    try s.beginArray();
    var i: u32 = 256;
    while (i < bpe.count) : (i += 1) {
        const split = findMergeSplit(bpe, i) orelse continue;
        try s.beginArray();
        try writeRawString(s, split.left);
        try writeRawString(s, split.right);
        try s.endArray();
    }
    try s.endArray();
}

fn writeBpeBody(s: *Stringify, bpe: *const Bpe, opts: WriteOptions) !void {
    try s.beginObject();
    try s.objectField("version");
    try s.write("1.0");
    try writeAddedTokens(s, opts.added_tokens);
    try writeNullField(s, "normalizer");
    try writeNullField(s, "pre_tokenizer");
    try writeNullField(s, "decoder");
    try s.objectField("model");
    try s.beginObject();
    try s.objectField("type");
    try s.write("BPE");
    try writeNullField(s, "dropout");
    try writeNullField(s, "unk_token");
    try writeNullField(s, "continuing_subword_prefix");
    try writeNullField(s, "end_of_word_suffix");
    try s.objectField("fuse_unk");
    try s.write(false);
    try s.objectField("byte_fallback");
    try s.write(false);
    try s.objectField("ignore_merges");
    try s.write(false);
    try writeBpeVocab(s, bpe);
    try writeBpeMerges(s, bpe);
    try s.endObject();
    try s.endObject();
}

pub fn writeBpe(allocator: std.mem.Allocator, bpe: *const Bpe, opts: WriteOptions) ![]u8 {
    var aw: Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    var s: Stringify = .{ .writer = &aw.writer, .options = buildOptions(opts.indent) };
    try writeBpeBody(&s, bpe, opts);
    return aw.toOwnedSlice();
}

fn writeWordPieceVocab(s: *Stringify, wp: *const WordPiece) !void {
    try s.objectField("vocab");
    try s.beginObject();
    var i: u32 = 0;
    while (i < wp.count) : (i += 1) {
        const key = wp.idBytes(i);
        try writeRawField(s, key);
        try s.write(@as(i64, @intCast(i)));
    }
    try s.endObject();
}

fn writeWordPieceBody(s: *Stringify, wp: *const WordPiece, opts: WriteOptions) !void {
    try s.beginObject();
    try s.objectField("version");
    try s.write("1.0");
    try writeAddedTokens(s, opts.added_tokens);
    try writeNullField(s, "normalizer");
    try writeNullField(s, "pre_tokenizer");
    try writeNullField(s, "decoder");
    try s.objectField("model");
    try s.beginObject();
    try s.objectField("type");
    try s.write("WordPiece");
    try s.objectField("unk_token");
    try writeRawString(s, wp.idBytes(wp.unk_id));
    try s.objectField("continuing_subword_prefix");
    try writeRawString(s, wp.continuing_subword_prefix);
    try s.objectField("max_input_chars_per_word");
    try s.write(@as(i64, @intCast(wp.max_input_chars_per_word)));
    try writeWordPieceVocab(s, wp);
    try s.endObject();
    try s.endObject();
}

pub fn writeWordPiece(allocator: std.mem.Allocator, wp: *const WordPiece, opts: WriteOptions) ![]u8 {
    var aw: Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    var s: Stringify = .{ .writer = &aw.writer, .options = buildOptions(opts.indent) };
    try writeWordPieceBody(&s, wp, opts);
    return aw.toOwnedSlice();
}

fn writeUnigramScore(s: *Stringify, score: f32) !void {
    // f32 -> f64 widening is exact; print with `{d}` for HF-compatible output.
    const f: f64 = @floatCast(score);
    try s.print("{d}", .{f});
}

fn writeUnigramVocab(s: *Stringify, u: *const Unigram) !void {
    try s.objectField("vocab");
    try s.beginArray();
    var i: u32 = 0;
    while (i < u.count) : (i += 1) {
        try s.beginArray();
        try writeRawString(s, u.idBytes(i));
        try writeUnigramScore(s, u.scores[i]);
        try s.endArray();
    }
    try s.endArray();
}

fn writeUnigramBody(s: *Stringify, u: *const Unigram, opts: WriteOptions) !void {
    try s.beginObject();
    try s.objectField("version");
    try s.write("1.0");
    try writeAddedTokens(s, opts.added_tokens);
    try writeNullField(s, "normalizer");
    try writeNullField(s, "pre_tokenizer");
    try writeNullField(s, "decoder");
    try s.objectField("model");
    try s.beginObject();
    try s.objectField("type");
    try s.write("Unigram");
    try s.objectField("unk_id");
    try s.write(@as(i64, @intCast(u.unk_id)));
    try writeUnigramVocab(s, u);
    try s.objectField("byte_fallback");
    try s.write(false);
    try s.endObject();
    try s.endObject();
}

pub fn writeUnigram(allocator: std.mem.Allocator, u: *const Unigram, opts: WriteOptions) ![]u8 {
    var aw: Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    var s: Stringify = .{ .writer = &aw.writer, .options = buildOptions(opts.indent) };
    try writeUnigramBody(&s, u, opts);
    return aw.toOwnedSlice();
}

fn writeBytesToFile(allocator: std.mem.Allocator, path: []const u8, data: []const u8) !void {
    _ = allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = data });
}

pub fn writeBpeFile(allocator: std.mem.Allocator, bpe: *const Bpe, path: []const u8, opts: WriteOptions) !void {
    const bytes = try writeBpe(allocator, bpe, opts);
    defer allocator.free(bytes);
    try writeBytesToFile(allocator, path, bytes);
}

pub fn writeWordPieceFile(allocator: std.mem.Allocator, wp: *const WordPiece, path: []const u8, opts: WriteOptions) !void {
    const bytes = try writeWordPiece(allocator, wp, opts);
    defer allocator.free(bytes);
    try writeBytesToFile(allocator, path, bytes);
}

pub fn writeUnigramFile(allocator: std.mem.Allocator, u: *const Unigram, path: []const u8, opts: WriteOptions) !void {
    const bytes = try writeUnigram(allocator, u, opts);
    defer allocator.free(bytes);
    try writeBytesToFile(allocator, path, bytes);
}

// ---------------------------------------------------------------------
// Post-processor writers
//
// The HuggingFace `post_processor` field accepts one of:
//   - BertProcessing      { sep:[tok,id], cls:[tok,id] }
//   - RobertaProcessing   { sep:[tok,id], cls:[tok,id],
//                           trim_offsets:bool, add_prefix_space:bool }
//   - ByteLevel           { add_prefix_space:bool, trim_offsets:bool,
//                           use_regex:bool }
//   - TemplateProcessing  { single:[piece...], pair:[piece...],
//                           special_tokens:{ id -> {id, ids:[..], tokens:[..]}}}
//   - Sequence            { processors:[sub-processor...] }
//   - null
//
// `PostProcessorSpec` is the value type passed to `writePostProcessor`.
// It mirrors the HF JSON shape and carries the full set of fields
// required for a lossless round-trip — most importantly the
// human-readable token strings for Bert/Roberta and the
// `special_tokens` map for Template, neither of which the runtime
// `PostProcessor` keeps around.

pub const PostProcessorSpec = union(enum) {
    none,
    bert: BertSpec,
    roberta: RobertaSpec,
    byte_level: ByteLevelSpec,
    template: TemplateSpec,
    sequence: SequenceSpec,

    pub const TokenRef = struct {
        token: []const u8,
        id: TokenId,
    };

    pub const BertSpec = struct {
        sep: TokenRef,
        cls: TokenRef,
    };

    pub const RobertaSpec = struct {
        sep: TokenRef,
        cls: TokenRef,
        trim_offsets: bool = true,
        add_prefix_space: bool = true,
    };

    pub const ByteLevelSpec = struct {
        add_prefix_space: bool = true,
        trim_offsets: bool = true,
        use_regex: bool = true,
    };

    pub const PieceKind = enum { sequence_a, sequence_b, special_token };

    pub const Piece = struct {
        kind: PieceKind,
        /// For `.special_token`: the special-token key (must exist in
        /// `special_tokens`). For `.sequence_a` and `.sequence_b` this
        /// field is ignored.
        id: []const u8 = "",
        type_id: u32 = 0,
    };

    pub const SpecialToken = struct {
        /// The map key, repeated as the inner `id` field per HF schema.
        id: []const u8,
        ids: []const TokenId,
        tokens: []const []const u8,
    };

    pub const TemplateSpec = struct {
        single: []const Piece,
        /// `null` (or empty slice) -> emit `[]` for `pair` (no pair rule).
        pair: ?[]const Piece = null,
        special_tokens: []const SpecialToken,
    };

    pub const SequenceSpec = struct {
        processors: []const PostProcessorSpec,
    };
};

fn writeTokenRef(s: *Stringify, ref: PostProcessorSpec.TokenRef) !void {
    try s.beginArray();
    try writeRawString(s, ref.token);
    try s.write(@as(i64, @intCast(ref.id)));
    try s.endArray();
}

fn writeBertProcessing(s: *Stringify, b: PostProcessorSpec.BertSpec) !void {
    try s.beginObject();
    // HF emits `type` first, then `sep`, then `cls` (order matches the
    // struct definition in tokenizers/src/processors/bert.rs).
    try s.objectField("type");
    try s.write("BertProcessing");
    try s.objectField("sep");
    try writeTokenRef(s, b.sep);
    try s.objectField("cls");
    try writeTokenRef(s, b.cls);
    try s.endObject();
}

fn writeRobertaProcessing(s: *Stringify, r: PostProcessorSpec.RobertaSpec) !void {
    try s.beginObject();
    try s.objectField("type");
    try s.write("RobertaProcessing");
    try s.objectField("sep");
    try writeTokenRef(s, r.sep);
    try s.objectField("cls");
    try writeTokenRef(s, r.cls);
    try s.objectField("trim_offsets");
    try s.write(r.trim_offsets);
    try s.objectField("add_prefix_space");
    try s.write(r.add_prefix_space);
    try s.endObject();
}

fn writeByteLevelPostProc(s: *Stringify, b: PostProcessorSpec.ByteLevelSpec) !void {
    try s.beginObject();
    try s.objectField("type");
    try s.write("ByteLevel");
    try s.objectField("add_prefix_space");
    try s.write(b.add_prefix_space);
    try s.objectField("trim_offsets");
    try s.write(b.trim_offsets);
    try s.objectField("use_regex");
    try s.write(b.use_regex);
    try s.endObject();
}

fn writePieces(s: *Stringify, pieces: []const PostProcessorSpec.Piece) !void {
    try s.beginArray();
    for (pieces) |p| {
        try s.beginObject();
        switch (p.kind) {
            .sequence_a, .sequence_b => {
                try s.objectField("Sequence");
                try s.beginObject();
                try s.objectField("id");
                try s.write(if (p.kind == .sequence_a) "A" else "B");
                try s.objectField("type_id");
                try s.write(@as(i64, @intCast(p.type_id)));
                try s.endObject();
            },
            .special_token => {
                try s.objectField("SpecialToken");
                try s.beginObject();
                try s.objectField("id");
                try writeRawString(s, p.id);
                try s.objectField("type_id");
                try s.write(@as(i64, @intCast(p.type_id)));
                try s.endObject();
            },
        }
        try s.endObject();
    }
    try s.endArray();
}

fn writeSpecialTokenEntry(s: *Stringify, st: PostProcessorSpec.SpecialToken) !void {
    try s.beginObject();
    try s.objectField("id");
    try writeRawString(s, st.id);
    try s.objectField("ids");
    try s.beginArray();
    for (st.ids) |id| try s.write(@as(i64, @intCast(id)));
    try s.endArray();
    try s.objectField("tokens");
    try s.beginArray();
    for (st.tokens) |tok| try writeRawString(s, tok);
    try s.endArray();
    try s.endObject();
}

fn writeTemplateProcessing(s: *Stringify, t: PostProcessorSpec.TemplateSpec) !void {
    try s.beginObject();
    try s.objectField("type");
    try s.write("TemplateProcessing");
    try s.objectField("single");
    try writePieces(s, t.single);
    try s.objectField("pair");
    if (t.pair) |p| {
        try writePieces(s, p);
    } else {
        // HF parsers accept `[]` for "no pair template"; mirror that.
        try s.beginArray();
        try s.endArray();
    }
    try s.objectField("special_tokens");
    try s.beginObject();
    for (t.special_tokens) |st| {
        try writeRawField(s, st.id);
        try writeSpecialTokenEntry(s, st);
    }
    try s.endObject();
    try s.endObject();
}

fn writePostProcessorValue(s: *Stringify, pp: PostProcessorSpec) (Writer.Error || error{OutOfMemory})!void {
    switch (pp) {
        .none => try s.write(null),
        .bert => |b| try writeBertProcessing(s, b),
        .roberta => |r| try writeRobertaProcessing(s, r),
        .byte_level => |b| try writeByteLevelPostProc(s, b),
        .template => |t| try writeTemplateProcessing(s, t),
        .sequence => |seq| {
            try s.beginObject();
            try s.objectField("type");
            try s.write("Sequence");
            try s.objectField("processors");
            try s.beginArray();
            for (seq.processors) |sub| {
                if (sub == .sequence) {
                    // HF only allows a single level of Sequence nesting in
                    // practice; emitting nested Sequence is well-formed
                    // JSON but the parser may reject. We allow it for
                    // round-trip; the caller is responsible for sanity.
                }
                try writePostProcessorValue(s, sub);
            }
            try s.endArray();
            try s.endObject();
        },
    }
}

/// Emit `pp` as the value of a `post_processor` field (no surrounding
/// key). For `.none`, emits the JSON literal `null`.
pub fn writePostProcessor(allocator: std.mem.Allocator, pp: PostProcessorSpec, opts: WriteOptions) ![]u8 {
    var aw: Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    var s: Stringify = .{ .writer = &aw.writer, .options = buildOptions(opts.indent) };
    try writePostProcessorValue(&s, pp);
    return aw.toOwnedSlice();
}

// --- tests ---

const testing = std.testing;

const TestEntry = struct { bytes: []const u8, rank: u32 };

// Build a tiktoken-format vocab string from (bytes, rank) pairs.
fn buildVocabSource(allocator: std.mem.Allocator, entries: []const TestEntry) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    const enc = std.base64.standard.Encoder;
    for (entries) |e| {
        const sz = enc.calcSize(e.bytes.len);
        const tmp = try allocator.alloc(u8, sz);
        defer allocator.free(tmp);
        const encoded = enc.encode(tmp, e.bytes);
        try buf.appendSlice(allocator, encoded);
        try buf.print(allocator, " {d}\n", .{e.rank});
    }
    return buf.toOwnedSlice(allocator);
}

// Build a Bpe with 256 single-byte tokens + extras (which must use ranks 256+).
fn buildByteBpe(allocator: std.mem.Allocator, extras: []const TestEntry) !Bpe {
    var entries: std.ArrayList(TestEntry) = .empty;
    defer entries.deinit(allocator);
    var byte_holders: [256][1]u8 = undefined;
    var i: u32 = 0;
    while (i < 256) : (i += 1) {
        byte_holders[i][0] = @intCast(i);
        try entries.append(allocator, .{ .bytes = byte_holders[i][0..1], .rank = i });
    }
    for (extras) |e| try entries.append(allocator, .{ .bytes = e.bytes, .rank = e.rank });
    const src = try buildVocabSource(allocator, entries.items);
    defer allocator.free(src);
    return Bpe.loadTiktokenBytes(allocator, src);
}

test "writeBpe emits parseable JSON" {
    const extras = [_]TestEntry{
        .{ .bytes = "ab", .rank = 256 },
        .{ .bytes = "abc", .rank = 257 },
        .{ .bytes = "abcd", .rank = 258 },
    };
    var bpe = try buildByteBpe(testing.allocator, &extras);
    defer bpe.deinit();

    const json = try writeBpe(testing.allocator, &bpe, .{});
    defer testing.allocator.free(json);

    var parsed = try hf_json.loadFromBytes(testing.allocator, json);
    defer parsed.deinit();
    try testing.expectEqual(hf_json.ModelKind.bpe, parsed.model_kind);
    try testing.expectEqual(@as(u32, 259), parsed.vocab.count);
    try testing.expectEqual(@as(usize, 3), parsed.merges.len);
}

test "writeBpe round-trips a real Bpe" {
    const extras = [_]TestEntry{
        .{ .bytes = "ab", .rank = 256 },
        .{ .bytes = "abc", .rank = 257 },
        .{ .bytes = "abcd", .rank = 258 },
        .{ .bytes = "abcde", .rank = 259 },
    };
    var bpe = try buildByteBpe(testing.allocator, &extras);
    defer bpe.deinit();

    const json = try writeBpe(testing.allocator, &bpe, .{});
    defer testing.allocator.free(json);

    var parsed = try hf_json.loadFromBytes(testing.allocator, json);
    defer parsed.deinit();

    try testing.expectEqual(bpe.count, parsed.vocab.count);
    // ASCII single-byte tokens (ids 0..127) round-trip exactly. High
    // bytes (0x80..0xFF, ids 128..255) get escaped as `\u00XX` and
    // round-trip into 2-byte UTF-8 sequences. Extras (id >= 256) are
    // ASCII so they round-trip exactly too.
    var i: u32 = 0;
    while (i < 128) : (i += 1) {
        try testing.expectEqualSlices(u8, bpe.idBytes(i), parsed.vocab.tokenBytes(i));
    }
    i = 256;
    while (i < bpe.count) : (i += 1) {
        try testing.expectEqualSlices(u8, bpe.idBytes(i), parsed.vocab.tokenBytes(i));
    }
    // Each merge should reproduce one of our extras when joining left+right
    // by id (i.e. the parsed vocab forms a valid merge tree).
    try testing.expectEqual(@as(usize, 4), parsed.merges.len);
    for (parsed.merges) |m| {
        try testing.expect(m.left < parsed.vocab.count);
        try testing.expect(m.right < parsed.vocab.count);
    }
}

test "writeWordPiece emits parseable JSON" {
    const vocab = [_][]const u8{ "[UNK]", "[CLS]", "the", "##s", "hello" };
    var wp = try WordPiece.init(testing.allocator, &vocab, .{ .unk_id = 0 });
    defer wp.deinit();

    const json = try writeWordPiece(testing.allocator, &wp, .{});
    defer testing.allocator.free(json);

    var parsed = try hf_json.loadFromBytes(testing.allocator, json);
    defer parsed.deinit();
    try testing.expectEqual(hf_json.ModelKind.wordpiece, parsed.model_kind);
    try testing.expectEqual(@as(u32, 5), parsed.vocab.count);
    try testing.expectEqualStrings("[UNK]", parsed.vocab.tokenBytes(0));
    try testing.expectEqualStrings("##s", parsed.vocab.tokenBytes(3));
    try testing.expect(parsed.wordpiece_continuing_subword_prefix != null);
    try testing.expectEqualStrings("##", parsed.wordpiece_continuing_subword_prefix.?);
    try testing.expectEqual(@as(u32, 100), parsed.wordpiece_max_input_chars_per_word);
    try testing.expect(parsed.unk_token != null);
    try testing.expectEqualStrings("[UNK]", parsed.unk_token.?);
}

test "writeUnigram emits parseable JSON" {
    var b = Unigram.Builder.init(testing.allocator);
    defer b.deinit();
    _ = try b.addToken("<unk>", 0.0);
    _ = try b.addToken("the", -3.14);
    _ = try b.addToken("a", -3.5);
    var u = try b.finalize(0);
    defer u.deinit();

    const json = try writeUnigram(testing.allocator, &u, .{});
    defer testing.allocator.free(json);

    var parsed = try hf_json.loadFromBytes(testing.allocator, json);
    defer parsed.deinit();
    try testing.expectEqual(hf_json.ModelKind.unigram, parsed.model_kind);
    try testing.expectEqual(@as(u32, 3), parsed.vocab.count);
    try testing.expect(parsed.unigram_scores != null);
    const scores = parsed.unigram_scores.?;
    try testing.expectApproxEqAbs(@as(f32, 0.0), scores[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, -3.14), scores[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, -3.5), scores[2], 1e-6);
    try testing.expectEqual(@as(?TokenId, 0), parsed.unigram_unk_id);
}

test "writeUnigram preserves piece order (id matters)" {
    var b = Unigram.Builder.init(testing.allocator);
    defer b.deinit();
    _ = try b.addToken("<unk>", 0.0);
    _ = try b.addToken("foo", -1.0);
    _ = try b.addToken("bar", -2.0);
    _ = try b.addToken("baz", -3.0);
    _ = try b.addToken("qux", -4.0);
    var u = try b.finalize(0);
    defer u.deinit();

    const json = try writeUnigram(testing.allocator, &u, .{});
    defer testing.allocator.free(json);

    var parsed = try hf_json.loadFromBytes(testing.allocator, json);
    defer parsed.deinit();
    try testing.expectEqual(u.count, parsed.vocab.count);
    var i: u32 = 0;
    while (i < u.count) : (i += 1) {
        try testing.expectEqualSlices(u8, u.idBytes(i), parsed.vocab.tokenBytes(i));
    }
}

test "writeBpe indent=0 produces single line" {
    const extras = [_]TestEntry{
        .{ .bytes = "ab", .rank = 256 },
    };
    var bpe = try buildByteBpe(testing.allocator, &extras);
    defer bpe.deinit();

    const json = try writeBpe(testing.allocator, &bpe, .{ .indent = 0 });
    defer testing.allocator.free(json);

    try testing.expect(std.mem.indexOfScalar(u8, json, '\n') == null);
    // Sanity: still parseable.
    var parsed = try hf_json.loadFromBytes(testing.allocator, json);
    defer parsed.deinit();
    try testing.expectEqual(hf_json.ModelKind.bpe, parsed.model_kind);
}

test "writeBpe with added_tokens" {
    const extras = [_]TestEntry{
        .{ .bytes = "ab", .rank = 256 },
    };
    var bpe = try buildByteBpe(testing.allocator, &extras);
    defer bpe.deinit();

    const unk_content = try testing.allocator.dupe(u8, "<|endoftext|>");
    defer testing.allocator.free(unk_content);
    const bos_content = try testing.allocator.dupe(u8, "<|bos|>");
    defer testing.allocator.free(bos_content);

    const added = [_]AddedToken{
        .{ .id = 257, .content = unk_content, .special = true, .normalized = false },
        .{ .id = 258, .content = bos_content, .special = true },
    };

    const json = try writeBpe(testing.allocator, &bpe, .{ .added_tokens = &added });
    defer testing.allocator.free(json);

    var parsed = try hf_json.loadFromBytes(testing.allocator, json);
    defer parsed.deinit();
    try testing.expectEqual(@as(usize, 2), parsed.added_tokens.len);
    try testing.expectEqual(@as(TokenId, 257), parsed.added_tokens[0].id);
    try testing.expectEqualStrings("<|endoftext|>", parsed.added_tokens[0].content);
    try testing.expect(parsed.added_tokens[0].special);
    try testing.expect(!parsed.added_tokens[0].normalized);
    try testing.expectEqualStrings("<|bos|>", parsed.added_tokens[1].content);
}

// --- post_processor writer tests ---

const post_processor = @import("post_processor.zig");

fn parseJsonValue(input: []const u8) !std.json.Parsed(std.json.Value) {
    return std.json.parseFromSlice(std.json.Value, testing.allocator, input, .{});
}

test "writePostProcessor: null when .none" {
    const json = try writePostProcessor(testing.allocator, .none, .{ .indent = 0 });
    defer testing.allocator.free(json);
    try testing.expectEqualStrings("null", json);
}

test "writePostProcessor: TemplateProcessing round-trips through ztok parser" {
    // Build a Bert-style template (single + pair) and write it. Then
    // re-parse via post_processor.parseFromJson and assert the resulting
    // PostProcessor applies identically to the original.
    const single_pieces = [_]PostProcessorSpec.Piece{
        .{ .kind = .special_token, .id = "[CLS]", .type_id = 0 },
        .{ .kind = .sequence_a, .type_id = 0 },
        .{ .kind = .special_token, .id = "[SEP]", .type_id = 0 },
    };
    const pair_pieces = [_]PostProcessorSpec.Piece{
        .{ .kind = .special_token, .id = "[CLS]", .type_id = 0 },
        .{ .kind = .sequence_a, .type_id = 0 },
        .{ .kind = .special_token, .id = "[SEP]", .type_id = 0 },
        .{ .kind = .sequence_b, .type_id = 1 },
        .{ .kind = .special_token, .id = "[SEP]", .type_id = 1 },
    };
    const cls_ids = [_]TokenId{101};
    const sep_ids = [_]TokenId{102};
    const cls_tokens = [_][]const u8{"[CLS]"};
    const sep_tokens = [_][]const u8{"[SEP]"};
    const specials = [_]PostProcessorSpec.SpecialToken{
        .{ .id = "[CLS]", .ids = &cls_ids, .tokens = &cls_tokens },
        .{ .id = "[SEP]", .ids = &sep_ids, .tokens = &sep_tokens },
    };
    const spec: PostProcessorSpec = .{ .template = .{
        .single = &single_pieces,
        .pair = &pair_pieces,
        .special_tokens = &specials,
    } };

    const json = try writePostProcessor(testing.allocator, spec, .{ .indent = 2 });
    defer testing.allocator.free(json);

    var parsed = try parseJsonValue(json);
    defer parsed.deinit();
    var pp = try post_processor.parseFromJson(testing.allocator, parsed.value);
    defer pp.deinit();
    try testing.expect(pp == .template);

    // Verify behaviour: applying to (5,6,7) yields [101, 5, 6, 7, 102].
    const ids_a = [_]TokenId{ 5, 6, 7 };
    var buf: [16]TokenId = undefined;
    const out_single = pp.applySingle(&ids_a, &buf, null);
    try testing.expectEqualSlices(TokenId, &[_]TokenId{ 101, 5, 6, 7, 102 }, out_single);

    // Pair: [101, 5, 6, 7, 102, 8, 9, 102]
    const ids_b = [_]TokenId{ 8, 9 };
    var pbuf: [32]TokenId = undefined;
    var tids: [32]u32 = undefined;
    const out_pair = try pp.applyPair(&ids_a, &ids_b, &pbuf, &tids);
    try testing.expectEqualSlices(TokenId, &[_]TokenId{ 101, 5, 6, 7, 102, 8, 9, 102 }, out_pair);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 0, 0, 0, 0, 1, 1, 1 }, tids[0..out_pair.len]);
}

test "writePostProcessor: BertProcessing round-trips through ztok parser" {
    const spec: PostProcessorSpec = .{ .bert = .{
        .sep = .{ .token = "[SEP]", .id = 102 },
        .cls = .{ .token = "[CLS]", .id = 101 },
    } };
    const json = try writePostProcessor(testing.allocator, spec, .{ .indent = 0 });
    defer testing.allocator.free(json);

    // Spot-check the minified shape — HF's exact emission per
    // refs/tokenizers/tokenizers/src/processors/bert.rs#serde test.
    try testing.expectEqualStrings(
        "{\"type\":\"BertProcessing\",\"sep\":[\"[SEP]\",102],\"cls\":[\"[CLS]\",101]}",
        json,
    );

    var parsed = try parseJsonValue(json);
    defer parsed.deinit();
    var pp = try post_processor.parseFromJson(testing.allocator, parsed.value);
    defer pp.deinit();
    try testing.expect(pp == .bert);
    try testing.expectEqual(@as(TokenId, 101), pp.bert.cls_id);
    try testing.expectEqual(@as(TokenId, 102), pp.bert.sep_id);
}

test "writePostProcessor: RobertaProcessing round-trip with non-default flags" {
    const spec: PostProcessorSpec = .{ .roberta = .{
        .sep = .{ .token = "</s>", .id = 2 },
        .cls = .{ .token = "<s>", .id = 0 },
        .trim_offsets = false,
        .add_prefix_space = false,
    } };
    const json = try writePostProcessor(testing.allocator, spec, .{ .indent = 0 });
    defer testing.allocator.free(json);

    // Spot-check the raw shape (per refs/tokenizers/.../processors/roberta.rs serde test).
    var parsed = try parseJsonValue(json);
    defer parsed.deinit();
    const obj = parsed.value.object;
    try testing.expectEqualStrings("RobertaProcessing", obj.get("type").?.string);

    // Now the ztok runtime parser knows RobertaProcessing; verify it
    // round-trips structurally.
    var pp = try post_processor.parseFromJson(testing.allocator, parsed.value);
    defer pp.deinit();
    try testing.expect(pp == .roberta);
    try testing.expectEqual(@as(TokenId, 0), pp.roberta.cls_id);
    try testing.expectEqual(@as(TokenId, 2), pp.roberta.sep_id);
    try testing.expectEqualStrings("<s>", pp.roberta.cls_token);
    try testing.expectEqualStrings("</s>", pp.roberta.sep_token);
    try testing.expectEqual(false, pp.roberta.trim_offsets);
    try testing.expectEqual(false, pp.roberta.add_prefix_space);
}

test "writePostProcessor: ByteLevel round-trip" {
    const spec: PostProcessorSpec = .{ .byte_level = .{
        .add_prefix_space = false,
        .trim_offsets = true,
        .use_regex = false,
    } };
    const json = try writePostProcessor(testing.allocator, spec, .{ .indent = 0 });
    defer testing.allocator.free(json);

    var parsed = try parseJsonValue(json);
    defer parsed.deinit();
    const obj = parsed.value.object;
    try testing.expectEqualStrings("ByteLevel", obj.get("type").?.string);

    var pp = try post_processor.parseFromJson(testing.allocator, parsed.value);
    defer pp.deinit();
    try testing.expect(pp == .byte_level);
    try testing.expectEqual(false, pp.byte_level.add_prefix_space);
    try testing.expectEqual(true, pp.byte_level.trim_offsets);
    try testing.expectEqual(false, pp.byte_level.use_regex);
}

test "writePostProcessor: Sequence chains sub-processors" {
    // ByteLevel (trim_offsets only) followed by RobertaProcessing — a
    // shape used by some real HF tokenizers to clean up byte-level
    // offsets before framing the sequence.
    const sub_byte_level: PostProcessorSpec = .{ .byte_level = .{
        .add_prefix_space = true,
        .trim_offsets = true,
        .use_regex = true,
    } };
    const sub_roberta: PostProcessorSpec = .{ .roberta = .{
        .sep = .{ .token = "</s>", .id = 2 },
        .cls = .{ .token = "<s>", .id = 0 },
    } };
    const subs = [_]PostProcessorSpec{ sub_byte_level, sub_roberta };
    const spec: PostProcessorSpec = .{ .sequence = .{ .processors = &subs } };

    const json = try writePostProcessor(testing.allocator, spec, .{ .indent = 0 });
    defer testing.allocator.free(json);

    var parsed = try parseJsonValue(json);
    defer parsed.deinit();
    const obj = parsed.value.object;
    try testing.expectEqualStrings("Sequence", obj.get("type").?.string);
    const procs = obj.get("processors").?.array;
    try testing.expectEqual(@as(usize, 2), procs.items.len);
    try testing.expectEqualStrings("ByteLevel", procs.items[0].object.get("type").?.string);
    try testing.expectEqualStrings("RobertaProcessing", procs.items[1].object.get("type").?.string);
    // And the inner Roberta carries default-ish flags.
    try testing.expectEqual(true, procs.items[1].object.get("trim_offsets").?.bool);
    try testing.expectEqual(true, procs.items[1].object.get("add_prefix_space").?.bool);

    // Now the ztok runtime parser knows Sequence; verify it round-trips
    // structurally and that apply delegates to the inner framer.
    var pp = try post_processor.parseFromJson(testing.allocator, parsed.value);
    defer pp.deinit();
    try testing.expect(pp == .sequence);
    try testing.expectEqual(@as(usize, 2), pp.sequence.processors.len);
    try testing.expect(pp.sequence.processors[0] == .byte_level);
    try testing.expect(pp.sequence.processors[1] == .roberta);
    const ids_a = [_]TokenId{ 12, 14 };
    var buf: [16]TokenId = undefined;
    const out = pp.applySingle(&ids_a, &buf, null);
    try testing.expectEqualSlices(TokenId, &[_]TokenId{ 0, 12, 14, 2 }, out);
}

test "writePostProcessor: real-world CLIP-style RobertaProcessing post_processor round-trip" {
    // The actual `post_processor` block from a real HuggingFace
    // `tokenizer.json` shipped with a CLIP/sam3-style image-text model
    // (verbatim from `/thearray/git/sam3/sam3/tokenizer.json` at the
    // time of writing). The Roberta variant is widely used outside
    // RoBERTa proper — CLIP's text tokenizer uses it with its own
    // <|startoftext|>/<|endoftext|> tokens and ids 49406/49407 instead
    // of <s>/</s> at 0/2.
    const real_fixture =
        \\{
        \\  "type": "RobertaProcessing",
        \\  "sep": ["<|endoftext|>", 49407],
        \\  "cls": ["<|startoftext|>", 49406],
        \\  "trim_offsets": false,
        \\  "add_prefix_space": false
        \\}
    ;
    var parsed = try parseJsonValue(real_fixture);
    defer parsed.deinit();
    var pp1 = try post_processor.parseFromJson(testing.allocator, parsed.value);
    defer pp1.deinit();
    try testing.expect(pp1 == .roberta);
    try testing.expectEqual(@as(TokenId, 49406), pp1.roberta.cls_id);
    try testing.expectEqual(@as(TokenId, 49407), pp1.roberta.sep_id);
    try testing.expectEqualStrings("<|startoftext|>", pp1.roberta.cls_token);
    try testing.expectEqualStrings("<|endoftext|>", pp1.roberta.sep_token);
    try testing.expectEqual(false, pp1.roberta.trim_offsets);
    try testing.expectEqual(false, pp1.roberta.add_prefix_space);

    // Round-trip via the writer.
    const spec: PostProcessorSpec = .{ .roberta = .{
        .sep = .{ .token = pp1.roberta.sep_token, .id = pp1.roberta.sep_id },
        .cls = .{ .token = pp1.roberta.cls_token, .id = pp1.roberta.cls_id },
        .trim_offsets = pp1.roberta.trim_offsets,
        .add_prefix_space = pp1.roberta.add_prefix_space,
    } };
    const json = try writePostProcessor(testing.allocator, spec, .{ .indent = 0 });
    defer testing.allocator.free(json);

    var parsed2 = try parseJsonValue(json);
    defer parsed2.deinit();
    var pp2 = try post_processor.parseFromJson(testing.allocator, parsed2.value);
    defer pp2.deinit();
    try testing.expect(pp2 == .roberta);
    try testing.expectEqual(pp1.roberta.cls_id, pp2.roberta.cls_id);
    try testing.expectEqual(pp1.roberta.sep_id, pp2.roberta.sep_id);
    try testing.expectEqualStrings(pp1.roberta.cls_token, pp2.roberta.cls_token);
    try testing.expectEqualStrings(pp1.roberta.sep_token, pp2.roberta.sep_token);
    try testing.expectEqual(pp1.roberta.trim_offsets, pp2.roberta.trim_offsets);
    try testing.expectEqual(pp1.roberta.add_prefix_space, pp2.roberta.add_prefix_space);

    // And the runtime apply produces the canonical Roberta framing
    // (single: [CLS] A [SEP]; pair: [CLS] A [SEP] [SEP] B [SEP]).
    const ids_a = [_]TokenId{ 320, 1125 };
    var buf: [16]TokenId = undefined;
    const out = pp2.applySingle(&ids_a, &buf, null);
    try testing.expectEqualSlices(TokenId, &[_]TokenId{ 49406, 320, 1125, 49407 }, out);
}
