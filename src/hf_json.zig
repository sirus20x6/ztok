//! Minimal HuggingFace tokenizer.json loader.
//!
//! Parses BPE, WordPiece, and Unigram model sections plus added_tokens,
//! and identifies which normalizer / pre-tokenizer / decoder variant is
//! in use so the integration layer can wire up the right pipeline stages.

const std = @import("std");
const TokenId = @import("token.zig").TokenId;
const Vocab = @import("vocab.zig").Vocab;
const hf_bytelevel = @import("hf_bytelevel_pretok.zig");
const hf_regex = @import("hf_regex.zig");

pub const AddedToken = struct {
    id: TokenId,
    content: []u8,
    special: bool,
    single_word: bool = false,
    lstrip: bool = false,
    rstrip: bool = false,
    normalized: bool = true,
};

pub const ModelKind = enum { bpe, wordpiece, unigram };

pub const NormalizerKind = enum { none, nfc, nfd, nfkc, nfkd, lowercase, sequence, bert, replace, strip, prepend, other };

/// Recursive HF normalizer spec captured from the `normalizer` section
/// of a tokenizer.json. The runtime `Normalizer` value is built from
/// this spec by `hf_bridge.normalizerFromHF`. Owns its strings + child
/// slices via `HFTokenizer.allocator`; `HFTokenizer.deinit` frees the
/// chain.
pub const HFNormalizerSpec = union(enum) {
    nfc,
    nfd,
    nfkc,
    nfkd,
    lowercase,
    strip: struct { strip_left: bool, strip_right: bool },
    bert: struct {
        clean_text: bool = true,
        handle_chinese_chars: bool = true,
        /// `null` mirrors HF's Option<bool> default ("follow lowercase").
        strip_accents: ?bool = null,
        lowercase: bool = true,
    },
    replace: struct {
        pattern: []u8,
        content: []u8,
        /// `true` when the source `pattern` field was tagged `Regex`
        /// rather than `String`. Bridge logs / promotes accordingly.
        is_regex: bool = false,
    },
    sequence: struct { items: []HFNormalizerSpec },
    /// HF `Prepend { prepend: "..." }` — SP-style literal prefix
    /// (post-1.18 agent B). Used inside `Sequence` chains by Phi-3 and
    /// other SP-reshelled HF tokenizers to prepend U+2581 before the
    /// `Replace(" "→"▁")` rewrites internal spaces.
    prepend: []u8,
    /// Any normalizer type we don't yet model (e.g. `Precompiled`,
    /// `Nmt`, `StripAccents`). The bridge drops these and records the
    /// name so the caller can warn.
    other: []u8,

    pub fn deinit(self: HFNormalizerSpec, allocator: std.mem.Allocator) void {
        switch (self) {
            .replace => |r| {
                if (r.pattern.len > 0) allocator.free(r.pattern);
                if (r.content.len > 0) allocator.free(r.content);
            },
            .sequence => |s| {
                for (s.items) |item| item.deinit(allocator);
                if (s.items.len > 0) allocator.free(s.items);
            },
            .prepend => |p| if (p.len > 0) allocator.free(p),
            .other => |o| if (o.len > 0) allocator.free(o),
            else => {},
        }
    }
};
pub const PreTokKind = enum { none, byte_level, whitespace, whitespace_split, metaspace, bert, sequence, other };
pub const DecoderKind = enum { none, byte_level, wordpiece, metaspace, sequence, other };

pub const MergePair = struct { left: TokenId, right: TokenId };

pub const HFTokenizer = struct {
    allocator: std.mem.Allocator,
    vocab: Vocab,
    merges: []MergePair,
    added_tokens: []AddedToken,
    model_kind: ModelKind,
    normalizer_kind: NormalizerKind,
    pre_tok_kind: PreTokKind,
    decoder_kind: DecoderKind,
    normalizer_other: ?[]u8 = null,
    pre_tok_other: ?[]u8 = null,
    decoder_other: ?[]u8 = null,
    /// Full HF normalizer spec, parsed when the tokenizer.json has a
    /// `normalizer` section. The legacy `normalizer_kind` enum still
    /// records the top-level type for back-compat; `normalizer_spec`
    /// gives the bridge layer the structured per-step config it needs
    /// to materialize the runtime `Normalizer` (Replace/Strip/Sequence
    /// /BertNormalizer aren't expressible as a flat enum). Owned by
    /// `allocator`; freed by `deinit`. Post-1.17 agent D.
    normalizer_spec: ?HFNormalizerSpec = null,
    byte_fallback: bool = false,
    fuse_unk: bool = false,
    ignore_merges: bool = false,
    unk_token: ?[]u8 = null,
    // Unigram-specific. `unigram_scores` is parallel to vocab ids.
    unigram_scores: ?[]f32 = null,
    unigram_unk_id: ?TokenId = null,
    /// Optional byte-fallback table populated when parsing a Unigram
    /// model that ships dedicated `<0xNN>` byte tokens for all 256
    /// bytes. Indexed by raw byte value -> token id. `null` when the
    /// vocab doesn't include the full byte bank — falling back to
    /// `unigram_unk_id` is safer than handing the encoder a half-
    /// populated table that silently emits wrong ids for missing
    /// bytes. Surfaced into `Unigram.byte_fallback` by
    /// `hf_bridge.unigramFromHF`.
    ///
    /// We trigger detection on either signal:
    ///   1. The model object's explicit `byte_fallback: bool` field
    ///      (set by HF's writer for SP-converted Unigram vocabs like
    ///      Gemma's HF export).
    ///   2. A scan of the vocab for `<0xNN>` pieces — populated even
    ///      if `byte_fallback` is absent, as long as all 256 bytes
    ///      are covered. Matches `sp_bridge.buildByteFallbackTable`.
    unigram_byte_fallback_table: ?[256]TokenId = null,
    // WordPiece-specific. Prefix is owned, defaults applied at parse time.
    wordpiece_continuing_subword_prefix: ?[]u8 = null,
    wordpiece_max_input_chars_per_word: u32 = 100,

    /// JSON-driven HF pretok chain, parsed when the `pre_tokenizer`
    /// section is a `Sequence` (or a single op we can model as a
    /// length-1 chain). Falcon-7B / Qwen2-7B / Llama-3-8B / many other
    /// post-GPT-2 BPE models carry per-model regex variants here.
    /// Owned via `allocator`; `deinit` frees the chain ops + regex
    /// programs. Post-1.18 agent A.
    pretok_chain: ?hf_bytelevel.Chain = null,

    /// True when this tokenizer.json is an SP-reshelled BPE — a
    /// SentencePiece BPE model that was converted to HF tokenizer.json
    /// without the GPT-2-style ByteLevel pretok / byte_to_unicode round
    /// trip. Phi-3-mini and Mistral-as-HF are canonical examples.
    ///
    /// Detection signature (all must hold):
    ///   - `model_kind == .bpe`
    ///   - `byte_fallback == true` (SP byte bank for unknown bytes)
    ///   - `merges.len > 0` (explicit merge ordering, SP-style)
    ///   - `pre_tok_kind == .none` AND `pretok_chain == null`
    ///     (no ByteLevel/regex pretok — GPT-2 style would set one)
    ///
    /// When true, `hf_bridge.bpeFromHF` configures the resulting `Bpe`
    /// with `encode_mode = .longest_match` + `piece_ranks` derived from
    /// the merge ordering, matching `sp_bridge.bpeFromSP`'s shape.
    /// When false, the historical `.bpe_merge` rank-by-id configuration
    /// applies — GPT-2 / Falcon / Qwen2 / Llama-3 / DeepSeek / Mistral
    /// HF-7B all stay on the existing path.
    ///
    /// Robustness: every regression-tested HF BPE vocab in the bench
    /// fails this signature for a structural reason:
    ///   - GPT-2 / DeepSeek-V2: `byte_fallback == false`
    ///   - Falcon-7B / Qwen2-7B / Llama-3-8B: `pretok_chain != null`
    ///     (ByteLevel/Split regex pretok present)
    /// so false positives that would mis-configure other vocabs are
    /// structurally excluded.
    pub fn isSpReshelled(self: *const HFTokenizer) bool {
        if (self.model_kind != .bpe) return false;
        if (!self.byte_fallback) return false;
        if (self.merges.len == 0) return false;
        if (self.pre_tok_kind != .none) return false;
        if (self.pretok_chain != null) return false;
        return true;
    }

    pub fn deinit(self: *HFTokenizer) void {
        self.vocab.deinit();
        if (self.merges.len > 0) self.allocator.free(self.merges);
        for (self.added_tokens) |*t| {
            if (t.content.len > 0) self.allocator.free(t.content);
        }
        if (self.added_tokens.len > 0) self.allocator.free(self.added_tokens);
        if (self.normalizer_other) |s| self.allocator.free(s);
        if (self.pre_tok_other) |s| self.allocator.free(s);
        if (self.decoder_other) |s| self.allocator.free(s);
        if (self.normalizer_spec) |spec| spec.deinit(self.allocator);
        if (self.pretok_chain) |*c| c.deinit();
        if (self.unk_token) |s| self.allocator.free(s);
        if (self.unigram_scores) |s| self.allocator.free(s);
        if (self.wordpiece_continuing_subword_prefix) |s| self.allocator.free(s);
    }
};

pub const Error = error{
    UnsupportedModel,
    MalformedJson,
    MissingField,
    DuplicateId,
} || std.mem.Allocator.Error || std.json.ParseError(std.json.Scanner);

pub fn loadFromBytes(allocator: std.mem.Allocator, json_bytes: []const u8) Error!HFTokenizer {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_bytes, .{});
    defer parsed.deinit();

    const root = parsed.value;
    if (root != .object) return error.MalformedJson;

    const model_v = root.object.get("model") orelse return error.MissingField;
    if (model_v != .object) return error.MalformedJson;

    // Most HF tokenizer.json files include an explicit `model.type`
    // (one of "BPE" / "WordPiece" / "Unigram"). Older exports like
    // OpenAI's gpt2 tokenizer.json predate that field — when it's
    // absent we infer from shape: a `merges` array means BPE; a
    // `vocab` array (not object) with parallel scores means Unigram;
    // a `vocab` object without `merges` falls through to WordPiece.
    const model_kind: ModelKind = blk: {
        if (model_v.object.get("type")) |type_v| {
            if (type_v != .string) return error.MalformedJson;
            if (std.mem.eql(u8, type_v.string, "BPE")) break :blk .bpe;
            if (std.mem.eql(u8, type_v.string, "WordPiece")) break :blk .wordpiece;
            if (std.mem.eql(u8, type_v.string, "Unigram")) break :blk .unigram;
            return error.UnsupportedModel;
        }
        // No explicit type — infer.
        if (model_v.object.get("merges") != null) break :blk .bpe;
        if (model_v.object.get("vocab")) |vv| {
            if (vv == .array) break :blk .unigram;
            if (vv == .object) break :blk .wordpiece;
        }
        return error.UnsupportedModel;
    };

    // Build the holder progressively; on any failure deinit unwinds.
    var hf: HFTokenizer = .{
        .allocator = allocator,
        .vocab = .empty(allocator),
        .merges = &.{},
        .added_tokens = &.{},
        .model_kind = model_kind,
        .normalizer_kind = .none,
        .pre_tok_kind = .none,
        .decoder_kind = .none,
    };
    errdefer hf.deinit();

    switch (model_kind) {
        .bpe => try parseModelBpe(&hf, model_v.object),
        .wordpiece => try parseModelWordPiece(&hf, model_v.object),
        .unigram => try parseModelUnigram(&hf, model_v.object),
    }
    try parseAddedTokens(&hf, root.object);
    try parseNormalizer(&hf, root.object);
    try parsePreTokenizer(&hf, root.object);
    try parseDecoder(&hf, root.object);

    return hf;
}

pub fn loadFromFile(allocator: std.mem.Allocator, path: []const u8) !HFTokenizer {
    const io = std.Io.Threaded.global_single_threaded.io();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(
        io,
        path,
        allocator,
        .limited(std.math.maxInt(usize)),
    );
    defer allocator.free(bytes);
    return loadFromBytes(allocator, bytes);
}

// ---------------------------------------------------------------------
// Parsing helpers. They take the user allocator (via `hf.allocator`) and
// produce owned arrays. The std.json arena owns the borrowed views we
// read from; we copy out before it dies.

fn parseModelBpe(hf: *HFTokenizer, model: std.json.ObjectMap) Error!void {
    const vocab_v = model.get("vocab") orelse return error.MissingField;
    if (vocab_v != .object) return error.MalformedJson;

    try buildVocabFromIdMap(hf, vocab_v.object);

    // Optional BPE flags.
    if (model.get("byte_fallback")) |v| if (v == .bool) {
        hf.byte_fallback = v.bool;
    };
    if (model.get("fuse_unk")) |v| if (v == .bool) {
        hf.fuse_unk = v.bool;
    };
    if (model.get("ignore_merges")) |v| if (v == .bool) {
        hf.ignore_merges = v.bool;
    };
    if (model.get("unk_token")) |v| if (v == .string) {
        hf.unk_token = try hf.allocator.dupe(u8, v.string);
    };

    // Merges (optional but expected for BPE).
    if (model.get("merges")) |merges_v| {
        if (merges_v != .array) return error.MalformedJson;
        try parseMerges(hf, vocab_v.object, merges_v.array);
    }
}

fn parseModelWordPiece(hf: *HFTokenizer, model: std.json.ObjectMap) Error!void {
    const vocab_v = model.get("vocab") orelse return error.MissingField;
    if (vocab_v != .object) return error.MalformedJson;

    try buildVocabFromIdMap(hf, vocab_v.object);

    // Prefix defaults to "##" per HF.
    if (model.get("continuing_subword_prefix")) |v| {
        if (v != .string) return error.MalformedJson;
        hf.wordpiece_continuing_subword_prefix = try hf.allocator.dupe(u8, v.string);
    } else {
        hf.wordpiece_continuing_subword_prefix = try hf.allocator.dupe(u8, "##");
    }

    if (model.get("max_input_chars_per_word")) |v| {
        if (v != .integer or v.integer < 0) return error.MalformedJson;
        hf.wordpiece_max_input_chars_per_word = @intCast(v.integer);
    }

    if (model.get("byte_fallback")) |v| if (v == .bool) {
        hf.byte_fallback = v.bool;
    };

    if (model.get("unk_token")) |v| if (v == .string) {
        hf.unk_token = try hf.allocator.dupe(u8, v.string);
    };
}

// HF Unigram vocab is `[[piece, score], ...]` with id = array position.
// We build the SoA Vocab directly off the array; no densification dance
// needed because indexing is positional.
fn parseModelUnigram(hf: *HFTokenizer, model: std.json.ObjectMap) Error!void {
    const vocab_v = model.get("vocab") orelse return error.MissingField;
    if (vocab_v != .array) return error.MalformedJson;
    const items = vocab_v.array.items;

    if (items.len == 0) {
        hf.vocab = .empty(hf.allocator);
    } else {
        const count: u32 = @intCast(items.len);

        // Validate up front + compute total byte length.
        var total_bytes: usize = 0;
        for (items) |entry| {
            if (entry != .array) return error.MalformedJson;
            if (entry.array.items.len != 2) return error.MalformedJson;
            const piece_v = entry.array.items[0];
            const score_v = entry.array.items[1];
            if (piece_v != .string) return error.MalformedJson;
            if (score_v != .integer and score_v != .float) return error.MalformedJson;
            total_bytes += piece_v.string.len;
        }

        const bytes = try hf.allocator.alloc(u8, total_bytes);
        errdefer hf.allocator.free(bytes);
        const offsets = try hf.allocator.alloc(u32, count + 1);
        errdefer hf.allocator.free(offsets);
        const scores = try hf.allocator.alloc(f32, count);
        errdefer hf.allocator.free(scores);

        var off: u32 = 0;
        for (items, 0..) |entry, i| {
            const piece = entry.array.items[0].string;
            const score_v = entry.array.items[1];
            const score: f32 = switch (score_v) {
                .integer => |x| @floatFromInt(x),
                .float => |x| @floatCast(x),
                else => unreachable,
            };
            offsets[i] = off;
            @memcpy(bytes[off .. off + piece.len], piece);
            off += @intCast(piece.len);
            scores[i] = score;
        }
        offsets[count] = off;

        hf.vocab = .{
            .allocator = hf.allocator,
            .bytes = bytes,
            .offsets = offsets,
            .ranks = null,
            .count = count,
        };
        hf.unigram_scores = scores;
    }

    if (model.get("unk_id")) |v| {
        if (v == .integer and v.integer >= 0) {
            hf.unigram_unk_id = @intCast(v.integer);
        }
    }

    if (model.get("byte_fallback")) |v| if (v == .bool) {
        hf.byte_fallback = v.bool;
    };

    // Scan for `<0xNN>` byte tokens and build the [256] table if all
    // 256 bytes are covered. Same gating rule as
    // `sp_bridge.buildByteFallbackTable`: a partial bank is treated as
    // "no byte_fallback" and we leave the field null. This populates
    // the table whether or not the model object set the explicit
    // `byte_fallback: bool` field — some HF-converted SP vocabs ship
    // the byte bank without the flag, and the table itself is the
    // source of truth for the encoder.
    hf.unigram_byte_fallback_table = buildUnigramByteFallbackTable(hf);
}

/// Scan the parsed Unigram vocab for `<0xNN>` byte-token names and
/// pack their ids into a `[256]TokenId` table indexed by the raw byte
/// value. Returns null if fewer than 256 distinct byte tokens are
/// present (i.e. the vocab isn't a real byte-fallback vocab).
fn buildUnigramByteFallbackTable(hf: *const HFTokenizer) ?[256]TokenId {
    var table: [256]TokenId = undefined;
    @memset(&table, std.math.maxInt(TokenId));
    var found: u32 = 0;
    var id: u32 = 0;
    while (id < hf.vocab.count) : (id += 1) {
        const name = hf.vocab.tokenBytes(id);
        const b = parseByteTokenName(name) orelse continue;
        if (table[b] == std.math.maxInt(TokenId)) {
            table[b] = id;
            found += 1;
        }
    }
    if (found < 256) return null;
    return table;
}

/// Parse `<0xNN>` -> byte. SP's writer uses uppercase hex; HF's
/// converter preserves that. Same convention as
/// `sp_bridge.parseByteTokenName`.
fn parseByteTokenName(s: []const u8) ?u8 {
    if (s.len != 6) return null;
    if (s[0] != '<' or s[1] != '0' or s[2] != 'x' or s[5] != '>') return null;
    const hi = hexNibble(s[3]) orelse return null;
    const lo = hexNibble(s[4]) orelse return null;
    return (hi << 4) | lo;
}

fn hexNibble(c: u8) ?u8 {
    return switch (c) {
        '0'...'9' => c - '0',
        'A'...'F' => 10 + (c - 'A'),
        else => null,
    };
}

// Shared between BPE and WordPiece: vocab is `{ "piece": id, ... }` with
// dense ids 0..count-1. Builds hf.vocab.
fn buildVocabFromIdMap(hf: *HFTokenizer, vocab_obj: std.json.ObjectMap) Error!void {
    var count: u32 = 0;
    var total_bytes: usize = 0;
    var max_id: i64 = -1;

    var it = vocab_obj.iterator();
    while (it.next()) |e| {
        const id_v = e.value_ptr.*;
        if (id_v != .integer or id_v.integer < 0) return error.MalformedJson;
        if (id_v.integer > max_id) max_id = id_v.integer;
        count += 1;
        total_bytes += e.key_ptr.len;
    }
    if (count == 0) {
        hf.vocab = .empty(hf.allocator);
        return;
    }
    if (max_id != @as(i64, count) - 1) return error.MalformedJson; // ids must be dense 0..count-1

    const bytes = try hf.allocator.alloc(u8, total_bytes);
    errdefer hf.allocator.free(bytes);
    const offsets = try hf.allocator.alloc(u32, count + 1);
    errdefer hf.allocator.free(offsets);

    // Pass 2: index by id, then lay out.
    var tmp_arena = std.heap.ArenaAllocator.init(hf.allocator);
    defer tmp_arena.deinit();
    const ids_to_keys = try tmp_arena.allocator().alloc([]const u8, count);
    for (ids_to_keys) |*p| p.* = "";

    var seen = try tmp_arena.allocator().alloc(bool, count);
    @memset(seen, false);

    it = vocab_obj.iterator();
    while (it.next()) |e| {
        const id: u32 = @intCast(e.value_ptr.integer);
        if (seen[id]) return error.DuplicateId;
        seen[id] = true;
        ids_to_keys[id] = e.key_ptr.*;
    }

    var off: u32 = 0;
    for (ids_to_keys, 0..) |key, i| {
        offsets[i] = off;
        @memcpy(bytes[off .. off + key.len], key);
        off += @intCast(key.len);
    }
    offsets[count] = off;

    hf.vocab = .{
        .allocator = hf.allocator,
        .bytes = bytes,
        .offsets = offsets,
        .ranks = null,
        .count = count,
    };
}

fn parseMerges(
    hf: *HFTokenizer,
    vocab_obj: std.json.ObjectMap,
    merges_arr: std.json.Array,
) Error!void {
    const n = merges_arr.items.len;
    if (n == 0) return;

    const out = try hf.allocator.alloc(MergePair, n);
    errdefer hf.allocator.free(out);

    for (merges_arr.items, 0..) |entry, i| {
        switch (entry) {
            .string => |s| {
                // Legacy "A B" form: split on first space.
                const sp = std.mem.indexOfScalar(u8, s, ' ') orelse return error.MalformedJson;
                const left_s = s[0..sp];
                const right_s = s[sp + 1 ..];
                out[i] = .{
                    .left = lookupId(vocab_obj, left_s) orelse return error.MalformedJson,
                    .right = lookupId(vocab_obj, right_s) orelse return error.MalformedJson,
                };
            },
            .array => |a| {
                if (a.items.len != 2) return error.MalformedJson;
                if (a.items[0] != .string or a.items[1] != .string) return error.MalformedJson;
                out[i] = .{
                    .left = lookupId(vocab_obj, a.items[0].string) orelse return error.MalformedJson,
                    .right = lookupId(vocab_obj, a.items[1].string) orelse return error.MalformedJson,
                };
            },
            else => return error.MalformedJson,
        }
    }

    hf.merges = out;
}

fn lookupId(vocab_obj: std.json.ObjectMap, key: []const u8) ?TokenId {
    const v = vocab_obj.get(key) orelse return null;
    if (v != .integer or v.integer < 0) return null;
    return @intCast(v.integer);
}

fn parseAddedTokens(hf: *HFTokenizer, root: std.json.ObjectMap) Error!void {
    const v = root.get("added_tokens") orelse return;
    if (v != .array) return error.MalformedJson;
    if (v.array.items.len == 0) return;

    const out = try hf.allocator.alloc(AddedToken, v.array.items.len);
    var filled: usize = 0;
    errdefer {
        for (out[0..filled]) |*t| if (t.content.len > 0) hf.allocator.free(t.content);
        hf.allocator.free(out);
    }

    for (v.array.items, 0..) |entry, i| {
        if (entry != .object) return error.MalformedJson;
        const id_v = entry.object.get("id") orelse return error.MissingField;
        const content_v = entry.object.get("content") orelse return error.MissingField;
        if (id_v != .integer or id_v.integer < 0) return error.MalformedJson;
        if (content_v != .string) return error.MalformedJson;

        var t: AddedToken = .{
            .id = @intCast(id_v.integer),
            .content = try hf.allocator.dupe(u8, content_v.string),
            .special = false,
        };
        if (entry.object.get("special")) |b| if (b == .bool) {
            t.special = b.bool;
        };
        if (entry.object.get("single_word")) |b| if (b == .bool) {
            t.single_word = b.bool;
        };
        if (entry.object.get("lstrip")) |b| if (b == .bool) {
            t.lstrip = b.bool;
        };
        if (entry.object.get("rstrip")) |b| if (b == .bool) {
            t.rstrip = b.bool;
        };
        if (entry.object.get("normalized")) |b| if (b == .bool) {
            t.normalized = b.bool;
        };

        out[i] = t;
        filled = i + 1;
    }

    hf.added_tokens = out;
}

fn parseNormalizer(hf: *HFTokenizer, root: std.json.ObjectMap) Error!void {
    const v = root.get("normalizer") orelse return;
    if (v == .null) return;
    if (v != .object) return error.MalformedJson;
    const type_v = v.object.get("type") orelse return error.MissingField;
    if (type_v != .string) return error.MalformedJson;
    const s = type_v.string;
    if (std.mem.eql(u8, s, "NFC")) {
        hf.normalizer_kind = .nfc;
    } else if (std.mem.eql(u8, s, "NFD")) {
        hf.normalizer_kind = .nfd;
    } else if (std.mem.eql(u8, s, "NFKC")) {
        hf.normalizer_kind = .nfkc;
    } else if (std.mem.eql(u8, s, "NFKD")) {
        hf.normalizer_kind = .nfkd;
    } else if (std.mem.eql(u8, s, "Lowercase")) {
        hf.normalizer_kind = .lowercase;
    } else if (std.mem.eql(u8, s, "Sequence")) {
        hf.normalizer_kind = .sequence;
    } else if (std.mem.eql(u8, s, "BertNormalizer")) {
        hf.normalizer_kind = .bert;
    } else if (std.mem.eql(u8, s, "Replace")) {
        hf.normalizer_kind = .replace;
    } else if (std.mem.eql(u8, s, "Strip")) {
        hf.normalizer_kind = .strip;
    } else if (std.mem.eql(u8, s, "Prepend")) {
        hf.normalizer_kind = .prepend;
    } else {
        hf.normalizer_kind = .other;
        hf.normalizer_other = try hf.allocator.dupe(u8, s);
    }
    // Parse the full recursive spec so the bridge can build a runtime
    // Normalizer value (Replace/Strip/Sequence/Bert require fields the
    // flat `normalizer_kind` enum doesn't carry). Independent of the
    // back-compat `normalizer_kind` assignment above so older callers
    // keep working.
    if (try parseNormalizerSpec(hf.allocator, v)) |spec| {
        hf.normalizer_spec = spec;
    }
}

/// Parse a HF normalizer JSON value (typed by `{"type": "..."}`) into
/// the recursive `HFNormalizerSpec`. Returns `null` for normalizer
/// types we don't model — the bridge treats null as identity.
fn parseNormalizerSpec(
    allocator: std.mem.Allocator,
    v: std.json.Value,
) Error!?HFNormalizerSpec {
    if (v != .object) return null;
    const type_v = v.object.get("type") orelse return null;
    if (type_v != .string) return null;
    const s = type_v.string;
    if (std.mem.eql(u8, s, "NFC")) return .nfc;
    if (std.mem.eql(u8, s, "NFD")) return .nfd;
    if (std.mem.eql(u8, s, "NFKC")) return .nfkc;
    if (std.mem.eql(u8, s, "NFKD")) return .nfkd;
    if (std.mem.eql(u8, s, "Lowercase")) return .lowercase;
    if (std.mem.eql(u8, s, "Strip")) {
        var left: bool = true;
        var right: bool = true;
        if (v.object.get("strip_left")) |b| if (b == .bool) {
            left = b.bool;
        };
        if (v.object.get("strip_right")) |b| if (b == .bool) {
            right = b.bool;
        };
        return .{ .strip = .{ .strip_left = left, .strip_right = right } };
    }
    if (std.mem.eql(u8, s, "BertNormalizer")) {
        var cfg = HFNormalizerSpec{ .bert = .{} };
        if (v.object.get("clean_text")) |b| if (b == .bool) {
            cfg.bert.clean_text = b.bool;
        };
        if (v.object.get("handle_chinese_chars")) |b| if (b == .bool) {
            cfg.bert.handle_chinese_chars = b.bool;
        };
        if (v.object.get("strip_accents")) |b| switch (b) {
            .bool => |x| {
                cfg.bert.strip_accents = x;
            },
            .null => {},
            else => {},
        };
        if (v.object.get("lowercase")) |b| if (b == .bool) {
            cfg.bert.lowercase = b.bool;
        };
        return cfg;
    }
    if (std.mem.eql(u8, s, "Replace")) {
        // HF Replace shape:
        //   { type: "Replace", pattern: { String | Regex: "..." }, content: "..." }
        //
        // `pattern` is preserved verbatim regardless of whether the
        // source said String or Regex. The bridge layer reads
        // `is_regex` and feeds the verbatim pattern to
        // `hf_regex.compile` for Regex sources; for String sources the
        // runtime normalizer matches `pattern` literally.
        const content_v = v.object.get("content") orelse return error.MissingField;
        if (content_v != .string) return error.MalformedJson;
        const pat_v = v.object.get("pattern") orelse return error.MissingField;
        if (pat_v != .object) return error.MalformedJson;
        var pattern: []u8 = &.{};
        errdefer if (pattern.len > 0) allocator.free(pattern);
        var is_regex = false;
        if (pat_v.object.get("String")) |ps| {
            if (ps != .string) return error.MalformedJson;
            pattern = try allocator.dupe(u8, ps.string);
        } else if (pat_v.object.get("Regex")) |pr| {
            if (pr != .string) return error.MalformedJson;
            pattern = try allocator.dupe(u8, pr.string);
            is_regex = true;
        } else return error.MalformedJson;
        const content = try allocator.dupe(u8, content_v.string);
        return .{ .replace = .{ .pattern = pattern, .content = content, .is_regex = is_regex } };
    }
    if (std.mem.eql(u8, s, "Sequence")) {
        const list_v = v.object.get("normalizers") orelse return error.MissingField;
        if (list_v != .array) return error.MalformedJson;
        const items_in = list_v.array.items;
        // Two-pass: parse each, drop nulls (unmodeled types), copy
        // remaining into a flat slice.
        var tmp: std.ArrayList(HFNormalizerSpec) = .empty;
        errdefer {
            for (tmp.items) |it| it.deinit(allocator);
            tmp.deinit(allocator);
        }
        for (items_in) |item_v| {
            const child = try parseNormalizerSpec(allocator, item_v);
            if (child) |c| try tmp.append(allocator, c);
        }
        const items = try tmp.toOwnedSlice(allocator);
        return .{ .sequence = .{ .items = items } };
    }
    if (std.mem.eql(u8, s, "Prepend")) {
        // HF Prepend shape: { type: "Prepend", prepend: "..." }
        // Post-1.18 agent B — unblocks Phi-3's
        // Sequence[Prepend("▁"), Replace(" "→"▁")] chain.
        const pre_v = v.object.get("prepend") orelse return error.MissingField;
        if (pre_v != .string) return error.MalformedJson;
        return .{ .prepend = try allocator.dupe(u8, pre_v.string) };
    }
    // Anything else (Precompiled, Nmt, StripAccents, ByteLevel
    // normalizer — note pre-tok ByteLevel is separate): record the name
    // and let the bridge decide what to do. We keep this distinct from
    // the .other variant so callers can distinguish "no spec at all"
    // from "spec exists but mentions an unmodeled type".
    return .{ .other = try allocator.dupe(u8, s) };
}

fn parsePreTokenizer(hf: *HFTokenizer, root: std.json.ObjectMap) Error!void {
    const v = root.get("pre_tokenizer") orelse return;
    if (v == .null) return;
    if (v != .object) return error.MalformedJson;
    const type_v = v.object.get("type") orelse return error.MissingField;
    if (type_v != .string) return error.MalformedJson;
    const s = type_v.string;
    if (std.mem.eql(u8, s, "ByteLevel")) {
        hf.pre_tok_kind = .byte_level;
    } else if (std.mem.eql(u8, s, "Whitespace")) {
        hf.pre_tok_kind = .whitespace;
    } else if (std.mem.eql(u8, s, "WhitespaceSplit")) {
        hf.pre_tok_kind = .whitespace_split;
    } else if (std.mem.eql(u8, s, "Metaspace")) {
        hf.pre_tok_kind = .metaspace;
    } else if (std.mem.eql(u8, s, "BertPreTokenizer")) {
        hf.pre_tok_kind = .bert;
    } else if (std.mem.eql(u8, s, "Sequence")) {
        hf.pre_tok_kind = .sequence;
    } else {
        hf.pre_tok_kind = .other;
        hf.pre_tok_other = try hf.allocator.dupe(u8, s);
    }
    // Also build the structured chain so the bridge can materialize a
    // runtime PreTokenizer.chain when the JSON describes a multi-op
    // Sequence (or any single op modeled by `hf_bytelevel.Chain`).
    if (try parsePretokChain(hf.allocator, v)) |chain| {
        hf.pretok_chain = chain;
    }
}

/// Parse the structured pretok JSON into a `hf_bytelevel.Chain`. Returns
/// null when the JSON describes a single op already covered by the
/// flat-enum dispatcher (so the bridge can keep using the cheap path).
fn parsePretokChain(
    allocator: std.mem.Allocator,
    v: std.json.Value,
) Error!?hf_bytelevel.Chain {
    var ops: std.ArrayList(hf_bytelevel.PretokOp) = .empty;
    errdefer {
        for (ops.items) |op| switch (op) {
            .split => |sp| {
                sp.re.deinit();
                allocator.destroy(sp.re);
            },
            else => {},
        };
        ops.deinit(allocator);
    }
    try collectPretokOps(allocator, v, &ops);
    if (ops.items.len == 0) return null;
    const owned = try ops.toOwnedSlice(allocator);
    return .{ .allocator = allocator, .ops = owned };
}

fn collectPretokOps(
    allocator: std.mem.Allocator,
    v: std.json.Value,
    ops: *std.ArrayList(hf_bytelevel.PretokOp),
) Error!void {
    if (v != .object) return;
    const type_v = v.object.get("type") orelse return;
    if (type_v != .string) return;
    const s = type_v.string;
    if (std.mem.eql(u8, s, "Sequence")) {
        const list = v.object.get("pretokenizers") orelse return;
        if (list != .array) return error.MalformedJson;
        for (list.array.items) |child| try collectPretokOps(allocator, child, ops);
        return;
    }
    if (std.mem.eql(u8, s, "Split")) {
        const pat_v = v.object.get("pattern") orelse return error.MissingField;
        if (pat_v != .object) return error.MalformedJson;
        var pattern: []const u8 = "";
        if (pat_v.object.get("Regex")) |pr| {
            if (pr != .string) return error.MalformedJson;
            pattern = pr.string;
        } else if (pat_v.object.get("String")) |ps| {
            if (ps != .string) return error.MalformedJson;
            pattern = ps.string;
        } else return error.MalformedJson;
        const behavior = parseBehavior(v.object.get("behavior")) orelse .Isolated;
        var invert = false;
        if (v.object.get("invert")) |b| if (b == .bool) {
            invert = b.bool;
        };

        // Compile the pattern. For literal `String` patterns we must
        // escape any regex metacharacters before compiling.
        const escaped = if (pat_v.object.get("String") != null)
            try escapeLiteralForRegex(allocator, pattern)
        else
            try allocator.dupe(u8, pattern);
        defer allocator.free(escaped);

        const re_ptr = try allocator.create(hf_regex.Regex);
        errdefer allocator.destroy(re_ptr);
        re_ptr.* = hf_regex.compile(allocator, escaped) catch |err| {
            // Compile failure: skip this op rather than aborting the
            // whole load. Surfaces upstream as the same diverging
            // behavior we had before — the model still loads.
            std.log.warn("hf_json: pretok Split regex compile failed ({s}): {s}", .{ @errorName(err), pattern });
            allocator.destroy(re_ptr);
            return;
        };
        try ops.append(allocator, .{ .split = .{ .re = re_ptr, .behavior = behavior, .invert = invert } });
        return;
    }
    if (std.mem.eql(u8, s, "ByteLevel")) {
        var add_prefix_space = false;
        var use_regex = true;
        var trim_offsets = true;
        if (v.object.get("add_prefix_space")) |b| if (b == .bool) {
            add_prefix_space = b.bool;
        };
        if (v.object.get("use_regex")) |b| if (b == .bool) {
            use_regex = b.bool;
        };
        if (v.object.get("trim_offsets")) |b| if (b == .bool) {
            trim_offsets = b.bool;
        };
        try ops.append(allocator, .{ .byte_level = .{
            .add_prefix_space = add_prefix_space,
            .use_regex = use_regex,
            .trim_offsets = trim_offsets,
        } });
        return;
    }
    if (std.mem.eql(u8, s, "Digits")) {
        var individual = false;
        if (v.object.get("individual_digits")) |b| if (b == .bool) {
            individual = b.bool;
        };
        try ops.append(allocator, .{ .digits = .{ .individual_digits = individual } });
        return;
    }
    if (std.mem.eql(u8, s, "Punctuation")) {
        const behavior = parseBehavior(v.object.get("behavior")) orelse .Isolated;
        try ops.append(allocator, .{ .punctuation = behavior });
        return;
    }
    if (std.mem.eql(u8, s, "Whitespace")) {
        try ops.append(allocator, .{ .whitespace = {} });
        return;
    }
    if (std.mem.eql(u8, s, "WhitespaceSplit")) {
        try ops.append(allocator, .{ .whitespace_split = {} });
        return;
    }
    if (std.mem.eql(u8, s, "Metaspace")) {
        var rep_cp: u21 = 0x2581;
        var add_prefix_space = true;
        if (v.object.get("replacement")) |rs| if (rs == .string and rs.string.len > 0) {
            // Decode first codepoint of the replacement string.
            const slen = std.unicode.utf8ByteSequenceLength(rs.string[0]) catch 1;
            if (slen <= rs.string.len) {
                rep_cp = std.unicode.utf8Decode(rs.string[0..slen]) catch rep_cp;
            }
        };
        if (v.object.get("prepend_scheme")) |ps| if (ps == .string) {
            // HF prepend_scheme values: "always", "never", "first".
            if (std.mem.eql(u8, ps.string, "never")) add_prefix_space = false;
        };
        if (v.object.get("add_prefix_space")) |b| if (b == .bool) {
            add_prefix_space = b.bool;
        };
        try ops.append(allocator, .{ .metaspace = .{ .replacement_cp = rep_cp, .add_prefix_space = add_prefix_space } });
        return;
    }
    if (std.mem.eql(u8, s, "BertPreTokenizer")) {
        try ops.append(allocator, .{ .bert = {} });
        return;
    }
    // Unrecognized op: drop silently. The flat enum dispatcher already
    // records the name via `pre_tok_other` for diagnostics.
}

fn parseBehavior(v_opt: ?std.json.Value) ?hf_bytelevel.SplitBehavior {
    const v = v_opt orelse return null;
    if (v != .string) return null;
    if (std.mem.eql(u8, v.string, "Removed")) return .Removed;
    if (std.mem.eql(u8, v.string, "Isolated")) return .Isolated;
    if (std.mem.eql(u8, v.string, "MergedWithPrevious")) return .MergedWithPrevious;
    if (std.mem.eql(u8, v.string, "MergedWithNext")) return .MergedWithNext;
    if (std.mem.eql(u8, v.string, "Contiguous")) return .Contiguous;
    return null;
}

/// Escape regex meta-characters in a literal `String` pattern so the
/// regex compiler treats them as literals. HF's `String` patterns are
/// raw byte strings — never regex.
fn escapeLiteralForRegex(allocator: std.mem.Allocator, src: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.ensureUnusedCapacity(allocator, src.len * 2);
    for (src) |b| {
        switch (b) {
            '\\', '.', '+', '*', '?', '(', ')', '[', ']', '{', '}', '|', '^', '$' => {
                try out.append(allocator, '\\');
                try out.append(allocator, b);
            },
            else => try out.append(allocator, b),
        }
    }
    return try out.toOwnedSlice(allocator);
}

fn parseDecoder(hf: *HFTokenizer, root: std.json.ObjectMap) Error!void {
    const v = root.get("decoder") orelse return;
    if (v == .null) return;
    if (v != .object) return error.MalformedJson;
    const type_v = v.object.get("type") orelse return error.MissingField;
    if (type_v != .string) return error.MalformedJson;
    const s = type_v.string;
    if (std.mem.eql(u8, s, "ByteLevel")) {
        hf.decoder_kind = .byte_level;
    } else if (std.mem.eql(u8, s, "WordPiece")) {
        hf.decoder_kind = .wordpiece;
    } else if (std.mem.eql(u8, s, "Metaspace")) {
        hf.decoder_kind = .metaspace;
    } else if (std.mem.eql(u8, s, "Sequence")) {
        hf.decoder_kind = .sequence;
    } else {
        hf.decoder_kind = .other;
        hf.decoder_other = try hf.allocator.dupe(u8, s);
    }
}

// ---------------------------------------------------------------------
// Tests

const testing = std.testing;

test "loadFromBytes parses minimal BPE" {
    const input =
        \\{
        \\  "version": "1.0",
        \\  "added_tokens": [],
        \\  "normalizer": null,
        \\  "pre_tokenizer": null,
        \\  "decoder": null,
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {"a": 0, "b": 1, "ab": 2, "c": 3},
        \\    "merges": [["a", "b"]]
        \\  }
        \\}
    ;
    var hf = try loadFromBytes(testing.allocator, input);
    defer hf.deinit();

    try testing.expectEqual(ModelKind.bpe, hf.model_kind);
    try testing.expectEqual(@as(u32, 4), hf.vocab.count);
    try testing.expectEqualStrings("a", hf.vocab.tokenBytes(0));
    try testing.expectEqualStrings("b", hf.vocab.tokenBytes(1));
    try testing.expectEqualStrings("ab", hf.vocab.tokenBytes(2));
    try testing.expectEqualStrings("c", hf.vocab.tokenBytes(3));
    try testing.expectEqual(@as(usize, 1), hf.merges.len);
    try testing.expectEqual(@as(TokenId, 0), hf.merges[0].left);
    try testing.expectEqual(@as(TokenId, 1), hf.merges[0].right);
    try testing.expectEqual(NormalizerKind.none, hf.normalizer_kind);
    try testing.expectEqual(PreTokKind.none, hf.pre_tok_kind);
    try testing.expectEqual(DecoderKind.none, hf.decoder_kind);
}

test "loadFromBytes legacy merges format" {
    const input =
        \\{
        \\  "version": "1.0",
        \\  "added_tokens": [],
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {"a": 0, "b": 1, "ab": 2, "c": 3},
        \\    "merges": ["a b"]
        \\  }
        \\}
    ;
    var hf = try loadFromBytes(testing.allocator, input);
    defer hf.deinit();

    try testing.expectEqual(@as(usize, 1), hf.merges.len);
    try testing.expectEqual(@as(TokenId, 0), hf.merges[0].left);
    try testing.expectEqual(@as(TokenId, 1), hf.merges[0].right);
}

test "loadFromBytes UnsupportedModel for unknown model type" {
    // BPE/WordPiece/Unigram are now all supported; an unknown model type
    // (e.g. WordLevel, not implemented here) should still reject.
    const input =
        \\{
        \\  "added_tokens": [],
        \\  "model": { "type": "WordLevel", "vocab": {} }
        \\}
    ;
    try testing.expectError(error.UnsupportedModel, loadFromBytes(testing.allocator, input));
}

test "loadFromBytes parses added_tokens" {
    const input =
        \\{
        \\  "added_tokens": [
        \\    {"id": 0, "content": "<unk>", "special": true, "single_word": false, "normalized": false},
        \\    {"id": 1, "content": "hello", "special": false}
        \\  ],
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {"<unk>": 0, "hello": 1},
        \\    "merges": []
        \\  }
        \\}
    ;
    var hf = try loadFromBytes(testing.allocator, input);
    defer hf.deinit();

    try testing.expectEqual(@as(usize, 2), hf.added_tokens.len);
    try testing.expectEqualStrings("<unk>", hf.added_tokens[0].content);
    try testing.expect(hf.added_tokens[0].special);
    try testing.expect(!hf.added_tokens[0].normalized);
    try testing.expectEqualStrings("hello", hf.added_tokens[1].content);
    try testing.expect(!hf.added_tokens[1].special);
}

test "loadFromBytes captures normalizer/pre_tok/decoder kinds" {
    const input =
        \\{
        \\  "added_tokens": [],
        \\  "normalizer": {"type": "NFC"},
        \\  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": true},
        \\  "decoder": {"type": "Metaspace"},
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {"a": 0},
        \\    "merges": []
        \\  }
        \\}
    ;
    var hf = try loadFromBytes(testing.allocator, input);
    defer hf.deinit();

    try testing.expectEqual(NormalizerKind.nfc, hf.normalizer_kind);
    try testing.expectEqual(PreTokKind.byte_level, hf.pre_tok_kind);
    try testing.expectEqual(DecoderKind.metaspace, hf.decoder_kind);
    try testing.expectEqual(@as(?[]u8, null), hf.normalizer_other);
    try testing.expectEqual(@as(?[]u8, null), hf.pre_tok_other);
    try testing.expectEqual(@as(?[]u8, null), hf.decoder_other);
}

test "loadFromBytes parses minimal Unigram" {
    const input =
        \\{
        \\  "added_tokens": [],
        \\  "model": {
        \\    "type": "Unigram",
        \\    "unk_id": 0,
        \\    "vocab": [
        \\      ["<unk>", 0.0],
        \\      ["the", -3.14],
        \\      ["a", -3.5]
        \\    ],
        \\    "byte_fallback": false
        \\  }
        \\}
    ;
    var hf = try loadFromBytes(testing.allocator, input);
    defer hf.deinit();

    try testing.expectEqual(ModelKind.unigram, hf.model_kind);
    try testing.expectEqual(@as(u32, 3), hf.vocab.count);
    try testing.expectEqualStrings("<unk>", hf.vocab.tokenBytes(0));
    try testing.expectEqualStrings("the", hf.vocab.tokenBytes(1));
    try testing.expectEqualStrings("a", hf.vocab.tokenBytes(2));
    try testing.expect(hf.unigram_scores != null);
    const scores = hf.unigram_scores.?;
    try testing.expectEqual(@as(usize, 3), scores.len);
    try testing.expectApproxEqAbs(@as(f32, 0.0), scores[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, -3.14), scores[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, -3.5), scores[2], 1e-6);
    try testing.expectEqual(@as(?TokenId, 0), hf.unigram_unk_id);
}

test "loadFromBytes parses minimal WordPiece" {
    const input =
        \\{
        \\  "added_tokens": [],
        \\  "model": {
        \\    "type": "WordPiece",
        \\    "unk_token": "[UNK]",
        \\    "continuing_subword_prefix": "##",
        \\    "max_input_chars_per_word": 100,
        \\    "vocab": {
        \\      "[UNK]": 0,
        \\      "[CLS]": 1,
        \\      "the": 2,
        \\      "##s": 3
        \\    }
        \\  }
        \\}
    ;
    var hf = try loadFromBytes(testing.allocator, input);
    defer hf.deinit();

    try testing.expectEqual(ModelKind.wordpiece, hf.model_kind);
    try testing.expectEqual(@as(u32, 4), hf.vocab.count);
    try testing.expectEqualStrings("[UNK]", hf.vocab.tokenBytes(0));
    try testing.expectEqualStrings("[CLS]", hf.vocab.tokenBytes(1));
    try testing.expectEqualStrings("the", hf.vocab.tokenBytes(2));
    try testing.expectEqualStrings("##s", hf.vocab.tokenBytes(3));
    try testing.expect(hf.wordpiece_continuing_subword_prefix != null);
    try testing.expectEqualStrings("##", hf.wordpiece_continuing_subword_prefix.?);
    try testing.expectEqual(@as(u32, 100), hf.wordpiece_max_input_chars_per_word);
    try testing.expect(hf.unk_token != null);
    try testing.expectEqualStrings("[UNK]", hf.unk_token.?);
}

test "loadFromBytes WordPiece custom prefix" {
    const input =
        \\{
        \\  "added_tokens": [],
        \\  "model": {
        \\    "type": "WordPiece",
        \\    "unk_token": "[UNK]",
        \\    "continuing_subword_prefix": ">>",
        \\    "max_input_chars_per_word": 64,
        \\    "vocab": {"[UNK]": 0, "foo": 1, ">>bar": 2}
        \\  }
        \\}
    ;
    var hf = try loadFromBytes(testing.allocator, input);
    defer hf.deinit();

    try testing.expectEqual(ModelKind.wordpiece, hf.model_kind);
    try testing.expect(hf.wordpiece_continuing_subword_prefix != null);
    try testing.expectEqualStrings(">>", hf.wordpiece_continuing_subword_prefix.?);
    try testing.expectEqual(@as(u32, 64), hf.wordpiece_max_input_chars_per_word);
}
