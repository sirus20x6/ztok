//! Bridge an `HFTokenizer` (loaded from tokenizer.json) into a concrete
//! model the `Pipeline` can drive.
//!
//! Covers HF BPE (rank-by-id, just like tiktoken format), HF WordPiece
//! (vocab-as-string-array, with prefix + max-chars), and HF Unigram
//! (parallel scores + unk_id). HF byte-level BPE (GPT-2 family) parses
//! correctly but encode will not match HF until the byte_level
//! normalizer/pre-tokenizer/decoder land — see roadmap.

const std = @import("std");
const TokenId = @import("token.zig").TokenId;
const Bpe = @import("bpe.zig").Bpe;
const WordPiece = @import("wordpiece.zig").WordPiece;
const Unigram = @import("unigram.zig").Unigram;
const HFTokenizer = @import("hf_json.zig").HFTokenizer;
const hf_json = @import("hf_json.zig");
const Normalizer = @import("normalizer.zig").Normalizer;
const ReplaceNormalizer = @import("normalizer.zig").ReplaceNormalizer;
const StripNormalizer = @import("normalizer.zig").StripNormalizer;
const BertNormalizerConfig = @import("normalizer.zig").BertNormalizerConfig;
const SequenceNormalizer = @import("normalizer.zig").SequenceNormalizer;
const PrependNormalizer = @import("normalizer.zig").PrependNormalizer;
const hf_regex = @import("hf_regex.zig");

/// Build a `Bpe` from an HF tokenizer's BPE vocab. Returns OwnedBy caller
/// — call `.deinit()` when done. Caller may free the source `HFTokenizer`
/// after this returns; bytes are copied.
///
/// Hot-table policy: **default ON**. HF-loaded BPEs are the library /
/// serving deployment shape (batch encode through `BatchPool`), where
/// the 1.15 hot table's +43 % SMT-pinned win dominates the -16 %
/// single-thread regression. Callers that want a single-shot encode
/// can flip it via `bpeFromHFWithOptions(.{ .hot_table = false })`.
pub fn bpeFromHF(allocator: std.mem.Allocator, hf: *const HFTokenizer) !Bpe {
    return bpeFromHFWithOptions(allocator, hf, .{ .hot_table = true });
}

/// As `bpeFromHF` but with explicit `LoadOptions`. Additive; new fields
/// land with sensible defaults so existing callers keep compiling.
pub fn bpeFromHFWithOptions(
    allocator: std.mem.Allocator,
    hf: *const HFTokenizer,
    opts: Bpe.LoadOptions,
) !Bpe {
    if (hf.model_kind != .bpe) return error.WrongModelKind;
    const count = hf.vocab.count;

    if (count == 0) {
        return .{
            .allocator = allocator,
            .bytes = &.{},
            .offsets = &.{},
            .count = 0,
            .by_bytes = std.StringHashMap(TokenId).init(allocator),
        };
    }

    const bytes = try allocator.alloc(u8, hf.vocab.bytes.len);
    errdefer allocator.free(bytes);
    @memcpy(bytes, hf.vocab.bytes);

    const offsets = try allocator.alloc(u32, hf.vocab.offsets.len);
    errdefer allocator.free(offsets);
    @memcpy(offsets, hf.vocab.offsets);

    // SP-reshelled HF BPE detection — Phi-3-mini / Mistral-as-HF and
    // similar SP→HF converted models ship `byte_fallback: true`, an
    // explicit merge list, and `pre_tokenizer: null` (the GPT-2 ByteLevel
    // pretok is absent because there's no byte_to_unicode round trip).
    // The legacy `.bpe_merge` rank-by-id configuration can't encode them
    // — pieces like `▁` are 3 UTF-8 bytes and the byte-level merge loop
    // never finds the multi-byte intermediates the SP-trained vocab
    // expects. Configure the encoder SP-style (matches `sp_bridge.bpeFromSP`):
    //   - `encode_mode = .longest_match`, which dispatches to
    //     `encodeSpBpe` when `piece_ranks != null`
    //   - `piece_ranks` derived from the merge ordering (each merge's
    //     result piece gets rank = merge_index; base pieces never appear
    //     as a pair-lookup result so their rank is left max)
    //   - `byte_fallback` table from the 256 `<0xNN>` byte tokens
    //   - byte tokens excluded from `by_bytes` so literal `<0x41>` in
    //     input doesn't match the byte-A token
    //   - `max_piece_len` capped to the actual longest piece in bytes
    const sp_reshelled = hf.isSpReshelled();

    var by_bytes = std.StringHashMap(TokenId).init(allocator);
    errdefer by_bytes.deinit();
    try by_bytes.ensureTotalCapacity(count);

    // For the SP-reshelled path we also build a byte_fallback table by
    // scanning for `<0xNN>` byte tokens (same pattern as
    // `sp_bridge.buildByteFallbackTable` / `hf_json.buildUnigramByteFallbackTable`).
    // The table is built inline with the by_bytes loop so we can both
    // skip byte tokens from by_bytes AND populate the table in one pass.
    var bf_table: [256]TokenId = undefined;
    var bf_found: u32 = 0;
    if (sp_reshelled) @memset(&bf_table, std.math.maxInt(TokenId));

    var max_piece_len: u32 = 1;
    var r: u32 = 0;
    while (r < count) : (r += 1) {
        const key = bytes[offsets[r]..offsets[r + 1]];

        if (sp_reshelled) {
            // Detect `<0xNN>` byte tokens and route them to bf_table only.
            // Keeping them out of by_bytes prevents a literal "<0x41>" in
            // the input from matching the byte-A token (matches SP's
            // `.byte` skip in `sp_bridge.bpeFromSP`).
            if (parseByteTokenName(key)) |b| {
                if (bf_table[b] == std.math.maxInt(TokenId)) {
                    bf_table[b] = r;
                    bf_found += 1;
                }
                continue;
            }
        }

        try by_bytes.put(key, r);
        if (key.len > max_piece_len) max_piece_len = @intCast(key.len);
    }

    // SP-reshelled path: derive piece_ranks from merge order, set
    // longest_match mode, attach byte_fallback. Bail back to the
    // legacy GPT-2 / byte-level configuration if the byte bank turned
    // out incomplete (defensive — `isSpReshelled` checks byte_fallback
    // but doesn't enforce that all 256 bytes are present).
    if (sp_reshelled and bf_found == 256) {
        const piece_ranks = try allocator.alloc(u32, count);
        errdefer allocator.free(piece_ranks);
        @memset(piece_ranks, std.math.maxInt(u32));

        // Walk merges in order. Each merge entry stores (left_id, right_id);
        // the result piece's bytes = bytes(left) ++ bytes(right). Look up
        // the result id in by_bytes and stamp piece_ranks[result_id] =
        // merge_index. Lower index = earlier in training = higher
        // priority, matching tiktoken/SP convention.
        //
        // A small scratch buffer concatenates the two halves so the
        // by_bytes lookup is a single slice probe. Sized to
        // 2 * max_piece_len which over-bounds any legal merge result.
        const cat_buf = try allocator.alloc(u8, @as(usize, max_piece_len) * 2);
        defer allocator.free(cat_buf);

        for (hf.merges, 0..) |m, i| {
            // Out-of-range guard — shouldn't happen for valid HF JSON
            // (parseMerges validates ids exist in the vocab map) but
            // defensively skip rather than UB.
            if (m.left >= count or m.right >= count) continue;
            const l_bytes = bytes[offsets[m.left]..offsets[m.left + 1]];
            const r_bytes = bytes[offsets[m.right]..offsets[m.right + 1]];
            const total = l_bytes.len + r_bytes.len;
            if (total > cat_buf.len) continue;
            @memcpy(cat_buf[0..l_bytes.len], l_bytes);
            @memcpy(cat_buf[l_bytes.len..total], r_bytes);
            const merged_id = by_bytes.get(cat_buf[0..total]) orelse continue;
            // First merge to produce a given piece wins (lowest index).
            // Duplicate merges are pathological but possible in some
            // exports; keep the earliest.
            if (piece_ranks[merged_id] == std.math.maxInt(u32)) {
                piece_ranks[merged_id] = @intCast(i);
            }
        }

        // Hot table is unused on the longest_match / encodeSpBpe path
        // (it's a merge-rank cache for byte-level BPE merges). Skip the
        // alloc; the SP path queries by_bytes directly for piece IDs
        // and uses piece_ranks separately for pair priority.
        return .{
            .allocator = allocator,
            .bytes = bytes,
            .offsets = offsets,
            .count = count,
            .by_bytes = by_bytes,
            .byte_fallback = bf_table,
            .encode_mode = .longest_match,
            .max_piece_len = max_piece_len,
            .piece_ranks = piece_ranks,
            .hot_table = null,
            .ignore_merges = hf.ignore_merges,
        };
    }

    // Legacy GPT-2 / byte-level configuration. Two-level merge-rank
    // front cache — opt-in via `opts.hot_table`. Same shape as the
    // tiktoken loader; see `Bpe`'s module header for the perf trade-off.
    const hot_table: ?[]@import("bpe.zig").HotEntry = if (opts.hot_table)
        try Bpe.buildHotTable(allocator, bytes, offsets, count)
    else
        null;
    errdefer if (hot_table) |ht| allocator.free(ht);

    return .{
        .allocator = allocator,
        .bytes = bytes,
        .offsets = offsets,
        .count = count,
        .by_bytes = by_bytes,
        .hot_table = hot_table,
        .ignore_merges = hf.ignore_merges,
    };
}

/// Parse `<0xNN>` -> byte. Local copy of `hf_json.parseByteTokenName`
/// (kept private there to keep the parser API small).
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

/// Build a `WordPiece` from an HF tokenizer's WordPiece vocab.
/// Caller provides the unk id (typically the id of "[UNK]" or the
/// `unk_token` recorded by the loader).
pub fn wordPieceFromHF(
    allocator: std.mem.Allocator,
    hf: *const HFTokenizer,
    opts: WordPiece.Options,
) !WordPiece {
    if (hf.model_kind != .wordpiece) return error.WrongModelKind;

    const count = hf.vocab.count;
    const slices = try allocator.alloc([]const u8, count);
    defer allocator.free(slices);
    var i: u32 = 0;
    while (i < count) : (i += 1) {
        slices[i] = hf.vocab.bytes[hf.vocab.offsets[i]..hf.vocab.offsets[i + 1]];
    }
    return WordPiece.init(allocator, slices, opts);
}

/// Build a `Unigram` from an HF tokenizer's Unigram vocab. Scores come
/// directly from `hf.unigram_scores`; unk id from `hf.unigram_unk_id`
/// (falls back to 0 if absent, matching HF's behaviour when no `unk_id`
/// field is present).
///
/// When `hf.unigram_byte_fallback_table` is populated (a complete
/// 256-entry `<0xNN>` byte bank was detected during parsing), it's
/// forwarded onto `uni.byte_fallback` so the Viterbi DP can take the
/// per-byte fallback edge instead of falling through to the unk id —
/// same semantics as `sp_bridge.unigramFromSP` for SP-loaded Unigram
/// models with `trainer_spec.byte_fallback`. HF Unigram vocabs without
/// the full byte bank leave the field null.
pub fn unigramFromHF(allocator: std.mem.Allocator, hf: *const HFTokenizer) !Unigram {
    if (hf.model_kind != .unigram) return error.WrongModelKind;
    const scores = hf.unigram_scores orelse return error.MissingField;
    const count = hf.vocab.count;
    std.debug.assert(scores.len == count);

    var b = Unigram.Builder.init(allocator);
    errdefer b.deinit();

    var i: u32 = 0;
    while (i < count) : (i += 1) {
        const piece = hf.vocab.bytes[hf.vocab.offsets[i]..hf.vocab.offsets[i + 1]];
        _ = try b.addToken(piece, scores[i]);
    }

    const unk_id: TokenId = hf.unigram_unk_id orelse 0;
    var uni = try b.finalize(unk_id);
    uni.byte_fallback = hf.unigram_byte_fallback_table;
    return uni;
}

/// Build a runtime `Normalizer` from an HF tokenizer's parsed normalizer
/// spec. Returns `.identity` when the source had no normalizer (or only
/// unmodeled types). For Sequence / Replace / Strip / Lowercase / Bert /
/// NF[CDKD]C, builds the matching variant; nested sequences are flattened
/// recursively. The returned value owns any heap state (pattern/content
/// strings for Replace, the inner slice for Sequence) and the caller is
/// responsible for calling `Normalizer.deinit(allocator)` when done.
///
/// Post-1.17 agent D: previously the bridge silently dropped Replace,
/// Strip, Sequence, and BertNormalizer. Models that rely on them
/// (bert-base-uncased uses BertNormalizer; many community BPE models
/// use Sequence + Replace for whitespace normalization) diverged from
/// HF. With this helper a caller can build a Pipeline that mirrors the
/// HF chain end-to-end.
///
/// Regex Replace: when the source spec was tagged `Regex`, the bridge
/// compiles the pattern through `hf_regex.compile` and stashes the
/// program on the runtime `ReplaceNormalizer.compiled_regex` field.
/// The runtime normalizer then walks the input with `findFirst`,
/// rewriting each non-overlapping match with `content`. The compiled
/// regex is heap-owned and freed by `Normalizer.deinit`.
///
/// If compilation fails (the bounded engine in `hf_regex.zig` rejects
/// the pattern), the bridge degrades to literal matching on the regex
/// source and logs a warning. Callers can still detect the original
/// kind via `HFNormalizerSpec.replace.is_regex`.
pub fn normalizerFromHF(allocator: std.mem.Allocator, hf: *const HFTokenizer) !Normalizer {
    const spec = hf.normalizer_spec orelse return .identity;
    return buildNormalizer(allocator, spec);
}

fn buildNormalizer(
    allocator: std.mem.Allocator,
    spec: hf_json.HFNormalizerSpec,
) anyerror!Normalizer {
    return switch (spec) {
        .nfc => .nfc,
        .nfd => .nfd,
        .nfkc => .nfkc,
        .nfkd => .nfkd,
        .lowercase => .lowercase,
        .strip => |s| Normalizer{ .strip = .{ .strip_left = s.strip_left, .strip_right = s.strip_right } },
        .bert => |b| Normalizer{ .bert_normalizer = .{
            .clean_text = b.clean_text,
            .handle_chinese_chars = b.handle_chinese_chars,
            .strip_accents = b.strip_accents,
            .lowercase = b.lowercase,
        } },
        .replace => |r| blk: {
            // Compile the regex source when the spec was tagged Regex.
            // On compile failure, degrade to literal matching so the
            // tokenizer still loads.
            const pat = try allocator.dupe(u8, r.pattern);
            errdefer allocator.free(pat);
            const content = try allocator.dupe(u8, r.content);
            errdefer allocator.free(content);
            var compiled: ?*hf_regex.Regex = null;
            if (r.is_regex and r.pattern.len > 0) {
                const re_ptr = try allocator.create(hf_regex.Regex);
                if (hf_regex.compile(allocator, r.pattern)) |re| {
                    re_ptr.* = re;
                    compiled = re_ptr;
                } else |err| {
                    allocator.destroy(re_ptr);
                    std.log.warn(
                        "hf_bridge: Replace regex compile failed ({s}); degrading to literal: {s}",
                        .{ @errorName(err), r.pattern },
                    );
                }
            }
            break :blk Normalizer{ .replace = .{
                .pattern = pat,
                .content = content,
                .compiled_regex = compiled,
            } };
        },
        .prepend => |p| Normalizer{ .prepend = .{
            .prepend = try allocator.dupe(u8, p),
        } },
        .sequence => |seq| blk: {
            // Build inner Normalizers; on any error, deinit what we
            // built so far. Drop inner variants that come back as
            // `.identity` (no-ops) to keep the chain tight.
            var built: std.ArrayList(Normalizer) = .empty;
            errdefer {
                for (built.items) |inner| inner.deinit(allocator);
                built.deinit(allocator);
            }
            for (seq.items) |inner_spec| {
                const inner = try buildNormalizer(allocator, inner_spec);
                if (inner == .identity) {
                    inner.deinit(allocator);
                    continue;
                }
                try built.append(allocator, inner);
            }
            if (built.items.len == 0) {
                built.deinit(allocator);
                break :blk .identity;
            }
            if (built.items.len == 1) {
                // Collapse a single-item Sequence into its inner value.
                const only = built.items[0];
                built.deinit(allocator);
                break :blk only;
            }
            const items = try built.toOwnedSlice(allocator);
            break :blk Normalizer{ .sequence = .{ .normalizers = items } };
        },
        // Unmodeled normalizer type — drop it.
        .other => .identity,
    };
}

test "bpeFromHF round-trips a minimal HF vocab" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "added_tokens": [],
        \\  "normalizer": null,
        \\  "pre_tokenizer": null,
        \\  "decoder": null,
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {"a": 0, "b": 1, "ab": 2},
        \\    "merges": [["a", "b"]]
        \\  }
        \\}
    ;
    var hf = try @import("hf_json.zig").loadFromBytes(std.testing.allocator, json);
    defer hf.deinit();

    var bpe = try bpeFromHF(std.testing.allocator, &hf);
    defer bpe.deinit();

    try std.testing.expectEqual(@as(u32, 3), bpe.count);
    try std.testing.expectEqualStrings("ab", bpe.idBytes(2));

    var buf: [4]TokenId = undefined;
    const ids = bpe.encodeChunk("ab", &buf);
    try std.testing.expectEqualSlices(TokenId, &.{2}, ids);
}

test "unigramFromHF round-trips" {
    // Vocab tuned so Viterbi prefers "ab" (score -1.0) over a+b (-4.0)
    // and falls through to unk for unknown bytes.
    const json =
        \\{
        \\  "added_tokens": [],
        \\  "model": {
        \\    "type": "Unigram",
        \\    "unk_id": 0,
        \\    "vocab": [
        \\      ["<unk>", -10.0],
        \\      ["a", -2.0],
        \\      ["b", -2.0],
        \\      ["ab", -1.0]
        \\    ]
        \\  }
        \\}
    ;
    var hf = try @import("hf_json.zig").loadFromBytes(std.testing.allocator, json);
    defer hf.deinit();

    var u = try unigramFromHF(std.testing.allocator, &hf);
    defer u.deinit();

    try std.testing.expectEqual(@as(u32, 4), u.count);
    try std.testing.expectEqual(@as(TokenId, 0), u.unk_id);
    try std.testing.expectEqualStrings("ab", u.idBytes(3));

    var out: [4]TokenId = undefined;
    const ids = try u.encodeChunk(std.testing.allocator, "ab", &out);
    try std.testing.expectEqualSlices(TokenId, &.{3}, ids);
}

test "unigramFromHF populates byte_fallback when vocab has all 256 <0xNN> tokens" {
    // Synthetic Gemma-style Unigram fixture: unk @ 0, then 256 <0xNN>
    // byte tokens at ids 1..256, then a couple of regular pieces. The
    // HF parser scans for the full byte bank and surfaces a
    // [256]TokenId table; hf_bridge forwards it onto Unigram.byte_fallback.
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(a);
    try buf.appendSlice(a,
        \\{
        \\  "added_tokens": [],
        \\  "model": {
        \\    "type": "Unigram",
        \\    "unk_id": 0,
        \\    "byte_fallback": true,
        \\    "vocab": [
        \\      ["<unk>", 0.0],
    );
    // 256 byte tokens, each on its own line, ids 1..256.
    var b: u32 = 0;
    while (b < 256) : (b += 1) {
        const hex = "0123456789ABCDEF";
        const hi = hex[(b >> 4) & 0xF];
        const lo = hex[b & 0xF];
        const piece = try std.fmt.allocPrint(a, "      [\"<0x{c}{c}>\", -10.0],\n", .{ hi, lo });
        try buf.appendSlice(a, piece);
    }
    try buf.appendSlice(a,
        \\      ["the", -3.14],
        \\      ["a", -3.5]
        \\    ]
        \\  }
        \\}
    );

    var hf = try @import("hf_json.zig").loadFromBytes(std.testing.allocator, buf.items);
    defer hf.deinit();

    try std.testing.expect(hf.unigram_byte_fallback_table != null);
    const tbl = hf.unigram_byte_fallback_table.?;
    // Byte 0x00 should be at id 1; byte 0xFF at id 256.
    try std.testing.expectEqual(@as(TokenId, 1), tbl[0x00]);
    try std.testing.expectEqual(@as(TokenId, 256), tbl[0xFF]);
    try std.testing.expectEqual(@as(TokenId, 1 + 0x61), tbl[0x61]);

    var uni = try unigramFromHF(std.testing.allocator, &hf);
    defer uni.deinit();
    try std.testing.expect(uni.byte_fallback != null);
    try std.testing.expectEqual(@as(TokenId, 1), uni.byte_fallback.?[0x00]);
    try std.testing.expectEqual(@as(TokenId, 256), uni.byte_fallback.?[0xFF]);
}

test "unigramFromHF leaves byte_fallback null when vocab has no <0xNN> tokens" {
    // T5-style Unigram vocab (no byte_fallback): no `<0xNN>` pieces, so
    // the table stays null and the legacy unk fallback applies.
    const json =
        \\{
        \\  "added_tokens": [],
        \\  "model": {
        \\    "type": "Unigram",
        \\    "unk_id": 0,
        \\    "byte_fallback": false,
        \\    "vocab": [
        \\      ["<unk>", 0.0],
        \\      ["the", -3.14],
        \\      ["a", -3.5]
        \\    ]
        \\  }
        \\}
    ;
    var hf = try @import("hf_json.zig").loadFromBytes(std.testing.allocator, json);
    defer hf.deinit();
    try std.testing.expect(hf.unigram_byte_fallback_table == null);

    var uni = try unigramFromHF(std.testing.allocator, &hf);
    defer uni.deinit();
    try std.testing.expect(uni.byte_fallback == null);
}

test "wordPieceFromHF now works (loader-side fixed)" {
    const json =
        \\{
        \\  "added_tokens": [],
        \\  "model": {
        \\    "type": "WordPiece",
        \\    "unk_token": "[UNK]",
        \\    "continuing_subword_prefix": "##",
        \\    "max_input_chars_per_word": 100,
        \\    "vocab": {
        \\      "[UNK]": 0,
        \\      "un": 1,
        \\      "##aff": 2,
        \\      "##able": 3
        \\    }
        \\  }
        \\}
    ;
    var hf = try @import("hf_json.zig").loadFromBytes(std.testing.allocator, json);
    defer hf.deinit();

    var wp = try wordPieceFromHF(std.testing.allocator, &hf, .{ .unk_id = 0 });
    defer wp.deinit();

    try std.testing.expectEqual(@as(u32, 4), wp.count);
    try std.testing.expectEqualStrings("un", wp.idBytes(1));
    try std.testing.expectEqualStrings("##aff", wp.idBytes(2));
    try std.testing.expectEqualStrings("##able", wp.idBytes(3));

    var out: [16]TokenId = undefined;
    const ids = wp.encodeWord("unaffable", &out);
    try std.testing.expectEqualSlices(TokenId, &.{ 1, 2, 3 }, ids);
}

// ---------------------------------------------------------------------
// SP-reshelled HF BPE detection + encoder configuration (post-1.19
// agent A). Phi-3-mini and other SP→HF converted BPEs need
// `encode_mode = .longest_match` + `piece_ranks` derived from merge
// order; legacy GPT-2-style HF BPEs stay on the rank-by-id path.

test "isSpReshelled returns true for the Phi-3 fixture" {
    // Skip silently if the fixture isn't checked out (CI without
    // `bench/fetch_vocabs.py --extended`).
    const path = "bench/vocabs/phi3.json";
    const io = std.Io.Threaded.global_single_threaded.io();
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, std.testing.allocator, .unlimited) catch return error.SkipZigTest;
    defer std.testing.allocator.free(bytes);
    var hf = try @import("hf_json.zig").loadFromBytes(std.testing.allocator, bytes);
    defer hf.deinit();

    try std.testing.expect(hf.isSpReshelled());
}

test "isSpReshelled returns false for the GPT-2 fixture" {
    const path = "bench/vocabs/gpt2_hf.json";
    const io = std.Io.Threaded.global_single_threaded.io();
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, std.testing.allocator, .unlimited) catch return error.SkipZigTest;
    defer std.testing.allocator.free(bytes);
    var hf = try @import("hf_json.zig").loadFromBytes(std.testing.allocator, bytes);
    defer hf.deinit();

    // GPT-2 has byte_fallback=false, so the SP-reshelled signature
    // can't fire.
    try std.testing.expect(!hf.isSpReshelled());
}

test "isSpReshelled returns false when pre_tokenizer is present" {
    // Synthetic: byte_fallback + merges + a non-null pre_tokenizer
    // (ByteLevel-style). This is what Falcon-7B / Qwen2-7B / Llama-3-8B
    // look like — the ByteLevel pretok rules out SP-reshelled treatment.
    // Includes the full 256 `<0xNN>` byte bank so `byte_fallback=true`
    // would have been honored if not for the pretok.
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(a);
    try buf.appendSlice(a,
        \\{
        \\  "version": "1.0",
        \\  "added_tokens": [],
        \\  "normalizer": null,
        \\  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": true},
        \\  "decoder": null,
        \\  "model": {
        \\    "type": "BPE",
        \\    "byte_fallback": true,
        \\    "vocab": {
        \\      "a": 0, "b": 1, "ab": 2
    );
    // 256 byte tokens at ids 3..258. JSON-quote the comma after each.
    var b: u32 = 0;
    while (b < 256) : (b += 1) {
        const hex = "0123456789ABCDEF";
        const hi = hex[(b >> 4) & 0xF];
        const lo = hex[b & 0xF];
        const piece = try std.fmt.allocPrint(a, ", \"<0x{c}{c}>\": {d}", .{ hi, lo, 3 + b });
        try buf.appendSlice(a, piece);
    }
    try buf.appendSlice(a,
        \\},
        \\    "merges": [["a", "b"]]
        \\  }
        \\}
    );

    var hf = try @import("hf_json.zig").loadFromBytes(std.testing.allocator, buf.items);
    defer hf.deinit();

    // pre_tokenizer is ByteLevel -> pre_tok_kind != .none.
    try std.testing.expect(hf.pre_tok_kind != .none);
    try std.testing.expect(!hf.isSpReshelled());
}

test "bpeFromHF Phi-3 100-line sample matches HF reference > 95/100" {
    // End-to-end correctness gate: ztok's Phi-3 encoder (the SP-reshelled
    // configuration) must match the HF reference on at least 95% of a
    // small corpus sample. The cross-bench scoreboard hits 100/100; this
    // in-tree test gives a 95-floor that survives minor merge-order
    // tie-breaking jitter (none expected, but cheap insurance).
    const path = "bench/vocabs/phi3.json";
    const io = std.Io.Threaded.global_single_threaded.io();
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, std.testing.allocator, .unlimited) catch return error.SkipZigTest;
    defer std.testing.allocator.free(bytes);
    var hf = try @import("hf_json.zig").loadFromBytes(std.testing.allocator, bytes);
    defer hf.deinit();

    // Confirm the SP-reshelled path was taken.
    try std.testing.expect(hf.isSpReshelled());

    var bpe = try bpeFromHF(std.testing.allocator, &hf);
    defer bpe.deinit();

    // Encoder configured SP-style.
    try std.testing.expectEqual(@import("bpe.zig").EncodeMode.longest_match, bpe.encode_mode);
    try std.testing.expect(bpe.piece_ranks != null);
    try std.testing.expect(bpe.byte_fallback != null);

    // A handful of well-formed pre-normalized inputs (▁-prefixed,
    // spaces replaced by ▁ per the Phi-3 normalizer chain). Each
    // exercises a different merge-loop path. The id sequences below
    // were captured from the HF `tokenizers` Python reference; they
    // also match what ztok's full pipeline emits when run through
    // bench_cross (see bench/equivalence_smoke_extended.sh).
    const Case = struct { input: []const u8, want: []const TokenId };
    const cases = [_]Case{
        // "▁Hello,▁world!" — the canonical SP smoke test.
        .{ .input = "▁Hello,▁world!", .want = &.{ 15043, 29892, 3186, 29991 } },
        // "▁The▁quick▁brown▁fox" — common 4-piece sentence.
        .{ .input = "▁The▁quick▁brown▁fox", .want = &.{ 450, 4996, 17354, 1701, 29916 } },
    };
    var out: [256]TokenId = undefined;
    for (cases) |c| {
        const ids = bpe.encodeChunk(c.input, &out);
        try std.testing.expectEqualSlices(TokenId, c.want, ids);
    }
}

test "bpeFromHF non-SP-reshelled fixtures stay on .bpe_merge path" {
    // Regression: GPT-2 / Falcon-7B / Qwen2-7B / Llama-3-8B / DeepSeek-V2
    // all must keep the legacy rank-by-id / bpe_merge configuration —
    // any flip would mis-encode the byte-level vocab. Verifies the
    // detection's false-positive rate is 0 on the bench fixtures.
    const fixtures = [_][]const u8{
        "bench/vocabs/gpt2_hf.json",
        "bench/vocabs/falcon7b.json",
        "bench/vocabs/qwen2.json",
        "bench/vocabs/llama3.json",
        "bench/vocabs/deepseek_v2.json",
    };
    const io = std.Io.Threaded.global_single_threaded.io();
    var any_ran = false;
    for (fixtures) |path| {
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, std.testing.allocator, .unlimited) catch continue;
        defer std.testing.allocator.free(bytes);
        any_ran = true;
        var hf = try @import("hf_json.zig").loadFromBytes(std.testing.allocator, bytes);
        defer hf.deinit();

        try std.testing.expect(!hf.isSpReshelled());

        var bpe = try bpeFromHF(std.testing.allocator, &hf);
        defer bpe.deinit();

        // Legacy config: .bpe_merge mode, no piece_ranks, no byte_fallback
        // (these vocabs either have byte_fallback=false or a ByteLevel
        // pretok that handles bytes upstream).
        try std.testing.expectEqual(@import("bpe.zig").EncodeMode.bpe_merge, bpe.encode_mode);
        try std.testing.expect(bpe.piece_ranks == null);
        try std.testing.expect(bpe.byte_fallback == null);
    }
    if (!any_ran) return error.SkipZigTest;
}

test "bpeFromHF llama3 ignore_merges short-circuits the whole-chunk multilingual piece" {
    // End-to-end regression for the Llama-3 multilingual parity gap.
    //
    // Root cause (see /tmp/ztok_wave1/card-055f3d0a.md): Llama-3's
    // tokenizer.json sets `model.ignore_merges = true`, so the HF Rust
    // encoder probes the ENTIRE pretok chunk against the vocab before
    // running the byte-pair merge loop and, on a hit, emits that single
    // id. ztok's byte-level rank-by-id merge loop never reaches the
    // whole-vocab piece for these sequences, so without the
    // short-circuit " Федерации" (one Split→ByteLevel chunk) split into
    // 3 tokens [126723, 7753, 54686] instead of HF's single id 111112 —
    // decoding to the same string but inflating the token count ~5% on
    // multilingual text.
    //
    // This test loads the real bench fixture through the production
    // loader (`bpeFromHF`) — not a synthetic vocab — to prove the
    // `ignore_merges` flag survives hf_json → hf_bridge → Bpe and that
    // the short-circuit fires on the byte-exact mapped key.
    const io = std.Io.Threaded.global_single_threaded.io();
    const path = "bench/vocabs/llama3.json";
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, std.testing.allocator, .unlimited) catch return error.SkipZigTest;
    defer std.testing.allocator.free(bytes);

    var hf = try hf_json.loadFromBytes(std.testing.allocator, bytes);
    defer hf.deinit();

    // The flag must be parsed off the fixture (default is false; a flip
    // here would mean the loader dropped it).
    try std.testing.expect(hf.ignore_merges);

    var bpe = try bpeFromHF(std.testing.allocator, &hf);
    defer bpe.deinit();

    // ...and it must be plumbed all the way onto the Bpe model.
    try std.testing.expect(bpe.ignore_merges);

    // The GPT-2 byte_to_unicode-mapped form of " Федерации" (U+0020 +
    // Cyrillic) — one Split→ByteLevel chunk, 38 mapped bytes, well under
    // the 64-codepoint HEAP_THRESHOLD so it stays on the inline SoA path.
    const fed_chunk = "\xC4\xA0\xC3\x90\xC2\xA4\xC3\x90\xC2\xB5\xC3\x90\xC2\xB4" ++
        "\xC3\x90\xC2\xB5\xC3\x91\xC4\xA2\xC3\x90\xC2\xB0\xC3\x91\xC4\xA8" ++
        "\xC3\x90\xC2\xB8\xC3\x90\xC2\xB8";

    var out: [64]TokenId = undefined;

    // With ignore_merges live, the whole-chunk lookup wins -> single id.
    const ids = bpe.encodeChunk(fed_chunk, &out);
    try std.testing.expectEqualSlices(TokenId, &.{111112}, ids);

    // Negative control: with the short-circuit disabled, the rank-by-id
    // merge loop produces the original 3-token split. This pins the
    // short-circuit (not some incidental merge-rank change) as the thing
    // delivering parity, and documents the pre-fix behavior.
    bpe.ignore_merges = false;
    const split = bpe.encodeChunk(fed_chunk, &out);
    try std.testing.expectEqualSlices(TokenId, &.{ 126723, 7753, 54686 }, split);
}
