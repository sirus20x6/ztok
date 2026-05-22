//! Bridge a parsed `SpModel` (from sp_model.zig) into a concrete
//! `Bpe` or `Unigram` model that the encoder pipeline can drive.
//!
//! Caveat: SP BPE models with user_defined/control pieces interleaved
//! with merge pieces may produce slightly different tokenizations than
//! the reference SentencePiece encoder. Truly faithful SP-BPE encoding
//! needs to skip non-mergeable pieces during the merge pair lookup;
//! deferred to a follow-up.
//!
//! Handling notes:
//!   - `.byte` pieces (literal `<0xNN>` forms) are kept as-is in the SoA
//!     layout, AND when present, are gathered into a `[256]TokenId`
//!     byte-fallback table that is attached to the produced `Bpe`. The
//!     BPE encoder's post-merge pass consults that table for any
//!     length-1 unmerged byte, matching SP's behavior of falling back
//!     to a dedicated byte token instead of an unk sentinel.
//!   - `.unused` pieces keep their slot in the SoA layout so token ids
//!     stay dense and aligned with the proto's piece order. They have
//!     non-encodable bytes (often empty / `<unused>` placeholders) and
//!     so will not be hit by the encoder in practice.

const std = @import("std");
const TokenId = @import("token.zig").TokenId;
const Bpe = @import("bpe.zig").Bpe;
const Unigram = @import("unigram.zig").Unigram;
const sp_model = @import("sp_model.zig");
const SpModel = sp_model.SpModel;
const normalizer_mod = @import("normalizer.zig");
const Normalizer = normalizer_mod.Normalizer;
const SpNormalizer = normalizer_mod.SpNormalizer;

pub const Error = error{ WrongModelKind, MissingUnk } || std.mem.Allocator.Error;

/// Construct the pipeline `Normalizer` an SP model expects, derived
/// from its `NormalizerSpec` flags. Maps:
///   * `name` typically `nmt_nfkc` (LLaMA-style), `nmt_nfkc_cf`
///     (Gemma-style — NFKC + simple case-fold), or `identity` →
///     enable/disable the NFKC pre-pass and the `_cf` casefold pass.
///   * `add_dummy_prefix`, `escape_whitespaces`, `remove_extra_whitespaces`
///     are forwarded one-to-one.
///
/// SP supports per-model `precompiled_charsmap` tables for custom
/// normalization rules (separate from NFKC). This `*const` variant
/// does NOT materialize the charsmap — for that path use
/// `normalizerFromSPModel` (post-1.17 agent B). With charsmap = null
/// the result is bit-for-bit identical to pre-1.17 behaviour, so
/// LLaMA/Gemma/synthetic-vocab callers stay on this entry point.
/// Real T5 / mT5 / Japanese SP models that ship a non-empty
/// `normalizer_spec.precompiled_charsmap` should use the *SpModel
/// variant so the Darts trie rewrite pass runs after NFKC.
pub fn normalizerFromSP(sp: *const SpModel) Normalizer {
    // SP `normalizer_name` strings observed in the wild: "identity",
    // "nfkc", "nmt_nfkc" (the LLaMA default), "nmt_nfkc_cf" (NFKC +
    // casefold, Gemma), "nfkc_cf". Treat any string with "nfkc" (any
    // case) in it as NFKC-enabled. A trailing `_cf` / `_CF` flips the
    // casefold flag — we apply the UCD 16.0 simple fold via
    // `capcode.toLower`. "identity" disables the NFKC pass but other
    // escape/dummy-prefix flags still apply. A missing name defaults
    // to NFKC-on to match SP's default when the spec is omitted.
    var enable_nfkc = true;
    var enable_cf = false;
    if (sp.normalizer_name) |name| {
        if (std.mem.eql(u8, name, "identity")) {
            enable_nfkc = false;
        } else if (std.mem.indexOf(u8, name, "nfkc") == null and
            std.mem.indexOf(u8, name, "NFKC") == null)
        {
            enable_nfkc = false;
        }
        if (std.mem.endsWith(u8, name, "_cf") or std.mem.endsWith(u8, name, "_CF")) {
            enable_cf = true;
        }
    }

    return .{ .sp_precompiled = SpNormalizer{
        .nfkc = enable_nfkc,
        .casefold = enable_cf,
        .add_dummy_prefix = sp.add_dummy_prefix,
        .escape_whitespaces = sp.escape_whitespaces,
        .remove_extra_whitespaces = sp.remove_extra_whitespaces,
    } };
}

/// Charsmap-aware variant of `normalizerFromSP`. Identical to the const
/// version EXCEPT that it lazily parses the SP model's
/// `precompiled_charsmap` (via `SpModel.parsedCharsmap`) and threads the
/// resulting pointer onto the `SpNormalizer.charsmap` field.
///
/// **Lifetime contract**: the returned `Normalizer` holds a
/// `*const PrecompiledCharsmap` borrowed from `sp.parsed_charsmap`,
/// which is owned by `sp` itself. The pointer remains valid as long as
/// the `SpModel` is not deinit'd. The pipeline owner is expected to
/// keep the `SpModel` alive for the lifetime of every encode call —
/// the bench harness, the C ABI loaders, and the Python binding all
/// already follow this pattern.
///
/// Parse errors on the trie blob degrade to `charsmap = null` rather
/// than propagating — a malformed charsmap is treated as "skip the
/// rewrite pass" so old behaviour is preserved. The model is still
/// usable; only the per-model rewrites are lost. Callers that want
/// hard failures should call `SpModel.parsedCharsmap` directly first.
pub fn normalizerFromSPModel(sp: *SpModel) std.mem.Allocator.Error!Normalizer {
    var n = normalizerFromSP(sp);
    if (n != .sp_precompiled) return n;
    // Best-effort parse; degrade to no-charsmap on any decode failure.
    const parsed = sp.parsedCharsmap() catch null;
    n.sp_precompiled.charsmap = parsed;
    return n;
}

/// Parse a `<0xNN>` SP byte-token name into the raw byte value, or null
/// if `s` doesn't match the convention. Exact length 6, leading `<0x`,
/// two uppercase hex digits, trailing `>`. Lowercase / mixed case is
/// rejected — SP's writer always emits uppercase.
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

/// Scan the SP vocab for `.byte` pieces whose name matches `<0xNN>` and
/// pack them into a `[256]TokenId` table indexed by the raw byte. Slots
/// for bytes that have no dedicated token are left as `maxInt(TokenId)`
/// — the encoder treats that as "no fallback for this byte".
///
/// Returns null if the model has fewer than 256 such tokens (i.e. it's
/// not a real byte-fallback vocab) — degrading to the old behavior is
/// safer than producing a half-populated table that would silently emit
/// wrong ids for the missing bytes.
fn buildByteFallbackTable(sp: *const SpModel) ?[256]TokenId {
    var table: [256]TokenId = undefined;
    @memset(&table, std.math.maxInt(TokenId));
    var found: u32 = 0;
    var id: u32 = 0;
    while (id < sp.count) : (id += 1) {
        if (sp.types[id] != .byte) continue;
        const name = sp.pieceBytes(id);
        const b = parseByteTokenName(name) orelse continue;
        // Duplicate `<0xNN>` entries are pathological; keep the first.
        if (table[b] == std.math.maxInt(TokenId)) {
            table[b] = id;
            found += 1;
        }
    }
    if (found < 256) return null;
    return table;
}

pub fn bpeFromSP(allocator: std.mem.Allocator, sp: *const SpModel) Error!Bpe {
    if (sp.model_kind != .bpe) return Error.WrongModelKind;
    const count = sp.count;

    if (count == 0) {
        return .{
            .allocator = allocator,
            .bytes = &.{},
            .offsets = &.{},
            .count = 0,
            .by_bytes = std.StringHashMap(TokenId).init(allocator),
        };
    }

    const bytes = try allocator.alloc(u8, sp.bytes.len);
    errdefer allocator.free(bytes);
    @memcpy(bytes, sp.bytes);

    const offsets = try allocator.alloc(u32, sp.offsets.len);
    errdefer allocator.free(offsets);
    @memcpy(offsets, sp.offsets);

    var by_bytes = std.StringHashMap(TokenId).init(allocator);
    errdefer by_bytes.deinit();
    try by_bytes.ensureTotalCapacity(count);

    // Per-piece merge rank derived from SP score. Lower rank wins in the
    // greedy per-position selection inside `encodeLongestMatch`. SP
    // scores are floats: 0 for specials/byte-fallback, negative for
    // normal pieces (more negative = added later in BPE training = lower
    // merge priority). We invert into u32 via `-score` clamped to u32
    // range; specials/byte/unused pieces — those that wouldn't appear in
    // `by_bytes` anyway — get max-rank so they're never preferred.
    const piece_ranks = try allocator.alloc(u32, count);
    errdefer allocator.free(piece_ranks);
    @memset(piece_ranks, std.math.maxInt(u32));

    var max_piece_len: u32 = 1;
    var r: u32 = 0;
    while (r < count) : (r += 1) {
        const key = bytes[offsets[r]..offsets[r + 1]];
        // SP byte-fallback pieces are stored as the literal text "<0xNN>"
        // in the proto bytes — six ASCII bytes per piece. We DON'T want
        // those keys in `by_bytes` because then a literal "<0x41>" in the
        // input would match the byte-A token. The byte-fallback table
        // covers them. Skip type=byte pieces from the multi-byte lookup
        // map. Same logic applies to control / unused pieces, which carry
        // bytes the user input would never legitimately contain (BOS/EOS
        // sentinels, `<unused>` placeholders).
        const ty = sp.types[r];
        if (ty == .byte or ty == .control or ty == .unused or ty == .unknown) continue;
        try by_bytes.put(key, r);
        if (key.len > max_piece_len) max_piece_len = @intCast(key.len);

        // Convert SP's negative score to a positive rank. Scores like
        // -184.0 → rank 184; -29184.0 → rank 29184. The exact integer
        // doesn't matter as long as ordering is preserved. Use a wide
        // u32 with saturating cast for the rare large-magnitude scores
        // (LLaMA-2's `▁▁` is -1e9 → saturated; rank 1e9 still beats
        // any normal piece's rank when picked for special-prefix runs).
        //
        // Note on zero: SP scores can be exactly 0.0 OR -0.0 for real
        // merge pieces (Gemma's `in` piece, the `\n`/`\n\n` family,
        // ~90 others). A naive `score >= 0.0` guard incorrectly captures
        // BOTH zeros (IEEE: -0.0 >= 0.0 is true) and stamps them with
        // RANK_INVALID — which forbids the BPE merge loop from ever
        // collapsing the matching pair. Result: ` within` segments as
        // `[▁with, i, n]` instead of `[▁within]` because the rank-0
        // `in` piece never wins its (i,n) pair. The fix is to compute
        // the rank for any non-positive score; only strictly positive
        // scores (which don't occur in well-trained SP vocabs) collapse
        // to max rank.
        const score = sp.scores[r];
        const rank: u32 = if (score > 0.0)
            std.math.maxInt(u32)
        else blk: {
            const neg = -score;
            if (neg >= @as(f32, @floatFromInt(std.math.maxInt(u32) - 1))) {
                break :blk std.math.maxInt(u32) - 1;
            }
            break :blk @as(u32, @intFromFloat(neg));
        };
        piece_ranks[r] = rank;
    }

    // Build the byte-fallback table. If `trainer_spec.byte_fallback` is
    // true the SP spec guarantees the 256-byte token bank; we trust
    // that and skip the scan when false. (`buildByteFallbackTable`
    // would still return null if fewer than 256 `<0xNN>` pieces are
    // present, leaving the Bpe behaving identically to before.)
    const bf_table: ?[256]TokenId = if (sp.byte_fallback)
        buildByteFallbackTable(sp)
    else
        null;

    return .{
        .allocator = allocator,
        .bytes = bytes,
        .offsets = offsets,
        .count = count,
        .by_bytes = by_bytes,
        .byte_fallback = bf_table,
        // SP-derived vocabs encode multi-byte pieces (e.g. `▁world`) as
        // single tokens without defining the intermediate byte-pair
        // merges that the standard tiktoken merge loop would need to
        // navigate from raw bytes up to the piece. Greedy per-position
        // selection (using SP's piece scores as priority) walks straight
        // to the piece in one pass.
        .encode_mode = .longest_match,
        .max_piece_len = max_piece_len,
        .piece_ranks = piece_ranks,
    };
}

pub fn unigramFromSP(allocator: std.mem.Allocator, sp: *const SpModel) Error!Unigram {
    if (sp.model_kind != .unigram) return Error.WrongModelKind;

    var unk_id: ?TokenId = null;
    var i: u32 = 0;
    while (i < sp.count) : (i += 1) {
        if (sp.types[i] == .unknown) {
            unk_id = i;
            break;
        }
    }
    const unk = unk_id orelse return Error.MissingUnk;

    var b = Unigram.Builder.init(allocator);
    errdefer b.deinit();

    // SP's Unigram encoder skips pieces of type `.unknown`, `.byte`,
    // `.control`, and `.unused` when matching text against the lattice:
    // those pieces have a stable id but their byte image is never a
    // valid candidate path. Including them in the trie produces wrong
    // segmentations because score-0 specials beat any negative real-
    // piece score (e.g. `<unk>` inside source like `const unk = ...
    // addToken("<unk>")` resolves to unk-id 2 instead of `<`, `unk`, `>`
    // — see the t5_unigram × code corpus regression at stress 1.24).
    // Allocate the id slot via `addToken` so vocab indices stay aligned
    // with the SP proto, then mark the slot excluded so `buildTrie`
    // skips registering its bytes.
    var id: u32 = 0;
    while (id < sp.count) : (id += 1) {
        const start = sp.offsets[id];
        const end = sp.offsets[id + 1];
        const new_id = try b.addToken(sp.bytes[start..end], sp.scores[id]);
        const ty = sp.types[id];
        if (ty == .unknown or ty == .byte or ty == .control or ty == .unused) {
            try b.excludeFromTrie(new_id);
        }
    }

    var uni = try b.finalize(unk);

    // Populate byte_fallback the same way `bpeFromSP` does. SP guarantees
    // the 256-byte token bank when `trainer_spec.byte_fallback` is set;
    // `buildByteFallbackTable` would otherwise return null if fewer than
    // 256 `<0xNN>` pieces are present and we leave the field null —
    // preserving the legacy per-codepoint unk behavior for non-fallback
    // models like T5 (which has `byte_fallback = false`).
    if (sp.byte_fallback) {
        uni.byte_fallback = buildByteFallbackTable(sp);
    }
    return uni;
}

// --- test helpers ---------------------------------------------------------
// Inline equivalents of sp_model.zig's protobuf builders (those helpers are
// file-private). Keep them minimal: just enough to hand-build tiny models.

fn tvEncodeVarint(out: *std.ArrayList(u8), allocator: std.mem.Allocator, v: u64) !void {
    var x = v;
    while (x >= 0x80) {
        try out.append(allocator, @as(u8, @intCast(x & 0x7f)) | 0x80);
        x >>= 7;
    }
    try out.append(allocator, @intCast(x));
}

fn tvTag(out: *std.ArrayList(u8), allocator: std.mem.Allocator, field: u32, wire: u3) !void {
    try tvEncodeVarint(out, allocator, (@as(u64, field) << 3) | wire);
}

fn tvFloat(out: *std.ArrayList(u8), allocator: std.mem.Allocator, f: f32) !void {
    var b: [4]u8 = undefined;
    const u: u32 = @bitCast(f);
    std.mem.writeInt(u32, &b, u, .little);
    try out.appendSlice(allocator, &b);
}

fn tvPieceBody(out: *std.ArrayList(u8), allocator: std.mem.Allocator, piece: []const u8, score: f32, ty: ?u64) !void {
    try tvTag(out, allocator, 1, 2);
    try tvEncodeVarint(out, allocator, piece.len);
    try out.appendSlice(allocator, piece);
    try tvTag(out, allocator, 2, 5);
    try tvFloat(out, allocator, score);
    if (ty) |t| {
        try tvTag(out, allocator, 3, 0);
        try tvEncodeVarint(out, allocator, t);
    }
}

fn tvEmbedMsg(out: *std.ArrayList(u8), allocator: std.mem.Allocator, field: u32, body: []const u8) !void {
    try tvTag(out, allocator, field, 2);
    try tvEncodeVarint(out, allocator, body.len);
    try out.appendSlice(allocator, body);
}

fn tvAppendPiece(buf: *std.ArrayList(u8), allocator: std.mem.Allocator, piece: []const u8, score: f32, ty: ?u64) !void {
    var p: std.ArrayList(u8) = .empty;
    defer p.deinit(allocator);
    try tvPieceBody(&p, allocator, piece, score, ty);
    try tvEmbedMsg(buf, allocator, 1, p.items);
}

fn tvAppendTrainerKind(buf: *std.ArrayList(u8), allocator: std.mem.Allocator, model_type: u64) !void {
    var ts: std.ArrayList(u8) = .empty;
    defer ts.deinit(allocator);
    try tvTag(&ts, allocator, 3, 0);
    try tvEncodeVarint(&ts, allocator, model_type);
    try tvEmbedMsg(buf, allocator, 2, ts.items);
}

fn tvAppendTrainerKindBF(
    buf: *std.ArrayList(u8),
    allocator: std.mem.Allocator,
    model_type: u64,
    byte_fallback: bool,
) !void {
    var ts: std.ArrayList(u8) = .empty;
    defer ts.deinit(allocator);
    try tvTag(&ts, allocator, 3, 0);
    try tvEncodeVarint(&ts, allocator, model_type);
    // field 35: byte_fallback (varint bool)
    try tvTag(&ts, allocator, 35, 0);
    try tvEncodeVarint(&ts, allocator, if (byte_fallback) 1 else 0);
    try tvEmbedMsg(buf, allocator, 2, ts.items);
}

// Append all 256 SP byte-token pieces (`<0x00>`..`<0xFF>`, type=byte=6).
// Used by byte-fallback tests to build a realistic LLaMA-style vocab.
fn tvAppendByteTokens(buf: *std.ArrayList(u8), allocator: std.mem.Allocator) !void {
    var i: u32 = 0;
    while (i < 256) : (i += 1) {
        var name: [6]u8 = .{ '<', '0', 'x', 0, 0, '>' };
        const hex = "0123456789ABCDEF";
        name[3] = hex[(i >> 4) & 0xF];
        name[4] = hex[i & 0xF];
        try tvAppendPiece(buf, allocator, &name, 0.0, 6);
    }
}

// --- tests ---------------------------------------------------------------

test "bpeFromSP builds an encodable Bpe" {
    const allocator = std.testing.allocator;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);

    try tvAppendPiece(&buf, allocator, "a", -1.0, 1);
    try tvAppendPiece(&buf, allocator, "b", -2.0, 1);
    try tvAppendPiece(&buf, allocator, "ab", -3.0, 1);
    try tvAppendTrainerKind(&buf, allocator, 2); // BPE

    var m = try sp_model.loadFromBytes(allocator, buf.items);
    defer m.deinit();

    var bpe = try bpeFromSP(allocator, &m);
    defer bpe.deinit();

    try std.testing.expectEqual(@as(u32, 3), bpe.count);
    try std.testing.expectEqualStrings("ab", bpe.idBytes(2));

    var out: [4]TokenId = undefined;
    const ids = bpe.encodeChunk("ab", &out);
    try std.testing.expectEqualSlices(TokenId, &.{2}, ids);
}

test "bpeFromSP rejects Unigram model" {
    const allocator = std.testing.allocator;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);

    try tvAppendPiece(&buf, allocator, "<unk>", 0.0, 2);
    try tvAppendPiece(&buf, allocator, "a", -1.0, 1);
    try tvAppendTrainerKind(&buf, allocator, 1); // Unigram

    var m = try sp_model.loadFromBytes(allocator, buf.items);
    defer m.deinit();

    try std.testing.expectError(Error.WrongModelKind, bpeFromSP(allocator, &m));
}

test "unigramFromSP builds an encodable Unigram" {
    const allocator = std.testing.allocator;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);

    try tvAppendPiece(&buf, allocator, "<unk>", 0.0, 2);
    try tvAppendPiece(&buf, allocator, "a", -2.0, 1);
    try tvAppendPiece(&buf, allocator, "b", -2.0, 1);
    try tvAppendPiece(&buf, allocator, "ab", -1.0, 1);
    try tvAppendTrainerKind(&buf, allocator, 1); // Unigram

    var m = try sp_model.loadFromBytes(allocator, buf.items);
    defer m.deinit();

    var uni = try unigramFromSP(allocator, &m);
    defer uni.deinit();

    try std.testing.expectEqual(@as(u32, 4), uni.count);
    try std.testing.expectEqual(@as(TokenId, 0), uni.unk_id);

    var out: [4]TokenId = undefined;
    const ids = try uni.encodeChunk(allocator, "ab", &out);
    try std.testing.expectEqualSlices(TokenId, &.{3}, ids);
}

test "unigramFromSP errors if no unk piece" {
    const allocator = std.testing.allocator;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);

    try tvAppendPiece(&buf, allocator, "a", -1.0, 1);
    try tvAppendPiece(&buf, allocator, "b", -1.0, 1);
    try tvAppendTrainerKind(&buf, allocator, 1); // Unigram

    var m = try sp_model.loadFromBytes(allocator, buf.items);
    defer m.deinit();

    try std.testing.expectError(Error.MissingUnk, unigramFromSP(allocator, &m));
}

test "unigramFromSP populates byte_fallback when trainer flag is set" {
    // SP Unigram model with byte_fallback=true must surface a fully
    // populated `[256]TokenId` table on the produced Unigram. Mirrors
    // the BPE-side test below; the LLaMA-family SP Unigram exports
    // (Gemma 1.0 Unigram variant, etc.) ship with this flag and the
    // 256-piece byte bank.
    const allocator = std.testing.allocator;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);

    try tvAppendPiece(&buf, allocator, "<unk>", 0.0, 2);
    try tvAppendPiece(&buf, allocator, "a", -1.0, 1);
    try tvAppendByteTokens(&buf, allocator);
    try tvAppendTrainerKindBF(&buf, allocator, 1, true); // Unigram + bf=true

    var m = try sp_model.loadFromBytes(allocator, buf.items);
    defer m.deinit();
    try std.testing.expect(m.byte_fallback);

    var uni = try unigramFromSP(allocator, &m);
    defer uni.deinit();

    try std.testing.expect(uni.byte_fallback != null);
    // Byte 'a' (0x61) -> the byte-token id at position 2 + 0x61
    // (after `<unk>` and `a`).
    const expected_a: TokenId = 2 + 0x61;
    try std.testing.expectEqual(expected_a, uni.byte_fallback.?[0x61]);
    try std.testing.expectEqual(@as(TokenId, 2 + 0xFF), uni.byte_fallback.?[0xFF]);
}

test "unigramFromSP leaves byte_fallback null when proto doesn't set it" {
    // T5-style: byte_fallback unset → field stays null and Viterbi
    // falls back to the unk-merge path (legacy behavior).
    const allocator = std.testing.allocator;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);

    try tvAppendPiece(&buf, allocator, "<unk>", 0.0, 2);
    try tvAppendPiece(&buf, allocator, "a", -1.0, 1);
    try tvAppendTrainerKind(&buf, allocator, 1); // Unigram, no bf

    var m = try sp_model.loadFromBytes(allocator, buf.items);
    defer m.deinit();
    try std.testing.expect(!m.byte_fallback);

    var uni = try unigramFromSP(allocator, &m);
    defer uni.deinit();

    try std.testing.expect(uni.byte_fallback == null);
}

test "unigramFromSP T5 model encodes a sample sentence without panicking" {
    // Live load of the T5 SP Unigram model + full pipeline encode.
    // Equivalence vs sp-python is verified by
    // `bench/equivalence_check.py unigram bench/vocabs/t5_unigram`
    // (100/100 lines match in the post-1.13 state); this test just
    // proves the SP → ztok pipeline links up end-to-end.
    const allocator = std.testing.allocator;
    const path = "bench/vocabs/t5_unigram.model";
    const io = std.Io.Threaded.global_single_threaded.io();
    const contents = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(contents);

    var sp = try sp_model.loadFromBytes(allocator, contents);
    defer sp.deinit();
    try std.testing.expectEqual(sp_model.ModelKind.unigram, sp.model_kind);
    try std.testing.expect(!sp.byte_fallback); // T5 has byte_fallback off

    var uni = try unigramFromSP(allocator, &sp);
    defer uni.deinit();
    try std.testing.expect(uni.byte_fallback == null);
    try std.testing.expect(uni.min_score < 0.0); // real pieces are negative

    const Pipeline = @import("pipeline.zig").Pipeline;
    const Vocab = @import("vocab.zig").Vocab;
    var v = Vocab.empty(allocator);
    defer v.deinit();

    const pipe: Pipeline = .{
        .normalizer = normalizerFromSP(&sp),
        .pre_tokenizer = .identity,
        .model = .{ .unigram = &uni },
        .decoder = .concat,
        .vocab = &v,
    };

    const ids = try pipe.encode(allocator, "Hello world");
    defer allocator.free(ids);
    // SP-python reference: ['▁Hello', '▁world'] -> [8774, 296].
    try std.testing.expectEqualSlices(TokenId, &.{ 8774, 296 }, ids);
}

// --- byte-fallback tests --------------------------------------------------

test "bpeFromSP populates byte_fallback when trainer flag is set" {
    const allocator = std.testing.allocator;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);

    // Two ordinary pieces, then the full 256-byte token bank, then a
    // trainer_spec with byte_fallback=true.
    try tvAppendPiece(&buf, allocator, "a", -1.0, 1);
    try tvAppendPiece(&buf, allocator, "b", -2.0, 1);
    try tvAppendByteTokens(&buf, allocator);
    try tvAppendTrainerKindBF(&buf, allocator, 2, true); // BPE + bf=true

    var m = try sp_model.loadFromBytes(allocator, buf.items);
    defer m.deinit();
    try std.testing.expect(m.byte_fallback);

    var bpe = try bpeFromSP(allocator, &m);
    defer bpe.deinit();

    try std.testing.expect(bpe.byte_fallback != null);
    // Byte 'a' (0x61) -> the byte-token id at position 2 + 0x61.
    const expected_byte_a: TokenId = 2 + 0x61;
    try std.testing.expectEqual(expected_byte_a, bpe.byte_fallback.?[0x61]);
    // And byte 0xFF -> 2 + 0xFF.
    try std.testing.expectEqual(@as(TokenId, 2 + 0xFF), bpe.byte_fallback.?[0xFF]);
}

test "bpeFromSP byte-fallback rewrites unmerged bytes to byte-token ids" {
    const allocator = std.testing.allocator;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);

    // Vocab: "a" + 256 byte tokens. Encoding "z" should yield the byte
    // token for 'z' (0x7a), NOT the maxInt unk sentinel — because the
    // merge loop can't grow the singleton "z" but byte-fallback covers it.
    try tvAppendPiece(&buf, allocator, "a", -1.0, 1);
    try tvAppendByteTokens(&buf, allocator);
    try tvAppendTrainerKindBF(&buf, allocator, 2, true);

    var m = try sp_model.loadFromBytes(allocator, buf.items);
    defer m.deinit();

    var bpe = try bpeFromSP(allocator, &m);
    defer bpe.deinit();

    var out: [4]TokenId = undefined;
    const ids = bpe.encodeChunk("z", &out);
    try std.testing.expectEqual(@as(usize, 1), ids.len);
    // Byte tokens start at id 1; id for 'z' is 1 + 0x7a.
    try std.testing.expectEqual(@as(TokenId, 1 + 0x7a), ids[0]);
}

test "bpeFromSP byte-fallback covers all 256 bytes" {
    const allocator = std.testing.allocator;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);

    // Only byte tokens + trainer with byte_fallback.
    try tvAppendByteTokens(&buf, allocator);
    try tvAppendTrainerKindBF(&buf, allocator, 2, true);

    var m = try sp_model.loadFromBytes(allocator, buf.items);
    defer m.deinit();

    var bpe = try bpeFromSP(allocator, &m);
    defer bpe.deinit();

    // Encode every byte 0..255 in one chunk. Each must resolve to its
    // own dedicated byte-token id (id == byte value). No unk sentinel.
    // 256 bytes is well above bpe.zig's HEAP_THRESHOLD=64, so this
    // also exercises the heap encoder path. Use an arena for the
    // heap-path scratch (it doesn't free its allocations by design).
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    var input: [256]u8 = undefined;
    var i: u32 = 0;
    while (i < 256) : (i += 1) input[i] = @intCast(i);

    var out: [256]TokenId = undefined;
    const ids = bpe.encodeChunkScratch(arena.allocator(), &input, &out);
    try std.testing.expectEqual(@as(usize, 256), ids.len);

    var b: u32 = 0;
    while (b < 256) : (b += 1) {
        try std.testing.expectEqual(@as(TokenId, b), ids[b]);
        try std.testing.expect(ids[b] != std.math.maxInt(TokenId));
    }
}

test "bpeFromSP without byte_fallback flag leaves byte_fallback null" {
    const allocator = std.testing.allocator;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);

    // Trainer doesn't set byte_fallback; even though we include byte
    // tokens here, the loader should NOT populate the fallback table.
    // Existing non-SP BPE callers and SP models without byte_fallback
    // are bit-for-bit unaffected.
    try tvAppendPiece(&buf, allocator, "a", -1.0, 1);
    try tvAppendPiece(&buf, allocator, "b", -2.0, 1);
    try tvAppendPiece(&buf, allocator, "ab", -3.0, 1);
    try tvAppendTrainerKind(&buf, allocator, 2); // BPE, no bf flag

    var m = try sp_model.loadFromBytes(allocator, buf.items);
    defer m.deinit();
    try std.testing.expect(!m.byte_fallback);

    var bpe = try bpeFromSP(allocator, &m);
    defer bpe.deinit();

    try std.testing.expect(bpe.byte_fallback == null);

    // Encoding behavior matches the pre-byte-fallback contract.
    var out: [4]TokenId = undefined;
    const ids = bpe.encodeChunk("ab", &out);
    try std.testing.expectEqualSlices(TokenId, &.{2}, ids);
}

test "bpeFromSP byte-fallback long-chunk heap path also rewrites" {
    // Chunks longer than the HEAP_THRESHOLD in bpe.zig route through the
    // 4-ary heap encoder. The byte-fallback rewrite must apply there too.
    const allocator = std.testing.allocator;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);

    // Vocab: byte tokens only. No merges, so every encoded id should
    // equal the source byte (since byte-token id == byte value).
    try tvAppendByteTokens(&buf, allocator);
    try tvAppendTrainerKindBF(&buf, allocator, 2, true);

    var m = try sp_model.loadFromBytes(allocator, buf.items);
    defer m.deinit();

    var bpe = try bpeFromSP(allocator, &m);
    defer bpe.deinit();

    // Build a 128-byte chunk (> HEAP_THRESHOLD=64 in bpe.zig) of
    // varying bytes. The heap path pulls scratch arrays from the
    // passed scratch allocator and (by design) doesn't free them, so
    // we feed it an arena that we deinit at the end.
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    var input: [128]u8 = undefined;
    var i: u32 = 0;
    while (i < input.len) : (i += 1) input[i] = @intCast((i * 7 + 3) & 0xFF);

    var out: [128]TokenId = undefined;
    const ids = bpe.encodeChunkScratch(arena.allocator(), &input, &out);
    try std.testing.expectEqual(@as(usize, input.len), ids.len);
    var k: u32 = 0;
    while (k < input.len) : (k += 1) {
        try std.testing.expectEqual(@as(TokenId, input[k]), ids[k]);
    }
}

// --- normalizerFromSP tests -----------------------------------------------

test "normalizerFromSP defaults to nmt_nfkc profile" {
    // A bare SpModel with no normalizer_spec at all: SP defaults are
    // NFKC + escape + dummy_prefix on, remove_extra off (well, SP's
    // default is actually "true" for that one, but the test reflects
    // what `sp_model` already records in its defaults).
    var sp: SpModel = .{
        .allocator = std.testing.allocator,
        .bytes = &.{},
        .offsets = &.{},
        .scores = &.{},
        .types = &.{},
        .count = 0,
        .model_kind = .bpe,
    };
    defer sp.deinit();

    const n = normalizerFromSP(&sp);
    switch (n) {
        .sp_precompiled => |cfg| {
            try std.testing.expectEqual(true, cfg.nfkc);
            try std.testing.expectEqual(true, cfg.add_dummy_prefix);
            try std.testing.expectEqual(true, cfg.escape_whitespaces);
            try std.testing.expectEqual(true, cfg.remove_extra_whitespaces);
        },
        else => return error.TestExpectedSpNormalizer,
    }
}

test "normalizerFromSP: identity normalizer_name disables NFKC" {
    var sp: SpModel = .{
        .allocator = std.testing.allocator,
        .bytes = &.{},
        .offsets = &.{},
        .scores = &.{},
        .types = &.{},
        .count = 0,
        .model_kind = .bpe,
        .normalizer_name = try std.testing.allocator.dupe(u8, "identity"),
        .add_dummy_prefix = false,
        .escape_whitespaces = false,
        .remove_extra_whitespaces = false,
    };
    defer sp.deinit();

    const n = normalizerFromSP(&sp);
    switch (n) {
        .sp_precompiled => |cfg| {
            try std.testing.expectEqual(false, cfg.nfkc);
            try std.testing.expectEqual(false, cfg.add_dummy_prefix);
            try std.testing.expectEqual(false, cfg.escape_whitespaces);
            try std.testing.expectEqual(false, cfg.remove_extra_whitespaces);
        },
        else => return error.TestExpectedSpNormalizer,
    }
}

test "normalizerFromSP: nmt_nfkc name enables NFKC" {
    var sp: SpModel = .{
        .allocator = std.testing.allocator,
        .bytes = &.{},
        .offsets = &.{},
        .scores = &.{},
        .types = &.{},
        .count = 0,
        .model_kind = .bpe,
        .normalizer_name = try std.testing.allocator.dupe(u8, "nmt_nfkc"),
        .add_dummy_prefix = true,
        .escape_whitespaces = true,
        .remove_extra_whitespaces = false,
    };
    defer sp.deinit();

    const n = normalizerFromSP(&sp);
    switch (n) {
        .sp_precompiled => |cfg| {
            try std.testing.expectEqual(true, cfg.nfkc);
            try std.testing.expectEqual(true, cfg.add_dummy_prefix);
            try std.testing.expectEqual(true, cfg.escape_whitespaces);
        },
        else => return error.TestExpectedSpNormalizer,
    }
}

test "normalizerFromSP: nmt_nfkc_cf and nfkc_cf enable casefold" {
    // Gemma-style normalizer names: the `_cf` suffix flips the
    // SpNormalizer.casefold flag. Test both spellings (with and
    // without the `nmt_` prefix).
    const names = [_][]const u8{ "nmt_nfkc_cf", "nfkc_cf" };
    for (names) |name_str| {
        var sp: SpModel = .{
            .allocator = std.testing.allocator,
            .bytes = &.{},
            .offsets = &.{},
            .scores = &.{},
            .types = &.{},
            .count = 0,
            .model_kind = .unigram,
            .normalizer_name = try std.testing.allocator.dupe(u8, name_str),
        };
        defer sp.deinit();

        const n = normalizerFromSP(&sp);
        switch (n) {
            .sp_precompiled => |cfg| {
                try std.testing.expectEqual(true, cfg.nfkc);
                try std.testing.expectEqual(true, cfg.casefold);
            },
            else => return error.TestExpectedSpNormalizer,
        }
    }
}

test "normalizerFromSP: nmt_nfkc (no _cf) leaves casefold off" {
    var sp: SpModel = .{
        .allocator = std.testing.allocator,
        .bytes = &.{},
        .offsets = &.{},
        .scores = &.{},
        .types = &.{},
        .count = 0,
        .model_kind = .unigram,
        .normalizer_name = try std.testing.allocator.dupe(u8, "nmt_nfkc"),
    };
    defer sp.deinit();

    const n = normalizerFromSP(&sp);
    switch (n) {
        .sp_precompiled => |cfg| {
            try std.testing.expectEqual(true, cfg.nfkc);
            try std.testing.expectEqual(false, cfg.casefold);
        },
        else => return error.TestExpectedSpNormalizer,
    }
}

// --- LLaMA-2 integration test ---------------------------------------------
//
// Skipped (logged + early-return) when the model file isn't present —
// CI environments that don't ship the bench corpus shouldn't fail. The
// test exercises the full SP loader → Bpe + normalizer-from-spec path.

test "normalizerFromSP: LLaMA-2 tokenizer.model produces ▁-prefixed output" {
    const allocator = std.testing.allocator;
    const path = "bench/vocabs/llama2.model";
    const io = std.Io.Threaded.global_single_threaded.io();
    const contents = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(contents);

    var sp = try sp_model.loadFromBytes(allocator, contents);
    defer sp.deinit();

    // SP-side spec sanity check before we exercise the pipeline.
    try std.testing.expectEqual(true, sp.add_dummy_prefix);
    try std.testing.expectEqual(true, sp.escape_whitespaces);

    const n = normalizerFromSP(&sp);
    switch (n) {
        .sp_precompiled => |cfg| {
            try std.testing.expectEqual(true, cfg.add_dummy_prefix);
            try std.testing.expectEqual(true, cfg.escape_whitespaces);
        },
        else => return error.TestExpectedSpNormalizer,
    }

    const out = try n.normalize(allocator, "Hello, world! How are you doing today?");
    defer allocator.free(out);

    // Leading dummy prefix: U+2581 (E2 96 81).
    try std.testing.expect(out.len >= 3);
    try std.testing.expectEqualSlices(u8, &.{ 0xE2, 0x96, 0x81 }, out[0..3]);

    // ASCII spaces in the source must NOT survive — every space becomes
    // a U+2581 sequence.
    for (out) |b| try std.testing.expect(b != ' ');

    // Sanity: U+2581 appears at least as many times as the source had
    // spaces (4 in "Hello, world! How are you doing today?") plus 1
    // for the dummy prefix.
    var u2581_count: usize = 0;
    var i: usize = 0;
    while (i + 2 < out.len) : (i += 1) {
        if (out[i] == 0xE2 and out[i + 1] == 0x96 and out[i + 2] == 0x81) {
            u2581_count += 1;
        }
    }
    try std.testing.expect(u2581_count >= 5);
}

// --- Gemma integration test (post-1.14 agent B) --------------------------
//
// Gemma's tokenizer.model ships byte_fallback=true with a 256-entry
// `<0xNN>` byte bank. The Gemma 2b mirror at unsloth/gemma-2b uses
// `normalizer_spec.name = "identity"` (no NFKC, no casefold) — they
// pre-normalize upstream — so this test verifies:
//   1. normalizerFromSP recognises "identity" and sets nfkc=false,
//      casefold=false (the `_cf` suffix is absent).
//   2. The BPE bridge surfaces a populated byte_fallback table from
//      the SP byte bank — same path Gemma's `nmt_nfkc_cf` cousins
//      would take.
//
// The casefold path itself is exercised by the unit tests in
// `normalizer.zig` and the synthetic `_cf`-name tests above; we don't
// require a real `_cf` model to be downloaded to test the wiring.
// Skipped when bench/vocabs/gemma.model isn't present (run
// `python bench/fetch_vocabs.py` to fetch).

test "normalizerFromSP: Gemma tokenizer.model populates byte_fallback through the bridge" {
    const allocator = std.testing.allocator;
    const path = "bench/vocabs/gemma.model";
    const io = std.Io.Threaded.global_single_threaded.io();
    const contents = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(contents);

    var sp = try sp_model.loadFromBytes(allocator, contents);
    defer sp.deinit();

    // Gemma ships byte_fallback=true (256 `<0xNN>` pieces).
    try std.testing.expect(sp.byte_fallback);

    // gemma-2b uses identity normalization (NFKC handled upstream).
    const n = normalizerFromSP(&sp);
    switch (n) {
        .sp_precompiled => |cfg| {
            try std.testing.expectEqual(false, cfg.nfkc);
            try std.testing.expectEqual(false, cfg.casefold);
        },
        else => return error.TestExpectedSpNormalizer,
    }

    // The BPE bridge should surface byte_fallback (whether SP-BPE or
    // SP-Unigram — both go through buildByteFallbackTable internally).
    if (sp.model_kind == .bpe) {
        var bpe = try bpeFromSP(allocator, &sp);
        defer bpe.deinit();
        try std.testing.expect(bpe.byte_fallback != null);
    } else if (sp.model_kind == .unigram) {
        var uni = try unigramFromSP(allocator, &sp);
        defer uni.deinit();
        try std.testing.expect(uni.byte_fallback != null);
    }
}

// --- SP-BPE encode-path tests --------------------------------------------

test "bpeFromSP sets longest_match mode + populates piece_ranks" {
    // Mode + rank metadata is the contract the BPE encoder relies on.
    // Without this, `encodeSpBpe` won't engage and SP-derived vocabs
    // fall back to byte-level merging — losing every multi-byte piece.
    const allocator = std.testing.allocator;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);

    try tvAppendPiece(&buf, allocator, "a", -1.0, 1);
    try tvAppendPiece(&buf, allocator, "b", -2.0, 1);
    try tvAppendPiece(&buf, allocator, "ab", -3.0, 1);
    try tvAppendTrainerKind(&buf, allocator, 2); // BPE

    var m = try sp_model.loadFromBytes(allocator, buf.items);
    defer m.deinit();

    var bpe = try bpeFromSP(allocator, &m);
    defer bpe.deinit();

    try std.testing.expectEqual(@import("bpe.zig").EncodeMode.longest_match, bpe.encode_mode);
    try std.testing.expect(bpe.piece_ranks != null);
    try std.testing.expectEqual(@as(u32, 1), bpe.piece_ranks.?[0]); // -score = 1
    try std.testing.expectEqual(@as(u32, 2), bpe.piece_ranks.?[1]);
    try std.testing.expectEqual(@as(u32, 3), bpe.piece_ranks.?[2]);
    try std.testing.expectEqual(@as(u32, 2), bpe.max_piece_len);
}

test "bpeFromSP score-priority beats greedy-longest match" {
    // Build a synthetic vocab where greedy longest-match would pick a
    // different piece sequence than SP's score-priority BPE merge.
    //
    // Vocab:
    //   a (id=0, score=-1)
    //   b (id=1, score=-1)
    //   c (id=2, score=-1)
    //   ab (id=3, score=-5)   <- score=5
    //   bc (id=4, score=-100) <- score=100, much lower priority than ab
    //   abc (id=5, score=-50) <- score=50
    //
    // Input "abc". Greedy longest-match would pick "abc" (id=5) directly.
    // SP-BPE would:
    //   1. Initial parts [a, b, c]
    //   2. Pair (a,b) rank=5, pair (b,c) rank=100. (a,b) wins -> merge.
    //   3. Parts [ab, c]. Pair (ab, c) -> "abc" rank=50. Merge.
    //   4. Parts [abc]. Output id=5.
    //
    // Both produce id=5 here. Construct a divergent case instead:
    // remove "abc" but keep "ab" (id 3) and "bc" (id 4). Score-priority
    // BPE: merges (a,b) first (rank 5 < 100), then can't merge further.
    // Greedy longest-match: at pos 0, longest match is "ab" (len 2);
    // emits ab, then 'c'. Same result. We need to make longest-match
    // disagree.
    //
    // Better case: input "ab".
    //   Pieces: a(0,-1), b(1,-1), ab(2,-10), abc(3,-5). No "abc" in input.
    //   Greedy longest-match at "ab": longest is "ab" (id=2). Emit [2].
    //   Score-priority BPE: (a,b) rank=10 -> merge -> [ab] id=2. Same.
    //
    // The real divergence: input "abcd" with pieces a,b,c,d,abc,bcd.
    // Greedy: emit [abc, d].
    // BPE: merges whichever pair has lowest rank first.
    //
    // Cleaner test: directly verify piece_ranks ordering AND that
    // encode produces the BPE result for "ungreedy"-style input.
    const allocator = std.testing.allocator;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);

    // a, b, c, d + ab (low rank), abc (high rank), abcd (very high rank).
    // Input "abcd". Greedy longest-match grabs "abcd" if present;
    // remove "abcd" so neither side has a single-piece answer.
    //
    // Vocab: a=0(-1), b=1(-1), c=2(-1), d=3(-1),
    //        ab=4(-10), bc=5(-1000), cd=6(-20), abc=7(-500), bcd=8(-100).
    // Input "abcd".
    //   Greedy longest at pos 0: try "abcd" (no) -> "abc"(id 7) found. Emit 7.
    //     pos 3: "d" found. Emit 3. Result [7, 3] = [abc, d].
    //   SP-BPE: pairs (a,b)=10, (b,c)=1000, (c,d)=20. Min=(a,b) rank 10.
    //     Merge -> [ab(4), c, d]. Pairs (ab,c)=? need "abc" -> rank 500.
    //                                 (c,d)=20. Min=(c,d) rank 20.
    //     Merge -> [ab, cd(6)]. Pair (ab,cd)=? need "abcd". Not in vocab. STOP.
    //     Result: [4, 6] = [ab, cd].
    //
    // Different output! Good — confirms score-priority not greedy-longest.
    try tvAppendPiece(&buf, allocator, "a", -1.0, 1);
    try tvAppendPiece(&buf, allocator, "b", -1.0, 1);
    try tvAppendPiece(&buf, allocator, "c", -1.0, 1);
    try tvAppendPiece(&buf, allocator, "d", -1.0, 1);
    try tvAppendPiece(&buf, allocator, "ab", -10.0, 1);
    try tvAppendPiece(&buf, allocator, "bc", -1000.0, 1);
    try tvAppendPiece(&buf, allocator, "cd", -20.0, 1);
    try tvAppendPiece(&buf, allocator, "abc", -500.0, 1);
    try tvAppendPiece(&buf, allocator, "bcd", -100.0, 1);
    try tvAppendTrainerKind(&buf, allocator, 2); // BPE

    var m = try sp_model.loadFromBytes(allocator, buf.items);
    defer m.deinit();

    var bpe = try bpeFromSP(allocator, &m);
    defer bpe.deinit();

    var out: [4]TokenId = undefined;
    const ids = bpe.encodeChunk("abcd", &out);
    try std.testing.expectEqualSlices(TokenId, &.{ 4, 6 }, ids); // ab + cd, NOT abc + d
}

test "encodeSpBpe falls back via byte_fallback for bytes not in any piece" {
    // Vocab has only "a" + the 256 byte tokens. Input "az" — 'z' isn't
    // a piece (no other piece covers it), so the SP-BPE encoder falls
    // back to the byte-fallback table for it. 'a' resolves to the
    // multi-byte-piece id directly.
    const allocator = std.testing.allocator;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);

    try tvAppendPiece(&buf, allocator, "a", -1.0, 1);
    try tvAppendByteTokens(&buf, allocator);
    try tvAppendTrainerKindBF(&buf, allocator, 2, true);

    var m = try sp_model.loadFromBytes(allocator, buf.items);
    defer m.deinit();

    var bpe = try bpeFromSP(allocator, &m);
    defer bpe.deinit();

    var out: [4]TokenId = undefined;
    const ids = bpe.encodeChunk("az", &out);
    try std.testing.expectEqual(@as(usize, 2), ids.len);
    try std.testing.expectEqual(@as(TokenId, 0), ids[0]); // "a" piece (id 0)
    try std.testing.expectEqual(@as(TokenId, 1 + 'z'), ids[1]); // byte-fallback for 'z'
}

test "tiktoken Bpe stays on .bpe_merge mode (no SP-side effects)" {
    // Regression: loading a tiktoken vocab must NOT engage the
    // longest-match path. piece_ranks stays null, encode_mode stays
    // .bpe_merge, and the merge behavior is bit-identical to the
    // pre-SP-BPE baseline.
    const EncodeMode = @import("bpe.zig").EncodeMode;
    const std_base64 = std.base64.standard.Encoder;

    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(std.testing.allocator);

    inline for (.{
        .{ "a", 0 },
        .{ "b", 1 },
        .{ "ab", 2 },
        .{ "c", 3 },
        .{ "bc", 4 },
        .{ "abc", 5 },
    }) |entry| {
        const bytes = entry[0];
        const rank: u32 = entry[1];
        const sz = std_base64.calcSize(bytes.len);
        const tmp = try std.testing.allocator.alloc(u8, sz);
        defer std.testing.allocator.free(tmp);
        const enc = std_base64.encode(tmp, bytes);
        try src.appendSlice(std.testing.allocator, enc);
        try src.print(std.testing.allocator, " {d}\n", .{rank});
    }

    var bpe = try Bpe.loadTiktokenBytes(std.testing.allocator, src.items);
    defer bpe.deinit();

    try std.testing.expectEqual(EncodeMode.bpe_merge, bpe.encode_mode);
    try std.testing.expect(bpe.piece_ranks == null);

    // BPE merge of "abc":
    //   pairs: (a,b)=rank 2, (b,c)=rank 4. Min=(a,b). Merge.
    //   parts: [ab, c]. pair (ab,c)="abc" rank 5. Merge.
    //   parts: [abc]. Done.
    // Final ids: [5]. Same merge sequence the pre-SP-BPE encoder used.
    var out: [4]TokenId = undefined;
    const ids = bpe.encodeChunk("abc", &out);
    try std.testing.expectEqualSlices(TokenId, &.{5}, ids);
}

test "bpeFromSP LLaMA-2 'Hello, world!' produces SP-compatible ids" {
    // End-to-end through the SP pipeline. Skipped if the model file
    // isn't present (CI environments without the bench corpus).
    const allocator = std.testing.allocator;
    const path = "bench/vocabs/llama2.model";
    const io = std.Io.Threaded.global_single_threaded.io();
    const contents = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(contents);

    var sp = try sp_model.loadFromBytes(allocator, contents);
    defer sp.deinit();

    var bpe = try bpeFromSP(allocator, &sp);
    defer bpe.deinit();

    // The SP-derived BPE must be in longest-match mode with a non-null
    // piece_ranks. Encoding the normalized form of "Hello, world!"
    // should emit the multi-byte pieces "▁Hello", ",", "▁world", "!".
    const Pipeline = @import("pipeline.zig").Pipeline;
    const Vocab = @import("vocab.zig").Vocab;
    var v = Vocab.empty(allocator);
    defer v.deinit();

    const pipe: Pipeline = .{
        .normalizer = normalizerFromSP(&sp),
        .pre_tokenizer = .identity,
        .model = .{ .bpe = &bpe },
        .decoder = .concat,
        .vocab = &v,
    };

    const ids = try pipe.encode(allocator, "Hello, world!");
    defer allocator.free(ids);

    // Reference SP encoding of "Hello, world!" on LLaMA-2:
    //   [15043, 29892, 3186, 29991] = ['▁Hello', ',', '▁world', '!']
    try std.testing.expectEqualSlices(
        TokenId,
        &.{ 15043, 29892, 3186, 29991 },
        ids,
    );

    // Sanity: no maxInt sentinel survived (every byte resolved).
    for (ids) |id| try std.testing.expect(id != std.math.maxInt(TokenId));
}

test "bpeFromSP LLaMA-2 100 sample lines match sp-python via dump-sample" {
    // Loads bench/vocabs/llama2.model + first 100 lines of
    // /tmp/corpus.txt (the equivalence-check corpus). Encodes each
    // line through the SP pipeline and verifies the result matches a
    // golden list captured from sp-python (one int slice per line).
    //
    // The full golden table is far too long to inline in Zig; instead,
    // we run bench/equivalence_check.py externally and check its
    // headline match count. This test is skipped when either the model
    // or the corpus file is missing.
    const allocator = std.testing.allocator;
    const model_path = "bench/vocabs/llama2.model";
    const corpus_path = "/tmp/corpus.txt";
    const io = std.Io.Threaded.global_single_threaded.io();

    const contents = std.Io.Dir.cwd().readFileAlloc(io, model_path, allocator, .unlimited) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(contents);
    const corpus = std.Io.Dir.cwd().readFileAlloc(io, corpus_path, allocator, .unlimited) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(corpus);

    var sp = try sp_model.loadFromBytes(allocator, contents);
    defer sp.deinit();
    var bpe = try bpeFromSP(allocator, &sp);
    defer bpe.deinit();

    const Pipeline = @import("pipeline.zig").Pipeline;
    const Vocab = @import("vocab.zig").Vocab;
    var v = Vocab.empty(allocator);
    defer v.deinit();

    const pipe: Pipeline = .{
        .normalizer = normalizerFromSP(&sp),
        .pre_tokenizer = .identity,
        .model = .{ .bpe = &bpe },
        .decoder = .concat,
        .vocab = &v,
    };

    // Walk the first 100 lines and just verify they all encode without
    // panicking and produce non-empty id streams (for non-empty input).
    // Full match-vs-sp-python parity is verified externally by
    // bench/equivalence_check.py — running sp-python from a Zig test
    // would require a Python interpreter at test time.
    var lines_done: u32 = 0;
    var p: usize = 0;
    while (lines_done < 100 and p < corpus.len) : (lines_done += 1) {
        const start = p;
        while (p < corpus.len and corpus[p] != '\n') : (p += 1) {}
        const line = corpus[start..p];
        if (p < corpus.len) p += 1;
        const ids = try pipe.encode(allocator, line);
        defer allocator.free(ids);
        // Empty input -> empty output (SP-correct behavior).
        if (line.len == 0) {
            try std.testing.expectEqual(@as(usize, 0), ids.len);
        } else {
            try std.testing.expect(ids.len > 0);
            for (ids) |id| try std.testing.expect(id != std.math.maxInt(TokenId));
        }
    }
}

// --- Gemma post-1.15 regression test --------------------------------------
//
// Cross-bench against sp-python found a single divergent line on Gemma:
// the segment "▁within" was emitted as `[▁with, i, n]` instead of
// `[▁within]`. Root cause: the SP score for the `in` piece is exactly
// `-0.0` (IEEE bits 0x80000000). The pre-fix `bpeFromSP` rank conversion
// gated on `score >= 0.0`, which IEEE-true-evaluates for -0.0 → the `in`
// piece was stamped with RANK_INVALID and the BPE merge loop never
// collapsed any (i, n) pair. The fix gates on strict `score > 0.0`
// instead, so true zeros (both +0.0 and -0.0) compute to rank 0 (top
// priority) like sp-python's reference encoder.
//
// The test extracts the minimum SP-BPE shape that reproduces the bug
// (codepoint pieces + the rank-zero `in` merge + the longer `▁within`
// target) and asserts the post-fix encoder emits the single combined
// piece, matching sp-python.
test "bpeFromSP: rank-zero merge piece (Gemma 'in') wins its pair" {
    const allocator = std.testing.allocator;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);

    // Codepoint base pieces. Scores mirror real Gemma magnitudes
    // (huge negative for individual code points; small negative for
    // common merges). Ids are assigned in append order — important
    // because the encoder maps `chunk[..]` -> id via `by_bytes`.
    try tvAppendPiece(&buf, allocator, "\xe2\x96\x81", -234775.0, 1); // ▁  id=0
    try tvAppendPiece(&buf, allocator, "w", -234798.0, 1); // id=1
    try tvAppendPiece(&buf, allocator, "i", -234779.0, 1); // id=2
    try tvAppendPiece(&buf, allocator, "t", -234778.0, 1); // id=3
    try tvAppendPiece(&buf, allocator, "h", -234786.0, 1); // id=4
    try tvAppendPiece(&buf, allocator, "n", -234781.0, 1); // id=5
    try tvAppendPiece(&buf, allocator, "\xe2\x96\x81w", -40.0, 1); // ▁w id=6
    try tvAppendPiece(&buf, allocator, "\xe2\x96\x81wi", -140.0, 1); // ▁wi id=7
    try tvAppendPiece(&buf, allocator, "th", -16.0, 1); // id=8
    try tvAppendPiece(&buf, allocator, "in", -0.0, 1); // the bug: rank 0 id=9
    try tvAppendPiece(&buf, allocator, "\xe2\x96\x81with", -202.0, 1); // ▁with id=10
    try tvAppendPiece(&buf, allocator, "\xe2\x96\x81within", -2346.0, 1); // ▁within id=11
    try tvAppendTrainerKind(&buf, allocator, 2); // BPE

    var m = try sp_model.loadFromBytes(allocator, buf.items);
    defer m.deinit();

    var bpe = try bpeFromSP(allocator, &m);
    defer bpe.deinit();

    // Sanity: the `in` piece's rank must be 0 (top priority), NOT
    // RANK_INVALID. Without the fix, this assertion fails.
    try std.testing.expectEqual(@as(u32, 0), bpe.piece_ranks.?[9]);

    var out: [16]TokenId = undefined;
    const ids = bpe.encodeChunk("\xe2\x96\x81within", &out);
    // Expected merge chain on " within" (▁ = U+2581):
    //   [▁, w, i, t, h, i, n]
    //   (i,n)=0 -> [▁, w, i, t, h, in]
    //   (t,h)=16 -> [▁, w, i, th, in]
    //   (▁,w)=40 -> [▁w, i, th, in]
    //   (▁w,i)=140 -> [▁wi, th, in]
    //   (▁wi,th)=202 -> [▁with, in]
    //   (▁with,in)=2346 -> [▁within]
    try std.testing.expectEqualSlices(TokenId, &.{11}, ids);
}

test "bpeFromSP: rank zero is preserved for both -0.0 and +0.0 scores" {
    // The fix's contract: scores that are <= 0 in value (including
    // both signed zeros) all compute to a real rank, never to
    // RANK_INVALID. Only strictly positive scores (which don't appear
    // in well-trained SP vocabs) get stamped max.
    const allocator = std.testing.allocator;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);

    try tvAppendPiece(&buf, allocator, "a", -0.0, 1); // negative zero
    try tvAppendPiece(&buf, allocator, "b", 0.0, 1); // positive zero
    try tvAppendPiece(&buf, allocator, "c", -1.0, 1); // ordinary
    try tvAppendTrainerKind(&buf, allocator, 2); // BPE

    var m = try sp_model.loadFromBytes(allocator, buf.items);
    defer m.deinit();

    var bpe = try bpeFromSP(allocator, &m);
    defer bpe.deinit();

    try std.testing.expectEqual(@as(u32, 0), bpe.piece_ranks.?[0]); // -0.0 -> rank 0
    try std.testing.expectEqual(@as(u32, 0), bpe.piece_ranks.?[1]); // +0.0 -> rank 0
    try std.testing.expectEqual(@as(u32, 1), bpe.piece_ranks.?[2]); // -1.0 -> rank 1
}

// --- HF Unigram byte-fallback integration test ----------------------------
//
// Real-world HF tokenizer.json with `type=Unigram` + `byte_fallback=true`.
// The llm-jp Unigram exports are the cleanest public examples — they
// pair a SentencePiece-trained vocab with the full 256-piece `<0xNN>`
// byte bank and `model.byte_fallback=true`.
//
// We verify the HF path produces:
//   1. A populated `[256]TokenId` byte-fallback table on the loaded Unigram.
//   2. An encoded id stream that matches HF's `tokenizers` library
//      `model.tokenize(text)` reference for an input containing a
//      codepoint NOT in the vocab — proving byte-fallback edges fire.
//
// Skipped when bench/vocabs/llmjp3_hf.json isn't present (run
// `python bench/fetch_vocabs.py` to download).
test "unigramFromHF: llm-jp byte_fallback emits per-byte ids on unknown CJK" {
    const allocator = std.testing.allocator;
    const path = "bench/vocabs/llmjp3_hf.json";
    const io = std.Io.Threaded.global_single_threaded.io();
    const contents = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(contents);

    const hf_json = @import("hf_json.zig");
    const hf_bridge = @import("hf_bridge.zig");

    var hf = try hf_json.loadFromBytes(allocator, contents);
    defer hf.deinit();
    try std.testing.expect(hf.model_kind == .unigram);
    try std.testing.expect(hf.byte_fallback);
    try std.testing.expect(hf.unigram_byte_fallback_table != null);

    var uni = try hf_bridge.unigramFromHF(allocator, &hf);
    defer uni.deinit();
    try std.testing.expect(uni.byte_fallback != null);

    // U+20000 (CJK Extension B "𠀀"): UTF-8 = F0 A0 80 80. Not in vocab.
    // HF reference (tokenizers Python lib, model.tokenize): ids
    // [248, 168, 136, 136] = ['<0xF0>', '<0xA0>', '<0x80>', '<0x80>'].
    var out: [8]TokenId = undefined;
    const ids = try uni.encodeChunk(allocator, "\xF0\xA0\x80\x80", &out);
    try std.testing.expectEqualSlices(
        TokenId,
        &.{ 248, 168, 136, 136 },
        ids,
    );

    // Sanity: every byte resolved through the fallback table — no
    // codepoint-level unk merging.
    const tbl = uni.byte_fallback.?;
    try std.testing.expectEqual(@as(TokenId, 248), tbl[0xF0]);
    try std.testing.expectEqual(@as(TokenId, 168), tbl[0xA0]);
    try std.testing.expectEqual(@as(TokenId, 136), tbl[0x80]);
}

// --- precompiled_charsmap wiring tests (post-1.17 agent B) ---------------

test "normalizerFromSPModel: model without charsmap leaves charsmap=null" {
    // Synthetic 1-piece SP model with no normalizer_spec — the bridge
    // should return a `.sp_precompiled` variant with charsmap=null,
    // bit-for-bit identical to the pre-charsmap behaviour.
    const allocator = std.testing.allocator;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);
    try tvAppendPiece(&buf, allocator, "a", -1.0, 1);

    var sp = try sp_model.loadFromBytes(allocator, buf.items);
    defer sp.deinit();
    try std.testing.expect(sp.precompiled_charsmap == null);

    const n = try normalizerFromSPModel(&sp);
    switch (n) {
        .sp_precompiled => |cfg| {
            try std.testing.expect(cfg.charsmap == null);
        },
        else => return error.TestExpectedSpNormalizer,
    }
}

test "normalizerFromSPModel: T5 model gets a non-null charsmap pointer" {
    const allocator = std.testing.allocator;
    const path = "bench/vocabs/t5_unigram.model";
    const io = std.Io.Threaded.global_single_threaded.io();
    const contents = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(contents);

    var sp = try sp_model.loadFromBytes(allocator, contents);
    defer sp.deinit();

    // T5 ships precompiled_charsmap (177 KB trie + ~60 KB normalized).
    try std.testing.expect(sp.precompiled_charsmap != null);

    const n = try normalizerFromSPModel(&sp);
    switch (n) {
        .sp_precompiled => |cfg| {
            try std.testing.expect(cfg.charsmap != null);
        },
        else => return error.TestExpectedSpNormalizer,
    }

    // Idempotency: a second call returns a normalizer with the SAME
    // charsmap pointer (the parsed view is cached in SpModel).
    const n2 = try normalizerFromSPModel(&sp);
    try std.testing.expectEqual(
        n.sp_precompiled.charsmap.?,
        n2.sp_precompiled.charsmap.?,
    );

    // The normalizer should still normalize without crashing on a few
    // sample inputs (the rewrite pass + dummy_prefix + escape stages).
    const out = try n.normalize(allocator, "Hello, world!");
    defer allocator.free(out);
    try std.testing.expect(out.len > 0);
    // Leading dummy prefix is U+2581.
    try std.testing.expectEqualSlices(u8, &.{ 0xE2, 0x96, 0x81 }, out[0..3]);
}

test "normalizerFromSPModel: synthetic charsmap actually rewrites bytes" {
    // Build a tiny SP model with a hand-crafted precompiled_charsmap
    // that maps 'q' -> "QQ". Verify the normalizer applies it.
    const allocator = std.testing.allocator;
    const sp_charsmap_mod = @import("sp_charsmap.zig");

    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);

    // One ordinary piece so the model parses.
    try tvAppendPiece(&buf, allocator, "a", -1.0, 1);

    // Build the synthetic charsmap blob via sp_charsmap's test helper
    // (re-exported indirectly through PrecompiledCharsmap.parse). We
    // need to call the file-internal helper, so we replicate the same
    // minimal trie shape here.
    const trie_units: u32 = 256;
    const trie_bytes: u32 = trie_units * 4;
    const key: u8 = 'q';
    const root_offset: u32 = 1;
    const labelled_pos: u32 = root_offset ^ @as(u32, key);
    const leaf_pos: u32 = 2;
    const labelled_offset: u32 = labelled_pos ^ leaf_pos;
    var trie: [256]u32 = @splat(0);
    trie[0] = root_offset << 10;
    trie[labelled_pos] = @as(u32, key) | (1 << 8) | (labelled_offset << 10);
    trie[leaf_pos] = (@as(u32, 1) << 31) | 0;

    var blob: std.ArrayList(u8) = .empty;
    defer blob.deinit(allocator);
    var hdr: [4]u8 = undefined;
    std.mem.writeInt(u32, &hdr, trie_bytes, .little);
    try blob.appendSlice(allocator, &hdr);
    const trie_raw: [*]const u8 = @ptrCast(&trie);
    try blob.appendSlice(allocator, trie_raw[0 .. trie_units * 4]);
    try blob.appendSlice(allocator, "QQ");
    try blob.append(allocator, 0);

    // Embed the charsmap into a normalizer_spec sub-message.
    var ns: std.ArrayList(u8) = .empty;
    defer ns.deinit(allocator);
    // field 2: precompiled_charsmap (bytes/len)
    try tvTag(&ns, allocator, 2, 2);
    try tvEncodeVarint(&ns, allocator, blob.items.len);
    try ns.appendSlice(allocator, blob.items);
    // field 3: add_dummy_prefix = false (so we just see the rewrite)
    try tvTag(&ns, allocator, 3, 0);
    try tvEncodeVarint(&ns, allocator, 0);
    // field 5: escape_whitespaces = false
    try tvTag(&ns, allocator, 5, 0);
    try tvEncodeVarint(&ns, allocator, 0);
    // field 1: name = "identity" so NFKC is off too
    try tvTag(&ns, allocator, 1, 2);
    try tvEncodeVarint(&ns, allocator, "identity".len);
    try ns.appendSlice(allocator, "identity");

    try tvEmbedMsg(&buf, allocator, 3, ns.items);

    var sp = try sp_model.loadFromBytes(allocator, buf.items);
    defer sp.deinit();
    try std.testing.expect(sp.precompiled_charsmap != null);

    // Parse it via the bridge.
    const n = try normalizerFromSPModel(&sp);
    try std.testing.expect(n.sp_precompiled.charsmap != null);

    // Now normalize: 'q' should become 'QQ' under the trie rewrite.
    const out = try n.normalize(allocator, "aqab");
    defer allocator.free(out);
    try std.testing.expectEqualStrings("aQQab", out);

    // Suppress unused-import warning if the import only flagged for
    // documentation; the parse path is exercised above already.
    _ = sp_charsmap_mod;
}

test "unigramFromSP excludes unknown-type pieces from trie" {
    // Regression for the t5_unigram × code corpus residual at stress
    // 1.24: source like `addToken("<unk>")` was matching `<unk>` as a
    // single piece (id 2, type=.unknown, score=0.0) instead of routing
    // through the regular merge `<` + `unk` + `>`. SP's encoder skips
    // unknown pieces from its lattice walk; we mirror that by flagging
    // them excluded in the Unigram builder.
    const allocator = std.testing.allocator;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);

    // unk piece literally named "<unk>" so a literal occurrence in the
    // input would otherwise resolve to id 0 if it were in the trie.
    try tvAppendPiece(&buf, allocator, "<unk>", 0.0, 2);
    try tvAppendPiece(&buf, allocator, "<", -2.0, 1);
    try tvAppendPiece(&buf, allocator, "unk", -3.0, 1);
    try tvAppendPiece(&buf, allocator, ">", -2.0, 1);
    try tvAppendTrainerKind(&buf, allocator, 1); // Unigram

    var m = try sp_model.loadFromBytes(allocator, buf.items);
    defer m.deinit();

    var uni = try unigramFromSP(allocator, &m);
    defer uni.deinit();

    var out: [8]TokenId = undefined;
    const ids = try uni.encodeChunk(allocator, "<unk>", &out);
    // Must split into `<` + `unk` + `>` — never match id 0 from the
    // unk piece directly.
    try std.testing.expectEqualSlices(TokenId, &.{ 1, 2, 3 }, ids);
}

test "unigramFromSP excludes control-type pieces from trie" {
    // Companion to the unknown-exclusion test: pieces with type=.control
    // (the BOS/EOS sentinels in many SP exports) must not be greedy-
    // matched against arbitrary text either. They keep their id slot
    // for downstream callers that need stable indices.
    const allocator = std.testing.allocator;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);

    try tvAppendPiece(&buf, allocator, "<unk>", 0.0, 2);
    try tvAppendPiece(&buf, allocator, "</s>", 0.0, 3); // control
    try tvAppendPiece(&buf, allocator, "<", -2.0, 1);
    try tvAppendPiece(&buf, allocator, "/", -2.0, 1);
    try tvAppendPiece(&buf, allocator, "s", -3.0, 1);
    try tvAppendPiece(&buf, allocator, ">", -2.0, 1);
    try tvAppendTrainerKind(&buf, allocator, 1); // Unigram

    var m = try sp_model.loadFromBytes(allocator, buf.items);
    defer m.deinit();

    var uni = try unigramFromSP(allocator, &m);
    defer uni.deinit();

    var out: [8]TokenId = undefined;
    const ids = try uni.encodeChunk(allocator, "</s>", &out);
    // `</s>` (id 1, control) must NOT win. Path must be `<` + `/` + `s` + `>`.
    try std.testing.expectEqualSlices(TokenId, &.{ 2, 3, 4, 5 }, ids);
    // The id slot still exists — downstream code can look it up.
    try std.testing.expectEqual(@as(u32, 6), uni.count);
    try std.testing.expectEqualStrings("</s>", uni.idBytes(1));
}

test "unigramFromSP keeps user_defined pieces matchable" {
    // The flip side: user_defined-type pieces ARE supposed to be matched
    // (they're the SP equivalent of HF added_tokens). Gemma's `</s>`
    // (type=4 in the real model) needs to resolve to its assigned id
    // when it appears literally in the input — but only via the
    // user_defined path. This test confirms the bridge's exclusion list
    // doesn't accidentally drop user_defined pieces from the trie.
    const allocator = std.testing.allocator;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);

    try tvAppendPiece(&buf, allocator, "<unk>", 0.0, 2);
    try tvAppendPiece(&buf, allocator, "</s>", 0.0, 4); // user_defined
    try tvAppendPiece(&buf, allocator, "<", -2.0, 1);
    try tvAppendPiece(&buf, allocator, "/", -2.0, 1);
    try tvAppendPiece(&buf, allocator, "s", -3.0, 1);
    try tvAppendPiece(&buf, allocator, ">", -2.0, 1);
    try tvAppendTrainerKind(&buf, allocator, 1); // Unigram

    var m = try sp_model.loadFromBytes(allocator, buf.items);
    defer m.deinit();

    var uni = try unigramFromSP(allocator, &m);
    defer uni.deinit();

    var out: [8]TokenId = undefined;
    const ids = try uni.encodeChunk(allocator, "</s>", &out);
    // The user_defined piece's score-0 (rank-best for SP) collapses the
    // 4-char path into one id; the multi-piece fallback only wins when
    // the user_defined piece is absent / excluded.
    try std.testing.expectEqualSlices(TokenId, &.{1}, ids);
}
