//! Vocab merge: take two vocabs of the SAME model kind (BPE+BPE,
//! Unigram+Unigram, or WordPiece+WordPiece) and produce a single merged
//! vocab whose token table is the union of the two.
//!
//! Guarantees:
//!   * Original vocab A ids are preserved bit-identical (id, bytes, and
//!     any auxiliary fields like Unigram scores / WordPiece unk_id).
//!   * Vocab B's tokens are appended starting at `vocab_a.count`.
//!     Optionally each B token is prefixed with `--prefix-b STR` so two
//!     unrelated vocabs can be unioned without byte collisions.
//!   * Same-bytes collisions between A and B are reported and resolved
//!     per the `OnConflict` policy (`.error_out`, `.keep_a`, `.keep_b`).
//!     `.keep_a` is the most useful default once you flip away from the
//!     CLI's `.error_out` default — it drops the colliding B token from
//!     the merged vocab.
//!   * Cross-kind merges (e.g. BPE + Unigram) are rejected up-front with
//!     `Error.IncompatibleModelKind`.
//!
//! The CLI entry point lives in `main.zig` as `ztok merge-vocab`. The
//! caller is expected to:
//!   1. Auto-detect both vocab paths.
//!   2. Reject cross-kind pairs (the merge function does this too, but
//!      the CLI prints a friendlier error first).
//!   3. Call `mergeBpe` / `mergeUnigram` / `mergeWordPiece`.
//!   4. Serialize the result back using whatever writer matches vocab
//!      A's original format (`hf_writer`, `sp_writer`, or `writeTiktoken`
//!      from `vocab_continued_pretrain.zig`).
//!   5. Write the JSON sidecar via `writeMergeMap`.
//!
//! `b_to_merged` in the JSON sidecar lets downstream tooling re-index
//! the embedding table for vocab B: row `b_id` in B's embedding matrix
//! becomes row `b_to_merged[b_id]` in the merged matrix. Entries whose
//! value is `null` were dropped (collision under `.keep_a`).

const std = @import("std");

const TokenId = @import("token.zig").TokenId;
const Bpe = @import("bpe.zig").Bpe;
const Unigram = @import("unigram.zig").Unigram;
const WordPiece = @import("wordpiece.zig").WordPiece;

pub const Error = error{
    /// Vocab A and B are different model kinds; merge is undefined.
    IncompatibleModelKind,
    /// `--on-conflict error` and at least one same-bytes collision
    /// occurred between A and B. Also raised when `--prefix-b` produces
    /// a token that happens to already exist in vocab A — the same
    /// logical situation as a raw collision.
    BytesConflict,
} || std.mem.Allocator.Error;

pub const OnConflict = enum {
    /// Reject the merge if any B token's bytes (after `prefix_b` is
    /// applied) match an existing A token. Default.
    error_out,
    /// Drop the conflicting B token; the merged vocab keeps A's id
    /// unchanged and the JSON sidecar records `b_to_merged[b_id] = null`.
    keep_a,
    /// Drop the conflicting A token (well — its merged-vocab id) and
    /// replace it with B's bytes/score. Rarely useful in practice
    /// because doing so would break the "A ids are bit-identical"
    /// guarantee, so this variant instead remaps the colliding B token
    /// to point at the existing A id (B "loses" its own slot, but its
    /// embedding-row gets reused from A). The JSON sidecar records the
    /// remap so the caller can rebuild the embedding table.
    keep_b,
};

pub const Options = struct {
    on_conflict: OnConflict = .error_out,
    /// If non-empty, prepended to every B token's bytes BEFORE the
    /// collision + append step. e.g. `prefix_b = "B_"` turns B's
    /// `"hello"` into `"B_hello"` in the merged vocab.
    prefix_b: []const u8 = "",
};

/// Common result shape across BPE / Unigram / WordPiece merges. The
/// caller owns `b_to_merged` (slice of `u32`) and must free it via the
/// allocator passed to the merge function. The contained model value
/// (`merged_bpe` / `merged_unigram` / `merged_wordpiece`) is also caller-
/// owned via its own `deinit()`.
pub fn MergeResult(comptime Model: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,

        /// Merged model. Tokens 0..original_a_size are bit-identical to
        /// vocab A; appended tokens follow in B's original order, minus
        /// any dropped collisions.
        merged: Model,

        /// Original counts before union.
        original_a_size: u32,
        original_b_size: u32,
        /// `merged.count` cached here for sidecar emission.
        merged_size: u32,

        /// For each B id, the corresponding merged id (or `NULL_ID` for
        /// dropped collisions under `.keep_a`). Length == original_b_size.
        b_to_merged: []TokenId,

        /// Count of same-bytes A/B collisions seen (before applying the
        /// policy). Useful diagnostic regardless of policy.
        conflicts_count: u32,

        pub fn deinit(self: *Self) void {
            if (self.b_to_merged.len > 0) self.allocator.free(self.b_to_merged);
            self.merged.deinit();
            self.* = undefined;
        }
    };
}

pub const NULL_ID: TokenId = std.math.maxInt(TokenId);

/// Apply the prefix once and return the borrowed-or-allocated B-bytes
/// slice for a single token. Caller frees only when `out_owned` is set.
fn prefixedBytes(
    allocator: std.mem.Allocator,
    prefix: []const u8,
    src: []const u8,
    scratch: *std.ArrayList(u8),
) ![]const u8 {
    if (prefix.len == 0) return src;
    scratch.clearRetainingCapacity();
    try scratch.ensureTotalCapacity(allocator, prefix.len + src.len);
    try scratch.appendSlice(allocator, prefix);
    try scratch.appendSlice(allocator, src);
    return scratch.items;
}

// =============================== BPE ===============================

pub fn mergeBpe(
    allocator: std.mem.Allocator,
    a: *const Bpe,
    b: *const Bpe,
    opts: Options,
) Error!MergeResult(Bpe) {
    const na = a.count;
    const nb = b.count;

    // Pass 1: count collisions and decide which B ids survive.
    var b_to_merged = try allocator.alloc(TokenId, nb);
    errdefer allocator.free(b_to_merged);
    @memset(b_to_merged, NULL_ID);

    var conflicts: u32 = 0;
    var append_count: u32 = 0;
    var total_b_extra_bytes: usize = 0;
    var scratch: std.ArrayList(u8) = .empty;
    defer scratch.deinit(allocator);

    var bi: u32 = 0;
    while (bi < nb) : (bi += 1) {
        const raw = b.idBytes(bi);
        const effective = try prefixedBytes(allocator, opts.prefix_b, raw, &scratch);
        if (a.by_bytes.get(effective)) |a_id| {
            conflicts += 1;
            switch (opts.on_conflict) {
                .error_out => return Error.BytesConflict,
                .keep_a => {
                    // B id dropped from merged vocab.
                    b_to_merged[bi] = NULL_ID;
                },
                .keep_b => {
                    // Remap B id onto A's existing id. No new slot.
                    b_to_merged[bi] = a_id;
                },
            }
        } else {
            b_to_merged[bi] = na + append_count;
            append_count += 1;
            total_b_extra_bytes += effective.len;
        }
    }

    // Pass 2: build the merged bytes/offsets buffers. We copy A verbatim
    // (preserves ids bit-identical) and then append the surviving B
    // tokens in B's original id order (skipping dropped ones).
    const merged_count = na + append_count;
    const total_bytes = a.bytes.len + total_b_extra_bytes;
    const new_bytes = try allocator.alloc(u8, total_bytes);
    errdefer allocator.free(new_bytes);
    const new_offsets = try allocator.alloc(u32, @as(usize, merged_count) + 1);
    errdefer allocator.free(new_offsets);

    @memcpy(new_bytes[0..a.bytes.len], a.bytes);
    @memcpy(new_offsets[0 .. na + 1], a.offsets);

    var write_off: u32 = @intCast(a.bytes.len);
    bi = 0;
    while (bi < nb) : (bi += 1) {
        const merged_id = b_to_merged[bi];
        if (merged_id == NULL_ID) continue;
        if (merged_id < na) continue; // .keep_b remap onto A — no new slot
        const raw = b.idBytes(bi);
        const effective = try prefixedBytes(allocator, opts.prefix_b, raw, &scratch);
        @memcpy(new_bytes[write_off .. write_off + effective.len], effective);
        write_off += @intCast(effective.len);
        new_offsets[merged_id + 1] = write_off;
    }
    std.debug.assert(write_off == total_bytes);

    // by_bytes for the merged vocab. Keys borrow from new_bytes.
    var by_bytes: std.StringHashMap(TokenId) = .init(allocator);
    errdefer by_bytes.deinit();
    try by_bytes.ensureTotalCapacity(merged_count);
    var i: u32 = 0;
    while (i < merged_count) : (i += 1) {
        const key = new_bytes[new_offsets[i]..new_offsets[i + 1]];
        try by_bytes.put(key, i);
    }

    // Hot table rebuild — match the prune module's approach so the
    // merged BPE stays on the fast L1d path.
    const hot_table = try Bpe.buildHotTable(allocator, new_bytes, new_offsets, merged_count);
    errdefer allocator.free(hot_table);

    // Recompute max_piece_len since B may have introduced longer pieces.
    var max_piece_len: u32 = a.max_piece_len;
    {
        var k: u32 = 0;
        while (k < merged_count) : (k += 1) {
            const len: u32 = new_offsets[k + 1] - new_offsets[k];
            if (len > max_piece_len) max_piece_len = len;
        }
    }
    if (max_piece_len == 0) max_piece_len = 1;

    const merged: Bpe = .{
        .allocator = allocator,
        .bytes = new_bytes,
        .offsets = new_offsets,
        .count = merged_count,
        .by_bytes = by_bytes,
        .hot_table = hot_table,
        .byte_fallback = a.byte_fallback,
        .encode_mode = a.encode_mode,
        .max_piece_len = max_piece_len,
        // piece_ranks: A's ranks no longer cover the appended ids, so we
        // drop them. Re-encoding falls back to the merge-rank-by-id
        // path, which is correct for the union (just not as compact).
        .piece_ranks = null,
        .ignore_merges = a.ignore_merges,
    };

    return .{
        .allocator = allocator,
        .merged = merged,
        .original_a_size = na,
        .original_b_size = nb,
        .merged_size = merged_count,
        .b_to_merged = b_to_merged,
        .conflicts_count = conflicts,
    };
}

// =============================== Unigram ===============================

pub fn mergeUnigram(
    allocator: std.mem.Allocator,
    a: *const Unigram,
    b: *const Unigram,
    opts: Options,
) Error!MergeResult(Unigram) {
    const na = a.count;
    const nb = b.count;

    // Build A's bytes-> id map up front so we can detect collisions in
    // O(1). Unlike BPE, the Unigram struct doesn't carry one.
    var a_by_bytes: std.StringHashMap(TokenId) = .init(allocator);
    defer a_by_bytes.deinit();
    try a_by_bytes.ensureTotalCapacity(na);
    {
        var i: u32 = 0;
        while (i < na) : (i += 1) {
            try a_by_bytes.put(a.idBytes(i), i);
        }
    }

    var b_to_merged = try allocator.alloc(TokenId, nb);
    errdefer allocator.free(b_to_merged);
    @memset(b_to_merged, NULL_ID);

    var scratch: std.ArrayList(u8) = .empty;
    defer scratch.deinit(allocator);

    var conflicts: u32 = 0;
    var append_count: u32 = 0;

    var bi: u32 = 0;
    while (bi < nb) : (bi += 1) {
        const raw = b.idBytes(bi);
        const effective = try prefixedBytes(allocator, opts.prefix_b, raw, &scratch);
        if (a_by_bytes.get(effective)) |a_id| {
            conflicts += 1;
            switch (opts.on_conflict) {
                .error_out => return Error.BytesConflict,
                .keep_a => b_to_merged[bi] = NULL_ID,
                .keep_b => b_to_merged[bi] = a_id,
            }
        } else {
            b_to_merged[bi] = na + append_count;
            append_count += 1;
        }
    }

    // Build a fresh Unigram via the Builder so the trie is rebuilt for
    // the merged vocab in one shot.
    var builder = Unigram.Builder.init(allocator);
    errdefer builder.deinit();

    // Copy A verbatim — same byte ranges and scores, ids 0..na-1.
    {
        var i: u32 = 0;
        while (i < na) : (i += 1) {
            _ = try builder.addToken(a.idBytes(i), a.scores[i]);
        }
    }
    // Append surviving B tokens in B-id order.
    bi = 0;
    while (bi < nb) : (bi += 1) {
        const merged_id = b_to_merged[bi];
        if (merged_id == NULL_ID) continue;
        if (merged_id < na) continue; // remapped to A — no new slot
        const raw = b.idBytes(bi);
        const effective = try prefixedBytes(allocator, opts.prefix_b, raw, &scratch);
        _ = try builder.addToken(effective, b.scores[bi]);
    }

    // unk_id stays A's. If A had no unk_id, fall back to 0 (Builder
    // contract requires unk_id < count).
    const unk_id: TokenId = if (a.count > 0) a.unk_id else 0;
    var merged = try builder.finalize(unk_id);
    errdefer merged.deinit();

    // bos/eos hints carry over from A.
    merged.bos_id = a.bos_id;
    merged.eos_id = a.eos_id;
    merged.byte_fallback = a.byte_fallback;

    const merged_count = merged.count;
    return .{
        .allocator = allocator,
        .merged = merged,
        .original_a_size = na,
        .original_b_size = nb,
        .merged_size = merged_count,
        .b_to_merged = b_to_merged,
        .conflicts_count = conflicts,
    };
}

// =============================== WordPiece ===============================

pub fn mergeWordPiece(
    allocator: std.mem.Allocator,
    a: *const WordPiece,
    b: *const WordPiece,
    opts: Options,
) Error!MergeResult(WordPiece) {
    const na = a.count;
    const nb = b.count;

    var b_to_merged = try allocator.alloc(TokenId, nb);
    errdefer allocator.free(b_to_merged);
    @memset(b_to_merged, NULL_ID);

    var scratch: std.ArrayList(u8) = .empty;
    defer scratch.deinit(allocator);

    var conflicts: u32 = 0;
    var append_count: u32 = 0;

    var bi: u32 = 0;
    while (bi < nb) : (bi += 1) {
        const raw = b.idBytes(bi);
        const effective = try prefixedBytes(allocator, opts.prefix_b, raw, &scratch);
        if (a.by_bytes.get(effective)) |a_id| {
            conflicts += 1;
            switch (opts.on_conflict) {
                .error_out => return Error.BytesConflict,
                .keep_a => b_to_merged[bi] = NULL_ID,
                .keep_b => b_to_merged[bi] = a_id,
            }
        } else {
            b_to_merged[bi] = na + append_count;
            append_count += 1;
        }
    }

    const merged_count = na + append_count;

    // Materialise the merged token list as a slice of byte slices so we
    // can hand it to WordPiece.init in one go. The slices borrow either
    // into A's `bytes` or into the per-token scratch we own here — we
    // need to hold both until init() finishes copying.
    const merged_tokens = try allocator.alloc([]const u8, merged_count);
    defer allocator.free(merged_tokens);

    // B tokens may need prefixing, so allocate per-prefixed slices on
    // an arena (only when the prefix is non-empty). Plain pass-through
    // for prefix_b == "" avoids the per-token alloc.
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const ar = arena.allocator();

    {
        var i: u32 = 0;
        while (i < na) : (i += 1) {
            merged_tokens[i] = a.idBytes(i);
        }
    }
    bi = 0;
    var write_idx: u32 = na;
    while (bi < nb) : (bi += 1) {
        const merged_id = b_to_merged[bi];
        if (merged_id == NULL_ID) continue;
        if (merged_id < na) continue;
        const raw = b.idBytes(bi);
        if (opts.prefix_b.len == 0) {
            merged_tokens[write_idx] = raw;
        } else {
            const combined = try ar.alloc(u8, opts.prefix_b.len + raw.len);
            @memcpy(combined[0..opts.prefix_b.len], opts.prefix_b);
            @memcpy(combined[opts.prefix_b.len..], raw);
            merged_tokens[write_idx] = combined;
        }
        write_idx += 1;
    }

    const unk_id: TokenId = if (a.unk_id < merged_count) a.unk_id else 0;
    var merged = try WordPiece.init(allocator, merged_tokens, .{
        .unk_id = unk_id,
        .continuing_subword_prefix = a.continuing_subword_prefix,
        .max_input_chars_per_word = a.max_input_chars_per_word,
    });
    errdefer merged.deinit();

    return .{
        .allocator = allocator,
        .merged = merged,
        .original_a_size = na,
        .original_b_size = nb,
        .merged_size = merged_count,
        .b_to_merged = b_to_merged,
        .conflicts_count = conflicts,
    };
}

// =============================== JSON sidecar ===============================

/// Common shape used by `writeMergeMap`. Pass any `MergeResult(*)`'s
/// fields manually so this function stays kind-agnostic.
pub const MergeMapMeta = struct {
    original_a_size: u32,
    original_b_size: u32,
    merged_size: u32,
    conflicts_count: u32,
    b_to_merged: []const TokenId,
};

pub fn writeMergeMap(
    allocator: std.mem.Allocator,
    meta: MergeMapMeta,
    out_path: []const u8,
) !void {
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var s: std.json.Stringify = .{
        .writer = &aw.writer,
        .options = .{ .whitespace = .indent_2 },
    };
    try s.beginObject();
    try s.objectField("original_a_size");
    try s.write(@as(i64, @intCast(meta.original_a_size)));
    try s.objectField("original_b_size");
    try s.write(@as(i64, @intCast(meta.original_b_size)));
    try s.objectField("merged_size");
    try s.write(@as(i64, @intCast(meta.merged_size)));
    try s.objectField("conflicts_count");
    try s.write(@as(i64, @intCast(meta.conflicts_count)));
    try s.objectField("b_to_merged");
    try s.beginObject();
    for (meta.b_to_merged, 0..) |merged_id, b_id| {
        var key_buf: [16]u8 = undefined;
        const key_len = std.fmt.printInt(&key_buf, b_id, 10, .lower, .{});
        try s.objectField(key_buf[0..key_len]);
        if (merged_id == NULL_ID) {
            try s.write(null);
        } else {
            try s.write(@as(i64, @intCast(merged_id)));
        }
    }
    try s.endObject();
    try s.endObject();

    const io = std.Io.Threaded.global_single_threaded.io();
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out_path, .data = aw.writer.buffered() });
}

// =============================== tests ===============================

const testing = std.testing;

const TestEntry = struct { bytes: []const u8, rank: u32 };

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

fn buildByteBpe(allocator: std.mem.Allocator, extras: []const TestEntry) !Bpe {
    var entries: std.ArrayList(TestEntry) = .empty;
    defer entries.deinit(allocator);
    var byte_holders: [256][1]u8 = undefined;
    var i: u32 = 0;
    while (i < 256) : (i += 1) {
        byte_holders[i][0] = @intCast(i);
        try entries.append(allocator, .{ .bytes = byte_holders[i][0..1], .rank = i });
    }
    for (extras) |e| try entries.append(allocator, e);
    const src = try buildVocabSource(allocator, entries.items);
    defer allocator.free(src);
    return Bpe.loadTiktokenBytes(allocator, src);
}

test "mergeBpe appends B at A.count and preserves A ids" {
    var a = try buildByteBpe(testing.allocator, &.{
        .{ .bytes = "ab", .rank = 256 },
        .{ .bytes = "cd", .rank = 257 },
    });
    defer a.deinit();
    // B uses a non-overlapping prefix so there's no natural collision
    // on the byte fallbacks (we prefix B to dodge them entirely).
    var b = try buildByteBpe(testing.allocator, &.{
        .{ .bytes = "ef", .rank = 256 },
        .{ .bytes = "gh", .rank = 257 },
    });
    defer b.deinit();

    var res = try mergeBpe(testing.allocator, &a, &b, .{
        .on_conflict = .keep_a,
        .prefix_b = "B_",
    });
    defer res.deinit();

    try testing.expectEqual(a.count, res.original_a_size);
    try testing.expectEqual(b.count, res.original_b_size);
    // No collisions because every B token gained the "B_" prefix.
    try testing.expectEqual(@as(u32, 0), res.conflicts_count);
    try testing.expectEqual(a.count + b.count, res.merged_size);

    // Every A id decodes to the same bytes.
    var id: u32 = 0;
    while (id < a.count) : (id += 1) {
        try testing.expectEqualSlices(u8, a.idBytes(id), res.merged.idBytes(id));
    }
    // Every B id appended at a.count and onward, in B id order, with
    // the "B_" prefix applied.
    var bi: u32 = 0;
    while (bi < b.count) : (bi += 1) {
        const merged_id = res.b_to_merged[bi];
        try testing.expect(merged_id != NULL_ID);
        try testing.expectEqual(a.count + bi, merged_id);
        const raw = b.idBytes(bi);
        const expected = try std.fmt.allocPrint(testing.allocator, "B_{s}", .{raw});
        defer testing.allocator.free(expected);
        try testing.expectEqualSlices(u8, expected, res.merged.idBytes(merged_id));
    }
}

test "mergeBpe prefix-b is applied to every B token" {
    var a = try buildByteBpe(testing.allocator, &.{
        .{ .bytes = "xy", .rank = 256 },
    });
    defer a.deinit();
    var b = try buildByteBpe(testing.allocator, &.{
        .{ .bytes = "qrs", .rank = 256 },
        .{ .bytes = "tuv", .rank = 257 },
    });
    defer b.deinit();

    var res = try mergeBpe(testing.allocator, &a, &b, .{
        .on_conflict = .keep_a,
        .prefix_b = "PFX_",
    });
    defer res.deinit();

    // Locate the prefixed pieces in the merged vocab via lookup.
    const id_qrs = res.merged.by_bytes.get("PFX_qrs");
    const id_tuv = res.merged.by_bytes.get("PFX_tuv");
    try testing.expect(id_qrs != null);
    try testing.expect(id_tuv != null);
    // And the un-prefixed forms must NOT exist (they came only from B).
    try testing.expect(res.merged.by_bytes.get("qrs") == null);
    try testing.expect(res.merged.by_bytes.get("tuv") == null);
}

test "mergeBpe .error_out errors on byte collision" {
    var a = try buildByteBpe(testing.allocator, &.{
        .{ .bytes = "shared", .rank = 256 },
    });
    defer a.deinit();
    var b = try buildByteBpe(testing.allocator, &.{
        .{ .bytes = "shared", .rank = 256 },
    });
    defer b.deinit();

    const err = mergeBpe(testing.allocator, &a, &b, .{ .on_conflict = .error_out });
    try testing.expectError(Error.BytesConflict, err);
}

test "mergeBpe .keep_a drops conflicting B tokens" {
    // Both vocabs share the 256 byte fallbacks AND one merge piece
    // ("shared"). With prefix_b="" we'll see 257 collisions. Apply a
    // prefix to dodge the byte-fallback collisions, then plant a single
    // genuine collision through the prefix.
    var a = try buildByteBpe(testing.allocator, &.{
        .{ .bytes = "PFX_shared", .rank = 256 }, // matches the prefixed B token
        .{ .bytes = "aonly", .rank = 257 },
    });
    defer a.deinit();
    var b = try buildByteBpe(testing.allocator, &.{
        .{ .bytes = "shared", .rank = 256 }, // becomes "PFX_shared" => collides
        .{ .bytes = "bonly", .rank = 257 }, // becomes "PFX_bonly" => unique
    });
    defer b.deinit();

    var res = try mergeBpe(testing.allocator, &a, &b, .{
        .on_conflict = .keep_a,
        .prefix_b = "PFX_",
    });
    defer res.deinit();

    // B id 0..255 are byte fallbacks; with the "PFX_" prefix they all
    // become 5-byte sequences not in A. Only the planted "shared" hit
    // collides at B id 256. B id 257 ("bonly" -> "PFX_bonly") is
    // appended after the 256 byte-fallbacks at merged id a.count+256.
    try testing.expectEqual(@as(u32, 1), res.conflicts_count);
    try testing.expectEqual(NULL_ID, res.b_to_merged[256]);
    try testing.expect(res.b_to_merged[257] != NULL_ID);
    try testing.expectEqual(a.count + 256, res.b_to_merged[257]);
    // Merged size = a.count + (256 prefixed byte-fallbacks) + 1
    // ("PFX_bonly"); the collided "PFX_shared" is dropped, not counted.
    try testing.expectEqual(a.count + 256 + 1, res.merged_size);
    // "PFX_shared" still resolves to A's original id (256).
    const shared_id = res.merged.by_bytes.get("PFX_shared");
    try testing.expect(shared_id != null);
    try testing.expectEqual(@as(TokenId, 256), shared_id.?);
}

test "writeMergeMap produces well-formed JSON" {
    var a = try buildByteBpe(testing.allocator, &.{
        .{ .bytes = "x", .rank = 256 },
    });
    defer a.deinit();
    var b = try buildByteBpe(testing.allocator, &.{
        .{ .bytes = "y", .rank = 256 },
        .{ .bytes = "z", .rank = 257 },
    });
    defer b.deinit();

    var res = try mergeBpe(testing.allocator, &a, &b, .{
        .on_conflict = .keep_a,
        .prefix_b = "B_",
    });
    defer res.deinit();

    const path = "/tmp/ztok_merge_map_test.json";
    try writeMergeMap(testing.allocator, .{
        .original_a_size = res.original_a_size,
        .original_b_size = res.original_b_size,
        .merged_size = res.merged_size,
        .conflicts_count = res.conflicts_count,
        .b_to_merged = res.b_to_merged,
    }, path);
    defer {
        const io = std.Io.Threaded.global_single_threaded.io();
        std.Io.Dir.cwd().deleteFile(io, path) catch {};
    }

    const io = std.Io.Threaded.global_single_threaded.io();
    const contents = try std.Io.Dir.cwd().readFileAlloc(io, path, testing.allocator, .unlimited);
    defer testing.allocator.free(contents);

    // Parse and check the shape: must round-trip as an object with the
    // four scalar fields + the b_to_merged map.
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, contents, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    try testing.expect(root.contains("original_a_size"));
    try testing.expect(root.contains("original_b_size"));
    try testing.expect(root.contains("merged_size"));
    try testing.expect(root.contains("conflicts_count"));
    try testing.expect(root.contains("b_to_merged"));
    const b2m = root.get("b_to_merged").?.object;
    try testing.expectEqual(@as(usize, res.original_b_size), b2m.count());
}

test "mergeUnigram appends + preserves A scores" {
    var ba = Unigram.Builder.init(testing.allocator);
    defer ba.deinit();
    _ = try ba.addToken("alpha", -1.0);
    _ = try ba.addToken("beta", -2.0);
    const unk_a = try ba.addToken("<unk>", -10.0);
    var a = try ba.finalize(unk_a);
    defer a.deinit();

    var bb = Unigram.Builder.init(testing.allocator);
    defer bb.deinit();
    _ = try bb.addToken("gamma", -3.0);
    _ = try bb.addToken("delta", -4.0);
    // No "<unk>" in B to avoid collision with A's "<unk>".
    var b = try bb.finalize(0);
    defer b.deinit();

    var res = try mergeUnigram(testing.allocator, &a, &b, .{ .on_conflict = .error_out });
    defer res.deinit();

    try testing.expectEqual(a.count + b.count, res.merged_size);
    // A's ids preserved bit-identical including scores.
    try testing.expectEqualSlices(u8, "alpha", res.merged.idBytes(0));
    try testing.expectEqual(@as(f32, -1.0), res.merged.scores[0]);
    try testing.expectEqualSlices(u8, "beta", res.merged.idBytes(1));
    try testing.expectEqual(@as(f32, -2.0), res.merged.scores[1]);
    // B appended.
    try testing.expectEqualSlices(u8, "gamma", res.merged.idBytes(a.count));
    try testing.expectEqual(@as(f32, -3.0), res.merged.scores[a.count]);
    // unk_id stays A's.
    try testing.expectEqual(a.unk_id, res.merged.unk_id);
}
