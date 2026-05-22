//! Token healing.
//!
//! When a prompt ends mid-word, the tokenizer is forced to emit a token
//! boundary that the model would (almost) never produce when generating
//! that same text. The classic example: a prompt ending in `"hello wor"`
//! tokenizes the trailing `"wor"` as its own token, but during generation
//! the model would emit a single `" world"` / `"world"` token. Conditioning
//! continuation on the artificial `"wor"` boundary pushes probability mass
//! onto unnatural continuations.
//!
//! Token healing fixes this by *trimming* the trailing token(s) whose bytes
//! are a strict prefix of some longer vocab token, handing those bytes back
//! as `boundary_bytes`. The caller re-encodes `boundary_bytes ++ generated`
//! as a single unit via `continueFrom`, so generation resumes on a natural
//! token boundary.
//!
//! ## Conservatism rule
//!
//! We only trim a trailing run of tokens when the concatenation of their
//! bytes is a *strict* prefix of a genuinely LONGER vocab token. "Strict"
//! means the candidate token's bytes are not themselves a complete vocab
//! token's full extent — i.e. there exists a vocab token `t` with
//! `t.bytes.len > candidate.len` and `t.bytes` starts with `candidate`.
//! A token that is not a prefix of any longer token is left untouched and
//! the boundary is empty (a true no-op). This keeps healing from ever
//! corrupting a prompt that already ends on a natural boundary.
//!
//! ## Multi-token trailing heal
//!
//! Some healable suffixes span more than one token (e.g. `" wo"` + `"r"`).
//! We greedily try trimming up to `MAX_HEAL_TOKENS` trailing tokens, taking
//! the LONGEST trailing run whose combined bytes are still a strict prefix
//! of a longer token. Taking the longest run hands the maximum number of
//! ambiguous bytes back to the re-encoder, which is exactly the bytes whose
//! tokenization the boundary was distorting.

const std = @import("std");
const bpe_mod = @import("bpe.zig");
const token = @import("token.zig");

pub const TokenId = token.TokenId;
const Bpe = bpe_mod.Bpe;

/// Maximum number of trailing tokens we will fold into the boundary. Three
/// is plenty for real vocabs (a healable suffix is at most `max_piece_len`
/// bytes, which a handful of single-byte tokens already covers) and bounds
/// the work to a small constant.
pub const MAX_HEAL_TOKENS: usize = 3;

pub const HealResult = struct {
    /// The prompt token ids with the healable trailing run removed. Borrows
    /// nothing from the input; freshly allocated.
    healed_ids: []TokenId,
    /// The trailing bytes that were trimmed, to be prepended to generated
    /// text before re-encoding. Empty when nothing was healed. Freshly
    /// allocated (may be zero-length, in which case it is an empty slice).
    boundary_bytes: []const u8,

    pub fn deinit(self: *HealResult, allocator: std.mem.Allocator) void {
        allocator.free(self.healed_ids);
        if (self.boundary_bytes.len > 0) allocator.free(self.boundary_bytes);
        self.* = .{ .healed_ids = &.{}, .boundary_bytes = &.{} };
    }
};

/// Returns true if some vocab token's bytes are STRICTLY longer than
/// `candidate` and start with `candidate`. This is the conservatism gate:
/// we only heal a suffix that a genuinely-longer token could absorb.
fn isStrictPrefixOfLongerToken(bpe: *const Bpe, candidate: []const u8) bool {
    if (candidate.len == 0) return false;
    // A suffix longer than the longest vocab token can never be a strict
    // prefix of anything; bail before the scan.
    if (candidate.len >= bpe.max_piece_len) return false;

    var id: TokenId = 0;
    while (id < bpe.count) : (id += 1) {
        const tb = bpe.idBytes(id);
        if (tb.len > candidate.len and std.mem.startsWith(u8, tb, candidate)) {
            return true;
        }
    }
    return false;
}

/// Trim the trailing token(s) whose combined bytes are a strict prefix of a
/// longer vocab token, returning the kept ids plus the trimmed boundary
/// bytes. When nothing is healable the result is the input ids (copied)
/// with an empty boundary.
///
/// Caller owns the returned `HealResult` and must `deinit` it.
pub fn heal(allocator: std.mem.Allocator, bpe: *const Bpe, ids: []const TokenId) !HealResult {
    if (ids.len == 0) {
        return .{ .healed_ids = try allocator.dupe(TokenId, ids), .boundary_bytes = &.{} };
    }

    // Greedily search for the LONGEST trailing run (1..MAX_HEAL_TOKENS) whose
    // concatenated bytes are a strict prefix of a longer vocab token. We
    // assemble the candidate suffix incrementally from the back.
    const max_run = @min(MAX_HEAL_TOKENS, ids.len);

    var best_run: usize = 0; // number of trailing tokens to trim
    var suffix: std.ArrayList(u8) = .empty;
    defer suffix.deinit(allocator);

    var run: usize = 1;
    while (run <= max_run) : (run += 1) {
        // Rebuild the suffix bytes for the trailing `run` tokens, in order.
        suffix.clearRetainingCapacity();
        var k: usize = ids.len - run;
        while (k < ids.len) : (k += 1) {
            try suffix.appendSlice(allocator, bpe.idBytes(ids[k]));
        }
        if (isStrictPrefixOfLongerToken(bpe, suffix.items)) {
            best_run = run; // keep searching for a longer healable run
        }
    }

    if (best_run == 0) {
        // No-op: prompt ends on a natural boundary.
        return .{ .healed_ids = try allocator.dupe(TokenId, ids), .boundary_bytes = &.{} };
    }

    // Materialize the boundary bytes for the chosen run.
    var boundary: std.ArrayList(u8) = .empty;
    errdefer boundary.deinit(allocator);
    var j: usize = ids.len - best_run;
    while (j < ids.len) : (j += 1) {
        try boundary.appendSlice(allocator, bpe.idBytes(ids[j]));
    }

    const healed = try allocator.dupe(TokenId, ids[0 .. ids.len - best_run]);
    errdefer allocator.free(healed);

    return .{
        .healed_ids = healed,
        .boundary_bytes = try boundary.toOwnedSlice(allocator),
    };
}

/// Re-encode `boundary_bytes ++ new_text` as a single unit. This is how a
/// caller resumes generation after healing: the trimmed boundary bytes and
/// the freshly generated text are tokenized together so the join lands on a
/// natural token boundary.
///
/// Caller owns the returned id slice and must free it with `allocator`.
pub fn continueFrom(
    allocator: std.mem.Allocator,
    bpe: *const Bpe,
    boundary_bytes: []const u8,
    new_text: []const u8,
) ![]TokenId {
    const joined = try allocator.alloc(u8, boundary_bytes.len + new_text.len);
    defer allocator.free(joined);
    @memcpy(joined[0..boundary_bytes.len], boundary_bytes);
    @memcpy(joined[boundary_bytes.len..], new_text);

    // Worst case is one id per byte (no merges). Allocate accordingly, then
    // shrink to the actual encoded length.
    const scratch = try allocator.alloc(TokenId, @max(joined.len, 1));
    defer allocator.free(scratch);

    const encoded = bpe.encodeChunk(joined, scratch);
    return allocator.dupe(TokenId, encoded);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

const TestEntry = struct { bytes: []const u8, rank: u32 };

// Mirror of the bpe.zig test helper: build a tiktoken-format vocab source
// (base64 bytes + space + rank per line) from (bytes, rank) pairs.
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

// Decode a run of ids back to bytes by concatenating their token bytes.
fn decodeIds(allocator: std.mem.Allocator, bpe: *const Bpe, ids: []const TokenId) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    for (ids) |id| try buf.appendSlice(allocator, bpe.idBytes(id));
    return buf.toOwnedSlice(allocator);
}

test "heal trims trailing prefix token (hello wor -> world)" {
    // Vocab: single-byte tokens for the letters we need, plus "wor" and the
    // longer "world" so "wor" is a strict prefix of a longer token.
    const src = try buildVocabSource(testing.allocator, &.{
        .{ .bytes = "h", .rank = 0 },
        .{ .bytes = "e", .rank = 1 },
        .{ .bytes = "l", .rank = 2 },
        .{ .bytes = "o", .rank = 3 },
        .{ .bytes = " ", .rank = 4 },
        .{ .bytes = "w", .rank = 5 },
        .{ .bytes = "r", .rank = 6 },
        .{ .bytes = "d", .rank = 7 },
        .{ .bytes = "wor", .rank = 8 },
        .{ .bytes = "world", .rank = 9 },
    });
    defer testing.allocator.free(src);

    var bpe = try Bpe.loadTiktokenBytes(testing.allocator, src);
    defer bpe.deinit();
    // Whole-chunk lookup so a complete vocab token re-encodes to a single id
    // regardless of the synthetic vocab's (absent) merge chain.
    bpe.ignore_merges = true;

    // Prompt ids: h e l l o ' ' wor  -> trailing "wor" should heal.
    const wor: TokenId = bpe.by_bytes.get("wor").?;
    const sp: TokenId = bpe.by_bytes.get(" ").?;
    const ids = [_]TokenId{
        bpe.by_bytes.get("h").?, bpe.by_bytes.get("e").?, bpe.by_bytes.get("l").?,
        bpe.by_bytes.get("l").?, bpe.by_bytes.get("o").?, sp,
        wor,
    };

    var res = try heal(testing.allocator, &bpe, &ids);
    defer res.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 6), res.healed_ids.len);
    try testing.expectEqualStrings("wor", res.boundary_bytes);

    // continueFrom("wor", "ld") must yield the single "world" token.
    const cont = try continueFrom(testing.allocator, &bpe, res.boundary_bytes, "ld");
    defer testing.allocator.free(cont);
    try testing.expectEqual(@as(usize, 1), cont.len);
    try testing.expectEqual(bpe.by_bytes.get("world").?, cont[0]);
}

test "heal no-op when last token is not a prefix of any longer token" {
    const src = try buildVocabSource(testing.allocator, &.{
        .{ .bytes = "a", .rank = 0 },
        .{ .bytes = "b", .rank = 1 },
        .{ .bytes = "ab", .rank = 2 },
        // "z" is a standalone token; nothing longer starts with "z".
        .{ .bytes = "z", .rank = 3 },
    });
    defer testing.allocator.free(src);

    var bpe = try Bpe.loadTiktokenBytes(testing.allocator, src);
    defer bpe.deinit();

    const ids = [_]TokenId{ bpe.by_bytes.get("ab").?, bpe.by_bytes.get("z").? };
    var res = try heal(testing.allocator, &bpe, &ids);
    defer res.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 2), res.healed_ids.len);
    try testing.expectEqual(@as(usize, 0), res.boundary_bytes.len);
    try testing.expectEqualSlices(TokenId, &ids, res.healed_ids);
}

test "heal round-trip: decode(healed) ++ boundary == original bytes" {
    const src = try buildVocabSource(testing.allocator, &.{
        .{ .bytes = "h", .rank = 0 },
        .{ .bytes = "e", .rank = 1 },
        .{ .bytes = "l", .rank = 2 },
        .{ .bytes = "o", .rank = 3 },
        .{ .bytes = " ", .rank = 4 },
        .{ .bytes = "w", .rank = 5 },
        .{ .bytes = "r", .rank = 6 },
        .{ .bytes = "d", .rank = 7 },
        .{ .bytes = "wor", .rank = 8 },
        .{ .bytes = "world", .rank = 9 },
    });
    defer testing.allocator.free(src);

    var bpe = try Bpe.loadTiktokenBytes(testing.allocator, src);
    defer bpe.deinit();

    const ids = [_]TokenId{
        bpe.by_bytes.get("h").?, bpe.by_bytes.get("e").?, bpe.by_bytes.get("l").?,
        bpe.by_bytes.get("l").?, bpe.by_bytes.get("o").?, bpe.by_bytes.get(" ").?,
        bpe.by_bytes.get("wor").?,
    };

    const original = try decodeIds(testing.allocator, &bpe, &ids);
    defer testing.allocator.free(original);

    var res = try heal(testing.allocator, &bpe, &ids);
    defer res.deinit(testing.allocator);

    const healed_bytes = try decodeIds(testing.allocator, &bpe, res.healed_ids);
    defer testing.allocator.free(healed_bytes);

    const rejoined = try std.mem.concat(testing.allocator, u8, &.{ healed_bytes, res.boundary_bytes });
    defer testing.allocator.free(rejoined);

    try testing.expectEqualStrings(original, rejoined);
}

test "multi-token trailing heal folds two tokens into boundary" {
    // No "wor" token here, so "wo" + "r" must be healed across two trailing
    // tokens. " world" is the longer token that absorbs the suffix.
    const src = try buildVocabSource(testing.allocator, &.{
        .{ .bytes = "h", .rank = 0 },
        .{ .bytes = "e", .rank = 1 },
        .{ .bytes = "l", .rank = 2 },
        .{ .bytes = "o", .rank = 3 },
        .{ .bytes = " ", .rank = 4 },
        .{ .bytes = "w", .rank = 5 },
        .{ .bytes = "r", .rank = 6 },
        .{ .bytes = "d", .rank = 7 },
        .{ .bytes = "wo", .rank = 8 },
        .{ .bytes = "world", .rank = 9 },
    });
    defer testing.allocator.free(src);

    var bpe = try Bpe.loadTiktokenBytes(testing.allocator, src);
    defer bpe.deinit();
    bpe.ignore_merges = true;

    // ... ' ' "wo" "r"  -> trailing "wo"+"r" = "wor", strict prefix of "world".
    const ids = [_]TokenId{
        bpe.by_bytes.get(" ").?,
        bpe.by_bytes.get("wo").?,
        bpe.by_bytes.get("r").?,
    };

    var res = try heal(testing.allocator, &bpe, &ids);
    defer res.deinit(testing.allocator);

    // Both trailing tokens trimmed; only the space remains.
    try testing.expectEqual(@as(usize, 1), res.healed_ids.len);
    try testing.expectEqual(bpe.by_bytes.get(" ").?, res.healed_ids[0]);
    try testing.expectEqualStrings("wor", res.boundary_bytes);

    const cont = try continueFrom(testing.allocator, &bpe, res.boundary_bytes, "ld");
    defer testing.allocator.free(cont);
    try testing.expectEqual(@as(usize, 1), cont.len);
    try testing.expectEqual(bpe.by_bytes.get("world").?, cont[0]);
}

test "heal empty input is a no-op" {
    const src = try buildVocabSource(testing.allocator, &.{
        .{ .bytes = "a", .rank = 0 },
    });
    defer testing.allocator.free(src);
    var bpe = try Bpe.loadTiktokenBytes(testing.allocator, src);
    defer bpe.deinit();

    const ids = [_]TokenId{};
    var res = try heal(testing.allocator, &bpe, &ids);
    defer res.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 0), res.healed_ids.len);
    try testing.expectEqual(@as(usize, 0), res.boundary_bytes.len);
}
