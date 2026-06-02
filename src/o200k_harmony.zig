//! Loader for the o200k_harmony (GPT-OSS) tokenizer.
//!
//! o200k_harmony is OpenAI's o200k_base BPE (199998 ranks) plus 1091 Harmony
//! special tokens (`<|start|>`, `<|message|>`, `<|end|>`, `<|channel|>`,
//! `<|return|>`, `<|call|>`, …) occupying ids 199998..201087, for a total
//! `n_vocab` of 201088.
//!
//! This module wires three fixtures into a ready-to-encode `Pipeline`:
//!   * `o200k_harmony.tiktoken`   — base64(token) SPACE rank, one per line.
//!   * `o200k_harmony.meta.json`  — { pat_str, special_tokens{}, n_vocab }.
//!
//! Encoding path:  added-token Scanner (splits specials out atomically)
//!                 → o200k pre-tokenizer (case-aware splitter, see o200k.zig)
//!                 → o200k_base BPE.
//!
//! The specials are registered as plain atomic added tokens (no lstrip /
//! rstrip / single_word), matching tiktoken's behavior when every special is
//! permitted: each `<|...|>` literal becomes exactly its assigned id and the
//! text between specials is pre-tokenized + BPE'd normally.

const std = @import("std");
const Bpe = @import("bpe.zig").Bpe;
const Vocab = @import("vocab.zig").Vocab;
const Pipeline = @import("pipeline.zig").Pipeline;
const added_tokens = @import("added_tokens.zig");
const TokenId = @import("token.zig").TokenId;

/// Owns every allocation backing an o200k_harmony tokenizer. Build with
/// `load` / `loadFromDir`, encode via `pipeline()`, free with `deinit`.
pub const Harmony = struct {
    allocator: std.mem.Allocator,
    bpe: Bpe,
    scanner: added_tokens.Scanner,
    vocab: Vocab,
    n_vocab: u32,

    /// Build a `Pipeline` view over this tokenizer. The returned pipeline
    /// borrows `self`; it must not outlive the `Harmony`.
    pub fn pipeline(self: *const Harmony) Pipeline {
        return .{
            .normalizer = .identity,
            .pre_tokenizer = .o200k,
            .model = .{ .bpe = &self.bpe },
            .decoder = .concat,
            .vocab = &self.vocab,
            .added_tokens = &self.scanner,
        };
    }

    pub fn deinit(self: *Harmony) void {
        self.bpe.deinit();
        self.scanner.deinit();
        self.vocab.deinit();
        self.* = undefined;
    }
};

/// Load from explicit byte buffers (ranks file + meta.json). Caller still
/// owns `tiktoken_bytes` and `meta_json` after return.
pub fn load(
    allocator: std.mem.Allocator,
    tiktoken_bytes: []const u8,
    meta_json: []const u8,
) !Harmony {
    var bpe = try Bpe.loadTiktokenBytes(allocator, tiktoken_bytes);
    errdefer bpe.deinit();

    // Parse meta.json and collect the special tokens into AddedToken[].
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, meta_json, .{});
    defer parsed.deinit();

    if (parsed.value != .object) return error.InvalidMetaJson;
    const root = parsed.value.object;

    const n_vocab: u32 = blk: {
        const v = root.get("n_vocab") orelse break :blk 0;
        break :blk switch (v) {
            .integer => |iv| @intCast(iv),
            else => 0,
        };
    };

    const specials_val = root.get("special_tokens") orelse return error.MissingSpecialTokens;
    if (specials_val != .object) return error.MissingSpecialTokens;
    const specials = specials_val.object;

    var toks: std.ArrayList(added_tokens.AddedToken) = .empty;
    defer toks.deinit(allocator);
    try toks.ensureTotalCapacity(allocator, specials.count());

    var it = specials.iterator();
    while (it.next()) |entry| {
        const content = entry.key_ptr.*;
        const id_val = entry.value_ptr.*;
        const id: TokenId = switch (id_val) {
            .integer => |iv| @intCast(iv),
            else => return error.InvalidSpecialId,
        };
        // Plain atomic specials: match anywhere, no whitespace absorption.
        toks.appendAssumeCapacity(.{ .id = id, .content = content });
    }

    // Scanner copies the content bytes into its own arena, so it's safe for
    // `parsed` to be freed after init.
    var scanner = try added_tokens.Scanner.init(allocator, toks.items);
    errdefer scanner.deinit();

    return .{
        .allocator = allocator,
        .bpe = bpe,
        .scanner = scanner,
        .vocab = Vocab.empty(allocator),
        .n_vocab = n_vocab,
    };
}

/// Load the three fixtures from a directory (default `bench/vocabs`). Returns
/// `error.FileNotFound` if a fixture is absent — callers that want a
/// skip-when-absent test should catch that.
pub fn loadFromDir(allocator: std.mem.Allocator, dir_path: []const u8) !Harmony {
    const io = std.Io.Threaded.global_single_threaded.io();
    const cwd = std.Io.Dir.cwd();

    const tik_path = try std.fs.path.join(allocator, &.{ dir_path, "o200k_harmony.tiktoken" });
    defer allocator.free(tik_path);
    const meta_path = try std.fs.path.join(allocator, &.{ dir_path, "o200k_harmony.meta.json" });
    defer allocator.free(meta_path);

    const tik_bytes = try cwd.readFileAlloc(io, tik_path, allocator, .unlimited);
    defer allocator.free(tik_bytes);
    const meta_bytes = try cwd.readFileAlloc(io, meta_path, allocator, .unlimited);
    defer allocator.free(meta_bytes);

    return load(allocator, tik_bytes, meta_bytes);
}

// --- golden test -------------------------------------------------------------

const golden = struct {
    text: []const u8,
    ids: []const TokenId,
};

// Bit-exact tiktoken golden ids for o200k_harmony.
const goldens = [_]golden{
    .{ .text = "Hello, world!", .ids = &.{ 13225, 11, 2375, 0 } },
    .{ .text = " the quick brown fox", .ids = &.{ 290, 4853, 19705, 68347 } },
    .{ .text = "def foo(x):\n    return x+1", .ids = &.{ 1314, 30551, 4061, 1883, 271, 622, 1215, 10, 16 } },
    .{ .text = "<|start|>user<|message|>hi<|end|>", .ids = &.{ 200006, 1428, 200008, 3686, 200007 } },
    .{ .text = "  123456789", .ids = &.{ 220, 220, 7633, 19354, 29338 } },
    .{ .text = "na\u{00ef}ve caf\u{00e9} ", .ids = &.{ 1503, 9954, 737, 30469, 220 } },
};

test "o200k_harmony golden ids (skip when fixtures absent)" {
    const allocator = std.testing.allocator;

    // Skip-when-absent: probe one fixture; if missing, the whole golden test
    // is a no-op so CI without the (untracked) vocab stays green.
    var harmony = loadFromDir(allocator, "bench/vocabs") catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer harmony.deinit();

    const pipe = harmony.pipeline();

    for (goldens) |g| {
        const ids = try pipe.encode(allocator, g.text);
        defer allocator.free(ids);
        std.testing.expectEqualSlices(TokenId, g.ids, ids) catch |err| {
            std.debug.print("o200k_harmony golden FAILED for {s:.40}\n  want: {any}\n  got:  {any}\n", .{ g.text, g.ids, ids });
            return err;
        };
    }
}
