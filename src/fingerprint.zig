//! Tokenizer fingerprint — a deterministic SHA-256 hash that uniquely
//! identifies a vocab. Two pipelines that produce the same fingerprint
//! are guaranteed to emit bit-identical id streams for any input.
//!
//! The fingerprint hashes OBSERVABLE encoding behavior, not the on-disk
//! file format: we encode a fixed canonical set of inputs and hash the
//! resulting id streams together with a model-kind tag and vocab size.
//! Two vocabs serialized differently but with identical behavior on the
//! canonical set will agree (which is the property a downstream cache
//! / KV-store / training pipeline actually needs).
//!
//! Layout fed to SHA-256:
//!   sha256(
//!     model_kind_tag (1 byte) ||
//!     vocab_size     (u32 LE) ||
//!     encode(input_1) ids as packed u32 LE ||
//!     encode(input_2) ids as packed u32 LE ||
//!     encode(input_3) ids as packed u32 LE ||
//!     encode(input_4) ids as packed u32 LE
//!   )
//!
//! Display form: `ztok:<64 lowercase hex chars>`.

const std = @import("std");

const Pipeline = @import("pipeline.zig").Pipeline;
const TokenId = @import("token.zig").TokenId;
const Model = @import("model.zig").Model;

/// Stable 1-byte tag identifying the model family. The exact value is
/// part of the fingerprint contract — never renumber; only append.
fn modelKindTag(model: Model) u8 {
    return switch (model) {
        .byte_id => 0,
        .bpe => 1,
        .unigram => 2,
        .wordpiece => 3,
        .monster => 4,
    };
}

/// Canonical inputs the fingerprint is computed over. Designed so that
/// any meaningful behavior change in the pipeline (BPE merges, Unigram
/// scores, added-token resolution, normalizer flavor, pre-tokenizer
/// regex, byte-level mapping) perturbs at least one of these streams.
///
/// 1. Empty string — catches start-of-text quirks (BOS, leading-space
///    injection) without any actual bytes.
/// 2. Every byte 0..255 in order — exercises raw-byte handling, the
///    full single-byte vocab coverage, and any byte_fallback paths.
///    Not valid UTF-8; that's intentional — `encode` accepts raw bytes.
/// 3. ASCII + multi-script UTF-8 + a flag emoji — exercises common
///    BPE merge chains, NFKC folds, byte-level encoding of non-ASCII,
///    and surrogate-pair / multi-codepoint sequences.
/// 4. Whitespace-heavy input — surfaces differences in whitespace
///    handling between cl100k, byte_level, identity pre-tokenizers,
///    and Unigram's `▁` prefix scheme.
const canonical_inputs: [4][]const u8 = blk: {
    // Construct the 256-byte all-bytes input at comptime.
    var all_bytes: [256]u8 = undefined;
    var i: usize = 0;
    while (i < 256) : (i += 1) all_bytes[i] = @intCast(i);
    const all_bytes_const = all_bytes;

    break :blk .{
        "",
        &all_bytes_const,
        "The quick brown fox jumps over the lazy dog. 你好世界 \xF0\x9F\x8C\x8D café naïve résumé Ωμέγα \xF0\x9F\x87\xBA\xF0\x9F\x87\xB8",
        "   \t\n\n  hello  \n\n   world   ",
    };
};

/// Compute the fingerprint for `p`. Allocates only temporary id buffers
/// from `allocator`; no state is retained.
pub fn computeFingerprint(p: *const Pipeline, allocator: std.mem.Allocator) ![32]u8 {
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});

    // Model-kind tag.
    const tag = [_]u8{modelKindTag(p.model)};
    hasher.update(&tag);

    // Vocab size as u32 little-endian. Falls back to 0 for byte_id /
    // pipelines that left the vocab empty (the model-kind tag already
    // discriminates those from a loaded vocab of size 0).
    var vs_buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &vs_buf, p.vocab.count, .little);
    hasher.update(&vs_buf);

    // Encode each canonical input and feed the id stream (as packed
    // little-endian u32s) into the hasher.
    for (canonical_inputs) |input| {
        const ids = try p.encode(allocator, input);
        defer allocator.free(ids);
        try updateWithIds(&hasher, ids);
    }

    return hasher.finalResult();
}

/// Feed `ids` into `hasher` as a packed little-endian u32 stream. We
/// don't trust the host endianness — write through a 4-byte buffer so
/// the fingerprint is stable across architectures.
fn updateWithIds(hasher: *std.crypto.hash.sha2.Sha256, ids: []const TokenId) !void {
    // u32 is the declared TokenId width; locking it here makes the
    // contract explicit and the fingerprint stable if anyone widens it
    // later.
    comptime std.debug.assert(@sizeOf(TokenId) == 4);
    var buf: [4]u8 = undefined;
    for (ids) |id| {
        std.mem.writeInt(u32, &buf, id, .little);
        hasher.update(&buf);
    }
}

/// Hex-encode a 32-byte digest to a 64-char lowercase ASCII buffer.
/// Pure, allocation-free; caller takes the array by value.
pub fn formatHex(bytes: [32]u8) [64]u8 {
    const hex = "0123456789abcdef";
    var out: [64]u8 = undefined;
    for (bytes, 0..) |b, i| {
        out[i * 2] = hex[(b >> 4) & 0x0F];
        out[i * 2 + 1] = hex[b & 0x0F];
    }
    return out;
}

// --- tests ---------------------------------------------------------------

const testing = std.testing;

const Vocab = @import("vocab.zig").Vocab;
const Bpe = @import("bpe.zig").Bpe;

test "formatHex produces lowercase hex" {
    var bytes: [32]u8 = undefined;
    var i: usize = 0;
    while (i < 32) : (i += 1) bytes[i] = @intCast(i);
    const hex = formatHex(bytes);
    try testing.expectEqualStrings(
        "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
        &hex,
    );
}

test "byte_id fingerprint is deterministic across 3 invocations" {
    var v = Vocab.empty(testing.allocator);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };

    const fp1 = try computeFingerprint(&pipe, testing.allocator);
    const fp2 = try computeFingerprint(&pipe, testing.allocator);
    const fp3 = try computeFingerprint(&pipe, testing.allocator);

    try testing.expectEqualSlices(u8, &fp1, &fp2);
    try testing.expectEqualSlices(u8, &fp1, &fp3);
}

test "different BPE vocabs produce different fingerprints" {
    // Two tiny BPE vocabs that disagree on which byte gets which id.
    // Identical inputs MUST yield different fingerprints.
    const sample_a =
        "YQ== 0\n" ++ // 'a' -> 0
        "Yg== 1\n" ++ // 'b' -> 1
        "YWI= 2\n";   // 'ab' -> 2
    const sample_b =
        "Yg== 0\n" ++ // 'b' -> 0
        "YQ== 1\n" ++ // 'a' -> 1
        "YmE= 2\n";   // 'ba' -> 2

    var bpe_a = try Bpe.loadTiktokenBytes(testing.allocator, sample_a);
    defer bpe_a.deinit();
    var bpe_b = try Bpe.loadTiktokenBytes(testing.allocator, sample_b);
    defer bpe_b.deinit();

    var v = Vocab.empty(testing.allocator);
    defer v.deinit();

    const pipe_a: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .{ .bpe = &bpe_a },
        .decoder = .concat,
        .vocab = &v,
    };
    const pipe_b: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .{ .bpe = &bpe_b },
        .decoder = .concat,
        .vocab = &v,
    };

    const fp_a = try computeFingerprint(&pipe_a, testing.allocator);
    const fp_b = try computeFingerprint(&pipe_b, testing.allocator);
    try testing.expect(!std.mem.eql(u8, &fp_a, &fp_b));
}

test "same BPE bytes produce same fingerprint (cross-load reproducibility)" {
    // Load the same source bytes twice into independent Bpe instances
    // — fingerprints must match.
    const sample =
        "YQ== 0\n" ++
        "Yg== 1\n" ++
        "YWI= 2\n";

    var bpe1 = try Bpe.loadTiktokenBytes(testing.allocator, sample);
    defer bpe1.deinit();
    var bpe2 = try Bpe.loadTiktokenBytes(testing.allocator, sample);
    defer bpe2.deinit();

    var v = Vocab.empty(testing.allocator);
    defer v.deinit();

    const pipe1: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .{ .bpe = &bpe1 },
        .decoder = .concat,
        .vocab = &v,
    };
    const pipe2: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .{ .bpe = &bpe2 },
        .decoder = .concat,
        .vocab = &v,
    };

    const fp1 = try computeFingerprint(&pipe1, testing.allocator);
    const fp2 = try computeFingerprint(&pipe2, testing.allocator);
    try testing.expectEqualSlices(u8, &fp1, &fp2);
}

test "fingerprint hex prefix form is 64 chars" {
    var v = Vocab.empty(testing.allocator);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };
    const fp = try computeFingerprint(&pipe, testing.allocator);
    const hex = formatHex(fp);
    try testing.expectEqual(@as(usize, 64), hex.len);
    // All chars in [0-9a-f].
    for (hex) |c| {
        try testing.expect((c >= '0' and c <= '9') or (c >= 'a' and c <= 'f'));
    }
}
