//! Minimal HuggingFace tokenizer_config.json loader.
//!
//! Sibling file to tokenizer.json in HF model repos. Parses the four
//! most-used special-token fields (bos/eos/pad/unk), each of which can
//! appear as a plain string OR a full AddedToken object, plus the
//! add_bos_token / add_eos_token bool flags and model_max_length.
//!
//! Everything else (chat_template, padding_side, tokenizer_class, ...)
//! is intentionally ignored — a later pass will layer those on.
//!
//! Integration note: the encoder is wired up by pairing this with
//! src/added_tokens.zig — if add_bos_token, prepend bos_token id; if
//! add_eos_token, append eos_token id. That lives in Pipeline.encode
//! and is out of scope here.

const std = @import("std");

pub const SpecialToken = struct {
    content: []u8,
    lstrip: bool = false,
    rstrip: bool = false,
    single_word: bool = false,
    normalized: bool = true,
};

pub const TokenizerConfig = struct {
    allocator: std.mem.Allocator,
    bos_token: ?SpecialToken = null,
    eos_token: ?SpecialToken = null,
    pad_token: ?SpecialToken = null,
    unk_token: ?SpecialToken = null,
    add_bos_token: bool = false,
    add_eos_token: bool = false,
    model_max_length: ?u64 = null,

    pub fn deinit(self: *TokenizerConfig) void {
        if (self.bos_token) |*t| self.allocator.free(t.content);
        if (self.eos_token) |*t| self.allocator.free(t.content);
        if (self.pad_token) |*t| self.allocator.free(t.content);
        if (self.unk_token) |*t| self.allocator.free(t.content);
    }
};

pub const Error = error{MalformedJson} || std.mem.Allocator.Error || std.json.ParseError(std.json.Scanner);

pub fn loadFromBytes(allocator: std.mem.Allocator, json_bytes: []const u8) Error!TokenizerConfig {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_bytes, .{});
    defer parsed.deinit();

    const root = parsed.value;
    if (root != .object) return error.MalformedJson;

    var cfg: TokenizerConfig = .{ .allocator = allocator };
    errdefer cfg.deinit();

    if (root.object.get("bos_token")) |v| cfg.bos_token = try parseSpecialToken(allocator, v);
    if (root.object.get("eos_token")) |v| cfg.eos_token = try parseSpecialToken(allocator, v);
    if (root.object.get("pad_token")) |v| cfg.pad_token = try parseSpecialToken(allocator, v);
    if (root.object.get("unk_token")) |v| cfg.unk_token = try parseSpecialToken(allocator, v);

    if (root.object.get("add_bos_token")) |v| if (v == .bool) {
        cfg.add_bos_token = v.bool;
    };
    if (root.object.get("add_eos_token")) |v| if (v == .bool) {
        cfg.add_eos_token = v.bool;
    };
    if (root.object.get("model_max_length")) |v| switch (v) {
        .integer => |i| if (i >= 0) {
            cfg.model_max_length = @intCast(i);
        },
        .float => |f| if (f >= 0 and std.math.isFinite(f)) {
            cfg.model_max_length = @intFromFloat(f);
        },
        else => {},
    };

    return cfg;
}

pub fn loadFromFile(allocator: std.mem.Allocator, path: []const u8) !TokenizerConfig {
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

fn parseSpecialToken(allocator: std.mem.Allocator, v: std.json.Value) Error!?SpecialToken {
    switch (v) {
        .null => return null,
        .string => |s| return .{ .content = try allocator.dupe(u8, s) },
        .object => |obj| {
            const content_v = obj.get("content") orelse return error.MalformedJson;
            if (content_v != .string) return error.MalformedJson;
            var t: SpecialToken = .{ .content = try allocator.dupe(u8, content_v.string) };
            if (obj.get("lstrip")) |b| if (b == .bool) {
                t.lstrip = b.bool;
            };
            if (obj.get("rstrip")) |b| if (b == .bool) {
                t.rstrip = b.bool;
            };
            if (obj.get("single_word")) |b| if (b == .bool) {
                t.single_word = b.bool;
            };
            if (obj.get("normalized")) |b| if (b == .bool) {
                t.normalized = b.bool;
            };
            return t;
        },
        else => return error.MalformedJson,
    }
}

// ---------------------------------------------------------------------
// Tests

const testing = std.testing;

test "string form" {
    const input =
        \\{"bos_token": "<s>", "eos_token": "</s>"}
    ;
    var cfg = try loadFromBytes(testing.allocator, input);
    defer cfg.deinit();

    try testing.expect(cfg.bos_token != null);
    try testing.expectEqualStrings("<s>", cfg.bos_token.?.content);
    try testing.expect(!cfg.bos_token.?.lstrip);
    try testing.expect(!cfg.bos_token.?.rstrip);
    try testing.expect(!cfg.bos_token.?.single_word);
    try testing.expect(cfg.bos_token.?.normalized);
    try testing.expect(cfg.eos_token != null);
    try testing.expectEqualStrings("</s>", cfg.eos_token.?.content);
    try testing.expectEqual(@as(?SpecialToken, null), cfg.pad_token);
    try testing.expectEqual(@as(?SpecialToken, null), cfg.unk_token);
}

test "object form" {
    const input =
        \\{
        \\  "bos_token": {
        \\    "content": "<s>",
        \\    "lstrip": true,
        \\    "rstrip": false,
        \\    "single_word": false,
        \\    "normalized": true
        \\  }
        \\}
    ;
    var cfg = try loadFromBytes(testing.allocator, input);
    defer cfg.deinit();

    try testing.expect(cfg.bos_token != null);
    try testing.expectEqualStrings("<s>", cfg.bos_token.?.content);
    try testing.expect(cfg.bos_token.?.lstrip);
    try testing.expect(!cfg.bos_token.?.rstrip);
    try testing.expect(!cfg.bos_token.?.single_word);
    try testing.expect(cfg.bos_token.?.normalized);
}

test "mixed string + object" {
    const input =
        \\{
        \\  "bos_token": "<s>",
        \\  "eos_token": {"content": "</s>", "lstrip": false, "rstrip": true, "single_word": false, "normalized": false}
        \\}
    ;
    var cfg = try loadFromBytes(testing.allocator, input);
    defer cfg.deinit();

    try testing.expect(cfg.bos_token != null);
    try testing.expectEqualStrings("<s>", cfg.bos_token.?.content);
    try testing.expect(!cfg.bos_token.?.rstrip);
    try testing.expect(cfg.bos_token.?.normalized);

    try testing.expect(cfg.eos_token != null);
    try testing.expectEqualStrings("</s>", cfg.eos_token.?.content);
    try testing.expect(cfg.eos_token.?.rstrip);
    try testing.expect(!cfg.eos_token.?.normalized);
}

test "missing fields default to null" {
    const input = "{}";
    var cfg = try loadFromBytes(testing.allocator, input);
    defer cfg.deinit();

    try testing.expectEqual(@as(?SpecialToken, null), cfg.bos_token);
    try testing.expectEqual(@as(?SpecialToken, null), cfg.eos_token);
    try testing.expectEqual(@as(?SpecialToken, null), cfg.pad_token);
    try testing.expectEqual(@as(?SpecialToken, null), cfg.unk_token);
    try testing.expect(!cfg.add_bos_token);
    try testing.expect(!cfg.add_eos_token);
    try testing.expectEqual(@as(?u64, null), cfg.model_max_length);
}

test "add_bos_token / add_eos_token bools" {
    const input =
        \\{"add_bos_token": true, "add_eos_token": false}
    ;
    var cfg = try loadFromBytes(testing.allocator, input);
    defer cfg.deinit();

    try testing.expect(cfg.add_bos_token);
    try testing.expect(!cfg.add_eos_token);
}

test "model_max_length integer" {
    const input =
        \\{"model_max_length": 4096}
    ;
    var cfg = try loadFromBytes(testing.allocator, input);
    defer cfg.deinit();

    try testing.expectEqual(@as(?u64, 4096), cfg.model_max_length);
}

test "unrecognized fields are ignored" {
    const input =
        \\{
        \\  "bos_token": "<s>",
        \\  "padding_side": "right",
        \\  "chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}",
        \\  "tokenizer_class": "LlamaTokenizer",
        \\  "clean_up_tokenization_spaces": false
        \\}
    ;
    var cfg = try loadFromBytes(testing.allocator, input);
    defer cfg.deinit();

    try testing.expect(cfg.bos_token != null);
    try testing.expectEqualStrings("<s>", cfg.bos_token.?.content);
}

test "null tokens stay null" {
    const input =
        \\{"bos_token": null, "eos_token": null, "pad_token": null, "unk_token": null}
    ;
    var cfg = try loadFromBytes(testing.allocator, input);
    defer cfg.deinit();

    try testing.expectEqual(@as(?SpecialToken, null), cfg.bos_token);
    try testing.expectEqual(@as(?SpecialToken, null), cfg.eos_token);
    try testing.expectEqual(@as(?SpecialToken, null), cfg.pad_token);
    try testing.expectEqual(@as(?SpecialToken, null), cfg.unk_token);
}
