//! Struct-of-arrays vocabulary table.
//!
//! Two flat byte buffers and an index — no per-token allocations, no
//! pointer chasing. Lookup is O(1) by id; reverse lookup (bytes → id) goes
//! through a separate hash map built on top of the same arena.

const std = @import("std");
const TokenId = @import("token.zig").TokenId;

pub const Vocab = struct {
    allocator: std.mem.Allocator,

    /// Concatenated UTF-8 bytes of every token, back to back.
    bytes: []u8,
    /// Offsets into `bytes`. `offsets[i]..offsets[i+1]` is token `i`.
    /// Length is `count + 1` so the final sentinel works.
    offsets: []u32,
    /// Optional BPE merge rank, parallel to id. `null` for non-BPE models.
    ranks: ?[]u32,

    count: u32,

    pub fn empty(allocator: std.mem.Allocator) Vocab {
        return .{
            .allocator = allocator,
            .bytes = &.{},
            .offsets = &.{},
            .ranks = null,
            .count = 0,
        };
    }

    pub fn deinit(self: *Vocab) void {
        if (self.bytes.len > 0) self.allocator.free(self.bytes);
        if (self.offsets.len > 0) self.allocator.free(self.offsets);
        if (self.ranks) |r| self.allocator.free(r);
        self.* = .empty(self.allocator);
    }

    pub fn tokenBytes(self: *const Vocab, id: TokenId) []const u8 {
        std.debug.assert(id < self.count);
        const start = self.offsets[id];
        const end = self.offsets[id + 1];
        return self.bytes[start..end];
    }
};

test "empty vocab" {
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();
    try std.testing.expectEqual(@as(u32, 0), v.count);
}
