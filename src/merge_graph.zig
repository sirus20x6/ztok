//! Ordered BPE merge relations plus an optional immutable lookup index.
//!
//! The relation slice is the source of truth and remains directly
//! inspectable for diagnostics, corpus analysis, and future n-gram/engram
//! data generation. The packed tables are only an acceleration overlay;
//! callers never need to reconstruct semantic relationships from them.

const std = @import("std");
const builtin = @import("builtin");
const TokenId = @import("token.zig").TokenId;

fn hintHugePages(comptime T: type, values: []T) void {
    if (builtin.os.tag != .linux or
        values.len < (2 * 1024 * 1024) / @sizeOf(T)) return;
    const bytes = std.mem.sliceAsBytes(values);
    const allocation_start = @intFromPtr(bytes.ptr);
    const start = std.mem.alignForward(usize, allocation_start, 4096);
    const end = std.mem.alignBackward(usize, allocation_start + bytes.len, 4096);
    if (end <= start) return;
    const ptr: [*]u8 = @ptrFromInt(start);
    _ = std.os.linux.madvise(ptr, end - start, std.os.linux.MADV.HUGEPAGE);
}

pub const Relation = struct {
    left: TokenId,
    right: TokenId,
    result: TokenId,
    /// Lower ranks merge first.
    rank: u32,
};

pub const MergeResult = struct {
    result: TokenId,
    rank: u32,
};

pub const BuildOptions = struct {
    /// A 2^N by 2^N direct grid for common low token IDs. Eleven matches
    /// the initial-byte and early-merge working set used by GPT-family BPE.
    dense_log2: u5 = 11,
    build_accelerator: bool = true,
};

pub const Graph = struct {
    allocator: std.mem.Allocator,
    ordered_relations: []Relation,
    /// Relation index for `(left,right)`, maxInt(u32) when absent.
    dense: []u32 = &.{},
    dense_log2: u5 = 0,
    /// Packed `(pair_key << 21) | relation_index`; maxInt(u64) is empty.
    slots: []u64 = &.{},
    slot_mask: usize = 0,
    hash_shift: u6 = 0,
    /// Common GPT/tiktoken layout: relation index == rank and
    /// result_id == rank + constant. This lets the encoding hot path derive
    /// both values directly from the indexed relation number without loading
    /// the larger inspectable relation record.
    affine_result_bias: ?u32 = null,

    const id_bits: u6 = 21;
    const id_limit: u32 = 1 << id_bits;
    const relation_mask: u64 = (1 << id_bits) - 1;
    const empty_slot = std.math.maxInt(u64);
    const no_relation = std.math.maxInt(u32);

    pub fn init(
        allocator: std.mem.Allocator,
        source_relations: []const Relation,
        options: BuildOptions,
    ) !Graph {
        const owned = try allocator.dupe(Relation, source_relations);
        errdefer allocator.free(owned);

        var graph: Graph = .{
            .allocator = allocator,
            .ordered_relations = owned,
        };
        errdefer graph.deinit();

        if (source_relations.len != 0) {
            const first = source_relations[0];
            if (first.result >= first.rank) {
                const bias = first.result - first.rank;
                var affine = true;
                for (source_relations, 0..) |rel, index| {
                    const rank_matches_index = rel.rank == @as(u32, @intCast(index));
                    const result_matches_rank = @as(u64, rel.result) ==
                        @as(u64, rel.rank) + @as(u64, bias);
                    if (!rank_matches_index or !result_matches_rank) {
                        affine = false;
                        break;
                    }
                }
                if (affine) graph.affine_result_bias = bias;
            }
        }

        if (!options.build_accelerator or source_relations.len == 0 or
            source_relations.len >= id_limit)
        {
            return graph;
        }
        for (source_relations) |rel| {
            if (rel.left >= id_limit or rel.right >= id_limit or rel.result >= id_limit) {
                return graph;
            }
        }

        const log2 = @min(options.dense_log2, id_bits);
        if (log2 != 0) {
            const dense_len = @as(usize, 1) << @as(u6, @intCast(log2 * 2));
            graph.dense = try allocator.alloc(u32, dense_len);
            hintHugePages(u32, graph.dense);
            @memset(graph.dense, no_relation);
            graph.dense_log2 = log2;
        }

        const wanted_slots = @max(@as(usize, 64), source_relations.len * 2);
        const slot_count = try std.math.ceilPowerOfTwo(usize, wanted_slots);
        graph.slots = try allocator.alloc(u64, slot_count);
        @memset(graph.slots, empty_slot);
        graph.slot_mask = slot_count - 1;
        const slot_log2: u6 = @intCast(@ctz(slot_count));
        graph.hash_shift = @intCast(@as(usize, 64) - @as(usize, slot_log2));

        for (source_relations, 0..) |rel, relation_index_usize| {
            const relation_index: u32 = @intCast(relation_index_usize);
            if (graph.dense.len != 0 and
                ((rel.left | rel.right) >> graph.dense_log2) == 0)
            {
                const idx = (@as(usize, rel.left) << graph.dense_log2) |
                    @as(usize, rel.right);
                // Duplicate pairs retain the earliest ordered merge.
                if (graph.dense[idx] == no_relation) graph.dense[idx] = relation_index;
            }

            const key = pairKey(rel.left, rel.right);
            var idx = graph.slotIndex(key);
            while (true) : (idx = (idx + 1) & graph.slot_mask) {
                const slot = graph.slots[idx];
                if (slot == empty_slot) {
                    graph.slots[idx] = (key << id_bits) | relation_index;
                    break;
                }
                if ((slot >> id_bits) == key) break;
            }
        }
        return graph;
    }

    pub fn deinit(self: *Graph) void {
        if (self.ordered_relations.len != 0) self.allocator.free(self.ordered_relations);
        if (self.dense.len != 0) self.allocator.free(self.dense);
        if (self.slots.len != 0) self.allocator.free(self.slots);
        self.* = .{ .allocator = self.allocator, .ordered_relations = &.{} };
    }

    pub fn relations(self: *const Graph) []const Relation {
        return self.ordered_relations;
    }

    pub fn hasAccelerator(self: *const Graph) bool {
        return self.slots.len != 0;
    }

    pub inline fn lookup(self: *const Graph, left: TokenId, right: TokenId) ?Relation {
        const relation_index = self.lookupRelationIndex(left, right) orelse {
            // Oversized/custom vocabularies retain inspectability and
            // correctness even when IDs cannot use the packed accelerator.
            if (self.slots.len == 0) {
                for (self.ordered_relations) |rel| {
                    if (rel.left == left and rel.right == right) return rel;
                }
            }
            return null;
        };
        return self.ordered_relations[relation_index];
    }

    /// Minimal lookup result for the merge hot path. In affine GPT-family
    /// graphs this avoids a second random load from `ordered_relations`.
    pub inline fn lookupMerge(self: *const Graph, left: TokenId, right: TokenId) ?MergeResult {
        if (self.lookupRelationIndex(left, right)) |relation_index| {
            if (self.affine_result_bias) |bias| return .{
                .result = relation_index + bias,
                .rank = relation_index,
            };
            const rel = self.ordered_relations[relation_index];
            return .{ .result = rel.result, .rank = rel.rank };
        }
        if (self.slots.len == 0) {
            for (self.ordered_relations) |rel| {
                if (rel.left == left and rel.right == right) return .{
                    .result = rel.result,
                    .rank = rel.rank,
                };
            }
        }
        return null;
    }

    /// Stage the accelerator location for a merge pair before its lookup.
    /// Callers use this only when independent list/heap work can cover the
    /// random table latency; it has no semantic effect.
    pub inline fn prefetchMerge(self: *const Graph, left: TokenId, right: TokenId) void {
        if (self.dense.len != 0 and ((left | right) >> self.dense_log2) == 0) {
            const idx = (@as(usize, left) << self.dense_log2) | @as(usize, right);
            @prefetch(&self.dense[idx], .{ .rw = .read, .locality = 3, .cache = .data });
            return;
        }
        if (self.slots.len != 0 and left < id_limit and right < id_limit) {
            const key = pairKey(left, right);
            @prefetch(&self.slots[self.slotIndex(key)], .{
                .rw = .read,
                .locality = 3,
                .cache = .data,
            });
        }
    }

    inline fn lookupRelationIndex(self: *const Graph, left: TokenId, right: TokenId) ?u32 {
        if (self.dense.len != 0 and ((left | right) >> self.dense_log2) == 0) {
            const idx = (@as(usize, left) << self.dense_log2) | @as(usize, right);
            const relation_index = self.dense[idx];
            if (relation_index != no_relation) return relation_index;
            return null;
        }
        if (self.slots.len != 0 and left < id_limit and right < id_limit) {
            const key = pairKey(left, right);
            var idx = self.slotIndex(key);
            while (true) : (idx = (idx + 1) & self.slot_mask) {
                const slot = self.slots[idx];
                if (slot == empty_slot) return null;
                if ((slot >> id_bits) == key) {
                    const relation_index: u32 = @truncate(slot & relation_mask);
                    return relation_index;
                }
            }
        }
        return null;
    }

    inline fn pairKey(left: TokenId, right: TokenId) u64 {
        return (@as(u64, left) << id_bits) | @as(u64, right);
    }

    inline fn slotIndex(self: *const Graph, key: u64) usize {
        return @intCast((key *% 0x9E37_79B9_7F4A_7C15) >> self.hash_shift);
    }
};

test "merge graph preserves order and accelerates exact pair lookup" {
    const relations = [_]Relation{
        .{ .left = 1, .right = 2, .result = 10, .rank = 0 },
        .{ .left = 10, .right = 3, .result = 11, .rank = 1 },
        .{ .left = 3000, .right = 4, .result = 12, .rank = 2 },
    };
    var graph = try Graph.init(std.testing.allocator, &relations, .{});
    defer graph.deinit();

    try std.testing.expect(graph.hasAccelerator());
    try std.testing.expectEqualSlices(Relation, &relations, graph.relations());
    try std.testing.expectEqual(relations[0], graph.lookup(1, 2).?);
    try std.testing.expectEqual(relations[2], graph.lookup(3000, 4).?);
    try std.testing.expectEqual(MergeResult{ .result = 11, .rank = 1 }, graph.lookupMerge(10, 3).?);
    try std.testing.expect(graph.lookup(2, 1) == null);
}

test "merge graph keeps first duplicate pair" {
    const relations = [_]Relation{
        .{ .left = 1, .right = 2, .result = 10, .rank = 0 },
        .{ .left = 1, .right = 2, .result = 11, .rank = 4 },
    };
    var graph = try Graph.init(std.testing.allocator, &relations, .{});
    defer graph.deinit();
    try std.testing.expectEqual(@as(TokenId, 10), graph.lookup(1, 2).?.result);
}
