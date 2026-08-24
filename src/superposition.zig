//! Experimental token-superposition plans.
//!
//! This module is deliberately an overlay over ordinary tokenization. It
//! never creates token ids, changes a vocabulary, or participates in the
//! normal encode/decode hot paths. Callers opt in by passing an already
//! encoded id/offset stream to `buildFixedPlan`, or by using the Pipeline
//! convenience method that delegates here.

const std = @import("std");
const token = @import("token.zig");

pub const TokenId = token.TokenId;
pub const TokenOffset = token.Span;

pub const SuperpositionSchemaVersion: u32 = 1;
pub const SuperpositionSchema = "ztok.superposition.v1";

pub const FusionKind = enum(u8) {
    mean,
    weighted_mean,
    norm_preserving_mean,

    pub fn jsonName(self: FusionKind) []const u8 {
        return switch (self) {
            .mean => "mean",
            .weighted_mean => "weighted_mean",
            .norm_preserving_mean => "norm_preserving_mean",
        };
    }

    pub fn cliName(self: FusionKind) []const u8 {
        return switch (self) {
            .mean => "mean",
            .weighted_mean => "weighted-mean",
            .norm_preserving_mean => "norm-preserving-mean",
        };
    }

    pub fn parse(value: []const u8) ?FusionKind {
        if (std.mem.eql(u8, value, "mean")) return .mean;
        if (std.mem.eql(u8, value, "weighted_mean") or
            std.mem.eql(u8, value, "weighted-mean")) return .weighted_mean;
        if (std.mem.eql(u8, value, "norm_preserving_mean") or
            std.mem.eql(u8, value, "norm-preserving-mean")) return .norm_preserving_mean;
        return null;
    }
};

/// Why a fixed-plan group exists. Serialized for auditability.
pub const FixedGroupKind = enum(u8) {
    fixed_window,
    partial_window,
    preserved_special,
    preserved_boundary,
    uncovered_tail,

    pub fn jsonName(self: FixedGroupKind) []const u8 {
        return switch (self) {
            .fixed_window => "fixed_window",
            .partial_window => "partial_window",
            .preserved_special => "preserved_special",
            .preserved_boundary => "preserved_boundary",
            .uncovered_tail => "uncovered_tail",
        };
    }
};

pub const SourceToken = struct {
    token_index: u32,
    token_id: TokenId,
    byte_start: u32,
    byte_end: u32,
    /// Normalized within each group. For a non-empty group, source weights
    /// sum to one (within ordinary f32 rounding).
    weight: f32,
};

pub const SuperpositionGroup = struct {
    output_index: u32,
    sources: []SourceToken,
    fusion: FusionKind,
    kind: FixedGroupKind,
    /// Half-open source-token range. Overlapping windows may cause ranges
    /// from adjacent output groups to overlap.
    position_start: u32,
    position_end: u32,
    byte_start: u32,
    byte_end: u32,
    /// Center in source-token coordinates, e.g. [0,4) has center 1.5.
    center_position: f32,
    /// Center normalized to [0,1] against the original token sequence.
    normalized_center: f32,
};

pub const SuperpositionPlan = struct {
    schema_version: u32 = SuperpositionSchemaVersion,
    original_token_count: u32,
    output_token_count: u32,
    groups: []SuperpositionGroup,

    // One allocation backs every `groups[*].sources` slice. Keeping it on
    // the plan avoids one heap allocation per group on large corpora.
    source_storage: []SourceToken,

    pub fn deinit(self: *SuperpositionPlan, allocator: std.mem.Allocator) void {
        allocator.free(self.groups);
        allocator.free(self.source_storage);
        self.* = undefined;
    }
};

pub const EncodedSuperposition = struct {
    ids: []TokenId,
    offsets: []TokenOffset,
    plan: SuperpositionPlan,

    pub fn deinit(self: *EncodedSuperposition, allocator: std.mem.Allocator) void {
        allocator.free(self.ids);
        allocator.free(self.offsets);
        self.plan.deinit(allocator);
        self.* = undefined;
    }
};

pub const FixedSuperpositionConfig = struct {
    group_size: u16 = 4,
    /// Defaults to `group_size`. Values greater than `group_size` are
    /// rejected because they would silently omit source tokens.
    stride: ?u16 = null,
    preserve_special_tokens: bool = true,
    preserve_boundary_tokens: bool = true,
    allow_partial_final_group: bool = true,
    fusion: FusionKind = .norm_preserving_mean,

    /// Optional 1:1 metadata aligned with the input token stream. A marked
    /// token becomes a singleton when its corresponding preserve flag is
    /// enabled. The Pipeline convenience API fills `special_token_mask`
    /// from the provenance overlay.
    special_token_mask: ?[]const bool = null,
    boundary_token_mask: ?[]const bool = null,

    /// Optional 1:1 mask. `hard_boundary_before[i]` prevents a group from
    /// containing tokens on both sides of token `i`. Index zero is ignored.
    /// Caption/document separators can be represented either as a preserved
    /// boundary token or as two hard boundaries around a caller-owned span.
    hard_boundary_before: ?[]const bool = null,

    /// Optional per-token source weights. Values must be finite and
    /// non-negative. Each output group's copied weights are normalized.
    source_weights: ?[]const f32 = null,
};

const TempGroup = struct {
    source_start: usize,
    source_len: usize,
    position_start: u32,
    position_end: u32,
    byte_start: u32,
    byte_end: u32,
    kind: FixedGroupKind,
};

/// Build a fixed contiguous superposition plan over an existing token stream.
///
/// This function owns no tokenizer state and cannot affect ordinary
/// tokenization. The returned plan owns two allocator-backed arrays; release
/// them with `SuperpositionPlan.deinit`.
pub fn buildFixedPlan(
    allocator: std.mem.Allocator,
    token_ids: []const TokenId,
    offsets: []const TokenOffset,
    config: FixedSuperpositionConfig,
) !SuperpositionPlan {
    if (token_ids.len != offsets.len) return error.LengthMismatch;
    if (token_ids.len > std.math.maxInt(u32)) return error.InputTooLarge;
    if (config.group_size == 0) return error.InvalidGroupSize;
    const stride = config.stride orelse config.group_size;
    if (stride == 0 or stride > config.group_size) return error.InvalidStride;

    try validateOptionalMask(config.special_token_mask, token_ids.len);
    try validateOptionalMask(config.boundary_token_mask, token_ids.len);
    try validateOptionalMask(config.hard_boundary_before, token_ids.len);
    if (config.source_weights) |weights| {
        if (weights.len != token_ids.len) return error.LengthMismatch;
        for (weights) |weight| {
            if (!std.math.isFinite(weight) or weight < 0) return error.InvalidWeight;
        }
    }
    for (offsets) |span| {
        if (span.end < span.start) return error.InvalidOffset;
    }

    var temp_groups: std.ArrayList(TempGroup) = .empty;
    defer temp_groups.deinit(allocator);
    var sources: std.ArrayList(SourceToken) = .empty;
    errdefer sources.deinit(allocator);

    var segment_start: usize = 0;
    while (segment_start < token_ids.len) {
        if (preservedKindAt(config, segment_start)) |kind| {
            try appendGroup(
                allocator,
                &temp_groups,
                &sources,
                token_ids,
                offsets,
                config,
                segment_start,
                segment_start + 1,
                kind,
            );
            segment_start += 1;
            continue;
        }

        var segment_end = segment_start + 1;
        while (segment_end < token_ids.len) : (segment_end += 1) {
            if (preservedKindAt(config, segment_end) != null) break;
            if (config.preserve_boundary_tokens and
                maskAt(config.hard_boundary_before, segment_end)) break;
        }

        try appendSegmentGroups(
            allocator,
            &temp_groups,
            &sources,
            token_ids,
            offsets,
            config,
            segment_start,
            segment_end,
            stride,
        );
        segment_start = segment_end;
    }

    const source_storage = try sources.toOwnedSlice(allocator);
    errdefer allocator.free(source_storage);
    const groups = try allocator.alloc(SuperpositionGroup, temp_groups.items.len);
    errdefer allocator.free(groups);

    for (temp_groups.items, 0..) |tmp, output_index| {
        const center = if (tmp.position_end > tmp.position_start)
            (@as(f32, @floatFromInt(tmp.position_start)) +
                @as(f32, @floatFromInt(tmp.position_end - 1))) / 2.0
        else
            @as(f32, @floatFromInt(tmp.position_start));
        const normalized = if (token_ids.len <= 1)
            0.0
        else
            center / @as(f32, @floatFromInt(token_ids.len - 1));
        groups[output_index] = .{
            .output_index = @intCast(output_index),
            .sources = source_storage[tmp.source_start .. tmp.source_start + tmp.source_len],
            .fusion = config.fusion,
            .kind = tmp.kind,
            .position_start = tmp.position_start,
            .position_end = tmp.position_end,
            .byte_start = tmp.byte_start,
            .byte_end = tmp.byte_end,
            .center_position = center,
            .normalized_center = normalized,
        };
    }

    return .{
        .original_token_count = @intCast(token_ids.len),
        .output_token_count = @intCast(groups.len),
        .groups = groups,
        .source_storage = source_storage,
    };
}

fn validateOptionalMask(mask: ?[]const bool, expected_len: usize) !void {
    if (mask) |values| {
        if (values.len != expected_len) return error.LengthMismatch;
    }
}

fn maskAt(mask: ?[]const bool, index: usize) bool {
    return if (mask) |values| values[index] else false;
}

fn preservedKindAt(config: FixedSuperpositionConfig, index: usize) ?FixedGroupKind {
    if (config.preserve_special_tokens and maskAt(config.special_token_mask, index))
        return .preserved_special;
    if (config.preserve_boundary_tokens and maskAt(config.boundary_token_mask, index))
        return .preserved_boundary;
    return null;
}

fn appendSegmentGroups(
    allocator: std.mem.Allocator,
    temp_groups: *std.ArrayList(TempGroup),
    sources: *std.ArrayList(SourceToken),
    token_ids: []const TokenId,
    offsets: []const TokenOffset,
    config: FixedSuperpositionConfig,
    segment_start: usize,
    segment_end: usize,
    stride: u16,
) !void {
    const group_size: usize = config.group_size;
    var start = segment_start;
    var covered_until = segment_start;

    while (start < segment_end) {
        const remaining = segment_end - start;
        if (remaining < group_size and !config.allow_partial_final_group) break;

        const end = @min(start + group_size, segment_end);
        try appendGroup(
            allocator,
            temp_groups,
            sources,
            token_ids,
            offsets,
            config,
            start,
            end,
            if (end - start == group_size) .fixed_window else .partial_window,
        );
        covered_until = @max(covered_until, end);
        if (end == segment_end) break;
        start += stride;
    }

    // A non-overlapping final remainder must never disappear merely because
    // partial grouping was disabled. Preserve uncovered tokens as auditable
    // singleton units.
    while (covered_until < segment_end) : (covered_until += 1) {
        try appendGroup(
            allocator,
            temp_groups,
            sources,
            token_ids,
            offsets,
            config,
            covered_until,
            covered_until + 1,
            .uncovered_tail,
        );
    }
}

fn appendGroup(
    allocator: std.mem.Allocator,
    temp_groups: *std.ArrayList(TempGroup),
    sources: *std.ArrayList(SourceToken),
    token_ids: []const TokenId,
    offsets: []const TokenOffset,
    config: FixedSuperpositionConfig,
    start: usize,
    end: usize,
    kind: FixedGroupKind,
) !void {
    std.debug.assert(start < end);
    const source_start = sources.items.len;

    var weight_sum: f64 = 0;
    for (start..end) |index| {
        const raw_weight = if (config.source_weights) |weights| weights[index] else 1.0;
        weight_sum += raw_weight;
    }
    if (!(weight_sum > 0) or !std.math.isFinite(weight_sum)) return error.InvalidWeight;

    for (start..end) |index| {
        const raw_weight = if (config.source_weights) |weights| weights[index] else 1.0;
        try sources.append(allocator, .{
            .token_index = @intCast(index),
            .token_id = token_ids[index],
            .byte_start = offsets[index].start,
            .byte_end = offsets[index].end,
            .weight = @floatCast(@as(f64, raw_weight) / weight_sum),
        });
    }

    var byte_start = offsets[start].start;
    var byte_end = offsets[start].end;
    for (offsets[start + 1 .. end]) |span| {
        byte_start = @min(byte_start, span.start);
        byte_end = @max(byte_end, span.end);
    }

    try temp_groups.append(allocator, .{
        .source_start = source_start,
        .source_len = end - start,
        .position_start = @intCast(start),
        .position_end = @intCast(end),
        .byte_start = byte_start,
        .byte_end = byte_end,
        .kind = kind,
    });
}

pub fn writeJson(writer: *std.Io.Writer, plan: *const SuperpositionPlan) !void {
    try writer.writeAll("{\"schema\":\"");
    try writer.writeAll(SuperpositionSchema);
    try writer.print("\",\"schema_version\":{d},\"original_token_count\":{d},\"output_token_count\":{d},\"groups\":[", .{
        plan.schema_version,
        plan.original_token_count,
        plan.output_token_count,
    });
    for (plan.groups, 0..) |group, group_index| {
        if (group_index != 0) try writer.writeByte(',');
        try writer.print(
            "{{\"output_index\":{d},\"kind\":\"{s}\",\"fusion\":\"{s}\",\"position_start\":{d},\"position_end\":{d},\"byte_start\":{d},\"byte_end\":{d},\"center_position\":{d},\"normalized_center\":{d},\"sources\":[",
            .{
                group.output_index,
                group.kind.jsonName(),
                group.fusion.jsonName(),
                group.position_start,
                group.position_end,
                group.byte_start,
                group.byte_end,
                group.center_position,
                group.normalized_center,
            },
        );
        for (group.sources, 0..) |source, source_index| {
            if (source_index != 0) try writer.writeByte(',');
            try writer.print(
                "{{\"token_index\":{d},\"token_id\":{d},\"byte_start\":{d},\"byte_end\":{d},\"weight\":{d}}}",
                .{
                    source.token_index,
                    source.token_id,
                    source.byte_start,
                    source.byte_end,
                    source.weight,
                },
            );
        }
        try writer.writeAll("]}");
    }
    try writer.writeAll("]}");
}

pub fn toJsonAlloc(
    allocator: std.mem.Allocator,
    plan: *const SuperpositionPlan,
) ![]u8 {
    var output: std.Io.Writer.Allocating = .init(allocator);
    errdefer output.deinit();
    try writeJson(&output.writer, plan);
    return output.toOwnedSlice();
}

test "fixed plan groups divisible input exactly" {
    const ids = [_]TokenId{ 10, 11, 12, 13, 14, 15, 16, 17 };
    const offsets = [_]TokenOffset{
        .{ .start = 0, .end = 1 },
        .{ .start = 1, .end = 2 },
        .{ .start = 2, .end = 3 },
        .{ .start = 3, .end = 4 },
        .{ .start = 4, .end = 5 },
        .{ .start = 5, .end = 6 },
        .{ .start = 6, .end = 7 },
        .{ .start = 7, .end = 8 },
    };
    var plan = try buildFixedPlan(std.testing.allocator, &ids, &offsets, .{});
    defer plan.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u32, 8), plan.original_token_count);
    try std.testing.expectEqual(@as(u32, 2), plan.output_token_count);
    try std.testing.expectEqualSlices(TokenId, ids[0..4], &.{
        plan.groups[0].sources[0].token_id,
        plan.groups[0].sources[1].token_id,
        plan.groups[0].sources[2].token_id,
        plan.groups[0].sources[3].token_id,
    });
    try std.testing.expectEqual(@as(u32, 0), plan.groups[0].byte_start);
    try std.testing.expectEqual(@as(u32, 4), plan.groups[0].byte_end);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), plan.groups[0].sources[0].weight, 0.00001);
}

test "fixed plan partial final group and no-partial preservation" {
    const ids = [_]TokenId{ 1, 2, 3, 4, 5 };
    const offsets = [_]TokenOffset{
        .{ .start = 0, .end = 1 },
        .{ .start = 1, .end = 2 },
        .{ .start = 2, .end = 3 },
        .{ .start = 3, .end = 4 },
        .{ .start = 4, .end = 5 },
    };

    var partial = try buildFixedPlan(std.testing.allocator, &ids, &offsets, .{});
    defer partial.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 2), partial.groups.len);
    try std.testing.expectEqual(FixedGroupKind.partial_window, partial.groups[1].kind);
    try std.testing.expectEqual(@as(usize, 1), partial.groups[1].sources.len);

    var preserved = try buildFixedPlan(std.testing.allocator, &ids, &offsets, .{
        .allow_partial_final_group = false,
    });
    defer preserved.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 2), preserved.groups.len);
    try std.testing.expectEqual(FixedGroupKind.uncovered_tail, preserved.groups[1].kind);
    try std.testing.expectEqual(@as(TokenId, 5), preserved.groups[1].sources[0].token_id);
}

test "fixed plan preserves special and boundary tokens" {
    const ids = [_]TokenId{ 1, 2, 99, 3, 77, 4, 5 };
    const offsets = [_]TokenOffset{
        .{ .start = 0, .end = 1 },
        .{ .start = 1, .end = 2 },
        .{ .start = 2, .end = 7 },
        .{ .start = 7, .end = 8 },
        .{ .start = 8, .end = 9 },
        .{ .start = 9, .end = 10 },
        .{ .start = 10, .end = 11 },
    };
    const specials = [_]bool{ false, false, true, false, false, false, false };
    const boundaries = [_]bool{ false, false, false, false, true, false, false };

    var plan = try buildFixedPlan(std.testing.allocator, &ids, &offsets, .{
        .group_size = 4,
        .special_token_mask = &specials,
        .boundary_token_mask = &boundaries,
    });
    defer plan.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 5), plan.groups.len);
    try std.testing.expectEqual(FixedGroupKind.preserved_special, plan.groups[1].kind);
    try std.testing.expectEqual(@as(TokenId, 99), plan.groups[1].sources[0].token_id);
    try std.testing.expectEqual(FixedGroupKind.preserved_boundary, plan.groups[3].kind);
    try std.testing.expectEqual(@as(TokenId, 77), plan.groups[3].sources[0].token_id);
}

test "fixed plan does not cross a hard boundary" {
    const ids = [_]TokenId{ 1, 2, 3, 4, 5, 6 };
    const offsets = [_]TokenOffset{
        .{ .start = 0, .end = 1 },
        .{ .start = 1, .end = 2 },
        .{ .start = 2, .end = 3 },
        .{ .start = 3, .end = 4 },
        .{ .start = 4, .end = 5 },
        .{ .start = 5, .end = 6 },
    };
    const hard = [_]bool{ false, false, false, true, false, false };
    var plan = try buildFixedPlan(std.testing.allocator, &ids, &offsets, .{
        .group_size = 4,
        .hard_boundary_before = &hard,
    });
    defer plan.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 2), plan.groups.len);
    try std.testing.expectEqual(@as(u32, 3), plan.groups[0].position_end);
    try std.testing.expectEqual(@as(u32, 3), plan.groups[1].position_start);
}

test "fixed plan empty, one-token, unicode offsets, and deterministic json" {
    var empty = try buildFixedPlan(std.testing.allocator, &.{}, &.{}, .{});
    defer empty.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 0), empty.groups.len);

    const ids = [_]TokenId{42};
    const offsets = [_]TokenOffset{.{ .start = 0, .end = 4 }};
    var one = try buildFixedPlan(std.testing.allocator, &ids, &offsets, .{});
    defer one.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 4), one.groups[0].byte_end);

    const json_a = try toJsonAlloc(std.testing.allocator, &one);
    defer std.testing.allocator.free(json_a);
    const json_b = try toJsonAlloc(std.testing.allocator, &one);
    defer std.testing.allocator.free(json_b);
    try std.testing.expectEqualStrings(json_a, json_b);
    try std.testing.expect(std.mem.indexOf(u8, json_a, "\"schema\":\"ztok.superposition.v1\"") != null);
}

test "fixed plan rejects invalid inputs" {
    const ids = [_]TokenId{1};
    const offsets = [_]TokenOffset{.{ .start = 0, .end = 1 }};
    try std.testing.expectError(
        error.InvalidGroupSize,
        buildFixedPlan(std.testing.allocator, &ids, &offsets, .{ .group_size = 0 }),
    );
    try std.testing.expectError(
        error.InvalidStride,
        buildFixedPlan(std.testing.allocator, &ids, &offsets, .{ .group_size = 2, .stride = 3 }),
    );
    try std.testing.expectError(
        error.LengthMismatch,
        buildFixedPlan(std.testing.allocator, &ids, &.{}, .{}),
    );
}

test "fixed plan normalizes caller weights and supports deterministic overlap" {
    const ids = [_]TokenId{ 1, 2, 3, 4, 5, 6 };
    const offsets = [_]TokenOffset{
        .{ .start = 0, .end = 1 },
        .{ .start = 1, .end = 2 },
        .{ .start = 2, .end = 3 },
        .{ .start = 3, .end = 4 },
        .{ .start = 4, .end = 5 },
        .{ .start = 5, .end = 6 },
    };
    const weights = [_]f32{ 1, 3, 1, 1, 2, 2 };
    var plan = try buildFixedPlan(std.testing.allocator, &ids, &offsets, .{
        .group_size = 4,
        .stride = 2,
        .fusion = .weighted_mean,
        .source_weights = &weights,
    });
    defer plan.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 2), plan.groups.len);
    try std.testing.expectEqual(@as(u32, 0), plan.groups[0].position_start);
    try std.testing.expectEqual(@as(u32, 4), plan.groups[0].position_end);
    try std.testing.expectEqual(@as(u32, 2), plan.groups[1].position_start);
    try std.testing.expectEqual(@as(u32, 6), plan.groups[1].position_end);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 6.0), plan.groups[0].sources[0].weight, 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), plan.groups[0].sources[1].weight, 0.00001);

    const zero_weights = [_]f32{ 0, 0, 0, 0, 0, 0 };
    try std.testing.expectError(
        error.InvalidWeight,
        buildFixedPlan(std.testing.allocator, &ids, &offsets, .{
            .source_weights = &zero_weights,
        }),
    );
}

test "fixed plan handles large input with contiguous source storage" {
    const token_count = 100_000;
    const ids = try std.testing.allocator.alloc(TokenId, token_count);
    defer std.testing.allocator.free(ids);
    const offsets = try std.testing.allocator.alloc(TokenOffset, token_count);
    defer std.testing.allocator.free(offsets);
    for (ids, offsets, 0..) |*id, *offset, index| {
        id.* = @intCast(index);
        offset.* = .{ .start = @intCast(index), .end = @intCast(index + 1) };
    }

    var plan = try buildFixedPlan(std.testing.allocator, ids, offsets, .{
        .group_size = 4,
    });
    defer plan.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 25_000), plan.output_token_count);
    try std.testing.expectEqual(token_count, plan.source_storage.len);
    try std.testing.expectEqual(
        plan.source_storage.ptr,
        plan.groups[0].sources.ptr,
    );
}
