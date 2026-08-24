//! JSON exchange format loader for `ztok.semantic_spans.v1`.
//!
//! Embeddings remain caller data: the loader only flattens inline JSON arrays
//! into the contiguous matrix consumed by the CCSS reference algorithm.

const std = @import("std");
const semantic = @import("semantic_superposition.zig");

const WireToken = struct {
    token_index: u32,
    token_id: u32,
    byte_start: u32,
    byte_end: u32,
};

const WireSpan = struct {
    span_id: []const u8,
    token_start: u32,
    token_end: u32,
    byte_start: ?u32 = null,
    byte_end: ?u32 = null,
    text: []const u8 = "",
    role: ?[]const u8 = null,
    entity_id: ?[]const u8 = null,
    confidence: f32 = 1.0,
    embedding: []const f32,
    grounding: ?[]const f32 = null,
    grounding_confidence: f32 = 1.0,
    section_reliability: f32 = 1.0,
    kind: ?[]const u8 = null,
    order_class: ?[]const u8 = null,
    relation_from_entity_id: ?[]const u8 = null,
    relation_to_entity_id: ?[]const u8 = null,
    contradicts: []const []const u8 = &.{},
};

const WireCaption = struct {
    caption_id: u32,
    section: ?[]const u8 = null,
    text: []const u8,
    quality: f32 = 1.0,
    tokens: []const WireToken = &.{},
    spans: []const WireSpan = &.{},
};

const WireDocument = struct {
    schema: []const u8,
    image_id: []const u8,
    original_token_count: ?u32 = null,
    captions: []const WireCaption,
};

pub const OwnedSemanticInput = struct {
    arena: std.heap.ArenaAllocator,
    input: semantic.SemanticInput,
    role_ids: std.StringHashMap(u32),
    entity_ids: std.StringHashMap(u32),
    section_ids: std.StringHashMap(u32),
    role_names: [][]const u8,
    entity_names: [][]const u8,
    section_names: [][]const u8,

    pub fn deinit(self: *OwnedSemanticInput) void {
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn roleId(self: *const OwnedSemanticInput, name: []const u8) ?u32 {
        return self.role_ids.get(name);
    }

    pub fn roleName(self: *const OwnedSemanticInput, id: u32) ?[]const u8 {
        return if (id < self.role_names.len) self.role_names[id] else null;
    }

    pub fn entityName(self: *const OwnedSemanticInput, id: u32) ?[]const u8 {
        return if (id < self.entity_names.len) self.entity_names[id] else null;
    }

    /// Conservative defaults for the protected roles named in the v1 brief.
    /// Directional/state/count roles are never fused unless a config file
    /// explicitly removes them from this list.
    pub fn defaultConfig(self: *OwnedSemanticInput) !semantic.SemanticSuperpositionConfig {
        const allocator = self.arena.allocator();
        var protected: std.ArrayList(u32) = .empty;
        var never: std.ArrayList(u32) = .empty;
        for (self.role_names, 0..) |name, id| {
            if (isProtectedRoleName(name)) try protected.append(allocator, @intCast(id));
            if (isNeverFuseRoleName(name)) try never.append(allocator, @intCast(id));
        }
        return .{
            .protected_role_ids = try protected.toOwnedSlice(allocator),
            .never_fuse_role_ids = try never.toOwnedSlice(allocator),
        };
    }

    pub fn loadConfig(
        self: *OwnedSemanticInput,
        json_bytes: []const u8,
    ) !semantic.SemanticSuperpositionConfig {
        var config = try self.defaultConfig();
        const allocator = self.arena.allocator();
        const root = try std.json.parseFromSliceLeaky(
            std.json.Value,
            allocator,
            json_bytes,
            .{},
        );
        if (root != .object) return error.MalformedConfig;
        const object = root.object;
        if (object.get("default_cosine_threshold")) |value|
            config.default_cosine_threshold = try jsonF32(value);
        if (object.get("protected_cosine_threshold")) |value|
            config.protected_cosine_threshold = try jsonF32(value);
        if (object.get("minimum_score")) |value|
            config.minimum_score = try jsonF32(value);
        if (object.get("semantic_weight")) |value|
            config.semantic_weight = try jsonF32(value);
        if (object.get("role_weight")) |value|
            config.role_weight = try jsonF32(value);
        if (object.get("entity_weight")) |value|
            config.entity_weight = try jsonF32(value);
        if (object.get("grounding_weight")) |value|
            config.grounding_weight = try jsonF32(value);
        if (object.get("contradiction_weight")) |value|
            config.contradiction_weight = try jsonF32(value);
        if (object.get("contradiction_block_threshold")) |value|
            config.contradiction_block_threshold = try jsonF32(value);
        if (object.get("maximum_cluster_size")) |value|
            config.maximum_cluster_size = try jsonU16(value);
        if (object.get("minimum_support_count")) |value|
            config.minimum_support_count = try jsonU16(value);
        if (object.get("minimum_support_fraction")) |value|
            config.minimum_support_fraction = try jsonF32(value);
        if (object.get("verbose_diagnostics")) |value| {
            if (value != .bool) return error.MalformedConfig;
            config.verbose_diagnostics = value.bool;
        }
        if (object.get("require_role_match")) |value| {
            if (value != .bool) return error.MalformedConfig;
            config.require_role_match = value.bool;
        }
        if (object.get("require_entity_match")) |value| {
            if (value != .bool) return error.MalformedConfig;
            config.require_entity_match = value.bool;
        }
        if (object.get("block_contradictions")) |value| {
            if (value != .bool) return error.MalformedConfig;
            config.block_contradictions = value.bool;
        }
        if (object.get("fusion")) |value| {
            if (value != .string) return error.MalformedConfig;
            config.fusion = @import("superposition.zig").FusionKind.parse(value.string) orelse
                return error.MalformedConfig;
        }
        if (object.get("role_thresholds")) |value| {
            if (value != .object) return error.MalformedConfig;
            const thresholds = try allocator.alloc(
                semantic.RoleThreshold,
                value.object.count(),
            );
            var written: usize = 0;
            var iterator = value.object.iterator();
            while (iterator.next()) |entry| {
                const role_id = self.roleId(entry.key_ptr.*) orelse
                    return error.UnknownRole;
                thresholds[written] = .{
                    .role_id = role_id,
                    .cosine_threshold = try jsonF32(entry.value_ptr.*),
                };
                written += 1;
            }
            config.role_thresholds = thresholds;
        }
        if (object.get("never_fuse_roles")) |value|
            config.never_fuse_role_ids = try parseRoleList(self, value);
        if (object.get("protected_roles")) |value|
            config.protected_role_ids = try parseRoleList(self, value);
        return config;
    }
};

fn jsonF32(value: std.json.Value) !f32 {
    return switch (value) {
        .float => |number| @floatCast(number),
        .integer => |number| @floatFromInt(number),
        else => error.MalformedConfig,
    };
}

fn jsonU16(value: std.json.Value) !u16 {
    if (value != .integer or value.integer < 0 or value.integer > std.math.maxInt(u16))
        return error.MalformedConfig;
    return @intCast(value.integer);
}

fn parseRoleList(
    owned: *OwnedSemanticInput,
    value: std.json.Value,
) ![]u32 {
    if (value != .array) return error.MalformedConfig;
    const result = try owned.arena.allocator().alloc(u32, value.array.items.len);
    for (value.array.items, 0..) |item, index| {
        if (item != .string) return error.MalformedConfig;
        result[index] = owned.roleId(item.string) orelse return error.UnknownRole;
    }
    return result;
}

fn normalizedRoleNameMatches(name: []const u8, choices: []const []const u8) bool {
    for (choices) |choice| {
        if (std.ascii.eqlIgnoreCase(name, choice)) return true;
    }
    return false;
}

fn isProtectedRoleName(name: []const u8) bool {
    return normalizedRoleNameMatches(name, &.{
        "count",
        "negation",
        "left_right",
        "above_below",
        "front_behind",
        "spatial_relation",
        "posture",
        "day_night",
        "clothing_state",
        "object_identity",
    });
}

fn isNeverFuseRoleName(name: []const u8) bool {
    return normalizedRoleNameMatches(name, &.{
        "count",
        "negation",
        "left_right",
        "above_below",
        "front_behind",
        "posture",
        "day_night",
        "clothing_state",
    });
}

fn parseSpanKind(value: ?[]const u8) !semantic.SpanKind {
    const name = value orelse return .other;
    inline for (@typeInfo(semantic.SpanKind).@"enum".fields) |field| {
        if (std.mem.eql(u8, name, field.name)) return @enumFromInt(field.value);
    }
    return error.UnknownSpanKind;
}

fn parseOrderClass(
    value: ?[]const u8,
    kind: semantic.SpanKind,
    section: ?[]const u8,
) !semantic.OrderClass {
    if (value) |name| {
        inline for (@typeInfo(semantic.OrderClass).@"enum".fields) |field| {
            if (std.mem.eql(u8, name, field.name)) return @enumFromInt(field.value);
        }
        return error.UnknownOrderClass;
    }
    if (section) |name| {
        if (std.ascii.eqlIgnoreCase(name, "lighting") or
            std.ascii.eqlIgnoreCase(name, "appearance")) return .lighting_appearance;
        if (std.ascii.eqlIgnoreCase(name, "style") or
            std.ascii.eqlIgnoreCase(name, "medium")) return .style_medium;
        if (std.ascii.eqlIgnoreCase(name, "environment")) return .environment;
        if (std.ascii.eqlIgnoreCase(name, "ocr") or
            std.ascii.eqlIgnoreCase(name, "visible_text")) return .visible_text;
    }
    return switch (kind) {
        .entity => .main_subject,
        .attribute => .subject_attribute,
        .action, .relation, .spatial_connector => .action_relation,
        .visible_text => .visible_text,
        .uncertainty => .uncertainty_alternative,
        else => .environment,
    };
}

fn intern(
    allocator: std.mem.Allocator,
    map: *std.StringHashMap(u32),
    names: *std.ArrayList([]const u8),
    value: ?[]const u8,
) !?u32 {
    const name = value orelse return null;
    if (map.get(name)) |id| return id;
    const id: u32 = @intCast(names.items.len);
    try map.put(name, id);
    try names.append(allocator, name);
    return id;
}

const ByteSpan = struct {
    start: u32,
    end: u32,
};

fn bytesForSpan(caption: WireCaption, span: WireSpan) !ByteSpan {
    if (span.byte_start != null or span.byte_end != null) {
        if (span.byte_start == null or span.byte_end == null) return error.IncompleteByteSpan;
        const result: ByteSpan = .{ .start = span.byte_start.?, .end = span.byte_end.? };
        if (result.end < result.start or
            (caption.text.len != 0 and result.end > caption.text.len))
            return error.InvalidByteSpan;
        return result;
    }
    if (span.token_start >= span.token_end or span.token_end > caption.tokens.len)
        return error.InvalidTokenSpan;
    var start = caption.tokens[span.token_start].byte_start;
    var end = caption.tokens[span.token_start].byte_end;
    for (caption.tokens[span.token_start + 1 .. span.token_end]) |token| {
        start = @min(start, token.byte_start);
        end = @max(end, token.byte_end);
    }
    if (end < start or (caption.text.len != 0 and end > caption.text.len))
        return error.InvalidByteSpan;
    return .{ .start = start, .end = end };
}

pub fn loadFromBytes(
    backing_allocator: std.mem.Allocator,
    json_bytes: []const u8,
) !OwnedSemanticInput {
    var arena = std.heap.ArenaAllocator.init(backing_allocator);
    errdefer arena.deinit();
    const allocator = arena.allocator();
    const wire = try std.json.parseFromSliceLeaky(
        WireDocument,
        allocator,
        json_bytes,
        .{
            .ignore_unknown_fields = true,
            .allocate = .alloc_always,
        },
    );
    if (!std.mem.eql(u8, wire.schema, semantic.SemanticSpansSchema))
        return error.UnsupportedSchema;

    var span_count: usize = 0;
    var token_count: usize = 0;
    var maximum_caption_id: u32 = 0;
    var caption_ids = std.AutoHashMap(u32, void).init(allocator);
    var embedding_dimensions: ?usize = null;
    var grounding_dimensions: ?usize = null;
    var grounding_present: ?bool = null;
    for (wire.captions) |caption| {
        if (caption_ids.contains(caption.caption_id)) return error.DuplicateCaptionId;
        try caption_ids.put(caption.caption_id, {});
        maximum_caption_id = @max(maximum_caption_id, caption.caption_id);
        span_count += caption.spans.len;
        token_count += caption.tokens.len;
        for (caption.tokens, 0..) |token, token_index| {
            if (token.token_index != token_index or
                token.byte_end < token.byte_start or
                (caption.text.len != 0 and token.byte_end > caption.text.len))
                return error.InvalidToken;
        }
        for (caption.spans) |span| {
            if (embedding_dimensions) |dimensions| {
                if (span.embedding.len != dimensions) return error.DimensionMismatch;
            } else {
                if (span.embedding.len == 0) return error.DimensionMismatch;
                embedding_dimensions = span.embedding.len;
            }
            if (grounding_present) |present| {
                if (present != (span.grounding != null)) return error.IncompleteGroundingMatrix;
            } else {
                grounding_present = span.grounding != null;
            }
            if (span.grounding) |grounding| {
                if (grounding_dimensions) |dimensions| {
                    if (grounding.len != dimensions) return error.DimensionMismatch;
                } else {
                    if (grounding.len == 0) return error.DimensionMismatch;
                    grounding_dimensions = grounding.len;
                }
            }
        }
    }
    if (span_count > 0 and embedding_dimensions == null) return error.DimensionMismatch;
    const dimensions = embedding_dimensions orelse 1;
    const grounding_width = grounding_dimensions orelse 1;
    const caption_count: u32 = if (wire.captions.len == 0) 0 else maximum_caption_id + 1;
    if (caption_ids.count() != caption_count) return error.NonContiguousCaptionIds;

    const spans = try allocator.alloc(semantic.SemanticSpan, span_count);
    const embedding_values = try allocator.alloc(f32, span_count * dimensions);
    const grounding_values = if (grounding_present orelse false)
        try allocator.alloc(f32, span_count * grounding_width)
    else
        null;
    const caption_quality = try allocator.alloc(f32, caption_count);
    @memset(caption_quality, 1.0);
    const grounding_confidence = try allocator.alloc(f32, span_count);
    const section_reliability = try allocator.alloc(f32, span_count);
    var role_ids = std.StringHashMap(u32).init(allocator);
    var entity_ids = std.StringHashMap(u32).init(allocator);
    var section_ids = std.StringHashMap(u32).init(allocator);
    var role_names: std.ArrayList([]const u8) = .empty;
    var entity_names: std.ArrayList([]const u8) = .empty;
    var section_names: std.ArrayList([]const u8) = .empty;
    var span_ids = std.StringHashMap(u32).init(allocator);

    var span_index: usize = 0;
    for (wire.captions) |caption| {
        caption_quality[caption.caption_id] = caption.quality;
        for (caption.spans) |wire_span| {
            if (span_ids.contains(wire_span.span_id)) return error.DuplicateSpanId;
            try span_ids.put(wire_span.span_id, @intCast(span_index));
            const byte_span = try bytesForSpan(caption, wire_span);
            const kind = try parseSpanKind(wire_span.kind);
            const role_id = try intern(
                allocator,
                &role_ids,
                &role_names,
                wire_span.role,
            );
            const entity_id = try intern(
                allocator,
                &entity_ids,
                &entity_names,
                wire_span.entity_id,
            );
            spans[span_index] = .{
                .span_id = wire_span.span_id,
                .text = wire_span.text,
                .caption_id = caption.caption_id,
                .section_id = try intern(
                    allocator,
                    &section_ids,
                    &section_names,
                    caption.section,
                ),
                .section_name = caption.section,
                .token_start = wire_span.token_start,
                .token_end = wire_span.token_end,
                .byte_start = byte_span.start,
                .byte_end = byte_span.end,
                .role_id = role_id,
                .role_name = wire_span.role,
                .entity_id = entity_id,
                .entity_name = wire_span.entity_id,
                .confidence = wire_span.confidence,
                .embedding_index = @intCast(span_index),
                .kind = kind,
                .order_class = try parseOrderClass(wire_span.order_class, kind, caption.section),
                .relation_from_entity_id = try intern(
                    allocator,
                    &entity_ids,
                    &entity_names,
                    wire_span.relation_from_entity_id,
                ),
                .relation_to_entity_id = try intern(
                    allocator,
                    &entity_ids,
                    &entity_names,
                    wire_span.relation_to_entity_id,
                ),
            };
            grounding_confidence[span_index] = wire_span.grounding_confidence;
            section_reliability[span_index] = wire_span.section_reliability;
            @memcpy(
                embedding_values[span_index * dimensions ..][0..dimensions],
                wire_span.embedding,
            );
            if (grounding_values) |values| @memcpy(
                values[span_index * grounding_width ..][0..grounding_width],
                wire_span.grounding.?,
            );
            span_index += 1;
        }
    }

    const contradictions = try allocator.alloc(bool, span_count * span_count);
    @memset(contradictions, false);
    span_index = 0;
    for (wire.captions) |caption| {
        for (caption.spans) |wire_span| {
            for (wire_span.contradicts) |other_id| {
                const other = span_ids.get(other_id) orelse return error.UnknownSpanId;
                contradictions[span_index * span_count + other] = true;
                contradictions[other * span_count + span_index] = true;
            }
            span_index += 1;
        }
    }

    return .{
        .arena = arena,
        .input = .{
            .image_id = wire.image_id,
            .caption_count = caption_count,
            .original_token_count = wire.original_token_count orelse @intCast(token_count),
            .spans = spans,
            .embeddings = .{
                .values = embedding_values,
                .rows = @intCast(span_count),
                .dimensions = @intCast(dimensions),
            },
            .grounding_embeddings = if (grounding_values) |values| .{
                .values = values,
                .rows = @intCast(span_count),
                .dimensions = @intCast(grounding_width),
            } else null,
            .contradictions = contradictions,
            .caption_quality = caption_quality,
            .grounding_confidence = grounding_confidence,
            .section_reliability = section_reliability,
        },
        .role_ids = role_ids,
        .entity_ids = entity_ids,
        .section_ids = section_ids,
        .role_names = try role_names.toOwnedSlice(allocator),
        .entity_names = try entity_names.toOwnedSlice(allocator),
        .section_names = try section_names.toOwnedSlice(allocator),
    };
}

test "semantic exchange loads spans embeddings roles boundaries and contradictions" {
    const json =
        \\{
        \\  "schema": "ztok.semantic_spans.v1",
        \\  "image_id": "image-1",
        \\  "captions": [
        \\    {"caption_id":0,"section":"lighting","text":"bright light","tokens":[
        \\      {"token_index":0,"token_id":1,"byte_start":0,"byte_end":6},
        \\      {"token_index":1,"token_id":2,"byte_start":7,"byte_end":12}
        \\    ],"spans":[
        \\      {"span_id":"c0-s0","token_start":0,"token_end":1,"text":"bright",
        \\       "role":"lighting_intensity","entity_id":"light-1",
        \\       "embedding":[1.0,0.0],"confidence":0.94,"contradicts":["c1-s0"]}
        \\    ]},
        \\    {"caption_id":1,"text":"dim light","tokens":[],"spans":[
        \\      {"span_id":"c1-s0","token_start":0,"token_end":1,
        \\       "byte_start":0,"byte_end":3,"text":"dim",
        \\       "role":"lighting_intensity","entity_id":"light-1",
        \\       "embedding":[-1.0,0.0],"confidence":0.9}
        \\    ]}
        \\  ]
        \\}
    ;
    var owned = try loadFromBytes(std.testing.allocator, json);
    defer owned.deinit();
    try std.testing.expectEqualStrings("image-1", owned.input.image_id);
    try std.testing.expectEqual(@as(usize, 2), owned.input.spans.len);
    try std.testing.expectEqual(@as(u32, 0), owned.input.spans[0].byte_start);
    try std.testing.expectEqual(@as(u32, 6), owned.input.spans[0].byte_end);
    try std.testing.expectEqual(owned.input.spans[0].role_id, owned.input.spans[1].role_id);
    try std.testing.expect(owned.input.contradictions.?[1]);
}

test "semantic config resolves role thresholds and protected defaults" {
    const json =
        \\{"schema":"ztok.semantic_spans.v1","image_id":"x","captions":[
        \\{"caption_id":0,"text":"one","spans":[
        \\{"span_id":"s","token_start":0,"token_end":1,"byte_start":0,"byte_end":3,
        \\"role":"count","embedding":[1.0]}
        \\]}]}
    ;
    var owned = try loadFromBytes(std.testing.allocator, json);
    defer owned.deinit();
    const config = try owned.loadConfig(
        \\{"default_cosine_threshold":0.9,
        \\"role_thresholds":{"count":0.999},
        \\"fusion":"weighted-mean","verbose_diagnostics":true}
    );
    try std.testing.expectEqual(@as(usize, 1), config.never_fuse_role_ids.len);
    try std.testing.expectApproxEqAbs(@as(f32, 0.999), config.role_thresholds[0].cosine_threshold, 0.00001);
    try std.testing.expect(config.verbose_diagnostics);
}
