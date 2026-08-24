//! Experimental Cross-Caption Semantic Superposition (CCSS).
//!
//! This module consumes caller-supplied contextual span embeddings and
//! semantic labels. It performs no model inference and has no dependency on
//! Qwen, PyTorch, CUDA, or an image model.

const std = @import("std");
const fixed = @import("superposition.zig");
const TokenOffset = @import("token.zig").Span;

pub const SemanticSpansSchema = "ztok.semantic_spans.v1";
pub const CcssSchema = "ztok.ccss.v1";
pub const CcssSchemaVersion: u32 = 1;

pub const SpanKind = enum(u8) {
    entity,
    attribute,
    action,
    relation,
    quantifier,
    negation,
    spatial_connector,
    uncertainty,
    visible_text,
    detail,
    separator,
    other,
};

/// Stable ordering buckets. Numeric order is intentional.
pub const OrderClass = enum(u8) {
    global_scene,
    main_subject,
    subject_attribute,
    action_relation,
    environment,
    lighting_appearance,
    style_medium,
    visible_text,
    uncertainty_alternative,
};

pub const SemanticSpan = struct {
    span_id: []const u8,
    text: []const u8 = "",
    caption_id: u32,
    section_id: ?u32 = null,
    section_name: ?[]const u8 = null,
    token_start: u32,
    token_end: u32,
    byte_start: u32,
    byte_end: u32,
    role_id: ?u32 = null,
    role_name: ?[]const u8 = null,
    entity_id: ?u32 = null,
    entity_name: ?[]const u8 = null,
    confidence: f32 = 1.0,
    embedding_index: u32,
    kind: SpanKind = .other,
    order_class: OrderClass = .environment,
    relation_from_entity_id: ?u32 = null,
    relation_to_entity_id: ?u32 = null,
};

pub const TokenRange = struct {
    token_start: u32,
    token_end: u32,
};

/// Map a non-empty source byte span to the half-open range of tokenizer
/// tokens that overlap it. This keeps multi-piece words and phrases whole
/// without performing semantic work on isolated subword IDs.
pub fn mapByteSpanToTokenRange(
    offsets: []const TokenOffset,
    byte_start: u32,
    byte_end: u32,
) !TokenRange {
    if (byte_end <= byte_start) return error.InvalidByteSpan;
    var first: ?usize = null;
    var last: usize = 0;
    for (offsets, 0..) |offset, index| {
        if (offset.end < offset.start) return error.InvalidOffset;
        if (offset.end > byte_start and offset.start < byte_end) {
            if (first == null) first = index;
            last = index + 1;
        }
    }
    const start = first orelse return error.ByteSpanNotCovered;
    if (last > std.math.maxInt(u32)) return error.TooManyTokens;
    return .{ .token_start = @intCast(start), .token_end = @intCast(last) };
}

pub const EmbeddingMatrix = struct {
    values: []const f32,
    rows: u32,
    dimensions: u32,

    pub fn row(self: EmbeddingMatrix, index: u32) ![]const f32 {
        if (self.dimensions == 0 or index >= self.rows) return error.InvalidEmbeddingIndex;
        const start = std.math.mul(usize, index, self.dimensions) catch
            return error.InvalidEmbeddingMatrix;
        return self.values[start .. start + self.dimensions];
    }

    fn validate(self: EmbeddingMatrix) !void {
        if (self.dimensions == 0) return error.InvalidEmbeddingMatrix;
        const expected = std.math.mul(usize, self.rows, self.dimensions) catch
            return error.InvalidEmbeddingMatrix;
        if (self.values.len != expected) return error.InvalidEmbeddingMatrix;
        for (self.values) |value| {
            if (!std.math.isFinite(value)) return error.InvalidEmbedding;
        }
    }
};

pub const RoleThreshold = struct {
    role_id: u32,
    cosine_threshold: f32,
};

pub const SemanticSuperpositionConfig = struct {
    default_cosine_threshold: f32 = 0.92,
    role_thresholds: []const RoleThreshold = &.{},
    protected_role_ids: []const u32 = &.{},
    never_fuse_role_ids: []const u32 = &.{},
    protected_cosine_threshold: f32 = 0.995,
    semantic_weight: f32 = 1.0,
    role_weight: f32 = 0.0,
    entity_weight: f32 = 0.0,
    grounding_weight: f32 = 0.0,
    contradiction_weight: f32 = 1.0,
    contradiction_block_threshold: f32 = 0.5,
    minimum_score: f32 = 0.92,
    maximum_cluster_size: u16 = 12,
    minimum_support_count: u16 = 2,
    minimum_support_fraction: f32 = 0.0,
    require_role_match: bool = true,
    require_entity_match: bool = true,
    block_contradictions: bool = true,
    fusion: fixed.FusionKind = .norm_preserving_mean,
    verbose_diagnostics: bool = false,
};

pub const SemanticInput = struct {
    image_id: []const u8,
    caption_count: u32,
    original_token_count: u32,
    spans: []const SemanticSpan,
    embeddings: EmbeddingMatrix,
    /// Optional span-aligned grounding vectors. When absent, grounding
    /// similarity contributes zero rather than being guessed.
    grounding_embeddings: ?EmbeddingMatrix = null,
    /// Optional row-major [span][span] explicit contradiction matrix.
    contradictions: ?[]const bool = null,
    /// Optional row-major [span][span] contradiction scores in [0,1].
    contradiction_scores: ?[]const f32 = null,
    caption_quality: ?[]const f32 = null,
    grounding_confidence: ?[]const f32 = null,
    section_reliability: ?[]const f32 = null,
};

pub const RejectionReason = enum(u8) {
    none,
    same_caption,
    role_mismatch,
    entity_mismatch,
    protected_role,
    explicit_contradiction,
    cosine_below_threshold,
    score_below_threshold,
    duplicate_caption_in_cluster,
    complete_link_failed,
    maximum_cluster_size,

    pub fn jsonName(self: RejectionReason) []const u8 {
        return switch (self) {
            .none => "none",
            .same_caption => "same_caption",
            .role_mismatch => "role_mismatch",
            .entity_mismatch => "entity_mismatch",
            .protected_role => "protected_role",
            .explicit_contradiction => "explicit_contradiction",
            .cosine_below_threshold => "cosine_below_threshold",
            .score_below_threshold => "score_below_threshold",
            .duplicate_caption_in_cluster => "duplicate_caption_in_cluster",
            .complete_link_failed => "complete_link_failed",
            .maximum_cluster_size => "maximum_cluster_size",
        };
    }
};

pub const PairEvaluation = struct {
    left_span_index: u32,
    right_span_index: u32,
    cosine_similarity: f32,
    grounding_similarity: f32,
    role_compatibility: f32,
    entity_compatibility: f32,
    contradiction: f32,
    score: f32,
    cosine_threshold: f32,
    accepted: bool,
    rejection: RejectionReason,
};

pub const ConsensusGroup = struct {
    group_id: u32,
    role_id: ?u32,
    entity_id: ?u32,
    member_span_indices: []u32,
    member_caption_ids: []u32,
    support_count: u16,
    support_fraction: f32,
    mean_similarity: f32,
    minimum_pair_similarity: f32,
    dispersion: f32,
    fusion: fixed.FusionKind,
    weights: []f32,
};

pub const UnitKind = enum(u8) {
    consensus,
    unique,
    relation,
    uncertainty,
    separator,

    pub fn jsonName(self: UnitKind) []const u8 {
        return @tagName(self);
    }
};

pub const SemanticUnit = struct {
    unit_id: u32,
    kind: UnitKind,
    role_id: ?u32,
    entity_id: ?u32,
    consensus_group_index: ?u32 = null,
    source_span_index: ?u32 = null,
    alternatives: []u32 = &.{},
    reason: ?[]const u8 = null,
    order_class: OrderClass,
    first_occurrence: u32,
};

pub const Relation = struct {
    unit_id: u32,
    source_span_index: u32,
    from_entity_id: ?u32,
    to_entity_id: ?u32,
};

pub const Diagnostics = struct {
    original_token_count: u32,
    semantic_span_count: u32,
    output_unit_count: u32,
    compression_ratio: f32,
    candidate_pair_count: u64,
    accepted_pair_count: u64,
    rejected_pairs: []PairEvaluation,
};

pub const SemanticPlan = struct {
    schema_version: u32 = CcssSchemaVersion,
    caption_count: u32,
    consensus_groups: []ConsensusGroup,
    units: []SemanticUnit,
    relations: []Relation,
    diagnostics: Diagnostics,

    pub fn deinit(self: *SemanticPlan, allocator: std.mem.Allocator) void {
        for (self.consensus_groups) |group| {
            allocator.free(group.member_span_indices);
            allocator.free(group.member_caption_ids);
            allocator.free(group.weights);
        }
        for (self.units) |unit| {
            if (unit.alternatives.len > 0) allocator.free(unit.alternatives);
        }
        allocator.free(self.consensus_groups);
        allocator.free(self.units);
        allocator.free(self.relations);
        allocator.free(self.diagnostics.rejected_pairs);
        self.* = undefined;
    }
};

pub fn cosineSimilarity(left: []const f32, right: []const f32) !f32 {
    if (left.len == 0 or left.len != right.len) return error.DimensionMismatch;
    var dot: f64 = 0;
    var left_norm_sq: f64 = 0;
    var right_norm_sq: f64 = 0;
    for (left, right) |a, b| {
        if (!std.math.isFinite(a) or !std.math.isFinite(b))
            return error.InvalidEmbedding;
        dot += @as(f64, a) * @as(f64, b);
        left_norm_sq += @as(f64, a) * @as(f64, a);
        right_norm_sq += @as(f64, b) * @as(f64, b);
    }
    if (!(left_norm_sq > 0) or !(right_norm_sq > 0))
        return error.ZeroNormEmbedding;
    const result = dot / @sqrt(left_norm_sq * right_norm_sq);
    return @floatCast(std.math.clamp(result, -1.0, 1.0));
}

fn containsId(ids: []const u32, needle: u32) bool {
    for (ids) |id| if (id == needle) return true;
    return false;
}

fn contradictionAt(input: SemanticInput, left: usize, right: usize) f32 {
    var score: f32 = 0.0;
    if (input.contradictions) |matrix| {
        if (matrix[left * input.spans.len + right] or
            matrix[right * input.spans.len + left]) score = 1.0;
    }
    if (input.contradiction_scores) |matrix| {
        score = @max(score, @max(
            matrix[left * input.spans.len + right],
            matrix[right * input.spans.len + left],
        ));
    }
    return score;
}

fn thresholdFor(config: SemanticSuperpositionConfig, role_id: ?u32) f32 {
    var threshold = config.default_cosine_threshold;
    if (role_id) |role| {
        for (config.role_thresholds) |override| {
            if (override.role_id == role) {
                threshold = override.cosine_threshold;
                break;
            }
        }
        if (containsId(config.protected_role_ids, role))
            threshold = @max(threshold, config.protected_cosine_threshold);
    }
    return threshold;
}

fn knownCompatibility(comptime field: []const u8, left: SemanticSpan, right: SemanticSpan) f32 {
    const a = @field(left, field);
    const b = @field(right, field);
    if (a == null or b == null) return 0.0;
    return if (a.? == b.?) 1.0 else 0.0;
}

pub fn evaluatePair(
    input: SemanticInput,
    config: SemanticSuperpositionConfig,
    left_index: u32,
    right_index: u32,
) !PairEvaluation {
    if (left_index >= input.spans.len or right_index >= input.spans.len or
        left_index == right_index) return error.InvalidSpanIndex;
    const left = input.spans[left_index];
    const right = input.spans[right_index];
    const role_compat = knownCompatibility("role_id", left, right);
    const entity_compat = knownCompatibility("entity_id", left, right);
    const contradiction = contradictionAt(input, left_index, right_index);
    const cosine = try cosineSimilarity(
        try input.embeddings.row(left.embedding_index),
        try input.embeddings.row(right.embedding_index),
    );
    const grounding = if (input.grounding_embeddings) |matrix|
        cosineSimilarity(
            try matrix.row(left.embedding_index),
            try matrix.row(right.embedding_index),
        ) catch |err| switch (err) {
            error.ZeroNormEmbedding => 0.0,
            else => return err,
        }
    else
        0.0;
    const score =
        config.semantic_weight * cosine +
        config.role_weight * role_compat +
        config.entity_weight * entity_compat +
        config.grounding_weight * grounding -
        config.contradiction_weight * contradiction;
    const common_role = if (left.role_id != null and left.role_id == right.role_id)
        left.role_id
    else
        null;
    const threshold = thresholdFor(config, common_role);

    var rejection: RejectionReason = .none;
    if (left.caption_id == right.caption_id)
        rejection = .same_caption
    else if (config.require_role_match and
        left.role_id != null and right.role_id != null and left.role_id != right.role_id)
        rejection = .role_mismatch
    else if (config.require_entity_match and
        left.entity_id != null and right.entity_id != null and left.entity_id != right.entity_id)
        rejection = .entity_mismatch
    else if ((left.role_id != null and containsId(config.never_fuse_role_ids, left.role_id.?)) or
        (right.role_id != null and containsId(config.never_fuse_role_ids, right.role_id.?)))
        rejection = .protected_role
    else if (config.block_contradictions and
        contradiction >= config.contradiction_block_threshold)
        rejection = .explicit_contradiction
    else if (cosine < threshold)
        rejection = .cosine_below_threshold
    else if (score < config.minimum_score)
        rejection = .score_below_threshold;

    return .{
        .left_span_index = left_index,
        .right_span_index = right_index,
        .cosine_similarity = cosine,
        .grounding_similarity = grounding,
        .role_compatibility = role_compat,
        .entity_compatibility = entity_compat,
        .contradiction = contradiction,
        .score = score,
        .cosine_threshold = threshold,
        .accepted = rejection == .none,
        .rejection = rejection,
    };
}

fn pairHigher(_: void, left: PairEvaluation, right: PairEvaluation) bool {
    if (left.score != right.score) return left.score > right.score;
    if (left.cosine_similarity != right.cosine_similarity)
        return left.cosine_similarity > right.cosine_similarity;
    if (left.left_span_index != right.left_span_index)
        return left.left_span_index < right.left_span_index;
    return left.right_span_index < right.right_span_index;
}

fn validateInput(input: SemanticInput, config: SemanticSuperpositionConfig) !void {
    try input.embeddings.validate();
    if (input.grounding_embeddings) |matrix| {
        try matrix.validate();
        if (matrix.rows != input.embeddings.rows) return error.InvalidEmbeddingMatrix;
    }
    if (input.spans.len > std.math.maxInt(u32)) return error.TooManySpans;
    if (input.caption_count == 0 and input.spans.len != 0) return error.InvalidCaptionCount;
    for (input.spans) |span| {
        if (span.caption_id >= input.caption_count) return error.InvalidCaptionId;
        if (span.token_end < span.token_start or span.byte_end < span.byte_start)
            return error.InvalidSpan;
        if (!std.math.isFinite(span.confidence) or span.confidence < 0)
            return error.InvalidConfidence;
        _ = try input.embeddings.row(span.embedding_index);
    }
    if (input.contradictions) |matrix| {
        const expected = std.math.mul(usize, input.spans.len, input.spans.len) catch
            return error.TooManySpans;
        if (matrix.len != expected) return error.LengthMismatch;
    }
    if (input.contradiction_scores) |matrix| {
        const expected = std.math.mul(usize, input.spans.len, input.spans.len) catch
            return error.TooManySpans;
        if (matrix.len != expected) return error.LengthMismatch;
        for (matrix) |score| {
            if (!std.math.isFinite(score) or score < 0 or score > 1)
                return error.InvalidContradictionScore;
        }
    }
    if (input.caption_quality) |quality| {
        if (quality.len != input.caption_count) return error.LengthMismatch;
        try validateNonNegativeFinite(quality);
    }
    if (input.grounding_confidence) |confidence| {
        if (confidence.len != input.spans.len) return error.LengthMismatch;
        try validateNonNegativeFinite(confidence);
    }
    if (input.section_reliability) |reliability| {
        if (reliability.len != input.spans.len) return error.LengthMismatch;
        try validateNonNegativeFinite(reliability);
    }
    if (config.maximum_cluster_size == 0 or config.minimum_support_count == 0)
        return error.InvalidConfig;
    if (!std.math.isFinite(config.default_cosine_threshold) or
        config.default_cosine_threshold < -1 or
        config.default_cosine_threshold > 1 or
        !std.math.isFinite(config.minimum_score) or
        !std.math.isFinite(config.protected_cosine_threshold) or
        config.protected_cosine_threshold < -1 or
        config.protected_cosine_threshold > 1 or
        !std.math.isFinite(config.contradiction_block_threshold) or
        config.contradiction_block_threshold < 0 or
        config.contradiction_block_threshold > 1 or
        config.minimum_support_fraction < 0 or
        config.minimum_support_fraction > 1) return error.InvalidConfig;
    const score_weights = [_]f32{
        config.semantic_weight,
        config.role_weight,
        config.entity_weight,
        config.grounding_weight,
        config.contradiction_weight,
    };
    for (score_weights) |weight| {
        if (!std.math.isFinite(weight) or weight < 0) return error.InvalidConfig;
    }
    for (config.role_thresholds) |threshold| {
        if (!std.math.isFinite(threshold.cosine_threshold) or
            threshold.cosine_threshold < -1 or threshold.cosine_threshold > 1)
            return error.InvalidConfig;
    }
}

fn validateNonNegativeFinite(values: []const f32) !void {
    for (values) |value| {
        if (!std.math.isFinite(value) or value < 0) return error.InvalidWeight;
    }
}

fn clusterSize(cluster_of: []const u32, root: u32) usize {
    var size: usize = 0;
    for (cluster_of) |cluster| if (cluster == root) {
        size += 1;
    };
    return size;
}

fn canMergeClusters(
    input: SemanticInput,
    cluster_of: []const u32,
    pair_pass: []const bool,
    left_root: u32,
    right_root: u32,
    maximum_size: usize,
) RejectionReason {
    if (clusterSize(cluster_of, left_root) + clusterSize(cluster_of, right_root) > maximum_size)
        return .maximum_cluster_size;
    for (cluster_of, 0..) |left_cluster, left_index| {
        if (left_cluster != left_root) continue;
        for (cluster_of, 0..) |right_cluster, right_index| {
            if (right_cluster != right_root) continue;
            if (input.spans[left_index].caption_id == input.spans[right_index].caption_id)
                return .duplicate_caption_in_cluster;
            if (!pair_pass[left_index * input.spans.len + right_index])
                return .complete_link_failed;
        }
    }
    return .none;
}

fn unitKindForSpan(kind: SpanKind) UnitKind {
    return switch (kind) {
        .relation, .spatial_connector, .action => .relation,
        .uncertainty => .uncertainty,
        .separator => .separator,
        else => .unique,
    };
}

fn unitLess(_: void, left: SemanticUnit, right: SemanticUnit) bool {
    if (@intFromEnum(left.order_class) != @intFromEnum(right.order_class))
        return @intFromEnum(left.order_class) < @intFromEnum(right.order_class);
    if (left.first_occurrence != right.first_occurrence)
        return left.first_occurrence < right.first_occurrence;
    const left_role = left.role_id orelse std.math.maxInt(u32);
    const right_role = right.role_id orelse std.math.maxInt(u32);
    return left_role < right_role;
}

fn minimumSupport(config: SemanticSuperpositionConfig, caption_count: u32) usize {
    const fraction_count: usize = @intFromFloat(@ceil(
        config.minimum_support_fraction * @as(f32, @floatFromInt(caption_count)),
    ));
    return @max(config.minimum_support_count, fraction_count);
}

/// Conservative greedy complete-link CCSS reference implementation.
pub fn buildSemanticPlan(
    allocator: std.mem.Allocator,
    input: SemanticInput,
    config: SemanticSuperpositionConfig,
) !SemanticPlan {
    try validateInput(input, config);
    const span_count = input.spans.len;
    const matrix_len = std.math.mul(usize, span_count, span_count) catch
        return error.TooManySpans;
    const pair_pass = try allocator.alloc(bool, matrix_len);
    defer allocator.free(pair_pass);
    @memset(pair_pass, false);
    for (0..span_count) |i| pair_pass[i * span_count + i] = true;

    var accepted_pairs: std.ArrayList(PairEvaluation) = .empty;
    defer accepted_pairs.deinit(allocator);
    var rejected_pairs: std.ArrayList(PairEvaluation) = .empty;
    errdefer rejected_pairs.deinit(allocator);
    var candidate_count: u64 = 0;

    for (0..span_count) |left| {
        for (left + 1..span_count) |right| {
            const evaluation = try evaluatePair(input, config, @intCast(left), @intCast(right));
            if (evaluation.rejection == .same_caption) {
                if (config.verbose_diagnostics)
                    try rejected_pairs.append(allocator, evaluation);
                continue;
            }
            candidate_count += 1;
            if (evaluation.accepted) {
                pair_pass[left * span_count + right] = true;
                pair_pass[right * span_count + left] = true;
                try accepted_pairs.append(allocator, evaluation);
            } else if (config.verbose_diagnostics) {
                try rejected_pairs.append(allocator, evaluation);
            }
        }
    }
    std.mem.sort(PairEvaluation, accepted_pairs.items, {}, pairHigher);

    const cluster_of = try allocator.alloc(u32, span_count);
    defer allocator.free(cluster_of);
    for (cluster_of, 0..) |*cluster, index| cluster.* = @intCast(index);

    for (accepted_pairs.items) |pair| {
        const left_root = cluster_of[pair.left_span_index];
        const right_root = cluster_of[pair.right_span_index];
        if (left_root == right_root) continue;
        const rejection = canMergeClusters(
            input,
            cluster_of,
            pair_pass,
            left_root,
            right_root,
            config.maximum_cluster_size,
        );
        if (rejection != .none) {
            if (config.verbose_diagnostics) {
                var diagnostic = pair;
                diagnostic.accepted = false;
                diagnostic.rejection = rejection;
                try rejected_pairs.append(allocator, diagnostic);
            }
            continue;
        }
        const keep = @min(left_root, right_root);
        const discard = @max(left_root, right_root);
        for (cluster_of) |*cluster| {
            if (cluster.* == discard) cluster.* = keep;
        }
    }

    var groups: std.ArrayList(ConsensusGroup) = .empty;
    errdefer {
        for (groups.items) |group| {
            allocator.free(group.member_span_indices);
            allocator.free(group.member_caption_ids);
            allocator.free(group.weights);
        }
        groups.deinit(allocator);
    }
    const required_support = minimumSupport(config, input.caption_count);
    for (0..span_count) |root| {
        if (cluster_of[root] != root) continue;
        const size = clusterSize(cluster_of, @intCast(root));
        if (size < required_support) continue;
        const group = try makeConsensusGroup(
            allocator,
            input,
            config,
            cluster_of,
            @intCast(root),
            @intCast(groups.items.len),
        );
        groups.append(allocator, group) catch |err| {
            allocator.free(group.member_span_indices);
            allocator.free(group.member_caption_ids);
            allocator.free(group.weights);
            return err;
        };
    }

    const span_to_group = try allocator.alloc(?u32, span_count);
    defer allocator.free(span_to_group);
    @memset(span_to_group, null);
    for (groups.items, 0..) |group, group_index| {
        for (group.member_span_indices) |span_index|
            span_to_group[span_index] = @intCast(group_index);
    }

    var units: std.ArrayList(SemanticUnit) = .empty;
    errdefer {
        for (units.items) |unit| if (unit.alternatives.len > 0)
            allocator.free(unit.alternatives);
        units.deinit(allocator);
    }
    for (groups.items, 0..) |group, group_index| {
        var first: u32 = std.math.maxInt(u32);
        var order: OrderClass = .uncertainty_alternative;
        for (group.member_span_indices) |span_index| {
            first = @min(first, span_index);
            if (@intFromEnum(input.spans[span_index].order_class) < @intFromEnum(order))
                order = input.spans[span_index].order_class;
        }
        try units.append(allocator, .{
            .unit_id = 0,
            .kind = .consensus,
            .role_id = group.role_id,
            .entity_id = group.entity_id,
            .consensus_group_index = @intCast(group_index),
            .order_class = order,
            .first_occurrence = first,
        });
    }
    for (input.spans, 0..) |span, span_index| {
        if (span_to_group[span_index] != null) continue;
        try units.append(allocator, .{
            .unit_id = 0,
            .kind = unitKindForSpan(span.kind),
            .role_id = span.role_id,
            .entity_id = span.entity_id,
            .source_span_index = @intCast(span_index),
            .order_class = span.order_class,
            .first_occurrence = @intCast(span_index),
        });
    }
    std.mem.sort(SemanticUnit, units.items, {}, unitLess);
    for (units.items, 0..) |*unit, index| unit.unit_id = @intCast(index);

    const span_to_unit = try allocator.alloc(u32, span_count);
    defer allocator.free(span_to_unit);
    for (units.items) |unit| {
        if (unit.source_span_index) |span_index| span_to_unit[span_index] = unit.unit_id;
        if (unit.consensus_group_index) |group_index| {
            for (groups.items[group_index].member_span_indices) |span_index|
                span_to_unit[span_index] = unit.unit_id;
        }
    }

    for (0..span_count) |left| {
        for (left + 1..span_count) |right| {
            if (contradictionAt(input, left, right) < config.contradiction_block_threshold)
                continue;
            const left_unit = span_to_unit[left];
            const right_unit = span_to_unit[right];
            if (left_unit == right_unit or hasAlternativePair(units.items, left_unit, right_unit))
                continue;
            const alternatives = try allocator.dupe(u32, &.{ left_unit, right_unit });
            units.append(allocator, .{
                .unit_id = @intCast(units.items.len),
                .kind = .uncertainty,
                .role_id = if (input.spans[left].role_id == input.spans[right].role_id)
                    input.spans[left].role_id
                else
                    null,
                .entity_id = if (input.spans[left].entity_id == input.spans[right].entity_id)
                    input.spans[left].entity_id
                else
                    null,
                .alternatives = alternatives,
                .reason = "explicit contradiction",
                .order_class = .uncertainty_alternative,
                .first_occurrence = @intCast(@min(left, right)),
            }) catch |err| {
                allocator.free(alternatives);
                return err;
            };
        }
    }

    var relations: std.ArrayList(Relation) = .empty;
    errdefer relations.deinit(allocator);
    for (input.spans, 0..) |span, span_index| {
        const unit_id = span_to_unit[span_index];
        if (unitKindForSpan(span.kind) != .relation) continue;
        try relations.append(allocator, .{
            .unit_id = unit_id,
            .source_span_index = @intCast(span_index),
            .from_entity_id = span.relation_from_entity_id,
            .to_entity_id = span.relation_to_entity_id,
        });
    }

    const owned_groups = try groups.toOwnedSlice(allocator);
    errdefer {
        for (owned_groups) |group| {
            allocator.free(group.member_span_indices);
            allocator.free(group.member_caption_ids);
            allocator.free(group.weights);
        }
        allocator.free(owned_groups);
    }
    const owned_units = try units.toOwnedSlice(allocator);
    errdefer {
        for (owned_units) |unit| if (unit.alternatives.len > 0)
            allocator.free(unit.alternatives);
        allocator.free(owned_units);
    }
    const owned_relations = try relations.toOwnedSlice(allocator);
    errdefer allocator.free(owned_relations);
    const owned_rejections = try rejected_pairs.toOwnedSlice(allocator);
    errdefer allocator.free(owned_rejections);

    return .{
        .caption_count = input.caption_count,
        .consensus_groups = owned_groups,
        .units = owned_units,
        .relations = owned_relations,
        .diagnostics = .{
            .original_token_count = input.original_token_count,
            .semantic_span_count = @intCast(span_count),
            .output_unit_count = @intCast(owned_units.len),
            .compression_ratio = if (input.original_token_count == 0)
                0.0
            else
                @as(f32, @floatFromInt(owned_units.len)) /
                    @as(f32, @floatFromInt(input.original_token_count)),
            .candidate_pair_count = candidate_count,
            .accepted_pair_count = @intCast(accepted_pairs.items.len),
            .rejected_pairs = owned_rejections,
        },
    };
}

fn makeConsensusGroup(
    allocator: std.mem.Allocator,
    input: SemanticInput,
    config: SemanticSuperpositionConfig,
    cluster_of: []const u32,
    root: u32,
    group_id: u32,
) !ConsensusGroup {
    const size = clusterSize(cluster_of, root);
    const members = try allocator.alloc(u32, size);
    errdefer allocator.free(members);
    const captions = try allocator.alloc(u32, size);
    errdefer allocator.free(captions);
    const weights = try allocator.alloc(f32, size);
    errdefer allocator.free(weights);
    var write_index: usize = 0;
    for (cluster_of, 0..) |cluster, span_index| {
        if (cluster != root) continue;
        members[write_index] = @intCast(span_index);
        captions[write_index] = input.spans[span_index].caption_id;
        write_index += 1;
    }

    var pair_sum: f64 = 0;
    var pair_count: usize = 0;
    var minimum: f32 = 1.0;
    for (members, 0..) |left, i| {
        for (members[i + 1 ..]) |right| {
            const similarity = try cosineSimilarity(
                try input.embeddings.row(input.spans[left].embedding_index),
                try input.embeddings.row(input.spans[right].embedding_index),
            );
            pair_sum += similarity;
            pair_count += 1;
            minimum = @min(minimum, similarity);
        }
    }
    const mean: f32 = if (pair_count == 0) 1.0 else @floatCast(pair_sum / @as(f64, @floatFromInt(pair_count)));

    var weight_sum: f64 = 0;
    for (members, 0..) |span_index, i| {
        const span = input.spans[span_index];
        var centrality: f32 = 1.0;
        if (members.len > 1) {
            var sum: f64 = 0;
            for (members) |other| {
                if (other == span_index) continue;
                sum += try cosineSimilarity(
                    try input.embeddings.row(span.embedding_index),
                    try input.embeddings.row(input.spans[other].embedding_index),
                );
            }
            centrality = @max(0.0, @as(f32, @floatCast(
                sum / @as(f64, @floatFromInt(members.len - 1)),
            )));
        }
        var raw = span.confidence * centrality;
        if (input.caption_quality) |quality| raw *= quality[span.caption_id];
        if (input.grounding_confidence) |confidence| raw *= confidence[span_index];
        if (input.section_reliability) |reliability| raw *= reliability[span_index];
        weights[i] = raw;
        weight_sum += raw;
    }
    if (!(weight_sum > 0)) {
        @memset(weights, 1.0 / @as(f32, @floatFromInt(weights.len)));
    } else {
        for (weights) |*weight| weight.* = @floatCast(@as(f64, weight.*) / weight_sum);
    }

    const first = input.spans[members[0]];
    return .{
        .group_id = group_id,
        .role_id = first.role_id,
        .entity_id = first.entity_id,
        .member_span_indices = members,
        .member_caption_ids = captions,
        .support_count = @intCast(size),
        .support_fraction = @as(f32, @floatFromInt(size)) /
            @as(f32, @floatFromInt(input.caption_count)),
        .mean_similarity = mean,
        .minimum_pair_similarity = minimum,
        .dispersion = 1.0 - mean,
        .fusion = config.fusion,
        .weights = weights,
    };
}

fn hasAlternativePair(units: []const SemanticUnit, left: u32, right: u32) bool {
    const low = @min(left, right);
    const high = @max(left, right);
    for (units) |unit| {
        if (unit.alternatives.len != 2) continue;
        if (@min(unit.alternatives[0], unit.alternatives[1]) == low and
            @max(unit.alternatives[0], unit.alternatives[1]) == high) return true;
    }
    return false;
}

fn writeJsonString(writer: *std.Io.Writer, value: []const u8) !void {
    try writer.writeByte('"');
    for (value) |byte| switch (byte) {
        '"' => try writer.writeAll("\\\""),
        '\\' => try writer.writeAll("\\\\"),
        '\n' => try writer.writeAll("\\n"),
        '\r' => try writer.writeAll("\\r"),
        '\t' => try writer.writeAll("\\t"),
        0...8, 11...12, 14...0x1f => try writer.print("\\u00{x:0>2}", .{byte}),
        else => try writer.writeByte(byte),
    };
    try writer.writeByte('"');
}

fn writeOptionalU32(writer: *std.Io.Writer, value: ?u32) !void {
    if (value) |number| try writer.print("{d}", .{number}) else try writer.writeAll("null");
}

fn roleNameFor(input: SemanticInput, role_id: ?u32) ?[]const u8 {
    const id = role_id orelse return null;
    for (input.spans) |span| {
        if (span.role_id != null and span.role_id.? == id and span.role_name != null)
            return span.role_name;
    }
    return null;
}

fn entityNameFor(input: SemanticInput, entity_id: ?u32) ?[]const u8 {
    const id = entity_id orelse return null;
    for (input.spans) |span| {
        if (span.entity_id != null and span.entity_id.? == id and span.entity_name != null)
            return span.entity_name;
    }
    return null;
}

pub fn writeJson(
    writer: *std.Io.Writer,
    input: SemanticInput,
    plan: *const SemanticPlan,
) !void {
    try writer.writeAll("{\"schema\":\"");
    try writer.writeAll(CcssSchema);
    try writer.writeAll("\",\"schema_version\":1,\"image_id\":");
    try writeJsonString(writer, input.image_id);
    try writer.print(",\"caption_count\":{d},\"units\":[", .{plan.caption_count});
    for (plan.units, 0..) |unit, index| {
        if (index != 0) try writer.writeByte(',');
        try writer.print("{{\"unit_id\":{d},\"kind\":\"{s}\",\"role_id\":", .{
            unit.unit_id,
            unit.kind.jsonName(),
        });
        try writeOptionalU32(writer, unit.role_id);
        try writer.writeAll(",\"role\":");
        if (roleNameFor(input, unit.role_id)) |name|
            try writeJsonString(writer, name)
        else
            try writer.writeAll("null");
        try writer.writeAll(",\"entity_id\":");
        try writeOptionalU32(writer, unit.entity_id);
        try writer.writeAll(",\"entity\":");
        if (entityNameFor(input, unit.entity_id)) |name|
            try writeJsonString(writer, name)
        else
            try writer.writeAll("null");
        if (unit.consensus_group_index) |group_index| {
            const group = plan.consensus_groups[group_index];
            try writer.writeAll(",\"members\":[");
            for (group.member_span_indices, 0..) |span_index, member_index| {
                if (member_index != 0) try writer.writeByte(',');
                try writeJsonString(writer, input.spans[span_index].span_id);
            }
            try writer.writeAll("],\"weights\":[");
            for (group.weights, 0..) |weight, weight_index| {
                if (weight_index != 0) try writer.writeByte(',');
                try writer.print("{d}", .{weight});
            }
            try writer.writeAll("],\"member_caption_ids\":[");
            for (group.member_caption_ids, 0..) |caption_id, caption_index| {
                if (caption_index != 0) try writer.writeByte(',');
                try writer.print("{d}", .{caption_id});
            }
            try writer.writeAll("],\"member_sections\":[");
            for (group.member_span_indices, 0..) |span_index, section_index| {
                if (section_index != 0) try writer.writeByte(',');
                if (input.spans[span_index].section_name) |section|
                    try writeJsonString(writer, section)
                else
                    try writer.writeAll("null");
            }
            try writer.print(
                "],\"support_count\":{d},\"support_fraction\":{d},\"mean_similarity\":{d},\"minimum_pair_similarity\":{d},\"dispersion\":{d},\"fusion\":\"{s}\",\"reason\":\"complete_link_similarity_and_constraints\"",
                .{
                    group.support_count,
                    group.support_fraction,
                    group.mean_similarity,
                    group.minimum_pair_similarity,
                    group.dispersion,
                    group.fusion.jsonName(),
                },
            );
        } else if (unit.source_span_index) |span_index| {
            try writer.writeAll(",\"span_id\":");
            try writeJsonString(writer, input.spans[span_index].span_id);
            try writer.writeAll(",\"section\":");
            if (input.spans[span_index].section_name) |section|
                try writeJsonString(writer, section)
            else
                try writer.writeAll("null");
            try writer.print(
                ",\"caption_id\":{d},\"token_start\":{d},\"token_end\":{d},\"byte_start\":{d},\"byte_end\":{d},\"confidence\":{d},\"text\":",
                .{
                    input.spans[span_index].caption_id,
                    input.spans[span_index].token_start,
                    input.spans[span_index].token_end,
                    input.spans[span_index].byte_start,
                    input.spans[span_index].byte_end,
                    input.spans[span_index].confidence,
                },
            );
            try writeJsonString(writer, input.spans[span_index].text);
        }
        if (unit.alternatives.len > 0) {
            try writer.writeAll(",\"alternatives\":[");
            for (unit.alternatives, 0..) |alternative, alternative_index| {
                if (alternative_index != 0) try writer.writeByte(',');
                try writer.print("{d}", .{alternative});
            }
            try writer.writeByte(']');
        }
        if (unit.reason) |reason| {
            try writer.writeAll(",\"reason\":");
            try writeJsonString(writer, reason);
        }
        try writer.writeByte('}');
    }
    try writer.writeAll("],\"relations\":[");
    for (plan.relations, 0..) |relation, index| {
        if (index != 0) try writer.writeByte(',');
        try writer.print("{{\"unit_id\":{d},\"span_id\":", .{relation.unit_id});
        try writeJsonString(writer, input.spans[relation.source_span_index].span_id);
        try writer.writeAll(",\"from_entity_id\":");
        try writeOptionalU32(writer, relation.from_entity_id);
        try writer.writeAll(",\"from_entity\":");
        if (entityNameFor(input, relation.from_entity_id)) |name|
            try writeJsonString(writer, name)
        else
            try writer.writeAll("null");
        try writer.writeAll(",\"to_entity_id\":");
        try writeOptionalU32(writer, relation.to_entity_id);
        try writer.writeAll(",\"to_entity\":");
        if (entityNameFor(input, relation.to_entity_id)) |name|
            try writeJsonString(writer, name)
        else
            try writer.writeAll("null");
        try writer.writeByte('}');
    }
    try writer.print(
        "],\"diagnostics\":{{\"original_token_count\":{d},\"semantic_span_count\":{d},\"output_unit_count\":{d},\"compression_ratio\":{d},\"candidate_pair_count\":{d},\"accepted_pair_count\":{d}",
        .{
            plan.diagnostics.original_token_count,
            plan.diagnostics.semantic_span_count,
            plan.diagnostics.output_unit_count,
            plan.diagnostics.compression_ratio,
            plan.diagnostics.candidate_pair_count,
            plan.diagnostics.accepted_pair_count,
        },
    );
    if (configIncludesRejected(plan)) {
        try writer.writeAll(",\"rejected_pairs\":[");
        for (plan.diagnostics.rejected_pairs, 0..) |pair, index| {
            if (index != 0) try writer.writeByte(',');
            try writer.writeAll("{\"left\":");
            try writeJsonString(writer, input.spans[pair.left_span_index].span_id);
            try writer.writeAll(",\"right\":");
            try writeJsonString(writer, input.spans[pair.right_span_index].span_id);
            try writer.print(
                ",\"cosine_similarity\":{d},\"score\":{d},\"threshold\":{d},\"reason\":\"{s}\"}}",
                .{
                    pair.cosine_similarity,
                    pair.score,
                    pair.cosine_threshold,
                    pair.rejection.jsonName(),
                },
            );
        }
        try writer.writeByte(']');
    }
    try writer.writeAll("}}");
}

fn configIncludesRejected(plan: *const SemanticPlan) bool {
    return plan.diagnostics.rejected_pairs.len > 0;
}

pub fn toJsonAlloc(
    allocator: std.mem.Allocator,
    input: SemanticInput,
    plan: *const SemanticPlan,
) ![]u8 {
    var output: std.Io.Writer.Allocating = .init(allocator);
    errdefer output.deinit();
    try writeJson(&output.writer, input, plan);
    return output.toOwnedSlice();
}

test "cosine similarity validates dimensions and zero vectors" {
    try std.testing.expectApproxEqAbs(
        @as(f32, 1.0),
        try cosineSimilarity(&.{ 1, 2 }, &.{ 2, 4 }),
        0.00001,
    );
    try std.testing.expectError(
        error.DimensionMismatch,
        cosineSimilarity(&.{1}, &.{ 1, 2 }),
    );
    try std.testing.expectError(
        error.ZeroNormEmbedding,
        cosineSimilarity(&.{ 0, 0 }, &.{ 1, 0 }),
    );
}

test "byte span maps whole Unicode and multi-token phrase" {
    const offsets = [_]TokenOffset{
        .{ .start = 0, .end = 1 },
        .{ .start = 2, .end = 4 },
        .{ .start = 4, .end = 6 },
        .{ .start = 7, .end = 12 },
    };
    const range = try mapByteSpanToTokenRange(&offsets, 2, 6);
    try std.testing.expectEqual(@as(u32, 1), range.token_start);
    try std.testing.expectEqual(@as(u32, 3), range.token_end);
    try std.testing.expectError(
        error.ByteSpanNotCovered,
        mapByteSpanToTokenRange(&offsets, 6, 7),
    );
}

test "CCSS empty and one-span inputs remain deterministic unique plans" {
    const empty_input: SemanticInput = .{
        .image_id = "empty",
        .caption_count = 0,
        .original_token_count = 0,
        .spans = &.{},
        .embeddings = .{ .values = &.{}, .rows = 0, .dimensions = 1 },
    };
    var empty = try buildSemanticPlan(std.testing.allocator, empty_input, .{});
    defer empty.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 0), empty.units.len);

    const spans = [_]SemanticSpan{.{
        .span_id = "only",
        .caption_id = 0,
        .token_start = 0,
        .token_end = 1,
        .byte_start = 0,
        .byte_end = 4,
        .embedding_index = 0,
        .kind = .detail,
    }};
    const input: SemanticInput = .{
        .image_id = "one",
        .caption_count = 1,
        .original_token_count = 1,
        .spans = &spans,
        .embeddings = .{ .values = &.{ 1, 0 }, .rows = 1, .dimensions = 2 },
    };
    var plan = try buildSemanticPlan(std.testing.allocator, input, .{});
    defer plan.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 1), plan.units.len);
    try std.testing.expectEqual(UnitKind.unique, plan.units[0].kind);
    const json = try toJsonAlloc(std.testing.allocator, input, &plan);
    defer std.testing.allocator.free(json);
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json, .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings(CcssSchema, parsed.value.object.get("schema").?.string);
}

test "CCSS hard constraints reject same caption role entity contradiction and protected roles" {
    const spans = [_]SemanticSpan{
        .{ .span_id = "a", .caption_id = 0, .token_start = 0, .token_end = 1, .byte_start = 0, .byte_end = 1, .role_id = 1, .entity_id = 1, .embedding_index = 0 },
        .{ .span_id = "b", .caption_id = 0, .token_start = 1, .token_end = 2, .byte_start = 2, .byte_end = 3, .role_id = 1, .entity_id = 1, .embedding_index = 1 },
        .{ .span_id = "c", .caption_id = 1, .token_start = 0, .token_end = 1, .byte_start = 0, .byte_end = 1, .role_id = 2, .entity_id = 1, .embedding_index = 2 },
        .{ .span_id = "d", .caption_id = 2, .token_start = 0, .token_end = 1, .byte_start = 0, .byte_end = 1, .role_id = 1, .entity_id = 2, .embedding_index = 3 },
        .{ .span_id = "e", .caption_id = 3, .token_start = 0, .token_end = 1, .byte_start = 0, .byte_end = 1, .role_id = 1, .entity_id = 1, .embedding_index = 4 },
        .{ .span_id = "count", .caption_id = 4, .token_start = 0, .token_end = 1, .byte_start = 0, .byte_end = 1, .role_id = 9, .entity_id = 1, .embedding_index = 5 },
        .{ .span_id = "count2", .caption_id = 5, .token_start = 0, .token_end = 1, .byte_start = 0, .byte_end = 1, .role_id = 9, .entity_id = 1, .embedding_index = 6 },
    };
    const embeddings = [_]f32{
        1, 0,
        1, 0,
        1, 0,
        1, 0,
        1, 0,
        1, 0,
        1, 0,
    };
    var contradictions = [_]bool{false} ** (spans.len * spans.len);
    contradictions[0 * spans.len + 4] = true;
    contradictions[4 * spans.len + 0] = true;
    const input: SemanticInput = .{
        .image_id = "constraints",
        .caption_count = 6,
        .original_token_count = 7,
        .spans = &spans,
        .embeddings = .{ .values = &embeddings, .rows = spans.len, .dimensions = 2 },
        .contradictions = &contradictions,
    };
    const config: SemanticSuperpositionConfig = .{ .never_fuse_role_ids = &.{9} };

    try std.testing.expectEqual(RejectionReason.same_caption, (try evaluatePair(input, config, 0, 1)).rejection);
    try std.testing.expectEqual(RejectionReason.role_mismatch, (try evaluatePair(input, config, 0, 2)).rejection);
    try std.testing.expectEqual(RejectionReason.entity_mismatch, (try evaluatePair(input, config, 0, 3)).rejection);
    try std.testing.expectEqual(RejectionReason.explicit_contradiction, (try evaluatePair(input, config, 0, 4)).rejection);
    try std.testing.expectEqual(RejectionReason.protected_role, (try evaluatePair(input, config, 5, 6)).rejection);
}

test "CCSS caller contradiction scores contribute to score and hard blocking" {
    const spans = [_]SemanticSpan{
        .{ .span_id = "warm", .caption_id = 0, .token_start = 0, .token_end = 1, .byte_start = 0, .byte_end = 4, .role_id = 1, .embedding_index = 0 },
        .{ .span_id = "cool", .caption_id = 1, .token_start = 0, .token_end = 1, .byte_start = 0, .byte_end = 4, .role_id = 1, .embedding_index = 1 },
    };
    const scores = [_]f32{ 0, 0.7, 0.7, 0 };
    const input: SemanticInput = .{
        .image_id = "scores",
        .caption_count = 2,
        .original_token_count = 2,
        .spans = &spans,
        .embeddings = .{ .values = &.{ 1, 0, 1, 0 }, .rows = 2, .dimensions = 2 },
        .contradiction_scores = &scores,
    };
    const evaluation = try evaluatePair(input, .{
        .contradiction_weight = 0.5,
        .contradiction_block_threshold = 0.6,
    }, 0, 1);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7), evaluation.contradiction, 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.65), evaluation.score, 0.00001);
    try std.testing.expectEqual(RejectionReason.explicit_contradiction, evaluation.rejection);
}

test "CCSS complete-link prevents semantic chaining" {
    const spans = [_]SemanticSpan{
        .{ .span_id = "bright", .caption_id = 0, .token_start = 0, .token_end = 1, .byte_start = 0, .byte_end = 6, .role_id = 1, .entity_id = 1, .embedding_index = 0 },
        .{ .span_id = "luminous", .caption_id = 1, .token_start = 0, .token_end = 1, .byte_start = 0, .byte_end = 8, .role_id = 1, .entity_id = 1, .embedding_index = 1 },
        .{ .span_id = "white", .caption_id = 2, .token_start = 0, .token_end = 1, .byte_start = 0, .byte_end = 5, .role_id = 1, .entity_id = 1, .embedding_index = 2 },
    };
    // A~B and B~C pass 0.75, while A~C does not.
    const embeddings = [_]f32{
        1.0,  0.0,
        0.8,  0.6,
        0.28, 0.96,
    };
    const input: SemanticInput = .{
        .image_id = "chain",
        .caption_count = 3,
        .original_token_count = 3,
        .spans = &spans,
        .embeddings = .{ .values = &embeddings, .rows = 3, .dimensions = 2 },
    };
    var plan = try buildSemanticPlan(std.testing.allocator, input, .{
        .default_cosine_threshold = 0.75,
        .minimum_score = 0.75,
        .verbose_diagnostics = true,
    });
    defer plan.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 1), plan.consensus_groups.len);
    try std.testing.expectEqual(@as(u16, 2), plan.consensus_groups[0].support_count);
    try std.testing.expectEqual(@as(usize, 2), plan.units.len);
    var found_complete_link_rejection = false;
    for (plan.diagnostics.rejected_pairs) |pair| {
        if (pair.rejection == .complete_link_failed) found_complete_link_rejection = true;
    }
    try std.testing.expect(found_complete_link_rejection);
}

test "CCSS synthetic captions preserve roles entities relations and contradictions" {
    const intensity: u32 = 1;
    const color: u32 = 2;
    const emission: u32 = 3;
    const posture: u32 = 4;
    const direction: u32 = 5;
    const count: u32 = 6;
    const phrase: u32 = 7;
    const light: u32 = 10;
    const person: u32 = 11;
    const room: u32 = 12;
    const dress: u32 = 13;
    const spans = [_]SemanticSpan{
        .{ .span_id = "bright", .text = "bright", .caption_id = 0, .token_start = 0, .token_end = 1, .byte_start = 0, .byte_end = 6, .role_id = intensity, .entity_id = light, .embedding_index = 0, .kind = .attribute, .order_class = .lighting_appearance },
        .{ .span_id = "brilliant", .text = "brilliant", .caption_id = 1, .token_start = 0, .token_end = 1, .byte_start = 0, .byte_end = 9, .role_id = intensity, .entity_id = light, .embedding_index = 1, .kind = .attribute, .order_class = .lighting_appearance },
        .{ .span_id = "intense", .text = "intense", .caption_id = 2, .token_start = 0, .token_end = 1, .byte_start = 0, .byte_end = 7, .role_id = intensity, .entity_id = light, .embedding_index = 2, .kind = .attribute, .order_class = .lighting_appearance },
        .{ .span_id = "white", .caption_id = 0, .token_start = 1, .token_end = 2, .byte_start = 7, .byte_end = 12, .role_id = color, .entity_id = light, .embedding_index = 3, .kind = .attribute },
        .{ .span_id = "pale", .caption_id = 2, .token_start = 1, .token_end = 2, .byte_start = 8, .byte_end = 12, .role_id = color, .entity_id = light, .embedding_index = 4, .kind = .attribute },
        .{ .span_id = "luminous", .caption_id = 0, .token_start = 2, .token_end = 3, .byte_start = 13, .byte_end = 21, .role_id = emission, .entity_id = light, .embedding_index = 5, .kind = .attribute },
        .{ .span_id = "glowing", .caption_id = 1, .token_start = 1, .token_end = 2, .byte_start = 10, .byte_end = 17, .role_id = emission, .entity_id = light, .embedding_index = 6, .kind = .attribute },
        .{ .span_id = "standing", .caption_id = 0, .token_start = 3, .token_end = 4, .byte_start = 22, .byte_end = 30, .role_id = posture, .entity_id = person, .embedding_index = 7, .kind = .action },
        .{ .span_id = "seated", .caption_id = 1, .token_start = 2, .token_end = 3, .byte_start = 18, .byte_end = 24, .role_id = posture, .entity_id = person, .embedding_index = 8, .kind = .action },
        .{ .span_id = "left", .caption_id = 0, .token_start = 4, .token_end = 5, .byte_start = 31, .byte_end = 35, .role_id = direction, .entity_id = person, .embedding_index = 9, .kind = .spatial_connector },
        .{ .span_id = "right", .caption_id = 1, .token_start = 3, .token_end = 4, .byte_start = 25, .byte_end = 30, .role_id = direction, .entity_id = person, .embedding_index = 10, .kind = .spatial_connector },
        .{ .span_id = "one-person", .caption_id = 0, .token_start = 5, .token_end = 7, .byte_start = 36, .byte_end = 46, .role_id = count, .entity_id = person, .embedding_index = 11, .kind = .quantifier },
        .{ .span_id = "two-people", .caption_id = 1, .token_start = 4, .token_end = 6, .byte_start = 31, .byte_end = 41, .role_id = count, .entity_id = person, .embedding_index = 12, .kind = .quantifier },
        .{ .span_id = "bright-room", .caption_id = 3, .token_start = 0, .token_end = 2, .byte_start = 0, .byte_end = 11, .role_id = intensity, .entity_id = room, .embedding_index = 13, .kind = .attribute },
        .{ .span_id = "bright-dress", .caption_id = 4, .token_start = 0, .token_end = 2, .byte_start = 0, .byte_end = 12, .role_id = intensity, .entity_id = dress, .embedding_index = 14, .kind = .attribute },
        .{ .span_id = "red-dress", .caption_id = 3, .token_start = 2, .token_end = 4, .byte_start = 12, .byte_end = 21, .role_id = color, .entity_id = dress, .embedding_index = 15, .kind = .attribute },
        .{ .span_id = "crimson-gown", .caption_id = 4, .token_start = 2, .token_end = 4, .byte_start = 13, .byte_end = 25, .role_id = color, .entity_id = dress, .embedding_index = 16, .kind = .attribute },
        .{ .span_id = "over-shoulder", .caption_id = 3, .token_start = 4, .token_end = 8, .byte_start = 22, .byte_end = 47, .role_id = phrase, .entity_id = person, .embedding_index = 17, .kind = .action },
        .{ .span_id = "glancing-backward", .caption_id = 4, .token_start = 4, .token_end = 6, .byte_start = 26, .byte_end = 44, .role_id = phrase, .entity_id = person, .embedding_index = 18, .kind = .action },
        .{ .span_id = "illuminates", .caption_id = 2, .token_start = 2, .token_end = 3, .byte_start = 13, .byte_end = 24, .role_id = 20, .entity_id = light, .embedding_index = 19, .kind = .relation, .order_class = .action_relation, .relation_from_entity_id = light, .relation_to_entity_id = room },
    };
    const embeddings = [_]f32{
        1.00,  0.00, 0.00, // intensity
        0.995, 0.05, 0.00,
        0.990, 0.08, 0.00,
        0.00, 1.00,  0.00, // white/pale
        0.04, 0.999, 0.00,
        0.00, 0.00, 1.00, // luminous/glowing
        0.02, 0.00, 0.999,
        0.70,  0.00, 0.70, // standing/seated distinct
        -0.70, 0.00, -0.70,
        0.60,  0.80,  0.00, // left/right protected
        -0.60, -0.80, 0.00,
        0.30,  0.40,  0.866, // counts protected
        -0.30, -0.40, -0.866,
        1.00, 0.00, 0.00, // bright but distinct entity
        1.00, 0.00, 0.00,
        0.20, 0.98,  0.00, // red/crimson
        0.22, 0.975, 0.00,
        0.45, 0.10, 0.887, // phrase paraphrases
        0.46, 0.11, 0.881,
        0.33, 0.33, 0.88, // relation singleton
    };
    var contradictions = [_]bool{false} ** (spans.len * spans.len);
    const contradiction_pairs = [_][2]usize{ .{ 7, 8 }, .{ 9, 10 }, .{ 11, 12 } };
    for (contradiction_pairs) |pair| {
        contradictions[pair[0] * spans.len + pair[1]] = true;
        contradictions[pair[1] * spans.len + pair[0]] = true;
    }
    const input: SemanticInput = .{
        .image_id = "synthetic",
        .caption_count = 5,
        .original_token_count = 80,
        .spans = &spans,
        .embeddings = .{ .values = &embeddings, .rows = spans.len, .dimensions = 3 },
        .contradictions = &contradictions,
    };
    var plan = try buildSemanticPlan(std.testing.allocator, input, .{
        .default_cosine_threshold = 0.94,
        .minimum_score = 0.94,
        .never_fuse_role_ids = &.{ direction, count },
        .verbose_diagnostics = true,
    });
    defer plan.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 5), plan.consensus_groups.len);
    try expectCluster(&plan, &.{ 0, 1, 2 });
    try expectCluster(&plan, &.{ 3, 4 });
    try expectCluster(&plan, &.{ 5, 6 });
    try expectCluster(&plan, &.{ 15, 16 });
    try expectCluster(&plan, &.{ 17, 18 });
    var found_illuminates = false;
    for (plan.relations) |relation| {
        if (relation.source_span_index == 19) {
            found_illuminates = true;
            try std.testing.expectEqual(light, relation.from_entity_id.?);
            try std.testing.expectEqual(room, relation.to_entity_id.?);
        }
    }
    try std.testing.expect(found_illuminates);

    var uncertainty_count: usize = 0;
    for (plan.units) |unit| if (unit.alternatives.len == 2) {
        uncertainty_count += 1;
    };
    try std.testing.expectEqual(@as(usize, 3), uncertainty_count);

    const json_a = try toJsonAlloc(std.testing.allocator, input, &plan);
    defer std.testing.allocator.free(json_a);
    const json_b = try toJsonAlloc(std.testing.allocator, input, &plan);
    defer std.testing.allocator.free(json_b);
    try std.testing.expectEqualStrings(json_a, json_b);
    try std.testing.expect(std.mem.indexOf(u8, json_a, "\"schema\":\"ztok.ccss.v1\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json_a, "\"span_id\":\"illuminates\"") != null);
}

fn expectCluster(plan: *const SemanticPlan, expected: []const u32) !void {
    for (plan.consensus_groups) |group| {
        if (std.mem.eql(u32, group.member_span_indices, expected)) return;
    }
    return error.ExpectedClusterNotFound;
}
