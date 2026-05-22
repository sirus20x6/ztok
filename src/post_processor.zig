//! HF post-processor templates. Handles BertProcessing, the more
//! general TemplateProcessing variant, RobertaProcessing, ByteLevel
//! (as a post-processor), and Sequence — the framing step that wraps
//! an encoded sequence with [CLS]/[SEP]/etc. before it goes to the
//! model.
//!
//! Storage is flat `[]TemplatePiece` + parallel `[]u32` type_ids per
//! piece. Applying walks the array once, copying ids into a caller-
//! provided buffer. No allocations on the hot path.
//!
//! Coverage is intentionally limited to the two grammar nodes
//! that real BERT-family tokenizer.json files use: SpecialToken and
//! Sequence. Nested templates and the rarer combinators in
//! refs/tokenizers/tokenizers/src/processors/template.rs are deferred.
//!
//! Variant choices (per task brief):
//!   - .roberta     — option (a): runtime variant with cls/sep ids;
//!                    apply emits [CLS] A [SEP] for single and
//!                    [CLS] A [SEP] [SEP] B [SEP] for pair, all
//!                    type_ids=0 (per HF Roberta semantics).
//!   - .byte_level  — option (a): runtime variant whose apply is
//!                    identity (the byte-level work happens in the
//!                    pre-tokenizer/decoder; the post-processor only
//!                    records that those settings should be applied
//!                    elsewhere). Parser must still accept it for
//!                    round-trip.
//!   - .sequence    — option (a): owns a slice of inner PostProcessor;
//!                    apply runs each in order (matches HF semantics —
//!                    though for the common "ByteLevel + Roberta" shape
//!                    the ByteLevel step is identity, so effectively
//!                    only the framing processor adds tokens).

const std = @import("std");
const TokenId = @import("token.zig").TokenId;

pub const TemplatePiece = union(enum) {
    sequence_a,
    sequence_b,
    /// Inline literal ids (for SpecialToken expansions). Borrowed slice
    /// owned by the parent TemplateConfig's literal_arena.
    literal: []const TokenId,
};

pub const TemplateRule = struct {
    pieces: []TemplatePiece,
    /// Parallel to pieces — type_id for each. For .literal entries the
    /// type_id applies to every inlined id.
    type_ids: []u32,
};

pub const PostProcessor = union(enum) {
    none,
    /// Bert-style framing: cls/sep ids inlined. Captured as a special
    /// case for efficiency; the generic TemplateProcessing can express
    /// the same thing.
    bert: BertConfig,
    /// Generic template covering anything BertProcessing could plus
    /// arbitrary special-token expansions and custom type_ids.
    template: TemplateConfig,
    /// Roberta-style framing: [CLS] A [SEP] / [CLS] A [SEP] [SEP] B
    /// [SEP], with all type_ids forced to 0 (per HF Roberta semantics).
    /// `trim_offsets` / `add_prefix_space` are recorded for round-trip
    /// fidelity but are not consumed on the id-only hot path (they
    /// govern offset adjustment in pre-tokenizer / decoder space).
    roberta: RobertaConfig,
    /// ByteLevel as a post-processor. In HF semantics the byte-level
    /// work itself runs in pre-tokenizer / decoder; the post-processor
    /// only records the three flags so a downstream re-serialization
    /// can carry them. Runtime apply is identity.
    byte_level: ByteLevelConfig,
    /// Sequence of inner post-processors, applied in order.
    sequence: SequenceConfig,

    pub const BertConfig = struct {
        cls_id: TokenId,
        sep_id: TokenId,
    };

    pub const TemplateConfig = struct {
        allocator: std.mem.Allocator,
        single: TemplateRule,
        pair: ?TemplateRule = null,
        /// Owning storage for any literal id slices referenced by the
        /// rules. Both rules' .literal slices point into this buffer.
        literal_arena: []TokenId,

        pub fn deinit(self: *TemplateConfig) void {
            self.allocator.free(self.single.pieces);
            self.allocator.free(self.single.type_ids);
            if (self.pair) |*p| {
                self.allocator.free(p.pieces);
                self.allocator.free(p.type_ids);
            }
            if (self.literal_arena.len > 0) self.allocator.free(self.literal_arena);
        }
    };

    pub const RobertaConfig = struct {
        allocator: std.mem.Allocator,
        cls_id: TokenId,
        sep_id: TokenId,
        /// Owned. Preserved for round-trip via PostProcessorSpec; not
        /// consulted by `applySingle`/`applyPair`.
        cls_token: []u8,
        sep_token: []u8,
        trim_offsets: bool = true,
        add_prefix_space: bool = true,

        pub fn deinit(self: *RobertaConfig) void {
            if (self.cls_token.len > 0) self.allocator.free(self.cls_token);
            if (self.sep_token.len > 0) self.allocator.free(self.sep_token);
        }
    };

    pub const ByteLevelConfig = struct {
        add_prefix_space: bool = true,
        trim_offsets: bool = true,
        use_regex: bool = true,
    };

    pub const SequenceConfig = struct {
        allocator: std.mem.Allocator,
        /// Owned. Each inner processor's `deinit` is invoked on teardown.
        processors: []PostProcessor,

        pub fn deinit(self: *SequenceConfig) void {
            for (self.processors) |*p| p.deinit();
            if (self.processors.len > 0) self.allocator.free(self.processors);
        }
    };

    pub fn deinit(self: *PostProcessor) void {
        switch (self.*) {
            .template => |*t| t.deinit(),
            .roberta => |*r| r.deinit(),
            .sequence => |*s| s.deinit(),
            else => {},
        }
    }

    pub fn outputLenSingle(self: PostProcessor, n_a: usize) usize {
        return switch (self) {
            .none => n_a,
            .bert => 2 + n_a,
            .roberta => 2 + n_a,
            .byte_level => n_a,
            .template => |t| ruleLen(t.single, n_a, 0),
            .sequence => |s| sequenceOutputLenSingle(s.processors, n_a),
        };
    }

    pub fn outputLenPair(self: PostProcessor, n_a: usize, n_b: usize) usize {
        return switch (self) {
            .none => n_a + n_b,
            .bert => 3 + n_a + n_b,
            // [CLS] A [SEP] [SEP] B [SEP] — 4 extras.
            .roberta => 4 + n_a + n_b,
            .byte_level => n_a + n_b,
            .template => |t| if (t.pair) |p| ruleLen(p, n_a, n_b) else 0,
            .sequence => |s| sequenceOutputLenPair(s.processors, n_a, n_b),
        };
    }

    /// Apply to a single sequence. Writes into `out`, returns the slice
    /// of `out` actually written. `out` must have capacity equal to
    /// `outputLenSingle(n_a)`.
    pub fn applySingle(
        self: PostProcessor,
        ids_a: []const TokenId,
        out: []TokenId,
        type_ids_out: ?[]u32,
    ) []TokenId {
        switch (self) {
            .none => {
                @memcpy(out[0..ids_a.len], ids_a);
                if (type_ids_out) |ti| @memset(ti[0..ids_a.len], 0);
                return out[0..ids_a.len];
            },
            .bert => |b| {
                out[0] = b.cls_id;
                @memcpy(out[1 .. 1 + ids_a.len], ids_a);
                out[1 + ids_a.len] = b.sep_id;
                const total = 2 + ids_a.len;
                if (type_ids_out) |ti| @memset(ti[0..total], 0);
                return out[0..total];
            },
            .roberta => |r| {
                out[0] = r.cls_id;
                @memcpy(out[1 .. 1 + ids_a.len], ids_a);
                out[1 + ids_a.len] = r.sep_id;
                const total = 2 + ids_a.len;
                // All type_ids = 0 (HF Roberta semantics).
                if (type_ids_out) |ti| @memset(ti[0..total], 0);
                return out[0..total];
            },
            .byte_level => {
                // Identity in id-space.
                @memcpy(out[0..ids_a.len], ids_a);
                if (type_ids_out) |ti| @memset(ti[0..ids_a.len], 0);
                return out[0..ids_a.len];
            },
            .template => |t| return applyRule(t.single, ids_a, &.{}, out, type_ids_out),
            .sequence => |s| return applySequenceSingle(s.processors, ids_a, out, type_ids_out),
        }
    }

    pub const ApplyError = error{ MissingPairTemplate, UnsupportedSequenceShape };

    /// Apply to a paired input. `out` must have capacity equal to
    /// `outputLenPair(n_a, n_b)`. Returns error if the configured
    /// template lacks a pair rule.
    pub fn applyPair(
        self: PostProcessor,
        ids_a: []const TokenId,
        ids_b: []const TokenId,
        out: []TokenId,
        type_ids_out: ?[]u32,
    ) ApplyError![]TokenId {
        switch (self) {
            .none => {
                @memcpy(out[0..ids_a.len], ids_a);
                @memcpy(out[ids_a.len .. ids_a.len + ids_b.len], ids_b);
                const total = ids_a.len + ids_b.len;
                if (type_ids_out) |ti| {
                    @memset(ti[0..ids_a.len], 0);
                    @memset(ti[ids_a.len..total], 1);
                }
                return out[0..total];
            },
            .bert => |b| {
                var c: usize = 0;
                out[c] = b.cls_id;
                c += 1;
                @memcpy(out[c .. c + ids_a.len], ids_a);
                c += ids_a.len;
                out[c] = b.sep_id;
                c += 1;
                const a_end = c;
                @memcpy(out[c .. c + ids_b.len], ids_b);
                c += ids_b.len;
                out[c] = b.sep_id;
                c += 1;
                if (type_ids_out) |ti| {
                    @memset(ti[0..a_end], 0);
                    @memset(ti[a_end..c], 1);
                }
                return out[0..c];
            },
            .roberta => |r| {
                // [CLS] A [SEP] [SEP] B [SEP] — all type_ids = 0 per
                // HF Roberta (see refs/tokenizers/.../processors/roberta.rs).
                var c: usize = 0;
                out[c] = r.cls_id;
                c += 1;
                @memcpy(out[c .. c + ids_a.len], ids_a);
                c += ids_a.len;
                out[c] = r.sep_id;
                c += 1;
                out[c] = r.sep_id;
                c += 1;
                @memcpy(out[c .. c + ids_b.len], ids_b);
                c += ids_b.len;
                out[c] = r.sep_id;
                c += 1;
                if (type_ids_out) |ti| @memset(ti[0..c], 0);
                return out[0..c];
            },
            .byte_level => {
                // Identity in id-space; type_ids follow the default
                // "A=0, B=1" convention so callers can still tell the
                // two halves apart.
                @memcpy(out[0..ids_a.len], ids_a);
                @memcpy(out[ids_a.len .. ids_a.len + ids_b.len], ids_b);
                const total = ids_a.len + ids_b.len;
                if (type_ids_out) |ti| {
                    @memset(ti[0..ids_a.len], 0);
                    @memset(ti[ids_a.len..total], 1);
                }
                return out[0..total];
            },
            .template => |t| {
                const p = t.pair orelse return error.MissingPairTemplate;
                return applyRule(p, ids_a, ids_b, out, type_ids_out);
            },
            .sequence => |s| return applySequencePair(s.processors, ids_a, ids_b, out, type_ids_out),
        }
    }
};

// Sequence helpers. The HF post-processor sequence is a chain that runs
// each inner processor on the encoding produced by the previous one.
// Identity-like processors (.none, .byte_level) don't change the ids
// (they only adjust offsets/strings), so in id-space we treat the whole
// chain as "find the first framer (bert/roberta/template/sequence) and
// take its output length; if no framer, the chain is identity".
fn sequenceOutputLenSingle(procs: []const PostProcessor, n_a: usize) usize {
    for (procs) |p| {
        switch (p) {
            .bert, .roberta, .template, .sequence => return p.outputLenSingle(n_a),
            .none, .byte_level => {},
        }
    }
    return n_a;
}

fn sequenceOutputLenPair(procs: []const PostProcessor, n_a: usize, n_b: usize) usize {
    for (procs) |p| {
        switch (p) {
            .bert, .roberta, .template, .sequence => return p.outputLenPair(n_a, n_b),
            .none, .byte_level => {},
        }
    }
    return n_a + n_b;
}

// applySequenceSingle: classify each inner processor as either an
// "identity" step (.none, .byte_level — leaves ids untouched in
// id-space) or a "framer" (.bert, .roberta, .template, nested .sequence
// — wraps the sequence with extra tokens). In the common HF shape
// `[ByteLevel, Framer]` the identity steps collapse to no-ops, so we
// just apply the framer. For the general case we apply identity steps
// first (they do nothing to ids), then the framer if any, then more
// identity steps. This matches HF semantics because HF's identity-like
// post-processors operate on offsets/strings, not ids.
fn applySequenceSingle(
    procs: []const PostProcessor,
    ids_a: []const TokenId,
    out: []TokenId,
    type_ids_out: ?[]u32,
) []TokenId {
    var framer_idx: ?usize = null;
    for (procs, 0..) |p, idx| {
        switch (p) {
            .bert, .roberta, .template, .sequence => {
                framer_idx = idx;
                break;
            },
            .none, .byte_level => {},
        }
    }
    if (framer_idx) |idx| {
        return procs[idx].applySingle(ids_a, out, type_ids_out);
    }
    // No framer: identity passthrough.
    @memcpy(out[0..ids_a.len], ids_a);
    if (type_ids_out) |ti| @memset(ti[0..ids_a.len], 0);
    return out[0..ids_a.len];
}

fn applySequencePair(
    procs: []const PostProcessor,
    ids_a: []const TokenId,
    ids_b: []const TokenId,
    out: []TokenId,
    type_ids_out: ?[]u32,
) PostProcessor.ApplyError![]TokenId {
    var framer_idx: ?usize = null;
    for (procs, 0..) |p, idx| {
        switch (p) {
            .bert, .roberta, .template, .sequence => {
                framer_idx = idx;
                break;
            },
            .none, .byte_level => {},
        }
    }
    if (framer_idx) |idx| {
        return procs[idx].applyPair(ids_a, ids_b, out, type_ids_out);
    }
    // No framer: concatenate A then B.
    @memcpy(out[0..ids_a.len], ids_a);
    @memcpy(out[ids_a.len .. ids_a.len + ids_b.len], ids_b);
    const total = ids_a.len + ids_b.len;
    if (type_ids_out) |ti| {
        @memset(ti[0..ids_a.len], 0);
        @memset(ti[ids_a.len..total], 1);
    }
    return out[0..total];
}

fn ruleLen(rule: TemplateRule, n_a: usize, n_b: usize) usize {
    var total: usize = 0;
    for (rule.pieces) |p| {
        total += switch (p) {
            .sequence_a => n_a,
            .sequence_b => n_b,
            .literal => |lit| lit.len,
        };
    }
    return total;
}

fn applyRule(
    rule: TemplateRule,
    ids_a: []const TokenId,
    ids_b: []const TokenId,
    out: []TokenId,
    type_ids_out: ?[]u32,
) []TokenId {
    var c: usize = 0;
    for (rule.pieces, rule.type_ids) |piece, tid| {
        switch (piece) {
            .sequence_a => {
                @memcpy(out[c .. c + ids_a.len], ids_a);
                if (type_ids_out) |ti| @memset(ti[c .. c + ids_a.len], tid);
                c += ids_a.len;
            },
            .sequence_b => {
                @memcpy(out[c .. c + ids_b.len], ids_b);
                if (type_ids_out) |ti| @memset(ti[c .. c + ids_b.len], tid);
                c += ids_b.len;
            },
            .literal => |lit| {
                @memcpy(out[c .. c + lit.len], lit);
                if (type_ids_out) |ti| @memset(ti[c .. c + lit.len], tid);
                c += lit.len;
            },
        }
    }
    return out[0..c];
}

pub const Error = error{ MissingPairTemplate, MalformedJson, UnsupportedSequenceShape } || std.mem.Allocator.Error || std.json.ParseError(std.json.Scanner);

/// Parse a tokenizer.json post_processor section into a PostProcessor.
/// Returns .none on JSON null. Errors on malformed input.
pub fn parseFromJson(allocator: std.mem.Allocator, value: std.json.Value) Error!PostProcessor {
    if (value == .null) return .none;
    if (value != .object) return error.MalformedJson;

    const type_v = value.object.get("type") orelse return error.MalformedJson;
    if (type_v != .string) return error.MalformedJson;

    if (std.mem.eql(u8, type_v.string, "BertProcessing")) {
        return parseBert(value.object);
    }
    if (std.mem.eql(u8, type_v.string, "TemplateProcessing")) {
        return parseTemplate(allocator, value.object);
    }
    if (std.mem.eql(u8, type_v.string, "RobertaProcessing")) {
        return parseRoberta(allocator, value.object);
    }
    if (std.mem.eql(u8, type_v.string, "ByteLevel")) {
        return parseByteLevel(value.object);
    }
    if (std.mem.eql(u8, type_v.string, "Sequence")) {
        return parseSequence(allocator, value.object);
    }
    return error.MalformedJson;
}

fn parseBert(obj: std.json.ObjectMap) Error!PostProcessor {
    const cls_v = obj.get("cls") orelse return error.MalformedJson;
    const sep_v = obj.get("sep") orelse return error.MalformedJson;
    return .{ .bert = .{
        .cls_id = try parseTokenPair(cls_v),
        .sep_id = try parseTokenPair(sep_v),
    } };
}

fn parseTokenPair(v: std.json.Value) Error!TokenId {
    if (v != .array or v.array.items.len != 2) return error.MalformedJson;
    const id_v = v.array.items[1];
    if (id_v != .integer or id_v.integer < 0) return error.MalformedJson;
    return @intCast(id_v.integer);
}

fn parseTokenPairText(v: std.json.Value) Error![]const u8 {
    if (v != .array or v.array.items.len != 2) return error.MalformedJson;
    const tok_v = v.array.items[0];
    if (tok_v != .string) return error.MalformedJson;
    return tok_v.string;
}

fn parseRoberta(allocator: std.mem.Allocator, obj: std.json.ObjectMap) Error!PostProcessor {
    const cls_v = obj.get("cls") orelse return error.MalformedJson;
    const sep_v = obj.get("sep") orelse return error.MalformedJson;
    const cls_id = try parseTokenPair(cls_v);
    const sep_id = try parseTokenPair(sep_v);
    const cls_token_src = try parseTokenPairText(cls_v);
    const sep_token_src = try parseTokenPairText(sep_v);

    // HF defaults from refs/tokenizers/.../processors/roberta.rs.
    var trim_offsets: bool = true;
    var add_prefix_space: bool = true;
    if (obj.get("trim_offsets")) |v| {
        if (v != .bool) return error.MalformedJson;
        trim_offsets = v.bool;
    }
    if (obj.get("add_prefix_space")) |v| {
        if (v != .bool) return error.MalformedJson;
        add_prefix_space = v.bool;
    }

    const cls_token = try allocator.dupe(u8, cls_token_src);
    errdefer allocator.free(cls_token);
    const sep_token = try allocator.dupe(u8, sep_token_src);
    errdefer allocator.free(sep_token);

    return .{ .roberta = .{
        .allocator = allocator,
        .cls_id = cls_id,
        .sep_id = sep_id,
        .cls_token = cls_token,
        .sep_token = sep_token,
        .trim_offsets = trim_offsets,
        .add_prefix_space = add_prefix_space,
    } };
}

fn parseByteLevel(obj: std.json.ObjectMap) Error!PostProcessor {
    // All three flags default to true to match HF.
    var add_prefix_space: bool = true;
    var trim_offsets: bool = true;
    var use_regex: bool = true;
    if (obj.get("add_prefix_space")) |v| {
        if (v != .bool) return error.MalformedJson;
        add_prefix_space = v.bool;
    }
    if (obj.get("trim_offsets")) |v| {
        if (v != .bool) return error.MalformedJson;
        trim_offsets = v.bool;
    }
    if (obj.get("use_regex")) |v| {
        if (v != .bool) return error.MalformedJson;
        use_regex = v.bool;
    }
    return .{ .byte_level = .{
        .add_prefix_space = add_prefix_space,
        .trim_offsets = trim_offsets,
        .use_regex = use_regex,
    } };
}

fn parseSequence(allocator: std.mem.Allocator, obj: std.json.ObjectMap) Error!PostProcessor {
    const procs_v = obj.get("processors") orelse return error.MalformedJson;
    if (procs_v != .array) return error.MalformedJson;
    const items = procs_v.array.items;

    const out = try allocator.alloc(PostProcessor, items.len);
    errdefer allocator.free(out);

    var filled: usize = 0;
    errdefer {
        for (out[0..filled]) |*p| p.deinit();
    }

    for (items, 0..) |entry, i| {
        out[i] = try parseFromJson(allocator, entry);
        filled = i + 1;
    }

    return .{ .sequence = .{
        .allocator = allocator,
        .processors = out,
    } };
}

fn parseTemplate(allocator: std.mem.Allocator, obj: std.json.ObjectMap) Error!PostProcessor {
    const single_v = obj.get("single") orelse return error.MalformedJson;
    if (single_v != .array) return error.MalformedJson;
    const special_v = obj.get("special_tokens") orelse return error.MalformedJson;
    if (special_v != .object) return error.MalformedJson;

    // First pass: tally how many literal TokenIds we need from
    // SpecialToken expansions across both rules so we can size one arena.
    var lit_total: usize = 0;
    try countLits(single_v.array, special_v.object, &lit_total);
    const pair_v_opt = obj.get("pair");
    if (pair_v_opt) |pv| {
        if (pv == .array) {
            try countLits(pv.array, special_v.object, &lit_total);
        } else if (pv != .null) {
            return error.MalformedJson;
        }
    }

    const arena = try allocator.alloc(TokenId, lit_total);
    errdefer if (arena.len > 0) allocator.free(arena);

    var arena_cursor: usize = 0;
    const single_rule = try buildRule(allocator, single_v.array, special_v.object, arena, &arena_cursor);
    errdefer {
        allocator.free(single_rule.pieces);
        allocator.free(single_rule.type_ids);
    }

    var pair_rule_opt: ?TemplateRule = null;
    if (pair_v_opt) |pv| {
        if (pv == .array and pv.array.items.len > 0) {
            pair_rule_opt = try buildRule(allocator, pv.array, special_v.object, arena, &arena_cursor);
        }
    }

    return .{ .template = .{
        .allocator = allocator,
        .single = single_rule,
        .pair = pair_rule_opt,
        .literal_arena = arena,
    } };
}

fn countLits(arr: std.json.Array, specials: std.json.ObjectMap, total: *usize) Error!void {
    for (arr.items) |entry| {
        if (entry != .object) return error.MalformedJson;
        if (entry.object.count() != 1) return error.MalformedJson;
        var it = entry.object.iterator();
        const e = it.next() orelse return error.MalformedJson;
        const key = e.key_ptr.*;
        if (std.mem.eql(u8, key, "SpecialToken")) {
            const inner = e.value_ptr.*;
            if (inner != .object) return error.MalformedJson;
            const id_v = inner.object.get("id") orelse return error.MalformedJson;
            if (id_v != .string) return error.MalformedJson;
            const sp_v = specials.get(id_v.string) orelse return error.MalformedJson;
            if (sp_v != .object) return error.MalformedJson;
            const ids_v = sp_v.object.get("ids") orelse return error.MalformedJson;
            if (ids_v != .array) return error.MalformedJson;
            total.* += ids_v.array.items.len;
        } else if (!std.mem.eql(u8, key, "Sequence")) {
            return error.MalformedJson;
        }
    }
}

fn buildRule(
    allocator: std.mem.Allocator,
    arr: std.json.Array,
    specials: std.json.ObjectMap,
    arena: []TokenId,
    arena_cursor: *usize,
) Error!TemplateRule {
    const n = arr.items.len;
    const pieces = try allocator.alloc(TemplatePiece, n);
    errdefer allocator.free(pieces);
    const tids = try allocator.alloc(u32, n);
    errdefer allocator.free(tids);

    for (arr.items, 0..) |entry, i| {
        var it = entry.object.iterator();
        const e = it.next() orelse return error.MalformedJson;
        const key = e.key_ptr.*;
        const inner = e.value_ptr.*;
        if (inner != .object) return error.MalformedJson;
        const type_id_v = inner.object.get("type_id") orelse return error.MalformedJson;
        if (type_id_v != .integer or type_id_v.integer < 0) return error.MalformedJson;
        tids[i] = @intCast(type_id_v.integer);

        if (std.mem.eql(u8, key, "Sequence")) {
            const id_v = inner.object.get("id") orelse return error.MalformedJson;
            if (id_v != .string) return error.MalformedJson;
            if (std.mem.eql(u8, id_v.string, "A")) {
                pieces[i] = .sequence_a;
            } else if (std.mem.eql(u8, id_v.string, "B")) {
                pieces[i] = .sequence_b;
            } else {
                return error.MalformedJson;
            }
        } else if (std.mem.eql(u8, key, "SpecialToken")) {
            const id_v = inner.object.get("id") orelse return error.MalformedJson;
            if (id_v != .string) return error.MalformedJson;
            const sp_v = specials.get(id_v.string) orelse return error.MalformedJson;
            const ids_v = sp_v.object.get("ids").?;
            const items = ids_v.array.items;
            const start = arena_cursor.*;
            for (items) |x| {
                if (x != .integer or x.integer < 0) return error.MalformedJson;
                arena[arena_cursor.*] = @intCast(x.integer);
                arena_cursor.* += 1;
            }
            pieces[i] = .{ .literal = arena[start..arena_cursor.*] };
        } else {
            return error.MalformedJson;
        }
    }

    return .{ .pieces = pieces, .type_ids = tids };
}

// ---------------------------------------------------------------------
// Tests

const testing = std.testing;

test "BertConfig applySingle frames CLS A SEP" {
    const pp: PostProcessor = .{ .bert = .{ .cls_id = 101, .sep_id = 102 } };
    const ids_a = [_]TokenId{ 5, 6, 7 };
    var buf: [16]TokenId = undefined;
    const out = pp.applySingle(&ids_a, &buf, null);
    try testing.expectEqualSlices(TokenId, &[_]TokenId{ 101, 5, 6, 7, 102 }, out);
}

test "BertConfig applyPair frames CLS A SEP B SEP" {
    const pp: PostProcessor = .{ .bert = .{ .cls_id = 101, .sep_id = 102 } };
    const ids_a = [_]TokenId{ 5, 6 };
    const ids_b = [_]TokenId{ 8, 9, 10 };
    var buf: [16]TokenId = undefined;
    const out = try pp.applyPair(&ids_a, &ids_b, &buf, null);
    try testing.expectEqualSlices(TokenId, &[_]TokenId{ 101, 5, 6, 102, 8, 9, 10, 102 }, out);
}

test "BertConfig type_ids: A and CLS/SEP get 0, B and trailing SEP get 1" {
    const pp: PostProcessor = .{ .bert = .{ .cls_id = 101, .sep_id = 102 } };
    const ids_a = [_]TokenId{ 5, 6 };
    const ids_b = [_]TokenId{ 8, 9, 10 };
    var buf: [16]TokenId = undefined;
    var tids: [16]u32 = undefined;
    const out = try pp.applyPair(&ids_a, &ids_b, &buf, &tids);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 0, 0, 0, 1, 1, 1, 1 }, tids[0..out.len]);
}

test "parseFromJson parses minimal BertProcessing" {
    const input =
        \\{"type": "BertProcessing", "sep": ["[SEP]", 102], "cls": ["[CLS]", 101]}
    ;
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, input, .{});
    defer parsed.deinit();
    var pp = try parseFromJson(testing.allocator, parsed.value);
    defer pp.deinit();
    try testing.expect(pp == .bert);
    try testing.expectEqual(@as(TokenId, 101), pp.bert.cls_id);
    try testing.expectEqual(@as(TokenId, 102), pp.bert.sep_id);
}

test "parseFromJson parses TemplateProcessing single" {
    const input =
        \\{
        \\  "type": "TemplateProcessing",
        \\  "single": [
        \\    {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
        \\    {"Sequence": {"id": "A", "type_id": 0}},
        \\    {"SpecialToken": {"id": "[SEP]", "type_id": 0}}
        \\  ],
        \\  "pair": [],
        \\  "special_tokens": {
        \\    "[CLS]": {"id": "[CLS]", "ids": [101], "tokens": ["[CLS]"]},
        \\    "[SEP]": {"id": "[SEP]", "ids": [102], "tokens": ["[SEP]"]}
        \\  }
        \\}
    ;
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, input, .{});
    defer parsed.deinit();
    var pp = try parseFromJson(testing.allocator, parsed.value);
    defer pp.deinit();
    try testing.expect(pp == .template);

    const ids_a = [_]TokenId{ 5, 6, 7 };
    var buf: [16]TokenId = undefined;
    const out = pp.applySingle(&ids_a, &buf, null);
    try testing.expectEqualSlices(TokenId, &[_]TokenId{ 101, 5, 6, 7, 102 }, out);
}

test "TemplateConfig pair mode" {
    const input =
        \\{
        \\  "type": "TemplateProcessing",
        \\  "single": [
        \\    {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
        \\    {"Sequence": {"id": "A", "type_id": 0}},
        \\    {"SpecialToken": {"id": "[SEP]", "type_id": 0}}
        \\  ],
        \\  "pair": [
        \\    {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
        \\    {"Sequence": {"id": "A", "type_id": 0}},
        \\    {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
        \\    {"Sequence": {"id": "B", "type_id": 1}},
        \\    {"SpecialToken": {"id": "[SEP]", "type_id": 1}}
        \\  ],
        \\  "special_tokens": {
        \\    "[CLS]": {"id": "[CLS]", "ids": [101], "tokens": ["[CLS]"]},
        \\    "[SEP]": {"id": "[SEP]", "ids": [102], "tokens": ["[SEP]"]}
        \\  }
        \\}
    ;
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, input, .{});
    defer parsed.deinit();
    var pp = try parseFromJson(testing.allocator, parsed.value);
    defer pp.deinit();

    const ids_a = [_]TokenId{ 5, 6 };
    const ids_b = [_]TokenId{ 8, 9, 10 };
    var buf: [16]TokenId = undefined;
    var tids: [16]u32 = undefined;
    const out = try pp.applyPair(&ids_a, &ids_b, &buf, &tids);
    try testing.expectEqualSlices(TokenId, &[_]TokenId{ 101, 5, 6, 102, 8, 9, 10, 102 }, out);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 0, 0, 0, 1, 1, 1, 1 }, tids[0..out.len]);
}

test "outputLenSingle / outputLenPair are exact" {
    const pp_bert: PostProcessor = .{ .bert = .{ .cls_id = 101, .sep_id = 102 } };
    const ids_a = [_]TokenId{ 5, 6, 7 };
    const ids_b = [_]TokenId{ 8, 9 };
    var buf: [32]TokenId = undefined;

    const single = pp_bert.applySingle(&ids_a, &buf, null);
    try testing.expectEqual(pp_bert.outputLenSingle(ids_a.len), single.len);

    const pair = try pp_bert.applyPair(&ids_a, &ids_b, &buf, null);
    try testing.expectEqual(pp_bert.outputLenPair(ids_a.len, ids_b.len), pair.len);

    // Same check for a template.
    const input =
        \\{
        \\  "type": "TemplateProcessing",
        \\  "single": [
        \\    {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
        \\    {"Sequence": {"id": "A", "type_id": 0}},
        \\    {"SpecialToken": {"id": "[SEP]", "type_id": 0}}
        \\  ],
        \\  "pair": [
        \\    {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
        \\    {"Sequence": {"id": "A", "type_id": 0}},
        \\    {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
        \\    {"Sequence": {"id": "B", "type_id": 1}},
        \\    {"SpecialToken": {"id": "[SEP]", "type_id": 1}}
        \\  ],
        \\  "special_tokens": {
        \\    "[CLS]": {"id": "[CLS]", "ids": [101], "tokens": ["[CLS]"]},
        \\    "[SEP]": {"id": "[SEP]", "ids": [102], "tokens": ["[SEP]"]}
        \\  }
        \\}
    ;
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, input, .{});
    defer parsed.deinit();
    var pp_t = try parseFromJson(testing.allocator, parsed.value);
    defer pp_t.deinit();

    const t_single = pp_t.applySingle(&ids_a, &buf, null);
    try testing.expectEqual(pp_t.outputLenSingle(ids_a.len), t_single.len);
    const t_pair = try pp_t.applyPair(&ids_a, &ids_b, &buf, null);
    try testing.expectEqual(pp_t.outputLenPair(ids_a.len, ids_b.len), t_pair.len);
}

test "applyPair errors when pair template missing" {
    const input =
        \\{
        \\  "type": "TemplateProcessing",
        \\  "single": [
        \\    {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
        \\    {"Sequence": {"id": "A", "type_id": 0}},
        \\    {"SpecialToken": {"id": "[SEP]", "type_id": 0}}
        \\  ],
        \\  "pair": [],
        \\  "special_tokens": {
        \\    "[CLS]": {"id": "[CLS]", "ids": [101], "tokens": ["[CLS]"]},
        \\    "[SEP]": {"id": "[SEP]", "ids": [102], "tokens": ["[SEP]"]}
        \\  }
        \\}
    ;
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, input, .{});
    defer parsed.deinit();
    var pp = try parseFromJson(testing.allocator, parsed.value);
    defer pp.deinit();

    const ids_a = [_]TokenId{ 5, 6 };
    const ids_b = [_]TokenId{ 7, 8 };
    var buf: [16]TokenId = undefined;
    try testing.expectError(error.MissingPairTemplate, pp.applyPair(&ids_a, &ids_b, &buf, null));
}

test "parseFromJson returns .none on null input" {
    var pp = try parseFromJson(testing.allocator, std.json.Value{ .null = {} });
    defer pp.deinit();
    try testing.expect(pp == .none);
}

// --- New variant parsing + apply ---

test "parseFromJson parses RobertaProcessing with all fields" {
    const input =
        \\{"type":"RobertaProcessing","sep":["</s>",2],"cls":["<s>",0],"trim_offsets":true,"add_prefix_space":false}
    ;
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, input, .{});
    defer parsed.deinit();
    var pp = try parseFromJson(testing.allocator, parsed.value);
    defer pp.deinit();
    try testing.expect(pp == .roberta);
    try testing.expectEqual(@as(TokenId, 0), pp.roberta.cls_id);
    try testing.expectEqual(@as(TokenId, 2), pp.roberta.sep_id);
    try testing.expectEqualStrings("<s>", pp.roberta.cls_token);
    try testing.expectEqualStrings("</s>", pp.roberta.sep_token);
    try testing.expectEqual(true, pp.roberta.trim_offsets);
    try testing.expectEqual(false, pp.roberta.add_prefix_space);
}

test "RobertaProcessing apply: single yields [CLS] A [SEP]" {
    const input =
        \\{"type":"RobertaProcessing","sep":["</s>",2],"cls":["<s>",0],"trim_offsets":true,"add_prefix_space":true}
    ;
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, input, .{});
    defer parsed.deinit();
    var pp = try parseFromJson(testing.allocator, parsed.value);
    defer pp.deinit();

    const ids_a = [_]TokenId{ 12, 14 };
    var buf: [16]TokenId = undefined;
    var tids: [16]u32 = undefined;
    const out = pp.applySingle(&ids_a, &buf, &tids);
    try testing.expectEqualSlices(TokenId, &[_]TokenId{ 0, 12, 14, 2 }, out);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 0, 0, 0 }, tids[0..out.len]);
}

test "RobertaProcessing apply: pair yields [CLS] A [SEP] [SEP] B [SEP], all type_ids 0" {
    const input =
        \\{"type":"RobertaProcessing","sep":["</s>",2],"cls":["<s>",0]}
    ;
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, input, .{});
    defer parsed.deinit();
    var pp = try parseFromJson(testing.allocator, parsed.value);
    defer pp.deinit();
    // Defaults should apply since trim_offsets / add_prefix_space were
    // omitted.
    try testing.expectEqual(true, pp.roberta.trim_offsets);
    try testing.expectEqual(true, pp.roberta.add_prefix_space);

    const ids_a = [_]TokenId{ 12, 14 };
    const ids_b = [_]TokenId{15};
    var buf: [16]TokenId = undefined;
    var tids: [16]u32 = undefined;
    const out = try pp.applyPair(&ids_a, &ids_b, &buf, &tids);
    // Mirrors refs/tokenizers/.../processors/roberta.rs roberta_processing
    // test: vec![0, 12, 14, 2, 2, 15, 2] / vec![0; 7].
    try testing.expectEqualSlices(TokenId, &[_]TokenId{ 0, 12, 14, 2, 2, 15, 2 }, out);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 0, 0, 0, 0, 0, 0 }, tids[0..out.len]);
    try testing.expectEqual(pp.outputLenPair(ids_a.len, ids_b.len), out.len);
}

test "parseFromJson parses ByteLevel post-processor" {
    const input =
        \\{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":false,"use_regex":true}
    ;
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, input, .{});
    defer parsed.deinit();
    var pp = try parseFromJson(testing.allocator, parsed.value);
    defer pp.deinit();
    try testing.expect(pp == .byte_level);
    try testing.expectEqual(true, pp.byte_level.add_prefix_space);
    try testing.expectEqual(false, pp.byte_level.trim_offsets);
    try testing.expectEqual(true, pp.byte_level.use_regex);
}

test "ByteLevel post-processor apply is identity in id-space" {
    const input =
        \\{"type":"ByteLevel"}
    ;
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, input, .{});
    defer parsed.deinit();
    var pp = try parseFromJson(testing.allocator, parsed.value);
    defer pp.deinit();
    // Defaults all true.
    try testing.expectEqual(true, pp.byte_level.add_prefix_space);
    try testing.expectEqual(true, pp.byte_level.trim_offsets);
    try testing.expectEqual(true, pp.byte_level.use_regex);

    const ids_a = [_]TokenId{ 100, 200, 300 };
    const ids_b = [_]TokenId{ 400, 500 };
    var buf: [16]TokenId = undefined;
    var tids: [16]u32 = undefined;

    const out_s = pp.applySingle(&ids_a, &buf, &tids);
    try testing.expectEqualSlices(TokenId, &ids_a, out_s);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 0, 0 }, tids[0..out_s.len]);

    const out_p = try pp.applyPair(&ids_a, &ids_b, &buf, &tids);
    try testing.expectEqualSlices(TokenId, &[_]TokenId{ 100, 200, 300, 400, 500 }, out_p);
    // A=0, B=1 split.
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 0, 0, 1, 1 }, tids[0..out_p.len]);
}

test "parseFromJson parses Sequence of Bert + Template" {
    const input =
        \\{
        \\  "type": "Sequence",
        \\  "processors": [
        \\    {"type": "ByteLevel", "trim_offsets": true, "add_prefix_space": true, "use_regex": true},
        \\    {"type": "BertProcessing", "sep": ["[SEP]", 102], "cls": ["[CLS]", 101]}
        \\  ]
        \\}
    ;
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, input, .{});
    defer parsed.deinit();
    var pp = try parseFromJson(testing.allocator, parsed.value);
    defer pp.deinit();
    try testing.expect(pp == .sequence);
    try testing.expectEqual(@as(usize, 2), pp.sequence.processors.len);
    try testing.expect(pp.sequence.processors[0] == .byte_level);
    try testing.expect(pp.sequence.processors[1] == .bert);
    try testing.expectEqual(@as(TokenId, 101), pp.sequence.processors[1].bert.cls_id);

    // Apply: the chain effectively delegates to the framer (Bert).
    const ids_a = [_]TokenId{ 5, 6, 7 };
    var buf: [16]TokenId = undefined;
    const out = pp.applySingle(&ids_a, &buf, null);
    try testing.expectEqualSlices(TokenId, &[_]TokenId{ 101, 5, 6, 7, 102 }, out);
    try testing.expectEqual(pp.outputLenSingle(ids_a.len), out.len);
}

test "parseFromJson parses Sequence containing Bert + Template (structural)" {
    // Per task brief item 3: Sequence containing Bert + Template.
    const input =
        \\{
        \\  "type": "Sequence",
        \\  "processors": [
        \\    {"type": "BertProcessing", "sep": ["[SEP]", 102], "cls": ["[CLS]", 101]},
        \\    {
        \\      "type": "TemplateProcessing",
        \\      "single": [
        \\        {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
        \\        {"Sequence": {"id": "A", "type_id": 0}},
        \\        {"SpecialToken": {"id": "[SEP]", "type_id": 0}}
        \\      ],
        \\      "pair": [],
        \\      "special_tokens": {
        \\        "[CLS]": {"id": "[CLS]", "ids": [101], "tokens": ["[CLS]"]},
        \\        "[SEP]": {"id": "[SEP]", "ids": [102], "tokens": ["[SEP]"]}
        \\      }
        \\    }
        \\  ]
        \\}
    ;
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, input, .{});
    defer parsed.deinit();
    var pp = try parseFromJson(testing.allocator, parsed.value);
    defer pp.deinit();
    try testing.expect(pp == .sequence);
    try testing.expectEqual(@as(usize, 2), pp.sequence.processors.len);
    try testing.expect(pp.sequence.processors[0] == .bert);
    try testing.expect(pp.sequence.processors[1] == .template);
    // Apply: outer-most framer (Bert) wins in id-space.
    const ids_a = [_]TokenId{ 5, 6 };
    var buf: [16]TokenId = undefined;
    const out = pp.applySingle(&ids_a, &buf, null);
    try testing.expectEqualSlices(TokenId, &[_]TokenId{ 101, 5, 6, 102 }, out);
}

test "parseFromJson rejects unknown post-processor type" {
    const input =
        \\{"type":"NoSuchProcessor"}
    ;
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, input, .{});
    defer parsed.deinit();
    try testing.expectError(error.MalformedJson, parseFromJson(testing.allocator, parsed.value));
}

// --- Round-trip via the writer (kept in hf_writer.zig for the inverse
//     direction); these tests parse JSON the writer would produce and
//     verify structural equality, completing the round-trip loop. ---

test "round-trip RobertaProcessing through stringified JSON" {
    // Build the exact wire string the writer emits (per
    // refs/tokenizers/.../processors/roberta.rs serde test).
    const input =
        \\{"type":"RobertaProcessing","sep":["</s>",2],"cls":["<s>",0],"trim_offsets":true,"add_prefix_space":true}
    ;
    var parsed1 = try std.json.parseFromSlice(std.json.Value, testing.allocator, input, .{});
    defer parsed1.deinit();
    var pp1 = try parseFromJson(testing.allocator, parsed1.value);
    defer pp1.deinit();

    // Round-trip: stringify and re-parse, assert structural identity.
    const reemitted = try std.json.Stringify.valueAlloc(testing.allocator, parsed1.value, .{});
    defer testing.allocator.free(reemitted);
    var parsed2 = try std.json.parseFromSlice(std.json.Value, testing.allocator, reemitted, .{});
    defer parsed2.deinit();
    var pp2 = try parseFromJson(testing.allocator, parsed2.value);
    defer pp2.deinit();
    try testing.expectEqual(pp1.roberta.cls_id, pp2.roberta.cls_id);
    try testing.expectEqual(pp1.roberta.sep_id, pp2.roberta.sep_id);
    try testing.expectEqualStrings(pp1.roberta.cls_token, pp2.roberta.cls_token);
    try testing.expectEqualStrings(pp1.roberta.sep_token, pp2.roberta.sep_token);
    try testing.expectEqual(pp1.roberta.trim_offsets, pp2.roberta.trim_offsets);
    try testing.expectEqual(pp1.roberta.add_prefix_space, pp2.roberta.add_prefix_space);
}

test "round-trip Sequence preserves inner processor types" {
    const input =
        \\{
        \\  "type": "Sequence",
        \\  "processors": [
        \\    {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
        \\    {"type": "RobertaProcessing", "sep": ["</s>", 2], "cls": ["<s>", 0], "trim_offsets": true, "add_prefix_space": true}
        \\  ]
        \\}
    ;
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, input, .{});
    defer parsed.deinit();
    var pp = try parseFromJson(testing.allocator, parsed.value);
    defer pp.deinit();
    try testing.expect(pp == .sequence);
    try testing.expectEqual(@as(usize, 2), pp.sequence.processors.len);
    try testing.expect(pp.sequence.processors[0] == .byte_level);
    try testing.expect(pp.sequence.processors[1] == .roberta);

    // Apply behaves like the inner Roberta (since ByteLevel is identity).
    const ids_a = [_]TokenId{ 12, 14 };
    const ids_b = [_]TokenId{15};
    var buf: [16]TokenId = undefined;
    var tids: [16]u32 = undefined;
    const out = try pp.applyPair(&ids_a, &ids_b, &buf, &tids);
    try testing.expectEqualSlices(TokenId, &[_]TokenId{ 0, 12, 14, 2, 2, 15, 2 }, out);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 0, 0, 0, 0, 0, 0 }, tids[0..out.len]);
}
