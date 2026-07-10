//! End-to-end pipeline: Normalizer → PreTokenizer → Model → Decoder.
//!
//! All stages are tagged unions so the whole pipeline is value-typed and
//! cheap to clone per worker thread. Batch encode dispatches via
//! `BatchPool` and is the default entry point; `encode` is the
//! single-string convenience.
//!
//! Optional `added_tokens` scanner runs BEFORE normalization to resolve
//! special tokens (`<|endoftext|>`, `[CLS]`, etc.) into their assigned
//! ids without subjecting them to BPE/Unigram tokenization.

const std = @import("std");

const TokenId = @import("token.zig").TokenId;
const Span = @import("token.zig").Span;
const OverlayKind = @import("token.zig").OverlayKind;
const Boundary = @import("token.zig").Boundary;
const Provenance = @import("token.zig").Provenance;
const Vocab = @import("vocab.zig").Vocab;
const Normalizer = @import("normalizer.zig").Normalizer;
const PreTokenizer = @import("pretok.zig").PreTokenizer;
const Model = @import("model.zig").Model;
const Decoder = @import("decoder.zig").Decoder;
const BatchPool = @import("thread_pool.zig").BatchPool;
const added_tokens_mod = @import("added_tokens.zig");
const trace_mod = @import("trace.zig");
const asm_normalizer = @import("asm_normalizer.zig");

/// Result of `Pipeline.encodeWithOffsets`. `ids` and `offsets` have the
/// same length; `offsets[i]` is the byte range in the ORIGINAL input
/// (not the post-normalized buffer) that produced `ids[i]`.
///
/// Under non-identity normalizers (NFC/NFD/NFKC/NFKD/byte_level), the
/// pipeline uses the normalizer's `origin` map to translate spans
/// from the post-normalization buffer back to original-input space.
/// For pre-tokenizers that also rewrite bytes (HF byte_level), the
/// pre-tokenizer's output buffer is still what those spans index into
/// — pair `byte_level` *normalizer* + `cl100k` pretok if you need
/// original-input chunking.
pub const EncodingWithOffsets = struct {
    ids: []TokenId,
    offsets: []Span,

    pub fn deinit(self: *EncodingWithOffsets, allocator: std.mem.Allocator) void {
        allocator.free(self.ids);
        allocator.free(self.offsets);
    }
};

/// One annotation channel of an `EncodingWithOverlays`. `values.len`
/// always equals the encoding's `ids.len`; `values[i]` annotates
/// `ids[i]`. Interpretation depends on `kind` — see `OverlayKind`.
pub const Overlay = struct {
    kind: OverlayKind,
    values: []u32,
};

/// `encodeWithOffsets` plus N caller-requested annotation channels, all
/// aligned 1:1 with `ids` (every slice — `ids`, `offsets`, and each
/// `overlays[*].values` — shares one length). Channels are populated in
/// the same order they were requested via `want`. See `OverlayKind` for
/// per-channel semantics; domain channels with no installed plugin come
/// back zero-filled rather than erroring.
pub const EncodingWithOverlays = struct {
    ids: []TokenId,
    offsets: []Span,
    overlays: []Overlay,

    /// Returns the values of the first channel matching `kind`, or null
    /// if it was not requested.
    pub fn channel(self: *const EncodingWithOverlays, kind: OverlayKind) ?[]const u32 {
        for (self.overlays) |o| {
            if (o.kind == kind) return o.values;
        }
        return null;
    }

    pub fn deinit(self: *EncodingWithOverlays, allocator: std.mem.Allocator) void {
        allocator.free(self.ids);
        allocator.free(self.offsets);
        for (self.overlays) |o| allocator.free(o.values);
        allocator.free(self.overlays);
    }
};

/// Translate a span produced against the post-normalization buffer
/// back to original-input space using a normalizer origin map.
///
/// - If `origin` is `null` (identity normalizer), the span is returned
///   unchanged.
/// - Otherwise, `span.start` maps to `origin[span.start]`. `span.end`
///   (exclusive) maps to `origin[span.end]` when that index is in
///   range, else to `original_len` (end-of-buffer semantics).
/// - Empty spans are projected to a single point.
///
/// `new_end < new_start` is clamped to `new_end = new_start` to defend
/// against pathological cases where canonical reordering may have
/// shuffled combining-mark origins across a span boundary.
pub fn translateSpan(origin: ?[]const u32, span: Span, original_len: u32) Span {
    const o = origin orelse return span;
    if (span.start == span.end) {
        const p = if (span.start < o.len) o[span.start] else original_len;
        return .{ .start = p, .end = p };
    }
    const new_start = if (span.start < o.len) o[span.start] else original_len;
    var new_end: u32 = if (span.end < o.len) o[span.end] else original_len;
    if (new_end < new_start) new_end = new_start;
    return .{ .start = new_start, .end = new_end };
}

/// Persistent per-thread scratch arena for `Pipeline.encodeWithScratch`.
///
/// Wraps a `std.heap.ArenaAllocator` that gets reset (with
/// `.retain_capacity`) at the start of every encode call, so the
/// underlying buffer is reused across calls. The first few calls grow
/// the arena to a steady-state high-water mark; subsequent calls hit
/// the bump path with no syscalls. Drop-in replacement for a
/// per-call GPA on hot loops over many small strings.
///
/// Construction is cheap (a single descriptor; no eager allocation —
/// the arena allocates lazily on first use). Each `ScratchArena`
/// owns its memory; create one per thread.
pub const ScratchArena = struct {
    arena: std.heap.ArenaAllocator,

    /// Create a new scratch arena backed by `child_allocator` (usually
    /// the same GPA you pass as the result allocator). Cheap — no
    /// memory is reserved up front.
    pub fn init(child_allocator: std.mem.Allocator) ScratchArena {
        return .{ .arena = std.heap.ArenaAllocator.init(child_allocator) };
    }

    /// Release the arena's underlying memory.
    pub fn deinit(self: *ScratchArena) void {
        self.arena.deinit();
    }

    /// The allocator interface to hand to `encodeWithScratch`.
    pub fn allocator(self: *ScratchArena) std.mem.Allocator {
        return self.arena.allocator();
    }

    /// Reset the arena to zero used bytes while keeping the underlying
    /// buffer. Called automatically by `encodeWithScratch` — exposed
    /// for callers that want to drive the lifecycle themselves.
    pub fn reset(self: *ScratchArena) void {
        _ = self.arena.reset(.retain_capacity);
    }

    /// Reserved-byte high-water mark; same semantics as
    /// `BatchPool.peakScratchBytes`.
    pub fn peakBytes(self: *const ScratchArena) usize {
        return self.arena.queryCapacity();
    }
};

pub const Pipeline = struct {
    /// Selects which (if any) domain-specific normalizer populates the
    /// `opcode_class` / `operand_class` overlay channels. `.none` (the
    /// default) preserves the historical zero-fill behavior; `.x86_64`
    /// walks the input through the x86-64 instruction classifier and
    /// labels each token with the class of the instruction covering its
    /// starting byte. Only consulted when the caller actually requests an
    /// affected channel via `want`.
    pub const OverlayDomain = enum { none, x86_64 };

    normalizer: Normalizer,
    pre_tokenizer: PreTokenizer,
    model: Model,
    decoder: Decoder,
    vocab: *const Vocab,
    added_tokens: ?*const added_tokens_mod.Scanner = null,
    /// Optional encoder-trace sink. When non-null, BPE / Unigram /
    /// Monster emit per-step decision records to `trace.writer`. When
    /// null (the default), the encoders pay one predicted-not-taken
    /// null check per decision point — no allocations, no formatting.
    trace: ?*trace_mod.Trace = null,
    /// Opt-in domain normalizer for the `opcode_class` / `operand_class`
    /// overlay channels. Default `.none` keeps those channels zero-filled.
    overlay_domain: OverlayDomain = .none,

    /// Factory: build a `ScratchArena` backed by `child_allocator`.
    /// Convenience for callers that don't want to import the scratch
    /// type directly; equivalent to `ScratchArena.init(child_allocator)`.
    pub fn makeScratch(_: *const Pipeline, child_allocator: std.mem.Allocator) ScratchArena {
        return ScratchArena.init(child_allocator);
    }

    /// Run normalize → pre-tokenize → model.encode on a single text
    /// chunk into the caller-provided `out` buffer. Returns the slice
    /// of `out` written.
    fn encodeText(
        self: *const Pipeline,
        scratch: std.mem.Allocator,
        text: []const u8,
        out: []TokenId,
    ) ![]TokenId {
        const normalized = try self.normalizer.normalize(scratch, text);
        defer scratch.free(normalized);
        var pr = try self.pre_tokenizer.split(scratch, normalized);
        defer pr.deinit(scratch);
        var n: usize = 0;
        for (pr.spans) |s| {
            const w = try self.model.encodeTraced(scratch, s.slice(pr.data), out[n..], self.trace);
            n += w.len;
        }
        return out[0..n];
    }

    /// Same as `encodeText` but also populates `out_offsets`. `base_offset`
    /// is added to each per-chunk span so callers can position the result
    /// inside the larger input buffer. Returns the number of ids written.
    ///
    /// Under a non-identity normalizer, offsets returned by the model
    /// index into the post-normalized buffer; we translate them back
    /// to original-input space via the normalizer's origin map before
    /// applying `base_offset` (which is in original-input coordinates).
    fn encodeTextWithOffsets(
        self: *const Pipeline,
        scratch: std.mem.Allocator,
        text: []const u8,
        base_offset: u32,
        out_ids: []TokenId,
        out_offsets: []Span,
    ) !usize {
        var nr = try self.normalizer.normalizeWithOrigin(scratch, text);
        defer nr.deinit(scratch);
        var pr = try self.pre_tokenizer.split(scratch, nr.bytes);
        defer pr.deinit(scratch);

        const text_len: u32 = @intCast(text.len);
        var n: usize = 0;
        for (pr.spans) |s| {
            // Model writes offsets relative to its input slice; we
            // pre-add the span's *post-normalization* start so model
            // outputs are in post-normalization buffer coordinates.
            const start = n;
            const w = try self.model.encodeWithOffsetsTraced(
                scratch,
                s.slice(pr.data),
                s.start,
                out_ids[n..],
                out_offsets[n..],
                self.trace,
            );
            n += w;
            // Translate the freshly written offsets back to original-
            // input space (no-op for identity normalizer / null origin).
            if (nr.origin) |origin| {
                var k: usize = start;
                while (k < n) : (k += 1) {
                    const orig_span = translateSpan(origin, out_offsets[k], text_len);
                    out_offsets[k] = .{
                        .start = orig_span.start + base_offset,
                        .end = orig_span.end + base_offset,
                    };
                }
            } else {
                // Identity normalizer: post-norm offsets == original
                // offsets; just apply base_offset.
                var k: usize = start;
                while (k < n) : (k += 1) {
                    out_offsets[k] = .{
                        .start = out_offsets[k].start + base_offset,
                        .end = out_offsets[k].end + base_offset,
                    };
                }
            }
        }
        return n;
    }

    /// Like `encodeTextWithOffsets` but also writes a chunk-start marker
    /// per token into `out_chunk_start`: 1 for the first token emitted
    /// from each pre-tokenizer span, 0 otherwise. Caller must pre-zero
    /// `out_chunk_start` (this only sets the 1s). Returns ids written.
    fn encodeTextWithOverlays(
        self: *const Pipeline,
        scratch: std.mem.Allocator,
        text: []const u8,
        base_offset: u32,
        out_ids: []TokenId,
        out_offsets: []Span,
        out_chunk_start: []u8,
    ) !usize {
        var nr = try self.normalizer.normalizeWithOrigin(scratch, text);
        defer nr.deinit(scratch);
        var pr = try self.pre_tokenizer.split(scratch, nr.bytes);
        defer pr.deinit(scratch);

        const text_len: u32 = @intCast(text.len);
        var n: usize = 0;
        for (pr.spans) |s| {
            const start = n;
            const w = try self.model.encodeWithOffsetsTraced(
                scratch,
                s.slice(pr.data),
                s.start,
                out_ids[n..],
                out_offsets[n..],
                self.trace,
            );
            n += w;
            if (w > 0) out_chunk_start[start] = 1;
            if (nr.origin) |origin| {
                var k: usize = start;
                while (k < n) : (k += 1) {
                    const orig_span = translateSpan(origin, out_offsets[k], text_len);
                    out_offsets[k] = .{
                        .start = orig_span.start + base_offset,
                        .end = orig_span.end + base_offset,
                    };
                }
            } else {
                var k: usize = start;
                while (k < n) : (k += 1) {
                    out_offsets[k] = .{
                        .start = out_offsets[k].start + base_offset,
                        .end = out_offsets[k].end + base_offset,
                    };
                }
            }
        }
        return n;
    }

    pub fn encode(
        self: *const Pipeline,
        allocator: std.mem.Allocator,
        input: []const u8,
    ) ![]TokenId {
        // Default-encode path: spin up a one-shot scratch arena backed
        // by `allocator` and delegate to `encodeWithScratch`. Callers
        // that encode many strings in a hot loop should construct a
        // `ScratchArena` once and reuse it via `encodeWithScratch`
        // directly — that's where the >1.3x small-string speedup
        // lives. This wrapper preserves the original "give me ids
        // from one input" API and keeps allocator semantics
        // unchanged: returned ids come from `allocator`, the arena
        // is fully released on return.
        var scratch = ScratchArena.init(allocator);
        defer scratch.deinit();
        return self.encodeWithScratch(allocator, input, &scratch);
    }

    /// Like `encode` but lets the caller supply a persistent
    /// `ScratchArena`. The arena is reset (`.retain_capacity`) at
    /// entry so it can be safely reused across many encode calls
    /// without leaking. The arena holds the per-call transient
    /// state (normalizer output, pre-tokenizer span arrays, BPE SoA
    /// scratch on chunks > 256 bytes) — after the first few calls
    /// the arena steady-states and subsequent calls hit O(0)
    /// allocations on the hot path.
    ///
    /// The returned `[]TokenId` is allocated from `result_allocator`
    /// (NOT the arena) and is owned by the caller. The arena must
    /// NEVER be used to back returned ids — calling `scratch.reset`
    /// or `scratch.deinit` after this returns would dangle them.
    ///
    /// Single-thread callers that encode 100k+ small strings should
    /// hold a single `ScratchArena` per thread; batch callers should
    /// instead use `BatchPool` workers (each worker arena is already
    /// retain-capacity-reset on every batch).
    pub fn encodeWithScratch(
        self: *const Pipeline,
        result_allocator: std.mem.Allocator,
        input: []const u8,
        scratch_arena: *ScratchArena,
    ) ![]TokenId {
        scratch_arena.reset();
        const scratch = scratch_arena.allocator();

        // Normalizers (NFD/NFKD/byte_level) AND some pre-tokenizers (HF
        // byte_level) can expand the byte stream that the model sees.
        // Multiply both factors so the model's `out.len >= chunk.len`
        // assertion holds.
        const expansion = self.normalizer.maxByteExpansion() * self.pre_tokenizer.maxByteExpansion();

        if (self.added_tokens) |scanner| {
            // Segments come from the arena — they're scratch state we
            // don't need to keep past this call.
            const segs = try added_tokens_mod.scan(scanner, scratch, input);

            var cap: usize = 0;
            for (segs) |seg| switch (seg) {
                .text => |t| cap += self.model.maxTokensFor((t.end - t.start) * expansion),
                .special => cap += 1,
            };

            const out = try result_allocator.alloc(TokenId, cap);
            errdefer result_allocator.free(out);
            var n: usize = 0;
            for (segs) |seg| switch (seg) {
                .text => |t| {
                    const w = try self.encodeText(scratch, input[t.start..t.end], out[n..]);
                    n += w.len;
                },
                .special => |sp| {
                    out[n] = sp.id;
                    n += 1;
                },
            };
            return result_allocator.realloc(out, n);
        }

        // Fast path: no added tokens, no segmentation.
        const cap = self.model.maxTokensFor(input.len * expansion);
        const out = try result_allocator.alloc(TokenId, cap);
        errdefer result_allocator.free(out);
        const written = try self.encodeText(scratch, input, out);
        return result_allocator.realloc(out, written.len);
    }

    pub fn decode(
        self: *const Pipeline,
        allocator: std.mem.Allocator,
        ids: []const TokenId,
    ) ![]u8 {
        return self.decoder.decode(allocator, ids, self.model, self.vocab);
    }

    /// Encode `input` and return ids alongside the byte range in the
    /// ORIGINAL `input` (translated back from any post-normalization
    /// buffer via the normalizer's origin map — see
    /// `EncodingWithOffsets` doc) that produced each id. Special
    /// tokens from the added-token scanner span their literal byte
    /// range plus any lstrip/rstrip whitespace they absorbed.
    pub fn encodeWithOffsets(
        self: *const Pipeline,
        allocator: std.mem.Allocator,
        input: []const u8,
    ) !EncodingWithOffsets {
        // Scale the per-text cap by the normalizer's AND pre-tokenizer's
        // max byte expansion: NFD/NFKD/byte_level normalizers and the
        // HF byte_level pre-tokenizer can produce more bytes than the
        // input, and the model writes one id per byte in the worst case
        // (byte_id, BPE pre-merge).
        const expansion = self.normalizer.maxByteExpansion() * self.pre_tokenizer.maxByteExpansion();

        if (self.added_tokens) |scanner| {
            const segs = try added_tokens_mod.scan(scanner, allocator, input);
            defer allocator.free(segs);

            var cap: usize = 0;
            for (segs) |seg| switch (seg) {
                .text => |t| cap += self.model.maxTokensFor((t.end - t.start) * expansion),
                .special => cap += 1,
            };

            const out_ids = try allocator.alloc(TokenId, cap);
            errdefer allocator.free(out_ids);
            const out_offsets = try allocator.alloc(Span, cap);
            errdefer allocator.free(out_offsets);

            var n: usize = 0;
            for (segs) |seg| switch (seg) {
                .text => |t| {
                    const w = try self.encodeTextWithOffsets(
                        allocator,
                        input[t.start..t.end],
                        t.start,
                        out_ids[n..],
                        out_offsets[n..],
                    );
                    n += w;
                },
                .special => |sp| {
                    out_ids[n] = sp.id;
                    out_offsets[n] = .{ .start = sp.start, .end = sp.end };
                    n += 1;
                },
            };

            const ids_final = try allocator.realloc(out_ids, n);
            errdefer allocator.free(ids_final);
            const off_final = try allocator.realloc(out_offsets, n);
            return .{ .ids = ids_final, .offsets = off_final };
        }

        const cap = self.model.maxTokensFor(input.len * expansion);
        const out_ids = try allocator.alloc(TokenId, cap);
        errdefer allocator.free(out_ids);
        const out_offsets = try allocator.alloc(Span, cap);
        errdefer allocator.free(out_offsets);

        const written = try self.encodeTextWithOffsets(allocator, input, 0, out_ids, out_offsets);
        const ids_final = try allocator.realloc(out_ids, written);
        errdefer allocator.free(ids_final);
        const off_final = try allocator.realloc(out_offsets, written);
        return .{ .ids = ids_final, .offsets = off_final };
    }

    /// Encode `input` and return ids + offsets (as `encodeWithOffsets`)
    /// plus one annotation channel per entry in `want`, each aligned
    /// 1:1 with the id stream. The id stream is byte-identical to a
    /// plain `encode` — overlays only describe tokenization.
    ///
    /// Cheap channels (`byte_start`, `byte_end`, `boundary`,
    /// `provenance`) are filled from state the encoder already produces.
    /// Domain channels (`opcode_class`, `operand_class`, `symbol_ref`,
    /// `hunk`) and any `user_base`+ kind are zero-filled until a domain
    /// normalizer plugin populates them. Duplicate kinds in `want`
    /// produce duplicate channels (each its own allocation).
    pub fn encodeWithOverlays(
        self: *const Pipeline,
        allocator: std.mem.Allocator,
        input: []const u8,
        want: []const OverlayKind,
    ) !EncodingWithOverlays {
        const expansion = self.normalizer.maxByteExpansion() * self.pre_tokenizer.maxByteExpansion();

        // Resolve added-token segments once (if any) so we can size the
        // output buffers and drive provenance in a single pass.
        const segs: ?[]added_tokens_mod.Segment = if (self.added_tokens) |scanner|
            try added_tokens_mod.scan(scanner, allocator, input)
        else
            null;
        defer if (segs) |s| allocator.free(s);

        var cap: usize = 0;
        if (segs) |s| {
            for (s) |seg| switch (seg) {
                .text => |t| cap += self.model.maxTokensFor((t.end - t.start) * expansion),
                .special => cap += 1,
            };
        } else {
            cap = self.model.maxTokensFor(input.len * expansion);
        }

        // `out_ids`/`out_offsets` are var so the post-encode shrink
        // realloc reassigns the same variable the errdefer tracks —
        // avoids a dangling double-free on the (effectively impossible)
        // shrink failure.
        var out_ids = try allocator.alloc(TokenId, cap);
        errdefer allocator.free(out_ids);
        var out_offsets = try allocator.alloc(Span, cap);
        errdefer allocator.free(out_offsets);

        // Scratch side-channels, full-cap; only [0..n) is meaningful.
        const chunk_start = try allocator.alloc(u8, cap);
        defer allocator.free(chunk_start);
        const provenance = try allocator.alloc(u8, cap);
        defer allocator.free(provenance);
        @memset(chunk_start, 0);
        @memset(provenance, @as(u8, @intCast(Provenance.model_text)));

        var n: usize = 0;
        if (segs) |s| {
            for (s) |seg| switch (seg) {
                .text => |t| {
                    const w = try self.encodeTextWithOverlays(
                        allocator,
                        input[t.start..t.end],
                        t.start,
                        out_ids[n..],
                        out_offsets[n..],
                        chunk_start[n..],
                    );
                    n += w;
                },
                .special => |sp| {
                    out_ids[n] = sp.id;
                    out_offsets[n] = .{ .start = sp.start, .end = sp.end };
                    chunk_start[n] = 1;
                    provenance[n] = @intCast(Provenance.special);
                    n += 1;
                },
            };
        } else {
            n = try self.encodeTextWithOverlays(allocator, input, 0, out_ids, out_offsets, chunk_start);
        }

        out_ids = try allocator.realloc(out_ids, n);
        out_offsets = try allocator.realloc(out_offsets, n);

        const overlays = try allocator.alloc(Overlay, want.len);
        errdefer allocator.free(overlays);
        var built: usize = 0;
        errdefer {
            var i: usize = 0;
            while (i < built) : (i += 1) allocator.free(overlays[i].values);
        }

        // Domain classification: only walk the input when an x86-64 domain
        // is selected AND the caller actually requested an affected channel.
        // The lookups map each ORIGINAL-INPUT byte to the class of the
        // instruction span covering it (out_offsets[i].start is in
        // original-input coords).
        var asm_opcode: ?[]u32 = null;
        var asm_operand: ?[]u32 = null;
        defer if (asm_opcode) |s| allocator.free(s);
        defer if (asm_operand) |s| allocator.free(s);
        if (self.overlay_domain == .x86_64) {
            var wants_asm = false;
            for (want) |k| {
                if (k == .opcode_class or k == .operand_class) {
                    wants_asm = true;
                    break;
                }
            }
            if (wants_asm) {
                const op_lut = try allocator.alloc(u32, input.len);
                errdefer allocator.free(op_lut);
                const opnd_lut = try allocator.alloc(u32, input.len);
                @memset(op_lut, 0);
                @memset(opnd_lut, 0);
                var off: usize = 0;
                while (off < input.len) {
                    const d = asm_normalizer.next(input[off..]);
                    const step = if (d.len == 0) 1 else d.len;
                    const end = @min(off + step, input.len);
                    var b = off;
                    while (b < end) : (b += 1) {
                        op_lut[b] = d.opcode_class;
                        opnd_lut[b] = d.operand_class;
                    }
                    off += step;
                }
                asm_opcode = op_lut;
                asm_operand = opnd_lut;
            }
        }

        for (want, 0..) |kind, idx| {
            const vals = try allocator.alloc(u32, n);
            var i: usize = 0;
            while (i < n) : (i += 1) {
                vals[i] = switch (kind) {
                    .byte_start => out_offsets[i].start,
                    .byte_end => out_offsets[i].end,
                    .boundary => blk: {
                        var b: u32 = 0;
                        if (chunk_start[i] != 0) b |= Boundary.chunk_start;
                        const bs = out_offsets[i].start;
                        if (bs < input.len and (input[bs] & 0xC0) != 0x80) b |= Boundary.codepoint_start;
                        break :blk b;
                    },
                    .provenance => provenance[i],
                    .opcode_class => if (asm_opcode) |lut| blk: {
                        const bs = out_offsets[i].start;
                        break :blk if (bs < lut.len) lut[bs] else 0;
                    } else 0,
                    .operand_class => if (asm_operand) |lut| blk: {
                        const bs = out_offsets[i].start;
                        break :blk if (bs < lut.len) lut[bs] else 0;
                    } else 0,
                    // Other domain / user channels: no plugin yet -> zero-filled.
                    else => 0,
                };
            }
            overlays[idx] = .{ .kind = kind, .values = vals };
            built += 1;
        }

        return .{ .ids = out_ids, .offsets = out_offsets, .overlays = overlays };
    }

    /// Split `input` into roughly `n_chunks` pieces along pre-tokenizer-
    /// safe boundaries, encode each piece in parallel through `pool`,
    /// and concatenate the resulting ids into a single buffer.
    ///
    /// Output is **bit-identical** to `encode(...)` on the same input as
    /// long as the pre-tokenizer supports `findSafeCut` (currently
    /// `identity` and `cl100k`). For other pre-tokenizers, or if no safe
    /// cut can be found within the search window for some boundary, the
    /// function falls back to single-shot encoding on that segment.
    ///
    /// This is the parallel single-input entry point — use it instead of
    /// the bench-style "split bytes and call `encodeBatch`" pattern,
    /// which produces chunk-boundary divergences (~6 ids per 3.4 MB on
    /// cl100k).
    ///
    /// When `added_tokens` is configured, the scanner runs first on the
    /// whole input and the resulting `Segment` list is used to anchor
    /// chunk boundaries: special segments are atomic (they never split),
    /// and each text segment between specials is further subdivided via
    /// `findSafeCut` to spread work across workers. The result is still
    /// bit-identical to single-shot `encode`.
    pub fn encodeChunked(
        self: *const Pipeline,
        result_allocator: std.mem.Allocator,
        pool: *BatchPool,
        input: []const u8,
        n_chunks: usize,
    ) ![]TokenId {
        // Splitting is correct only when normalization is byte-identical and
        // the pre-tokenizer exposes truly independent model spans. Identity
        // pre-tokenization presents the entire input as one model span, so
        // only byte_id can be divided without changing model decisions.
        if (!self.normalizer.isIdentity()) return self.encode(result_allocator, input);
        switch (self.pre_tokenizer) {
            .identity => if (self.model != .byte_id) return self.encode(result_allocator, input),
            // Sequence chains currently use an approximate newline heuristic
            // rather than a proof for every supported operation.
            .chain => return self.encode(result_allocator, input),
            else => {},
        }
        if (self.added_tokens) |scanner| {
            return self.encodeChunkedWithAddedTokens(result_allocator, pool, scanner, input, n_chunks);
        }
        const chunks = if (n_chunks == 0) 1 else n_chunks;

        // Single-chunk or trivially small input: skip the chunking dance.
        if (chunks == 1 or input.len < 2 * chunks) {
            return self.encode(result_allocator, input);
        }

        // Compute safe chunk boundaries. We always produce `chunks`
        // segments (or fewer if the input is too small after snapping);
        // empty segments are dropped.
        const boundaries = try result_allocator.alloc(usize, chunks + 1);
        defer result_allocator.free(boundaries);
        boundaries[0] = 0;
        boundaries[chunks] = input.len;

        // Search window: half of the nominal chunk size, capped at 64 KiB.
        // 64 KiB is enough to find a `\n` in even the long-line corpora
        // we've seen (max line length ~11 KB). Snap cost is O(window) per
        // boundary, executed `chunks` times — negligible vs the O(N) BPE
        // encode that follows.
        const nominal = input.len / chunks;
        const window = @min(@max(nominal / 2, 256), 64 * 1024);

        var i: usize = 1;
        while (i < chunks) : (i += 1) {
            const desired = i * nominal;
            const snapped = self.pre_tokenizer.findSafeCut(input, desired, window) orelse
                return self.encode(result_allocator, input);
            // Enforce monotonicity (snapping backwards into a previous
            // segment would create a zero-length slice; clamp instead).
            boundaries[i] = @max(snapped, boundaries[i - 1]);
        }

        // Drop empty segments by compacting boundaries.
        var out_count: usize = 0;
        var j: usize = 0;
        while (j < chunks) : (j += 1) {
            if (boundaries[j] < boundaries[j + 1]) {
                boundaries[out_count] = boundaries[j];
                boundaries[out_count + 1] = boundaries[j + 1];
                out_count += 1;
            }
        }
        if (out_count == 0) {
            // Entire input is empty.
            return result_allocator.alloc(TokenId, 0);
        }
        if (out_count == 1) {
            // All chunks collapsed to one — just encode single-shot.
            return self.encode(result_allocator, input);
        }

        // Put worst-case partial outputs in the prewarmed worker-local
        // arenas. They remain valid until the next batch-wide reset, so a
        // worker may process multiple jobs without touching the shared GPA.
        // Only the final right-sized result is globally allocated.
        const expansion = self.pre_tokenizer.maxByteExpansion();
        const partials = try result_allocator.alloc([]const TokenId, out_count);
        defer result_allocator.free(partials);
        @memset(partials, &.{});
        pool.resetAllArenas();
        const Ctx = struct {
            pipe: *const Pipeline,
            pool: *BatchPool,
            input: []const u8,
            boundaries: []const usize,
            partials: [][]const TokenId,
            expansion: usize,
            errored: std.atomic.Value(u32) = .init(0),

            pub fn run(c: *@This(), idx: usize, worker_idx: usize) void {
                const scratch = c.pool.arenaAllocator(worker_idx);
                const text = c.input[c.boundaries[idx]..c.boundaries[idx + 1]];
                const cap = c.pipe.model.maxTokensFor(text.len * c.expansion);
                const region = scratch.alloc(TokenId, cap) catch {
                    _ = c.errored.fetchAdd(1, .acq_rel);
                    return;
                };
                c.partials[idx] = c.pipe.encodeText(scratch, text, region) catch {
                    _ = c.errored.fetchAdd(1, .acq_rel);
                    return;
                };
            }
        };
        var ctx: Ctx = .{
            .pipe = self,
            .pool = pool,
            .input = input,
            .boundaries = boundaries[0 .. out_count + 1],
            .partials = partials,
            .expansion = expansion,
        };
        try pool.runBatch(Ctx, &ctx, out_count);
        if (ctx.errored.load(.acquire) != 0) return error.BatchEncodeFailed;

        var total: usize = 0;
        for (partials) |partial| total += partial.len;
        const out = try result_allocator.alloc(TokenId, total);
        var written: usize = 0;
        for (partials) |partial| {
            @memcpy(out[written .. written + partial.len], partial);
            written += partial.len;
        }
        return out;
    }

    /// One unit of work the chunked-with-added-tokens path hands to a
    /// pool worker. Either "encode this absolute text slice" (the worker
    /// runs the full normalize → pre-tokenize → model encode pipeline) or
    /// "emit this special id" (cheap; we still go through the pool for
    /// uniform output handling).
    const ChunkJob = union(enum) {
        text: struct { start: u32, end: u32 },
        special: TokenId,
    };

    /// Split a `Segment` list (output of `added_tokens_mod.scan`) into a
    /// flat list of `ChunkJob` items honoring `n_target_chunks`. Special
    /// segments become single-id jobs as-is; text segments are subdivided
    /// by `pretok.findSafeCut`, with the number of sub-pieces per segment
    /// proportional to its byte length. The returned slice is owned by
    /// the caller.
    fn segmentSafeCuts(
        scratch: std.mem.Allocator,
        input: []const u8,
        segments: []const added_tokens_mod.Segment,
        n_target_chunks: usize,
        pretok: PreTokenizer,
    ) ![]ChunkJob {
        // Count text bytes and number of specials up front so we can
        // distribute the remaining chunk budget across text segments only.
        var text_bytes: usize = 0;
        var n_specials: usize = 0;
        var n_text_segs: usize = 0;
        for (segments) |seg| switch (seg) {
            .text => |t| {
                if (t.end > t.start) {
                    text_bytes += t.end - t.start;
                    n_text_segs += 1;
                }
            },
            .special => n_specials += 1,
        };

        // Specials are atomic and always emit one job each. Subtract them
        // from the budget so the text-pieces total roughly matches the
        // caller's requested chunk count. Floor at one chunk per non-empty
        // text segment so we still split when budget is tight.
        const target = if (n_target_chunks == 0) 1 else n_target_chunks;
        const remaining_for_text: usize = if (n_specials >= target)
            n_text_segs
        else
            @max(n_text_segs, target - n_specials);

        var jobs: std.ArrayList(ChunkJob) = .empty;
        defer jobs.deinit(scratch);
        // Upper bound: every text segment may produce up to
        // `remaining_for_text` sub-pieces, plus specials.
        try jobs.ensureTotalCapacity(scratch, remaining_for_text + n_specials);

        for (segments) |seg| switch (seg) {
            .special => |sp| {
                try jobs.append(scratch, .{ .special = sp.id });
            },
            .text => |t| {
                if (t.end <= t.start) continue;
                const seg_len: usize = t.end - t.start;

                // Proportional share of the remaining-text budget, with a
                // floor of 1 sub-piece per non-empty text segment.
                var share: usize = 1;
                if (text_bytes > 0 and n_text_segs > 0) {
                    // (seg_len * remaining_for_text + text_bytes/2) / text_bytes
                    // gives a rounded-to-nearest distribution.
                    const num = seg_len * remaining_for_text + text_bytes / 2;
                    share = @max(1, num / text_bytes);
                }
                // Don't ask for more sub-pieces than bytes; that just
                // wastes findSafeCut calls on degenerate boundaries.
                if (share > seg_len) share = seg_len;
                if (share == 0) share = 1;

                // Find sub-cut points inside this text segment. Returned
                // values are positions in `input[t.start..t.end]`.
                const sub_slice = input[t.start..t.end];
                const sub_nominal: usize = seg_len / share;
                const window: usize = @min(@max(sub_nominal / 2, 256), 64 * 1024);

                var cut_starts: std.ArrayList(u32) = .empty;
                defer cut_starts.deinit(scratch);
                try cut_starts.ensureTotalCapacity(scratch, share + 1);
                try cut_starts.append(scratch, 0);

                var i: usize = 1;
                while (i < share) : (i += 1) {
                    const desired = i * sub_nominal;
                    // Omit an unsafe desired cut. The surrounding safe cuts
                    // (or the segment endpoints) still form a correct job.
                    const snapped = pretok.findSafeCut(sub_slice, desired, window) orelse continue;
                    const prev = cut_starts.items[cut_starts.items.len - 1];
                    const monotone: u32 = @intCast(@max(snapped, @as(usize, prev)));
                    try cut_starts.append(scratch, monotone);
                }
                try cut_starts.append(scratch, @intCast(seg_len));

                // Emit one text job per non-empty sub-range.
                var k: usize = 0;
                while (k + 1 < cut_starts.items.len) : (k += 1) {
                    const sub_start = cut_starts.items[k];
                    const sub_end = cut_starts.items[k + 1];
                    if (sub_end <= sub_start) continue;
                    try jobs.append(scratch, .{
                        .text = .{
                            .start = @as(u32, @intCast(t.start)) + sub_start,
                            .end = @as(u32, @intCast(t.start)) + sub_end,
                        },
                    });
                }
            },
        };

        return jobs.toOwnedSlice(scratch);
    }

    /// Chunked encode path when an `added_tokens` scanner is configured.
    /// Composes `added_tokens_mod.scan` (segment boundaries are atomic for
    /// specials) with `segmentSafeCuts` (text segments split via the
    /// pre-tokenizer's `findSafeCut`) and dispatches every job through
    /// the BatchPool. Output is bit-identical to single-shot `encode`.
    fn encodeChunkedWithAddedTokens(
        self: *const Pipeline,
        result_allocator: std.mem.Allocator,
        pool: *BatchPool,
        scanner: *const added_tokens_mod.Scanner,
        input: []const u8,
        n_chunks: usize,
    ) ![]TokenId {
        const segs = try added_tokens_mod.scan(scanner, result_allocator, input);
        defer result_allocator.free(segs);

        if (segs.len == 0) {
            // Empty input.
            return result_allocator.alloc(TokenId, 0);
        }

        // Fast path: nothing to parallelize over.
        const chunks_req = if (n_chunks == 0) 1 else n_chunks;
        if (chunks_req == 1 or input.len < 2 * chunks_req) {
            return self.encode(result_allocator, input);
        }

        const jobs = try segmentSafeCuts(result_allocator, input, segs, chunks_req, self.pre_tokenizer);
        defer result_allocator.free(jobs);

        if (jobs.len == 0) return result_allocator.alloc(TokenId, 0);
        if (jobs.len == 1) {
            // One job — single-shot is cheaper than the pool dance.
            return self.encode(result_allocator, input);
        }

        const expansion = self.normalizer.maxByteExpansion() * self.pre_tokenizer.maxByteExpansion();
        const partials = try result_allocator.alloc([]const TokenId, jobs.len);
        defer result_allocator.free(partials);
        @memset(partials, &.{});
        pool.resetAllArenas();

        const Ctx = struct {
            pipe: *const Pipeline,
            input: []const u8,
            jobs: []const ChunkJob,
            partials: [][]const TokenId,
            expansion: usize,
            pool: *BatchPool,
            errored: std.atomic.Value(u32),

            const Self = @This();

            pub fn run(c: *Self, idx: usize, worker_idx: usize) void {
                const job = c.jobs[idx];
                const scratch = c.pool.arenaAllocator(worker_idx);
                switch (job) {
                    .special => |id| {
                        const region = scratch.alloc(TokenId, 1) catch {
                            _ = c.errored.fetchAdd(1, .acq_rel);
                            return;
                        };
                        region[0] = id;
                        c.partials[idx] = region;
                    },
                    .text => |t| {
                        const text = c.input[t.start..t.end];
                        const cap = c.pipe.model.maxTokensFor(text.len * c.expansion);
                        const region = scratch.alloc(TokenId, cap) catch {
                            _ = c.errored.fetchAdd(1, .acq_rel);
                            return;
                        };
                        c.partials[idx] = c.pipe.encodeText(scratch, text, region) catch {
                            _ = c.errored.fetchAdd(1, .acq_rel);
                            return;
                        };
                    },
                }
            }
        };

        var ctx: Ctx = .{
            .pipe = self,
            .input = input,
            .jobs = jobs,
            .partials = partials,
            .expansion = expansion,
            .pool = pool,
            .errored = .init(0),
        };

        try pool.runBatch(Ctx, &ctx, jobs.len);
        if (ctx.errored.load(.acquire) != 0) return error.BatchEncodeFailed;
        var total: usize = 0;
        for (partials) |partial| total += partial.len;
        const out = try result_allocator.alloc(TokenId, total);
        var written: usize = 0;
        for (partials) |partial| {
            @memcpy(out[written .. written + partial.len], partial);
            written += partial.len;
        }
        return out;
    }

    /// Encode many inputs in parallel. Each result is allocated from
    /// the caller-supplied `result_allocator` (a thread-safe allocator
    /// is required), so the caller owns the returned `[]TokenId`. The
    /// per-worker arena inside `pool` holds only scratch.
    pub fn encodeBatch(
        self: *const Pipeline,
        result_allocator: std.mem.Allocator,
        pool: *BatchPool,
        inputs: []const []const u8,
        results: [][]TokenId,
    ) !void {
        std.debug.assert(results.len == inputs.len);
        for (results) |*r| r.* = &.{};

        const Ctx = struct {
            pipe: *const Pipeline,
            inputs: []const []const u8,
            results: [][]TokenId,
            result_allocator: std.mem.Allocator,
            pool: *BatchPool,
            errored: std.atomic.Value(u32),

            const Self = @This();

            pub fn run(c: *Self, idx: usize, worker_idx: usize) void {
                const scratch = c.pool.resetArena(worker_idx);
                c.results[idx] = encodeOne(c.pipe, scratch, c.result_allocator, c.inputs[idx]) catch {
                    _ = c.errored.fetchAdd(1, .acq_rel);
                    return;
                };
            }

            fn encodeOne(
                pipe: *const Pipeline,
                scratch: std.mem.Allocator,
                ra: std.mem.Allocator,
                input: []const u8,
            ) ![]TokenId {
                const exp = pipe.normalizer.maxByteExpansion() * pipe.pre_tokenizer.maxByteExpansion();
                if (pipe.added_tokens) |scanner| {
                    const segs = try added_tokens_mod.scan(scanner, scratch, input);
                    var cap: usize = 0;
                    for (segs) |seg| switch (seg) {
                        .text => |t| cap += pipe.model.maxTokensFor((t.end - t.start) * exp),
                        .special => cap += 1,
                    };
                    const out = try ra.alloc(TokenId, cap);
                    var n: usize = 0;
                    for (segs) |seg| switch (seg) {
                        .text => |t| {
                            const w = try pipe.encodeText(scratch, input[t.start..t.end], out[n..]);
                            n += w.len;
                        },
                        .special => |sp| {
                            out[n] = sp.id;
                            n += 1;
                        },
                    };
                    return ra.realloc(out, n);
                }

                const cap = pipe.model.maxTokensFor(input.len * exp);
                const out = try ra.alloc(TokenId, cap);
                const w = try pipe.encodeText(scratch, input, out);
                return ra.realloc(out, w.len);
            }
        };

        var ctx: Ctx = .{
            .pipe = self,
            .inputs = inputs,
            .results = results,
            .result_allocator = result_allocator,
            .pool = pool,
            .errored = .init(0),
        };

        pool.runBatch(Ctx, &ctx, inputs.len) catch |err| {
            for (results) |*r| {
                if (r.len > 0) result_allocator.free(r.*);
                r.* = &.{};
            }
            return err;
        };
        if (ctx.errored.load(.acquire) != 0) {
            for (results) |*r| {
                if (r.len > 0) result_allocator.free(r.*);
                r.* = &.{};
            }
            return error.BatchEncodeFailed;
        }
    }
};

test "pipeline encode/decode byte_id round-trip" {
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();

    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };

    const ids = try pipe.encode(std.testing.allocator, "hello");
    defer std.testing.allocator.free(ids);
    try std.testing.expectEqualSlices(TokenId, &.{ 'h', 'e', 'l', 'l', 'o' }, ids);

    const out = try pipe.decode(std.testing.allocator, ids);
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("hello", out);
}

test "end-to-end: cl100k + bpe round-trip" {
    const Bpe = @import("bpe.zig").Bpe;

    const a = std.testing.allocator;
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(a);

    const b64 = std.base64.standard.Encoder;
    var enc_buf: [16]u8 = undefined;
    var rank: u32 = 0;
    var b: u32 = 0;
    while (b < 256) : (b += 1) {
        const byte: [1]u8 = .{@intCast(b)};
        const encoded = b64.encode(&enc_buf, &byte);
        try src.print(a, "{s} {d}\n", .{ encoded, rank });
        rank += 1;
    }
    const extra = [_][]const u8{
        "he",     "hel", "hell", "hello",
        " w",     " wo", " wor", " worl",
        " world",
    };
    for (extra) |bytes| {
        const encoded = b64.encode(&enc_buf, bytes);
        try src.print(a, "{s} {d}\n", .{ encoded, rank });
        rank += 1;
    }

    var bpe = try Bpe.loadTiktokenBytes(std.testing.allocator, src.items);
    defer bpe.deinit();

    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();

    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .cl100k,
        .model = .{ .bpe = &bpe },
        .decoder = .concat,
        .vocab = &v,
    };

    const ids = try pipe.encode(std.testing.allocator, "hello world");
    defer std.testing.allocator.free(ids);
    try std.testing.expectEqual(@as(usize, 2), ids.len);

    const round = try pipe.decode(std.testing.allocator, ids);
    defer std.testing.allocator.free(round);
    try std.testing.expectEqualStrings("hello world", round);
}

test "added_tokens resolves special before model encode" {
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();

    // Build a Scanner with one special token "<EOS>" -> id 50000.
    const at_tokens = [_]added_tokens_mod.AddedToken{
        .{ .id = 50000, .content = "<EOS>" },
    };
    var scanner = try added_tokens_mod.Scanner.init(std.testing.allocator, &at_tokens);
    defer scanner.deinit();

    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
        .added_tokens = &scanner,
    };

    const ids = try pipe.encode(std.testing.allocator, "hi<EOS>");
    defer std.testing.allocator.free(ids);
    // "hi" -> two byte_id tokens (104, 105), then special 50000.
    try std.testing.expectEqualSlices(TokenId, &.{ 'h', 'i', 50000 }, ids);
}

test "encodeBatch parallel byte_id" {
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();

    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };

    var bp = try BatchPool.init(std.testing.allocator, 2);
    defer bp.deinit();

    const inputs = [_][]const u8{ "ab", "cde", "f", "ghij" };
    var results: [4][]TokenId = undefined;
    try pipe.encodeBatch(std.testing.allocator, &bp, &inputs, &results);
    defer for (results) |r| std.testing.allocator.free(r);

    try std.testing.expectEqualSlices(TokenId, &.{ 'a', 'b' }, results[0]);
    try std.testing.expectEqualSlices(TokenId, &.{ 'c', 'd', 'e' }, results[1]);
    try std.testing.expectEqualSlices(TokenId, &.{'f'}, results[2]);
    try std.testing.expectEqualSlices(TokenId, &.{ 'g', 'h', 'i', 'j' }, results[3]);
}

test "encodeBatch with added_tokens" {
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();

    const at_tokens = [_]added_tokens_mod.AddedToken{
        .{ .id = 42, .content = "[X]" },
    };
    var scanner = try added_tokens_mod.Scanner.init(std.testing.allocator, &at_tokens);
    defer scanner.deinit();

    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
        .added_tokens = &scanner,
    };

    var bp = try BatchPool.init(std.testing.allocator, 2);
    defer bp.deinit();

    const inputs = [_][]const u8{ "a[X]b", "[X][X]" };
    var results: [2][]TokenId = undefined;
    try pipe.encodeBatch(std.testing.allocator, &bp, &inputs, &results);
    defer for (results) |r| std.testing.allocator.free(r);

    try std.testing.expectEqualSlices(TokenId, &.{ 'a', 42, 'b' }, results[0]);
    try std.testing.expectEqualSlices(TokenId, &.{ 42, 42 }, results[1]);
}

test "Pipeline.encodeWithOffsets identity normalizer byte_id" {
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();

    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };

    var enc = try pipe.encodeWithOffsets(std.testing.allocator, "hi");
    defer enc.deinit(std.testing.allocator);

    try std.testing.expectEqualSlices(TokenId, &.{ 'h', 'i' }, enc.ids);
    try std.testing.expectEqual(@as(usize, 2), enc.offsets.len);
    try std.testing.expectEqual(@as(u32, 0), enc.offsets[0].start);
    try std.testing.expectEqual(@as(u32, 1), enc.offsets[0].end);
    try std.testing.expectEqual(@as(u32, 1), enc.offsets[1].start);
    try std.testing.expectEqual(@as(u32, 2), enc.offsets[1].end);
}

test "Pipeline.encodeWithOverlays byte_id channels align + boundary + zero-fill" {
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();

    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };

    const want = [_]OverlayKind{ .byte_start, .byte_end, .boundary, .provenance, .opcode_class };
    var enc = try pipe.encodeWithOverlays(std.testing.allocator, "hi", &want);
    defer enc.deinit(std.testing.allocator);

    // ids unchanged vs a plain encode; every channel aligns 1:1.
    try std.testing.expectEqualSlices(TokenId, &.{ 'h', 'i' }, enc.ids);
    try std.testing.expectEqual(@as(usize, want.len), enc.overlays.len);
    for (enc.overlays) |o| try std.testing.expectEqual(enc.ids.len, o.values.len);

    try std.testing.expectEqualSlices(u32, &.{ 0, 1 }, enc.channel(.byte_start).?);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, enc.channel(.byte_end).?);

    // token0 begins the pre-tok chunk AND a codepoint; token1 only a codepoint.
    const b = enc.channel(.boundary).?;
    try std.testing.expectEqual(Boundary.chunk_start | Boundary.codepoint_start, b[0]);
    try std.testing.expectEqual(Boundary.codepoint_start, b[1]);

    try std.testing.expectEqualSlices(u32, &.{ Provenance.model_text, Provenance.model_text }, enc.channel(.provenance).?);

    // domain channel with no installed plugin -> zero-filled (not an error).
    try std.testing.expectEqualSlices(u32, &.{ 0, 0 }, enc.channel(.opcode_class).?);

    // an unrequested channel is absent.
    try std.testing.expectEqual(@as(?[]const u32, null), enc.channel(.hunk));
}

test "Pipeline.encodeWithOverlays codepoint_start skips UTF-8 continuation bytes" {
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();

    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };

    const want = [_]OverlayKind{.boundary};
    // "é" == 0xC3 0xA9 → two byte_id tokens; only the lead byte is a
    // codepoint start.
    var enc = try pipe.encodeWithOverlays(std.testing.allocator, "é", &want);
    defer enc.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 2), enc.ids.len);
    const b = enc.channel(.boundary).?;
    try std.testing.expectEqual(Boundary.chunk_start | Boundary.codepoint_start, b[0]);
    try std.testing.expectEqual(@as(u32, 0), b[1]);
}

test "Pipeline.encodeWithOverlays marks special-token provenance" {
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();

    const at_tokens = [_]added_tokens_mod.AddedToken{
        .{ .id = 50000, .content = "<EOS>" },
    };
    var scanner = try added_tokens_mod.Scanner.init(std.testing.allocator, &at_tokens);
    defer scanner.deinit();

    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
        .added_tokens = &scanner,
    };

    const want = [_]OverlayKind{ .provenance, .byte_start };
    var enc = try pipe.encodeWithOverlays(std.testing.allocator, "hi<EOS>", &want);
    defer enc.deinit(std.testing.allocator);

    try std.testing.expectEqualSlices(TokenId, &.{ 'h', 'i', 50000 }, enc.ids);
    try std.testing.expectEqualSlices(u32, &.{ Provenance.model_text, Provenance.model_text, Provenance.special }, enc.channel(.provenance).?);
    // the special token's byte_start is where its literal bytes begin.
    try std.testing.expectEqual(@as(u32, 2), enc.channel(.byte_start).?[2]);
}

test "Pipeline.encodeWithOverlays x86_64 domain populates opcode/operand classes" {
    const asm_norm = @import("asm_normalizer.zig");
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();

    // byte_id model + identity pre-tokenizer => one token per input byte,
    // so token i's byte_start == i and we can reason per-byte.
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
        .overlay_domain = .x86_64,
    };

    // 48 89 d8   mov rax, rbx   -> mov / reg_reg   (bytes 0..2)
    // e8 00..00  call rel32     -> call / rel32    (bytes 3..7)
    // c3         ret            -> ret / none      (byte 8)
    const code = [_]u8{ 0x48, 0x89, 0xd8, 0xe8, 0x00, 0x00, 0x00, 0x00, 0xc3 };

    const want = [_]OverlayKind{ .opcode_class, .operand_class, .byte_start };
    var enc = try pipe.encodeWithOverlays(std.testing.allocator, &code, &want);
    defer enc.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, code.len), enc.ids.len);

    const mov = @intFromEnum(asm_norm.Class.mov);
    const reg_reg = @intFromEnum(asm_norm.Operand.reg_reg);
    const call = @intFromEnum(asm_norm.Class.call);
    const rel32 = @intFromEnum(asm_norm.Operand.rel32);
    const ret = @intFromEnum(asm_norm.Class.ret);
    const none = @intFromEnum(asm_norm.Operand.none);

    const oc = enc.channel(.opcode_class).?;
    const opnd = enc.channel(.operand_class).?;

    // Every byte of the MOV instruction carries mov / reg_reg.
    try std.testing.expectEqualSlices(u32, &.{ mov, mov, mov }, oc[0..3]);
    try std.testing.expectEqualSlices(u32, &.{ reg_reg, reg_reg, reg_reg }, opnd[0..3]);
    // Every byte of the CALL instruction carries call / rel32.
    try std.testing.expectEqualSlices(u32, &.{ call, call, call, call, call }, oc[3..8]);
    try std.testing.expectEqualSlices(u32, &.{ rel32, rel32, rel32, rel32, rel32 }, opnd[3..8]);
    // RET.
    try std.testing.expectEqual(ret, oc[8]);
    try std.testing.expectEqual(none, opnd[8]);

    // Sanity: the labels match a direct asm_normalizer decode of each span.
    try std.testing.expectEqual(asm_norm.next(code[0..]).opcode_class, oc[0]);
    try std.testing.expectEqual(asm_norm.next(code[3..]).opcode_class, oc[3]);
}

test "Pipeline.encodeWithOverlays domain=none keeps opcode/operand zero-filled" {
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();

    // Default overlay_domain (.none) must preserve the historical zero-fill.
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };

    const code = [_]u8{ 0x48, 0x89, 0xd8, 0xc3 };
    const want = [_]OverlayKind{ .opcode_class, .operand_class };
    var enc = try pipe.encodeWithOverlays(std.testing.allocator, &code, &want);
    defer enc.deinit(std.testing.allocator);

    try std.testing.expectEqualSlices(u32, &.{ 0, 0, 0, 0 }, enc.channel(.opcode_class).?);
    try std.testing.expectEqualSlices(u32, &.{ 0, 0, 0, 0 }, enc.channel(.operand_class).?);
}

test "Pipeline.encodeWithOffsets with cl100k+bpe covers input" {
    const Bpe = @import("bpe.zig").Bpe;

    const a = std.testing.allocator;
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(a);

    const b64 = std.base64.standard.Encoder;
    var enc_buf: [16]u8 = undefined;
    var rank: u32 = 0;
    var b: u32 = 0;
    while (b < 256) : (b += 1) {
        const byte: [1]u8 = .{@intCast(b)};
        const encoded = b64.encode(&enc_buf, &byte);
        try src.print(a, "{s} {d}\n", .{ encoded, rank });
        rank += 1;
    }
    const extra = [_][]const u8{
        "he",     "hel", "hell", "hello",
        " w",     " wo", " wor", " worl",
        " world",
    };
    for (extra) |bytes| {
        const encoded = b64.encode(&enc_buf, bytes);
        try src.print(a, "{s} {d}\n", .{ encoded, rank });
        rank += 1;
    }

    var bpe = try Bpe.loadTiktokenBytes(std.testing.allocator, src.items);
    defer bpe.deinit();

    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();

    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .cl100k,
        .model = .{ .bpe = &bpe },
        .decoder = .concat,
        .vocab = &v,
    };

    var enc = try pipe.encodeWithOffsets(std.testing.allocator, "hello world");
    defer enc.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 2), enc.ids.len);
    try std.testing.expectEqual(enc.ids.len, enc.offsets.len);
    // First piece starts at 0, last ends at 11; spans are non-decreasing
    // and abut.
    try std.testing.expectEqual(@as(u32, 0), enc.offsets[0].start);
    try std.testing.expectEqual(@as(u32, 11), enc.offsets[enc.offsets.len - 1].end);
    var i: usize = 0;
    while (i + 1 < enc.offsets.len) : (i += 1) {
        try std.testing.expect(enc.offsets[i].start <= enc.offsets[i].end);
        try std.testing.expect(enc.offsets[i].end <= enc.offsets[i + 1].start);
    }
}

test "Pipeline.encodeWithOffsets with added_tokens specials" {
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();

    const at_tokens = [_]added_tokens_mod.AddedToken{
        .{ .id = 50000, .content = "<EOS>" },
    };
    var scanner = try added_tokens_mod.Scanner.init(std.testing.allocator, &at_tokens);
    defer scanner.deinit();

    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
        .added_tokens = &scanner,
    };

    var enc = try pipe.encodeWithOffsets(std.testing.allocator, "hi<EOS>");
    defer enc.deinit(std.testing.allocator);

    try std.testing.expectEqualSlices(TokenId, &.{ 'h', 'i', 50000 }, enc.ids);
    try std.testing.expectEqual(@as(usize, 3), enc.offsets.len);
    try std.testing.expectEqual(@as(u32, 0), enc.offsets[0].start);
    try std.testing.expectEqual(@as(u32, 1), enc.offsets[0].end);
    try std.testing.expectEqual(@as(u32, 1), enc.offsets[1].start);
    try std.testing.expectEqual(@as(u32, 2), enc.offsets[1].end);
    // Special "<EOS>" spans bytes 2..7.
    try std.testing.expectEqual(@as(u32, 2), enc.offsets[2].start);
    try std.testing.expectEqual(@as(u32, 7), enc.offsets[2].end);
}

test "Pipeline.encodeWithOffsets lstrip absorbed ws goes to special" {
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();

    const at_tokens = [_]added_tokens_mod.AddedToken{
        .{ .id = 9000, .content = "<s>", .lstrip = true },
    };
    var scanner = try added_tokens_mod.Scanner.init(std.testing.allocator, &at_tokens);
    defer scanner.deinit();

    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
        .added_tokens = &scanner,
    };

    // "a   <s>" — "a" then three spaces eaten by lstrip then "<s>".
    var enc = try pipe.encodeWithOffsets(std.testing.allocator, "a   <s>");
    defer enc.deinit(std.testing.allocator);

    try std.testing.expectEqualSlices(TokenId, &.{ 'a', 9000 }, enc.ids);
    try std.testing.expectEqual(@as(u32, 0), enc.offsets[0].start);
    try std.testing.expectEqual(@as(u32, 1), enc.offsets[0].end);
    // Special covers the absorbed whitespace (1..4) plus "<s>" (4..7).
    try std.testing.expectEqual(@as(u32, 1), enc.offsets[1].start);
    try std.testing.expectEqual(@as(u32, 7), enc.offsets[1].end);
}

// === translateSpan / origin-map offset translation ===

test "translateSpan: null origin returns span unchanged" {
    const s = Span{ .start = 3, .end = 7 };
    const got = translateSpan(null, s, 100);
    try std.testing.expectEqual(s.start, got.start);
    try std.testing.expectEqual(s.end, got.end);
}

test "translateSpan: maps via origin and handles end-of-buffer" {
    // Synthesize an origin map equivalent to NFD of "ab" expanding 'a'
    // -> [0,0,0] and 'b' -> [1]: post-norm bytes index map.
    const origin = [_]u32{ 0, 0, 0, 1 };
    const original_len: u32 = 2;

    // Whole-buffer span: post-norm [0..4) -> original [0..end-of-input).
    const all = translateSpan(&origin, .{ .start = 0, .end = 4 }, original_len);
    try std.testing.expectEqual(@as(u32, 0), all.start);
    try std.testing.expectEqual(@as(u32, 2), all.end);

    // Span over just 'a' decomposition: post-norm [0..3) -> original [0..1).
    const a_only = translateSpan(&origin, .{ .start = 0, .end = 3 }, original_len);
    try std.testing.expectEqual(@as(u32, 0), a_only.start);
    try std.testing.expectEqual(@as(u32, 1), a_only.end);

    // Empty span at offset 2 -> original offset 0 (still inside 'a').
    const empty = translateSpan(&origin, .{ .start = 2, .end = 2 }, original_len);
    try std.testing.expectEqual(@as(u32, 0), empty.start);
    try std.testing.expectEqual(@as(u32, 0), empty.end);
}

test "encodeWithOffsets: nfc normalizer returns spans in original-input space" {
    // Input has a precomposed é = U+00E9 (UTF-8 0xC3 0xA9). NFC is a
    // no-op here so origin is just per-cp 1:1 (within the multi-byte
    // codepoint each byte shares the cp start offset).
    // Use byte_id model so we can check each byte's span directly.
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .nfc,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };
    // "a" + "é" + "b" = 1 + 2 + 1 = 4 input bytes.
    const input = "a\xC3\xA9b";
    var enc = try pipe.encodeWithOffsets(std.testing.allocator, input);
    defer enc.deinit(std.testing.allocator);

    // NFC of this is the same — 4 output bytes -> 4 byte_id tokens.
    try std.testing.expectEqual(@as(usize, 4), enc.ids.len);
    // First token corresponds to 'a' at original offset 0.
    try std.testing.expectEqual(@as(u32, 0), enc.offsets[0].start);
    try std.testing.expectEqual(@as(u32, 1), enc.offsets[0].end);
    // Last token corresponds to 'b' at original offset 3.
    try std.testing.expectEqual(@as(u32, 3), enc.offsets[3].start);
    try std.testing.expectEqual(@as(u32, 4), enc.offsets[3].end);
    // All spans must be within the original input.
    for (enc.offsets) |sp| {
        try std.testing.expect(sp.start <= input.len);
        try std.testing.expect(sp.end <= input.len);
        try std.testing.expect(sp.start <= sp.end);
    }
}

test "encodeWithOffsets: nfd normalizer collapses decomposed bytes to original codepoint" {
    // Input has precomposed é. NFD decomposes it to 'e' + U+0301.
    // The decomposed bytes should all map back to the original cp's
    // start offset under our origin convention.
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .nfd,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };
    // Input "é" (precomposed, 2 bytes). NFD -> "e" (1) + U+0301 (2) = 3 byte_id tokens.
    const input = "\xC3\xA9";
    var enc = try pipe.encodeWithOffsets(std.testing.allocator, input);
    defer enc.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 3), enc.ids.len);
    // All three output tokens should report spans inside [0..2) of the
    // original input, since the only source codepoint started at 0.
    for (enc.offsets) |sp| {
        try std.testing.expect(sp.start <= input.len);
        try std.testing.expect(sp.end <= input.len);
        try std.testing.expect(sp.start <= sp.end);
    }
    // First span starts at original offset 0.
    try std.testing.expectEqual(@as(u32, 0), enc.offsets[0].start);
}

test "encodeWithOffsets: byte_level normalizer maps spans back to original input" {
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .byte_level,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };
    // " a" -> Ġa under byte_level (' ' becomes 0xC4 0xA0, 'a' stays as 'a').
    const input = " a";
    var enc = try pipe.encodeWithOffsets(std.testing.allocator, input);
    defer enc.deinit(std.testing.allocator);

    // 3 output bytes -> 3 byte_id tokens.
    try std.testing.expectEqual(@as(usize, 3), enc.ids.len);
    // First two output bytes come from input byte 0 (the space).
    try std.testing.expectEqual(@as(u32, 0), enc.offsets[0].start);
    try std.testing.expectEqual(@as(u32, 1), enc.offsets[1].end);
    // Third byte comes from input byte 1 ('a').
    try std.testing.expectEqual(@as(u32, 1), enc.offsets[2].start);
    try std.testing.expectEqual(@as(u32, 2), enc.offsets[2].end);
    // All spans must be within the original input.
    for (enc.offsets) |sp| {
        try std.testing.expect(sp.start <= input.len);
        try std.testing.expect(sp.end <= input.len);
    }
}

// === encodeChunked: chunk-boundary-safe parallel encoding ===

/// Build a small, dependency-free cl100k_base BPE vocab covering all 256
/// byte tokens plus a few merges, for use in chunk-boundary tests. Cribbed
/// from the existing round-trip tests.
fn buildTinyBpe(a: std.mem.Allocator) !@import("bpe.zig").Bpe {
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(a);

    const b64 = std.base64.standard.Encoder;
    var enc_buf: [16]u8 = undefined;
    var rank: u32 = 0;
    var b: u32 = 0;
    while (b < 256) : (b += 1) {
        const byte: [1]u8 = .{@intCast(b)};
        const encoded = b64.encode(&enc_buf, &byte);
        try src.print(a, "{s} {d}\n", .{ encoded, rank });
        rank += 1;
    }
    // A handful of merges so the BPE has interesting fan-in.
    const extra = [_][]const u8{
        "he",     "hel", "hell", "hello",
        " w",     " wo", " wor", " worl",
        " world", "th",  "the",  " the",
        " and",   " of", "in",   "ing",
        " a",     " to", " is",  "\n\n",
        " \n",    " *",  "##",
    };
    for (extra) |bytes| {
        const encoded = b64.encode(&enc_buf, bytes);
        try src.print(a, "{s} {d}\n", .{ encoded, rank });
        rank += 1;
    }

    return @import("bpe.zig").Bpe.loadTiktokenBytes(a, src.items);
}

fn expectChunkedMatchesSingleShot(
    pipe: *const Pipeline,
    pool: *BatchPool,
    input: []const u8,
    n_chunks: usize,
) !void {
    const a = std.testing.allocator;
    const single = try pipe.encode(a, input);
    defer a.free(single);
    const chunked = try pipe.encodeChunked(a, pool, input, n_chunks);
    defer a.free(chunked);
    try std.testing.expectEqualSlices(TokenId, single, chunked);
}

test "encodeChunked identity byte_id matches single-shot" {
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };
    var bp = try BatchPool.init(std.testing.allocator, 4);
    defer bp.deinit();

    try expectChunkedMatchesSingleShot(&pipe, &bp, "hello world this is a test", 4);
    try expectChunkedMatchesSingleShot(&pipe, &bp, "", 4);
    try expectChunkedMatchesSingleShot(&pipe, &bp, "x", 4);
    // Multi-byte UTF-8: ensure cuts don't land mid-codepoint.
    try expectChunkedMatchesSingleShot(&pipe, &bp, "αβγδεζηθικλμνξοπρστυφχψω", 4);
}

test "encodeChunked identity BPE falls back instead of splitting a model span" {
    var bpe = try buildTinyBpe(std.testing.allocator);
    defer bpe.deinit();
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .{ .bpe = &bpe },
        .decoder = .concat,
        .vocab = &v,
    };
    var bp = try BatchPool.init(std.testing.allocator, 4);
    defer bp.deinit();

    // Every chunk cut would otherwise be inside the single identity span,
    // allowing BPE merges such as "hello" to cross the boundary.
    try expectChunkedMatchesSingleShot(
        &pipe,
        &bp,
        "hellohellohellohello worldhellohellohellohello",
        7,
    );
}

test "encodeChunked cl100k+bpe matches single-shot — basic sentences" {
    var bpe = try buildTinyBpe(std.testing.allocator);
    defer bpe.deinit();
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .cl100k,
        .model = .{ .bpe = &bpe },
        .decoder = .concat,
        .vocab = &v,
    };
    var bp = try BatchPool.init(std.testing.allocator, 4);
    defer bp.deinit();

    // Newlines give the safe-cut search plenty of targets.
    const sample =
        "hello world\nthe quick brown fox\n" ++
        "jumps over the lazy dog\nand is happy of it\n" ++
        "more text here\nfinal line\n";
    try expectChunkedMatchesSingleShot(&pipe, &bp, sample, 4);
    try expectChunkedMatchesSingleShot(&pipe, &bp, sample, 1);
    try expectChunkedMatchesSingleShot(&pipe, &bp, sample, 8);
}

test "encodeChunked cl100k handles pathological whitespace and double newlines" {
    var bpe = try buildTinyBpe(std.testing.allocator);
    defer bpe.deinit();
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .cl100k,
        .model = .{ .bpe = &bpe },
        .decoder = .concat,
        .vocab = &v,
    };
    var bp = try BatchPool.init(std.testing.allocator, 4);
    defer bp.deinit();

    // Whitespace runs spanning newlines (the bug we fixed): pattern 5
    // `\s*[\r\n]+` would absorb mid-cut spaces in single-shot.
    try expectChunkedMatchesSingleShot(&pipe, &bp, "foo\n   \n bar\n\nbaz", 4);
    try expectChunkedMatchesSingleShot(&pipe, &bp, "a\n\n\n\nb\n\n\nc\n", 4);
    // Contractions at boundaries.
    try expectChunkedMatchesSingleShot(&pipe, &bp, "it's a day\nthey're here\nwe'll see\n", 4);
    // Multi-byte UTF-8 at boundaries.
    try expectChunkedMatchesSingleShot(&pipe, &bp, "café\nαβγ\nthe\nαβγδ\nend\n", 4);
}

test "encodeChunked cl100k random large corpus matches single-shot" {
    var bpe = try buildTinyBpe(std.testing.allocator);
    defer bpe.deinit();
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .cl100k,
        .model = .{ .bpe = &bpe },
        .decoder = .concat,
        .vocab = &v,
    };
    var bp = try BatchPool.init(std.testing.allocator, 4);
    defer bp.deinit();

    // ~100 KB of pseudo-random ASCII with frequent newlines and a sprinkle
    // of UTF-8. Deterministic seed so failures are reproducible.
    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const rng = prng.random();
    var corpus: std.ArrayList(u8) = .empty;
    defer corpus.deinit(std.testing.allocator);
    var i: usize = 0;
    while (i < 100_000) : (i += 1) {
        const r = rng.intRangeAtMost(u32, 0, 99);
        if (r < 4) {
            try corpus.append(std.testing.allocator, '\n');
        } else if (r < 8) {
            try corpus.append(std.testing.allocator, ' ');
        } else if (r < 10) {
            // UTF-8 codepoint U+00E9 (é) = 0xC3 0xA9.
            try corpus.appendSlice(std.testing.allocator, "\xC3\xA9");
        } else {
            // ASCII letter.
            const c: u8 = @intCast('a' + rng.intRangeAtMost(u32, 0, 25));
            try corpus.append(std.testing.allocator, c);
        }
    }
    try corpus.appendSlice(std.testing.allocator, "\n"); // trailing newline

    try expectChunkedMatchesSingleShot(&pipe, &bp, corpus.items, 8);
    try expectChunkedMatchesSingleShot(&pipe, &bp, corpus.items, 16);
}

test "encodeChunked hf_byte_level+bpe matches single-shot — multi-paragraph corpus" {
    var bpe = try buildTinyBpe(std.testing.allocator);
    defer bpe.deinit();
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .hf_byte_level,
        .model = .{ .bpe = &bpe },
        .decoder = .byte_level,
        .vocab = &v,
    };
    var bp = try BatchPool.init(std.testing.allocator, 4);
    defer bp.deinit();

    // Newlines between paragraphs give findSafeCut plenty of targets.
    const sample =
        "hello world\nthe quick brown fox\n" ++
        "jumps over the lazy dog\nand is happy of it\n" ++
        "more text here\nfinal line\n";
    try expectChunkedMatchesSingleShot(&pipe, &bp, sample, 4);
    try expectChunkedMatchesSingleShot(&pipe, &bp, sample, 1);
    try expectChunkedMatchesSingleShot(&pipe, &bp, sample, 8);
    // Contractions across the corpus.
    try expectChunkedMatchesSingleShot(
        &pipe,
        &bp,
        "it's a day\nthey're here\nwe'll see\nfoo\nbar baz\n",
        4,
    );
}

test "encodeChunked cl100k+bpe+added_tokens matches single-shot" {
    var bpe = try buildTinyBpe(std.testing.allocator);
    defer bpe.deinit();
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();

    const at_tokens = [_]added_tokens_mod.AddedToken{
        .{ .id = 50000, .content = "<|endoftext|>" },
        .{ .id = 50001, .content = "<|im_start|>" },
        .{ .id = 50002, .content = "<|im_end|>" },
    };
    var scanner = try added_tokens_mod.Scanner.init(std.testing.allocator, &at_tokens);
    defer scanner.deinit();

    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .cl100k,
        .model = .{ .bpe = &bpe },
        .decoder = .concat,
        .vocab = &v,
        .added_tokens = &scanner,
    };
    var bp = try BatchPool.init(std.testing.allocator, 4);
    defer bp.deinit();

    const sample =
        "<|im_start|>system\nyou are helpful<|im_end|>\n" ++
        "<|im_start|>user\nhello world the quick brown fox\n" ++
        "jumps over the lazy dog<|im_end|>\n" ++
        "<|im_start|>assistant\nthe answer is here<|im_end|>\n" ++
        "<|endoftext|>";
    try expectChunkedMatchesSingleShot(&pipe, &bp, sample, 4);
    try expectChunkedMatchesSingleShot(&pipe, &bp, sample, 1);
    try expectChunkedMatchesSingleShot(&pipe, &bp, sample, 8);
}

test "encodeChunked added_tokens that fragment input into many tiny segments" {
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();

    // A specials-heavy corpus: every few characters interrupts with a
    // special. Forces the chunked path to honor lots of atomic boundaries.
    const at_tokens = [_]added_tokens_mod.AddedToken{
        .{ .id = 1001, .content = "<|s|>" },
        .{ .id = 1002, .content = "<|e|>" },
    };
    var scanner = try added_tokens_mod.Scanner.init(std.testing.allocator, &at_tokens);
    defer scanner.deinit();

    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
        .added_tokens = &scanner,
    };
    var bp = try BatchPool.init(std.testing.allocator, 4);
    defer bp.deinit();

    const sample =
        "ab<|s|>cd<|e|>ef<|s|>gh<|e|>ij<|s|>kl<|e|>" ++
        "mn<|s|>op<|e|>qr<|s|>st<|e|>uv<|s|>wx<|e|>" ++
        "yz<|s|>01<|e|>23<|s|>45<|e|>67<|s|>89<|e|>";
    try expectChunkedMatchesSingleShot(&pipe, &bp, sample, 8);
    try expectChunkedMatchesSingleShot(&pipe, &bp, sample, 16);
}

test "encodeChunked added_tokens land at potential cut points" {
    var bpe = try buildTinyBpe(std.testing.allocator);
    defer bpe.deinit();
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();

    // A corpus engineered so that special tokens fall near where the
    // proportional splitter would otherwise pick a boundary — exercises
    // the "specials anchor; text sub-chunks slot in" composition.
    const at_tokens = [_]added_tokens_mod.AddedToken{
        .{ .id = 7777, .content = "<|sep|>" },
    };
    var scanner = try added_tokens_mod.Scanner.init(std.testing.allocator, &at_tokens);
    defer scanner.deinit();

    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .cl100k,
        .model = .{ .bpe = &bpe },
        .decoder = .concat,
        .vocab = &v,
        .added_tokens = &scanner,
    };
    var bp = try BatchPool.init(std.testing.allocator, 4);
    defer bp.deinit();

    // ~40-char repeating block ending in <|sep|>. With n_chunks=4 the
    // splitter wants cuts near positions 40, 80, 120 — right next to or
    // straddling the separators.
    const block = "the quick brown fox jumps over a dog\n<|sep|>";
    var corpus: std.ArrayList(u8) = .empty;
    defer corpus.deinit(std.testing.allocator);
    var k: usize = 0;
    while (k < 16) : (k += 1) {
        try corpus.appendSlice(std.testing.allocator, block);
    }
    try expectChunkedMatchesSingleShot(&pipe, &bp, corpus.items, 4);
    try expectChunkedMatchesSingleShot(&pipe, &bp, corpus.items, 8);
}

test "encodeChunked added_tokens with overlong text segments forces sub-cuts" {
    var bpe = try buildTinyBpe(std.testing.allocator);
    defer bpe.deinit();
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();

    const at_tokens = [_]added_tokens_mod.AddedToken{
        .{ .id = 31337, .content = "<|brk|>" },
    };
    var scanner = try added_tokens_mod.Scanner.init(std.testing.allocator, &at_tokens);
    defer scanner.deinit();

    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .cl100k,
        .model = .{ .bpe = &bpe },
        .decoder = .concat,
        .vocab = &v,
        .added_tokens = &scanner,
    };
    var bp = try BatchPool.init(std.testing.allocator, 4);
    defer bp.deinit();

    // Two huge text segments with one special between them — each segment
    // must split into multiple sub-pieces to use all workers.
    var corpus: std.ArrayList(u8) = .empty;
    defer corpus.deinit(std.testing.allocator);
    var i: usize = 0;
    while (i < 1500) : (i += 1) {
        try corpus.appendSlice(std.testing.allocator, "hello world\n");
    }
    try corpus.appendSlice(std.testing.allocator, "<|brk|>");
    i = 0;
    while (i < 1500) : (i += 1) {
        try corpus.appendSlice(std.testing.allocator, "the quick brown fox\n");
    }
    try expectChunkedMatchesSingleShot(&pipe, &bp, corpus.items, 8);
    try expectChunkedMatchesSingleShot(&pipe, &bp, corpus.items, 16);
}

test "encodeChunked added_tokens round-trip: encode→decode reconstructs input" {
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();

    // byte_id model + identity pretok + identity normalizer lets us verify
    // the chunked path doesn't corrupt the id stream by decoding back to
    // the original bytes. Specials (id >= 256) get stripped before decode.
    const at_tokens = [_]added_tokens_mod.AddedToken{
        .{ .id = 50000, .content = "<|sep|>" },
    };
    var scanner = try added_tokens_mod.Scanner.init(std.testing.allocator, &at_tokens);
    defer scanner.deinit();

    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
        .added_tokens = &scanner,
    };
    var bp = try BatchPool.init(std.testing.allocator, 4);
    defer bp.deinit();

    const input = "alpha<|sep|>beta gamma<|sep|>delta epsilon<|sep|>zeta";
    const ids = try pipe.encodeChunked(std.testing.allocator, &bp, input, 4);
    defer std.testing.allocator.free(ids);

    // Strip specials, then decode the byte-level ids back to text.
    var byte_ids: std.ArrayList(TokenId) = .empty;
    defer byte_ids.deinit(std.testing.allocator);
    for (ids) |id| if (id < 256) try byte_ids.append(std.testing.allocator, id);

    const decoded = try pipe.decode(std.testing.allocator, byte_ids.items);
    defer std.testing.allocator.free(decoded);

    // Decoded text should equal the input with all "<|sep|>" removed.
    var expected: std.ArrayList(u8) = .empty;
    defer expected.deinit(std.testing.allocator);
    var seg_it = std.mem.splitSequence(u8, input, "<|sep|>");
    while (seg_it.next()) |s| try expected.appendSlice(std.testing.allocator, s);
    try std.testing.expectEqualStrings(expected.items, decoded);
}

// --- persistent scratch-arena (encodeWithScratch) tests --------------

test "encodeWithScratch matches encode bit-identical (byte_id)" {
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };

    var scratch = ScratchArena.init(std.testing.allocator);
    defer scratch.deinit();

    const inputs = [_][]const u8{
        "",
        "h",
        "hello world",
        "the quick brown fox jumps over the lazy dog",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ" ** 8, // exercise the heap-spillover path
    };
    for (inputs) |inp| {
        const a = try pipe.encode(std.testing.allocator, inp);
        defer std.testing.allocator.free(a);
        const b = try pipe.encodeWithScratch(std.testing.allocator, inp, &scratch);
        defer std.testing.allocator.free(b);
        try std.testing.expectEqualSlices(TokenId, a, b);
    }
}

test "encodeWithScratch matches encode bit-identical (cl100k+bpe)" {
    const Bpe = @import("bpe.zig").Bpe;
    const a = std.testing.allocator;

    // Build a small byte-id BPE vocab via the standard tiktoken format so
    // we exercise the real encodeChunkScratch path including chunks
    // > STACK_LIMIT (heap fallback) and > HEAP_THRESHOLD (4-ary heap).
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(a);
    const b64 = std.base64.standard.Encoder;
    var enc_buf: [16]u8 = undefined;
    var rank: u32 = 0;
    while (rank < 256) : (rank += 1) {
        const byte: [1]u8 = .{@intCast(rank)};
        const encoded = b64.encode(&enc_buf, &byte);
        try src.print(a, "{s} {d}\n", .{ encoded, rank });
    }
    var bpe = try Bpe.loadTiktokenBytes(a, src.items);
    defer bpe.deinit();

    var v = Vocab.empty(a);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .cl100k,
        .model = .{ .bpe = &bpe },
        .decoder = .concat,
        .vocab = &v,
    };

    var scratch = ScratchArena.init(a);
    defer scratch.deinit();

    // 600 byte run: forces both the STACK_LIMIT (256) and HEAP_THRESHOLD
    // (64) spillover paths inside encodeChunkScratch.
    const big_input = "x" ** 600;
    const samples = [_][]const u8{ "hello", "the quick brown fox", big_input };
    for (samples) |inp| {
        const exp = try pipe.encode(a, inp);
        defer a.free(exp);
        const got = try pipe.encodeWithScratch(a, inp, &scratch);
        defer a.free(got);
        try std.testing.expectEqualSlices(TokenId, exp, got);
    }
}

test "encodeWithScratch arena bytes stay bounded across many calls" {
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };

    var scratch = ScratchArena.init(std.testing.allocator);
    defer scratch.deinit();

    // Warm up the arena to its steady state on a representative payload.
    const payload = "the quick brown fox jumps over the lazy dog 0123456789";
    var i: usize = 0;
    while (i < 32) : (i += 1) {
        const ids = try pipe.encodeWithScratch(std.testing.allocator, payload, &scratch);
        std.testing.allocator.free(ids);
    }
    const warm_cap = scratch.peakBytes();

    // Now hammer the same arena 1000 more times; capacity must NOT grow
    // (retain_capacity guarantees the buffer is reused). A small jitter
    // up to the page-aligned next bucket is theoretically possible but
    // not for identical-size payloads, so we expect strict equality.
    var j: usize = 0;
    while (j < 1000) : (j += 1) {
        const ids = try pipe.encodeWithScratch(std.testing.allocator, payload, &scratch);
        std.testing.allocator.free(ids);
    }
    try std.testing.expectEqual(warm_cap, scratch.peakBytes());
    // Sanity floor: a 54-byte input + identity normalizer + identity
    // pretok shouldn't reserve much. < 4 KB is a generous cap.
    try std.testing.expect(scratch.peakBytes() < 4 * 1024);
}

test "BatchPool per-worker scratch is race-free across 1000 calls" {
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };

    var bp = try BatchPool.init(std.testing.allocator, 8);
    defer bp.deinit();

    // 1000 inputs cycled across 8 workers — each worker hits its own
    // arena dozens of times. Compare every parallel result to the
    // single-shot encode of the same input.
    const N = 1000;
    var inputs: [N][]const u8 = undefined;
    const pool_strs = [_][]const u8{ "alpha", "beta gamma", "the quick brown fox jumps over the lazy dog", "x", "" };
    for (&inputs, 0..) |*slot, idx| slot.* = pool_strs[idx % pool_strs.len];

    var results: [N][]TokenId = undefined;
    try pipe.encodeBatch(std.testing.allocator, &bp, &inputs, &results);
    defer for (results) |r| std.testing.allocator.free(r);

    for (inputs, results) |inp, got| {
        const want = try pipe.encode(std.testing.allocator, inp);
        defer std.testing.allocator.free(want);
        try std.testing.expectEqualSlices(TokenId, want, got);
    }

    // peakScratchBytes for byte_id + identity + identity is dominated by
    // the per-tokenizer-result spans array (one Span per input byte)
    // plus the per-worker prewarm scratch buffer (1.19: 256 KiB default,
    // see `Options.prewarm_scratch_bytes`). For inputs < 64 bytes, the
    // actual scratch claim is well below the prewarm size — so the
    // total peak is bounded by `prewarm_scratch_bytes + a few KB`.
    var w: usize = 0;
    while (w < bp.workerCount()) : (w += 1) {
        const peak = bp.peakScratchBytes(w);
        try std.testing.expect(peak < 512 * 1024);
    }
}
