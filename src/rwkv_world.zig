//! RWKV "World" tokenizer — a greedy longest-match byte-trie model.
//!
//! RWKV (and the RWKV-8 ROSA line) does NOT use BPE. Its "World" vocab
//! (`rwkv_vocab_v20230424.txt`) is a flat list of byte strings; encoding
//! is a single left-to-right pass that, at each position, emits the id
//! of the LONGEST vocab entry matching there, then advances past it. The
//! vocab contains all 256 single bytes, so a match always exists and the
//! tokenizer is byte-lossless (no UNK, exact round-trip).
//!
//! There is no normalization and no pre-tokenization: the greedy walk
//! runs over the whole input as one chunk. Wire it with an identity
//! normalizer + identity pre-tokenizer + concat decoder.
//!
//! Storage mirrors `wordpiece.zig`: a flat `bytes` arena plus an
//! `offsets` table indexed by id (SoA), so `idBytes` is a slice and the
//! decoder concatenates without per-token allocation. The match trie is
//! a flat SoA (`Node`/`Child`, binary-searched children) like
//! `added_tokens.zig`, with a `root_child[256]` table for O(1)
//! first-byte dispatch.

const std = @import("std");
const TokenId = @import("token.zig").TokenId;
const Span = @import("token.zig").Span;

pub const RwkvWorld = struct {
    allocator: std.mem.Allocator,
    /// Flat token bytes, indexed via `offsets`. `idBytes(id)` =
    /// bytes[offsets[id]..offsets[id+1]]. Absent ids (gaps in the vocab
    /// id space) have a zero-length range.
    bytes: []u8,
    offsets: []u32, // len == count + 1
    count: u32, // max_id + 1; the id space size
    nodes: []Node,
    children: []Child,
    /// First-byte dispatch: root_child[b] is the trie node index reached
    /// by byte `b` from the root, or -1 if no vocab entry starts with `b`.
    root_child: [256]i32,

    pub const Node = struct {
        token_id: i32, // -1 if this node is not itself a token
        children_start: u32,
        children_len: u32,
    };
    pub const Child = struct { byte: u8, node: u32 };

    pub const Entry = struct { id: TokenId, bytes: []const u8 };

    pub fn deinit(self: *RwkvWorld) void {
        if (self.bytes.len > 0) self.allocator.free(self.bytes);
        if (self.offsets.len > 0) self.allocator.free(self.offsets);
        if (self.nodes.len > 0) self.allocator.free(self.nodes);
        if (self.children.len > 0) self.allocator.free(self.children);
        self.* = undefined;
    }

    /// Build from an in-memory list of (id, bytes) entries. The entry
    /// byte slices are copied into an owned arena; the caller keeps
    /// ownership of its inputs. Ids may be sparse and unordered.
    pub fn init(allocator: std.mem.Allocator, entries: []const Entry) !RwkvWorld {
        var max_id: u32 = 0;
        var total: usize = 0;
        for (entries) |e| {
            if (e.id > max_id) max_id = e.id;
            total += e.bytes.len;
        }
        const count: u32 = if (entries.len == 0) 0 else max_id + 1;

        const bytes = try allocator.alloc(u8, total);
        errdefer allocator.free(bytes);
        const offsets = try allocator.alloc(u32, count + 1);
        errdefer allocator.free(offsets);

        // Map id -> entry index so we can lay bytes out in id order
        // (offsets must be monotonic in id). -1 marks an absent id.
        const slot = try allocator.alloc(i32, count);
        defer allocator.free(slot);
        @memset(slot, -1);
        for (entries, 0..) |e, i| slot[e.id] = @intCast(i);

        var cursor: u32 = 0;
        for (0..count) |id| {
            offsets[id] = cursor;
            const si = slot[id];
            if (si >= 0) {
                const b = entries[@intCast(si)].bytes;
                @memcpy(bytes[cursor .. cursor + b.len], b);
                cursor += @intCast(b.len);
            }
        }
        offsets[count] = cursor;

        // --- build the match trie ---
        var nodes_b: std.ArrayList(BuildNode) = .empty;
        defer {
            for (nodes_b.items) |*nd| nd.children.deinit(allocator);
            nodes_b.deinit(allocator);
        }
        try nodes_b.append(allocator, .{ .token_id = -1, .children = .empty });

        for (entries) |e| {
            if (e.bytes.len == 0) continue;
            var cur: u32 = 0;
            for (e.bytes) |b| cur = try descendOrCreate(allocator, &nodes_b, cur, b);
            // Last writer wins on duplicate content (a vocab bug otherwise).
            nodes_b.items[cur].token_id = @intCast(e.id);
        }

        const node_count = nodes_b.items.len;
        var total_children: usize = 0;
        for (nodes_b.items) |nd| total_children += nd.children.items.len;

        const flat_nodes = try allocator.alloc(Node, node_count);
        errdefer allocator.free(flat_nodes);
        const flat_children = try allocator.alloc(Child, total_children);
        errdefer allocator.free(flat_children);

        var write: u32 = 0;
        for (nodes_b.items, 0..) |nd, idx| {
            const len: u32 = @intCast(nd.children.items.len);
            flat_nodes[idx] = .{
                .token_id = nd.token_id,
                .children_start = write,
                .children_len = len,
            };
            @memcpy(flat_children[write .. write + len], nd.children.items);
            write += len;
        }

        var self: RwkvWorld = .{
            .allocator = allocator,
            .bytes = bytes,
            .offsets = offsets,
            .count = count,
            .nodes = flat_nodes,
            .children = flat_children,
            .root_child = undefined,
        };
        // Populate the first-byte dispatch table from the root's children.
        @memset(&self.root_child, -1);
        const root = flat_nodes[0];
        for (flat_children[root.children_start .. root.children_start + root.children_len]) |c| {
            self.root_child[c.byte] = @intCast(c.node);
        }
        return self;
    }

    /// Parse `rwkv_vocab_v20230424.txt`. Each non-empty line is
    /// `<id> <python-repr> <byte-len>`, where the repr is a Python `str`
    /// or `bytes` literal (the BlinkDL reference `eval`s it). The trailing
    /// length is validated against the decoded byte count.
    pub fn loadFromFile(allocator: std.mem.Allocator, path: []const u8) !RwkvWorld {
        const io = std.Io.Threaded.global_single_threaded.io();
        const src = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
        defer allocator.free(src);
        return loadFromBytes(allocator, src);
    }

    pub fn loadFromBytes(allocator: std.mem.Allocator, src: []const u8) !RwkvWorld {
        var entries: std.ArrayList(Entry) = .empty;
        defer {
            for (entries.items) |e| allocator.free(@constCast(e.bytes));
            entries.deinit(allocator);
        }

        var lines = std.mem.splitScalar(u8, src, '\n');
        while (lines.next()) |raw| {
            const line = std.mem.trimEnd(u8, raw, "\r");
            if (line.len == 0) continue;

            const first_sp = std.mem.indexOfScalar(u8, line, ' ') orelse return error.InvalidVocab;
            const last_sp = std.mem.lastIndexOfScalar(u8, line, ' ') orelse return error.InvalidVocab;
            if (last_sp <= first_sp) return error.InvalidVocab;

            const id = std.fmt.parseInt(TokenId, line[0..first_sp], 10) catch return error.InvalidVocab;
            const repr = std.mem.trim(u8, line[first_sp + 1 .. last_sp], " ");
            const want_len = std.fmt.parseInt(usize, std.mem.trim(u8, line[last_sp + 1 ..], " "), 10) catch return error.InvalidVocab;

            const decoded = try parseReprBytes(allocator, repr);
            errdefer allocator.free(decoded);
            if (decoded.len != want_len) return error.VocabLenMismatch;
            try entries.append(allocator, .{ .id = id, .bytes = decoded });
        }

        return init(allocator, entries.items);
    }

    pub fn vocabSize(self: *const RwkvWorld) u32 {
        return self.count;
    }

    pub fn idBytes(self: *const RwkvWorld, id: TokenId) []const u8 {
        if (id >= self.count) return &.{};
        return self.bytes[self.offsets[id]..self.offsets[id + 1]];
    }

    fn findChild(self: *const RwkvWorld, node: u32, byte: u8) ?u32 {
        const n = self.nodes[node];
        if (n.children_len == 0) return null;
        const kids = self.children[n.children_start .. n.children_start + n.children_len];
        var l: usize = 0;
        var r: usize = kids.len;
        while (l < r) {
            const m = l + (r - l) / 2;
            const b = kids[m].byte;
            if (b == byte) return kids[m].node;
            if (b < byte) l = m + 1 else r = m;
        }
        return null;
    }

    /// Greedy longest-match encode over the whole chunk. `out` must hold
    /// at least `chunk.len` ids (the worst case: every byte its own
    /// token). Returns error.NoTokenMatch if some position has no
    /// matching entry — impossible with the real World vocab (all 256
    /// single bytes present), but possible with a partial fixture.
    pub fn encodeChunk(self: *const RwkvWorld, chunk: []const u8, out: []TokenId) ![]TokenId {
        if (chunk.len == 0) return out[0..0];
        std.debug.assert(out.len >= chunk.len);

        var pos: usize = 0;
        var n_out: usize = 0;
        while (pos < chunk.len) {
            const first = self.root_child[chunk[pos]];
            if (first < 0) return error.NoTokenMatch;

            var walk: u32 = @intCast(first);
            var best_id: i32 = self.nodes[walk].token_id;
            var best_len: usize = 1;

            var k: usize = pos + 1;
            while (k < chunk.len) {
                const child = self.findChild(walk, chunk[k]) orelse break;
                walk = child;
                k += 1;
                if (self.nodes[walk].token_id >= 0) {
                    best_id = self.nodes[walk].token_id;
                    best_len = k - pos;
                }
            }

            if (best_id < 0) return error.NoTokenMatch;
            out[n_out] = @intCast(best_id);
            n_out += 1;
            pos += best_len;
        }
        return out[0..n_out];
    }

    /// Same as `encodeChunk` but also fills `out_offsets` with the byte
    /// span of each emitted token, relative to `chunk_offset` (the
    /// position of `chunk[0]` in the buffer the caller indexes).
    pub fn encodeChunkWithOffsets(
        self: *const RwkvWorld,
        chunk: []const u8,
        chunk_offset: u32,
        out_ids: []TokenId,
        out_offsets: []Span,
    ) !usize {
        if (chunk.len == 0) return 0;
        std.debug.assert(out_ids.len >= chunk.len);
        std.debug.assert(out_offsets.len >= chunk.len);

        var pos: usize = 0;
        var n_out: usize = 0;
        while (pos < chunk.len) {
            const first = self.root_child[chunk[pos]];
            if (first < 0) return error.NoTokenMatch;

            var walk: u32 = @intCast(first);
            var best_id: i32 = self.nodes[walk].token_id;
            var best_len: usize = 1;

            var k: usize = pos + 1;
            while (k < chunk.len) {
                const child = self.findChild(walk, chunk[k]) orelse break;
                walk = child;
                k += 1;
                if (self.nodes[walk].token_id >= 0) {
                    best_id = self.nodes[walk].token_id;
                    best_len = k - pos;
                }
            }

            if (best_id < 0) return error.NoTokenMatch;
            out_ids[n_out] = @intCast(best_id);
            out_offsets[n_out] = .{
                .start = chunk_offset + @as(u32, @intCast(pos)),
                .end = chunk_offset + @as(u32, @intCast(pos + best_len)),
            };
            n_out += 1;
            pos += best_len;
        }
        return n_out;
    }
};

const BuildNode = struct {
    token_id: i32,
    children: std.ArrayList(RwkvWorld.Child),
};

fn descendOrCreate(
    allocator: std.mem.Allocator,
    nodes: *std.ArrayList(BuildNode),
    parent: u32,
    byte: u8,
) !u32 {
    {
        const kids = nodes.items[parent].children.items;
        var l: usize = 0;
        var r: usize = kids.len;
        while (l < r) {
            const m = l + (r - l) / 2;
            const b = kids[m].byte;
            if (b == byte) return kids[m].node;
            if (b < byte) l = m + 1 else r = m;
        }
    }

    const new_idx: u32 = @intCast(nodes.items.len);
    try nodes.append(allocator, .{ .token_id = -1, .children = .empty });

    const kids_ptr = &nodes.items[parent].children;
    var l2: usize = 0;
    var r2: usize = kids_ptr.items.len;
    while (l2 < r2) {
        const m = l2 + (r2 - l2) / 2;
        if (kids_ptr.items[m].byte < byte) l2 = m + 1 else r2 = m;
    }
    try kids_ptr.insert(allocator, l2, .{ .byte = byte, .node = new_idx });
    return new_idx;
}

// --- Python str/bytes-repr parser -----------------------------------
//
// Decodes a single Python literal as it appears in the RWKV vocab file:
// `'...'`, `"..."`, or the `b'...'` / `b"..."` byte-string forms. For
// str literals, `\xHH` / `\uHHHH` name a Unicode codepoint that we
// UTF-8 encode, and raw (already-UTF-8) bytes are copied verbatim. For
// bytes literals, `\xHH` is a raw byte and unescaped chars are raw
// bytes. The result is the token's UTF-8 / raw byte sequence.

/// Caller owns the returned slice.
pub fn parseReprBytes(allocator: std.mem.Allocator, repr_in: []const u8) ![]u8 {
    var repr = repr_in;
    var is_bytes = false;
    if (repr.len >= 1 and (repr[0] == 'b' or repr[0] == 'B')) {
        is_bytes = true;
        repr = repr[1..];
    }
    if (repr.len < 2) return error.BadRepr;
    const quote = repr[0];
    if (quote != '\'' and quote != '"') return error.BadRepr;
    if (repr[repr.len - 1] != quote) return error.BadRepr;
    const body = repr[1 .. repr.len - 1];

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);

    var i: usize = 0;
    while (i < body.len) {
        const c = body[i];
        if (c != '\\') {
            // Raw byte. For both str and bytes forms, copying the byte
            // verbatim is correct: str bodies are already UTF-8 and a
            // codepoint's UTF-8 bytes are exactly these bytes.
            try out.append(allocator, c);
            i += 1;
            continue;
        }
        // Escape sequence.
        i += 1;
        if (i >= body.len) return error.BadRepr;
        const e = body[i];
        i += 1;
        switch (e) {
            'n' => try out.append(allocator, '\n'),
            't' => try out.append(allocator, '\t'),
            'r' => try out.append(allocator, '\r'),
            '\\' => try out.append(allocator, '\\'),
            '\'' => try out.append(allocator, '\''),
            '"' => try out.append(allocator, '"'),
            '0' => try out.append(allocator, 0),
            'a' => try out.append(allocator, 0x07),
            'b' => try out.append(allocator, 0x08),
            'f' => try out.append(allocator, 0x0C),
            'v' => try out.append(allocator, 0x0B),
            'x' => {
                if (i + 2 > body.len) return error.BadRepr;
                const v = std.fmt.parseInt(u8, body[i .. i + 2], 16) catch return error.BadRepr;
                i += 2;
                if (is_bytes) {
                    try out.append(allocator, v);
                } else {
                    try appendCodepoint(allocator, &out, v);
                }
            },
            'u' => {
                if (i + 4 > body.len) return error.BadRepr;
                const v = std.fmt.parseInt(u21, body[i .. i + 4], 16) catch return error.BadRepr;
                i += 4;
                try appendCodepoint(allocator, &out, v);
            },
            'U' => {
                if (i + 8 > body.len) return error.BadRepr;
                const v = std.fmt.parseInt(u21, body[i .. i + 8], 16) catch return error.BadRepr;
                i += 8;
                try appendCodepoint(allocator, &out, v);
            },
            else => {
                // Unknown escape: Python keeps the backslash + char.
                try out.append(allocator, '\\');
                try out.append(allocator, e);
            },
        }
    }

    return out.toOwnedSlice(allocator);
}

fn appendCodepoint(allocator: std.mem.Allocator, out: *std.ArrayList(u8), cp: u21) !void {
    var buf: [4]u8 = undefined;
    const n = std.unicode.utf8Encode(cp, &buf) catch {
        // Lone surrogate / invalid scalar — store the low byte raw so we
        // never crash on a malformed vocab line.
        try out.append(allocator, @intCast(cp & 0xFF));
        return;
    };
    try out.appendSlice(allocator, buf[0..n]);
}

// --- tests ----------------------------------------------------------

const testing = std.testing;

fn buildTiny(alloc: std.mem.Allocator) !RwkvWorld {
    // A fixture covering every byte used below as a single token, plus a
    // few multi-byte tokens so greedy longest-match has real choices.
    const entries = [_]RwkvWorld.Entry{
        .{ .id = 0, .bytes = "a" },
        .{ .id = 1, .bytes = "b" },
        .{ .id = 2, .bytes = "c" },
        .{ .id = 3, .bytes = "ab" },
        .{ .id = 4, .bytes = "abc" },
        .{ .id = 5, .bytes = "bc" },
        .{ .id = 6, .bytes = " " },
    };
    return RwkvWorld.init(alloc, &entries);
}

test "greedy longest match prefers the longest entry" {
    var m = try buildTiny(testing.allocator);
    defer m.deinit();

    var out: [16]TokenId = undefined;
    // "abc" -> single token id 4 (longest), not a/b/c or ab+c.
    const ids = try m.encodeChunk("abc", &out);
    try testing.expectEqualSlices(TokenId, &.{4}, ids);
}

test "greedy falls back to shorter then resumes" {
    var m = try buildTiny(testing.allocator);
    defer m.deinit();

    var out: [16]TokenId = undefined;
    // "abca" -> "abc"(4) then "a"(0).
    const ids = try m.encodeChunk("abca", &out);
    try testing.expectEqualSlices(TokenId, &.{ 4, 0 }, ids);

    // "abbc" -> "ab"(3) then "bc"(5).
    const ids2 = try m.encodeChunk("abbc", &out);
    try testing.expectEqualSlices(TokenId, &.{ 3, 5 }, ids2);
}

test "round-trips through idBytes concatenation" {
    var m = try buildTiny(testing.allocator);
    defer m.deinit();

    var out: [16]TokenId = undefined;
    const ids = try m.encodeChunk("ab abc", &out);

    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(testing.allocator);
    for (ids) |id| try buf.appendSlice(testing.allocator, m.idBytes(id));
    try testing.expectEqualStrings("ab abc", buf.items);
}

test "encodeChunkWithOffsets spans cover the matched bytes" {
    var m = try buildTiny(testing.allocator);
    defer m.deinit();

    var ids: [16]TokenId = undefined;
    var off: [16]Span = undefined;
    const n = try m.encodeChunkWithOffsets("abca", 0, &ids, &off);
    try testing.expectEqual(@as(usize, 2), n);
    try testing.expectEqual(@as(u32, 0), off[0].start);
    try testing.expectEqual(@as(u32, 3), off[0].end); // "abc"
    try testing.expectEqual(@as(u32, 3), off[1].start);
    try testing.expectEqual(@as(u32, 4), off[1].end); // "a"
}

test "no matching token errors on partial vocab" {
    const entries = [_]RwkvWorld.Entry{.{ .id = 0, .bytes = "a" }};
    var m = try RwkvWorld.init(testing.allocator, &entries);
    defer m.deinit();

    var out: [4]TokenId = undefined;
    try testing.expectError(error.NoTokenMatch, m.encodeChunk("ax", &out));
}

test "parseReprBytes str form, plain ASCII" {
    const b = try parseReprBytes(testing.allocator, "'hello'");
    defer testing.allocator.free(b);
    try testing.expectEqualStrings("hello", b);
}

test "parseReprBytes str form with escapes" {
    const b = try parseReprBytes(testing.allocator, "'a\\tb\\n'");
    defer testing.allocator.free(b);
    try testing.expectEqualSlices(u8, &.{ 'a', '\t', 'b', '\n' }, b);
}

test "parseReprBytes str \\x is a codepoint (UTF-8 encoded)" {
    // '\xa0' is U+00A0 -> UTF-8 C2 A0 in str form.
    const b = try parseReprBytes(testing.allocator, "'\\xa0'");
    defer testing.allocator.free(b);
    try testing.expectEqualSlices(u8, &.{ 0xC2, 0xA0 }, b);
}

test "parseReprBytes bytes form \\x is a raw byte" {
    // b'\xa0' is the single raw byte 0xA0.
    const b = try parseReprBytes(testing.allocator, "b'\\xa0'");
    defer testing.allocator.free(b);
    try testing.expectEqualSlices(u8, &.{0xA0}, b);
}

test "parseReprBytes bytes form multi-byte UTF-8 sequence" {
    // b'\xe4\xbd\xa0' is the 3 raw bytes of U+4F60 (你).
    const b = try parseReprBytes(testing.allocator, "b'\\xe4\\xbd\\xa0'");
    defer testing.allocator.free(b);
    try testing.expectEqualSlices(u8, &.{ 0xE4, 0xBD, 0xA0 }, b);
}

test "parseReprBytes str form raw non-ASCII copied verbatim" {
    // '你' is already UTF-8 in the file; copy the 3 bytes through.
    const b = try parseReprBytes(testing.allocator, "'\xe4\xbd\xa0'");
    defer testing.allocator.free(b);
    try testing.expectEqualSlices(u8, &.{ 0xE4, 0xBD, 0xA0 }, b);
}

test "parseReprBytes escaped quote" {
    const b = try parseReprBytes(testing.allocator, "'it\\'s'");
    defer testing.allocator.free(b);
    try testing.expectEqualStrings("it's", b);
}

test "loadFromBytes parses an id/repr/len vocab and encodes" {
    const vocab =
        \\0 'a' 1
        \\1 'b' 1
        \\2 'ab' 2
        \\3 ' ' 1
    ;
    var m = try RwkvWorld.loadFromBytes(testing.allocator, vocab);
    defer m.deinit();
    try testing.expectEqual(@as(u32, 4), m.vocabSize());

    var out: [8]TokenId = undefined;
    const ids = try m.encodeChunk("ab a", &out);
    // "ab"(2) " "(3) "a"(0)
    try testing.expectEqualSlices(TokenId, &.{ 2, 3, 0 }, ids);
}

test "loadFromBytes validates the trailing length" {
    const bad = "0 'ab' 1\n"; // says len 1 but 'ab' is 2
    try testing.expectError(error.VocabLenMismatch, RwkvWorld.loadFromBytes(testing.allocator, bad));
}
