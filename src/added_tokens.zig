//! Added-token scanner.
//!
//! Runs before pre-tokenization. The HF tokenizer.json `added_tokens` array
//! lists strings (`<|endoftext|>`, `<|im_start|>`, `[CLS]`, `<s>`, ...) that
//! must resolve to a fixed id rather than be BPE-encoded as ordinary text.
//! `scan` walks the input once and emits a flat list of segments — either a
//! literal text byte range or a resolved special-token id — which the
//! pipeline encodes piece by piece.
//!
//! Algorithm: build a byte trie over the added-token strings at init time.
//! Children at each node are kept sorted by byte so lookup is a binary
//! search (the root with up to 256 children benefits the most). Scan is
//! O(input.len * max_token_len) worst case; near-linear in practice because
//! special tokens are rare. The hot loop allocates only into a pre-sized
//! segments buffer (we upper-bound it to 2*input.len + 1 before scanning).
//!
//! lstrip / rstrip: when an added-token match is found, lstrip absorbs
//! the run of whitespace immediately BEFORE the match (silently — the
//! special id replaces both the whitespace and the literal match).
//! rstrip absorbs the run of whitespace immediately AFTER. Both flags
//! compose. The whitespace class follows Unicode `\s` (via
//! `unicode_props.isWhitespace`), so NBSP, U+2028 line separators, and
//! the full White_Space=Yes set are honored — matching HF's `regex`
//! library behavior.
//!
//! Reference: `tokenizers/src/tokenizer/added_vocabulary.rs` in
//! huggingface/tokenizers. The `single_word` boundary filter, lstrip,
//! and rstrip are all honored.

const std = @import("std");
const TokenId = @import("token.zig").TokenId;

pub const AddedToken = struct {
    id: TokenId,
    content: []const u8,
    single_word: bool = false,
    lstrip: bool = false,
    rstrip: bool = false,
};

pub const Segment = union(enum) {
    text: struct { start: u32, end: u32 },
    special: struct { id: TokenId, start: u32, end: u32 },
};

pub const Scanner = struct {
    allocator: std.mem.Allocator,
    tokens: []AddedToken,
    arena: []u8, // owns all token content bytes back-to-back
    nodes: []Node,
    children: []Child,

    pub const Node = struct {
        token_idx: i32, // -1 if not terminal; else index into `tokens`
        children_start: u32,
        children_len: u32,
    };
    pub const Child = struct { byte: u8, node: u32 };

    pub fn init(allocator: std.mem.Allocator, tokens_in: []const AddedToken) !Scanner {
        // Empty scanner: still need a one-node root so `scan` can no-op.
        if (tokens_in.len == 0) {
            const tokens = try allocator.alloc(AddedToken, 0);
            errdefer allocator.free(tokens);
            const arena = try allocator.alloc(u8, 0);
            errdefer allocator.free(arena);
            const nodes = try allocator.alloc(Node, 1);
            errdefer allocator.free(nodes);
            nodes[0] = .{ .token_idx = -1, .children_start = 0, .children_len = 0 };
            const children = try allocator.alloc(Child, 0);
            return .{
                .allocator = allocator,
                .tokens = tokens,
                .arena = arena,
                .nodes = nodes,
                .children = children,
            };
        }

        // Copy content bytes into one owned arena, point per-token slices into it.
        var arena_len: usize = 0;
        for (tokens_in) |t| arena_len += t.content.len;
        const arena = try allocator.alloc(u8, arena_len);
        errdefer allocator.free(arena);

        const tokens = try allocator.alloc(AddedToken, tokens_in.len);
        errdefer allocator.free(tokens);

        var off: usize = 0;
        for (tokens_in, 0..) |t, i| {
            @memcpy(arena[off .. off + t.content.len], t.content);
            tokens[i] = .{
                .id = t.id,
                .content = arena[off .. off + t.content.len],
                .single_word = t.single_word,
                .lstrip = t.lstrip,
                .rstrip = t.rstrip,
            };
            off += t.content.len;
        }

        // Build the trie. Use the intermediate growable form from
        // unigram.zig in spirit — collapse to flat SoA at the end so the
        // hot path stays cache-friendly.
        var nodes_b: std.ArrayList(BuildNode) = .empty;
        defer {
            for (nodes_b.items) |*n| n.children.deinit(allocator);
            nodes_b.deinit(allocator);
        }
        try nodes_b.append(allocator, .{ .token_idx = -1, .children = .empty });

        for (tokens, 0..) |t, i| {
            if (t.content.len == 0) continue; // skip empty content defensively
            var cur: u32 = 0;
            for (t.content) |b| {
                cur = try descendOrCreate(allocator, &nodes_b, cur, b);
            }
            // Last writer wins on duplicate content; HF's id assignment is
            // the source of truth, and duplicates would be a vocab bug.
            nodes_b.items[cur].token_idx = @intCast(i);
        }

        const node_count = nodes_b.items.len;
        var total_children: usize = 0;
        for (nodes_b.items) |n| total_children += n.children.items.len;

        const flat_nodes = try allocator.alloc(Node, node_count);
        errdefer allocator.free(flat_nodes);
        const flat_children = try allocator.alloc(Child, total_children);
        errdefer allocator.free(flat_children);

        var write: u32 = 0;
        for (nodes_b.items, 0..) |n, idx| {
            const len: u32 = @intCast(n.children.items.len);
            flat_nodes[idx] = .{
                .token_idx = n.token_idx,
                .children_start = write,
                .children_len = len,
            };
            @memcpy(flat_children[write .. write + len], n.children.items);
            write += len;
        }

        return .{
            .allocator = allocator,
            .tokens = tokens,
            .arena = arena,
            .nodes = flat_nodes,
            .children = flat_children,
        };
    }

    pub fn deinit(self: *Scanner) void {
        if (self.tokens.len > 0) self.allocator.free(self.tokens);
        if (self.arena.len > 0) self.allocator.free(self.arena);
        self.allocator.free(self.nodes);
        if (self.children.len > 0) self.allocator.free(self.children);
        self.* = undefined;
    }

    fn findChild(self: *const Scanner, node: u32, byte: u8) ?u32 {
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

    /// Find the next byte that can begin any added token. Common HF
    /// vocabularies have one root byte (`<` or `[`), so this lowers to the
    /// standard library's vectorized memchr instead of probing the trie at
    /// every corpus byte.
    fn nextRootCandidate(self: *const Scanner, input: []const u8, start: u32) ?u32 {
        const root = self.nodes[0];
        if (root.children_len == 0) return null;
        const kids = self.children[root.children_start .. root.children_start + root.children_len];
        if (kids.len <= 4) {
            var best: ?usize = null;
            for (kids) |kid| {
                const found = std.mem.indexOfScalarPos(u8, input, start, kid.byte) orelse continue;
                best = if (best) |current| @min(current, found) else found;
            }
            return if (best) |found| @intCast(found) else null;
        }

        // Unusual many-root vocabulary: retain the exact scalar fallback.
        var pos: usize = start;
        while (pos < input.len) : (pos += 1) {
            if (self.findChild(0, input[pos]) != null) return @intCast(pos);
        }
        return null;
    }

    /// Whether independently scanning disjoint chunks preserves added-token
    /// whitespace semantics. Strip-enabled tokens need overlap/state across a
    /// cut and stay on the conservative whole-input scan path.
    pub fn supportsIndependentChunks(self: *const Scanner) bool {
        for (self.tokens) |token| {
            if (token.lstrip or token.rstrip) return false;
        }
        return true;
    }

    /// True when no added-token byte string straddles `pos`. Call only after
    /// `supportsIndependentChunks`; matching or rejected `single_word`
    /// occurrences are both treated conservatively.
    pub fn isIndependentCut(self: *const Scanner, input: []const u8, pos: usize) bool {
        if (pos == 0 or pos == input.len) return true;
        for (self.tokens) |token| {
            const len = token.content.len;
            if (len < 2) continue;
            const first = pos -| (len - 1);
            const last_exclusive = @min(pos, input.len -| len) + 1;
            // A token spelling longer than the entire input cannot straddle
            // this cut. Saturating arithmetic above may otherwise produce an
            // inverted range near the end of very short inputs.
            if (first >= last_exclusive) continue;
            for (first..last_exclusive) |start| {
                const end = start + len;
                if (start < pos and end > pos and end <= input.len and
                    std.mem.eql(u8, input[start..end], token.content)) return false;
            }
        }
        return true;
    }
};

const BuildNode = struct {
    token_idx: i32,
    children: std.ArrayList(Scanner.Child),
};

fn descendOrCreate(
    allocator: std.mem.Allocator,
    nodes: *std.ArrayList(BuildNode),
    parent: u32,
    byte: u8,
) !u32 {
    // Search existing children first; the appended-node path may realloc
    // `nodes.items`, so we must not hold a pointer across it.
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
    try nodes.append(allocator, .{ .token_idx = -1, .children = .empty });

    const kids_ptr = &nodes.items[parent].children;
    var l2: usize = 0;
    var r2: usize = kids_ptr.items.len;
    while (l2 < r2) {
        const m = l2 + (r2 - l2) / 2;
        const b = kids_ptr.items[m].byte;
        if (b < byte) l2 = m + 1 else r2 = m;
    }
    try kids_ptr.insert(allocator, l2, .{ .byte = byte, .node = new_idx });
    return new_idx;
}

const unicode_props = @import("unicode_props.zig");

fn isUnicodeWs(cp: u21) bool {
    if (cp < 0x80) return cp == ' ' or cp == '\t' or cp == '\n' or cp == '\r' or cp == 0x0B or cp == 0x0C;
    return unicode_props.isWhitespace(cp);
}

const Decoded = struct { cp: u21, len: u32 };
fn decodeOne(s: []const u8, i: u32) ?Decoded {
    if (i >= s.len) return null;
    const b0 = s[i];
    if (b0 < 0x80) return .{ .cp = b0, .len = 1 };
    const seq_len = std.unicode.utf8ByteSequenceLength(b0) catch return .{ .cp = b0, .len = 1 };
    if (@as(usize, i) + seq_len > s.len) return .{ .cp = b0, .len = 1 };
    const cp = std.unicode.utf8Decode(s[i .. i + seq_len]) catch return .{ .cp = b0, .len = 1 };
    return .{ .cp = cp, .len = seq_len };
}

/// Find the start byte position of the codepoint whose LAST byte is at
/// `end-1`. Walks back at most 4 bytes (max UTF-8 sequence length).
fn utf8RevStart(s: []const u8, low: u32, end: u32) u32 {
    if (end <= low) return low;
    var p: u32 = end - 1;
    var k: u32 = 0;
    while (p > low and (s[p] & 0xC0) == 0x80 and k < 3) : (k += 1) p -= 1;
    return p;
}

fn isWordCp(cp: u21) bool {
    // ASCII fast path.
    if (cp < 0x80) {
        return (cp >= 'A' and cp <= 'Z') or (cp >= 'a' and cp <= 'z') or
            (cp >= '0' and cp <= '9') or cp == '_';
    }
    return unicode_props.isLetter(cp) or unicode_props.isNumber(cp);
}

pub fn scan(
    self: *const Scanner,
    allocator: std.mem.Allocator,
    input: []const u8,
) ![]Segment {
    // Fast path: nothing to match. One text segment (or zero if empty).
    if (self.tokens.len == 0) {
        if (input.len == 0) return try allocator.alloc(Segment, 0);
        const out = try allocator.alloc(Segment, 1);
        out[0] = .{ .text = .{ .start = 0, .end = @intCast(input.len) } };
        return out;
    }
    if (input.len == 0) return try allocator.alloc(Segment, 0);

    // Added tokens are normally sparse. Reserving the theoretical 2*N+1
    // worst case on a dataset-sized input creates a multi-gigabyte virtual
    // allocation just to return one text segment; grow only when matches
    // actually occur.
    var segs: std.ArrayList(Segment) = .empty;
    defer segs.deinit(allocator);
    try segs.ensureTotalCapacity(allocator, 64);

    var i: u32 = 0;
    var text_start: u32 = 0;
    const n: u32 = @intCast(input.len);

    while (i < n) {
        i = self.nextRootCandidate(input, i) orelse break;
        // Walk the trie from position i, recording the longest accepted hit.
        var node: u32 = 0;
        var j: u32 = i;
        var best_len: u32 = 0;
        var best_tok: i32 = -1;
        while (j < n) {
            const next = self.findChild(node, input[j]) orelse break;
            node = next;
            j += 1;
            const ti = self.nodes[node].token_idx;
            if (ti >= 0) {
                const idx_u: usize = @intCast(ti);
                const tok = self.tokens[idx_u];
                // single_word boundary check: surrounding bytes must be
                // non-word (or absent). Reject this length but keep walking
                // — a longer match further down might be accepted.
                if (tok.single_word) {
                    // Walk one codepoint back from `i` and one forward
                    // from `j`. Outside the input bounds counts as "not
                    // a word character" (boundary OK).
                    const left_ok = blk: {
                        if (i == 0) break :blk true;
                        const cp_start = utf8RevStart(input, 0, i);
                        const d = decodeOne(input, cp_start) orelse break :blk true;
                        break :blk !isWordCp(d.cp);
                    };
                    const right_ok = blk: {
                        if (j == n) break :blk true;
                        const d = decodeOne(input, j) orelse break :blk true;
                        break :blk !isWordCp(d.cp);
                    };
                    if (!(left_ok and right_ok)) continue;
                }
                best_len = j - i;
                best_tok = ti;
            }
        }

        if (best_len > 0) {
            const idx_u: usize = @intCast(best_tok);
            const tok = self.tokens[idx_u];

            // lstrip: absorb the run of Unicode whitespace immediately
            // before the match by shrinking the preceding text segment.
            // We walk codepoints backwards via prefix re-decode since
            // UTF-8 lookbehind has no constant-time analog.
            var text_end: u32 = i;
            if (tok.lstrip) {
                while (text_end > text_start) {
                    const cp_start = utf8RevStart(input, text_start, text_end);
                    const d = decodeOne(input, cp_start) orelse break;
                    if (!isUnicodeWs(d.cp)) break;
                    text_end = cp_start;
                }
            }
            if (text_end > text_start) {
                try segs.append(allocator, .{ .text = .{ .start = text_start, .end = text_end } });
            }
            // Special's source range absorbs lstrip'd ws on the left and rstrip'd
            // ws on the right; record it before/after we advance `i`.
            const special_start: u32 = text_end;
            i += best_len;
            if (tok.rstrip) {
                while (i < n) {
                    const d = decodeOne(input, i) orelse break;
                    if (!isUnicodeWs(d.cp)) break;
                    i += d.len;
                }
            }
            try segs.append(allocator, .{ .special = .{ .id = tok.id, .start = special_start, .end = i } });
            text_start = i;
        } else {
            i += 1;
        }
    }

    if (text_start < n) {
        try segs.append(allocator, .{ .text = .{ .start = text_start, .end = n } });
    }

    return segs.toOwnedSlice(allocator);
}

// --- tests ---

test "empty scanner returns single text segment" {
    var s = try Scanner.init(std.testing.allocator, &.{});
    defer s.deinit();
    const segs = try scan(&s, std.testing.allocator, "hello");
    defer std.testing.allocator.free(segs);
    try std.testing.expectEqual(@as(usize, 1), segs.len);
    try std.testing.expect(segs[0] == .text);
    try std.testing.expectEqual(@as(u32, 0), segs[0].text.start);
    try std.testing.expectEqual(@as(u32, 5), segs[0].text.end);
}

test "single special match" {
    const toks = [_]AddedToken{
        .{ .id = 50256, .content = "<|endoftext|>" },
    };
    var s = try Scanner.init(std.testing.allocator, &toks);
    defer s.deinit();
    const segs = try scan(&s, std.testing.allocator, "Hello <|endoftext|> world");
    defer std.testing.allocator.free(segs);
    try std.testing.expectEqual(@as(usize, 3), segs.len);
    try std.testing.expect(segs[0] == .text);
    try std.testing.expectEqual(@as(u32, 0), segs[0].text.start);
    try std.testing.expectEqual(@as(u32, 6), segs[0].text.end);
    try std.testing.expect(segs[1] == .special);
    try std.testing.expectEqual(@as(TokenId, 50256), segs[1].special.id);
    try std.testing.expectEqual(@as(u32, 6), segs[1].special.start);
    try std.testing.expectEqual(@as(u32, 19), segs[1].special.end);
    try std.testing.expect(segs[2] == .text);
    try std.testing.expectEqual(@as(u32, 19), segs[2].text.start);
    try std.testing.expectEqual(@as(u32, 25), segs[2].text.end);
}

test "two adjacent specials" {
    const toks = [_]AddedToken{
        .{ .id = 50256, .content = "<|endoftext|>" },
    };
    var s = try Scanner.init(std.testing.allocator, &toks);
    defer s.deinit();
    const segs = try scan(&s, std.testing.allocator, "<|endoftext|><|endoftext|>");
    defer std.testing.allocator.free(segs);
    try std.testing.expectEqual(@as(usize, 2), segs.len);
    try std.testing.expect(segs[0] == .special);
    try std.testing.expectEqual(@as(TokenId, 50256), segs[0].special.id);
    try std.testing.expect(segs[1] == .special);
    try std.testing.expectEqual(@as(TokenId, 50256), segs[1].special.id);
}

test "independent chunk cuts reject a straddling added token" {
    const toks = [_]AddedToken{
        .{ .id = 7, .content = "<|special|>" },
    };
    var scanner = try Scanner.init(std.testing.allocator, &toks);
    defer scanner.deinit();
    const input = "left <|special|> right";
    try std.testing.expect(scanner.supportsIndependentChunks());
    try std.testing.expect(!scanner.isIndependentCut(input, 10));
    try std.testing.expect(scanner.isIndependentCut(input, 5));
}

test "independent cut handles token spelling longer than input" {
    const toks = [_]AddedToken{
        .{ .id = 7, .content = "<|a-very-long-special-token|>" },
    };
    var scanner = try Scanner.init(std.testing.allocator, &toks);
    defer scanner.deinit();
    try std.testing.expect(scanner.isIndependentCut("tiny", 3));
}

test "strip-enabled added tokens require conservative whole-input scan" {
    const toks = [_]AddedToken{
        .{ .id = 7, .content = "<s>", .lstrip = true },
    };
    var scanner = try Scanner.init(std.testing.allocator, &toks);
    defer scanner.deinit();
    try std.testing.expect(!scanner.supportsIndependentChunks());
}

test "longest-match wins" {
    const toks = [_]AddedToken{
        .{ .id = 1, .content = "<|im_" },
        .{ .id = 2, .content = "<|im_start|>" },
    };
    var s = try Scanner.init(std.testing.allocator, &toks);
    defer s.deinit();
    const segs = try scan(&s, std.testing.allocator, "<|im_start|>");
    defer std.testing.allocator.free(segs);
    try std.testing.expectEqual(@as(usize, 1), segs.len);
    try std.testing.expect(segs[0] == .special);
    try std.testing.expectEqual(@as(TokenId, 2), segs[0].special.id);
}

test "single_word filter rejects in-word match" {
    const toks = [_]AddedToken{
        .{ .id = 5, .content = "gpt", .single_word = true },
    };
    var s = try Scanner.init(std.testing.allocator, &toks);
    defer s.deinit();
    const segs = try scan(&s, std.testing.allocator, "chatgpt-3");
    defer std.testing.allocator.free(segs);
    try std.testing.expectEqual(@as(usize, 1), segs.len);
    try std.testing.expect(segs[0] == .text);
    try std.testing.expectEqual(@as(u32, 0), segs[0].text.start);
    try std.testing.expectEqual(@as(u32, 9), segs[0].text.end);
}

test "single_word filter accepts boundary match" {
    const toks = [_]AddedToken{
        .{ .id = 5, .content = "gpt", .single_word = true },
    };
    var s = try Scanner.init(std.testing.allocator, &toks);
    defer s.deinit();
    const segs = try scan(&s, std.testing.allocator, "chat gpt 4");
    defer std.testing.allocator.free(segs);
    try std.testing.expectEqual(@as(usize, 3), segs.len);
    try std.testing.expect(segs[0] == .text);
    try std.testing.expectEqual(@as(u32, 0), segs[0].text.start);
    try std.testing.expectEqual(@as(u32, 5), segs[0].text.end);
    try std.testing.expect(segs[1] == .special);
    try std.testing.expectEqual(@as(TokenId, 5), segs[1].special.id);
    try std.testing.expect(segs[2] == .text);
    try std.testing.expectEqual(@as(u32, 8), segs[2].text.start);
    try std.testing.expectEqual(@as(u32, 10), segs[2].text.end);
}

test "single_word filter recognizes Unicode letters as word chars" {
    // "café<TOK>" — the 'é' before <TOK> is a Unicode letter, so the
    // single_word filter must REJECT this match (no word boundary).
    const toks = [_]AddedToken{
        .{ .id = 7, .content = "TOK", .single_word = true },
    };
    var s = try Scanner.init(std.testing.allocator, &toks);
    defer s.deinit();
    const input = "caféTOK ok";
    const segs = try scan(&s, std.testing.allocator, input);
    defer std.testing.allocator.free(segs);
    // No match → single text segment.
    try std.testing.expectEqual(@as(usize, 1), segs.len);
    try std.testing.expect(segs[0] == .text);
}

test "single_word filter accepts Unicode boundary" {
    // "我 TOK 你" — Chinese letter then space then TOK then space then Chinese.
    // Spaces are non-word, so single_word accepts the TOK match.
    const toks = [_]AddedToken{
        .{ .id = 8, .content = "TOK", .single_word = true },
    };
    var s = try Scanner.init(std.testing.allocator, &toks);
    defer s.deinit();
    const input = "我 TOK 你";
    const segs = try scan(&s, std.testing.allocator, input);
    defer std.testing.allocator.free(segs);
    try std.testing.expectEqual(@as(usize, 3), segs.len);
    try std.testing.expect(segs[1] == .special);
    try std.testing.expectEqual(@as(TokenId, 8), segs[1].special.id);
}

test "non-overlap: trailing partial doesn't false-match" {
    const toks = [_]AddedToken{
        .{ .id = 1, .content = "abc" },
    };
    var s = try Scanner.init(std.testing.allocator, &toks);
    defer s.deinit();
    const segs = try scan(&s, std.testing.allocator, "abx");
    defer std.testing.allocator.free(segs);
    try std.testing.expectEqual(@as(usize, 1), segs.len);
    try std.testing.expect(segs[0] == .text);
    try std.testing.expectEqual(@as(u32, 0), segs[0].text.start);
    try std.testing.expectEqual(@as(u32, 3), segs[0].text.end);
}

test "lstrip absorbs preceding whitespace" {
    const toks = [_]AddedToken{
        .{ .id = 7, .content = "<s>", .lstrip = true },
    };
    var s = try Scanner.init(std.testing.allocator, &toks);
    defer s.deinit();
    const segs = try scan(&s, std.testing.allocator, "hi   <s> world");
    defer std.testing.allocator.free(segs);
    try std.testing.expectEqual(@as(usize, 3), segs.len);
    // text "hi" (whitespace eaten by lstrip), then <s>, then " world"
    try std.testing.expectEqualStrings("hi", "hi   <s> world"[segs[0].text.start..segs[0].text.end]);
    try std.testing.expectEqual(@as(TokenId, 7), segs[1].special.id);
    try std.testing.expectEqualStrings(" world", "hi   <s> world"[segs[2].text.start..segs[2].text.end]);
}

test "rstrip absorbs following whitespace" {
    const toks = [_]AddedToken{
        .{ .id = 9, .content = "<s>", .rstrip = true },
    };
    var s = try Scanner.init(std.testing.allocator, &toks);
    defer s.deinit();
    const segs = try scan(&s, std.testing.allocator, "hi <s>   world");
    defer std.testing.allocator.free(segs);
    try std.testing.expectEqual(@as(usize, 3), segs.len);
    try std.testing.expectEqualStrings("hi ", "hi <s>   world"[segs[0].text.start..segs[0].text.end]);
    try std.testing.expectEqual(@as(TokenId, 9), segs[1].special.id);
    try std.testing.expectEqualStrings("world", "hi <s>   world"[segs[2].text.start..segs[2].text.end]);
}

test "lstrip + rstrip compose" {
    const toks = [_]AddedToken{
        .{ .id = 11, .content = "<x>", .lstrip = true, .rstrip = true },
    };
    var s = try Scanner.init(std.testing.allocator, &toks);
    defer s.deinit();
    const segs = try scan(&s, std.testing.allocator, "a   <x>   b");
    defer std.testing.allocator.free(segs);
    try std.testing.expectEqual(@as(usize, 3), segs.len);
    try std.testing.expectEqualStrings("a", "a   <x>   b"[segs[0].text.start..segs[0].text.end]);
    try std.testing.expectEqual(@as(TokenId, 11), segs[1].special.id);
    try std.testing.expectEqualStrings("b", "a   <x>   b"[segs[2].text.start..segs[2].text.end]);
}

test "lstrip without preceding text leaves empty prefix" {
    const toks = [_]AddedToken{
        .{ .id = 13, .content = "<s>", .lstrip = true },
    };
    var s = try Scanner.init(std.testing.allocator, &toks);
    defer s.deinit();
    const segs = try scan(&s, std.testing.allocator, "   <s>hi");
    defer std.testing.allocator.free(segs);
    // 3 leading spaces all absorbed; only special + " hi"... wait there's no leading space before hi
    // segs should be: special "<s>", text "hi"
    try std.testing.expectEqual(@as(usize, 2), segs.len);
    try std.testing.expectEqual(@as(TokenId, 13), segs[0].special.id);
    try std.testing.expectEqualStrings("hi", "   <s>hi"[segs[1].text.start..segs[1].text.end]);
}
