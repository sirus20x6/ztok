//! Bounded regex compiler + interpreter for HF tokenizer.json pretokenizers.
//!
//! HF `pre_tokenizer.Split` carries arbitrary regex patterns; full PCRE is
//! a non-goal — we only support the subset HF tokenizers actually use:
//!
//!   Literals:           any byte / Unicode codepoint
//!   Character classes:  [abc], [^abc], ranges `[a-z]`, named escapes
//!                       (`\d`, `\D`, `\w`, `\W`, `\s`, `\S`),
//!                       `\p{L}`, `\p{N}`, `\p{P}`, `\p{M}`, `\p{S}`, `\p{Z}`
//!                       and `\P{...}` negations
//!   Quantifiers:        `*`, `+`, `?`, `{N}`, `{N,M}`, `{N,}` (greedy only —
//!                       HF patterns never need lazy quantifiers; we treat
//!                       `*?` as greedy and document)
//!   Groups:             `(?:...)` non-capturing, `(?i:...)` case-insensitive
//!                       inline-flag group, `(...)` bare capturing groups
//!                       are accepted but treated as non-capturing (we don't
//!                       expose capture groups — HF Split only needs match
//!                       boundaries)
//!   Alternation:        `a|b|c`  — leftmost-first NFA semantics
//!   Anchors:            `^` and `$` (start/end of input only; no multiline)
//!   Special escapes:    `\n`, `\r`, `\t`, `\0`, `\\`, `\.`, `\(`, etc.
//!   Lookahead:          `(?!...)` and `(?=...)` zero-width assertions
//!                       (GPT-2's `\s+(?!\S)` needs this)
//!
//! Compile target: a flat array of `Inst` (Pike-style program); evaluation
//! uses backtracking with NFA semantics — leftmost-first alternation, greedy
//! quantifiers. The HF patterns we care about are small (a few dozen ops);
//! backtracking matches what the upstream `regex` crate produces for these
//! inputs at the byte boundaries we care about.
//!
//! Intentionally NOT supported (route to follow-up if a future model needs):
//!   - Backreferences (`\1`).
//!   - Lookbehind (`(?<=...)`, `(?<!...)`).
//!   - Possessive quantifiers (`a++`, `a*+`).
//!   - `\b` / `\B` word boundaries.
//!   - Unicode script properties beyond the L/N/P/M/S/Z categories.

const std = @import("std");
const unicode_props = @import("unicode_props.zig");

// ---------------------------------------------------------------------
// Public surface

pub const CompileError = error{
    UnexpectedEnd,
    UnexpectedChar,
    InvalidEscape,
    InvalidQuantifier,
    InvalidCharClass,
    InvalidGroupFlags,
} || std.mem.Allocator.Error;

/// Compiled regex program. Owns its bytecode + any class range slices via
/// `allocator`.
pub const Regex = struct {
    allocator: std.mem.Allocator,
    insts: []Inst,
    /// True if the top-level pattern is anchored at `^`. We strip the
    /// anchor at compile time and short-circuit `findFirst` accordingly.
    anchored_start: bool = false,

    pub fn deinit(self: *Regex) void {
        for (self.insts) |inst| {
            if (inst.op == .class) {
                if (inst.cls.ranges.len > 0) self.allocator.free(@constCast(inst.cls.ranges));
            }
        }
        self.allocator.free(self.insts);
    }

    /// Find the first match starting at or after `start`. Returns the
    /// half-open `[begin, end)` byte range; null if no match.
    pub fn findFirst(self: *const Regex, input: []const u8, start: usize) ?Match {
        if (self.anchored_start) {
            if (start != 0) return null;
            if (match(self.insts, input, 0)) |end| {
                return .{ .start = 0, .end = end };
            }
            return null;
        }
        var i: usize = start;
        while (i <= input.len) {
            if (match(self.insts, input, i)) |end| {
                return .{ .start = i, .end = end };
            }
            if (i == input.len) break;
            // Advance by one codepoint (so we don't split mid-UTF8).
            const d = decodeAt(input, i) orelse {
                i += 1;
                continue;
            };
            i += d.len;
        }
        return null;
    }

    /// Match anchored at `pos`. Returns the end byte of the longest
    /// leftmost match starting there; null if no match anchored at `pos`.
    pub fn matchAt(self: *const Regex, input: []const u8, pos: usize) ?usize {
        return match(self.insts, input, pos);
    }

    /// Prefix-acceptance classification for grammar-constrained generation.
    ///
    /// Treats the whole pattern as anchored at byte 0 (the natural reading
    /// for "does this generated text obey the grammar so far") and asks
    /// whether `input` can still grow into a full match:
    ///
    ///   .full    — `input` is itself a complete match (reaches `.match`
    ///              with the cursor at `input.len`). Generation MAY stop.
    ///   .partial — `input` matches no complete string yet, but at least
    ///              one live NFA thread is still alive at `input.len`, so
    ///              some continuation could complete the match.
    ///   .dead    — no live thread survives consuming `input`; no suffix
    ///              can rescue it. The byte sequence has left the grammar.
    ///
    /// Implementation note: this is a Thompson NFA simulation over the same
    /// compiled `Inst` program used by `match`, run codepoint-by-codepoint.
    /// Unlike the backtracking `match` (which only answers "is there a full
    /// match?"), the thread-set walk lets us detect the live/dead frontier
    /// in a single left-to-right pass — O(insts * input) with no
    /// backtracking blowup. Lookahead assertions are evaluated against the
    /// CURRENTLY KNOWN bytes only; a positive lookahead that needs unseen
    /// bytes is treated optimistically as still-live (it cannot be refuted
    /// until those bytes arrive), which is the correct conservative answer
    /// for prefix acceptance.
    pub fn prefixStatus(self: *const Regex, input: []const u8) PrefixStatus {
        return prefixWalk(self.insts, input);
    }
};

pub const PrefixStatus = enum { dead, partial, full };

pub const Match = struct { start: usize, end: usize };

/// Compile a pattern. Returns an owned `Regex` the caller must `deinit`.
pub fn compile(allocator: std.mem.Allocator, pattern: []const u8) CompileError!Regex {
    var p: Parser = .{
        .src = pattern,
        .pos = 0,
        .allocator = allocator,
        .insts = .empty,
        .case_insensitive_depth = 0,
    };
    errdefer {
        for (p.insts.items) |inst| {
            if (inst.op == .class) {
                if (inst.cls.ranges.len > 0) allocator.free(@constCast(inst.cls.ranges));
            }
        }
        p.insts.deinit(allocator);
    }

    var anchored = false;
    if (p.peek() == @as(?u8, '^')) {
        anchored = true;
        p.pos += 1;
    }

    try compileAlt(&p);
    try p.emit(.{ .op = .match });

    if (p.pos != pattern.len) return error.UnexpectedChar;

    return .{
        .allocator = allocator,
        .insts = try p.insts.toOwnedSlice(allocator),
        .anchored_start = anchored,
    };
}

// ---------------------------------------------------------------------
// IR

const Inst = struct {
    op: Op,
    target: u32 = 0,
    target_b: u32 = 0,
    cp: u21 = 0,
    cls: CharClass = .{},
};

const Op = enum {
    /// `match` — terminate with success at the current position.
    match,
    /// `char` — match a single codepoint equal to `cp`.
    char,
    /// `char_ci` — case-insensitive single ASCII codepoint match. `cp`
    /// is stored lowercased; the matcher lowercases the input byte
    /// before comparing.
    char_ci,
    /// `any_char` — match any single codepoint (UTF-8 decode at runtime).
    any_char,
    /// `class` — match a codepoint against an embedded character class.
    class,
    /// `jmp` — unconditional jump to `target`.
    jmp,
    /// `split` — try `target` first; on failure, try `target_b`.
    split,
    /// `anchor_start` — succeed iff the cursor is at position 0.
    anchor_start,
    /// `anchor_end` — succeed iff the cursor is at `input.len`.
    anchor_end,
    /// `assert` — zero-width lookahead. `target` = body start (body
    /// terminates with `assert_end`), `target_b` = post-assertion PC.
    /// `cp = 0` for positive lookahead, `1` for negative.
    assert,
    /// `assert_end` — terminator emitted at the end of a lookahead body;
    /// the engine treats it like `match` while evaluating an assertion.
    assert_end,
};

const CharClass = struct {
    ascii: AsciiSet = .{},
    ranges: []const Range = &.{},
    cats: CatMask = .{},
    negated: bool = false,
};

const Range = struct { lo: u21, hi: u21 };

const AsciiSet = packed struct {
    a: u64 = 0,
    b: u64 = 0,

    fn add(self: *AsciiSet, b: u8) void {
        if (b < 64) {
            self.a |= @as(u64, 1) << @intCast(b);
        } else if (b < 128) {
            self.b |= @as(u64, 1) << @intCast(b - 64);
        }
    }
    fn contains(self: AsciiSet, b: u8) bool {
        if (b < 64) return (self.a & (@as(u64, 1) << @intCast(b))) != 0;
        if (b < 128) return (self.b & (@as(u64, 1) << @intCast(b - 64))) != 0;
        return false;
    }
    fn addRange(self: *AsciiSet, lo: u8, hi: u8) void {
        var c: u16 = lo;
        while (c <= hi) : (c += 1) self.add(@intCast(c));
    }
};

const CatMask = packed struct {
    letter: bool = false,
    number: bool = false,
    mark: bool = false,
    whitespace: bool = false,
    punctuation: bool = false,
    symbol: bool = false,
    separator: bool = false,

    fn any(self: CatMask) bool {
        return self.letter or self.number or self.mark or self.whitespace or
            self.punctuation or self.symbol or self.separator;
    }
};

// ---------------------------------------------------------------------
// Parser
//
// Emits a Pike-style program. Every control-flow construct uses two-pass
// patching: emit a placeholder target, remember the patch index, fix it
// after the next-known address.

const Parser = struct {
    src: []const u8,
    pos: usize,
    allocator: std.mem.Allocator,
    insts: std.ArrayList(Inst),
    case_insensitive_depth: u32,

    fn peek(self: *const Parser) ?u8 {
        if (self.pos >= self.src.len) return null;
        return self.src[self.pos];
    }
    fn peekAt(self: *const Parser, k: usize) ?u8 {
        if (self.pos + k >= self.src.len) return null;
        return self.src[self.pos + k];
    }
    fn eat(self: *Parser) ?u8 {
        if (self.pos >= self.src.len) return null;
        const c = self.src[self.pos];
        self.pos += 1;
        return c;
    }
    fn emit(self: *Parser, inst: Inst) !void {
        try self.insts.append(self.allocator, inst);
    }
    fn here(self: *const Parser) u32 {
        return @intCast(self.insts.items.len);
    }
};

/// Compile an alternation. Emits standard NFA layout:
///   split L1, L2_split
///   L1: <branch1>
///   jmp END
///   L2_split: split L2, L3_split
///   L2: <branch2>
///   jmp END
///   ...
///   Llast: <branchN>
///   END:
///
/// For a single-alternative pattern (no `|`), we just emit the branch
/// inline — no splits, no jumps.
fn compileAlt(p: *Parser) CompileError!void {
    // Detect whether there's any '|' at the top level of the upcoming
    // chunk. Easiest way: parse the first branch, then loop.
    const branch1_start = p.here();
    try compileConcat(p);

    if (p.peek() != @as(?u8, '|')) return;

    // We have at least one `|`. Materialize the standard layout by
    // RE-EMITTING with splits + jumps. To avoid the cost of re-parsing,
    // we instead splice in the head split and a tail jmp.
    //
    // Strategy: collect ALL branch bodies as parsed fragments first by
    // saving each branch's [start..end) byte ranges in the instruction
    // stream and the trailing `jmp` placeholders. After all branches are
    // parsed, we know the END address and can patch the jumps and splits.

    var branch_starts: std.ArrayList(u32) = .empty;
    defer branch_starts.deinit(p.allocator);
    try branch_starts.append(p.allocator, branch1_start);

    var jmp_indices: std.ArrayList(u32) = .empty;
    defer jmp_indices.deinit(p.allocator);

    // Emit a trailing jmp at the end of branch 1.
    try jmp_indices.append(p.allocator, p.here());
    try p.emit(.{ .op = .jmp, .target = 0xFFFF_FFFF });

    while (p.peek() == @as(?u8, '|')) {
        _ = p.eat();
        try branch_starts.append(p.allocator, p.here());
        try compileConcat(p);
        try jmp_indices.append(p.allocator, p.here());
        try p.emit(.{ .op = .jmp, .target = 0xFFFF_FFFF });
    }

    // We have N branches. We need to insert N-1 `split` instructions and
    // re-route execution so that branch_starts[i] gets reached via the
    // i-th split (which falls through to the next split, etc.).
    //
    // To avoid shifting downstream targets, we instead rewrite the head
    // by SHIFTING the entire stream right by (N-1) split slots. We track
    // every absolute target (`target` and `target_b` fields) and add the
    // shift if it points at or past `branch1_start`.

    const n_branches = branch_starts.items.len;
    const n_splits: u32 = @intCast(n_branches - 1);
    const shift: u32 = n_splits;
    const old_len: u32 = @intCast(p.insts.items.len);

    // Make room at the head of the alternation.
    try p.insts.ensureUnusedCapacity(p.allocator, n_splits);
    p.insts.items.len += n_splits;
    // Move tail right.
    {
        var i: usize = old_len;
        while (i > branch1_start) : (i -= 1) {
            p.insts.items[i + n_splits - 1] = p.insts.items[i - 1];
        }
    }
    // Shift all absolute targets that landed at or past branch1_start.
    for (p.insts.items[branch1_start + n_splits ..]) |*it| {
        if (it.op == .jmp or it.op == .split or it.op == .assert) {
            if (it.target != 0xFFFF_FFFF and it.target >= branch1_start) it.target += shift;
            if ((it.op == .split or it.op == .assert) and it.target_b != 0xFFFF_FFFF and it.target_b >= branch1_start) it.target_b += shift;
        }
    }
    // Same shift for the branch_starts and jmp_indices arrays.
    for (branch_starts.items) |*v| v.* += shift;
    for (jmp_indices.items) |*v| v.* += shift;

    // Now write splits at [branch1_start - shift .. branch1_start). After
    // the shift, the head splits live at [branch1_start .. branch1_start
    // + n_splits) — but we want them at the ORIGINAL branch1_start. Wait:
    // we shifted [branch1_start..old_len) right by `shift` — so the
    // original head address (branch1_start) is now empty. Fill it.
    //
    // Layout:
    //   [branch1_start + 0]            = split branch_starts[0], next_split (=branch1_start+1)
    //   [branch1_start + 1]            = split branch_starts[1], next_split (=branch1_start+2)
    //   ...
    //   [branch1_start + n_splits - 1] = split branch_starts[n-2], branch_starts[n-1]
    {
        var s: usize = 0;
        while (s < n_splits) : (s += 1) {
            const try_branch = branch_starts.items[s];
            const else_target: u32 = if (s + 1 < n_splits)
                branch1_start + @as(u32, @intCast(s + 1))
            else
                branch_starts.items[n_branches - 1];
            p.insts.items[branch1_start + s] = .{
                .op = .split,
                .target = try_branch,
                .target_b = else_target,
            };
        }
    }

    // Patch all jmp stubs to point at END (current here()).
    const end: u32 = @intCast(p.insts.items.len);
    for (jmp_indices.items) |idx| p.insts.items[idx].target = end;
}

fn compileConcat(p: *Parser) CompileError!void {
    while (p.peek()) |c| {
        if (c == '|' or c == ')') break;
        try compileAtomQuantified(p);
    }
}

fn compileAtomQuantified(p: *Parser) CompileError!void {
    const atom_start = p.here();
    try compileAtom(p);
    const atom_end = p.here();

    const q = p.peek() orelse return;
    switch (q) {
        '?' => {
            _ = p.eat();
            if (p.peek() == @as(?u8, '?')) _ = p.eat();
            // Wrap: [split atom_start after][atom...] — insert split at atom_start.
            try insertSplitBefore(p, atom_start, atom_end);
            // The original atom now sits at atom_start+1 .. atom_end+1.
            // The split's target = atom_start+1; target_b = atom_end+1.
            p.insts.items[atom_start].target = atom_start + 1;
            p.insts.items[atom_start].target_b = atom_end + 1;
        },
        '*' => {
            _ = p.eat();
            if (p.peek() == @as(?u8, '?')) _ = p.eat();
            // [split atom after][atom...][jmp split]
            try insertSplitBefore(p, atom_start, atom_end);
            const new_atom_end = atom_end + 1;
            // Emit the jmp back to the split.
            try p.emit(.{ .op = .jmp, .target = atom_start });
            const after = p.here();
            p.insts.items[atom_start].target = atom_start + 1;
            p.insts.items[atom_start].target_b = after;
            _ = new_atom_end;
        },
        '+' => {
            _ = p.eat();
            if (p.peek() == @as(?u8, '?')) _ = p.eat();
            // [atom...][split atom after]
            const split_here = p.here();
            try p.emit(.{ .op = .split, .target = atom_start, .target_b = 0xFFFF_FFFF });
            const after = p.here();
            p.insts.items[split_here].target_b = after;
        },
        '{' => {
            _ = p.eat();
            const n_min = try parseInt(p);
            var n_max: ?u32 = null;
            if (p.peek() == @as(?u8, ',')) {
                _ = p.eat();
                if (p.peek() != @as(?u8, '}')) {
                    n_max = try parseInt(p);
                } else {
                    n_max = null;
                }
            } else {
                n_max = n_min;
            }
            if (p.peek() != @as(?u8, '}')) return error.InvalidQuantifier;
            _ = p.eat();
            if (p.peek() == @as(?u8, '?')) _ = p.eat();
            try emitRepeat(p, atom_start, atom_end, n_min, n_max);
        },
        else => {},
    }
}

fn insertSplitBefore(p: *Parser, atom_start: u32, atom_end: u32) !void {
    _ = atom_end;
    try p.insts.insert(p.allocator, atom_start, .{ .op = .split, .target = 0, .target_b = 0 });
    // Shift all targets that pointed STRICTLY PAST atom_start by 1. A
    // target == atom_start was the "next instruction after the previous
    // op finished" — i.e. the slot we just filled with the new split.
    // The caller wants such pointers to land on the new split (so the
    // outer `+`-split's failure path falls through into the new `?`/`*`
    // split that wraps the atom), so we DO NOT shift them.
    for (p.insts.items, 0..) |*it, idx| {
        if (idx == atom_start) continue;
        if (it.op == .jmp or it.op == .split or it.op == .assert) {
            if (it.target != 0xFFFF_FFFF and it.target > atom_start) it.target += 1;
            if ((it.op == .split or it.op == .assert) and it.target_b != 0xFFFF_FFFF and it.target_b > atom_start) it.target_b += 1;
        }
    }
}

fn parseInt(p: *Parser) !u32 {
    var n: u32 = 0;
    var any = false;
    while (p.peek()) |c| {
        if (c < '0' or c > '9') break;
        n = n * 10 + (c - '0');
        _ = p.eat();
        any = true;
    }
    if (!any) return error.InvalidQuantifier;
    return n;
}

/// Repeat an atom to satisfy `{n_min,n_max}` (n_max=null = unbounded).
///
/// We currently have ONE physical copy of the atom at [atom_start..atom_end).
/// Strategy:
///   - For n_min == 0: turn the existing atom into "split atom after" (optional).
///   - For n_min >= 1: leave the original mandatory; clone (n_min - 1) more
///     copies behind it. After that, for bounded suffix (n_max - n_min),
///     emit a chain of optionals; for unbounded, emit a `*`-style loop.
///
/// Cloning is only safe for atoms with no internal control flow. We
/// detect that case (no split/jmp/assert inside [atom_start..atom_end))
/// and fall back to a runtime error if a grouped atom needs cloning.
fn emitRepeat(
    p: *Parser,
    atom_start: u32,
    atom_end: u32,
    n_min: u32,
    n_max_opt: ?u32,
) !void {
    if (n_max_opt) |mx| {
        if (mx < n_min) return error.InvalidQuantifier;
    }

    // Snapshot the atom bytecode.
    const atom_slice = try p.allocator.dupe(Inst, p.insts.items[atom_start..atom_end]);
    defer p.allocator.free(atom_slice);

    // Check cloneability.
    var has_ctrl_flow = false;
    for (atom_slice) |inst| {
        if (inst.op == .jmp or inst.op == .split or inst.op == .assert) {
            has_ctrl_flow = true;
            break;
        }
    }

    if (n_min == 0 and (n_max_opt == null or n_max_opt.? >= 1)) {
        // Wrap the original in a split (optional).
        try insertSplitBefore(p, atom_start, atom_end);
        p.insts.items[atom_start].target = atom_start + 1;
        const after_atom = atom_end + 1;
        p.insts.items[atom_start].target_b = after_atom;

        if (n_max_opt) |mx| {
            // Emit (mx - 1) more optional copies.
            var k: u32 = 1;
            while (k < mx) : (k += 1) {
                if (has_ctrl_flow) return error.InvalidQuantifier; // can't clone groups
                const sp = p.here();
                try p.emit(.{ .op = .split, .target = 0, .target_b = 0 });
                for (atom_slice) |inst| try p.emit(inst);
                const after = p.here();
                p.insts.items[sp].target = sp + 1;
                p.insts.items[sp].target_b = after;
            }
        } else {
            // Unbounded: emit [split atom after][atom][jmp split].
            if (has_ctrl_flow) return error.InvalidQuantifier;
            const sp = p.here();
            try p.emit(.{ .op = .split, .target = 0, .target_b = 0 });
            for (atom_slice) |inst| try p.emit(inst);
            try p.emit(.{ .op = .jmp, .target = sp });
            const after = p.here();
            p.insts.items[sp].target = sp + 1;
            p.insts.items[sp].target_b = after;
        }
        return;
    }

    if (n_min == 0 and n_max_opt != null and n_max_opt.? == 0) {
        // {0} — turn the atom into nothing. Remove the bytecode.
        p.insts.items.len = atom_start;
        return;
    }

    // n_min >= 1: leave the first copy mandatory; clone (n_min - 1) more.
    var k: u32 = 1;
    while (k < n_min) : (k += 1) {
        if (has_ctrl_flow) return error.InvalidQuantifier;
        for (atom_slice) |inst| try p.emit(inst);
    }

    if (n_max_opt) |mx| {
        const optional = mx - n_min;
        var i: u32 = 0;
        while (i < optional) : (i += 1) {
            if (has_ctrl_flow) return error.InvalidQuantifier;
            const sp = p.here();
            try p.emit(.{ .op = .split, .target = 0, .target_b = 0 });
            for (atom_slice) |inst| try p.emit(inst);
            const after = p.here();
            p.insts.items[sp].target = sp + 1;
            p.insts.items[sp].target_b = after;
        }
    } else {
        // Unbounded tail.
        if (has_ctrl_flow) return error.InvalidQuantifier;
        const sp = p.here();
        try p.emit(.{ .op = .split, .target = 0, .target_b = 0 });
        for (atom_slice) |inst| try p.emit(inst);
        try p.emit(.{ .op = .jmp, .target = sp });
        const after = p.here();
        p.insts.items[sp].target = sp + 1;
        p.insts.items[sp].target_b = after;
    }
}

fn compileAtom(p: *Parser) CompileError!void {
    const c = p.peek() orelse return error.UnexpectedEnd;
    switch (c) {
        '(' => {
            _ = p.eat();
            var is_lookahead: ?bool = null; // null = not LH; false = positive; true = negative
            var case_i = false;
            if (p.peek() == @as(?u8, '?')) {
                _ = p.eat();
                const c2 = p.peek() orelse return error.UnexpectedEnd;
                switch (c2) {
                    ':' => {
                        _ = p.eat();
                    },
                    'i' => {
                        _ = p.eat();
                        if (p.peek() == @as(?u8, ':')) {
                            _ = p.eat();
                            case_i = true;
                        } else return error.InvalidGroupFlags;
                    },
                    '=' => {
                        _ = p.eat();
                        is_lookahead = false;
                    },
                    '!' => {
                        _ = p.eat();
                        is_lookahead = true;
                    },
                    else => return error.InvalidGroupFlags,
                }
            }
            if (case_i) p.case_insensitive_depth += 1;
            defer if (case_i) {
                p.case_insensitive_depth -= 1;
            };

            if (is_lookahead) |neg| {
                const assert_idx = p.here();
                try p.emit(.{ .op = .assert, .target = 0, .target_b = 0, .cp = @intFromBool(neg) });
                const body_start = p.here();
                try compileAlt(p);
                try p.emit(.{ .op = .assert_end });
                const after_body = p.here();
                if (p.eat() != @as(?u8, ')')) return error.UnexpectedChar;
                p.insts.items[assert_idx].target = body_start;
                p.insts.items[assert_idx].target_b = after_body;
            } else {
                try compileAlt(p);
                if (p.eat() != @as(?u8, ')')) return error.UnexpectedChar;
            }
        },
        ')' => return error.UnexpectedChar,
        '|' => return error.UnexpectedChar,
        '[' => try compileClass(p),
        '.' => {
            _ = p.eat();
            try p.emit(.{ .op = .any_char });
        },
        '\\' => {
            _ = p.eat();
            try compileEscape(p);
        },
        '^' => {
            _ = p.eat();
            try p.emit(.{ .op = .anchor_start });
        },
        '$' => {
            _ = p.eat();
            try p.emit(.{ .op = .anchor_end });
        },
        else => try compileLiteralCp(p),
    }
}

fn compileLiteralCp(p: *Parser) !void {
    const c0 = p.peek().?;
    if (c0 < 0x80) {
        _ = p.eat();
        if (p.case_insensitive_depth > 0 and isAsciiLetter(c0)) {
            try p.emit(.{ .op = .char_ci, .cp = toLowerAscii(c0) });
        } else {
            try p.emit(.{ .op = .char, .cp = c0 });
        }
        return;
    }
    const seq_len = std.unicode.utf8ByteSequenceLength(c0) catch return error.InvalidEscape;
    if (p.pos + seq_len > p.src.len) return error.InvalidEscape;
    const cp = std.unicode.utf8Decode(p.src[p.pos .. p.pos + seq_len]) catch return error.InvalidEscape;
    p.pos += seq_len;
    try p.emit(.{ .op = .char, .cp = cp });
}

fn compileEscape(p: *Parser) !void {
    const c = p.eat() orelse return error.InvalidEscape;
    switch (c) {
        'n' => try p.emit(.{ .op = .char, .cp = '\n' }),
        'r' => try p.emit(.{ .op = .char, .cp = '\r' }),
        't' => try p.emit(.{ .op = .char, .cp = '\t' }),
        '0' => try p.emit(.{ .op = .char, .cp = 0 }),
        '\\' => try p.emit(.{ .op = .char, .cp = '\\' }),
        '.', '+', '*', '?', '(', ')', '[', ']', '{', '}', '|', '^', '$', '/', '\'', '"' => try p.emit(.{ .op = .char, .cp = c }),
        'd' => try p.emit(.{ .op = .class, .cls = .{ .cats = .{ .number = true } } }),
        'D' => try p.emit(.{ .op = .class, .cls = .{ .cats = .{ .number = true }, .negated = true } }),
        'w' => {
            var cls: CharClass = .{};
            cls.ascii.addRange('A', 'Z');
            cls.ascii.addRange('a', 'z');
            cls.ascii.addRange('0', '9');
            cls.ascii.add('_');
            try p.emit(.{ .op = .class, .cls = cls });
        },
        'W' => {
            var cls: CharClass = .{};
            cls.ascii.addRange('A', 'Z');
            cls.ascii.addRange('a', 'z');
            cls.ascii.addRange('0', '9');
            cls.ascii.add('_');
            cls.negated = true;
            try p.emit(.{ .op = .class, .cls = cls });
        },
        's' => try p.emit(.{ .op = .class, .cls = .{ .cats = .{ .whitespace = true } } }),
        'S' => try p.emit(.{ .op = .class, .cls = .{ .cats = .{ .whitespace = true }, .negated = true } }),
        'p' => try compilePropEscape(p, false),
        'P' => try compilePropEscape(p, true),
        else => {
            if (c < 0x80) {
                try p.emit(.{ .op = .char, .cp = c });
            } else return error.InvalidEscape;
        },
    }
}

fn compilePropEscape(p: *Parser, negated: bool) !void {
    if (p.eat() != @as(?u8, '{')) return error.InvalidEscape;
    var cats: CatMask = .{};
    while (true) {
        const c = p.peek() orelse return error.InvalidEscape;
        if (c == '}') break;
        _ = p.eat();
        switch (c) {
            'L' => cats.letter = true,
            'N' => cats.number = true,
            'M' => cats.mark = true,
            'Z' => {
                cats.separator = true;
                cats.whitespace = true;
            },
            'P' => cats.punctuation = true,
            'S' => cats.symbol = true,
            // Subcategory letters (e.g. \p{Lu}, \p{Nd}) — treat as parent.
            'u', 'l', 'd', 'o', 'm', 'c', 'e', 't' => {},
            else => return error.InvalidEscape,
        }
    }
    _ = p.eat(); // '}'
    try p.emit(.{ .op = .class, .cls = .{ .cats = cats, .negated = negated } });
}

fn compileClass(p: *Parser) !void {
    if (p.eat() != @as(?u8, '[')) return error.InvalidCharClass;
    var cls: CharClass = .{};
    if (p.peek() == @as(?u8, '^')) {
        _ = p.eat();
        cls.negated = true;
    }
    var ranges: std.ArrayList(Range) = .empty;
    defer ranges.deinit(p.allocator);

    while (true) {
        const c = p.peek() orelse return error.InvalidCharClass;
        if (c == ']') {
            _ = p.eat();
            break;
        }
        const lo_cp = try classAtom(p, &cls);
        if (p.peek() == @as(?u8, '-') and p.peekAt(1) != @as(?u8, ']')) {
            _ = p.eat();
            const hi_cp = try classAtom(p, &cls);
            if (lo_cp == null or hi_cp == null) return error.InvalidCharClass;
            const a = lo_cp.?;
            const b = hi_cp.?;
            if (b < a) return error.InvalidCharClass;
            if (a < 0x80 and b < 0x80) {
                var k: u21 = a;
                while (k <= b) : (k += 1) cls.ascii.add(@intCast(k));
            } else if (b < 0x80) {
                var k: u21 = a;
                while (k <= b) : (k += 1) cls.ascii.add(@intCast(k));
            } else {
                // Split ASCII tail into ascii set if needed.
                if (a < 0x80) {
                    var k: u21 = a;
                    while (k < 0x80) : (k += 1) cls.ascii.add(@intCast(k));
                    try ranges.append(p.allocator, .{ .lo = 0x80, .hi = b });
                } else {
                    try ranges.append(p.allocator, .{ .lo = a, .hi = b });
                }
            }
        } else if (lo_cp) |cp| {
            if (cp < 0x80) {
                cls.ascii.add(@intCast(cp));
            } else {
                try ranges.append(p.allocator, .{ .lo = cp, .hi = cp });
            }
        }
    }

    if (ranges.items.len > 0) {
        const slice = try ranges.toOwnedSlice(p.allocator);
        cls.ranges = slice;
    }
    try p.emit(.{ .op = .class, .cls = cls });
}

fn classAtom(p: *Parser, cls: *CharClass) !?u21 {
    const c = p.peek() orelse return error.InvalidCharClass;
    if (c == '\\') {
        _ = p.eat();
        const e = p.eat() orelse return error.InvalidEscape;
        switch (e) {
            'n' => return '\n',
            'r' => return '\r',
            't' => return '\t',
            '0' => return 0,
            '\\' => return '\\',
            'd' => {
                cls.cats.number = true;
                return null;
            },
            'D' => {
                // Approximation; HF doesn't use \D inside a class.
                return null;
            },
            'w' => {
                cls.ascii.addRange('A', 'Z');
                cls.ascii.addRange('a', 'z');
                cls.ascii.addRange('0', '9');
                cls.ascii.add('_');
                return null;
            },
            'W' => return null,
            's' => {
                cls.cats.whitespace = true;
                return null;
            },
            'S' => return null,
            'p' => {
                if (p.eat() != @as(?u8, '{')) return error.InvalidEscape;
                while (true) {
                    const x = p.peek() orelse return error.InvalidEscape;
                    if (x == '}') break;
                    _ = p.eat();
                    switch (x) {
                        'L' => cls.cats.letter = true,
                        'N' => cls.cats.number = true,
                        'M' => cls.cats.mark = true,
                        'Z' => {
                            cls.cats.separator = true;
                            cls.cats.whitespace = true;
                        },
                        'P' => cls.cats.punctuation = true,
                        'S' => cls.cats.symbol = true,
                        'u', 'l', 'd', 'o', 'm', 'c', 'e', 't' => {},
                        else => return error.InvalidEscape,
                    }
                }
                _ = p.eat();
                return null;
            },
            '.', '+', '*', '?', '(', ')', '[', ']', '{', '}', '|', '^', '$', '/', '-', '\'', '"' => return e,
            else => {
                if (e < 0x80) return e;
                return error.InvalidEscape;
            },
        }
    }
    if (c < 0x80) {
        _ = p.eat();
        return c;
    }
    const seq_len = std.unicode.utf8ByteSequenceLength(c) catch return error.InvalidCharClass;
    if (p.pos + seq_len > p.src.len) return error.InvalidCharClass;
    const cp = std.unicode.utf8Decode(p.src[p.pos .. p.pos + seq_len]) catch return error.InvalidCharClass;
    p.pos += seq_len;
    return cp;
}

// ---------------------------------------------------------------------
// Interpreter

fn classMatches(cls: CharClass, cp: u21) bool {
    var hit = false;
    if (cp < 0x80) {
        if (cls.ascii.contains(@intCast(cp))) hit = true;
    }
    if (!hit) {
        for (cls.ranges) |r| {
            if (cp >= r.lo and cp <= r.hi) {
                hit = true;
                break;
            }
        }
    }
    if (!hit and cls.cats.any()) {
        const c = unicode_props.classifyCp(cp);
        if (cls.cats.letter and c.letter) hit = true;
        if (!hit and cls.cats.number and c.number) hit = true;
        if (!hit and cls.cats.mark and c.mark) hit = true;
        if (!hit and cls.cats.whitespace and c.whitespace) hit = true;
        if (!hit and cls.cats.separator and c.whitespace) hit = true;
        if (!hit and cls.cats.punctuation) {
            // Approximation: ASCII punct already covered by ascii set;
            // BMP "General Punctuation" / CJK / Fullwidth blocks for
            // non-ASCII inputs. Tightens up the common cases without a
            // full Pd/Po/Ps/Pe/Pi/Pf table.
            if (!c.letter and !c.number and !c.mark and !c.whitespace) {
                if (cp < 0x80) {
                    // ASCII punctuation: chars that aren't alnum/space/ctrl.
                    if ((cp >= '!' and cp <= '/') or (cp >= ':' and cp <= '@') or
                        (cp >= '[' and cp <= '`') or (cp >= '{' and cp <= '~'))
                    {
                        hit = true;
                    }
                } else if ((cp >= 0x2000 and cp <= 0x206F) or
                    (cp >= 0x3000 and cp <= 0x303F) or
                    (cp >= 0xFF00 and cp <= 0xFFEF))
                {
                    hit = true;
                }
            }
        }
        if (!hit and cls.cats.symbol) {
            // No general S* support beyond ASCII.
        }
    }
    return if (cls.negated) !hit else hit;
}

const Decoded = struct { cp: u21, len: usize };

fn decodeAt(s: []const u8, i: usize) ?Decoded {
    if (i >= s.len) return null;
    const b0 = s[i];
    if (b0 < 0x80) return .{ .cp = b0, .len = 1 };
    const seq_len = std.unicode.utf8ByteSequenceLength(b0) catch return .{ .cp = b0, .len = 1 };
    if (i + seq_len > s.len) return .{ .cp = b0, .len = 1 };
    const cp = std.unicode.utf8Decode(s[i .. i + seq_len]) catch return .{ .cp = b0, .len = 1 };
    return .{ .cp = cp, .len = seq_len };
}

fn isAsciiLetter(b: u8) bool {
    return (b >= 'A' and b <= 'Z') or (b >= 'a' and b <= 'z');
}
fn toLowerAscii(b: u8) u8 {
    if (b >= 'A' and b <= 'Z') return b + 32;
    return b;
}

const MAX_DEPTH: u32 = 4096;

fn match(insts: []const Inst, input: []const u8, pos: usize) ?usize {
    return matchInner(insts, input, pos, 0, MAX_DEPTH, false);
}

// ---------------------------------------------------------------------
// Thompson NFA prefix walker (drives constrained.zig's prefix automaton).
//
// We maintain a set of live program counters ("threads"). At each input
// codepoint we epsilon-close the current set (following jmp/split/anchor/
// assert), record whether any thread sits on `.match`, then advance every
// consuming thread (char/class/any) over the codepoint into the next set.
// A pattern is `dead` once the set empties before input is exhausted.

const MAX_PROG: usize = 4096;

const ThreadSet = struct {
    /// Bitset over instruction indices (cap MAX_PROG); insts are small.
    seen: [MAX_PROG / 64]u64 = [_]u64{0} ** (MAX_PROG / 64),
    /// Consuming threads, to advance on the next codepoint.
    list: [MAX_PROG]u32 = undefined,
    len: usize = 0,
    /// Every pc marked during the epsilon closure (consuming + zero-width),
    /// so `clear` can reset only the bits it touched.
    marked: [MAX_PROG]u32 = undefined,
    marked_len: usize = 0,
    saw_match: bool = false,

    fn clear(self: *ThreadSet) void {
        for (self.marked[0..self.marked_len]) |pc| self.seen[pc / 64] = 0;
        self.len = 0;
        self.marked_len = 0;
        self.saw_match = false;
    }
    fn has(self: *const ThreadSet, pc: u32) bool {
        return (self.seen[pc / 64] & (@as(u64, 1) << @intCast(pc % 64))) != 0;
    }
    fn mark(self: *ThreadSet, pc: u32) void {
        self.seen[pc / 64] |= @as(u64, 1) << @intCast(pc % 64);
        if (self.marked_len < self.marked.len) {
            self.marked[self.marked_len] = pc;
            self.marked_len += 1;
        }
    }
};

/// Epsilon-close `pc` into `set`, executing zero-width ops (jmp, split,
/// anchor, assert) against `input` at byte `pos`. Consuming ops land in
/// the set as live threads. `.match` sets `saw_match`.
fn addThread(
    set: *ThreadSet,
    insts: []const Inst,
    pc: u32,
    input: []const u8,
    pos: usize,
) void {
    if (pc >= insts.len) return;
    if (set.has(pc)) return;
    set.mark(pc);
    const inst = insts[pc];
    switch (inst.op) {
        .match, .assert_end => set.saw_match = true,
        .jmp => addThread(set, insts, inst.target, input, pos),
        .split => {
            addThread(set, insts, inst.target, input, pos);
            addThread(set, insts, inst.target_b, input, pos);
        },
        .anchor_start => {
            if (pos == 0) addThread(set, insts, pc + 1, input, pos);
        },
        .anchor_end => {
            // Only satisfiable if we're at the end of the KNOWN input.
            // For prefix acceptance that means: succeeds at input.len.
            if (pos == input.len) addThread(set, insts, pc + 1, input, pos);
        },
        .assert => {
            // Evaluate the lookahead body against known bytes via the
            // backtracking matcher. If the body needs bytes we don't have
            // yet, treat positive lookahead as still-live (optimistic) so
            // we don't prematurely kill a viable prefix.
            const body_ok = matchInner(insts, input, pos, inst.target, MAX_DEPTH, true) != null;
            const want = inst.cp == 0; // 0 = positive, 1 = negative
            const at_frontier = pos == input.len;
            const live = if (at_frontier and want) true else (body_ok == want);
            if (live) addThread(set, insts, inst.target_b, input, pos);
        },
        // Consuming ops: leave as a live thread to be advanced by the
        // next codepoint.
        .char, .char_ci, .any_char, .class => {
            if (set.len < set.list.len) {
                set.list[set.len] = pc;
                set.len += 1;
            }
        },
    }
}

fn prefixWalk(insts: []const Inst, input: []const u8) PrefixStatus {
    if (insts.len > MAX_PROG or insts.len == 0) {
        // Programs this large don't occur for HF/constraint patterns; fall
        // back to a conservative answer.
        return .partial;
    }
    var cur: ThreadSet = .{};
    var next: ThreadSet = .{};

    addThread(&cur, insts, 0, input, 0);

    var pos: usize = 0;
    while (true) {
        const at_end = pos >= input.len;
        if (at_end) {
            // Re-close at the true frontier so anchors/asserts at input.len
            // and `.match` are observed with pos == input.len.
            return if (cur.saw_match) .full else if (cur.len > 0) .partial else .dead;
        }
        if (cur.len == 0) return .dead;

        const d = decodeAt(input, pos) orelse return .dead;
        next.clear();
        for (cur.list[0..cur.len]) |pc| {
            const inst = insts[pc];
            const ok = switch (inst.op) {
                .char => d.cp == inst.cp,
                .char_ci => blk: {
                    if (d.cp >= 0x80) break :blk d.cp == inst.cp;
                    const lo: u21 = if (d.cp >= 'A' and d.cp <= 'Z') d.cp + 32 else d.cp;
                    break :blk lo == inst.cp;
                },
                .any_char => true,
                .class => classMatches(inst.cls, d.cp),
                else => false,
            };
            if (ok) addThread(&next, insts, pc + 1, input, pos + d.len);
        }
        const tmp = cur;
        cur = next;
        next = tmp;
        pos += d.len;
    }
}

/// Recursive backtracking interpreter. `assertion_mode` = true while
/// evaluating a lookahead body — `assert_end` becomes a terminator that
/// returns the body's end position (without affecting outer cursor).
fn matchInner(
    insts: []const Inst,
    input: []const u8,
    pos_in: usize,
    pc_in: u32,
    depth: u32,
    assertion_mode: bool,
) ?usize {
    if (depth == 0) return null;
    var pos = pos_in;
    var pc = pc_in;
    while (pc < insts.len) {
        const inst = insts[pc];
        switch (inst.op) {
            .match => {
                if (assertion_mode) {
                    // Shouldn't hit `match` inside an assertion body — the
                    // body has its own `assert_end`. Defensive: succeed.
                    return pos;
                }
                return pos;
            },
            .assert_end => {
                // Terminator for an assertion body: succeed.
                return pos;
            },
            .char => {
                const d = decodeAt(input, pos) orelse return null;
                if (d.cp != inst.cp) return null;
                pos += d.len;
                pc += 1;
            },
            .char_ci => {
                const d = decodeAt(input, pos) orelse return null;
                if (d.cp >= 0x80) {
                    if (d.cp != inst.cp) return null;
                } else {
                    const lo: u21 = if (d.cp >= 'A' and d.cp <= 'Z') d.cp + 32 else d.cp;
                    if (lo != inst.cp) return null;
                }
                pos += d.len;
                pc += 1;
            },
            .any_char => {
                const d = decodeAt(input, pos) orelse return null;
                pos += d.len;
                pc += 1;
            },
            .class => {
                const d = decodeAt(input, pos) orelse return null;
                if (!classMatches(inst.cls, d.cp)) return null;
                pos += d.len;
                pc += 1;
            },
            .anchor_start => {
                if (pos != 0) return null;
                pc += 1;
            },
            .anchor_end => {
                if (pos != input.len) return null;
                pc += 1;
            },
            .jmp => {
                pc = inst.target;
            },
            .split => {
                if (matchInner(insts, input, pos, inst.target, depth - 1, assertion_mode)) |end| {
                    return end;
                }
                pc = inst.target_b;
            },
            .assert => {
                // Evaluate the body in assertion mode (no cursor advance).
                const body_ok = matchInner(insts, input, pos, inst.target, depth - 1, true) != null;
                const want_match = inst.cp == 0; // 0 = positive
                if (body_ok != want_match) return null;
                pc = inst.target_b;
            },
        }
    }
    return pos;
}

// ---------------------------------------------------------------------
// Split helpers used by the HF pretokenizer chain

pub const SplitBehavior = enum { Removed, Isolated, MergedWithPrevious, MergedWithNext, Contiguous };

pub const Segment = struct { start: usize, end: usize, is_match: bool };

/// Split `input` by `re`, emitting segments per `behavior`.
pub fn splitWith(
    allocator: std.mem.Allocator,
    re: *const Regex,
    input: []const u8,
    behavior: SplitBehavior,
) ![]Segment {
    var out: std.ArrayList(Segment) = .empty;
    errdefer out.deinit(allocator);

    var cursor: usize = 0;
    while (cursor <= input.len) {
        const m = re.findFirst(input, cursor) orelse break;
        if (m.end == m.start) {
            // Zero-width match: advance to avoid infinite loop.
            if (cursor >= input.len) break;
            const d = decodeAt(input, cursor) orelse break;
            cursor += d.len;
            continue;
        }
        const gap_start = cursor;
        const gap_end = m.start;
        switch (behavior) {
            .Removed => {
                if (gap_end > gap_start) try out.append(allocator, .{ .start = gap_start, .end = gap_end, .is_match = false });
            },
            .Isolated => {
                if (gap_end > gap_start) try out.append(allocator, .{ .start = gap_start, .end = gap_end, .is_match = false });
                try out.append(allocator, .{ .start = m.start, .end = m.end, .is_match = true });
            },
            .MergedWithPrevious => {
                // Append match onto preceding non-match (or, if there is
                // no preceding, emit as a standalone non-match segment).
                if (out.items.len > 0 and !out.items[out.items.len - 1].is_match and out.items[out.items.len - 1].end == gap_start and gap_end == gap_start) {
                    out.items[out.items.len - 1].end = m.end;
                } else if (gap_end > gap_start) {
                    try out.append(allocator, .{ .start = gap_start, .end = m.end, .is_match = false });
                } else {
                    try out.append(allocator, .{ .start = m.start, .end = m.end, .is_match = false });
                }
            },
            .MergedWithNext => {
                if (gap_end > gap_start) try out.append(allocator, .{ .start = gap_start, .end = gap_end, .is_match = false });
                try out.append(allocator, .{ .start = m.start, .end = m.end, .is_match = true });
            },
            .Contiguous => {
                if (gap_end > gap_start) try out.append(allocator, .{ .start = gap_start, .end = gap_end, .is_match = false });
                if (out.items.len > 0 and out.items[out.items.len - 1].is_match) {
                    out.items[out.items.len - 1].end = m.end;
                } else {
                    try out.append(allocator, .{ .start = m.start, .end = m.end, .is_match = true });
                }
            },
        }
        cursor = m.end;
    }
    if (cursor < input.len) {
        try out.append(allocator, .{ .start = cursor, .end = input.len, .is_match = false });
    }

    if (behavior == .MergedWithNext) {
        // Pull each match into the FOLLOWING non-match.
        var merged: std.ArrayList(Segment) = .empty;
        errdefer merged.deinit(allocator);
        var i: usize = 0;
        while (i < out.items.len) {
            const s = out.items[i];
            if (s.is_match and i + 1 < out.items.len and !out.items[i + 1].is_match) {
                try merged.append(allocator, .{ .start = s.start, .end = out.items[i + 1].end, .is_match = false });
                i += 2;
            } else {
                try merged.append(allocator, .{ .start = s.start, .end = s.end, .is_match = false });
                i += 1;
            }
        }
        out.deinit(allocator);
        return try merged.toOwnedSlice(allocator);
    }

    return try out.toOwnedSlice(allocator);
}

// ---------------------------------------------------------------------
// Tests

const testing = std.testing;

test "compile and match contractions alternation" {
    var re = try compile(testing.allocator, "'s|'t|'re|'ve|'m|'ll|'d");
    defer re.deinit();
    const text = "it's it'll go";
    const m1 = re.findFirst(text, 0).?;
    try testing.expectEqualStrings("'s", text[m1.start..m1.end]);
    const m2 = re.findFirst(text, m1.end).?;
    try testing.expectEqualStrings("'ll", text[m2.start..m2.end]);
}

test "case-insensitive inline-flag group" {
    var re = try compile(testing.allocator, "(?i:hello)");
    defer re.deinit();
    try testing.expect(re.findFirst("say hello", 0) != null);
    try testing.expect(re.findFirst("HELLO world", 0) != null);
    try testing.expect(re.findFirst("Hello", 0) != null);
    try testing.expect(re.findFirst("nope", 0) == null);
}

test "fixed-count digit quantifier" {
    var re = try compile(testing.allocator, "\\d{3}");
    defer re.deinit();
    const m = re.findFirst("ab12345", 0).?;
    try testing.expectEqualStrings("123", "ab12345"[m.start..m.end]);
    try testing.expect(re.findFirst("12", 0) == null);
}

test "p N bounded quantifier matches up to 3 digits greedily" {
    var re = try compile(testing.allocator, "\\p{N}{1,3}");
    defer re.deinit();
    const m = re.findFirst("12345", 0).?;
    try testing.expectEqualStrings("123", "12345"[m.start..m.end]);
    const m2 = re.findFirst("x1y", 0).?;
    try testing.expectEqualStrings("1", "x1y"[m2.start..m2.end]);
    try testing.expect(re.findFirst("abc", 0) == null);
}

test "splitWith Isolated on digit runs" {
    var re = try compile(testing.allocator, "\\d+");
    defer re.deinit();
    const segs = try splitWith(testing.allocator, &re, "abc123def", .Isolated);
    defer testing.allocator.free(segs);
    try testing.expectEqual(@as(usize, 3), segs.len);
    try testing.expectEqualStrings("abc", "abc123def"[segs[0].start..segs[0].end]);
    try testing.expectEqualStrings("123", "abc123def"[segs[1].start..segs[1].end]);
    try testing.expectEqualStrings("def", "abc123def"[segs[2].start..segs[2].end]);
}

test "splitWith Removed drops matches" {
    var re = try compile(testing.allocator, "\\s+");
    defer re.deinit();
    const segs = try splitWith(testing.allocator, &re, "  hello  world  ", .Removed);
    defer testing.allocator.free(segs);
    try testing.expectEqual(@as(usize, 2), segs.len);
    try testing.expectEqualStrings("hello", "  hello  world  "[segs[0].start..segs[0].end]);
    try testing.expectEqualStrings("world", "  hello  world  "[segs[1].start..segs[1].end]);
}

test "negative lookahead implements GPT-2 trailing-ws pattern" {
    var re = try compile(testing.allocator, "\\s+(?!\\S)");
    defer re.deinit();
    // EOF satisfies (?!\S): no next char, so the next is NOT \S.
    const m = re.findFirst("hi   ", 0).?;
    try testing.expectEqualStrings("   ", "hi   "[m.start..m.end]);
}

test "alternation picks leftmost branch" {
    var re = try compile(testing.allocator, "foo|foobar");
    defer re.deinit();
    const m = re.findFirst("foobar", 0).?;
    // Leftmost-first NFA: should match "foo".
    try testing.expectEqualStrings("foo", "foobar"[m.start..m.end]);
}

test "char class with range and unicode property" {
    var re = try compile(testing.allocator, "[\\p{L}\\p{N}]+");
    defer re.deinit();
    const m = re.findFirst("héllo123!", 0).?;
    // Match runs until the '!' (not letter/digit).
    try testing.expectEqualStrings("héllo123", "héllo123!"[m.start..m.end]);
}

test "Llama-3 pretok regex compiles and clamps digit groups to 1-3" {
    // The verbatim Llama-3 Split pattern from tokenizer.json. Must compile
    // and must split a digit run of 5 into "123" + "45" (not one greedy
    // "12345"). Earlier the `\p{N}+` rewriting bug produced one span.
    const pat = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    var re = try compile(testing.allocator, pat);
    defer re.deinit();

    // Five digits: first match must be "123" (the {1,3} branch caps at 3).
    const m1 = re.findFirst("12345", 0).?;
    try testing.expectEqualStrings("123", "12345"[m1.start..m1.end]);
    // Next match starting after position 3 picks up "45".
    const m2 = re.findFirst("12345", m1.end).?;
    try testing.expectEqualStrings("45", "12345"[m2.start..m2.end]);

    // Four digits with surrounding text: digits must split 3+1, not 4+0.
    const m3 = re.findFirst("n=1024)", 0).?;
    try testing.expectEqualStrings("n", "n=1024)"[m3.start..m3.end]);
    const m4 = re.findFirst("n=1024)", m3.end).?;
    // After "n", "=" is in the punct branch.
    try testing.expectEqualStrings("=", "n=1024)"[m4.start..m4.end]);
    const m5 = re.findFirst("n=1024)", m4.end).?;
    try testing.expectEqualStrings("102", "n=1024)"[m5.start..m5.end]);
    const m6 = re.findFirst("n=1024)", m5.end).?;
    try testing.expectEqualStrings("4", "n=1024)"[m6.start..m6.end]);
}
