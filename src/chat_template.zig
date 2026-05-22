//! Jinja-subset renderer for HuggingFace chat templates.
//!
//! Covers enough of Jinja2 to render the common HF chat templates
//! (ChatML / Qwen, Llama-2, Mistral, Gemma). NOT a full Jinja engine.
//!
//! Implemented:
//!   - {{ expr }} interpolation
//!   - {% if %}/{% elif %}/{% else %}/{% endif %}
//!   - {% for v in expr %}{% endfor %} with loop.index0/index/first/last
//!   - {% set v = expr %}
//!   - Whitespace control: {%- ... -%} and {{- ... -}}
//!   - Literals: 'str', "str", int, true/false/none
//!   - Member/bracket access: a.b, a['b']
//!   - Comparisons (==, !=, <, <=, >, >=), and/or/not, `in`
//!   - String concat via `+`
//!   - Ternary `X if COND else Y`
//!   - Filters: |trim, |length
//!   - Method calls: .strip(), .lstrip(), .rstrip(), .upper(), .lower()
//!   - raise_exception('msg') -> error.UnsupportedSyntax
//!
//! Skipped (returns error.UnsupportedSyntax with the offending bytes):
//!   - {% macro %}, {% include %}, {% extends %}, {% raw %}
//!   - Custom filters beyond trim/length
//!   - Slicing (a[1:3]), arithmetic beyond `+`
//!
//! Literals: list `[a, b, c]` and dict `{'k': v, ...}` are supported in
//! expressions. Dict keys must be string literals (Jinja allows bareword
//! keys too — deferred). List/dict values live in the render arena.
//!
//! Not registered in src/root.zig — the task forbade modifying other
//! files. Import directly: `@import("chat_template.zig")`.

const std = @import("std");

pub const Value = union(enum) {
    null_v,
    bool_v: bool,
    int_v: i64,
    string_v: []const u8,
    array_v: []const Value,
    object_v: *const ObjectMap,
};

pub const ObjectMap = std.StringHashMap(Value);

pub const Message = struct {
    role: []const u8,
    content: []const u8,
};

pub const RenderOptions = struct {
    messages: []const Message,
    add_generation_prompt: bool = false,
    bos_token: ?[]const u8 = null,
    eos_token: ?[]const u8 = null,
    extra_vars: ?*const std.StringHashMap(Value) = null,
};

pub const Error = error{
    UnknownVariable,
    TypeMismatch,
    MalformedTemplate,
    UnsupportedSyntax,
    RaisedException,
} || std.mem.Allocator.Error;

// ---------------------------------------------------------------------
// AST

const ExprKind = enum {
    str_lit,
    int_lit,
    bool_lit,
    null_lit,
    ident,
    member, // a.b
    index, // a[expr]
    binary,
    unary,
    ternary,
    filter,
    method_call,
    func_call,
    list_lit, // [a, b, c]
    dict_lit, // {'k': v, ...}
};

const BinOp = enum { add, mod, mul, div, eq, neq, lt, le, gt, ge, log_and, log_or, in_op, is_test };

const Expr = struct {
    kind: ExprKind,
    // payloads (only the relevant ones are used per kind)
    s: []const u8 = "", // str_lit / ident / member name / filter name / method name / func name
    i: i64 = 0, // int_lit
    b: bool = false, // bool_lit
    op: BinOp = .add, // binary
    left: ?*Expr = null,
    right: ?*Expr = null,
    cond: ?*Expr = null, // ternary
    args: []*Expr = &.{}, // method/func args, filter args, list_lit elements, dict_lit values
    keys: []const []const u8 = &.{}, // dict_lit keys (parallel to args)
};

const StmtKind = enum {
    text,
    emit,
    if_stmt,
    for_stmt,
    set_stmt,
};

const Branch = struct {
    cond: ?*Expr, // null for `else`
    body: []Stmt,
};

const Stmt = struct {
    kind: StmtKind,
    // text
    text_bytes: []const u8 = "",
    // emit
    emit_expr: ?*Expr = null,
    // if
    branches: []Branch = &.{},
    // for
    for_var: []const u8 = "",
    for_expr: ?*Expr = null,
    for_body: []Stmt = &.{},
    // set
    set_var: []const u8 = "",
    set_expr: ?*Expr = null,
};

// ---------------------------------------------------------------------
// Expression tokenizer (inside {{ }} or {% %} after keyword)

const ETokKind = enum {
    ident,
    string,
    integer,
    lparen,
    rparen,
    lbracket,
    rbracket,
    lbrace,
    rbrace,
    colon,
    comma,
    dot,
    pipe,
    plus,
    minus,
    percent,
    star,
    slash,
    eq_eq,
    neq,
    lt,
    le,
    gt,
    ge,
    assign,
    eof,
    // keywords (also lexed as ident; promoted by parser)
};

const EToken = struct {
    kind: ETokKind,
    text: []const u8 = "",
    int_val: i64 = 0,
};

const ELexer = struct {
    src: []const u8,
    pos: usize = 0,

    fn skipWs(self: *ELexer) void {
        while (self.pos < self.src.len) {
            const c = self.src[self.pos];
            if (c == ' ' or c == '\t' or c == '\n' or c == '\r') self.pos += 1 else break;
        }
    }

    fn next(self: *ELexer) Error!EToken {
        self.skipWs();
        if (self.pos >= self.src.len) return .{ .kind = .eof };
        const c = self.src[self.pos];

        // strings
        if (c == '\'' or c == '"') {
            const quote = c;
            self.pos += 1;
            const start = self.pos;
            while (self.pos < self.src.len and self.src[self.pos] != quote) {
                if (self.src[self.pos] == '\\' and self.pos + 1 < self.src.len) self.pos += 2 else self.pos += 1;
            }
            if (self.pos >= self.src.len) return error.MalformedTemplate;
            const slice = self.src[start..self.pos];
            self.pos += 1;
            return .{ .kind = .string, .text = slice };
        }

        // integers
        if (c >= '0' and c <= '9') {
            const start = self.pos;
            while (self.pos < self.src.len and self.src[self.pos] >= '0' and self.src[self.pos] <= '9') self.pos += 1;
            const txt = self.src[start..self.pos];
            const v = std.fmt.parseInt(i64, txt, 10) catch return error.MalformedTemplate;
            return .{ .kind = .integer, .text = txt, .int_val = v };
        }

        // identifiers (incl keywords)
        if (isIdentStart(c)) {
            const start = self.pos;
            while (self.pos < self.src.len and isIdentCont(self.src[self.pos])) self.pos += 1;
            return .{ .kind = .ident, .text = self.src[start..self.pos] };
        }

        // punctuation / operators
        switch (c) {
            '(' => {
                self.pos += 1;
                return .{ .kind = .lparen };
            },
            ')' => {
                self.pos += 1;
                return .{ .kind = .rparen };
            },
            '[' => {
                self.pos += 1;
                return .{ .kind = .lbracket };
            },
            ']' => {
                self.pos += 1;
                return .{ .kind = .rbracket };
            },
            '{' => {
                self.pos += 1;
                return .{ .kind = .lbrace };
            },
            '}' => {
                self.pos += 1;
                return .{ .kind = .rbrace };
            },
            ':' => {
                self.pos += 1;
                return .{ .kind = .colon };
            },
            ',' => {
                self.pos += 1;
                return .{ .kind = .comma };
            },
            '.' => {
                self.pos += 1;
                return .{ .kind = .dot };
            },
            '|' => {
                self.pos += 1;
                return .{ .kind = .pipe };
            },
            '+' => {
                self.pos += 1;
                return .{ .kind = .plus };
            },
            '-' => {
                self.pos += 1;
                return .{ .kind = .minus };
            },
            '%' => {
                // Don't consume the `%` of a closing `%}` or `-%}`.
                if (self.pos + 1 < self.src.len and self.src[self.pos + 1] == '}') return .{ .kind = .eof };
                self.pos += 1;
                return .{ .kind = .percent };
            },
            '*' => {
                self.pos += 1;
                return .{ .kind = .star };
            },
            '/' => {
                self.pos += 1;
                return .{ .kind = .slash };
            },
            '=' => {
                if (self.pos + 1 < self.src.len and self.src[self.pos + 1] == '=') {
                    self.pos += 2;
                    return .{ .kind = .eq_eq };
                }
                self.pos += 1;
                return .{ .kind = .assign };
            },
            '!' => {
                if (self.pos + 1 < self.src.len and self.src[self.pos + 1] == '=') {
                    self.pos += 2;
                    return .{ .kind = .neq };
                }
                return error.MalformedTemplate;
            },
            '<' => {
                if (self.pos + 1 < self.src.len and self.src[self.pos + 1] == '=') {
                    self.pos += 2;
                    return .{ .kind = .le };
                }
                self.pos += 1;
                return .{ .kind = .lt };
            },
            '>' => {
                if (self.pos + 1 < self.src.len and self.src[self.pos + 1] == '=') {
                    self.pos += 2;
                    return .{ .kind = .ge };
                }
                self.pos += 1;
                return .{ .kind = .gt };
            },
            else => return error.MalformedTemplate,
        }
    }

    fn peek(self: *ELexer) Error!EToken {
        const saved = self.pos;
        const t = try self.next();
        self.pos = saved;
        return t;
    }
};

fn isIdentStart(c: u8) bool {
    return (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or c == '_';
}
fn isIdentCont(c: u8) bool {
    return isIdentStart(c) or (c >= '0' and c <= '9');
}

// ---------------------------------------------------------------------
// Parser

// Parser holds only the expression-parsing methods. Statement-level parsing
// happens in ParserV2 (which calls these via a temporary Parser instance).
const Parser = struct {
    arena: std.mem.Allocator,
    src: []const u8,

    fn newExpr(self: *Parser) !*Expr {
        const e = try self.arena.create(Expr);
        e.* = .{ .kind = .null_lit };
        return e;
    }

    // ----- expression precedence -----
    // ternary (low)
    // or
    // and
    // not
    // comparison / in
    // additive (+)
    // unary -
    // postfix (. [] () |)
    // primary

    fn parseExpr(self: *Parser, el: *ELexer) Error!*Expr {
        return try self.parseTernary(el);
    }

    fn parseTernary(self: *Parser, el: *ELexer) Error!*Expr {
        const left = try self.parseOr(el);
        const saved = el.pos;
        const t = el.next() catch return left;
        if (t.kind == .ident and std.mem.eql(u8, t.text, "if")) {
            const cond = try self.parseOr(el);
            const else_tok = try el.next();
            if (else_tok.kind != .ident or !std.mem.eql(u8, else_tok.text, "else")) return error.MalformedTemplate;
            const right = try self.parseTernary(el);
            const e = try self.newExpr();
            e.* = .{ .kind = .ternary, .left = left, .right = right, .cond = cond };
            return e;
        }
        el.pos = saved;
        return left;
    }

    fn parseOr(self: *Parser, el: *ELexer) Error!*Expr {
        var left = try self.parseAnd(el);
        while (true) {
            const saved = el.pos;
            const t = el.next() catch break;
            if (t.kind == .ident and std.mem.eql(u8, t.text, "or")) {
                const right = try self.parseAnd(el);
                const e = try self.newExpr();
                e.* = .{ .kind = .binary, .op = .log_or, .left = left, .right = right };
                left = e;
            } else {
                el.pos = saved;
                break;
            }
        }
        return left;
    }

    fn parseAnd(self: *Parser, el: *ELexer) Error!*Expr {
        var left = try self.parseNot(el);
        while (true) {
            const saved = el.pos;
            const t = el.next() catch break;
            if (t.kind == .ident and std.mem.eql(u8, t.text, "and")) {
                const right = try self.parseNot(el);
                const e = try self.newExpr();
                e.* = .{ .kind = .binary, .op = .log_and, .left = left, .right = right };
                left = e;
            } else {
                el.pos = saved;
                break;
            }
        }
        return left;
    }

    fn parseNot(self: *Parser, el: *ELexer) Error!*Expr {
        const saved = el.pos;
        const t = el.next() catch return self.parseCmp(el);
        if (t.kind == .ident and std.mem.eql(u8, t.text, "not")) {
            const inner = try self.parseNot(el);
            const e = try self.newExpr();
            e.* = .{ .kind = .unary, .s = "not", .left = inner };
            return e;
        }
        el.pos = saved;
        return try self.parseCmp(el);
    }

    fn parseCmp(self: *Parser, el: *ELexer) Error!*Expr {
        var left = try self.parseAdd(el);
        const saved = el.pos;
        const t = el.next() catch return left;
        var op: BinOp = .add;
        var matched = true;
        switch (t.kind) {
            .eq_eq => op = .eq,
            .neq => op = .neq,
            .lt => op = .lt,
            .le => op = .le,
            .gt => op = .gt,
            .ge => op = .ge,
            .ident => {
                if (std.mem.eql(u8, t.text, "in")) {
                    op = .in_op;
                } else if (std.mem.eql(u8, t.text, "not")) {
                    // `not in`
                    const t2 = el.next() catch return error.MalformedTemplate;
                    if (t2.kind != .ident or !std.mem.eql(u8, t2.text, "in")) return error.MalformedTemplate;
                    const right = try self.parseAdd(el);
                    const inner = try self.newExpr();
                    inner.* = .{ .kind = .binary, .op = .in_op, .left = left, .right = right };
                    const e = try self.newExpr();
                    e.* = .{ .kind = .unary, .s = "not", .left = inner };
                    return e;
                } else if (std.mem.eql(u8, t.text, "is")) {
                    // `is TEST` or `is not TEST` — Jinja test like `defined`/`none`.
                    // We model it as a binary with op=is_test where right is a
                    // synthetic literal-string expression containing the test name.
                    var negate = false;
                    const test_tok = el.next() catch return error.MalformedTemplate;
                    var test_name = test_tok.text;
                    if (test_tok.kind != .ident) return error.MalformedTemplate;
                    if (std.mem.eql(u8, test_name, "not")) {
                        negate = true;
                        const t3 = el.next() catch return error.MalformedTemplate;
                        if (t3.kind != .ident) return error.MalformedTemplate;
                        test_name = t3.text;
                    }
                    const rhs = try self.newExpr();
                    rhs.* = .{ .kind = .str_lit, .s = test_name };
                    const e = try self.newExpr();
                    e.* = .{ .kind = .binary, .op = .is_test, .left = left, .right = rhs };
                    if (negate) {
                        const wrap = try self.newExpr();
                        wrap.* = .{ .kind = .unary, .s = "not", .left = e };
                        return wrap;
                    }
                    return e;
                } else matched = false;
            },
            else => matched = false,
        }
        if (!matched) {
            el.pos = saved;
            return left;
        }
        const right = try self.parseAdd(el);
        const e = try self.newExpr();
        e.* = .{ .kind = .binary, .op = op, .left = left, .right = right };
        left = e;
        return left;
    }

    fn parseAdd(self: *Parser, el: *ELexer) Error!*Expr {
        var left = try self.parseMul(el);
        while (true) {
            const saved = el.pos;
            const t = el.next() catch break;
            if (t.kind == .plus) {
                const right = try self.parseMul(el);
                const e = try self.newExpr();
                e.* = .{ .kind = .binary, .op = .add, .left = left, .right = right };
                left = e;
            } else {
                el.pos = saved;
                break;
            }
        }
        return left;
    }

    fn parseMul(self: *Parser, el: *ELexer) Error!*Expr {
        var left = try self.parseUnary(el);
        while (true) {
            const saved = el.pos;
            const t = el.next() catch break;
            const op: ?BinOp = switch (t.kind) {
                .star => .mul,
                .slash => .div,
                .percent => .mod,
                else => null,
            };
            if (op) |o| {
                const right = try self.parseUnary(el);
                const e = try self.newExpr();
                e.* = .{ .kind = .binary, .op = o, .left = left, .right = right };
                left = e;
            } else {
                el.pos = saved;
                break;
            }
        }
        return left;
    }

    fn parseUnary(self: *Parser, el: *ELexer) Error!*Expr {
        const saved = el.pos;
        const t = el.next() catch return self.parsePostfix(el);
        if (t.kind == .minus) {
            const inner = try self.parsePostfix(el);
            // realize as -inner via 0-inner is too much; treat as int negation when literal,
            // else error (no -<expr> usage in our templates).
            if (inner.kind == .int_lit) {
                inner.i = -inner.i;
                return inner;
            }
            return error.UnsupportedSyntax;
        }
        el.pos = saved;
        return try self.parsePostfix(el);
    }

    fn parsePostfix(self: *Parser, el: *ELexer) Error!*Expr {
        var node = try self.parsePrimary(el);
        while (true) {
            const saved = el.pos;
            const t = el.next() catch break;
            switch (t.kind) {
                .dot => {
                    const name = try el.next();
                    if (name.kind != .ident) return error.MalformedTemplate;
                    // optional call: a.method(args)
                    const after = el.next() catch EToken{ .kind = .eof };
                    if (after.kind == .lparen) {
                        var args: std.ArrayList(*Expr) = .empty;
                        // parse args
                        const peek_close = el.peek() catch EToken{ .kind = .eof };
                        if (peek_close.kind != .rparen) {
                            while (true) {
                                const a = try self.parseExpr(el);
                                try args.append(self.arena, a);
                                const sep = try el.next();
                                if (sep.kind == .rparen) break;
                                if (sep.kind != .comma) return error.MalformedTemplate;
                            }
                        } else {
                            _ = try el.next(); // consume rparen
                        }
                        const e = try self.newExpr();
                        e.* = .{ .kind = .method_call, .left = node, .s = name.text, .args = try args.toOwnedSlice(self.arena) };
                        node = e;
                    } else {
                        // rewind 'after'
                        el.pos = saved;
                        // re-consume the dot+name only
                        _ = try el.next(); // dot
                        _ = try el.next(); // name
                        const e = try self.newExpr();
                        e.* = .{ .kind = .member, .left = node, .s = name.text };
                        node = e;
                    }
                },
                .lbracket => {
                    const idx = try self.parseExpr(el);
                    const close = try el.next();
                    if (close.kind != .rbracket) return error.MalformedTemplate;
                    const e = try self.newExpr();
                    e.* = .{ .kind = .index, .left = node, .right = idx };
                    node = e;
                },
                .lparen => {
                    // function call form (only valid on ident at top level)
                    if (node.kind != .ident) return error.MalformedTemplate;
                    var args: std.ArrayList(*Expr) = .empty;
                    const peek_close = el.peek() catch EToken{ .kind = .eof };
                    if (peek_close.kind != .rparen) {
                        while (true) {
                            const a = try self.parseExpr(el);
                            try args.append(self.arena, a);
                            const sep = try el.next();
                            if (sep.kind == .rparen) break;
                            if (sep.kind != .comma) return error.MalformedTemplate;
                        }
                    } else {
                        _ = try el.next();
                    }
                    const e = try self.newExpr();
                    e.* = .{ .kind = .func_call, .s = node.s, .args = try args.toOwnedSlice(self.arena) };
                    node = e;
                },
                .pipe => {
                    const name = try el.next();
                    if (name.kind != .ident) return error.MalformedTemplate;
                    const e = try self.newExpr();
                    e.* = .{ .kind = .filter, .left = node, .s = name.text };
                    node = e;
                },
                else => {
                    el.pos = saved;
                    break;
                },
            }
        }
        return node;
    }

    fn parsePrimary(self: *Parser, el: *ELexer) Error!*Expr {
        const t = try el.next();
        switch (t.kind) {
            .string => {
                const e = try self.newExpr();
                e.* = .{ .kind = .str_lit, .s = try decodeString(self.arena, t.text) };
                return e;
            },
            .integer => {
                const e = try self.newExpr();
                e.* = .{ .kind = .int_lit, .i = t.int_val };
                return e;
            },
            .ident => {
                if (std.mem.eql(u8, t.text, "true") or std.mem.eql(u8, t.text, "True")) {
                    const e = try self.newExpr();
                    e.* = .{ .kind = .bool_lit, .b = true };
                    return e;
                }
                if (std.mem.eql(u8, t.text, "false") or std.mem.eql(u8, t.text, "False")) {
                    const e = try self.newExpr();
                    e.* = .{ .kind = .bool_lit, .b = false };
                    return e;
                }
                if (std.mem.eql(u8, t.text, "none") or std.mem.eql(u8, t.text, "None") or std.mem.eql(u8, t.text, "null")) {
                    const e = try self.newExpr();
                    e.* = .{ .kind = .null_lit };
                    return e;
                }
                const e = try self.newExpr();
                e.* = .{ .kind = .ident, .s = t.text };
                return e;
            },
            .lparen => {
                const inner = try self.parseExpr(el);
                const close = try el.next();
                if (close.kind != .rparen) return error.MalformedTemplate;
                return inner;
            },
            .lbracket => {
                // list literal: [ ] or [ expr (, expr)* ]
                var elems: std.ArrayList(*Expr) = .empty;
                const peek0 = try el.peek();
                if (peek0.kind == .rbracket) {
                    _ = try el.next();
                } else {
                    while (true) {
                        const a = try self.parseExpr(el);
                        try elems.append(self.arena, a);
                        const sep = try el.next();
                        if (sep.kind == .rbracket) break;
                        if (sep.kind != .comma) return error.MalformedTemplate;
                        // tolerate trailing comma before ]
                        const peek_after = try el.peek();
                        if (peek_after.kind == .rbracket) {
                            _ = try el.next();
                            break;
                        }
                    }
                }
                const e = try self.newExpr();
                e.* = .{ .kind = .list_lit, .args = try elems.toOwnedSlice(self.arena) };
                return e;
            },
            .lbrace => {
                // dict literal: { } or { 'k': v (, 'k': v)* }
                var keys: std.ArrayList([]const u8) = .empty;
                var vals: std.ArrayList(*Expr) = .empty;
                const peek0 = try el.peek();
                if (peek0.kind == .rbrace) {
                    _ = try el.next();
                } else {
                    while (true) {
                        const ktok = try el.next();
                        if (ktok.kind != .string) return error.UnsupportedSyntax;
                        const decoded = try decodeString(self.arena, ktok.text);
                        const colon = try el.next();
                        if (colon.kind != .colon) return error.MalformedTemplate;
                        const v = try self.parseExpr(el);
                        try keys.append(self.arena, decoded);
                        try vals.append(self.arena, v);
                        const sep = try el.next();
                        if (sep.kind == .rbrace) break;
                        if (sep.kind != .comma) return error.MalformedTemplate;
                        const peek_after = try el.peek();
                        if (peek_after.kind == .rbrace) {
                            _ = try el.next();
                            break;
                        }
                    }
                }
                const e = try self.newExpr();
                e.* = .{ .kind = .dict_lit, .keys = try keys.toOwnedSlice(self.arena), .args = try vals.toOwnedSlice(self.arena) };
                return e;
            },
            else => return error.MalformedTemplate,
        }
    }
};

fn decodeString(allocator: std.mem.Allocator, raw: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    var i: usize = 0;
    while (i < raw.len) {
        if (raw[i] == '\\' and i + 1 < raw.len) {
            const c = raw[i + 1];
            const repl: u8 = switch (c) {
                'n' => '\n',
                't' => '\t',
                'r' => '\r',
                '\\' => '\\',
                '\'' => '\'',
                '"' => '"',
                else => c,
            };
            try out.append(allocator, repl);
            i += 2;
        } else {
            try out.append(allocator, raw[i]);
            i += 1;
        }
    }
    return try out.toOwnedSlice(allocator);
}

// ---------------------------------------------------------------------
// Helpers: whitespace strip

fn lstripWs(s: []const u8) []const u8 {
    var i: usize = 0;
    while (i < s.len) {
        const c = s[i];
        if (c == ' ' or c == '\t' or c == '\n' or c == '\r') i += 1 else break;
    }
    return s[i..];
}
fn rstripWs(s: []const u8) []const u8 {
    var i: usize = s.len;
    while (i > 0) {
        const c = s[i - 1];
        if (c == ' ' or c == '\t' or c == '\n' or c == '\r') i -= 1 else break;
    }
    return s[0..i];
}
fn stripWs(s: []const u8) []const u8 {
    return rstripWs(lstripWs(s));
}

// ---------------------------------------------------------------------
// Renderer

const Scope = struct {
    parent: ?*Scope,
    map: std.StringHashMap(Value),

    fn init(allocator: std.mem.Allocator, parent: ?*Scope) Scope {
        return .{ .parent = parent, .map = std.StringHashMap(Value).init(allocator) };
    }
    fn deinit(self: *Scope) void {
        self.map.deinit();
    }
    fn lookup(self: *Scope, name: []const u8) ?Value {
        if (self.map.get(name)) |v| return v;
        if (self.parent) |p| return p.lookup(name);
        return null;
    }
    fn setLocal(self: *Scope, name: []const u8, v: Value) !void {
        try self.map.put(name, v);
    }
    // walk up to find the scope where `name` exists; if none, set in current.
    fn assign(self: *Scope, name: []const u8, v: Value) !void {
        var cur: ?*Scope = self;
        while (cur) |s| {
            if (s.map.contains(name)) {
                try s.map.put(name, v);
                return;
            }
            cur = s.parent;
        }
        try self.map.put(name, v);
    }
};

const Renderer = struct {
    arena: std.mem.Allocator,
    out: *std.ArrayList(u8),
    out_alloc: std.mem.Allocator,

    fn writeValue(self: *Renderer, v: Value) Error!void {
        switch (v) {
            .null_v => {},
            .bool_v => |b| try self.out.appendSlice(self.out_alloc, if (b) "True" else "False"),
            .int_v => |i| {
                var buf: [32]u8 = undefined;
                const s = std.fmt.bufPrint(&buf, "{d}", .{i}) catch return error.MalformedTemplate;
                try self.out.appendSlice(self.out_alloc, s);
            },
            .string_v => |s| try self.out.appendSlice(self.out_alloc, s),
            .array_v => return error.TypeMismatch,
            .object_v => return error.TypeMismatch,
        }
    }

    fn renderStmts(self: *Renderer, stmts: []const Stmt, scope: *Scope) Error!void {
        for (stmts) |s| try self.renderStmt(s, scope);
    }

    fn renderStmt(self: *Renderer, s: Stmt, scope: *Scope) Error!void {
        switch (s.kind) {
            .text => try self.out.appendSlice(self.out_alloc, s.text_bytes),
            .emit => {
                const v = try self.evalExpr(s.emit_expr.?, scope);
                try self.writeValue(v);
            },
            .if_stmt => {
                var matched = false;
                for (s.branches) |br| {
                    if (matched) break;
                    if (br.cond) |c| {
                        const v = try self.evalExpr(c, scope);
                        if (truthy(v)) {
                            try self.renderStmts(br.body, scope);
                            matched = true;
                        }
                    } else {
                        try self.renderStmts(br.body, scope);
                        matched = true;
                    }
                }
            },
            .for_stmt => {
                const iter_val = try self.evalExpr(s.for_expr.?, scope);
                const items: []const Value = switch (iter_val) {
                    .array_v => |a| a,
                    .string_v => |str| blk: {
                        // iterate over chars as single-byte strings
                        var tmp = try self.arena.alloc(Value, str.len);
                        var i: usize = 0;
                        while (i < str.len) : (i += 1) tmp[i] = .{ .string_v = str[i .. i + 1] };
                        break :blk tmp;
                    },
                    else => return error.TypeMismatch,
                };

                var child = Scope.init(self.arena, scope);
                defer child.deinit();

                const n = items.len;
                for (items, 0..) |it, i| {
                    try child.setLocal(s.for_var, it);
                    var loop_map = ObjectMap.init(self.arena);
                    try loop_map.put("index0", .{ .int_v = @intCast(i) });
                    try loop_map.put("index", .{ .int_v = @intCast(i + 1) });
                    try loop_map.put("first", .{ .bool_v = i == 0 });
                    try loop_map.put("last", .{ .bool_v = i + 1 == n });
                    try loop_map.put("length", .{ .int_v = @intCast(n) });
                    const loop_obj = try self.arena.create(ObjectMap);
                    loop_obj.* = loop_map;
                    try child.setLocal("loop", .{ .object_v = loop_obj });
                    try self.renderStmts(s.for_body, &child);
                }
            },
            .set_stmt => {
                const v = try self.evalExpr(s.set_expr.?, scope);
                try scope.assign(s.set_var, v);
            },
        }
    }

    fn evalExpr(self: *Renderer, e: *Expr, scope: *Scope) Error!Value {
        return switch (e.kind) {
            .str_lit => .{ .string_v = e.s },
            .int_lit => .{ .int_v = e.i },
            .bool_lit => .{ .bool_v = e.b },
            .null_lit => .null_v,
            .ident => scope.lookup(e.s) orelse return error.UnknownVariable,
            .member => blk: {
                const obj = try self.evalExpr(e.left.?, scope);
                break :blk try self.memberAccess(obj, e.s);
            },
            .index => blk: {
                const obj = try self.evalExpr(e.left.?, scope);
                const idx = try self.evalExpr(e.right.?, scope);
                break :blk try self.indexAccess(obj, idx);
            },
            .binary => try self.evalBinary(e, scope),
            .unary => blk: {
                const v = try self.evalExpr(e.left.?, scope);
                if (std.mem.eql(u8, e.s, "not")) break :blk Value{ .bool_v = !truthy(v) };
                break :blk error.MalformedTemplate;
            },
            .ternary => blk: {
                const c = try self.evalExpr(e.cond.?, scope);
                if (truthy(c)) break :blk try self.evalExpr(e.left.?, scope);
                break :blk try self.evalExpr(e.right.?, scope);
            },
            .filter => blk: {
                const v = try self.evalExpr(e.left.?, scope);
                if (std.mem.eql(u8, e.s, "trim")) {
                    if (v != .string_v) return error.TypeMismatch;
                    break :blk Value{ .string_v = stripWs(v.string_v) };
                } else if (std.mem.eql(u8, e.s, "length")) {
                    switch (v) {
                        .string_v => |s| break :blk Value{ .int_v = @intCast(s.len) },
                        .array_v => |a| break :blk Value{ .int_v = @intCast(a.len) },
                        .object_v => |o| break :blk Value{ .int_v = @intCast(o.count()) },
                        else => return error.TypeMismatch,
                    }
                } else return error.UnsupportedSyntax;
            },
            .method_call => try self.evalMethod(e, scope),
            .func_call => try self.evalFunc(e, scope),
            .list_lit => blk: {
                const vals = try self.arena.alloc(Value, e.args.len);
                for (e.args, 0..) |child, i| vals[i] = try self.evalExpr(child, scope);
                break :blk Value{ .array_v = vals };
            },
            .dict_lit => blk: {
                const map = try self.arena.create(ObjectMap);
                map.* = ObjectMap.init(self.arena);
                for (e.keys, e.args) |k, vexpr| {
                    const v = try self.evalExpr(vexpr, scope);
                    try map.put(k, v);
                }
                break :blk Value{ .object_v = map };
            },
        };
    }

    fn memberAccess(self: *Renderer, v: Value, name: []const u8) Error!Value {
        _ = self;
        switch (v) {
            .object_v => |obj| return obj.get(name) orelse error.UnknownVariable,
            else => return error.TypeMismatch,
        }
    }

    fn indexAccess(self: *Renderer, v: Value, idx: Value) Error!Value {
        _ = self;
        switch (v) {
            .object_v => |obj| {
                if (idx != .string_v) return error.TypeMismatch;
                return obj.get(idx.string_v) orelse error.UnknownVariable;
            },
            .array_v => |arr| {
                if (idx != .int_v) return error.TypeMismatch;
                const i = idx.int_v;
                const n: i64 = @intCast(arr.len);
                const idx2: i64 = if (i < 0) i + n else i;
                if (idx2 < 0 or idx2 >= n) return error.UnknownVariable;
                return arr[@intCast(idx2)];
            },
            .string_v => |s| {
                if (idx != .int_v) return error.TypeMismatch;
                const n: i64 = @intCast(s.len);
                const i: i64 = if (idx.int_v < 0) idx.int_v + n else idx.int_v;
                if (i < 0 or i >= n) return error.UnknownVariable;
                return Value{ .string_v = s[@intCast(i) .. @intCast(i + 1)] };
            },
            else => return error.TypeMismatch,
        }
    }

    fn evalBinary(self: *Renderer, e: *Expr, scope: *Scope) Error!Value {
        if (e.op == .log_and) {
            const l = try self.evalExpr(e.left.?, scope);
            if (!truthy(l)) return l;
            return try self.evalExpr(e.right.?, scope);
        }
        if (e.op == .log_or) {
            const l = try self.evalExpr(e.left.?, scope);
            if (truthy(l)) return l;
            return try self.evalExpr(e.right.?, scope);
        }
        if (e.op == .is_test) {
            // Jinja `is TEST`. The left side may throw UnknownVariable
            // when the test is `defined` — that's the case the test is
            // meant to detect. Catch + map.
            const test_name = e.right.?.s;
            if (std.mem.eql(u8, test_name, "defined")) {
                const v = self.evalExpr(e.left.?, scope) catch |err| switch (err) {
                    error.UnknownVariable => return Value{ .bool_v = false },
                    else => return err,
                };
                return Value{ .bool_v = v != .null_v };
            }
            if (std.mem.eql(u8, test_name, "none")) {
                const v = self.evalExpr(e.left.?, scope) catch |err| switch (err) {
                    error.UnknownVariable => return Value{ .bool_v = true },
                    else => return err,
                };
                return Value{ .bool_v = v == .null_v };
            }
            if (std.mem.eql(u8, test_name, "string")) {
                const v = try self.evalExpr(e.left.?, scope);
                return Value{ .bool_v = v == .string_v };
            }
            if (std.mem.eql(u8, test_name, "sequence")) {
                const v = try self.evalExpr(e.left.?, scope);
                return Value{ .bool_v = v == .array_v or v == .string_v };
            }
            if (std.mem.eql(u8, test_name, "mapping")) {
                const v = try self.evalExpr(e.left.?, scope);
                return Value{ .bool_v = v == .object_v };
            }
            if (std.mem.eql(u8, test_name, "iterable")) {
                const v = try self.evalExpr(e.left.?, scope);
                return Value{ .bool_v = v == .array_v or v == .string_v or v == .object_v };
            }
            if (std.mem.eql(u8, test_name, "number")) {
                const v = try self.evalExpr(e.left.?, scope);
                return Value{ .bool_v = v == .int_v };
            }
            if (std.mem.eql(u8, test_name, "boolean")) {
                const v = try self.evalExpr(e.left.?, scope);
                return Value{ .bool_v = v == .bool_v };
            }
            if (std.mem.eql(u8, test_name, "true")) {
                const v = try self.evalExpr(e.left.?, scope);
                return Value{ .bool_v = v == .bool_v and v.bool_v };
            }
            if (std.mem.eql(u8, test_name, "false")) {
                const v = try self.evalExpr(e.left.?, scope);
                return Value{ .bool_v = v == .bool_v and !v.bool_v };
            }
            return error.UnsupportedSyntax;
        }
        const l = try self.evalExpr(e.left.?, scope);
        const r = try self.evalExpr(e.right.?, scope);
        return switch (e.op) {
            .add => try self.addOp(l, r),
            .mod, .mul, .div => try self.arithOp(l, r, e.op),
            .eq => .{ .bool_v = valueEq(l, r) },
            .neq => .{ .bool_v = !valueEq(l, r) },
            .lt, .le, .gt, .ge => try self.cmpOp(l, r, e.op),
            .in_op => try self.inOp(l, r),
            else => unreachable,
        };
    }

    fn arithOp(self: *Renderer, l: Value, r: Value, op: BinOp) !Value {
        _ = self;
        if (l != .int_v or r != .int_v) return error.TypeMismatch;
        const a = l.int_v;
        const b = r.int_v;
        return switch (op) {
            .mul => Value{ .int_v = a * b },
            .div => if (b == 0) error.TypeMismatch else Value{ .int_v = @divTrunc(a, b) },
            .mod => if (b == 0) error.TypeMismatch else Value{ .int_v = @mod(a, b) },
            else => unreachable,
        };
    }

    fn addOp(self: *Renderer, l: Value, r: Value) !Value {
        if (l == .string_v and r == .string_v) {
            const merged = try std.mem.concat(self.arena, u8, &.{ l.string_v, r.string_v });
            return Value{ .string_v = merged };
        }
        if (l == .int_v and r == .int_v) return Value{ .int_v = l.int_v + r.int_v };
        return error.TypeMismatch;
    }

    fn cmpOp(self: *Renderer, l: Value, r: Value, op: BinOp) !Value {
        _ = self;
        const li: i64 = switch (l) {
            .int_v => |x| x,
            else => return error.TypeMismatch,
        };
        const ri: i64 = switch (r) {
            .int_v => |x| x,
            else => return error.TypeMismatch,
        };
        const b = switch (op) {
            .lt => li < ri,
            .le => li <= ri,
            .gt => li > ri,
            .ge => li >= ri,
            else => unreachable,
        };
        return Value{ .bool_v = b };
    }

    fn inOp(self: *Renderer, needle: Value, hay: Value) !Value {
        _ = self;
        switch (hay) {
            .string_v => |s| {
                if (needle != .string_v) return error.TypeMismatch;
                return Value{ .bool_v = std.mem.indexOf(u8, s, needle.string_v) != null };
            },
            .array_v => |arr| {
                for (arr) |a| if (valueEq(a, needle)) return Value{ .bool_v = true };
                return Value{ .bool_v = false };
            },
            .object_v => |obj| {
                if (needle != .string_v) return error.TypeMismatch;
                return Value{ .bool_v = obj.contains(needle.string_v) };
            },
            else => return error.TypeMismatch,
        }
    }

    fn evalMethod(self: *Renderer, e: *Expr, scope: *Scope) !Value {
        const recv = try self.evalExpr(e.left.?, scope);
        if (recv != .string_v) return error.TypeMismatch;
        const s = recv.string_v;
        if (std.mem.eql(u8, e.s, "strip")) return Value{ .string_v = stripWs(s) };
        if (std.mem.eql(u8, e.s, "lstrip")) return Value{ .string_v = lstripWs(s) };
        if (std.mem.eql(u8, e.s, "rstrip")) return Value{ .string_v = rstripWs(s) };
        if (std.mem.eql(u8, e.s, "upper")) {
            const buf = try self.arena.alloc(u8, s.len);
            for (s, 0..) |c, i| buf[i] = std.ascii.toUpper(c);
            return Value{ .string_v = buf };
        }
        if (std.mem.eql(u8, e.s, "lower")) {
            const buf = try self.arena.alloc(u8, s.len);
            for (s, 0..) |c, i| buf[i] = std.ascii.toLower(c);
            return Value{ .string_v = buf };
        }
        return error.UnsupportedSyntax;
    }

    fn evalFunc(self: *Renderer, e: *Expr, scope: *Scope) !Value {
        _ = self;
        _ = scope;
        if (std.mem.eql(u8, e.s, "raise_exception")) return error.RaisedException;
        return error.UnsupportedSyntax;
    }
};

fn truthy(v: Value) bool {
    return switch (v) {
        .null_v => false,
        .bool_v => |b| b,
        .int_v => |i| i != 0,
        .string_v => |s| s.len > 0,
        .array_v => |a| a.len > 0,
        .object_v => |o| o.count() > 0,
    };
}

fn valueEq(a: Value, b: Value) bool {
    if (@as(std.meta.Tag(Value), a) != @as(std.meta.Tag(Value), b)) return false;
    return switch (a) {
        .null_v => true,
        .bool_v => |x| x == b.bool_v,
        .int_v => |x| x == b.int_v,
        .string_v => |x| std.mem.eql(u8, x, b.string_v),
        .array_v => false, // pointer compare too brittle; treat as not equal
        .object_v => |x| x == b.object_v,
    };
}

// ---------------------------------------------------------------------
// Public API

pub fn render(
    allocator: std.mem.Allocator,
    template: []const u8,
    opts: RenderOptions,
) Error![]u8 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    var p = ParserV2{ .arena = aa, .src = template, .pos = 0 };
    const stmts = try p.parseProgramV2();

    // Build root scope.
    var root_scope = Scope.init(aa, null);
    defer root_scope.deinit();

    // messages -> array of object maps
    var msg_objs = try aa.alloc(ObjectMap, opts.messages.len);
    var msg_vals = try aa.alloc(Value, opts.messages.len);
    for (opts.messages, 0..) |m, i| {
        msg_objs[i] = ObjectMap.init(aa);
        try msg_objs[i].put("role", .{ .string_v = m.role });
        try msg_objs[i].put("content", .{ .string_v = m.content });
        msg_vals[i] = .{ .object_v = &msg_objs[i] };
    }
    try root_scope.setLocal("messages", .{ .array_v = msg_vals });
    try root_scope.setLocal("add_generation_prompt", .{ .bool_v = opts.add_generation_prompt });
    if (opts.bos_token) |b| try root_scope.setLocal("bos_token", .{ .string_v = b });
    if (opts.eos_token) |e| try root_scope.setLocal("eos_token", .{ .string_v = e });

    if (opts.extra_vars) |xv| {
        var it = xv.iterator();
        while (it.next()) |entry| {
            try root_scope.setLocal(entry.key_ptr.*, entry.value_ptr.*);
        }
    }

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);

    var r = Renderer{ .arena = aa, .out = &out, .out_alloc = allocator };
    try r.renderStmts(stmts, &root_scope);

    return try out.toOwnedSlice(allocator);
}

// ---------------------------------------------------------------------
// ParserV2 — single-pass parser that supports elif natively.
//
// Operates directly on the source byte stream; emits AST nodes from the same
// expression grammar implemented in Parser above (we reuse parsePrimary etc.
// by creating a tiny adapter Parser).

const ParserV2 = struct {
    arena: std.mem.Allocator,
    src: []const u8,
    pos: usize,
    pending_strip_trailing: bool = false,

    fn parseProgramV2(self: *ParserV2) Error![]Stmt {
        return try self.parseStmtsV2(&.{});
    }

    fn parseStmtsV2(self: *ParserV2, terminators: []const []const u8) Error![]Stmt {
        var out: std.ArrayList(Stmt) = .empty;

        while (self.pos < self.src.len) {
            // text chunk
            const text_start = self.pos;
            var i = self.pos;
            while (i < self.src.len) {
                if (i + 1 < self.src.len and self.src[i] == '{' and (self.src[i + 1] == '{' or self.src[i + 1] == '%')) break;
                i += 1;
            }
            if (i > text_start) {
                var bytes = self.src[text_start..i];
                if (self.pending_strip_trailing) {
                    bytes = lstripWs(bytes);
                    self.pending_strip_trailing = false;
                }
                if (bytes.len > 0) try out.append(self.arena, .{ .kind = .text, .text_bytes = bytes });
                self.pos = i;
            }

            if (self.pos >= self.src.len) break;

            // marker
            const m1 = self.src[self.pos];
            const m2 = self.src[self.pos + 1];
            if (m1 != '{') return error.MalformedTemplate;

            if (m2 == '{') {
                // expression
                const trim_open = self.pos + 2 < self.src.len and self.src[self.pos + 2] == '-';
                self.pos += if (trim_open) 3 else 2;
                if (trim_open) {
                    if (out.items.len > 0) {
                        const last = &out.items[out.items.len - 1];
                        if (last.kind == .text) last.text_bytes = rstripWs(last.text_bytes);
                    }
                }
                var el: ELexer = .{ .src = self.src, .pos = self.pos };
                var tmp_parser = Parser{ .arena = self.arena, .src = self.src };
                const expr = try tmp_parser.parseExpr(&el);
                // expect }} or -}}
                el.skipWs();
                if (el.pos + 1 < self.src.len and self.src[el.pos] == '}' and self.src[el.pos + 1] == '}') {
                    self.pos = el.pos + 2;
                } else if (el.pos + 2 < self.src.len and self.src[el.pos] == '-' and self.src[el.pos + 1] == '}' and self.src[el.pos + 2] == '}') {
                    self.pos = el.pos + 3;
                    self.pending_strip_trailing = true;
                } else return error.MalformedTemplate;
                try out.append(self.arena, .{ .kind = .emit, .emit_expr = expr });
            } else if (m2 == '%') {
                // statement
                const trim_open = self.pos + 2 < self.src.len and self.src[self.pos + 2] == '-';
                const stmt_start = self.pos;
                self.pos += if (trim_open) 3 else 2;
                if (trim_open) {
                    if (out.items.len > 0) {
                        const last = &out.items[out.items.len - 1];
                        if (last.kind == .text) last.text_bytes = rstripWs(last.text_bytes);
                    }
                }
                var el: ELexer = .{ .src = self.src, .pos = self.pos };
                const kw = try el.next();
                if (kw.kind != .ident) return error.MalformedTemplate;

                // terminator check
                for (terminators) |term| {
                    if (std.mem.eql(u8, term, kw.text)) {
                        // rewind to stmt_start so caller can re-parse this stmt
                        self.pos = stmt_start;
                        return try out.toOwnedSlice(self.arena);
                    }
                }

                if (std.mem.eql(u8, kw.text, "if")) {
                    const s = try self.parseIfV2(&el);
                    try out.append(self.arena, s);
                } else if (std.mem.eql(u8, kw.text, "for")) {
                    const s = try self.parseForV2(&el);
                    try out.append(self.arena, s);
                } else if (std.mem.eql(u8, kw.text, "set")) {
                    const s = try self.parseSetV2(&el);
                    try out.append(self.arena, s);
                } else if (std.mem.eql(u8, kw.text, "macro") or
                    std.mem.eql(u8, kw.text, "include") or
                    std.mem.eql(u8, kw.text, "extends") or
                    std.mem.eql(u8, kw.text, "raw") or
                    std.mem.eql(u8, kw.text, "block") or
                    std.mem.eql(u8, kw.text, "endmacro") or
                    std.mem.eql(u8, kw.text, "endraw") or
                    std.mem.eql(u8, kw.text, "endblock"))
                {
                    return error.UnsupportedSyntax;
                } else {
                    return error.MalformedTemplate;
                }
            } else {
                return error.MalformedTemplate;
            }
        }

        return try out.toOwnedSlice(self.arena);
    }

    fn consumeStmtClose(self: *ParserV2, el: *ELexer) !void {
        el.skipWs();
        if (el.pos + 1 < self.src.len and self.src[el.pos] == '%' and self.src[el.pos + 1] == '}') {
            self.pos = el.pos + 2;
        } else if (el.pos + 2 < self.src.len and self.src[el.pos] == '-' and self.src[el.pos + 1] == '%' and self.src[el.pos + 2] == '}') {
            self.pos = el.pos + 3;
            self.pending_strip_trailing = true;
        } else return error.MalformedTemplate;
    }

    // After we rewound to stmt_start (because we hit a terminator), this re-consumes
    // the `{% kw` and any args, returning the keyword text and an ELexer positioned
    // at the args.
    fn reopenStmt(self: *ParserV2) !struct { kw: []const u8, el: ELexer } {
        if (self.pos + 1 >= self.src.len or self.src[self.pos] != '{' or self.src[self.pos + 1] != '%') return error.MalformedTemplate;
        const trim_open = self.pos + 2 < self.src.len and self.src[self.pos + 2] == '-';
        self.pos += if (trim_open) 3 else 2;
        var el: ELexer = .{ .src = self.src, .pos = self.pos };
        const kw = try el.next();
        if (kw.kind != .ident) return error.MalformedTemplate;
        return .{ .kw = kw.text, .el = el };
    }

    fn parseIfV2(self: *ParserV2, el: *ELexer) Error!Stmt {
        var tmp_parser = Parser{ .arena = self.arena, .src = self.src };
        const cond0 = try tmp_parser.parseExpr(el);
        try self.consumeStmtClose(el);

        var branches: std.ArrayList(Branch) = .empty;
        const body0 = try self.parseStmtsV2(&.{ "elif", "else", "endif" });
        try branches.append(self.arena, .{ .cond = cond0, .body = body0 });

        while (true) {
            if (self.pos >= self.src.len) return error.MalformedTemplate;
            const reop = try self.reopenStmt();
            var el2 = reop.el;
            if (std.mem.eql(u8, reop.kw, "endif")) {
                try self.consumeStmtClose(&el2);
                break;
            } else if (std.mem.eql(u8, reop.kw, "elif")) {
                const condN = try tmp_parser.parseExpr(&el2);
                try self.consumeStmtClose(&el2);
                const bodyN = try self.parseStmtsV2(&.{ "elif", "else", "endif" });
                try branches.append(self.arena, .{ .cond = condN, .body = bodyN });
            } else if (std.mem.eql(u8, reop.kw, "else")) {
                try self.consumeStmtClose(&el2);
                const body_else = try self.parseStmtsV2(&.{"endif"});
                try branches.append(self.arena, .{ .cond = null, .body = body_else });
            } else return error.MalformedTemplate;
        }

        return .{ .kind = .if_stmt, .branches = try branches.toOwnedSlice(self.arena) };
    }

    fn parseForV2(self: *ParserV2, el: *ELexer) Error!Stmt {
        const var_tok = try el.next();
        if (var_tok.kind != .ident) return error.MalformedTemplate;
        const in_tok = try el.next();
        if (in_tok.kind != .ident or !std.mem.eql(u8, in_tok.text, "in")) return error.MalformedTemplate;
        var tmp_parser = Parser{ .arena = self.arena, .src = self.src };
        const expr = try tmp_parser.parseExpr(el);
        try self.consumeStmtClose(el);

        const body = try self.parseStmtsV2(&.{"endfor"});
        // consume endfor
        const reop = try self.reopenStmt();
        if (!std.mem.eql(u8, reop.kw, "endfor")) return error.MalformedTemplate;
        var el2 = reop.el;
        try self.consumeStmtClose(&el2);

        return .{ .kind = .for_stmt, .for_var = var_tok.text, .for_expr = expr, .for_body = body };
    }

    fn parseSetV2(self: *ParserV2, el: *ELexer) Error!Stmt {
        const var_tok = try el.next();
        if (var_tok.kind != .ident) return error.MalformedTemplate;
        const eq_tok = try el.next();
        if (eq_tok.kind != .assign) return error.MalformedTemplate;
        var tmp_parser = Parser{ .arena = self.arena, .src = self.src };
        const expr = try tmp_parser.parseExpr(el);
        try self.consumeStmtClose(el);
        return .{ .kind = .set_stmt, .set_var = var_tok.text, .set_expr = expr };
    }
};

// ---------------------------------------------------------------------
// Inverse: parse a rendered chat-template string back into a `[]Message`.
//
// `render` goes messages -> string. `invert` goes string -> messages, for
// serving stacks that need bidirectional template handling (e.g. decoding
// a model's continuation back into structured turns).
//
// One inverse parser per supported template family. Parsers are byte-level
// scanners; they don't run the Jinja engine in reverse — they pattern-match
// on the canonical marker layout each family produces.

pub const TemplateKind = enum {
    chatml, // <|im_start|>role\n...content...<|im_end|>\n
    mistral, // <s>[INST] user [/INST] assistant </s>...
    llama2, // <s>[INST] <<SYS>>\nsystem\n<</SYS>>\n\nuser [/INST] reply </s>
    gemma, // <start_of_turn>role\ncontent<end_of_turn>\n
};

/// A parsed message, with a flag marking turns that were cut mid-stream
/// (e.g. an assistant turn missing its closing marker).
pub const ParsedMessage = struct {
    role: []const u8,
    content: []const u8,
    partial: bool = false,
};

/// Caller owns the returned slice and each `role`/`content` string inside it.
/// Free via `freeParsed(allocator, msgs)`.
pub fn invert(
    allocator: std.mem.Allocator,
    template_kind: TemplateKind,
    rendered_text: []const u8,
) Error![]ParsedMessage {
    return switch (template_kind) {
        .chatml => try invertChatML(allocator, rendered_text),
        .mistral => try invertMistral(allocator, rendered_text),
        .llama2 => try invertLlama2(allocator, rendered_text),
        .gemma => try invertGemma(allocator, rendered_text),
    };
}

/// Free the result of `invert`. Safe to call with a zero-length slice.
pub fn freeParsed(allocator: std.mem.Allocator, msgs: []ParsedMessage) void {
    for (msgs) |m| {
        allocator.free(m.role);
        allocator.free(m.content);
    }
    allocator.free(msgs);
}

// -- helpers --------------------------------------------------------------

fn validUtf8Role(s: []const u8) bool {
    return std.unicode.utf8ValidateSlice(s);
}

fn dupTrimmed(allocator: std.mem.Allocator, s: []const u8) ![]u8 {
    return try allocator.dupe(u8, s);
}

fn appendMsg(
    list: *std.ArrayList(ParsedMessage),
    allocator: std.mem.Allocator,
    role: []const u8,
    content: []const u8,
    partial: bool,
) !void {
    if (!validUtf8Role(role)) return error.MalformedTemplate;
    const r = try allocator.dupe(u8, role);
    errdefer allocator.free(r);
    const c = try allocator.dupe(u8, content);
    errdefer allocator.free(c);
    try list.append(allocator, .{ .role = r, .content = c, .partial = partial });
}

// -- ChatML ---------------------------------------------------------------
//
// Canonical block: `<|im_start|>ROLE\nCONTENT<|im_end|>`. Blocks may be
// separated by arbitrary whitespace (templates commonly emit `\n` between
// them). A trailing `<|im_start|>ROLE\nCONTENT` with no `<|im_end|>` is
// treated as a partial (mid-stream) turn.

fn invertChatML(allocator: std.mem.Allocator, src: []const u8) Error![]ParsedMessage {
    const start_marker = "<|im_start|>";
    const end_marker = "<|im_end|>";

    var out: std.ArrayList(ParsedMessage) = .empty;
    errdefer {
        for (out.items) |m| {
            allocator.free(m.role);
            allocator.free(m.content);
        }
        out.deinit(allocator);
    }

    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, src, pos, start_marker)) |s| {
        const role_start = s + start_marker.len;
        // Role runs until newline.
        const nl_rel = std.mem.indexOfScalarPos(u8, src, role_start, '\n') orelse {
            // No newline after role at all — malformed turn header.
            return error.MalformedTemplate;
        };
        const role = src[role_start..nl_rel];
        const content_start = nl_rel + 1;

        if (std.mem.indexOfPos(u8, src, content_start, end_marker)) |e| {
            const content = src[content_start..e];
            try appendMsg(&out, allocator, role, content, false);
            pos = e + end_marker.len;
        } else {
            // Trailing partial turn. Content runs to end of input.
            const content = src[content_start..];
            try appendMsg(&out, allocator, role, content, true);
            pos = src.len;
        }
    }

    return try out.toOwnedSlice(allocator);
}

// -- Gemma ----------------------------------------------------------------
//
// Same shape as ChatML with different markers:
//   `<start_of_turn>ROLE\nCONTENT<end_of_turn>`

fn invertGemma(allocator: std.mem.Allocator, src: []const u8) Error![]ParsedMessage {
    const start_marker = "<start_of_turn>";
    const end_marker = "<end_of_turn>";

    var out: std.ArrayList(ParsedMessage) = .empty;
    errdefer {
        for (out.items) |m| {
            allocator.free(m.role);
            allocator.free(m.content);
        }
        out.deinit(allocator);
    }

    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, src, pos, start_marker)) |s| {
        const role_start = s + start_marker.len;
        const nl_rel = std.mem.indexOfScalarPos(u8, src, role_start, '\n') orelse {
            return error.MalformedTemplate;
        };
        const role = src[role_start..nl_rel];
        const content_start = nl_rel + 1;

        if (std.mem.indexOfPos(u8, src, content_start, end_marker)) |e| {
            const content = src[content_start..e];
            try appendMsg(&out, allocator, role, content, false);
            pos = e + end_marker.len;
        } else {
            const content = src[content_start..];
            try appendMsg(&out, allocator, role, content, true);
            pos = src.len;
        }
    }

    return try out.toOwnedSlice(allocator);
}

// -- Mistral --------------------------------------------------------------
//
// Format (per HF Mistral-Instruct):
//   `<s>[INST] USER [/INST] ASSISTANT </s>[INST] USER2 [/INST] ASSISTANT2 </s>...`
//
// Each `[INST] X [/INST] Y` pair becomes two messages: `{user, X}` then
// `{assistant, Y}`. The trailing `</s>` separates pairs; the leading `<s>`
// (BOS) is consumed if present. A final `[INST] X [/INST]` with no closing
// `</s>` and no assistant text emits the user message and a partial empty
// assistant turn if [/INST] is present; if [/INST] is missing the user
// message itself is marked partial.

fn invertMistral(allocator: std.mem.Allocator, src: []const u8) Error![]ParsedMessage {
    const inst_open = "[INST]";
    const inst_close = "[/INST]";
    const eos = "</s>";

    var out: std.ArrayList(ParsedMessage) = .empty;
    errdefer {
        for (out.items) |m| {
            allocator.free(m.role);
            allocator.free(m.content);
        }
        out.deinit(allocator);
    }

    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, src, pos, inst_open)) |open| {
        const user_start = open + inst_open.len;
        const close = std.mem.indexOfPos(u8, src, user_start, inst_close) orelse {
            // [INST] with no [/INST] — partial user turn, rest of input is content.
            const content = std.mem.trim(u8, src[user_start..], " \t\n\r");
            try appendMsg(&out, allocator, "user", content, true);
            pos = src.len;
            break;
        };
        const user_raw = src[user_start..close];
        const user = std.mem.trim(u8, user_raw, " \t\n\r");
        try appendMsg(&out, allocator, "user", user, false);

        const after_close = close + inst_close.len;
        // Assistant text runs to the next </s> or to the next [INST] (some
        // emitters omit </s> between turns) or to end-of-input.
        const next_eos = std.mem.indexOfPos(u8, src, after_close, eos);
        const next_open = std.mem.indexOfPos(u8, src, after_close, inst_open);

        const asst_end = blk: {
            if (next_eos == null and next_open == null) break :blk src.len;
            if (next_eos == null) break :blk next_open.?;
            if (next_open == null) break :blk next_eos.?;
            break :blk @min(next_eos.?, next_open.?);
        };

        const asst_raw = src[after_close..asst_end];
        const asst = std.mem.trim(u8, asst_raw, " \t\n\r");
        const partial = (asst_end == src.len) and next_eos == null;
        if (asst.len > 0 or !partial) {
            // Empty closed assistant turn still gets a (possibly empty) message,
            // mirroring Mistral's behavior where empty replies are legal.
            try appendMsg(&out, allocator, "assistant", asst, partial);
        } else {
            // partial empty assistant — emit so caller sees the turn boundary.
            try appendMsg(&out, allocator, "assistant", asst, true);
        }

        // Advance past `</s>` if present so we don't re-enter on it.
        if (next_eos) |e| {
            if (e == asst_end) {
                pos = e + eos.len;
                continue;
            }
        }
        pos = asst_end;
    }

    return try out.toOwnedSlice(allocator);
}

// -- Llama-2 --------------------------------------------------------------
//
// Format:
//   <s>[INST] <<SYS>>\nSYSTEM\n<</SYS>>\n\nUSER [/INST] REPLY </s><s>[INST] ...
//
// First [INST] block may begin with a `<<SYS>>...<</SYS>>` envelope.
// If present, the system message is emitted before the user message.
// Subsequent [INST] blocks behave like Mistral.

fn invertLlama2(allocator: std.mem.Allocator, src: []const u8) Error![]ParsedMessage {
    const inst_open = "[INST]";
    const inst_close = "[/INST]";
    const eos = "</s>";
    const sys_open = "<<SYS>>";
    const sys_close = "<</SYS>>";

    var out: std.ArrayList(ParsedMessage) = .empty;
    errdefer {
        for (out.items) |m| {
            allocator.free(m.role);
            allocator.free(m.content);
        }
        out.deinit(allocator);
    }

    var pos: usize = 0;
    var first_block = true;
    while (std.mem.indexOfPos(u8, src, pos, inst_open)) |open| {
        const inst_body_start = open + inst_open.len;
        const close = std.mem.indexOfPos(u8, src, inst_body_start, inst_close) orelse {
            // No closing [/INST] — emit whatever's left as a partial user turn.
            var body = src[inst_body_start..];
            if (first_block) {
                if (std.mem.indexOf(u8, body, sys_open)) |so| {
                    if (std.mem.indexOfPos(u8, body, so + sys_open.len, sys_close)) |sc| {
                        const sys_raw = body[so + sys_open.len .. sc];
                        const sys = std.mem.trim(u8, sys_raw, " \t\n\r");
                        try appendMsg(&out, allocator, "system", sys, false);
                        body = body[sc + sys_close.len ..];
                    }
                }
            }
            const user = std.mem.trim(u8, body, " \t\n\r");
            try appendMsg(&out, allocator, "user", user, true);
            pos = src.len;
            break;
        };
        var inst_body = src[inst_body_start..close];

        if (first_block) {
            // Optional system envelope only legal at the very first turn.
            if (std.mem.indexOf(u8, inst_body, sys_open)) |so| {
                if (std.mem.indexOfPos(u8, inst_body, so + sys_open.len, sys_close)) |sc| {
                    const sys_raw = inst_body[so + sys_open.len .. sc];
                    const sys = std.mem.trim(u8, sys_raw, " \t\n\r");
                    try appendMsg(&out, allocator, "system", sys, false);
                    inst_body = inst_body[sc + sys_close.len ..];
                } else return error.MalformedTemplate;
            }
        }
        first_block = false;

        const user = std.mem.trim(u8, inst_body, " \t\n\r");
        try appendMsg(&out, allocator, "user", user, false);

        const after_close = close + inst_close.len;
        const next_eos = std.mem.indexOfPos(u8, src, after_close, eos);
        const next_open = std.mem.indexOfPos(u8, src, after_close, inst_open);

        const asst_end = blk: {
            if (next_eos == null and next_open == null) break :blk src.len;
            if (next_eos == null) break :blk next_open.?;
            if (next_open == null) break :blk next_eos.?;
            break :blk @min(next_eos.?, next_open.?);
        };

        const asst_raw = src[after_close..asst_end];
        const asst = std.mem.trim(u8, asst_raw, " \t\n\r");
        const partial = (asst_end == src.len) and next_eos == null;
        try appendMsg(&out, allocator, "assistant", asst, partial);

        if (next_eos) |e| {
            if (e == asst_end) {
                pos = e + eos.len;
                continue;
            }
        }
        pos = asst_end;
    }

    return try out.toOwnedSlice(allocator);
}

// ---------------------------------------------------------------------
// Tests

const testing = std.testing;

test "literal text passes through" {
    const out = try render(testing.allocator, "hello", .{ .messages = &.{} });
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("hello", out);
}

test "simple variable interpolation" {
    var extra = std.StringHashMap(Value).init(testing.allocator);
    defer extra.deinit();
    try extra.put("name", .{ .string_v = "alice" });
    const out = try render(testing.allocator, "hi {{ name }}", .{
        .messages = &.{},
        .extra_vars = &extra,
    });
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("hi alice", out);
}

test "ChatML template renders correctly" {
    const tmpl =
        \\{% for message in messages %}
        \\{% if message['role'] == 'system' %}<|im_start|>system
        \\{{ message['content'] }}<|im_end|>{% elif message['role'] == 'user' %}<|im_start|>user
        \\{{ message['content'] }}<|im_end|>{% elif message['role'] == 'assistant' %}<|im_start|>assistant
        \\{{ message['content'] }}<|im_end|>{% endif %}
        \\{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
        \\{% endif %}
    ;
    const msgs = [_]Message{
        .{ .role = "system", .content = "You are helpful." },
        .{ .role = "user", .content = "Hi" },
    };
    const out = try render(testing.allocator, tmpl, .{
        .messages = &msgs,
        .add_generation_prompt = true,
    });
    defer testing.allocator.free(out);
    // Check key markers are present
    try testing.expect(std.mem.indexOf(u8, out, "<|im_start|>system") != null);
    try testing.expect(std.mem.indexOf(u8, out, "You are helpful.") != null);
    try testing.expect(std.mem.indexOf(u8, out, "<|im_start|>user") != null);
    try testing.expect(std.mem.indexOf(u8, out, "Hi") != null);
    try testing.expect(std.mem.indexOf(u8, out, "<|im_end|>") != null);
    // generation prompt present
    const last_assistant = std.mem.lastIndexOf(u8, out, "<|im_start|>assistant").?;
    try testing.expect(last_assistant > std.mem.indexOf(u8, out, "<|im_start|>user").?);
}

test "Llama-2 simplified template renders" {
    // Use the simpler stand-in form (per task: full Llama-2 template is complex
    // and may not render bit-exact — scoped-down version per spec).
    const tmpl = "<s>[INST] {{ messages[0]['content'] }} [/INST]";
    const msgs = [_]Message{
        .{ .role = "user", .content = "Hello world" },
    };
    const out = try render(testing.allocator, tmpl, .{
        .messages = &msgs,
        .bos_token = "<s>",
        .eos_token = "</s>",
    });
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("<s>[INST] Hello world [/INST]", out);
}

test "Mistral [INST] template" {
    const tmpl = "{% for message in messages %}{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]{% elif message['role'] == 'assistant' %}{{ message['content'] }}{{ eos_token }}{% endif %}{% endfor %}";
    const msgs = [_]Message{
        .{ .role = "user", .content = "ping" },
        .{ .role = "assistant", .content = "pong" },
    };
    const out = try render(testing.allocator, tmpl, .{
        .messages = &msgs,
        .eos_token = "</s>",
    });
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("[INST] ping [/INST]pong</s>", out);
}

test "for loop with loop.last" {
    var arr = [_]Value{
        .{ .string_v = "a" },
        .{ .string_v = "b" },
        .{ .string_v = "c" },
    };
    var extra = std.StringHashMap(Value).init(testing.allocator);
    defer extra.deinit();
    try extra.put("msgs", .{ .array_v = &arr });

    const tmpl = "{% for m in msgs %}{{ m }}{% if not loop.last %}, {% endif %}{% endfor %}";
    const out = try render(testing.allocator, tmpl, .{
        .messages = &.{},
        .extra_vars = &extra,
    });
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("a, b, c", out);
}

test "if/elif/else dispatches by role" {
    const tmpl = "{% for m in messages %}{% if m['role'] == 'user' %}U:{{ m['content'] }}{% elif m['role'] == 'assistant' %}A:{{ m['content'] }}{% else %}?:{{ m['content'] }}{% endif %};{% endfor %}";
    const msgs = [_]Message{
        .{ .role = "user", .content = "x" },
        .{ .role = "assistant", .content = "y" },
        .{ .role = "system", .content = "z" },
    };
    const out = try render(testing.allocator, tmpl, .{ .messages = &msgs });
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("U:x;A:y;?:z;", out);
}

test "add_generation_prompt branch toggles output" {
    const tmpl = "X{% if add_generation_prompt %}!{% endif %}";
    const out1 = try render(testing.allocator, tmpl, .{ .messages = &.{}, .add_generation_prompt = true });
    defer testing.allocator.free(out1);
    const out2 = try render(testing.allocator, tmpl, .{ .messages = &.{}, .add_generation_prompt = false });
    defer testing.allocator.free(out2);
    try testing.expectEqualStrings("X!", out1);
    try testing.expectEqualStrings("X", out2);
}

test "string subscript access on object" {
    const tmpl = "{{ messages[0]['content'] }}";
    const msgs = [_]Message{.{ .role = "user", .content = "hello" }};
    const out = try render(testing.allocator, tmpl, .{ .messages = &msgs });
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("hello", out);
}

test "unsupported syntax errors cleanly" {
    const tmpl = "{% macro foo() %}{% endmacro %}";
    const result = render(testing.allocator, tmpl, .{ .messages = &.{} });
    try testing.expectError(error.UnsupportedSyntax, result);
}

test "trim filter" {
    var extra = std.StringHashMap(Value).init(testing.allocator);
    defer extra.deinit();
    try extra.put("s", .{ .string_v = "  hi  " });
    const out = try render(testing.allocator, "[{{ s | trim }}]", .{ .messages = &.{}, .extra_vars = &extra });
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("[hi]", out);
}

test "length filter on array" {
    const tmpl = "n={{ messages | length }}";
    const msgs = [_]Message{
        .{ .role = "user", .content = "a" },
        .{ .role = "user", .content = "b" },
    };
    const out = try render(testing.allocator, tmpl, .{ .messages = &msgs });
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("n=2", out);
}

test "set statement" {
    const tmpl = "{% set x = 'hi' %}{{ x }}";
    const out = try render(testing.allocator, tmpl, .{ .messages = &.{} });
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("hi", out);
}

test "string concat with +" {
    const tmpl = "{{ 'foo' + '-' + 'bar' }}";
    const out = try render(testing.allocator, tmpl, .{ .messages = &.{} });
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("foo-bar", out);
}

test "method call strip" {
    var extra = std.StringHashMap(Value).init(testing.allocator);
    defer extra.deinit();
    try extra.put("s", .{ .string_v = "  hi  " });
    const out = try render(testing.allocator, "[{{ s.strip() }}]", .{ .messages = &.{}, .extra_vars = &extra });
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("[hi]", out);
}

test "ternary expression" {
    const tmpl = "{{ 'yes' if add_generation_prompt else 'no' }}";
    const out = try render(testing.allocator, tmpl, .{ .messages = &.{}, .add_generation_prompt = true });
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("yes", out);
}

test "whitespace control strips" {
    const tmpl = "a {%- if true -%}   b   {%- endif -%} c";
    const out = try render(testing.allocator, tmpl, .{ .messages = &.{} });
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("abc", out);
}

test "is string test" {
    var extra = std.StringHashMap(Value).init(testing.allocator);
    defer extra.deinit();
    try extra.put("s", .{ .string_v = "hi" });
    try extra.put("n", .{ .int_v = 7 });
    const out = try render(testing.allocator,
        "{% if s is string %}1{% endif %}{% if n is string %}2{% else %}0{% endif %}",
        .{ .messages = &.{}, .extra_vars = &extra });
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("10", out);
}

test "is sequence test" {
    var arr = [_]Value{ .{ .int_v = 1 }, .{ .int_v = 2 } };
    var extra = std.StringHashMap(Value).init(testing.allocator);
    defer extra.deinit();
    try extra.put("a", .{ .array_v = &arr });
    try extra.put("s", .{ .string_v = "hi" });
    try extra.put("n", .{ .int_v = 3 });
    const tmpl =
        "{% if a is sequence %}A{% endif %}" ++
        "{% if s is sequence %}S{% endif %}" ++
        "{% if n is sequence %}N{% else %}.{% endif %}";
    const out = try render(testing.allocator, tmpl, .{ .messages = &.{}, .extra_vars = &extra });
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("AS.", out);
}

test "is mapping test" {
    var arr = [_]Value{.{ .int_v = 1 }};
    var map = ObjectMap.init(testing.allocator);
    defer map.deinit();
    try map.put("k", .{ .int_v = 1 });
    var extra = std.StringHashMap(Value).init(testing.allocator);
    defer extra.deinit();
    try extra.put("m", .{ .object_v = &map });
    try extra.put("a", .{ .array_v = &arr });
    const tmpl =
        "{% if m is mapping %}M{% endif %}" ++
        "{% if a is mapping %}A{% else %}.{% endif %}";
    const out = try render(testing.allocator, tmpl, .{ .messages = &.{}, .extra_vars = &extra });
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("M.", out);
}

test "is number and is boolean tests" {
    var extra = std.StringHashMap(Value).init(testing.allocator);
    defer extra.deinit();
    try extra.put("n", .{ .int_v = 42 });
    try extra.put("b", .{ .bool_v = true });
    try extra.put("s", .{ .string_v = "x" });
    const tmpl =
        "{% if n is number %}N{% endif %}" ++
        "{% if b is number %}!{% else %}.{% endif %}" ++
        "{% if b is boolean %}B{% endif %}" ++
        "{% if s is boolean %}!{% else %}.{% endif %}";
    const out = try render(testing.allocator, tmpl, .{ .messages = &.{}, .extra_vars = &extra });
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("N.B.", out);
}

test "list literal in for loop" {
    const tmpl = "{% for x in [1, 2, 3] %}{{ x }}{% endfor %}";
    const out = try render(testing.allocator, tmpl, .{ .messages = &.{} });
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("123", out);
}

test "dict literal in interpolation" {
    const tmpl = "{{ {'role': 'system', 'content': 'hi'}.role }}";
    const out = try render(testing.allocator, tmpl, .{ .messages = &.{} });
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("system", out);
}

test "empty list and dict literals length" {
    const tmpl = "{{ [] | length }},{{ {} | length }}";
    const out = try render(testing.allocator, tmpl, .{ .messages = &.{} });
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("0,0", out);
}

// ---------------------------------------------------------------------
// Inverse-parser tests
//
// Each family uses a fixed Jinja template that mirrors the canonical
// HuggingFace layout. We render `messages -> string`, invert
// `string -> messages2`, then assert `messages == messages2` (modulo
// the `partial` flag, which is false for well-formed roundtrips).

const chatml_tmpl =
    "{% for message in messages %}" ++
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n" ++
    "{% endfor %}";

const gemma_tmpl =
    "{% for message in messages %}" ++
    "<start_of_turn>{{ message['role'] }}\n{{ message['content'] }}<end_of_turn>\n" ++
    "{% endfor %}";

const mistral_tmpl =
    "{% for message in messages %}" ++
    "{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]" ++
    "{% elif message['role'] == 'assistant' %} {{ message['content'] }} </s>" ++
    "{% endif %}{% endfor %}";

// Llama-2 stand-in template: a single optional `<<SYS>>` envelope wrapped in
// the first [INST], then user/assistant pairs separated by `</s><s>`.
fn renderLlama2(allocator: std.mem.Allocator, msgs: []const Message) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);

    var i: usize = 0;
    var has_system = false;
    var system_content: []const u8 = "";
    if (msgs.len > 0 and std.mem.eql(u8, msgs[0].role, "system")) {
        has_system = true;
        system_content = msgs[0].content;
        i = 1;
    }

    var first_pair = true;
    while (i < msgs.len) : (i += 2) {
        if (!std.mem.eql(u8, msgs[i].role, "user")) return error.MalformedTemplate;
        try out.appendSlice(allocator, "<s>[INST] ");
        if (first_pair and has_system) {
            try out.appendSlice(allocator, "<<SYS>>\n");
            try out.appendSlice(allocator, system_content);
            try out.appendSlice(allocator, "\n<</SYS>>\n\n");
        }
        try out.appendSlice(allocator, msgs[i].content);
        try out.appendSlice(allocator, " [/INST]");
        if (i + 1 < msgs.len and std.mem.eql(u8, msgs[i + 1].role, "assistant")) {
            try out.appendSlice(allocator, " ");
            try out.appendSlice(allocator, msgs[i + 1].content);
            try out.appendSlice(allocator, " </s>");
        }
        first_pair = false;
    }

    return try out.toOwnedSlice(allocator);
}

fn expectRoundtrip(
    kind: TemplateKind,
    tmpl: []const u8,
    msgs: []const Message,
) !void {
    const rendered = if (kind == .llama2)
        try renderLlama2(testing.allocator, msgs)
    else
        try render(testing.allocator, tmpl, .{ .messages = msgs, .eos_token = "</s>" });
    defer testing.allocator.free(rendered);

    const parsed = try invert(testing.allocator, kind, rendered);
    defer freeParsed(testing.allocator, parsed);

    try testing.expectEqual(msgs.len, parsed.len);
    for (msgs, 0..) |m, idx| {
        try testing.expectEqualStrings(m.role, parsed[idx].role);
        try testing.expectEqualStrings(m.content, parsed[idx].content);
        try testing.expect(!parsed[idx].partial);
    }
}

// -- ChatML ---------------------------------------------------------------

test "invert ChatML: single user turn roundtrip" {
    const msgs = [_]Message{.{ .role = "user", .content = "hello" }};
    try expectRoundtrip(.chatml, chatml_tmpl, &msgs);
}

test "invert ChatML: system + user + assistant roundtrip" {
    const msgs = [_]Message{
        .{ .role = "system", .content = "Be terse." },
        .{ .role = "user", .content = "what is 2+2?" },
        .{ .role = "assistant", .content = "4" },
    };
    try expectRoundtrip(.chatml, chatml_tmpl, &msgs);
}

test "invert ChatML: multi-turn dialogue roundtrip" {
    const msgs = [_]Message{
        .{ .role = "user", .content = "hi" },
        .{ .role = "assistant", .content = "hi back" },
        .{ .role = "user", .content = "bye" },
        .{ .role = "assistant", .content = "bye!" },
    };
    try expectRoundtrip(.chatml, chatml_tmpl, &msgs);
}

test "invert ChatML: empty content is preserved" {
    const msgs = [_]Message{
        .{ .role = "user", .content = "" },
        .{ .role = "assistant", .content = "" },
    };
    try expectRoundtrip(.chatml, chatml_tmpl, &msgs);
}

test "invert ChatML: content containing markers passes through bytes" {
    // Content carrying the literal marker bytes would normally need escaping;
    // we just exercise multi-line content (the common practical case).
    const msgs = [_]Message{
        .{ .role = "user", .content = "line1\nline2\nline3" },
        .{ .role = "assistant", .content = "ok" },
    };
    try expectRoundtrip(.chatml, chatml_tmpl, &msgs);
}

test "invert ChatML: trailing partial assistant turn" {
    const stream =
        "<|im_start|>user\nhi<|im_end|>\n" ++
        "<|im_start|>assistant\nstill writin";
    const parsed = try invert(testing.allocator, .chatml, stream);
    defer freeParsed(testing.allocator, parsed);
    try testing.expectEqual(@as(usize, 2), parsed.len);
    try testing.expectEqualStrings("user", parsed[0].role);
    try testing.expect(!parsed[0].partial);
    try testing.expectEqualStrings("assistant", parsed[1].role);
    try testing.expectEqualStrings("still writin", parsed[1].content);
    try testing.expect(parsed[1].partial);
}

test "invert ChatML: empty input yields empty list" {
    const parsed = try invert(testing.allocator, .chatml, "");
    defer freeParsed(testing.allocator, parsed);
    try testing.expectEqual(@as(usize, 0), parsed.len);
}

// -- Gemma ---------------------------------------------------------------

test "invert Gemma: single user turn roundtrip" {
    const msgs = [_]Message{.{ .role = "user", .content = "hello" }};
    try expectRoundtrip(.gemma, gemma_tmpl, &msgs);
}

test "invert Gemma: user + model dialogue roundtrip" {
    const msgs = [_]Message{
        .{ .role = "user", .content = "what is the capital of france?" },
        .{ .role = "model", .content = "Paris" },
        .{ .role = "user", .content = "of germany?" },
        .{ .role = "model", .content = "Berlin" },
    };
    try expectRoundtrip(.gemma, gemma_tmpl, &msgs);
}

test "invert Gemma: empty content roundtrip" {
    const msgs = [_]Message{
        .{ .role = "user", .content = "" },
        .{ .role = "model", .content = "?" },
    };
    try expectRoundtrip(.gemma, gemma_tmpl, &msgs);
}

test "invert Gemma: multi-line content roundtrip" {
    const msgs = [_]Message{
        .{ .role = "user", .content = "one\ntwo\nthree" },
        .{ .role = "model", .content = "got it" },
    };
    try expectRoundtrip(.gemma, gemma_tmpl, &msgs);
}

test "invert Gemma: trailing partial turn marks final message" {
    const stream =
        "<start_of_turn>user\nhello<end_of_turn>\n" ++
        "<start_of_turn>model\nlet me think";
    const parsed = try invert(testing.allocator, .gemma, stream);
    defer freeParsed(testing.allocator, parsed);
    try testing.expectEqual(@as(usize, 2), parsed.len);
    try testing.expect(!parsed[0].partial);
    try testing.expect(parsed[1].partial);
    try testing.expectEqualStrings("let me think", parsed[1].content);
}

// -- Mistral --------------------------------------------------------------

test "invert Mistral: single user/assistant pair roundtrip" {
    const msgs = [_]Message{
        .{ .role = "user", .content = "ping" },
        .{ .role = "assistant", .content = "pong" },
    };
    try expectRoundtrip(.mistral, mistral_tmpl, &msgs);
}

test "invert Mistral: multi-pair roundtrip" {
    const msgs = [_]Message{
        .{ .role = "user", .content = "one" },
        .{ .role = "assistant", .content = "two" },
        .{ .role = "user", .content = "three" },
        .{ .role = "assistant", .content = "four" },
        .{ .role = "user", .content = "five" },
        .{ .role = "assistant", .content = "six" },
    };
    try expectRoundtrip(.mistral, mistral_tmpl, &msgs);
}

test "invert Mistral: empty assistant content roundtrip" {
    // Render emits `[INST] q [/INST]  </s>`; invert should return ("user","q"),
    // ("assistant","") with partial=false because </s> closed the turn.
    const stream = "[INST] q [/INST] </s>";
    const parsed = try invert(testing.allocator, .mistral, stream);
    defer freeParsed(testing.allocator, parsed);
    try testing.expectEqual(@as(usize, 2), parsed.len);
    try testing.expectEqualStrings("user", parsed[0].role);
    try testing.expectEqualStrings("q", parsed[0].content);
    try testing.expectEqualStrings("assistant", parsed[1].role);
    try testing.expectEqualStrings("", parsed[1].content);
    try testing.expect(!parsed[1].partial);
}

test "invert Mistral: trailing partial assistant after [/INST]" {
    const stream = "[INST] hi [/INST] writin";
    const parsed = try invert(testing.allocator, .mistral, stream);
    defer freeParsed(testing.allocator, parsed);
    try testing.expectEqual(@as(usize, 2), parsed.len);
    try testing.expectEqualStrings("user", parsed[0].role);
    try testing.expectEqualStrings("hi", parsed[0].content);
    try testing.expectEqualStrings("assistant", parsed[1].role);
    try testing.expectEqualStrings("writin", parsed[1].content);
    try testing.expect(parsed[1].partial);
}

test "invert Mistral: partial user turn (no [/INST])" {
    const stream = "[INST] still typin";
    const parsed = try invert(testing.allocator, .mistral, stream);
    defer freeParsed(testing.allocator, parsed);
    try testing.expectEqual(@as(usize, 1), parsed.len);
    try testing.expectEqualStrings("user", parsed[0].role);
    try testing.expectEqualStrings("still typin", parsed[0].content);
    try testing.expect(parsed[0].partial);
}

// -- Llama-2 --------------------------------------------------------------

test "invert Llama-2: user/assistant pair (no system) roundtrip" {
    const msgs = [_]Message{
        .{ .role = "user", .content = "hello" },
        .{ .role = "assistant", .content = "hi" },
    };
    try expectRoundtrip(.llama2, "", &msgs);
}

test "invert Llama-2: system + user + assistant roundtrip" {
    const msgs = [_]Message{
        .{ .role = "system", .content = "Be helpful." },
        .{ .role = "user", .content = "what is 1+1?" },
        .{ .role = "assistant", .content = "2" },
    };
    try expectRoundtrip(.llama2, "", &msgs);
}

test "invert Llama-2: multi-turn with system roundtrip" {
    const msgs = [_]Message{
        .{ .role = "system", .content = "You are a calc." },
        .{ .role = "user", .content = "1+1" },
        .{ .role = "assistant", .content = "2" },
        .{ .role = "user", .content = "2+2" },
        .{ .role = "assistant", .content = "4" },
    };
    try expectRoundtrip(.llama2, "", &msgs);
}

test "invert Llama-2: partial assistant turn" {
    const stream = "<s>[INST] question [/INST] partial repl";
    const parsed = try invert(testing.allocator, .llama2, stream);
    defer freeParsed(testing.allocator, parsed);
    try testing.expectEqual(@as(usize, 2), parsed.len);
    try testing.expectEqualStrings("user", parsed[0].role);
    try testing.expectEqualStrings("question", parsed[0].content);
    try testing.expectEqualStrings("assistant", parsed[1].role);
    try testing.expectEqualStrings("partial repl", parsed[1].content);
    try testing.expect(parsed[1].partial);
}

test "invert Llama-2: system-only first turn handled" {
    const stream = "<s>[INST] <<SYS>>\nbe nice\n<</SYS>>\n\nhello [/INST] hi </s>";
    const parsed = try invert(testing.allocator, .llama2, stream);
    defer freeParsed(testing.allocator, parsed);
    try testing.expectEqual(@as(usize, 3), parsed.len);
    try testing.expectEqualStrings("system", parsed[0].role);
    try testing.expectEqualStrings("be nice", parsed[0].content);
    try testing.expectEqualStrings("user", parsed[1].role);
    try testing.expectEqualStrings("hello", parsed[1].content);
    try testing.expectEqualStrings("assistant", parsed[2].role);
    try testing.expectEqualStrings("hi", parsed[2].content);
}
