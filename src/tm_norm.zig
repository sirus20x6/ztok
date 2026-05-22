//! TokenMonster-Go faithful capcode normalizer port.
//!
//! Implements the exact byte-output of TM's `capcode` Go package
//! (`refs/tokenmonster_capcode/capcode.go`) for both:
//!
//!   * `NoCapcodeEncode`  (TM ln 478-540) — \x7F + ' ' insertion at
//!                        word/digit boundaries WITH the apostrophe
//!                        bridge clause and the start-of-input
//!                        boundary treatment.
//!   * `Encode`           (TM ln 30-234) — full capcode with the W
//!                        marker REPLACING the preceding space (or
//!                        prefixed with `D + W + ' '` when no space
//!                        precedes); the retro `multiLetter` rewrite
//!                        that turns a W run into a chain of C markers
//!                        when a single lowercase letter follows; the
//!                        inside-word D-spurious-boundary checks.
//!
//! This is the byte-for-byte truth for the `tm_compat_space=true` and
//! `marker_style=.tm_printable` arms of `CapcodeNormalizer`. ztok's
//! existing `src/capcode.zig` remains the back-compat path for the
//! `.ztok` marker style and `tm_compat_space=false` mode (no W-replaces-
//! space rewrite, single-byte markers, no synthetic leading boundary).
//!
//! Each function has two flavors: `*Bytes` returns just the encoded
//! bytes (used by the encoder hot path, no per-byte attribution
//! required) and `*WithOrigin` returns both bytes and a per-byte
//! u32 origin array mapping each output byte back to the byte offset
//! of the source-input codepoint that produced it.
//!
//! Origin convention (matches `src/normalizer.zig` capcode/nocapcode
//! convention): every output byte references the start byte offset of
//! the source codepoint it derives from. Inserted markers and the
//! synthetic ' ' bytes attribute to the offset of the codepoint that
//! triggered the insertion (i.e., the codepoint being emitted NEXT,
//! since the marker prefaces it). For the W-overwrites-space case the
//! W's origin is the offset of the upper-case rune (TM's logic
//! semantically replaces the preceding space with W, so the byte at
//! position pos-1 is no longer a passthrough of the space — it's a
//! marker for the upper rune).

const std = @import("std");
const unicode_props = @import("unicode_props.zig");
const capcode_mod = @import("capcode.zig");

pub const NOCAPCODE_DELETE: u8 = capcode_mod.NOCAPCODE_DELETE; // 0x7F
pub const NOCAPCODE_SUBSTITUTE: u8 = 0x14;

// TM constants (capcode.go ln 13-14).
const APOSTROPHE: u21 = 0x27;
const APOSTROPHE2: u21 = 0x2019;

// --- 1.21 perf: precomputed ASCII attribute table ----------------
//
// One byte per ASCII char packs every classification bit the hot loop
// needs into a single load, replacing ~6 separate predicate calls
// per ASCII codepoint (letter / upper / lower / number / modifier /
// apostrophe). Profile showed ~25-30% of capcode-encode CPU was the
// per-byte classifier dispatch.
//
// Bits (LSB→MSB):
//   0: letter
//   1: upper
//   2: lower
//   3: number
//   4: modifier (always 0 for ASCII — no combining marks below U+0080)
//   5: apostrophe (ASCII 0x27 only — U+2019 needs the non-ASCII path)
//   6: ascii_space (== ' ' literally — matches TM's `rlast == ' '` test)
//   7: reserved
const ATTR_LETTER: u8 = 0x01;
const ATTR_UPPER: u8 = 0x02;
const ATTR_LOWER: u8 = 0x04;
const ATTR_NUMBER: u8 = 0x08;
const ATTR_MODIFIER: u8 = 0x10;
const ATTR_APOS: u8 = 0x20;
const ATTR_ASCII_SPACE: u8 = 0x40;

const ascii_attr_table: [128]u8 = blk: {
    var t: [128]u8 = undefined;
    for (0..128) |i| {
        var a: u8 = 0;
        const c: u8 = @intCast(i);
        if (c >= 'A' and c <= 'Z') a |= ATTR_LETTER | ATTR_UPPER;
        if (c >= 'a' and c <= 'z') a |= ATTR_LETTER | ATTR_LOWER;
        if (c >= '0' and c <= '9') a |= ATTR_NUMBER;
        if (c == 0x27) a |= ATTR_APOS;
        if (c == ' ') a |= ATTR_ASCII_SPACE;
        t[i] = a;
    }
    break :blk t;
};

// === codepoint helpers ============================================

const Cp = struct { cp: u21, len: u3 };

fn decodeCp(s: []const u8, i: usize) ?Cp {
    if (i >= s.len) return null;
    const lead = s[i];
    const seq = std.unicode.utf8ByteSequenceLength(lead) catch {
        return Cp{ .cp = lead, .len = 1 };
    };
    if (i + seq > s.len) return Cp{ .cp = lead, .len = 1 };
    const cp = std.unicode.utf8Decode(s[i..][0..seq]) catch {
        return Cp{ .cp = lead, .len = 1 };
    };
    return Cp{ .cp = cp, .len = @intCast(seq) };
}

fn isApostrophe(cp: u21) bool {
    return cp == APOSTROPHE or cp == APOSTROPHE2;
}

// TM's `unicode.IsUpper` / `unicode.IsLower` go through Go's full
// Unicode case tables. Mirror this via the existing ztok capcode case
// table — the same UCD 16.0 upper/lower pairs are embedded there.
fn isUpper(cp: u21) bool {
    return capcode_mod.toLower(cp) != cp;
}

fn isLower(cp: u21) bool {
    // A codepoint is lower iff its uppercase form (if any) differs.
    // ASCII fast path.
    if (cp >= 'a' and cp <= 'z') return true;
    if (cp < 0x80) return false;
    // For non-ASCII we don't have a direct lower->upper table public
    // from capcode_mod, so use the merged classifier: cur is letter
    // AND not upper.
    if (!unicode_props.classifyCp(cp).letter) return false;
    return !isUpper(cp);
}

fn isLetter(cp: u21) bool {
    return unicode_props.classifyCp(cp).letter;
}

fn isNumber(cp: u21) bool {
    return unicode_props.classifyCp(cp).number;
}

fn isModifier(cp: u21) bool {
    // TM's `isModifier` = Mn|Mc|Me (capcode.go ln 20-22). ztok's
    // `classifyCp.mark` returns the union of the three.
    return unicode_props.classifyCp(cp).mark;
}

fn toLowerCp(cp: u21) u21 {
    return capcode_mod.toLower(cp);
}

fn encodeCp(buf: []u8, cp: u21) usize {
    return std.unicode.utf8Encode(cp, buf) catch blk: {
        // Mirror TM's fallback — write '?' to keep buffer length
        // predictable. Triggered only on out-of-range codepoints.
        buf[0] = '?';
        break :blk 1;
    };
}

// === NoCapcode (port of TM `NoCapcodeEncode`, capcode.go ln 478-540) ===
//
// Per-rune state (TM ln 479-482):
//   rlast2, rlast    : previous two runes
//   rlast == 0       : start-of-input (treated as NOT space, NOT letter)
//
// For each rune r:
//   if isLetter(r) AND NOT (rlast == ' ' OR isLetter(rlast) OR
//                           (isLetter(rlast2) AND (rlast == '\'' OR
//                            rlast == '’')) OR isModifier(rlast)):
//      emit \x7F + ' '
//   else if isNumber(r) AND NOT (rlast == ' ' OR isNumber(rlast)):
//      emit \x7F + ' '
//   if r == \x7F: emit \x14 (substitute)
//   else: emit r (utf-8 bytes verbatim)
//
// Note: TM ALWAYS emits "\x7F + ' '" at boundaries. ztok's existing
// `capcode.NoCapcode.encode` emits only "\x7F", and the wrapping
// normalizer expands it. Here we emit both directly to mirror TM
// byte-for-byte and to make origin attribution exact.

pub fn normalizeNocapcode(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    // 1.21 perf: pre-size to the absolute worst-case envelope so the
    // inner loop can use a raw `[*]u8` cursor and never bounds-check
    // or grow. The TM nocapcode emits at most DEL+' ' (2 bytes) per
    // codepoint boundary plus the codepoint itself (up to the input's
    // byte length, since case folding doesn't apply here) — so the
    // worst case is 3× input.len for an alternating letter/digit ASCII
    // stream like "a1a1a1...". Measured expansion on a 10 MB English
    // corpus is ~1.59×; the headroom is harmless because we `realloc`
    // the buffer down to the actual written length at function return.
    const cap = input.len * 3 + 8;
    var buf = try allocator.alloc(u8, cap);
    errdefer allocator.free(buf);

    var rlast_attr: u8 = 0; // packed ASCII attrs of rlast (or 0 for non-ASCII / start)
    var rlast2_is_letter: bool = false;
    // Mirror rlast bits for the non-ASCII path. ASCII state uses
    // rlast_attr; these stay in sync (and ASCII paths only set the
    // bit-bool variants they need from cur_attr).
    var rlast_is_letter: bool = false;
    var rlast_is_number: bool = false;
    var rlast_is_modifier: bool = false;
    var rlast_is_apos: bool = false;

    var w: usize = 0;
    var i: usize = 0;
    while (i < input.len) {
        const b0 = input[i];
        if (b0 < 0x80) {
            @branchHint(.likely);
            // -- ASCII fast path --------------------------------------
            // Single packed-attribute load instead of classifyCp +
            // isApostrophe + isUpper + isLower + isModifier.
            const cur_attr = ascii_attr_table[b0];
            const cur_is_letter = (cur_attr & ATTR_LETTER) != 0;
            const cur_is_number = (cur_attr & ATTR_NUMBER) != 0;
            const cur_is_apos = (cur_attr & ATTR_APOS) != 0;

            if (cur_is_letter) {
                // TM ln 493-498: skip DEL if rlast is space|letter|
                // (letter rlast2 AND apos rlast)|modifier.
                const rlast_is_ascii_space = (rlast_attr & ATTR_ASCII_SPACE) != 0;
                const apos_bridge = rlast_is_apos and rlast2_is_letter;
                const skip = rlast_is_ascii_space or rlast_is_letter or apos_bridge or rlast_is_modifier;
                if (!skip) {
                    buf[w] = NOCAPCODE_DELETE;
                    buf[w + 1] = ' ';
                    w += 2;
                }
            } else if (cur_is_number) {
                const rlast_is_ascii_space = (rlast_attr & ATTR_ASCII_SPACE) != 0;
                if (!(rlast_is_ascii_space or rlast_is_number)) {
                    buf[w] = NOCAPCODE_DELETE;
                    buf[w + 1] = ' ';
                    w += 2;
                }
            }

            if (b0 == NOCAPCODE_DELETE) {
                buf[w] = NOCAPCODE_SUBSTITUTE;
                w += 1;
            } else {
                buf[w] = b0;
                w += 1;
            }

            // Roll state forward.
            rlast2_is_letter = rlast_is_letter;
            rlast_attr = cur_attr;
            rlast_is_letter = cur_is_letter;
            rlast_is_number = cur_is_number;
            rlast_is_modifier = false; // no ASCII modifiers
            rlast_is_apos = cur_is_apos;
            i += 1;
        } else {
            // -- Non-ASCII slow path ----------------------------------
            const dec = decodeCp(input, i) orelse break;
            const r = dec.cp;
            const n: usize = dec.len;

            const cls = unicode_props.classifyCp(r);
            const cur_is_letter = cls.letter;
            const cur_is_number = cls.number;
            const cur_is_modifier = cls.mark;
            const cur_is_apos = (r == APOSTROPHE2); // APOSTROPHE handled above

            if (cur_is_letter) {
                const rlast_is_ascii_space = (rlast_attr & ATTR_ASCII_SPACE) != 0;
                const apos_bridge = rlast_is_apos and rlast2_is_letter;
                const skip = rlast_is_ascii_space or rlast_is_letter or apos_bridge or rlast_is_modifier;
                if (!skip) {
                    buf[w] = NOCAPCODE_DELETE;
                    buf[w + 1] = ' ';
                    w += 2;
                }
            } else if (cur_is_number) {
                const rlast_is_ascii_space = (rlast_attr & ATTR_ASCII_SPACE) != 0;
                if (!(rlast_is_ascii_space or rlast_is_number)) {
                    buf[w] = NOCAPCODE_DELETE;
                    buf[w + 1] = ' ';
                    w += 2;
                }
            }

            // Non-ASCII can never be \x7F so no substitute needed.
            @memcpy(buf[w .. w + n], input[i .. i + n]);
            w += n;

            rlast2_is_letter = rlast_is_letter;
            rlast_attr = 0;
            rlast_is_letter = cur_is_letter;
            rlast_is_number = cur_is_number;
            rlast_is_modifier = cur_is_modifier;
            rlast_is_apos = cur_is_apos;
            i += n;
        }
    }

    return allocator.realloc(buf, w);
}

pub const OriginResult = struct {
    bytes: []u8,
    origin: []u32,
};

pub fn normalizeNocapcodeWithOrigin(
    allocator: std.mem.Allocator,
    input: []const u8,
    /// `input_origin` maps each byte of `input` back to a byte offset
    /// in the CALLER'S original input. When non-null, the returned
    /// origin array uses these offsets (lets the caller chain through
    /// an NFD pre-pass). When null, each input byte is its own origin
    /// (the identity map).
    input_origin: ?[]const u32,
) !OriginResult {
    var out = try std.ArrayList(u8).initCapacity(allocator, input.len + input.len / 4 + 8);
    errdefer out.deinit(allocator);
    var origin = try std.ArrayList(u32).initCapacity(allocator, input.len + input.len / 4 + 8);
    errdefer origin.deinit(allocator);

    var rlast: u21 = 0;
    var rlast2: u21 = 0;
    var rlast_is_letter = false;
    var rlast_is_number = false;
    var rlast_is_modifier = false;
    var rlast_is_apos = false;
    var rlast2_is_letter = false;

    var i: usize = 0;
    while (i < input.len) {
        const dec = decodeCp(input, i) orelse break;
        const r = dec.cp;
        const n: usize = dec.len;

        const anchor: u32 = if (input_origin) |io| (if (i < io.len) io[i] else @intCast(i)) else @intCast(i);

        const cls = unicode_props.classifyCp(r);
        const cur_is_letter = cls.letter;
        const cur_is_number = cls.number;
        const cur_is_modifier = cls.mark;
        const cur_is_apos = isApostrophe(r);

        if (cur_is_letter) {
            const rlast_is_ascii_space = (rlast == ' ');
            const apos_bridge = rlast_is_apos and rlast2_is_letter;
            const skip = rlast_is_ascii_space or rlast_is_letter or apos_bridge or rlast_is_modifier;
            if (!skip) {
                try out.append(allocator, NOCAPCODE_DELETE);
                try origin.append(allocator, anchor);
                try out.append(allocator, ' ');
                try origin.append(allocator, anchor);
            }
        } else if (cur_is_number) {
            const rlast_is_ascii_space = (rlast == ' ');
            if (!(rlast_is_ascii_space or rlast_is_number)) {
                try out.append(allocator, NOCAPCODE_DELETE);
                try origin.append(allocator, anchor);
                try out.append(allocator, ' ');
                try origin.append(allocator, anchor);
            }
        }

        if (n == 1 and input[i] == NOCAPCODE_DELETE) {
            try out.append(allocator, NOCAPCODE_SUBSTITUTE);
            try origin.append(allocator, anchor);
        } else {
            var k: usize = 0;
            while (k < n) : (k += 1) {
                try out.append(allocator, input[i + k]);
                const off: u32 = if (input_origin) |io| (if (i + k < io.len) io[i + k] else anchor) else @intCast(i + k);
                try origin.append(allocator, off);
            }
        }

        rlast2 = rlast;
        rlast2_is_letter = rlast_is_letter;
        rlast = r;
        rlast_is_letter = cur_is_letter;
        rlast_is_number = cur_is_number;
        rlast_is_modifier = cur_is_modifier;
        rlast_is_apos = cur_is_apos;

        i += n;
    }
    const bytes = try out.toOwnedSlice(allocator);
    errdefer allocator.free(bytes);
    const ori = try origin.toOwnedSlice(allocator);
    return .{ .bytes = bytes, .origin = ori };
}

// === Full capcode (port of TM `Encode`, capcode.go ln 30-234) ======
//
// State (TM ln 31-33):
//   rlast2, rlast        : previous two runes
//   i, i2, n, n2         : byte indices / lengths
//   pos                  : output write position
//   wordTokenPos         : index of the most recently emitted W marker
//                          (rewritten to C when a lone lowercase letter
//                          follows the W run)
//   inWord               : currently inside a capital run
//   multiLetter          : the current run has 2+ uppercase letters
//
// The encoder's two outer branches:
//
//   inWord == true:
//     if r is upper:
//       if rlast NOT (letter | apos | apos2 | modifier):
//         emit D + ' '
//       multiLetter = true
//       emit lower(r)
//     else if r is lower:
//       inWord = false
//       buf[wordTokenPos] = C            # rewrite W → C
//       if multiLetter:
//         retro-scan: replace each D+' ' inside the run with D+C+' ';
//         insert C before each remaining bare letter
//       if rlast NOT (letter | apos | apos2 | modifier):
//         emit D + ' '
//       emit r
//     else (non-letter):
//       if r is number:
//         inWord = false
//         if rlast NOT (space | number) and rlast != 0:
//           emit D + ' '
//       else if r NOT (apos | apos2 | modifier):
//         inWord = false
//       emit r
//
//   inWord == false:
//     if r is lower:
//       if rlast NOT (space | lower | (letter rlast2 + apos rlast) | modifier):
//         emit D + ' '
//       emit r
//     else if r is upper:
//       if rlast == ' ':
//         wordTokenPos = pos - 1
//         buf[pos-1] = W                 # rewrite trailing space → W
//         buf[pos]   = ' '
//         pos++
//       else:
//         wordTokenPos = pos + 1
//         emit D + W + ' '
//       emit lower(r); n2 = pos; multiLetter = false; inWord = true
//     else if r is number:
//       if rlast NOT (space | number):
//         emit D + ' '
//       emit r
//     else:
//       emit r
//
// rlast2, rlast updated each iteration.
//
// NOTE on TM's `rlast != 0` start-of-input gate (ln 102, 179): when
// rlast == 0 (start of input), TM still emits a D for a leading
// uppercase (rlast=0 != ' ' so the else branch fires; pos=0 so it
// writes D+W+' ' starting at 0). For a leading lowercase rlast=0 is
// neither space nor lower nor modifier, so D+' ' is emitted. For a
// leading digit, TM ln 179 explicitly checks rlast != ' ' && !IsNumber,
// which fires when rlast=0 too. Match all three.

const Markers = struct { c: u8, w: u8, d: u8 };

pub fn normalizeCapcode(
    allocator: std.mem.Allocator,
    input: []const u8,
    style: capcode_mod.MarkerStyle,
) ![]u8 {
    const mk = capcode_mod.markersFor(style);
    const m: Markers = .{ .c = mk.c, .w = mk.w, .d = mk.d };

    // 1.21 perf: pre-size to the worst-case envelope so the inner
    // loop can use a raw `[*]u8` cursor. The full capcode encoder
    // emits at most D+W+' ' (3 bytes) before a codepoint, plus the
    // codepoint's bytes (case-folded). For ASCII the absolute worst
    // case is alternating letter/digit input "A1A1A1..." that emits
    // "DW alowerD 1..." — still <4× input. Measured expansion on
    // a 10 MB English corpus is ~1.74×; the headroom is harmless
    // (realloc-down at return).
    //
    // The retro-walk in retroInsertCMarkers can insert C bytes
    // mid-buffer via std.mem.copyBackwards; that grows the live
    // length but does NOT overrun the original capacity unless a
    // multi-letter run is long enough to insert >cap bytes. We use
    // an `std.ArrayList` for the retro path (it has its own grow
    // logic) but a raw cursor for the inner loop. The lift comes
    // from the inner loop, not from the retro walk.
    const cap = input.len * 4 + 16;
    var buf_storage = try allocator.alloc(u8, cap);
    errdefer allocator.free(buf_storage);

    var w: usize = 0;
    var in_word = false;
    var multi_letter = false;
    var word_token_pos: usize = 0;
    var n2: usize = 0; // position just after the first emitted letter of the run

    var rlast_attr: u8 = 0;
    var rlast_is_letter: bool = false;
    var rlast_is_lower: bool = false;
    var rlast_is_number: bool = false;
    var rlast_is_modifier: bool = false;
    var rlast_is_apos: bool = false;
    var rlast2_is_letter: bool = false;

    var i: usize = 0;
    while (i < input.len) {
        // 1.21 perf: tight ASCII-lowercase passthrough fast-path.
        // When !in_word AND rlast was a lowercase ASCII letter, any
        // run of subsequent lowercase ASCII letters bridges without
        // state change — memcpy through. Hits English prose hard:
        // ~25% of bytes in a code/docs corpus are letters in runs of
        // 3-12 lowers. Branchless prologue cost (~1 cmp + jcc) is
        // dwarfed by the saved per-byte work (~6 attr loads + cond
        // moves + bool roll-forward) when span >= 2.
        if (!in_word and rlast_is_lower) {
            const start = i;
            while (i < input.len) {
                const c = input[i];
                if (c < 'a' or c > 'z') break;
                i += 1;
            }
            const span = i - start;
            if (span > 0) {
                @memcpy(buf_storage[w .. w + span], input[start .. start + span]);
                w += span;
                rlast_attr = ATTR_LETTER | ATTR_LOWER;
                rlast2_is_letter = true;
                rlast_is_letter = true;
                rlast_is_lower = true;
                rlast_is_number = false;
                rlast_is_modifier = false;
                rlast_is_apos = false;
                continue;
            }
        }

        // Inside-word uppercase-letter run fast path. After the first
        // upper, in_word=true and rlast_is_letter=true; subsequent
        // upper ASCII letters all hit the "bridge_ok" skip and emit
        // `b0 + 32` (case fold). Tight loop with the same branchless
        // shape as the lowercase passthrough above. Triggers on
        // acronyms ("USA", "TODO", "AWS"), shell variables, etc. —
        // any 2+ uppercase run.
        if (in_word and rlast_is_letter) {
            const start = i;
            while (i < input.len) {
                const c = input[i];
                if (c < 'A' or c > 'Z') break;
                buf_storage[w] = c + 32;
                w += 1;
                i += 1;
            }
            const span = i - start;
            if (span > 0) {
                multi_letter = true;
                rlast_attr = ATTR_LETTER | ATTR_UPPER;
                rlast2_is_letter = true;
                rlast_is_letter = true;
                rlast_is_lower = false;
                rlast_is_number = false;
                rlast_is_modifier = false;
                rlast_is_apos = false;
                continue;
            }
        }

        const b0 = input[i];
        var r: u21 = undefined;
        var n: usize = undefined;
        var cur_attr: u8 = 0;
        var cur_is_letter: bool = false;
        var cur_is_number: bool = false;
        var cur_is_modifier: bool = false;
        var cur_is_apos: bool = false;
        var cur_is_upper: bool = false;
        var cur_is_lower: bool = false;

        if (b0 < 0x80) {
            @branchHint(.likely);
            r = b0;
            n = 1;
            cur_attr = ascii_attr_table[b0];
            cur_is_letter = (cur_attr & ATTR_LETTER) != 0;
            cur_is_number = (cur_attr & ATTR_NUMBER) != 0;
            cur_is_apos = (cur_attr & ATTR_APOS) != 0;
            cur_is_upper = (cur_attr & ATTR_UPPER) != 0;
            cur_is_lower = (cur_attr & ATTR_LOWER) != 0;
            // cur_is_modifier already false; no ASCII modifiers.
        } else {
            const dec = decodeCp(input, i) orelse break;
            r = dec.cp;
            n = dec.len;
            const cls = unicode_props.classifyCp(r);
            cur_is_letter = cls.letter;
            cur_is_number = cls.number;
            cur_is_modifier = cls.mark;
            cur_is_apos = (r == APOSTROPHE2);
            if (cur_is_letter) {
                cur_is_upper = isUpper(r);
                cur_is_lower = if (cur_is_upper) false else isLower(r);
            }
        }

        const rlast_is_ascii_space = (rlast_attr & ATTR_ASCII_SPACE) != 0;

        if (in_word) {
            if (cur_is_upper) {
                // TM ln 47-54: D+' ' if rlast not letter/apos/modifier.
                const bridge_ok = rlast_is_letter or rlast_is_apos or rlast_is_modifier;
                if (!bridge_ok) {
                    buf_storage[w] = m.d;
                    buf_storage[w + 1] = ' ';
                    w += 2;
                }
                multi_letter = true;
                if (b0 >= 'A' and b0 <= 'Z') {
                    buf_storage[w] = b0 + 32;
                    w += 1;
                } else {
                    var tmp: [4]u8 = undefined;
                    const en = encodeCp(&tmp, toLowerCp(r));
                    @memcpy(buf_storage[w .. w + en], tmp[0..en]);
                    w += en;
                }
            } else if (cur_is_lower) {
                // TM ln 55-99: end of W run; rewrite W → C, retro
                // rewrite if multi_letter, then emit r.
                in_word = false;
                buf_storage[word_token_pos] = m.c;
                if (multi_letter) {
                    // Retro-walk: fall back to the (rare) heap path
                    // through ArrayList. Move our raw buffer into an
                    // ArrayList for the duration of the insert, then
                    // pull it back out. Even on a pathological corpus
                    // this fires once per W-run; the inner loop's
                    // amortized cost dominates.
                    var al = std.ArrayList(u8){
                        .items = buf_storage[0..w],
                        .capacity = buf_storage.len,
                    };
                    try retroInsertCMarkers(allocator, &al, n2, m);
                    buf_storage = al.items.ptr[0..al.capacity];
                    w = al.items.len;
                }
                const bridge_ok = rlast_is_letter or rlast_is_apos or rlast_is_modifier;
                if (!bridge_ok) {
                    buf_storage[w] = m.d;
                    buf_storage[w + 1] = ' ';
                    w += 2;
                }
                if (n == 1) {
                    buf_storage[w] = b0;
                    w += 1;
                } else {
                    @memcpy(buf_storage[w .. w + n], input[i .. i + n]);
                    w += n;
                }
            } else {
                // TM ln 100-131.
                if (cur_is_number) {
                    in_word = false;
                    if (!rlast_is_ascii_space and !rlast_is_number) {
                        buf_storage[w] = m.d;
                        buf_storage[w + 1] = ' ';
                        w += 2;
                    }
                } else if (!(cur_is_apos or cur_is_modifier)) {
                    in_word = false;
                }
                if (n == 1) {
                    buf_storage[w] = b0;
                    w += 1;
                } else {
                    @memcpy(buf_storage[w .. w + n], input[i .. i + n]);
                    w += n;
                }
            }
        } else {
            if (cur_is_lower) {
                // TM ln 134-160.
                const bridge = rlast2_is_letter and rlast_is_apos;
                const skip = rlast_is_ascii_space or rlast_is_lower or bridge or rlast_is_modifier;
                if (!skip) {
                    buf_storage[w] = m.d;
                    buf_storage[w + 1] = ' ';
                    w += 2;
                }
                if (n == 1) {
                    buf_storage[w] = b0;
                    w += 1;
                } else {
                    @memcpy(buf_storage[w .. w + n], input[i .. i + n]);
                    w += n;
                }
            } else if (cur_is_upper) {
                // TM ln 161-177: begin capital run.
                if (rlast_is_ascii_space) {
                    word_token_pos = w - 1;
                    buf_storage[w - 1] = m.w;
                    buf_storage[w] = ' ';
                    w += 1;
                } else {
                    word_token_pos = w + 1;
                    buf_storage[w] = m.d;
                    buf_storage[w + 1] = m.w;
                    buf_storage[w + 2] = ' ';
                    w += 3;
                }
                if (b0 >= 'A' and b0 <= 'Z') {
                    buf_storage[w] = b0 + 32;
                    w += 1;
                } else {
                    var tmp: [4]u8 = undefined;
                    const en = encodeCp(&tmp, toLowerCp(r));
                    @memcpy(buf_storage[w .. w + en], tmp[0..en]);
                    w += en;
                }
                n2 = w;
                multi_letter = false;
                in_word = true;
            } else if (cur_is_number) {
                if (!rlast_is_ascii_space and !rlast_is_number) {
                    buf_storage[w] = m.d;
                    buf_storage[w + 1] = ' ';
                    w += 2;
                }
                if (n == 1) {
                    buf_storage[w] = b0;
                    w += 1;
                } else {
                    @memcpy(buf_storage[w .. w + n], input[i .. i + n]);
                    w += n;
                }
            } else {
                if (n == 1) {
                    buf_storage[w] = b0;
                    w += 1;
                } else {
                    @memcpy(buf_storage[w .. w + n], input[i .. i + n]);
                    w += n;
                }
            }
        }

        rlast2_is_letter = rlast_is_letter;
        rlast_attr = cur_attr;
        rlast_is_letter = cur_is_letter;
        rlast_is_lower = cur_is_lower;
        rlast_is_number = cur_is_number;
        rlast_is_modifier = cur_is_modifier;
        rlast_is_apos = cur_is_apos;
        i += n;
    }
    return allocator.realloc(buf_storage, w);
}

// Retro-insert C markers across a multi-letter W run when a lone
// lowercase letter follows. Mirrors TM-Go `multiLetter` block
// (capcode.go Encode, the `if multiLetter { for i2 := n2; ... }` loop;
// = JS `tokenmonster.js` capcode_encode ln 924-951).
//
// On entry, `out.items[word_token_pos..]` is the run: `C` (just
// rewritten from W), ' ', <first lowercased letter>, optionally
// D+' ' before each subsequent letter for in-word boundary D's. We
// walk from `start` (=n2, position just after the FIRST letter) and
// give EVERY subsequent lowercase letter of the run its own
// capitalize-next prefix, exactly as TM-Go does:
//   * if a `D ' '` already precedes the letter (in-word boundary that
//     emitted a spurious-boundary delete), rewrite `D ' ' x` →
//     `D C ' ' x` (insert ONE `C`; net +1 byte). TM-Go Branch A:
//     copy(buf[i2+3:pos+1], buf[i2+2:pos]); buf[i2..i2+3]=D,C,' ';
//     pos++.
//   * if the letter is BARE (no preceding D+' '), insert the full
//     three-byte `D C ' '` prefix in front of it (net +3 bytes).
//     TM-Go Branch B: copy(buf[i2+3:pos+3], buf[i2:pos]);
//     buf[i2..i2+3]=D,C,' '; pos+=3.
//
// The earlier ztok revision inserted only a bare `C` in the second
// case (`C x` instead of `D C ' ' x`). That decoded back correctly
// under ztok's own decoder, but produced a byte stream that did NOT
// match the `D C ' '`-framed pieces a TM `.vocab` was trained on —
// so every CamelCase / mixed-cap-run token diverged from TM, costing
// full-capcode parity. This restores byte-for-byte TM-Go framing.
//
// TM's retro logic is byte-position based (uses i2 += 3 then += n2 in
// Go, where n2 is the just-decoded rune length). Here we walk
// codepoint-by-codepoint over the run; `p = p + 3 + dn.len` is the
// same advance (3 prefix bytes + the rune we just classified).
fn retroInsertCMarkers(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    start: usize,
    m: Markers,
) !void {
    var p = start;
    while (p < out.items.len) {
        if (p + 1 < out.items.len and out.items[p] == m.d and out.items[p + 1] == ' ') {
            // D+' '+letter — rewrite as D+C+' '+letter (TM-Go Branch A).
            // Decode the letter that follows the D+' '.
            const letter_pos = p + 2;
            if (letter_pos >= out.items.len) return;
            const dn = decodeCp(out.items, letter_pos) orelse return;
            // TM only rewrites if the next char is lower.
            if (!isLower(dn.cp)) {
                p = letter_pos + dn.len;
                continue;
            }
            // Convert "D ' ' x" → "D C ' ' x" by inserting C at p+1.
            try out.insert(allocator, p + 1, m.c);
            // After insert: out.items[p]=D, out.items[p+1]=C, out.items[p+2]=' ',
            // out.items[p+3..p+3+dn.len]=letter. Skip past it.
            p = p + 3 + dn.len;
        } else {
            // Bare lowercase letter — insert the full D+C+' ' prefix
            // (TM-Go Branch B). Three bytes inserted before the letter.
            const dn = decodeCp(out.items, p) orelse return;
            if (!isLower(dn.cp)) {
                // Not a lowercase letter (D/space/marker/etc — TM only
                // rewrites bare lowercase letters). Skip past.
                p += dn.len;
                continue;
            }
            // Insert "D C ' '" before the letter at p. ArrayList.insert
            // shifts the tail right; do it back-to-front so each insert
            // keeps the letter at the growing front edge.
            try out.insert(allocator, p, ' ');
            try out.insert(allocator, p, m.c);
            try out.insert(allocator, p, m.d);
            // Now out.items[p..p+3] = D,C,' ' and the letter is at p+3.
            p = p + 3 + dn.len;
        }
    }
}

pub fn normalizeCapcodeWithOrigin(
    allocator: std.mem.Allocator,
    input: []const u8,
    style: capcode_mod.MarkerStyle,
    input_origin: ?[]const u32,
) !OriginResult {
    const mk = capcode_mod.markersFor(style);
    const m: Markers = .{ .c = mk.c, .w = mk.w, .d = mk.d };

    var out = try std.ArrayList(u8).initCapacity(allocator, input.len + input.len / 2 + 8);
    errdefer out.deinit(allocator);
    var origin = try std.ArrayList(u32).initCapacity(allocator, input.len + input.len / 2 + 8);
    errdefer origin.deinit(allocator);

    var rlast: u21 = 0;
    var rlast2: u21 = 0;
    var in_word = false;
    var multi_letter = false;
    var word_token_pos: usize = 0;
    var n2: usize = 0;

    var rlast_is_letter = false;
    var rlast_is_lower = false;
    var rlast_is_number = false;
    var rlast_is_modifier = false;
    var rlast_is_apos = false;
    var rlast2_is_letter = false;

    var i: usize = 0;
    while (i < input.len) {
        const dec = decodeCp(input, i) orelse break;
        const r = dec.cp;
        const n: usize = dec.len;

        const anchor: u32 = if (input_origin) |io| (if (i < io.len) io[i] else @intCast(i)) else @intCast(i);

        const cls = unicode_props.classifyCp(r);
        const cur_is_letter = cls.letter;
        const cur_is_number = cls.number;
        const cur_is_modifier = cls.mark;
        const cur_is_apos = isApostrophe(r);
        var cur_is_upper = false;
        var cur_is_lower = false;
        if (cur_is_letter) {
            if (r < 0x80) {
                cur_is_upper = (r >= 'A' and r <= 'Z');
                cur_is_lower = (r >= 'a' and r <= 'z');
            } else {
                cur_is_upper = isUpper(r);
                cur_is_lower = if (cur_is_upper) false else isLower(r);
            }
        }

        if (in_word) {
            if (cur_is_upper) {
                const bridge_ok = rlast_is_letter or rlast_is_apos or rlast_is_modifier;
                if (!bridge_ok) {
                    try out.append(allocator, m.d);
                    try origin.append(allocator, anchor);
                    try out.append(allocator, ' ');
                    try origin.append(allocator, anchor);
                }
                multi_letter = true;
                var buf: [4]u8 = undefined;
                const en = encodeCp(&buf, toLowerCp(r));
                var k: usize = 0;
                while (k < en) : (k += 1) {
                    try out.append(allocator, buf[k]);
                    try origin.append(allocator, anchor);
                }
            } else if (cur_is_lower) {
                in_word = false;
                out.items[word_token_pos] = m.c;
                // (origin at word_token_pos already records its anchor.)
                if (multi_letter) {
                    try retroInsertCMarkersOrigin(allocator, &out, &origin, n2, m);
                }
                const bridge_ok = rlast_is_letter or rlast_is_apos or rlast_is_modifier;
                if (!bridge_ok) {
                    try out.append(allocator, m.d);
                    try origin.append(allocator, anchor);
                    try out.append(allocator, ' ');
                    try origin.append(allocator, anchor);
                }
                var k: usize = 0;
                while (k < n) : (k += 1) {
                    try out.append(allocator, input[i + k]);
                    const off: u32 = if (input_origin) |io| (if (i + k < io.len) io[i + k] else anchor) else @intCast(i + k);
                    try origin.append(allocator, off);
                }
            } else {
                if (cur_is_number) {
                    in_word = false;
                    const rlast_is_ascii_space = (rlast == ' ');
                    if (!rlast_is_ascii_space and !rlast_is_number) {
                        try out.append(allocator, m.d);
                        try origin.append(allocator, anchor);
                        try out.append(allocator, ' ');
                        try origin.append(allocator, anchor);
                    }
                } else if (!(cur_is_apos or cur_is_modifier)) {
                    in_word = false;
                }
                var k: usize = 0;
                while (k < n) : (k += 1) {
                    try out.append(allocator, input[i + k]);
                    const off: u32 = if (input_origin) |io| (if (i + k < io.len) io[i + k] else anchor) else @intCast(i + k);
                    try origin.append(allocator, off);
                }
            }
        } else {
            if (cur_is_lower) {
                const rlast_is_ascii_space = (rlast == ' ');
                const bridge = rlast2_is_letter and rlast_is_apos;
                const skip = rlast_is_ascii_space or rlast_is_lower or bridge or rlast_is_modifier;
                if (!skip) {
                    try out.append(allocator, m.d);
                    try origin.append(allocator, anchor);
                    try out.append(allocator, ' ');
                    try origin.append(allocator, anchor);
                }
                var k: usize = 0;
                while (k < n) : (k += 1) {
                    try out.append(allocator, input[i + k]);
                    const off: u32 = if (input_origin) |io| (if (i + k < io.len) io[i + k] else anchor) else @intCast(i + k);
                    try origin.append(allocator, off);
                }
            } else if (cur_is_upper) {
                const rlast_is_ascii_space = (rlast == ' ');
                if (rlast_is_ascii_space) {
                    // Overwrite trailing ' '. word_token_pos = pos-1.
                    // The byte at pos-1 was a passthrough of the input
                    // space; reassign its origin to the upper rune.
                    word_token_pos = out.items.len - 1;
                    out.items[out.items.len - 1] = m.w;
                    origin.items[out.items.len - 1] = anchor;
                    try out.append(allocator, ' ');
                    try origin.append(allocator, anchor);
                } else {
                    try out.append(allocator, m.d);
                    try origin.append(allocator, anchor);
                    word_token_pos = out.items.len;
                    try out.append(allocator, m.w);
                    try origin.append(allocator, anchor);
                    try out.append(allocator, ' ');
                    try origin.append(allocator, anchor);
                }
                var buf: [4]u8 = undefined;
                const en = encodeCp(&buf, toLowerCp(r));
                var k: usize = 0;
                while (k < en) : (k += 1) {
                    try out.append(allocator, buf[k]);
                    try origin.append(allocator, anchor);
                }
                n2 = out.items.len;
                multi_letter = false;
                in_word = true;
            } else if (cur_is_number) {
                const rlast_is_ascii_space = (rlast == ' ');
                if (!rlast_is_ascii_space and !rlast_is_number) {
                    try out.append(allocator, m.d);
                    try origin.append(allocator, anchor);
                    try out.append(allocator, ' ');
                    try origin.append(allocator, anchor);
                }
                var k: usize = 0;
                while (k < n) : (k += 1) {
                    try out.append(allocator, input[i + k]);
                    const off: u32 = if (input_origin) |io| (if (i + k < io.len) io[i + k] else anchor) else @intCast(i + k);
                    try origin.append(allocator, off);
                }
            } else {
                var k: usize = 0;
                while (k < n) : (k += 1) {
                    try out.append(allocator, input[i + k]);
                    const off: u32 = if (input_origin) |io| (if (i + k < io.len) io[i + k] else anchor) else @intCast(i + k);
                    try origin.append(allocator, off);
                }
            }
        }

        rlast2 = rlast;
        rlast2_is_letter = rlast_is_letter;
        rlast = r;
        rlast_is_letter = cur_is_letter;
        rlast_is_lower = cur_is_lower;
        rlast_is_number = cur_is_number;
        rlast_is_modifier = cur_is_modifier;
        rlast_is_apos = cur_is_apos;

        i += n;
    }

    const bytes = try out.toOwnedSlice(allocator);
    errdefer allocator.free(bytes);
    const ori = try origin.toOwnedSlice(allocator);
    return .{ .bytes = bytes, .origin = ori };
}

// Origin-aware version of retroInsertCMarkers. The inserted marker
// bytes inherit the origin of the lowercase letter they prefix (the
// letter is the codepoint that "triggered" the per-character cap).
// We read that origin BEFORE inserting (insert shifts the tail right).
//
// Mirrors the byte framing of `retroInsertCMarkers` above:
//   * Branch A (`D ' '` already present): insert one `C` → `D C ' '`.
//   * Branch B (bare letter): insert the full `D C ' '` prefix.
fn retroInsertCMarkersOrigin(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    origin: *std.ArrayList(u32),
    start: usize,
    m: Markers,
) !void {
    var p = start;
    while (p < out.items.len) {
        if (p + 1 < out.items.len and out.items[p] == m.d and out.items[p + 1] == ' ') {
            const letter_pos = p + 2;
            if (letter_pos >= out.items.len) return;
            const dn = decodeCp(out.items, letter_pos) orelse return;
            if (!isLower(dn.cp)) {
                p = letter_pos + dn.len;
                continue;
            }
            // Insert C at p+1; new C's origin = origin of the letter at p+2.
            const anchor = if (letter_pos < origin.items.len) origin.items[letter_pos] else origin.items[origin.items.len - 1];
            try out.insert(allocator, p + 1, m.c);
            try origin.insert(allocator, p + 1, anchor);
            p = p + 3 + dn.len;
        } else {
            const dn = decodeCp(out.items, p) orelse return;
            if (!isLower(dn.cp)) {
                p += dn.len;
                continue;
            }
            // Insert the full D+C+' ' prefix before the letter at p.
            // All three bytes inherit the letter's origin anchor.
            const anchor = if (p < origin.items.len) origin.items[p] else origin.items[origin.items.len - 1];
            try out.insert(allocator, p, ' ');
            try origin.insert(allocator, p, anchor);
            try out.insert(allocator, p, m.c);
            try origin.insert(allocator, p, anchor);
            try out.insert(allocator, p, m.d);
            try origin.insert(allocator, p, anchor);
            p = p + 3 + dn.len;
        }
    }
}

// ===================== TESTS =====================

const testing = std.testing;

test "tm_norm nocapcode: lowercase word with no leading space gets D+' ' prefix" {
    // TM-Go: 'pretraining' has rlast=0 (not space) at first letter,
    // so we emit \x7F+' ' before 'p'. No internal boundaries (all
    // letters bridge via isLetter(rlast)).
    const out = try normalizeNocapcode(testing.allocator, "pretraining");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("\x7F pretraining", out);
}

test "tm_norm nocapcode: input with leading space — no DEL prefix" {
    // rlast=0 at start, then we see ' '. The space is not a letter or
    // number, so no DEL is emitted. Then 'a' sees rlast=' ' (literal
    // ASCII), so the skip clause matches; no DEL.
    const out = try normalizeNocapcode(testing.allocator, " apples");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings(" apples", out);
}

test "tm_norm nocapcode: 'a/b' triggers DEL at start AND after '/'" {
    const out = try normalizeNocapcode(testing.allocator, "a/b");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("\x7F a/\x7F b", out);
}

test "tm_norm nocapcode: apostrophe bridge — don't D+' ' inside 'don't'" {
    // 'd' at start: DEL prefix.
    // 'o','n' bridge via rlast letter.
    // '\'' is not letter — skipped to literal.
    // 't' check: rlast='\'', rlast2='n' (letter) → bridge → skip.
    const out = try normalizeNocapcode(testing.allocator, "don't");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("\x7F don't", out);
}

test "tm_norm nocapcode: literal \\x7F in input → substitute 0x14" {
    const out = try normalizeNocapcode(testing.allocator, "a\x7Fb");
    defer testing.allocator.free(out);
    // 'a' → "\x7F a". '\x7F' is not letter/number, rlast='a' (letter).
    // letter branch skip check: rlast is letter → skip. number branch
    // doesn't fire (not number). So no DEL inserted, and the literal
    // \x7F gets substituted to \x14. Then 'b': rlast=\x7F (now char
    // 127, not letter not number not space not modifier not apos),
    // letter branch: not space, not letter, no apos bridge, not
    // modifier → emit DEL+' '.
    try testing.expectEqualStrings("\x7F a\x14\x7F b", out);
}

test "tm_norm nocapcode origin: each output byte has a valid origin into the input" {
    const r = try normalizeNocapcodeWithOrigin(testing.allocator, "a/b", null);
    defer testing.allocator.free(r.bytes);
    defer testing.allocator.free(r.origin);
    try testing.expectEqualStrings("\x7F a/\x7F b", r.bytes);
    try testing.expectEqual(@as(usize, 7), r.origin.len);
    // Bytes: \x7F ' ' 'a' '/' \x7F ' ' 'b'
    //   anchors: 0   0  0   1  2   2   2
    try testing.expectEqual(@as(u32, 0), r.origin[0]);
    try testing.expectEqual(@as(u32, 0), r.origin[1]);
    try testing.expectEqual(@as(u32, 0), r.origin[2]);
    try testing.expectEqual(@as(u32, 1), r.origin[3]);
    try testing.expectEqual(@as(u32, 2), r.origin[4]);
    try testing.expectEqual(@as(u32, 2), r.origin[5]);
    try testing.expectEqual(@as(u32, 2), r.origin[6]);
}

test "tm_norm capcode: 'HELLO' (all caps) — W-replaces-space, lowercase emission" {
    // 'H': rlast=0 (not space). Emit D+W+' '+'h'. wordTokenPos=1, n2=4.
    //   out: "DW h" (4 bytes).
    // 'E': inWord; rlast='H' letter → no D. multi_letter=true. emit 'e'.
    //   out: "DW he"
    // 'L': same → "DW hel"
    // 'L': same → "DW hell"
    // 'O': same → "DW hello"
    // End of input; no rewrite. Final: "DW hello" with TM-printable markers.
    const out = try normalizeCapcode(testing.allocator, "HELLO", .tm_printable);
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("DW hello", out);
}

test "tm_norm capcode: 'Hello' (one upper followed by lowers) — W rewritten to C" {
    // 'H': rlast=0. emit D+W+' '+'h'. wordTokenPos=1, n2=4. inWord=true.
    // 'e': lower; in_word → exit. buf[1]=C. multi_letter=false (only 1 letter).
    //   No D needed (rlast='H' letter). Emit 'e'.
    //   Final: "DC hello".
    const out = try normalizeCapcode(testing.allocator, "Hello", .tm_printable);
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("DC hello", out);
}

test "tm_norm capcode: 'Hello world' — leading word with no space, then a plain word" {
    // "Hello" → "DC hello" (4+1=5 chars)
    // ' ': non-letter, in_word=false, just emit. → "DC hello "
    // 'w': lower; rlast=' ' → skip. → "DC hello w"
    // o,r,l,d: lower bridges. → "DC hello world"
    const out = try normalizeCapcode(testing.allocator, "Hello world", .tm_printable);
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("DC hello world", out);
}

test "tm_norm capcode: 'Hello WORLD' — second word starts with space → W replaces space" {
    // "Hello " (after Hello): out="DC hello "
    // 'W': upper, rlast=' '. wordTokenPos=pos-1=8. buf[8]='W'.
    //   Append ' '. emit 'w'. n2=11. inWord=true.
    //   out = "DC helloW w"
    // 'O': in_word, upper. rlast='W' (letter). No D. multi=true. emit 'o'.
    //   out = "DC helloW wo"
    // R,L,D: same. Final: "DC helloW world"
    const out = try normalizeCapcode(testing.allocator, "Hello WORLD", .tm_printable);
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("DC helloW world", out);
}

test "tm_norm capcode: 'USA today' — multiLetter retro rewrite, then space + lower" {
    // 'U': rlast=0. emit D W ' ' u. out="DW u". wordTokenPos=1, n2=4.
    // 'S': in_word upper. rlast='U' letter. No D. multi=true. emit s.
    //   out="DW us"
    // 'A': same. out="DW usa". multi=true.
    // ' ': non-letter. in_word=true (still, since space is not letter/digit/apos/modifier).
    //   Wait — TM ln 100-122: in_word, !upper, !lower. Then check is_number.
    //   Not number. Then check NOT (apos or modifier). Space is neither, so
    //   in_word=false. Emit ' '. out="DW usa "
    // 't': lower, in_word=false. rlast=' ' → skip. emit t. out="DW usa t"
    // o,d,a,y: bridges. Final: "DW usa today"
    const out = try normalizeCapcode(testing.allocator, "USA today", .tm_printable);
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("DW usa today", out);
}

test "tm_norm capcode: 'PCs' (multiLetter then lower) — retro rewrite triggers" {
    // 'P': rlast=0. emit D W ' ' p. out="DW p". wordTokenPos=1, n2=4.
    // 'C': in_word upper. rlast='P' letter. No D. multi=true. emit c.
    //   out="DW pc"
    // 's': in_word, lower. Exit. buf[1]=C → "DC pc". multi=true,
    //   so retro from n2=4: items[4..]="c" (single bare lowercase
    //   letter, no preceding D+' '). TM-Go Branch B inserts the full
    //   "D C ' '" prefix before 'c': out becomes "DC pDC c". Then
    //   no D needed (rlast='C' letter). Emit 's'. Final: "DC pDC cs".
    //   Decodes: "DC p"→P, "DC c"→C, "s"→s = "PCs". This is the
    //   TM-Go byte framing (each subsequent run letter gets its own
    //   D+C+' ' capitalize-next prefix), NOT a bare C.
    const out = try normalizeCapcode(testing.allocator, "PCs", .tm_printable);
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("DC pDC cs", out);
}

test "tm_norm capcode origin: every output byte maps into the input" {
    const inputs = [_][]const u8{
        "Hello",
        "HELLO",
        "Hello world",
        "Hello WORLD",
        "USA today",
        "PCs",
        "abc 123",
    };
    for (inputs) |inp| {
        const r = try normalizeCapcodeWithOrigin(testing.allocator, inp, .tm_printable, null);
        defer testing.allocator.free(r.bytes);
        defer testing.allocator.free(r.origin);
        try testing.expectEqual(r.bytes.len, r.origin.len);
        for (r.origin) |off| {
            try testing.expect(off <= inp.len);
        }
    }
}

test "tm_norm capcode: leading lowercase 'pretraining' — D+' ' prefix" {
    // 'p': not in_word, lower. rlast=0 (not space, not lower, no apos
    // bridge, not modifier). Emit D+' '+'p'. Then "retraining" all
    // bridge via lower-after-lower. Final: "D pretraining"
    const out = try normalizeCapcode(testing.allocator, "pretraining", .tm_printable);
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("D pretraining", out);
}

test "tm_norm capcode: leading digit '42' — D+' ' prefix" {
    // '4': not in_word, number. rlast=0 → not space → emit D+' '+'4'.
    // '2': not in_word, number. rlast='4' number → skip. emit '2'.
    const out = try normalizeCapcode(testing.allocator, "42", .tm_printable);
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("D 42", out);
}

// === 1.21 perf-rewrite regression tests ===
//
// These tests pin the bit-exact output of the optimized inner loop
// against the documented golden cases so a future micro-optimization
// can't silently regress behavior. The 1.18-baseline tests above
// already cover the algorithmic shape; these add coverage for the
// fast-path branches and the larger-sample equivalence the 1.21
// rewrite touched.

test "1.21 perf: capcode bit-identical across marker_style on 1 KB sample" {
    // The fast-path optimizations are marker-style-agnostic; verify
    // that .ztok and .tm_printable produce outputs that differ ONLY
    // in the marker byte values (one-to-one substitution of
    // C/W/D = 0x43/0x57/0x44 ↔ 0x0E/0x0F/0x11) — never in length
    // or structure.
    const sample =
        \\Hello world. This is a TEST of the capcode normalizer's
        \\ASCII fast path. It includes "quotes" and 'apos' and don't
        \\and McKenzie's Mac and SoMe weird MixedCAPS. Also digits:
        \\42, 1024, 65536, and 3.14159. The Quick Brown Fox Jumps
        \\Over The Lazy Dog at 12:34 on Wed 2026-05-18. NASA, USA,
        \\IBM, AT&T, McDonald's, O'Brien. End of sample.
    ;
    // Repeat a few times to push past 1 KB and exercise the
    // multi-letter retro-walk + ASCII run paths.
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(testing.allocator);
    var ri: usize = 0;
    while (ri < 4) : (ri += 1) try buf.appendSlice(testing.allocator, sample);

    const ztok_out = try normalizeCapcode(testing.allocator, buf.items, .ztok);
    defer testing.allocator.free(ztok_out);
    const tm_out = try normalizeCapcode(testing.allocator, buf.items, .tm_printable);
    defer testing.allocator.free(tm_out);

    try testing.expectEqual(ztok_out.len, tm_out.len);
    const z = capcode_mod.markersFor(.ztok);
    const t = capcode_mod.markersFor(.tm_printable);
    for (ztok_out, tm_out) |a, b| {
        if (a == z.c) try testing.expectEqual(t.c, b) else if (a == z.w) try testing.expectEqual(t.w, b) else if (a == z.d) try testing.expectEqual(t.d, b) else try testing.expectEqual(a, b);
    }
}

test "1.21 perf: nocapcode ASCII-only input — DEL+' ' only at word starts (TM-Go regression)" {
    // Pin the byte-exact DEL injection pattern on a non-trivial ASCII
    // string. Each spurious boundary (start of letter run, start of
    // digit run after non-digit, etc.) produces DEL+' ' before the
    // codepoint. The apostrophe bridge inside "don't" stays unfired.
    const inp = "Hello world don't 42 stop.";
    const out = try normalizeNocapcode(testing.allocator, inp);
    defer testing.allocator.free(out);
    // Walkthrough (rlast=0 at start):
    //   H -> letter, !space, !letter, no apos-bridge, !mod => DEL+' '+H
    //   e,l,l,o -> letter, rlast=letter => bridge => emit
    //   ' '    -> non-letter non-num => emit
    //   w      -> letter, rlast=' ' => bridge => emit
    //   o,r,l,d -> bridge => emit
    //   ' '    -> emit
    //   d      -> letter, rlast=' ' => bridge => emit
    //   o,n    -> bridge => emit
    //   '      -> non-letter => emit
    //   t      -> letter, rlast=', rlast2=n (letter) => apos bridge => emit
    //   ' '    -> emit
    //   4      -> number, rlast=' ' => bridge => emit
    //   2      -> number, rlast=4 (number) => bridge => emit
    //   ' '    -> emit
    //   s      -> letter, rlast=' ' => bridge => emit
    //   t,o,p  -> bridge => emit
    //   .      -> emit
    const expected = "\x7F Hello world don't 42 stop.";
    try testing.expectEqualStrings(expected, out);
}

test "1.21 perf: capcode .tm_printable on 'HELLO World' — correct D/W/C runs" {
    // 'H' -> rlast=0 not space => D+W+' '+'h'. wordTokenPos=1, n2=4.
    //   in_word=true. multi_letter=false. buf = "DW h"
    // 'E' -> in_word + upper. rlast='H' letter => no D. multi=true.
    //   emit 'e'. buf = "DW he"
    // L,L,O -> same. buf = "DW hello"
    // ' ' -> non-letter. in_word=true, but space is !apos !mod !num.
    //   in_word=false. emit ' '. buf = "DW hello "
    // 'W' -> upper, !in_word. rlast=' ' (ascii space) => overwrite ' '
    //   with W; emit ' '. wordTokenPos = pos-1 = 8. emit 'w'.
    //   buf = "DW helloW w". n2=11. in_word=true.
    // 'o' -> in_word + lower. Exit. buf[wordTokenPos=8] = C => "DW helloC w".
    //   Wait — that breaks "Wo". Let me re-walk.
    //
    // Actually let me re-verify against the existing test:
    //   test "tm_norm capcode: 'Hello WORLD'" expects "DC helloW world"
    //   so for "HELLO World":
    //   'H' -> D+W+' '+'h'. n2=4. wTokenPos=1. buf="DW h"
    //   'E','L','L','O' -> in_word upper bridge. multi=true. buf="DW hello"
    //   ' ' -> non-letter, not number not apos not mod. in_word=false.
    //     emit ' '. buf="DW hello "
    //   'W' -> upper !in_word. rlast=' '. Overwrite last ' ' with W,
    //     append ' '. wTokenPos = pos-1 = 8 (W's position). Emit 'w'.
    //     buf="DW helloW w". n2=11. in_word=true. multi=false.
    //   'o' -> in_word lower. Exit. buf[wTokenPos=8] = C (rewrite W to C).
    //     buf becomes "DW helloC w". multi=false so no retro. rlast='W'
    //     was letter, bridge_ok=true so no D. Emit 'o'. buf="DW helloC wo"
    //   'r','l','d' -> !in_word lower. rlast=lower => skip D. Emit.
    //     Final: "DW helloC world"
    //
    // Now back to "HELLO World":
    //   "HELLO" -> "DW hello", in_word=true, multi=true at end.
    //   ' ' -> in_word, not upper/lower/digit/apos/mod -> in_word=false.
    //     Emit ' '. buf="DW hello "
    //   'W' -> upper !in_word. rlast=' '. Overwrite with W; emit ' '.
    //     wTokenPos=8. Emit 'w'. buf="DW helloW w". n2=11. in_word=true.
    //   'o' -> in_word lower. Exit. buf[8]=C (DW helloC w). multi=false.
    //     bridge_ok (rlast=W letter). Emit 'o'. buf="DW helloC wo".
    //   'r','l','d' -> !in_word lower. rlast=lower => skip. Emit.
    //     Final: "DW helloC world"
    const out = try normalizeCapcode(testing.allocator, "HELLO World", .tm_printable);
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("DW helloC world", out);
}

test "1.21 perf: capcode origin round-trip — every output byte's origin is a valid input offset" {
    // Sweeps the WithOrigin path against several inputs and verifies:
    //   * out.len == origin.len
    //   * every origin offset is < input.len (i.e. references a real
    //     input byte) — synthetic marker bytes inherit the offset of
    //     the codepoint that triggered them, which is always a real
    //     input position.
    const inputs = [_][]const u8{
        "Hello",
        "HELLO",
        "Hello world",
        "Hello WORLD",
        "USA today",
        "PCs",
        "abc 123",
        "Hello, World! Don't stop at 42.",
        "MixedCASE With UPPER and lower runs",
    };
    for (inputs) |inp| {
        const r = try normalizeCapcodeWithOrigin(testing.allocator, inp, .tm_printable, null);
        defer testing.allocator.free(r.bytes);
        defer testing.allocator.free(r.origin);
        try testing.expectEqual(r.bytes.len, r.origin.len);
        for (r.origin) |off| {
            try testing.expect(off < inp.len);
        }
    }
}

test "tm_norm capcode: encoded bytes are non-empty and contain no uppercase ASCII letters except markers" {
    // Documenting an invariant of the encoder: every uppercase ASCII
    // input letter is case-folded to lowercase. The only uppercase
    // ASCII letters in the output are the C/W/D markers themselves.
    // This is the test we can actually run against the in-tree
    // decoder; full round-trip via ztok's `decodeStyled` is gated on
    // future decoder work (TM-Go's decoder uses an `ignore` flag for
    // the W's synthetic space and treats C/W/D as state flags rather
    // than consuming-prefixes). Once the decoder is brought to TM-Go
    // parity those round-trips will become exact.
    const cases = [_][]const u8{
        "Hello",
        "Hello world",
        "Hello WORLD",
        "abc 123",
        "PCs",
    };
    for (cases) |inp| {
        const enc = try normalizeCapcode(testing.allocator, inp, .tm_printable);
        defer testing.allocator.free(enc);
        try testing.expect(enc.len > 0);
        for (enc) |b| {
            if (b >= 'A' and b <= 'Z') {
                try testing.expect(b == 'C' or b == 'W' or b == 'D');
            }
        }
    }
}
