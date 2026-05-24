const std = @import("std");
const monster_io = @import("monster_io.zig");

pub const Format = enum {
    tiktoken,
    hf_json,
    sentencepiece,
    /// ztok's own Monster (.ztm) binary format. Magic: `'Z' 'T' 'M' 0x01`
    /// — see `monster_io.MAGIC`.
    ztm,
    /// Mistral Tekken (`tekken.json`): tiktoken-style BPE vocab with
    /// base64-encoded `token_bytes`, an explicit `special_tokens` block,
    /// and a top-level `config.pattern` regex. Distinguishable from a
    /// generic HF tokenizer.json by the `"type": "Tekkenizer"` field
    /// (set on v7+) or by the absence of HF's `model` object combined
    /// with the presence of `vocab` + `special_tokens` arrays.
    tekken,
    /// RWKV "World" vocab (`rwkv_vocab_v20230424.txt`): one entry per
    /// line, `<id> <python-repr> <byte-len>`, where the middle field is a
    /// Python `str`/`bytes` literal. Distinguished from tiktoken (`<b64>
    /// <rank>`) by the quoted repr in the middle column.
    rwkv,
    unknown,
};

/// First-pass peek for cheap magic checks (ztm, sentencepiece, tiktoken).
const PEEK = 256;
/// Extended peek for JSON-shape disambiguation. Tekken's `"type":
/// "Tekkenizer"` marker generally lands within the first KiB on real
/// files; we bump to 4 KiB to also catch hand-formatted variants that
/// put `config` first.
const JSON_PEEK = 4096;

fn isWs(c: u8) bool {
    return c == ' ' or c == '\t' or c == '\n' or c == '\r';
}

fn isB64(c: u8) bool {
    return (c >= 'A' and c <= 'Z') or
        (c >= 'a' and c <= 'z') or
        (c >= '0' and c <= '9') or
        c == '+' or c == '/' or c == '=';
}

fn isDigit(c: u8) bool {
    return c >= '0' and c <= '9';
}

fn looksLikeTiktokenLine(bytes: []const u8) bool {
    // Expect: [A-Za-z0-9+/=]+ ' ' [0-9]+ ('\n' or EOF)
    if (bytes.len == 0) return false;
    var i: usize = 0;
    const left_start = i;
    while (i < bytes.len and isB64(bytes[i])) : (i += 1) {}
    if (i == left_start) return false;
    if (i >= bytes.len) return false;
    if (bytes[i] != ' ') return false;
    i += 1;
    const right_start = i;
    while (i < bytes.len and isDigit(bytes[i])) : (i += 1) {}
    if (i == right_start) return false;
    // Must end with newline, CR, or EOF (within the peek window).
    if (i < bytes.len and bytes[i] != '\n' and bytes[i] != '\r') return false;
    return true;
}

/// RWKV World vocab line: `<id> <python-repr> <byte-len>`, e.g.
/// `0 'a' 1` or `127 b'\x7f' 1`. We require leading id digits, a space,
/// a repr that opens with a quote (optionally `b`-prefixed), and a
/// trailing space + length digits — enough to separate it from a
/// tiktoken `<base64> <rank>` line (no quote, only two columns).
fn looksLikeRwkvLine(bytes: []const u8) bool {
    var i: usize = 0;
    const id_start = i;
    while (i < bytes.len and isDigit(bytes[i])) : (i += 1) {}
    if (i == id_start) return false; // need a leading numeric id
    if (i >= bytes.len or bytes[i] != ' ') return false;
    i += 1;

    // The repr must open with a quote, optionally `b`/`B`-prefixed.
    var j = i;
    if (j < bytes.len and (bytes[j] == 'b' or bytes[j] == 'B')) j += 1;
    if (j >= bytes.len) return false;
    if (bytes[j] != '\'' and bytes[j] != '"') return false;

    // Isolate the line and require a trailing ` <digits>` length column.
    var end = i;
    while (end < bytes.len and bytes[end] != '\n' and bytes[end] != '\r') : (end += 1) {}
    const line = bytes[0..end];
    var k = line.len;
    while (k > 0 and isDigit(line[k - 1])) k -= 1;
    if (k == line.len) return false; // no trailing digits
    if (k == 0 or line[k - 1] != ' ') return false;
    return true;
}

pub fn detect(bytes: []const u8) Format {
    if (bytes.len == 0) return .unknown;

    // ztok Monster (.ztm): 4-byte magic `'Z' 'T' 'M' V` where V is the
    // format version (currently 0x01 v1, 0x02 v2, 0x03 v3). Cheapest
    // check — fixed 3-byte prefix; the 4th byte is the version. We
    // accept any known version so .ztm files written by any released
    // ztok loader.
    if (bytes.len >= 4 and bytes[0] == 'Z' and bytes[1] == 'T' and bytes[2] == 'M' and
        (bytes[3] == 0x01 or bytes[3] == 0x02 or bytes[3] == 0x03))
    {
        return .ztm;
    }

    // SentencePiece: first byte is the wire-format tag for field 1,
    // type len-delimited: (1 << 3) | 2 == 0x0A. Real SP models have a
    // small-but-nonzero varint length next. A tiktoken file with a
    // leading blank line also starts with 0x0A but the byte after is
    // typically a base64 char (ASCII letter, digit, +, /, =).
    if (bytes[0] == 0x0A and bytes.len >= 2) {
        const b1 = bytes[1];
        // If byte 1 looks like base64 (or another whitespace), treat
        // it as text and fall through to tiktoken / unknown.
        if (!isB64(b1) and !isWs(b1) and b1 != 0) {
            // Plausible varint length byte (1..127 or has high bit).
            return .sentencepiece;
        }
    }

    // Skip leading whitespace for the text-based detectors.
    var i: usize = 0;
    while (i < bytes.len and isWs(bytes[i])) : (i += 1) {}
    if (i >= bytes.len) return .unknown;

    if (bytes[i] == '{') {
        // Tekken vs generic HF tokenizer.json: both are JSON objects, but
        // Tekken has the unique markers `"type": "Tekkenizer"` (v7+) or
        // the combo `"special_tokens"` + `"token_bytes"` (any version).
        // HF files instead carry a top-level `"model"` object. We
        // substring-scan the peek window: it's a coarse signal but
        // sufficient for auto-detect's "best-effort sniff" contract.
        if (looksLikeTekkenJson(bytes[i..])) return .tekken;
        return .hf_json;
    }

    // RWKV World vocab: a quoted repr in the middle column. Checked
    // before tiktoken — both are line-based text, but the quote makes
    // RWKV unambiguous and tiktoken's two-column shape never matches.
    if (looksLikeRwkvLine(bytes[i..])) return .rwkv;

    // Try tiktoken on the first non-blank line.
    if (looksLikeTiktokenLine(bytes[i..])) return .tiktoken;

    return .unknown;
}

/// Substring-scan the JSON peek window for Tekken-distinctive markers.
/// Cheap, doesn't try to actually parse the JSON. Returns false on the
/// generic HF tokenizer.json shape (which contains a top-level `"model"`
/// object Tekken files never have).
fn looksLikeTekkenJson(bytes: []const u8) bool {
    // The strongest positive signal — explicit Tekkenizer type tag on
    // v7+ files. Match the lenient form `"type"<ws>:<ws>"Tekkenizer"`
    // by just searching for the quoted value.
    if (std.mem.indexOf(u8, bytes, "\"Tekkenizer\"") != null) return true;

    // Older / hand-rolled files may omit `type`. Fall back on shape: a
    // top-level `"special_tokens"` array AND a `"token_bytes"` field
    // (which appears in every Tekken vocab entry) together are enough
    // to distinguish from an HF tokenizer.json — HF uses `"vocab"` +
    // `"merges"` and never names a field `token_bytes`.
    const has_specials = std.mem.indexOf(u8, bytes, "\"special_tokens\"") != null;
    const has_token_bytes = std.mem.indexOf(u8, bytes, "\"token_bytes\"") != null;
    if (has_specials and has_token_bytes) return true;

    return false;
}

pub fn detectFile(path: []const u8) !Format {
    const io = std.Io.Threaded.global_single_threaded.io();
    // Use the larger JSON peek window so Tekken's `"Tekkenizer"` marker
    // (which may not land in the first 256 bytes if `vocab` is listed
    // before `type`) is reachable. Non-JSON formats are unaffected;
    // their cheap magic checks all fit inside the first few bytes.
    var buf: [JSON_PEEK]u8 = undefined;
    const slice = try std.Io.Dir.cwd().readFile(io, path, &buf);
    // Path-based hint: a file literally named `tekken.json` is the
    // canonical Mistral on-disk name, so we trust the extension even if
    // the substring scan misses the type tag (e.g. the file is so large
    // the markers fall outside JSON_PEEK).
    const f = detect(slice);
    if (f == .hf_json) {
        const base = std.fs.path.basename(path);
        if (std.mem.eql(u8, base, "tekken.json")) return .tekken;
        if (std.mem.endsWith(u8, base, ".tekken.json")) return .tekken;
    }
    return f;
}

pub fn detectWithExtension(bytes: []const u8, filename: []const u8) Format {
    const f = detect(bytes);
    if (f != .unknown) {
        // Path-based override mirrors detectFile: Mistral's canonical
        // `tekken.json` filename is itself a strong signal even when the
        // peek window doesn't carry the type marker.
        if (f == .hf_json) {
            const base = std.fs.path.basename(filename);
            if (std.mem.eql(u8, base, "tekken.json")) return .tekken;
            if (std.mem.endsWith(u8, base, ".tekken.json")) return .tekken;
        }
        return f;
    }
    const base = std.fs.path.basename(filename);
    if (std.mem.eql(u8, base, "tekken.json")) return .tekken;
    if (std.mem.endsWith(u8, base, ".tekken.json")) return .tekken;
    if (std.mem.endsWith(u8, filename, ".tiktoken")) return .tiktoken;
    if (std.mem.endsWith(u8, filename, ".json")) return .hf_json;
    if (std.mem.endsWith(u8, filename, ".model")) return .sentencepiece;
    if (std.mem.endsWith(u8, filename, ".ztm")) return .ztm;
    return .unknown;
}

// --- tests -----------------------------------------------------------------

const testing = std.testing;

test "detect hf_json with leading whitespace" {
    try testing.expectEqual(Format.hf_json, detect("  \n{ \"version\": \"1.0\" }"));
}

test "detect hf_json bare" {
    try testing.expectEqual(Format.hf_json, detect("{\"model\":{}}"));
}

test "detect sentencepiece" {
    try testing.expectEqual(Format.sentencepiece, detect("\x0A\x05<unk>"));
}

test "detect tiktoken simple" {
    try testing.expectEqual(Format.tiktoken, detect("aGVsbG8= 0\n"));
}

test "detect tiktoken with leading blank line" {
    try testing.expectEqual(Format.tiktoken, detect("\naGVsbG8= 0\n"));
}

test "detect unknown random bytes" {
    try testing.expectEqual(Format.unknown, detect("random text"));
}

test "detect rwkv world vocab" {
    try testing.expectEqual(Format.rwkv, detect("0 'a' 1\n1 'b' 1\n"));
    // bytes-literal form with an escaped hex byte
    try testing.expectEqual(Format.rwkv, detect("127 b'\\x7f' 1\n"));
    // leading blank line tolerated
    try testing.expectEqual(Format.rwkv, detect("\n0 'a' 1\n"));
}

test "rwkv sniff does not swallow tiktoken" {
    // tiktoken has no quoted middle column -> still tiktoken.
    try testing.expectEqual(Format.tiktoken, detect("aGVsbG8= 0\n"));
}

test "detectWithExtension falls back to extension" {
    // Random bytes that don't match any magic, but filename hints .model.
    const raw = "\xff\xfe\xfd\xfc";
    try testing.expectEqual(Format.sentencepiece, detectWithExtension(raw, "vocab.model"));
    try testing.expectEqual(Format.tiktoken, detectWithExtension(raw, "cl100k.tiktoken"));
    try testing.expectEqual(Format.hf_json, detectWithExtension(raw, "tokenizer.json"));
    try testing.expectEqual(Format.unknown, detectWithExtension(raw, "mystery.bin"));
}

test "detectFile reads a real file" {
    const io = std.Io.Threaded.global_single_threaded.io();
    const path = "/tmp/ztok_auto_detect_test.json";
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = "{\"version\":\"1.0\"}" });
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};
    const fmt = try detectFile(path);
    try testing.expectEqual(Format.hf_json, fmt);
}

test "detect ztm by magic" {
    // Just the 4-byte magic plus zero padding is enough — `detect` only
    // looks at the prefix and doesn't parse the rest of the header.
    var buf: [monster_io.HEADER_SIZE]u8 = @splat(0);
    @memcpy(buf[0..monster_io.MAGIC.len], &monster_io.MAGIC);
    try testing.expectEqual(Format.ztm, detect(&buf));
}

test "detect ztm by v2 magic" {
    // v2 .ztm files stamp `ZTM\x02` in the 4-byte magic plus 0x02 at
    // offset 4 (version). The detect path only inspects the magic
    // bytes so this is enough for `Format.ztm` classification.
    var buf: [monster_io.HEADER_SIZE]u8 = @splat(0);
    @memcpy(buf[0..monster_io.MAGIC_V2.len], &monster_io.MAGIC_V2);
    buf[4] = 0x02;
    try testing.expectEqual(Format.ztm, detect(&buf));
}

test "detect ztm by v3 magic" {
    // v3 .ztm files stamp `ZTM\x03` in the 4-byte magic plus 0x03 at
    // offset 4 (version). The detect path only inspects the magic
    // bytes so this is enough for `Format.ztm` classification.
    var buf: [monster_io.HEADER_SIZE]u8 = @splat(0);
    @memcpy(buf[0..monster_io.MAGIC_V3.len], &monster_io.MAGIC_V3);
    buf[4] = 0x03;
    try testing.expectEqual(Format.ztm, detect(&buf));
}

test "detect tekken by type tag" {
    const bytes = "{ \"version\": 7, \"type\": \"Tekkenizer\", \"vocab\": [] }";
    try testing.expectEqual(Format.tekken, detect(bytes));
}

test "detect tekken by shape signature" {
    // Hand-rolled Tekken JSON omitting the `type` field — disambiguates
    // via the special_tokens + token_bytes combo.
    const bytes =
        \\{
        \\  "version": 7,
        \\  "vocab": [{"rank":0,"token_bytes":"AA==","token_str":"<0x00>"}],
        \\  "special_tokens": [{"rank":0,"token_str":"<unk>","is_control":true}]
        \\}
    ;
    try testing.expectEqual(Format.tekken, detect(bytes));
}

test "detectWithExtension treats `tekken.json` as Tekken" {
    // Bytes don't carry the type tag, but the filename signals Tekken.
    const raw = "\xff\xfe\xfd\xfc";
    try testing.expectEqual(Format.tekken, detectWithExtension(raw, "tekken.json"));
    try testing.expectEqual(Format.tekken, detectWithExtension(raw, "models/foo.tekken.json"));
}

test "detectWithExtension falls back to .ztm extension" {
    // Bytes that don't match any magic; filename hints `.ztm`.
    const raw = "\xff\xfe\xfd\xfc";
    try testing.expectEqual(Format.ztm, detectWithExtension(raw, "vocab.ztm"));
}

test "detectFile recognizes a real .ztm file" {
    // Build a tiny Monster via monster_io.writeFile and detect it.
    const Monster = @import("monster.zig").Monster;
    var b = Monster.Builder.init(testing.allocator);
    defer b.deinit();
    _ = try b.addToken("a");
    const unk = try b.addToken("<unk>");
    var m = try b.finalize(unk);
    defer m.deinit();

    const path = "/tmp/ztok_auto_detect_test.ztm";
    try monster_io.writeFile(testing.allocator, &m, path);
    defer {
        const io = std.Io.Threaded.global_single_threaded.io();
        std.Io.Dir.cwd().deleteFile(io, path) catch {};
    }
    const fmt = try detectFile(path);
    try testing.expectEqual(Format.ztm, fmt);
}
