#!/usr/bin/env python3
"""Convert a TokenMonster `.vocab` file into ztok's `.ztm` format.

The TM `.vocab` format is described in
`refs/tokenmonster/go/tokenmonster.go::Save`. It carries capcode markers,
normalization flags, alt-token branch metadata, deleted-token history, a
256-entry beginByte LUT, and the dictionary itself. ztok's Monster
encoder doesn't run capcode or store alt indices — it just needs the
plain token byte sequences (the alt branches are recomputed from the
prefix trie at runtime), so we strip the rest.

If the vocab has capcode == 1 (deleteToken only) or capcode == 2 (full
capcode), the converter drops the literal `D` deleteToken (0x44 alone) /
capcode markers since ztok's pipeline doesn't apply them. This means
token id sequences will not match TM's native tokenization byte-for-byte
on inputs that exercise capcode — that divergence is reported by the
equivalence checker, not silently swept under the rug.

The `.ztm` format is in `src/monster_io.zig::writeBytes`:
  0..4   magic = "ZTM\\x01"
  4      version = 1
  5      capcode (0=none, 1=nocapcode/forward-delete, 2=full capcode with
          ztok-native C0 markers, 3=full capcode with TM-Go printable
          markers 'C'/'W'/'D'). When the source TM `.vocab` has
          `usingCapcode == 2` (TM-Go always uses printable markers), we
          translate to 3 (CAPCODE_FULL_TM) so ztok's loader wires a
          `.capcode { .marker_style = .tm_printable }` normalizer via
          `monster_io.LoadedMonster.recommendedNormalizer()` — that
          normalizer emits the same byte stream the vocab pieces were
          carved out of. CAPCODE_FULL (2) is reserved for ztok-side
          writers that want the C0-control markers.
  6      max_token_length (u8, cap 255)
  7      norm_flag (TM-style normalizer flag bits: bit 0 = NFD,
          bit 1 = Lowercase, ...; propagated from TM's
          `norm.Normalizer.Flag`. ztok currently only acts on the NFD
          bit; other bits are stamped for future use.)
  8..12  count (u32 LE)
  12..16 unk_id (u32 LE)
  16..20 bytes_total (u32 LE)
  20..   bytes
  ...    offsets[count+1] (u32 LE each)
  ...    nwords[count] (u8 each — recomputed on load; we write zeros)
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path


def read_uint24(buf: bytes, off: int) -> int:
    return buf[off] | (buf[off + 1] << 8) | (buf[off + 2] << 16)


def read_uint8(buf: bytes, off: int) -> int:
    return buf[off]


def read_float32(buf: bytes, off: int) -> float:
    return struct.unpack_from("<f", buf, off)[0]


# --- rune helpers (charset 0/1 == UTF-8) ----------------------------------

def _decode_rune(buf: bytes, off: int) -> tuple[int, int]:
    """Decode one UTF-8 rune at `buf[off:]`; return (codepoint, nbytes).

    Mirrors TM-Go's `decodeRune` for charset 0/1. On a malformed lead
    byte we fall back to treating the single byte as its own codepoint
    (matching Go's RuneError-with-width-1 behavior closely enough for
    the letter/number/space classifiers used below).
    """
    if off >= len(buf):
        return (0, 0)
    b0 = buf[off]
    if b0 < 0x80:
        return (b0, 1)
    if b0 >> 5 == 0b110 and off + 1 < len(buf):
        return (((b0 & 0x1F) << 6) | (buf[off + 1] & 0x3F), 2)
    if b0 >> 4 == 0b1110 and off + 2 < len(buf):
        return (
            ((b0 & 0x0F) << 12) | ((buf[off + 1] & 0x3F) << 6) | (buf[off + 2] & 0x3F),
            3,
        )
    if b0 >> 3 == 0b11110 and off + 3 < len(buf):
        return (
            ((b0 & 0x07) << 18)
            | ((buf[off + 1] & 0x3F) << 12)
            | ((buf[off + 2] & 0x3F) << 6)
            | (buf[off + 3] & 0x3F),
            4,
        )
    return (b0, 1)


def _decode_last_rune(buf: bytes) -> int:
    """Codepoint of the LAST rune in `buf` (charset 0/1 == UTF-8)."""
    if not buf:
        return 0
    i = len(buf) - 1
    # Walk back over continuation bytes (0b10xxxxxx).
    while i > 0 and (buf[i] & 0xC0) == 0x80:
        i -= 1
    cp, _ = _decode_rune(buf, i)
    return cp


def _is_letter(cp: int, using_capcode: int) -> bool:
    # TM-Go: unicode.IsLetter(r) && (capcode!=2 || r not in {C,W,D}) || Mn/Mc/Me.
    # We approximate unicode.IsLetter with str.isalpha on the single
    # codepoint (Python's isalpha covers the Unicode letter categories;
    # combining marks Mn/Mc/Me are not isalpha but are rare in these
    # vocabs and TM treats them as letters — handled below for the
    # common ASCII/Latin case).
    if using_capcode == 2 and cp in (ord("C"), ord("W"), ord("D")):
        return False
    ch = chr(cp)
    if ch.isalpha():
        return True
    # Combining marks: Python doesn't expose Mn/Mc/Me via str directly;
    # use unicodedata.
    import unicodedata

    return unicodedata.category(ch) in ("Mn", "Mc", "Me")


def _is_number(cp: int) -> bool:
    # TM-Go uses unicode.IsNumber. Python's str.isnumeric is the closest
    # match to Go's unicode.IsNumber (Nd/Nl/No categories).
    return chr(cp).isnumeric()


def _is_space(cp: int) -> bool:
    # TM-Go uses unicode.IsSpace.
    return chr(cp).isspace()


def _is_capcode(cp: int, using_capcode: int) -> bool:
    if using_capcode == 1:
        return cp == 0x7F
    if using_capcode == 2:
        return cp in (ord("C"), ord("W"), ord("D"))
    return False


def _is_alphanum(cp: int, using_capcode: int) -> bool:
    return _is_letter(cp, using_capcode) or _is_number(cp)


def parse_tm_vocab(raw: bytes) -> dict:
    off = 0
    usingCapcode = raw[off]; off += 1
    charset = raw[off]; off += 1
    normalizer_flag = raw[off]; off += 1
    level = raw[off]; off += 1
    reserve = raw[off]; off += 1
    off += 3  # reserved bytes

    # charset: 0 = none/byte, 1 = UTF-8, 2 = UTF-16. ztok's Monster only
    # handles charset 0/1 (1-byte-aligned keys). For charset 2, TM-Go uses
    # `lilbufOffset = 2` and the twin-prefix is the 2-byte UTF-16 encoding
    # of the marker+space, so the byte-length-based primary selection and
    # ztok's `\x7F `/`D ` prefix-strip in monster.zig would both be wrong.
    # Refuse rather than emit a silently-broken .ztm.
    if charset == 2:
        raise SystemExit(
            "charset==2 (UTF-16) TM vocabs are not supported by this "
            "converter: ztok's Monster assumes 1-byte-aligned keys and a "
            "1-byte marker prefix. Re-export the vocab as charset 0/1."
        )

    unkToken = read_uint24(raw, off); off += 3
    vocabSize = read_uint24(raw, off); off += 3
    nReverse = read_uint24(raw, off); off += 3
    nInfo = read_uint24(raw, off); off += 3
    deleteToken = read_uint24(raw, off); off += 3
    maxTokenLength = read_uint8(raw, off); off += 1

    # `info` stores tokens in their internal (sorted/processed) order; the
    # actual TM id is `token.alt.id`. We need the tokens ordered by id, so
    # collect (id, bytes) pairs and sort. MULTIPLE info entries can map
    # to the SAME alt_id (TM-Go's "twin entries" — e.g. `train` and
    # `\x7F train` both at alt_id 9735). We keep the FIRST byte sequence
    # seen at each id as the "primary"; subsequent ones are emitted as
    # aliases in the v2 .ztm alias section so the trie can match either
    # form. For the primary, we prefer the SHORTEST form (the bare
    # letter) because the bare form is what mid-word `LongestSubstring`
    # in TM-Go reaches; the marker-prefixed form is reachable through
    # the encoder's lilbuf machinery anyway. Empirically this lifts
    # equivalence more than picking the marker-prefixed primary.
    id_to_bytes: dict[int, bytes] = {}
    id_to_aliases: dict[int, list[bytes]] = {}
    # Per-info-entry records (TM-Go's `tokenInfo` / twin entries). Each
    # is a dict {key, flag, nwords, id}. We keep EVERY info entry,
    # including both twins (`sti` and `\x7F sti`), so the v3 per-twin
    # alt computation below can derive each twin's own alt table in its
    # own byte units — the structural fix the .ztm v3 format exposes.
    info_entries: list[dict] = []
    for _ in range(nInfo):
        klen = raw[off]; off += 1
        key = raw[off:off + klen]; off += klen
        flag = raw[off]; off += 1
        nWords = raw[off]; off += 1
        alt_index = read_uint24(raw, off); off += 3
        alt_index2 = read_uint24(raw, off); off += 3
        alt_id = read_uint24(raw, off); off += 3
        # The float `score` is the TRAINING score. TM-Go's encoder
        # (tokenmonster.go tokenize hot loop, ~:1075-1209) computes its
        # ungreedy ranking purely from the integer/byte fields
        # (`flag`, `nWords`, token `length`, and `beginByte`) — the float
        # score is NEVER read during encoding. It is only consulted at
        # vocab-GENERATION time (the `< -0.5` "duplicate"-token marker and
        # the `> 0` YAML-export filter). ztok recomputes flag/nWords/
        # beginByte itself in monster.zig's Builder, so neither the float
        # score nor the stored flag/nWords needs to be carried into the
        # .ztm. We read+discard the score here purely to advance `off`.
        _score = read_float32(raw, off); off += 4
        kbytes = bytes(key)
        info_entries.append(
            {"key": kbytes, "flag": flag, "nwords": nWords, "id": alt_id}
        )
        if alt_id in id_to_bytes:
            existing = id_to_bytes[alt_id]
            if kbytes == existing:
                continue  # exact duplicate, skip
            # Pick the LONGER as primary so decoding via `idBytes(id)`
            # yields the marker-prefixed form that the TM-Go decoder
            # also produces (capcode markers preserved for the
            # round-trip). The SHORTER form becomes an alias inserted
            # into the trie at load time, so `LongestSubstring` lookups
            # mid-word can still match it directly.
            if len(kbytes) > len(existing):
                id_to_aliases.setdefault(alt_id, []).append(existing)
                id_to_bytes[alt_id] = kbytes
            else:
                id_to_aliases.setdefault(alt_id, []).append(kbytes)
        else:
            id_to_bytes[alt_id] = kbytes

    # Skip 256-byte beginByte LUT (we don't need it).
    off += 256

    # Skip deleted-token block (vocab manipulation history; not active).
    nDeleted = read_uint24(raw, off); off += 3
    for _ in range(nDeleted):
        dlen = raw[off]; off += 1
        off += dlen   # token bytes
        off += 3      # id (uint24)
        off += 4      # score (f32)

    if off != len(raw):
        sys.stderr.write(
            f"warning: trailing {len(raw) - off} bytes after parse "
            "(may be a newer-format vocab — ignoring)\n"
        )

    # Build a dense id-ordered list. TM ids should be contiguous 0..N-1.
    max_id = max(id_to_bytes) if id_to_bytes else 0
    if max_id + 1 != len(id_to_bytes):
        sys.stderr.write(
            f"warning: TM vocab has gaps in id space (max={max_id}, "
            f"count={len(id_to_bytes)}); converted file will have gaps "
            "filled with empty tokens.\n"
        )
    ordered = [id_to_bytes.get(i, b"") for i in range(max_id + 1)]
    # Flatten aliases into a list of `(id, bytes)` pairs ordered by id.
    aliases: list[tuple[int, bytes]] = []
    for aid in sorted(id_to_aliases):
        for ab in id_to_aliases[aid]:
            aliases.append((aid, ab))
    # v3: derive per-twin alt tables. TM-Go computes each info entry's
    # `tokenOuter.{index,length,index2,length2}` from that entry's OWN
    # token bytes (refs/tokenmonster/go/tokenmonster.go:3597-3772). The
    # `.vocab` binary does NOT persist these (they're recomputed at
    # Load time), so we replicate the ladder here over the parsed info
    # entries. The result is the structural data the v2 converter
    # COLLAPSED: separate alt tables for `sti` (alts in bare units) and
    # `\x7F sti` (alts in `\x7F `-prefixed units).
    per_twin_alts = compute_per_twin_alts(info_entries, usingCapcode)

    return {
        "tokens": ordered,
        "aliases": aliases,
        "per_twin_alts": per_twin_alts,
        "unkToken": unkToken,
        "vocabSize": vocabSize,
        "maxTokenLength": maxTokenLength,
        "usingCapcode": usingCapcode,
        "charset": charset,
        "normalizer_flag": normalizer_flag,
        "deleteToken": deleteToken,
    }


def compute_per_twin_alts(
    info_entries: list[dict], using_capcode: int
) -> list[dict]:
    """Replicate TM-Go's per-info-entry alt derivation.

    For each info entry (twin), find the two best-priority in-vocab
    proper-prefix subtokens of its OWN key, ranked by TM-Go's priority
    ladder (refs/tokenmonster/go/tokenmonster.go:3597-3772). Returns a
    list of dicts:
        {"key": bytes, "id": int, "alts": [(alt_id, alt_byte_len), ...]}
    with 0, 1, or 2 alt entries (alt1 first — the better one). Only
    entries with >= 1 alt are returned; alt-less entries are skipped to
    keep the section compact (the encoder treats a missing entry as
    "no alts").

    `alt_byte_len` is the byte length of the subtoken IN THIS TWIN'S OWN
    UNITS — exactly TM-Go's `tokenOuter.length`/`length2`. The alt's
    vocab id is `vocab.info[index].alt.id`, resolved here via the
    subtoken-bytes -> id map.

    NOTE: TM-Go's priority-8 ungreedy-suffix rule (:3719-3735) depends on
    the YAML `ungreedy_suffixes` list, which is NOT carried in the binary
    `.vocab`. We therefore set hasSuffix = -1 (no suffix), matching what
    the binary file actually contains. ztok's runtime `computeAlts` makes
    the same choice (the rule is deferred — see bench/TM_AUDIT.md Phase 6).
    """
    # bytes -> id (dictionary.Find -> vocabList[index].alt.id). Each
    # distinct byte sequence maps to exactly one id in TM-Go.
    bytes_to_id: dict[bytes, int] = {}
    for e in info_entries:
        bytes_to_id.setdefault(e["key"], e["id"])

    DOES_NOT_EXIST = -1
    out: list[dict] = []
    for e in info_entries:
        token = e["key"]
        tlen = len(token)
        if tlen <= 1:
            continue

        # --- replicate the minAltSize / flag preamble (:3512-3583) ---
        r, n = _decode_rune(token, 0)
        r2, n2 = _decode_rune(token, n)
        min_alt_size = 1
        nwords = 0
        if r == ord(" "):
            if _is_alphanum(r2, using_capcode):
                nwords += 1
                min_alt_size = 2
        # word counting loop (only needed for the minAltSize<=1 reset)
        i = n + n2
        rr, rn = r2, n2
        while i < tlen:
            prev = rr
            rr, rn2 = _decode_rune(token, i)
            if prev == ord(" ") and _is_alphanum(rr, using_capcode):
                nwords += 1
            i += rn2 if rn2 > 0 else 1
        if min_alt_size == 2 and nwords <= 1:
            min_alt_size = 1

        # --- the priority ladder (:3597-3752) ---
        index1 = DOES_NOT_EXIST
        length1 = 0
        index2 = DOES_NOT_EXIST
        length2 = 0
        priority1 = 0
        priority2 = 0
        # alt index here means the alt's vocab id (we store ids directly).
        length = tlen - 1
        while length >= min_alt_size:
            subword = token[:length]
            sub_id = bytes_to_id.get(subword)
            if sub_id is not None:
                placed = False
                # priority 10: anything | space_letter or space_number
                if length <= tlen - 2 and token[length] == ord(" "):
                    nr, _ = _decode_rune(token, length + 1)
                    if _is_letter(nr, using_capcode) or _is_number(nr):
                        if priority1 < priority2 or (
                            priority1 == priority2 and length1 <= length2
                        ):
                            if priority1 < 10:
                                index1, length1, priority1 = sub_id, length, 10
                        else:
                            if priority2 < 10:
                                index2, length2, priority2 = sub_id, length, 10
                        length -= 1
                        continue

                rl = _decode_last_rune(subword)
                nr, _ = _decode_rune(token, length)

                # priority 9 (capcode==0 only): non-letter|letter, non-num|num
                if using_capcode == 0:
                    cond = (
                        (not _is_letter(rl, using_capcode) and rl != ord("_"))
                        and (_is_letter(nr, using_capcode) or nr == ord("_"))
                    ) or (not _is_number(rl) and _is_number(nr))
                    if cond:
                        if priority1 < priority2 or (
                            priority1 == priority2 and length1 <= length2
                        ):
                            if priority1 < 9:
                                index1, length1, priority1 = sub_id, length, 9
                        else:
                            if priority2 < 9:
                                index2, length2, priority2 = sub_id, length, 9
                        length -= 1
                        continue

                # priority 9: letter|non-letter, number|non-number (_ = letter)
                cond9 = (
                    (_is_letter(rl, using_capcode) or rl == ord("_"))
                    and (not _is_letter(nr, using_capcode) and nr != ord("_"))
                ) or (_is_number(rl) and not _is_number(nr))
                # priority 7: space | non-space
                cond7 = _is_space(rl) and not _is_space(nr)
                # priority 8: non-space | space
                cond8 = not _is_space(rl) and _is_space(nr)
                # priority 9: everything | capcode
                cond9c = _is_capcode(nr, using_capcode)
                if cond9:
                    if priority1 < priority2 or (
                        priority1 == priority2 and length1 <= length2
                    ):
                        if priority1 < 9:
                            index1, length1, priority1 = sub_id, length, 9
                    else:
                        if priority2 < 9:
                            index2, length2, priority2 = sub_id, length, 9
                    length -= 1
                    continue
                if cond7:
                    if priority1 < priority2 or (
                        priority1 == priority2 and length1 <= length2
                    ):
                        if priority1 < 7:
                            index1, length1, priority1 = sub_id, length, 7
                    else:
                        if priority2 < 7:
                            index2, length2, priority2 = sub_id, length, 7
                    length -= 1
                    continue
                if cond8:
                    if priority1 < priority2 or (
                        priority1 == priority2 and length1 <= length2
                    ):
                        if priority1 < 8:
                            index1, length1, priority1 = sub_id, length, 8
                    else:
                        if priority2 < 8:
                            index2, length2, priority2 = sub_id, length, 8
                    length -= 1
                    continue
                if cond9c:
                    if priority1 < priority2 or (
                        priority1 == priority2 and length1 <= length2
                    ):
                        if priority1 < 9:
                            index1, length1, priority1 = sub_id, length, 9
                    else:
                        if priority2 < 9:
                            index2, length2, priority2 = sub_id, length, 9
                    length -= 1
                    continue

                # suffix rule deferred (hasSuffix = -1); never matches.

                # priority 1: everything else
                if priority1 < priority2 or (
                    priority1 == priority2 and length1 <= length2
                ):
                    if priority1 < 1:
                        index1, length1, priority1 = sub_id, length, 1
                else:
                    if priority2 < 1:
                        index2, length2, priority2 = sub_id, length, 1
            length -= 1

        # Make sure the first alternative is the better one (:3760-3764).
        if length2 > 0 and (
            priority2 > priority1 or (priority2 == priority1 and length2 > length1)
        ):
            index1, index2 = index2, index1
            length1, length2 = length2, length1

        alts: list[tuple[int, int]] = []
        if length1 > 0 and index1 != DOES_NOT_EXIST:
            alts.append((index1, length1))
            if length2 > 0 and index2 != DOES_NOT_EXIST:
                alts.append((index2, length2))
        if alts:
            out.append({"key": token, "id": e["id"], "alts": alts})
    return out


def write_ztm(
    tokens: list[bytes],
    unk_id: int,
    path: Path,
    *,
    capcode: int = 0,
    norm_flag: int = 0,
    aliases: list[tuple[int, bytes]] | None = None,
    per_twin_alts: list[dict] | None = None,
) -> None:
    """Write the `.ztm` v2 binary that `src/monster_io.zig::readBytes` parses.

    `capcode` (0/1/2/3) and `norm_flag` (TM `norm.Normalizer.Flag` bits)
    are propagated from the source TM vocab so ztok's loader can pick a
    matching pipeline normalizer via `LoadedMonster.recommendedNormalizer()`.
    Default 0/0 preserves legacy behavior for callers that build .ztm
    files from scratch.

    `aliases` is a list of `(id, alt_bytes)` pairs encoding TM-Go's twin
    entries — additional byte sequences that resolve to an existing id
    in the primary `tokens` list. The v2 alias section is always
    emitted (a zero-count section is 4 bytes); ztok's v2 reader parses
    it and seeds the encoder trie so direct longest-match lookups can
    hit either form. v1 readers stop at the v1 EOF (`max_token_length`
    bytes + offsets + nwords) and ignore the alias section — but the
    magic byte at offset 3 / version at offset 4 are bumped to 0x02 so
    a strict v1 reader will error out cleanly rather than silently
    truncate.
    """
    if aliases is None:
        aliases = []
    if per_twin_alts is None:
        per_twin_alts = []
    count = len(tokens)
    bytes_total = sum(len(t) for t in tokens)
    if bytes_total > 0xFFFFFFFF:
        raise SystemExit("token bytes total exceeds u32 max")
    primary_max = max((len(t) for t in tokens), default=0)
    alias_max = max((len(b) for _, b in aliases), default=0)
    max_len = max(primary_max, alias_max)
    max_len_u8 = min(max_len, 255)

    if not 0 <= unk_id < count:
        # The TM unkToken can legitimately be 16777215 (DOES_NOT_EXIST),
        # which is the case for TM-Go's prebuilt 32K vocabs. When
        # unkToken == DOES_NOT_EXIST, TM-Go's encoder emits NOTHING for an
        # unmatched byte (tokenmonster.go ~:1269-1275: it only appends
        # `vocab.unkToken` when `unkToken != DOES_NOT_EXIST`, otherwise it
        # just advances and increments `missing`).
        #
        # ztok's .ztm format CANNOT represent "no unk token": the reader
        # requires `unk_id < count` (monster_io.zig ~:246) and the encoder
        # unconditionally emits `self.unk_id` for an unmatched byte
        # (monster.zig ~:1316-1326). So we fall back to the first
        # single-byte token. This is a FIDELITY GAP: on inputs containing
        # a byte outside the vocab's coverage, ztok emits a spurious
        # single-byte token id where TM-Go emits nothing. For TM-Go's
        # prebuilt 32K vocabs this never fires in practice — they are
        # byte-coverage vocabs (every byte is reachable via a single-byte
        # token), so the unmatched path is unreachable. The proper fix
        # (a DOES_NOT_EXIST sentinel honored by monster_io.zig + the
        # encoder) is owned by the Zig side; see the report.
        sys.stderr.write(
            "warning: TM unkToken == DOES_NOT_EXIST (no UNK). ztok cannot "
            "represent this; falling back to first single-byte token id. "
            "On byte-coverage vocabs (TM prebuilts) the unmatched path is "
            "unreachable, so ids stay faithful; on partial-coverage vocabs "
            "ztok will emit a spurious token where TM-Go emits nothing.\n"
        )
        unk_id = next(
            (i for i, t in enumerate(tokens) if len(t) == 1),
            0,
        )

    # alias_max must also consider per-twin alt keys (which are full
    # twin token byte sequences) so the reader's max-token-length stays
    # large enough for any byte run it must walk.
    twin_max = max((len(t.get("key", b"")) for t in per_twin_alts), default=0)
    if twin_max > max_len_u8:
        max_len_u8 = min(twin_max, 255)

    out = bytearray()
    out += b"ZTM\x03"
    out.append(3)              # version
    out.append(capcode & 0xFF) # capcode (0=none, 1=nocapcode, 2=full, 3=full_tm)
    out.append(max_len_u8)     # max_token_length (across primaries + aliases)
    out.append(norm_flag & 0xFF) # norm_flag (TM normalizer flag bits)
    out += struct.pack("<I", count)
    out += struct.pack("<I", unk_id)
    out += struct.pack("<I", bytes_total)
    for t in tokens:
        out += t
    # offsets: count+1 entries, monotonic, first==0, last==bytes_total
    acc = 0
    out += struct.pack("<I", 0)
    for t in tokens:
        acc += len(t)
        out += struct.pack("<I", acc)
    # nwords block: count bytes, all zero (Builder recomputes on load).
    out += b"\x00" * count

    # v2 alias section: appended after the v1 payload.
    #   u32 alias_count
    #   { u32 id, u32 alt_byte_len, u8[alt_byte_len] } * alias_count
    # Filter out zero-length aliases (ztok's reader rejects those).
    valid_aliases = [(aid, ab) for aid, ab in aliases if len(ab) > 0]
    out += struct.pack("<I", len(valid_aliases))
    for aid, ab in valid_aliases:
        if not 0 <= aid < count:
            raise SystemExit(f"alias id {aid} out of bounds (count={count})")
        if len(ab) > 0xFFFFFFFF:
            raise SystemExit("alias bytes length exceeds u32 max")
        out += struct.pack("<I", aid)
        out += struct.pack("<I", len(ab))
        out += ab

    # v3 per-twin alt section: appended after the v2 alias section.
    #   u32 twin_count
    #   for each twin:
    #     u32 id            (vocab id of this twin; shared across twins)
    #     u32 key_len
    #     u8[key_len] key   (this twin's byte sequence — the disambiguator)
    #     u8  n_alts        (0..2)
    #     for each alt:
    #       u32 alt_id      (vocab id to emit for the alt)
    #       u32 alt_byte_len (bytes consumed in THIS twin's units)
    valid_twins = [
        t for t in per_twin_alts
        if len(t.get("key", b"")) > 0 and len(t.get("alts", [])) > 0
    ]
    out += struct.pack("<I", len(valid_twins))
    for t in valid_twins:
        tid = t["id"]
        key = t["key"]
        talts = t["alts"][:2]  # at most 2 alts (TM-Go's alt1/alt2)
        if not 0 <= tid < count:
            raise SystemExit(f"twin id {tid} out of bounds (count={count})")
        out += struct.pack("<I", tid)
        out += struct.pack("<I", len(key))
        out += key
        out.append(len(talts))
        for alt_id, alt_len in talts:
            if not 0 <= alt_id < count:
                raise SystemExit(
                    f"twin alt id {alt_id} out of bounds (count={count})"
                )
            out += struct.pack("<I", alt_id)
            out += struct.pack("<I", alt_len)

    path.write_bytes(bytes(out))


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: convert_tm_to_ztm.py <input.vocab> <output.ztm>", file=sys.stderr)
        return 2
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    raw = src.read_bytes()
    parsed = parse_tm_vocab(raw)

    print(f"source: {src} ({len(raw)} bytes)", file=sys.stderr)
    print(f"  vocabSize:      {parsed['vocabSize']}", file=sys.stderr)
    print(f"  tokens parsed:  {len(parsed['tokens'])}", file=sys.stderr)
    print(f"  aliases:        {len(parsed['aliases'])}", file=sys.stderr)
    print(f"  per-twin alts:  {len(parsed['per_twin_alts'])}", file=sys.stderr)
    print(f"  maxTokenLength: {parsed['maxTokenLength']}", file=sys.stderr)
    print(f"  capcode:        {parsed['usingCapcode']}", file=sys.stderr)
    print(f"  charset:        {parsed['charset']}", file=sys.stderr)
    print(f"  normalizer:     0x{parsed['normalizer_flag']:02x}", file=sys.stderr)
    print(f"  unkToken:       {parsed['unkToken']}", file=sys.stderr)
    print(f"  deleteToken:    {parsed['deleteToken']}", file=sys.stderr)

    # TM-Go's `capcode.Encode` always emits printable 'C'/'W'/'D'
    # markers for `usingCapcode == 2`. Map that to ztok's
    # CAPCODE_FULL_TM (= 3) so the loader picks a normalizer that emits
    # the matching byte stream. `usingCapcode == 1` (nocapcode/0x7F) is
    # the same byte in both worlds — no remap. `usingCapcode == 0` (none)
    # — pass through unchanged.
    src_cap = parsed["usingCapcode"]
    dst_cap = 3 if src_cap == 2 else src_cap

    write_ztm(
        parsed["tokens"],
        parsed["unkToken"],
        dst,
        capcode=dst_cap,
        norm_flag=parsed["normalizer_flag"],
        aliases=parsed["aliases"],
        per_twin_alts=parsed["per_twin_alts"],
    )
    print(f"wrote {dst} ({dst.stat().st_size} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
