#!/usr/bin/env python3
"""TokenMonster diff-categorizer for ztok's Monster encoder.

ztok's Monster encoder now matches TokenMonster-Go (TM-Go) at ~92-93/100
on nocapcode vocabs and ~96/100 on full-capcode vocabs at 100 lines
(was ~80-85 nocapcode / ~56-77 full-capcode before the .ztm v2 alias +
encoder fixes — see bench/RESULTS.md "TM Monster equivalence" and the
bench/tm_parity_gate.sh floor). This tool buckets each still-diverging
line by root cause so the remaining gap can be attacked next.

It reuses the existing harness plumbing:
  * ztok side: shells out to `bench_cross --kind monster ... --dump-sample`
    exactly like `bench/equivalence_check.py` does, and parses the same
    `LINE <idx> <id> <id> ...` stderr format via `parse_dump()`.
  * TM-Go side: drives the `tokenmonster` Python package (the same path
    `bench/bench_competitors.py --lib tokenmonster` uses) — `vocab.tokenize`
    for ids and `vocab.decode` for bytes.

For decoding ztok ids -> bytes we read the `.ztm` vocab file DIRECTLY rather
than shelling to a `ztok decode` CLI. Rationale:
  * The `.ztm` v2 binary layout is fully specified by the writer in
    `bench/convert_tm_to_ztm.py::write_ztm` (and the reader in
    `src/monster_io.zig::readBytes`); parsing it is ~30 lines and has no
    runtime dependency on a built ztok binary.
  * `idBytes(id)` in ztok is literally `bytes[offsets[id]:offsets[id+1]]`,
    so a direct file read reproduces the decoder's per-token bytes exactly
    (the `.concat` decoder ztok uses for Monster just concatenates them).
  * It keeps the tool self-contained and runnable even before the Zig
    binary is built.

Usage:
  python3 bench/tm_diff_report.py \\
      --vocab bench/vocabs/tm_englishcode_32k.ztm \\
      --corpus bench/corpora/english.txt --lines 100 [--json] [--out report.md]

Exit codes:
  0  ran (report emitted; a decode-mismatch flag is reported in-band)
  1  bad inputs (missing vocab/corpus, ztok dump empty, etc.)
  2  the `tokenmonster` package isn't importable
"""
from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys

# Reuse the existing harness's stderr-dump parser instead of reinventing it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from equivalence_check import parse_dump  # noqa: E402

ZTOK_BENCH = os.environ.get("ZTOK_BENCH_CROSS", "zig-out/bin/bench_cross")

# ---------------------------------------------------------------------------
# Capcode marker byte sets. Mirrors src/monster.zig::isCapcodeMarker and the
# CAPCODE_* constants in src/monster_io.zig:
#   nocapcode (capcode==1): 0x7F is the forward-DEL marker.
#   full ztok-native (capcode==2): 0x0E / 0x0F / 0x11.
#   full TM-printable (capcode==3): 'C'/'W'/'D' = 0x43 / 0x57 / 0x44.
# We test against the UNION so the categorizer doesn't have to know which
# marker style a given vocab uses; any of these bytes appearing at the
# divergence is enough to flag `capcode_marker`.
# ---------------------------------------------------------------------------
CAPCODE_MARKER_BYTES = frozenset(
    {0x7F, 0x0E, 0x0F, 0x11, 0x43, 0x57, 0x44}
)
# The TM-printable markers C/W/D are also ordinary letters in plain English
# text, so naively flagging every 'C'/'W'/'D' byte would mislabel huge
# numbers of normal-text diffs. We treat the printable trio as a marker only
# for full-capcode vocabs (capcode==3); for nocapcode/none vocabs only 0x7F
# (and the C0 trio, which never appear in text) count.
CAPCODE_MARKER_C0 = frozenset({0x7F, 0x0E, 0x0F, 0x11})
CAPCODE_MARKER_TM_PRINTABLE = frozenset({0x43, 0x57, 0x44})


# ---------------------------------------------------------------------------
# .ztm v2 vocab reader. Layout (little-endian), from convert_tm_to_ztm.py and
# src/monster_io.zig:
#   0..4    magic "ZTM\x01" (v1) or "ZTM\x02" (v2)
#   4       version (1 or 2)
#   5       capcode (0 none / 1 nocapcode / 2 full-ztok / 3 full-TM)
#   6       max_token_length (u8)
#   7       norm_flag (u8)
#   8..12   count        (u32 LE)
#   12..16  unk_id       (u32 LE)
#   16..20  bytes_total  (u32 LE)
#   20..              bytes[bytes_total]
#   ...               offsets[count+1] (u32 LE each, monotonic, first 0)
#   ...               nwords[count]    (u8 each)
#   v2 only:  u32 alias_count, then { u32 id, u32 len, u8[len] } * count
# We only need the primary `tokens` list (id -> bytes) plus capcode to decode
# and categorize; aliases are extra byte-forms that resolve to existing ids,
# which don't change the decoded bytes of a given id.
# ---------------------------------------------------------------------------
class ZtmVocab:
    def __init__(self, tokens: list[bytes], capcode: int, unk_id: int):
        self.tokens = tokens          # id -> bytes
        self.capcode = capcode        # 0/1/2/3
        self.unk_id = unk_id
        # Reverse map bytes -> id for the missing_piece check. If two ids
        # share bytes (shouldn't happen for primaries, but be safe) the
        # lowest id wins.
        self.bytes_to_id: dict[bytes, int] = {}
        for i, b in enumerate(tokens):
            if b and b not in self.bytes_to_id:
                self.bytes_to_id[b] = i

    def id_bytes(self, tid: int) -> bytes:
        if 0 <= tid < len(self.tokens):
            return self.tokens[tid]
        return b""

    def marker_bytes(self) -> frozenset:
        """Marker byte set appropriate for this vocab's capcode mode."""
        if self.capcode == 3:
            # full TM-printable: C/W/D plus the C0 trio (never both in
            # one vocab, but harmless to union).
            return CAPCODE_MARKER_C0 | CAPCODE_MARKER_TM_PRINTABLE
        # none / nocapcode / full-ztok: only the C0-style markers; the
        # printable C/W/D are plain letters here.
        return CAPCODE_MARKER_C0


def read_ztm(path: str) -> ZtmVocab:
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) < 20 or raw[0:3] != b"ZTM":
        raise SystemExit(f"not a .ztm file (bad magic): {path}")
    version = raw[4]
    # v3 = v2 token table + an appended per-twin alt section. The token
    # table (offsets/bytes) this reader needs is byte-identical to v2;
    # the appended section lives past it and is simply ignored here.
    if version not in (1, 2, 3):
        raise SystemExit(f"unsupported .ztm version {version}: {path}")
    capcode = raw[5]
    # raw[6] = max_token_length, raw[7] = norm_flag — not needed here.
    count = struct.unpack_from("<I", raw, 8)[0]
    unk_id = struct.unpack_from("<I", raw, 12)[0]
    bytes_total = struct.unpack_from("<I", raw, 16)[0]
    off = 20
    blob = raw[off:off + bytes_total]
    off += bytes_total
    # offsets: count+1 u32 entries.
    noff = count + 1
    offsets = list(struct.unpack_from("<%dI" % noff, raw, off))
    off += noff * 4
    # (nwords block + optional alias section follow; not needed for decode.)
    tokens: list[bytes] = []
    for i in range(count):
        tokens.append(blob[offsets[i]:offsets[i + 1]])
    return ZtmVocab(tokens, capcode, unk_id)


# ---------------------------------------------------------------------------
# Stream alignment + byte-divergence computation.
#
# We turn each id stream into a list of (id, decoded_bytes) and walk a
# cumulative byte cursor down both. The first byte position at which the two
# streams' token boundaries / contents disagree is the "byte-divergence
# offset". Concretely: we concatenate decoded bytes on both sides and find
# the first index `k` where ztok_bytes[k] != tm_bytes[k] (or where one runs
# out). When both decode to the SAME total bytes (the normal case for a pure
# segmentation difference) this is the index just past the longest common
# byte prefix. When the decoded bytes differ in CONTENT (not just split
# points) that's a decode/normalization bug and we flag it loudly.
# ---------------------------------------------------------------------------
def decode_ztok(ids: list[int], vocab: ZtmVocab) -> bytes:
    return b"".join(vocab.id_bytes(i) for i in ids)


# ---------------------------------------------------------------------------
# Capcode-aware decode via the shared TM vocab.
#
# THE OLD BUG: this tool reconstructed each side's bytes by raw-concatenating
# `vocab.id_bytes(i)` straight out of the .ztm. But the .ztm token bytes still
# carry embedded capcode markers — in nocapcode mode the forward-delete marker
# `\x7F` prefixes word-continuation tokens (e.g. id 760 = b"\x7F Hu", id 1054 =
# b"\x7F ck"). TM's real decoder COLLAPSES `\x7F ` (DEL+space) to stitch words
# back together, so `Huck` round-trips correctly. Raw concatenation instead
# produced b"\x7F Hu\x7F ck..." — which differs from TM's clean decode starting
# at BYTE 0 (the leading `\x7F`). Result: every nocapcode line with a mid-word
# split got mislabeled `decode_bug_length @byte 0`. There were never any real
# decode bugs.
#
# THE FIX: ztok's Monster .ztm ids live in the SAME id-space as the TM vocab
# (the converter preserves ids), so we decode BOTH id streams through TM's own
# `vocab.decode` (capcode/marker-aware). To map a byte offset back onto a token
# boundary we decode incrementally: tm_token_spans(ids) returns the cumulative
# decoded-byte length after each token, computed via prefix decodes. This makes
# the byte cursor land on CLEAN decoded text, so straddling-token analysis is
# meaningful.
# ---------------------------------------------------------------------------
def tm_decode_ids(tm_vocab, ids: list[int]) -> bytes:
    if not ids:
        return b""
    try:
        dec = tm_vocab.decode([int(x) for x in ids])
    except Exception:
        return b""
    return dec.encode("utf-8") if isinstance(dec, str) else bytes(dec)


def tm_token_spans(tm_vocab, ids: list[int]) -> list[int]:
    """Cumulative decoded-byte length after each successive token.

    spans[k] = len(tm.decode(ids[:k+1])). Because capcode markers can make a
    token contribute zero or "negative-looking" bytes (a DEL deletes a prior
    space), spans are clamped to be monotonic non-decreasing so the cursor
    walk below stays well-defined.
    """
    spans: list[int] = []
    prev = 0
    for k in range(len(ids)):
        cur = len(tm_decode_ids(tm_vocab, ids[: k + 1]))
        if cur < prev:
            cur = prev
        spans.append(cur)
        prev = cur
    return spans


def token_index_at_byte_spans(spans: list[int], byte_pos: int) -> int:
    """Token index whose decoded span covers `byte_pos` (cumulative spans)."""
    for i, end in enumerate(spans):
        if byte_pos < end:
            return i
    return len(spans)


def first_byte_divergence(a: bytes, b: bytes) -> int:
    """Index of first differing byte; len(min) if one is a prefix of the
    other; -1 if identical."""
    n = min(len(a), len(b))
    for k in range(n):
        if a[k] != b[k]:
            return k
    if len(a) == len(b):
        return -1
    return n


def token_index_at_byte(ids: list[int], piece_bytes: list[bytes], byte_pos: int) -> int:
    """Which token index covers `byte_pos` in the concatenated stream."""
    cum = 0
    for i, pb in enumerate(piece_bytes):
        nxt = cum + len(pb)
        if byte_pos < nxt:
            return i
        cum = nxt
    return len(ids)  # past the end


def context_window(data: bytes, center: int, radius: int = 24) -> bytes:
    lo = max(0, center - radius)
    hi = min(len(data), center + radius)
    return data[lo:hi]


# ---------------------------------------------------------------------------
# Categorization. For each diverging line we already have the byte-divergence
# offset `d`. We find the ztok token and the TM-Go token that straddle `d`
# and classify:
#
#   decode_bug        — ztok and TM decode to DIFFERENT total bytes/content.
#                       Reported separately and loudly; everything else
#                       assumes the two streams cover the same text.
#   capcode_marker    — either side's straddling token bytes contain a
#                       capcode marker byte (per the vocab's marker set).
#   missing_piece     — TM-Go's straddling token's bytes don't resolve to any
#                       single ztok vocab id (vocab-conversion gap: TM picked
#                       a token ztok's converted vocab can't name).
#   score_tiebreak    — ztok exposes scores AND the two straddling pieces tie
#                       on score. ztok's .ztm carries no scores, so in
#                       practice we emit `score_tiebreak_suspected` whenever
#                       greedy_vs_ungreedy holds but neither side is longer
#                       (equal-length alternative pieces at the same offset).
#   greedy_vs_ungreedy— both straddling pieces exist in the ztok vocab and
#                       ztok chose a different segmentation length than TM at
#                       the same position.
#   unknown           — none of the above.
# ---------------------------------------------------------------------------
def categorize(
    z_ids: list[int],
    t_ids: list[int],
    z_spans: list[int],
    t_spans: list[int],
    z_bytes: bytes,
    t_bytes: bytes,
    vocab: ZtmVocab,
) -> dict:
    """Categorize one diverging line.

    z_bytes / t_bytes are the CAPCODE-AWARE decodes (via TM's `vocab.decode`)
    of each id stream — markers already applied, so a byte mismatch here is a
    genuine decode bug, not a stray marker. z_spans / t_spans are the
    cumulative decoded-byte lengths after each token (from tm_token_spans),
    used to map a byte offset onto the straddling token.

    The "straddling token" we report is the RAW .ztm bytes of the id covering
    the divergence (so the hex is human-readable and capcode markers are
    visible for the capcode_marker bucket); length comparisons for
    greedy/ungreedy use the DECODED span length, which is what actually
    determines segmentation.
    """
    markers = vocab.marker_bytes()

    # --- decode-bug check first, now on capcode-AWARE bytes. With markers
    # applied, the two streams should reconstruct the SAME source text; if not
    # it's a real normalization/decode bug.
    decode_mismatch = z_bytes != t_bytes
    d = first_byte_divergence(z_bytes, t_bytes)

    rec: dict = {
        "byte_divergence": d,
        "ztok_bytes_len": len(z_bytes),
        "tm_bytes_len": len(t_bytes),
    }

    if decode_mismatch:
        kind = "decode_bug_length" if len(z_bytes) != len(t_bytes) else "decode_bug_content"
        rec["category"] = "decode_bug"
        rec["decode_bug_kind"] = kind
        rec["ztok_ctx"] = context_window(z_bytes, max(d, 0)).hex()
        rec["tm_ctx"] = context_window(t_bytes, max(d, 0)).hex()
        return rec

    if d < 0:
        # Identical decoded text => pure re-segmentation. Locate the first
        # token whose id differs and use its decoded-byte start as the cursor.
        d = _first_id_divergence_byte(z_ids, t_ids, z_spans)
        rec["byte_divergence"] = d

    zi = token_index_at_byte_spans(z_spans, d)
    ti = token_index_at_byte_spans(t_spans, d)
    z_id = z_ids[zi] if zi < len(z_ids) else None
    t_id = t_ids[ti] if ti < len(t_ids) else None
    # Raw .ztm bytes (markers visible) for display + marker detection.
    z_tok = vocab.id_bytes(z_id) if z_id is not None else b""
    t_tok = vocab.id_bytes(t_id) if t_id is not None else b""
    # Decoded-byte span length of each straddling token (segmentation length).
    z_span_len = (z_spans[zi] - (z_spans[zi - 1] if zi > 0 else 0)) if zi < len(z_spans) else 0
    t_span_len = (t_spans[ti] - (t_spans[ti - 1] if ti > 0 else 0)) if ti < len(t_spans) else 0
    rec["ztok_token_idx"] = zi
    rec["tm_token_idx"] = ti
    rec["ztok_token"] = z_tok.hex()
    rec["tm_token"] = t_tok.hex()
    rec["ztok_token_id"] = z_id
    rec["tm_token_id"] = t_id
    rec["ztok_span_len"] = z_span_len
    rec["tm_span_len"] = t_span_len

    # --- missing_piece: TM's chosen token's id falls outside the ztok vocab
    # range (the converted .ztm can't name what TM picked). Since ids are
    # shared this should be rare; it flags a real conversion gap.
    if t_id is not None and t_id >= len(vocab.tokens):
        rec["category"] = "missing_piece"
        rec["tm_token_in_ztok_vocab"] = False
        return rec
    rec["tm_token_in_ztok_vocab"] = True

    # --- Marker analysis. In nocapcode mode the DEL marker `\x7F` prefixes
    # almost every mid-word continuation token, so a marker BYTE being present
    # is NOT a distinguishing signal — both sides usually carry it. The marker
    # only matters when it's ASYMMETRIC: one side's straddling token starts
    # with a `\x7F `-style DEL+space stitch and the other does NOT. That is the
    # ground-truth ungreedy pattern (TM picks a leading-SPACE token + emits a
    # forward-delete to stitch mid-word, where ztok keeps a DEL-prefixed
    # token). We surface that as its own bucket so it doesn't hide inside the
    # generic greedy/ungreedy count.
    def has_del_prefix(tok: bytes) -> bool:
        # DEL+space (`\x7F `) for nocapcode/none, or printable `D `/C0 for full.
        if tok[:2] == b"\x7f\x20":
            return True
        return bool(tok) and tok[0] in markers and tok[:1] != tok[:0]

    z_mark = has_del_prefix(z_tok)
    t_mark = has_del_prefix(t_tok)
    if z_mark != t_mark:
        # Asymmetric DEL stitch: the core ungreedy delete-marker divergence.
        rec["category"] = "capcode_marker"
        rec["marker_side"] = "ztok" if z_mark else "tm"
        rec["ztok_longer"] = z_span_len > t_span_len
        return rec

    # --- both ids valid, symmetric markers: pure segmentation choice. Use
    # DECODED span lengths (capcode-aware), which is what determines greediness.
    if z_span_len == t_span_len and z_id != t_id:
        rec["category"] = "score_tiebreak_suspected"
        return rec
    if z_span_len != t_span_len:
        rec["category"] = "greedy_vs_ungreedy"
        rec["ztok_longer"] = z_span_len > t_span_len
        return rec

    rec["category"] = "unknown"
    return rec


def _first_id_divergence_byte(
    z_ids: list[int],
    t_ids: list[int],
    z_spans: list[int],
) -> int:
    """Decoded-byte offset at which the two id streams first disagree.

    Walk both id lists in lockstep; at the first differing index return the
    decoded-byte offset reached just BEFORE that token on the ztok side
    (z_spans[i-1], or 0 for the first token).
    """
    n = min(len(z_ids), len(t_ids))
    for i in range(n):
        if z_ids[i] != t_ids[i]:
            return z_spans[i - 1] if i > 0 else 0
    return z_spans[n - 1] if n > 0 else 0


# ---------------------------------------------------------------------------
# Driver: run ztok dump, run TM-Go reference, align, categorize, report.
# ---------------------------------------------------------------------------
def run_ztok_dump(vocab_path: str, corpus: str, lines_n: int) -> dict[int, list[int]]:
    p = subprocess.run(
        [
            ZTOK_BENCH,
            "--kind", "monster", "--model", vocab_path,
            "--corpus", corpus, "--iters", "1",
            "--dump-sample", "--sample-lines", str(lines_n),
        ],
        capture_output=True,
    )
    return parse_dump(p.stderr)


def load_tm_vocab(vocab_path: str):
    """Load the TM vocab handle (kept alive for incremental decode)."""
    import tokenmonster  # raised to caller for the exit-2 path

    # Prefer a sibling .vocab next to the .ztm (that's what the converter
    # consumed). If absent, pass the given path through — tokenmonster.load
    # also resolves vocab *names* and bare paths.
    tm_path = vocab_path
    if vocab_path.endswith(".ztm"):
        cand = vocab_path[:-4] + ".vocab"
        if os.path.exists(cand):
            tm_path = cand
    return tokenmonster.load(tm_path)


def run_tm_reference(tm_vocab, corpus: str, lines_n: int):
    """Tokenize each line with the given TM vocab handle. Returns
    ids_by_line. (Decoding is now done lazily/incrementally during
    categorization via tm_decode_ids / tm_token_spans.)"""
    ids_by_line: dict[int, list[int]] = {}
    with open(corpus, "rb") as f:
        raw = f.read()
    for i, line in enumerate(raw.split(b"\n")[:lines_n]):
        r = tm_vocab.tokenize(line)
        ids_by_line[i] = [int(x) for x in (r if r is not None else [])]
    return ids_by_line


def emit_markdown(out_path: str, vocab_path: str, corpus: str, records: list[dict],
                  buckets: dict[str, int], total_lines: int, decode_bugs: list[dict]):
    lines: list[str] = []
    lines.append(f"# TM diff report\n")
    lines.append(f"- vocab: `{vocab_path}`")
    lines.append(f"- corpus: `{corpus}`")
    lines.append(f"- lines compared: {total_lines}")
    lines.append(f"- diverging lines: {len(records)}\n")

    lines.append("## Summary buckets\n")
    if buckets:
        order = sorted(buckets.items(), key=lambda kv: -kv[1])
        lines.append(", ".join(f"**{n} {cat}**" for cat, n in order))
    else:
        lines.append("_no divergences_")
    lines.append("")

    if decode_bugs:
        lines.append("## !! DECODE BUGS (decoded text differs) !!\n")
        lines.append(f"{len(decode_bugs)} line(s) where ztok and TM-Go do NOT "
                     "decode to the same bytes. These are correctness bugs, "
                     "not segmentation choices.\n")
        lines.append("| line | kind | div@ | ztok ctx (hex) | tm ctx (hex) |")
        lines.append("|---|---|---|---|---|")
        for r in decode_bugs:
            lines.append(
                f"| {r['line']} | {r['decode_bug_kind']} | {r['byte_divergence']} "
                f"| `{r['ztok_ctx']}` | `{r['tm_ctx']}` |"
            )
        lines.append("")

    lines.append("## Diverging lines\n")
    lines.append("| line | category | div@ | ztok tok (hex) | tm tok (hex) | text |")
    lines.append("|---|---|---|---|---|---|")
    for r in records:
        text = r.get("text", "")
        if len(text) > 60:
            text = text[:57] + "..."
        text = text.replace("|", "\\|").replace("\n", "\\n")
        lines.append(
            f"| {r['line']} | {r['category']} | {r.get('byte_divergence', '?')} "
            f"| `{r.get('ztok_token', '')}` | `{r.get('tm_token', '')}` | {text} |"
        )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vocab", required=True, help="ztok .ztm vocab path")
    ap.add_argument("--corpus", required=True, help="corpus text path")
    ap.add_argument("--lines", type=int, default=100, help="lines to compare")
    ap.add_argument("--json", action="store_true",
                    help="emit one NDJSON record per diverging line on stdout")
    ap.add_argument("--out", default=None, help="write a markdown report here")
    args = ap.parse_args()

    if not os.path.exists(args.vocab):
        print(f"missing vocab: {args.vocab}", file=sys.stderr)
        return 1
    if not os.path.exists(args.corpus):
        print(f"missing corpus: {args.corpus}", file=sys.stderr)
        return 1

    # Be defensive: TM package missing => clear message + exit 2.
    try:
        import tokenmonster  # noqa: F401
    except ImportError:
        print("the `tokenmonster` package is not installed; run "
              "`pip install tokenmonster`", file=sys.stderr)
        return 2

    vocab = read_ztm(args.vocab)

    ztok_ids = run_ztok_dump(args.vocab, args.corpus, args.lines)
    if not ztok_ids:
        print("FAIL: ztok dump empty (is bench_cross built? "
              "set $ZTOK_BENCH_CROSS)", file=sys.stderr)
        return 1

    tm_vocab = load_tm_vocab(args.vocab)
    try:
        tm_ids = run_tm_reference(tm_vocab, args.corpus, args.lines)
        if not tm_ids:
            print("FAIL: TM-Go reference produced no ids", file=sys.stderr)
            return 1

        # Source lines (split on \n only, matching dump semantics).
        with open(args.corpus, "rb") as f:
            src_lines = f.read().split(b"\n")[:args.lines]

        total = min(max(ztok_ids), max(tm_ids)) + 1

        records: list[dict] = []
        decode_bugs: list[dict] = []
        buckets: dict[str, int] = {}

        for i in range(total):
            z = ztok_ids.get(i, [])
            t = tm_ids.get(i, [])
            if z == t:
                continue  # agrees, nothing to categorize

            # Capcode-AWARE decodes + per-token decoded-byte spans via the
            # shared TM vocab (markers applied — no more stray-\x7F false bugs).
            z_bytes = tm_decode_ids(tm_vocab, z)
            t_bytes = tm_decode_ids(tm_vocab, t)
            z_spans = tm_token_spans(tm_vocab, z)
            t_spans = tm_token_spans(tm_vocab, t)

            rec = categorize(z, t, z_spans, t_spans, z_bytes, t_bytes, vocab)
            rec["line"] = i
            rec["text"] = (src_lines[i].decode("utf-8", "replace")
                           if i < len(src_lines) else "")
            rec["ztok_ids"] = z
            rec["tm_ids"] = t

            cat = rec["category"]
            buckets[cat] = buckets.get(cat, 0) + 1
            if cat == "decode_bug":
                decode_bugs.append(rec)
            records.append(rec)

            if args.json:
                import json
                print(json.dumps(rec))
    finally:
        try:
            tokenmonster.disconnect()
        except Exception:
            pass

    # Human summary always goes to stderr so it's visible even with --json.
    if decode_bugs:
        print(f"\n!! {len(decode_bugs)} DECODE BUG line(s) detected — "
              "ztok and TM-Go decode to DIFFERENT bytes. Fix these first.",
              file=sys.stderr)
        for r in decode_bugs[:5]:
            print(f"   line {r['line']}: {r['decode_bug_kind']} @byte "
                  f"{r['byte_divergence']}", file=sys.stderr)

    summary = ", ".join(f"{n} {cat}" for cat, n in
                        sorted(buckets.items(), key=lambda kv: -kv[1]))
    print(f"\n{len(records)}/{total} lines diverge. Buckets: "
          f"{summary or '(none)'}", file=sys.stderr)

    if args.out:
        emit_markdown(args.out, args.vocab, args.corpus, records, buckets,
                      total, decode_bugs)
        print(f"wrote markdown report to {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
