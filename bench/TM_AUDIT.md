# TM-Go ↔ ztok Monster Encoder Audit (Wave 6)

READ-ONLY line-by-line audit of `src/monster.zig` against the TokenMonster-Go
reference encoder at `refs/tokenmonster/go/tokenmonster.go`. No source was
edited; no build/bench/sweep was run.

- Reference encoder: `refs/tokenmonster/go/tokenmonster.go` — `tokenize` (`:1017-1279`),
  alt/flag/beginByte build inside `PrivateGenerateVocab` (`:3486-3801`), helpers
  `hasSuffixPos` (`:287-299`), `isLetter/isAlphaNum/isCapcode` (`:359-369`).
- ztok encoder: `src/monster.zig` — `encodeChunkImpl` (`:1154-2003`), duplicate
  `encodeChunkWithOffsetsImpl` (`:2050-…`), `tmScore` (`:177-263`), `computeFlags`
  (`:3066-3167`), `computeNwords*` (`:2963-3043`), `computeAlts`/`altPriority`/
  `hasSuffixPos` (`:3392-3666`), `finalizeWithCapcode` (`:581-964`).

Legend: ✅ matches · ⚠️ partial/suspect · ❌ missing.

---

## Phase 1 — Trie walk / candidate collection

| TM-Go behavior | TM-Go file:line | Status | ztok file:line | Notes |
|---|---|---|---|---|
| Greedy = `LongestSubstring` over `min(lenData-i, maxTokenLength)` | `:1049` | ✅ | `collectPrefixMatches` `:2661-2698`, `limit` `:2671` | ztok collects ALL terminals shortest→longest; greedy = last entry `:1381`. |
| 1-byte look-ahead pad on input (read `data[i1+length1]` past end) | `:1038-1046`,`:1073` | ✅ (different mechanism) | `next_bb = if (tail >= chunk.len) 0` `:1634` | TM pads buffer + reads `beginByte[0]`(=0). ztok maps past-end → `next_bb=0`. Same numeric effect. |
| Second look-ahead = `LongestSubstring` at `i1`/`i2`/`i3` | `:1068,:1113,:1165` | ✅ | `longestMatchIdAndLen` at `after` `:1480` | One walk yields id+len. |
| `i2 = i + original.length - forwardDelete` (alt look-ahead origin uses ALT length) | `:1112,:1164` | ✅ | `first_real_len`/`after` for b≥1 use `br_lens[b]=ap.length` `:1443-1467` | Alt branch first-token length = alt subtoken length, not greedy. Correct. |
| `forwardDelete` shrinks branchLength and nWords | `:1112,:1117,:1120` | ⚠️ | `first_len_i`/`nw1` subtract `forward_delete` `:1388,:1458,:1592` | Applied to greedy+alt uniformly; see Phase 5 `forward_delete` row for the state-set gap. |

---

## Phase 2 — Skip-gate (the single-whole-word early-out)

| TM-Go behavior | TM-Go file:line | Status | ztok file:line | Notes |
|---|---|---|---|---|
| **`flag&32` skip-gate**: if `i1<lenData && (flag&32==0 ‖ beginByte[data[i1]]!=12)` is FALSE, skip ALL scoring and emit greedy | `:1057` | ❌ | none | ztok **computes** `FLAG_SINGLE_WORD` (`:3159`) but the encoder **never reads it**. ztok always runs the branch loop. When greedy is a single whole word followed by a space, TM emits greedy unconditionally; ztok scores alts and can diverge if an alt ties/wins. **Top-tier suspect — see prioritized gaps.** |
| `i1 < lenData` end-of-input guard ⇒ no scoring, emit greedy | `:1057` | ⚠️ | greedy still scored with `second_len=0` `:1475-1483` | Outcome usually identical (empty-second greedy wins), but it is not a structural match; chunk-tail ties can resolve differently. |

---

## Phase 3 — Scoring formula (score1/2/3 and score1b/2b/3b)

| TM-Go term | TM-Go file:line | Status | ztok file:line | Notes |
|---|---|---|---|---|
| base `length + length1` (branch length) | `:1075` | ✅ | `branch_len` `:190` | |
| `+ (first>>7)+(second>>7)` (all-letter/all-punct, ×1 each) | `:1076` | ✅ | `:195-196` | |
| `+ maxZeroAnd(nWords-1)` (first), `+ maxZeroAnd(second.nWords-1)` | `:1077-1078` | ✅ | `:199-200` | |
| `+ (second>>2)&1` (second begins space) — present in score1/2/3, dropped in b-variants | `:1079` | ✅ | gated by `drop_begin_space_bonus` `:205-207` | |
| `+ (nextByte>>2)&1` (next byte is space) | `:1080` | ✅ | `:210` | |
| `+ (nWords+second.nWords+(nextByte>>3))*100` (whole-word ×100) | `:1081` | ✅ | `:215-216` | `>>3` = "not a letter" bit. |
| `- 103 * (first&1 & (second>>1))` split-word — **gated** in score1/2/3 | `:1082` | ✅ | `:235-239` (else-branch) | |
| `- 103 * (first&1)` split-word — **ungated** in score1b/2b/3b | `:1102,:1152,:1204` | ✅ | `:235-239` (`drop_begin_space_bonus` branch) | Wave-3D port confirmed. |
| `- 100 * ((first>>3)&1 & (second>>4))` split-capcode | `:1083` | ✅ | `:243-244` | |
| `- 3 * (second&1 & nextByte)` second ends inside word | `:1084` | ✅ | `:249-250` | `nextByte&1==1` only when letter. |
| `- 1` extra-token penalty (b-variants only) | `:1105,:1155,:1207` | ✅ | `extra_token_penalty` `:253`, caller passes 1 `:1770` | |
| `- 100 * LessThan(branchLength,length)` (alt shorter than greedy) | `:1132,:1156,…` | ✅ | `:257-258` gated `is_alt` | |
| `- 10000 * Equal(branchLength,length)` (alt same size as greedy) | `:1133,:1157,…` | ✅ | `:257-259` | |
| score-b uses **plain** second not phantom for its GATE | `:1088` gate | ✅ | uses `plain_second_id`/`plain_second_len` `:1700,:1719` | Wave-3D. |
| score-b gate `second.flag&2 && nextByte==1 && second.nWords==0` | `:1088,:1137,:1189` | ⚠️ | `inside_word` `:1738-1741` | ztok recasts gate as `isAsciiLetter(chunk[after])` + `isAsciiLetter(chunk[after_second])` + `nwords_tm[second]==0`. ASCII-letter ≠ TM's `beginByte==1`/`flag&2` for multi-byte runes; non-ASCII letters (Greek/Cyrillic) miss the gate. |
| score-b `length2b > length2 + 1` (lilbuf strictly longer by ≥2) | `:1092,:1141,:1193` | ⚠️ | `lb_real > eff_second_len` `:1750` | TM requires `> length2 + 1`; ztok uses `> length2`. **Off-by-one: ztok fires score-b when lilbuf is only 1 byte longer; TM requires ≥2.** See prioritized gaps. |

---

## Phase 4 — Candidate set / branch ordering / tie-break

| TM-Go behavior | TM-Go file:line | Status | ztok file:line | Notes |
|---|---|---|---|---|
| Exactly 3 first-token candidates: greedy(`original`), `original.index`(alt1), `original.index2`(alt2) | `:1053,:1111,:1163` | ✅ | `br_ids[3]` from `alts[greedy]` `:1399-1435` | precomp-alts path. |
| Look-ahead depth = **2 tokens** (first + one second); NO 3-deep look | `:1068` etc. | ✅ | single second at `after` `:1475-1483` | Both are depth-2. No 3-deep. |
| Each first-token gets ONE optional b-variant (score-b) | `:1088,:1137,:1189` | ✅ | per-branch score-b block `:1695-1789` | |
| **Tie-break = source order via `switch maxScore`**: score1 ▸ score2 ▸ score3 ▸ score1b ▸ score2b ▸ score3b (first matching case wins) | `:1217-1262` | ⚠️ | `score > best_score` (strict) walked greedy→alt1→alt2, score-b inline per branch, then path-b (7th) `:1801`, path-a (8th) `:1851` | **Two divergences:** (1) ztok interleaves score1b BEFORE score2 (b-loop does score1+score1b at b=0, then score2 at b=1). On a `score1b == score2` tie, TM picks score2 (listed first); ztok keeps score1b (strict `>` blocks score2). (2) ztok adds lilbuf path-b/path-a as 7th/8th branches with NO TM analog in the switch. |
| score1/2/3/1b/2b/3b all initialised to `-1000000`; if maxScore stays `-1000000`, emit greedy plain | `:1059-1065,:1218-1219` | ✅ | `best_score=minInt`, fall-through emits seeded greedy `:1363,:1350-1362` | |
| `case score1`: emit `original.id`, advance `length`, `goto checkpoint` | `:1220-1226` | ✅ | `.normal` emit `:1916-1921` | But ztok does NOT loop-goto; it advances `i` and re-enters the while. Equivalent. |

---

## Phase 5 — forward_delete & goto-checkpoint state machine

| TM-Go behavior | TM-Go file:line | Status | ztok file:line | Notes |
|---|---|---|---|---|
| `case score1b`: emit `[original.id, deleteToken]`, `forwardDelete=1`, set `length=length1b`,`index=index1b`, `goto checkpoint` | `:1241-1247` | ⚠️ | path-a (`del_before_first`) does NOT set forward_delete (`winner_sets_fd=false` `:1894`) | ztok's path-a is `[DEL, first]` and explicitly does NOT carry forwardDelete (comment `:1991-1995` admits "residual gap"). TM's score1b is `[first, DEL]` then forwardDelete=1. **Order + state mismatch.** |
| `case score2b/3b`: emit `[id1/id2, deleteToken]`, `forwardDelete=1`, `goto checkpoint` | `:1248-1261` | ✅ (when `use_goto_checkpoint`) | `.first_del_second` + `next_forward_lilbuf` `:1937-1970,:1996` | goto path emits `[first, DEL]`, seeds next iter. Only active when lilbuf+score2b3b+goto all on (`:1027`). |
| `forwardDelete` cleared after every non-b branch | `:1225,:1232,:1239,:1267,:1275` | ✅ | `forward_delete = if (winner_sets_fd) 1 else 0` `:1996` | |
| goto-checkpoint re-evaluates the lilbuf token's OWN alts at the new position | `:1226` (loop back to `checkpoint:`) | ⚠️ | seeded iter `:1228-1253` injects seed as single candidate; alts via precomp `:1402` | Seed is treated as greedy with `len = real+1`; alt re-eval depends on `alts[seed_id]`. Plausible but only covers the score2b/3b case, not score1b (path-a). |
| `goto checkpoint` keeps SAME `i` baseline, recomputes `original=info[index].alt` | `:1051-1053` | ✅ | re-enter while-loop, `collectPrefixMatches(chunk[i..])` | Equivalent: re-derives candidates+alts at advanced `i`. |

---

## Phase 6 — Alt computation (`tokenData.alt`)

| TM-Go behavior | TM-Go file:line | Status | ztok file:line | Notes |
|---|---|---|---|---|
| Iterate `length = len(token)-1 … minAltSize` | `:3597` | ✅ | `while (length > min_alt_size) length-=1` `:3606-3608` | |
| `minAltSize`: ` `+alphanum start ⇒ 2, else 1; reset to 1 if `nWords<=1` | `:3527,:3581-3583` | ✅ | `computeMinAltSize` `:3559-3573` | Uses `computeNwordsScore<=1`. |
| Priority-10: `token[length]==' '` & next is letter/number | `:3602-3621` | ✅ | `altPriority` `:3497-3504` | |
| Priority-9 (capcode==0 only): non-letter|letter, non-number|number | `:3627-3647` | ✅ | `:3516-3525` gated `capcode==.none` | |
| Priority-9: letter|non-letter, number|non-number (`_` = letter) | `:3649-3668` | ✅ | `:3529-3535` | |
| Priority-7: space|non-space | `:3669-3684` | ✅ | `:3537-3539` | |
| Priority-8: non-space|space | `:3685-3700` | ✅ | `:3541-3543` | |
| Priority-9: everything|capcode (`isCapcode(r2)`) | `:3701-3717` | ✅ | `:3545-3547` | |
| **Priority-8 suffix** (`length==hasSuffix`) with loop `break` | `:3719-3735` | ❌ (deferred) | `hasSuffixPos` ported `:3458-3474` but **not wired into `computeAlts`** | TM evaluates the suffix rule BETWEEN the switch ladder and the priority-1 fallback, and `break`s the length loop on match. ztok's `computeAlts` never calls `hasSuffixPos`; the priority-1 fallback `:3553` runs instead, and there is no `break`. Inert for `'s`/`’s` per ztok's own note (letter|non-letter priority-9 fires first), but the missing `break` can change which longer subtoken wins in slot assignment. Wave-3D explicitly DEFERRED this. |
| Priority-1 fallback "everything else" | `:3737-3750` | ✅ | `:3548-3553` (`is_fallback`) | |
| Promotion: `priority1<priority2 ‖ (eq & len1<=len2)` chooses slot | `:3606-3617,:3632-3644,…` | ✅ | `target_slot1` `:3633-3647` | |
| Slot swap so slot1 is better: `len2>0 & (p2>p1 ‖ (eq & len2>len1))` | `:3760-3764` | ✅ | `:3650-3661` | |
| `id1=info[index].id`, `id2=info[index2].id` precomputed | `:3766-3772` | ✅ (implicit) | ztok stores `index`/`index2` as ids directly via `trieExactLookup`→token_id `:3614,:3637` | ztok stores token ids, not info-indices. Equivalent end state. |

---

## Phase 7 — Per-piece flag bits

| TM-Go bit | TM-Go file:line | Status | ztok file:line | Notes |
|---|---|---|---|---|
| `flag=4` (begins space) + nWords++/minAlt=2 when ` `+alphanum | `:3522-3528` | ✅ | `:3083-3085` | |
| `flag=2` (begins letter) | `:3529-3531` | ✅ | `:3086-3087` | |
| capcode CharacterToken/WordToken ⇒ `flag=4`; `flag|=16` begins-capcode | `:3532-3537` | ⚠️ | `:3088-3097` | ztok sets `FLAG_BEGINS_SPACE` only for `.full` C/W; for `.nocapcode` DEL (`\x7F`) it sets just `FLAG_BEGINS_CAPCODE`. TM-Go nocapcode `\x7F` is `NoCapcodeDeleteToken` — not Character/Word, so no `4` bit. **Matches** for nocapcode; for full-capcode the `D` (DeleteToken) also gets no space bit — consistent with TM. OK but worth a targeted test. |
| `onlyLetterSpace`/`onlyNumberSpace`/`onlyPunc` rune-loop | `:3543-3572` | ✅ | `:3100-3135` | |
| `nWords++` inside rune loop | `:3554-3560` | ✅ (separate fn) | `computeNwordsScore` `:3017-3043` | ztok counts words in a separate pass, not inside `computeFlags`. Same result. |
| `flag|=32` SINGLE_WORD: `minAlt==2 & isLetter(last) & onlyLetterSpace & nWords==1` | `:3576-3580` | ✅ (computed) / ❌ (used) | `:3144-3160` | Computed correctly but **never consumed by the encoder** (see Phase 2). |
| `flag|=8` ends-capcode | `:3584-3586` | ✅ | `:3162` | |
| `flag|=1` ends-letter | `:3588-3590` | ✅ | `:3163` | |
| `flag|=128` all-letters/all-punct | `:3591-3593` | ✅ | `:3164` | |

---

## Phase 8 — beginByte 256-table

| TM-Go behavior | TM-Go file:line | Status | ztok file:line | Notes |
|---|---|---|---|---|
| 4-way per-byte tally [space,letter,number,punct/capcode] | `:3522-3542` | ✅ | `bb_tally` `:652,:702-712` | |
| Majority vote, strict `>` others AND `> 2`, else 0 | `:3780-3788` | ✅ | `:798-810` | letter→1, space→4+8, punct→2+8. |
| `\x7F X` / `X` twin double-tally | `:3522,:3779` | ✅ (heuristic) | `:739-764` bare-form re-tally + alias tally `:776-793` | Wave-3D vocab-collapse compensation. ⚠️ heuristic double-counts twinned ids (comment `:723-738` acknowledges); claims it cancels in the vote. |

---

## Phase 9 — capcode rune classifiers

| TM-Go behavior | TM-Go file:line | Status | ztok file:line | Notes |
|---|---|---|---|---|
| `isLetter`: unicode.IsLetter & (capcode!=2 ‖ not C/W/D) + Mn/Mc/Me | `:359-361` | ✅ | `isLetterCp` (referenced throughout) | combining-mark handling assumed via `unicode_props`. |
| `isAlphaNum` | `:363-365` | ✅ | `isAlphaNumCp` | |
| `isCapcode`: (cap1 & `\x7F`) ‖ (cap2 & C/W/D) | `:367-369` | ✅ | `isCapcodeMarker` | |
| `hasSuffixPos` decode-last-rune-before-suffix is-letter | `:287-299` | ✅ | `:3458-3474` | Faithful port (but unused — Phase 6). |

---

## Prioritized top-5 gaps to fix next

1. **score-b length off-by-one (`> length2+1` vs `> length2`)** — `monster.zig:1750`
   vs TM `:1092/:1141/:1193`. ztok fires score-b (DEL-insert) when lilbuf covers
   only **1** extra byte; TM requires **≥2**. Cheapest high-confidence fix; directly
   changes which positions emit a DEL token. **Most likely single-line win.**

2. **Tie-break ordering: score1b beats score2 in ztok, score2 beats score1b in TM**
   — `monster.zig:1650/1773` (strict `>` + interleaved b-loop) vs TM `switch` `:1217-1262`.
   To match TM, all of score1/2/3 must be compared before any score1b/2b/3b, with
   first-source-order winning ties. Affects every position where a greedy-b ties a
   plain alt.

3. **Missing `flag&32` single-whole-word skip-gate** — TM `:1057`; ztok has no
   equivalent (`FLAG_SINGLE_WORD` computed at `:3159`, never read). When greedy is a
   lone space-led all-letter word followed by a space, TM emits greedy unconditionally;
   ztok scores alts and can pick differently. Big surface on prose (English).

4. **Priority-8 ungreedy-suffix rule + loop `break` not wired into `computeAlts`**
   — TM `:3719-3735`; ztok `hasSuffixPos` exists `:3458-3474` but `computeAlts`
   (`:3607-3648`) never calls it and never `break`s. Even if inert for `'s`/`’s`, the
   absent `break` lets a longer lower-priority subtoken overwrite a slot TM would have
   frozen. Deferred by Wave-3D; revisit.

5. **score-b gate uses ASCII-letter test, not TM's `flag&2`/`beginByte==1`** —
   `monster.zig:1738-1741` vs TM `:1088`. Non-ASCII letters (Greek/Cyrillic/accented)
   at the look-ahead fail ztok's `isAsciiLetter` gate but pass TM's `nextByte==1`.
   Also `nwords_tm` strip vs TM's raw `second.nWords==0`. Hits multilingual/full-capcode
   corpora.

---

## Decision-point tally

Counting the rows above (excluding pure helper rows): **~46 decision points**.
- ✅ matches: **34**
- ⚠️ partial / suspect: **8** (forwardDelete-shrink, end-of-input guard, score-b gate
  shape, score-b length off-by-one, tie-break order, goto-checkpoint coverage,
  CharacterToken flag, beginByte twin heuristic)
- ❌ missing: **4** (`flag&32` skip-gate, priority-8 suffix wiring, suffix loop-`break`,
  score1b `[first,DEL]`+forwardDelete via path-a)

---

## Hypotheses

**nocapcode 80→100 gap (single best hypothesis):** the **tie-break ordering
divergence (gap #2)** combined with the **missing `flag&32` skip-gate (gap #3)**.
On English prose almost every position is a space-led whole word where TM's
`:1057` gate emits greedy with zero scoring, but ztok runs the full branch loop and
can be pulled onto a score1b/path-a/path-b alternative that ties or marginally beats
greedy under ztok's strict-`>` semantics. The skip-gate is exactly the construct TM
uses to force greedy on the common case; its absence is the most plausible source of
the residual ~15-20 point band on plain-letter vocabs.

**full-capcode 56→77 gap (single best hypothesis):** the **score-b gate/length
mismatch (gaps #1 + #5)**. Full-capcode vocabs lean heavily on the DEL-insert
(score2b/3b) machinery to re-enter mid-word, so the off-by-one `> length2` (firing DEL
one byte too eagerly) and the ASCII-only look-ahead gate (missing non-ASCII and
capcode-marker look-aheads that TM's `beginByte`/`flag&2` accepts) compound on exactly
the path that capcode mode exercises most. The `forward_delete`-not-set path-a branch
(gap #5/Phase-5) further desyncs the post-DEL `branchLength`/`nWords` bookkeeping that
capcode relies on.

---

## Process confirmation

- Wrote **`bench/TM_AUDIT.md`** (this file). No other file created.
- Did **NOT** edit `src/monster.zig` or any other source.
- Did **NOT** run `zig build`, tests, benchmarks, or sweeps.
