// ztok demo — app.js
//
// Vanilla ES module. Wires the UI to the ztok-wasm.js loader.
// Three live displays share the same encoded ids:
//   1. The playground textarea overlay (colored token spans)
//   2. The sidebar id list (clickable, hoverable)
//   3. The hero stats ticker
// Comparison mode encodes the same text with a second pipeline.

import { loadZtok, hasSimd128, ZtokError } from "./vendor/ztok-wasm.js";

// -------------------------- constants ------------------------------

const WASM_URL = hasSimd128()
  ? "./vendor/ztok-wasm.wasm"
  : "./vendor/ztok-wasm-scalar.wasm";

// Bundled vocabs. `format: "tiktoken"` means the binding can load it.
// `format: "hf-json"` is included so we can surface a friendly UI
// message ("this format isn't supported by the wasm binding yet")
// rather than silently dropping the option.
const BUILTIN_VOCABS = [
  {
    id: "cl100k_base",
    label: "cl100k_base (GPT-4, GPT-3.5)",
    url: "./vocabs/cl100k_base.tiktoken",
    format: "tiktoken",
  },
  {
    id: "gpt2",
    label: "gpt2 (HuggingFace JSON) — preview",
    url: "./vocabs/gpt2.json",
    format: "hf-json",
  },
  {
    id: "llama2",
    label: "llama2 (SentencePiece) — not yet supported",
    url: null,
    format: "spm",
  },
];

// In-memory cache: vocab id -> Pipeline | null
const pipelineCache = new Map();
const vocabBytesCache = new Map();

// -------------------------- dom refs -------------------------------

const $ = (id) => document.getElementById(id);

const els = {
  pill: $("status-pill"),
  pillLabel: $("status-label"),
  ver: $("ver-string"),
  tokenizerSelect: $("tokenizer-select"),
  customVocab: $("custom-vocab"),
  vocabStatus: $("vocab-status"),
  input: $("input"),
  overlay: $("overlay"),
  mTokens: $("m-tokens"),
  mBytes: $("m-bytes"),
  mRatio: $("m-ratio"),
  mTime: $("m-time"),
  idList: $("id-list"),
  hoverInfo: $("hover-info"),
  heroTokens: $("hero-tokens"),
  heroBytes: $("hero-bytes"),
  heroRatio: $("hero-ratio"),
  heroTime: $("hero-time"),
  ticker: $("ticker"),
  compareToggle: $("compare-toggle"),
  comparePickers: document.querySelector(".compare-pickers"),
  compareGrid: $("compare-grid"),
  compareHint: $("compare-hint"),
  compareA: $("compare-a"),
  compareB: $("compare-b"),
  compareAName: $("compare-a-name"),
  compareBName: $("compare-b-name"),
  compareAOverlay: $("compare-a-overlay"),
  compareBOverlay: $("compare-b-overlay"),
  compareAMeta: $("compare-a-meta"),
  compareBMeta: $("compare-b-meta"),
  pricePreset: $("price-preset"),
  pricePerMillion: $("price-per-million"),
  reqsPerDay: $("reqs-per-day"),
  costPerReq: $("cost-per-req"),
  costPerDay: $("cost-per-day"),
  costPerMonth: $("cost-per-month"),
};

// -------------------------- state ---------------------------------

let ztok = null;
let activePipelineId = null;
let activePipeline = null;
let lastIds = new Uint32Array(0);
let lastText = "";

// Per-id byte spans into `lastText` — populated by encodeAndRender via a
// length-incremental decode that does not call into wasm per token.
let lastSpans = []; // [{start, end}]

// -------------------------- status pill ---------------------------

function setStatus(state, label) {
  els.pill.dataset.state = state;
  els.pillLabel.textContent = label;
}

// -------------------------- vocab loading -------------------------

function populateVocabSelect(select, includeCustom = true) {
  select.innerHTML = "";
  for (const v of BUILTIN_VOCABS) {
    const opt = document.createElement("option");
    opt.value = v.id;
    opt.textContent = v.label;
    if (v.format !== "tiktoken") opt.disabled = true;
    select.appendChild(opt);
  }
  if (includeCustom) {
    const opt = document.createElement("option");
    opt.value = "__custom__";
    opt.textContent = "— custom upload —";
    opt.disabled = true;
    opt.id = `${select.id}-custom-opt`;
    select.appendChild(opt);
  }
}

async function fetchVocabBytes(spec) {
  if (vocabBytesCache.has(spec.id)) return vocabBytesCache.get(spec.id);
  const resp = await fetch(spec.url);
  if (!resp.ok) throw new ZtokError(`fetch ${spec.url}: ${resp.status}`, -1);
  const bytes = new Uint8Array(await resp.arrayBuffer());
  vocabBytesCache.set(spec.id, bytes);
  return bytes;
}

async function getPipeline(specOrId) {
  const spec = typeof specOrId === "string"
    ? BUILTIN_VOCABS.find((v) => v.id === specOrId)
    : specOrId;
  if (!spec) throw new ZtokError("unknown vocab", -1);
  if (pipelineCache.has(spec.id)) return pipelineCache.get(spec.id);
  if (spec.format !== "tiktoken") {
    throw new ZtokError(
      `${spec.label} uses a vocab format the WASM binding doesn't load yet (${spec.format}). Try cl100k_base.`,
      -1,
    );
  }
  const bytes = await fetchVocabBytes(spec);
  const t0 = performance.now();
  const pipe = ztok.pipelineFromTiktokenBytes(bytes);
  pipe._loadMs = performance.now() - t0;
  pipe._vocabBytes = bytes.length;
  pipe._label = spec.label;
  pipelineCache.set(spec.id, pipe);
  return pipe;
}

async function setActiveVocab(id) {
  els.vocabStatus.textContent = `loading ${id}…`;
  els.vocabStatus.className = "vocab-status";
  try {
    const pipe = await getPipeline(id);
    activePipelineId = id;
    activePipeline = pipe;
    const kb = (pipe._vocabBytes / 1024).toFixed(0);
    els.vocabStatus.textContent = `loaded ${id} (${kb} KB) in ${pipe._loadMs.toFixed(1)} ms`;
    els.vocabStatus.className = "vocab-status ok";
    encodeAndRender();
    refreshCompare();
  } catch (err) {
    els.vocabStatus.textContent = err.message;
    els.vocabStatus.className = "vocab-status err";
  }
}

async function loadCustomFile(file) {
  els.vocabStatus.textContent = `loading ${file.name}…`;
  els.vocabStatus.className = "vocab-status";
  try {
    const bytes = new Uint8Array(await file.arrayBuffer());
    const t0 = performance.now();
    const pipe = ztok.pipelineFromTiktokenBytes(bytes);
    pipe._loadMs = performance.now() - t0;
    pipe._vocabBytes = bytes.length;
    pipe._label = `${file.name} (uploaded)`;
    const id = `__upload__:${file.name}`;
    // Replace any earlier upload by this name. Free its pipeline first.
    const prior = pipelineCache.get(id);
    if (prior) prior.free();
    pipelineCache.set(id, pipe);

    // Mirror the upload as a selectable entry in all dropdowns.
    upsertCustomOption(els.tokenizerSelect, id, file.name);
    upsertCustomOption(els.compareA, id, file.name);
    upsertCustomOption(els.compareB, id, file.name);
    els.tokenizerSelect.value = id;

    activePipelineId = id;
    activePipeline = pipe;
    const kb = (bytes.length / 1024).toFixed(0);
    els.vocabStatus.textContent = `loaded ${file.name} (${kb} KB) in ${pipe._loadMs.toFixed(1)} ms`;
    els.vocabStatus.className = "vocab-status ok";
    encodeAndRender();
    refreshCompare();
  } catch (err) {
    els.vocabStatus.textContent = `failed: ${err.message}`;
    els.vocabStatus.className = "vocab-status err";
  }
}

function upsertCustomOption(select, id, name) {
  let opt = select.querySelector(`option[value="${CSS.escape(id)}"]`);
  if (!opt) {
    opt = document.createElement("option");
    opt.value = id;
    select.appendChild(opt);
  }
  opt.textContent = `${name} (uploaded)`;
}

// -------------------- encode + render ------------------------------

// Use a generic id->bytes resolver per pipeline. We decode each unique
// id once and cache the resulting bytes; on a 20-char paragraph that's
// ~10 wasm calls instead of one per token per keystroke.
const idBytesCache = new WeakMap(); // pipeline -> Map<id, Uint8Array>

function bytesForId(pipe, id) {
  let cache = idBytesCache.get(pipe);
  if (!cache) { cache = new Map(); idBytesCache.set(pipe, cache); }
  let bytes = cache.get(id);
  if (bytes === undefined) {
    // `decodeOne` returns a string, but we want bytes — re-encode the
    // returned string with TextEncoder. This isn't a roundtrip of the
    // original bytes for non-utf8 ids, but for cl100k all leaf bytes
    // get merged into utf8 strings.
    const s = pipe.decodeOne(id);
    bytes = new TextEncoder().encode(s);
    cache.set(id, bytes);
  }
  return bytes;
}

// Walk ids, emit {start, end} byte spans into the input. Works for
// cl100k because the BPE is byte-level: concatenated leaf bytes
// reproduce the input bytes exactly.
function computeSpans(pipe, text, ids) {
  const inputBytes = new TextEncoder().encode(text);
  const spans = [];
  let cursor = 0;
  for (const id of ids) {
    const tb = bytesForId(pipe, id);
    const start = cursor;
    let end = cursor + tb.length;
    if (end > inputBytes.length) end = inputBytes.length;
    spans.push({ start, end });
    cursor = end;
  }
  return { spans, inputBytes };
}

// Convert byte-offset spans into utf16-codeunit spans for slicing the
// JS string in the overlay. Walks the input string + a parallel byte
// counter computed via TextEncoder.encodeInto on a scratch buffer.
function bytesToCharSpans(text, byteSpans) {
  const out = [];
  let charIdx = 0;
  let byteIdx = 0;
  // Greedy: for each span end, advance charIdx until the byte cursor reaches it.
  const enc = new TextEncoder();
  // Precompute a parallel array: char[i] -> byte position at start of char[i].
  const charStart = new Uint32Array(text.length + 1);
  for (let i = 0; i < text.length; i++) {
    charStart[i] = byteIdx;
    // Surrogate pair handling
    const code = text.charCodeAt(i);
    let cp;
    if (code >= 0xd800 && code <= 0xdbff && i + 1 < text.length) {
      cp = text.codePointAt(i);
      byteIdx += enc.encode(String.fromCodePoint(cp)).length;
      // mark the trailing surrogate too
      charStart[i + 1] = charStart[i]; // both halves share start
    } else {
      byteIdx += enc.encode(text[i]).length;
    }
  }
  charStart[text.length] = byteIdx;

  // For each byte span, find char index whose start == span.start (or just past).
  // Linear scan, monotonic.
  let ci = 0;
  for (const sp of byteSpans) {
    while (ci < text.length && charStart[ci] < sp.start) ci++;
    const cStart = ci;
    while (ci < text.length && charStart[ci] < sp.end) ci++;
    out.push({ cStart, cEnd: ci });
  }
  return out;
}

function renderOverlay(text, spans, ids, targetEl) {
  if (spans.length === 0) {
    targetEl.textContent = text || " ";
    return;
  }
  const charSpans = bytesToCharSpans(text, spans);
  const frag = document.createDocumentFragment();
  let written = 0;
  for (let i = 0; i < charSpans.length; i++) {
    const { cStart, cEnd } = charSpans[i];
    if (cStart > written) {
      frag.appendChild(document.createTextNode(text.slice(written, cStart)));
    }
    const slice = text.slice(cStart, cEnd);
    if (slice.length === 0) continue;
    const span = document.createElement("span");
    span.className = "tk";
    span.dataset.h = (i & 7).toString();
    span.dataset.idx = i.toString();
    span.dataset.id = ids[i].toString();
    span.textContent = slice;
    frag.appendChild(span);
    written = cEnd;
  }
  if (written < text.length) {
    frag.appendChild(document.createTextNode(text.slice(written)));
  }
  // Trailing newline for textarea baseline alignment.
  frag.appendChild(document.createTextNode("\n"));
  targetEl.textContent = "";
  targetEl.appendChild(frag);
}

function renderIdList(ids) {
  const html = Array.from(ids).map((id, i) =>
    `<span class="id" data-h="${i & 7}" data-idx="${i}" data-id="${id}" title="id ${id}">${id}</span>`
  ).join("");
  els.idList.innerHTML = html;
}

function updateMetrics(text, ids, encodeMs) {
  const bytes = new TextEncoder().encode(text).length;
  const ratio = ids.length === 0 ? 0 : bytes / ids.length;
  els.mTokens.textContent = ids.length.toString();
  els.mBytes.textContent = bytes.toString();
  els.mRatio.textContent = ratio === 0 ? "—" : ratio.toFixed(2);
  els.mTime.textContent = `${encodeMs.toFixed(2)} ms`;

  els.heroTokens.textContent = ids.length.toString();
  els.heroBytes.textContent = bytes.toString();
  els.heroRatio.textContent = ratio === 0 ? "—" : ratio.toFixed(2);
  els.heroTime.textContent = `${encodeMs.toFixed(2)} ms`;

  updateCost();
}

function encodeAndRender() {
  if (!activePipeline) return;
  const text = els.input.value;
  lastText = text;

  if (text.length === 0) {
    lastIds = new Uint32Array(0);
    lastSpans = [];
    els.overlay.textContent = " ";
    els.idList.innerHTML = "";
    updateMetrics("", lastIds, 0);
    return;
  }

  const t0 = performance.now();
  let ids;
  try {
    ids = activePipeline.encode(text);
  } catch (err) {
    els.overlay.textContent = `ERROR: ${err.message}`;
    return;
  }
  const ms = performance.now() - t0;
  lastIds = ids;

  const { spans } = computeSpans(activePipeline, text, ids);
  lastSpans = spans;
  renderOverlay(text, spans, ids, els.overlay);
  renderIdList(ids);
  updateMetrics(text, ids, ms);
  tickHero(text, ids);
}

// -------------------- hover wiring ---------------------------------

function wireHover() {
  function highlight(idx, id, src) {
    document.querySelectorAll(".tk.hovered, .id.hovered").forEach((n) =>
      n.classList.remove("hovered")
    );
    if (idx == null) {
      els.hoverInfo.textContent = "hover a token to inspect";
      els.hoverInfo.classList.remove("active");
      return;
    }
    document.querySelectorAll(
      `.tk[data-idx="${idx}"], .id[data-idx="${idx}"]`,
    ).forEach((n) => n.classList.add("hovered"));
    let bytesStr = "";
    try {
      const decoded = activePipeline.decodeOne(parseInt(id, 10));
      // Show printable; for control bytes, show \x escapes.
      bytesStr = decoded.replace(/[\x00-\x1f\x7f]/g, (c) =>
        `\\x${c.charCodeAt(0).toString(16).padStart(2, "0")}`,
      ).replace(/ /g, "·");
    } catch { bytesStr = "(decode failed)"; }
    els.hoverInfo.innerHTML =
      `<strong>id ${id}</strong> · pos ${idx} · ` +
      `<code>${escapeHtml(bytesStr)}</code>`;
    els.hoverInfo.classList.add("active");
  }

  function handle(e) {
    const t = e.target.closest(".tk, .id");
    if (!t) return;
    highlight(t.dataset.idx, t.dataset.id);
  }
  function leave(e) {
    // Only clear if leaving the playground entirely.
    if (e.target.closest("#playground")) highlight(null);
  }
  document.querySelector("#playground").addEventListener("mouseover", handle);
  document.querySelector("#playground").addEventListener("mouseleave", () => highlight(null));
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]
  );
}

// -------------------- hero ticker ---------------------------------

function tickHero(text, ids) {
  // Render the last ~10 tokens as a stack-effect ticker for visual interest.
  const tail = Array.from(ids).slice(-12);
  if (tail.length === 0) {
    els.ticker.innerHTML = '<span class="tk">type something →</span>';
    return;
  }
  const html = tail.map((id, i) => {
    let s;
    try {
      s = activePipeline.decodeOne(id).replace(/\n/g, "\\n");
    } catch { s = `[${id}]`; }
    if (s.length > 12) s = s.slice(0, 12) + "…";
    return `<span class="tk" style="background:var(--tk-${i & 7});opacity:${0.4 + (i / tail.length) * 0.6}">${escapeHtml(s) || " "}</span>`;
  }).join("");
  els.ticker.innerHTML = html;
}

// -------------------- compare mode --------------------------------

function refreshCompare() {
  if (!els.compareToggle.checked) return;
  const aId = els.compareA.value;
  const bId = els.compareB.value;
  if (aId) renderComparePane("a", aId);
  if (bId) renderComparePane("b", bId);
}

async function renderComparePane(side, id) {
  const overlay = side === "a" ? els.compareAOverlay : els.compareBOverlay;
  const meta = side === "a" ? els.compareAMeta : els.compareBMeta;
  const nameEl = side === "a" ? els.compareAName : els.compareBName;
  const text = els.input.value;

  let pipe;
  if (id.startsWith("__upload__:")) {
    pipe = pipelineCache.get(id);
    if (!pipe) { overlay.textContent = "(upload no longer available)"; return; }
  } else {
    try { pipe = await getPipeline(id); }
    catch (err) { overlay.textContent = err.message; meta.textContent = ""; return; }
  }

  nameEl.textContent = pipe._label || id;
  const t0 = performance.now();
  const ids = pipe.encode(text);
  const ms = performance.now() - t0;
  const { spans } = computeSpans(pipe, text, ids);
  renderOverlay(text, spans, ids, overlay);
  const bytes = new TextEncoder().encode(text).length;
  meta.textContent = `${ids.length} tokens · ${bytes} B · ${(bytes / Math.max(1, ids.length)).toFixed(2)} B/tok · ${ms.toFixed(2)} ms`;
}

function wireCompare() {
  els.compareToggle.addEventListener("change", () => {
    const on = els.compareToggle.checked;
    els.comparePickers.hidden = !on;
    els.compareGrid.hidden = !on;
    els.compareHint.hidden = !on;
    if (on) {
      // Default A to active vocab, B to a different one if possible.
      els.compareA.value = activePipelineId || "cl100k_base";
      const others = Array.from(els.compareB.options)
        .filter((o) => !o.disabled && o.value !== els.compareA.value);
      els.compareB.value = others[0]?.value || els.compareA.value;
      refreshCompare();
    }
  });
  els.compareA.addEventListener("change", refreshCompare);
  els.compareB.addEventListener("change", refreshCompare);
}

// -------------------- cost estimator -------------------------------

function updateCost() {
  const pricePerMillion = parseFloat(els.pricePerMillion.value) || 0;
  const reqsPerDay = parseFloat(els.reqsPerDay.value) || 0;
  const tokens = lastIds.length;
  const perReq = (tokens / 1_000_000) * pricePerMillion;
  const perDay = perReq * reqsPerDay;
  const perMonth = perDay * 30;
  const fmt = (n) =>
    n < 0.01 && n > 0
      ? `$${n.toExponential(2)}`
      : `$${n.toLocaleString("en-US", { minimumFractionDigits: 4, maximumFractionDigits: 4 })}`;
  els.costPerReq.textContent = fmt(perReq);
  els.costPerDay.textContent = fmt(perDay);
  els.costPerMonth.textContent = fmt(perMonth);
}

function wireCost() {
  els.pricePreset.addEventListener("change", () => {
    const val = els.pricePreset.value;
    if (val === "custom") {
      els.pricePerMillion.focus();
      return;
    }
    // Per-1K price → per-1M: ×1000.
    const perK = parseFloat(val);
    els.pricePerMillion.value = (perK * 1000).toFixed(2);
    updateCost();
  });
  els.pricePerMillion.addEventListener("input", updateCost);
  els.reqsPerDay.addEventListener("input", updateCost);
}

// -------------------- init ----------------------------------------

function debounce(fn, ms) {
  let t = 0;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

async function init() {
  setStatus("loading", "loading wasm");

  populateVocabSelect(els.tokenizerSelect);
  populateVocabSelect(els.compareA, false);
  populateVocabSelect(els.compareB, false);
  els.tokenizerSelect.value = "cl100k_base";
  els.compareA.value = "cl100k_base";
  els.compareB.value = "cl100k_base";

  try {
    ztok = await loadZtok({ wasmUrl: WASM_URL });
    els.ver.textContent = `${ztok.version} · WASM ${hasSimd128() ? "SIMD128" : "scalar"}`;
    setStatus("ready", `wasm ${ztok.version}`);
  } catch (err) {
    setStatus("error", err.message);
    els.vocabStatus.textContent = `wasm load failed: ${err.message}`;
    els.vocabStatus.className = "vocab-status err";
    return;
  }

  await setActiveVocab("cl100k_base");

  // Initial price preset → fill input.
  els.pricePerMillion.value = (parseFloat(els.pricePreset.value) * 1000).toFixed(2);
  updateCost();

  els.input.addEventListener("input", debounce(() => {
    encodeAndRender();
    refreshCompare();
  }, 80));

  els.tokenizerSelect.addEventListener("change", (e) => {
    const v = e.target.value;
    if (!v || v === "__custom__") return;
    if (v.startsWith("__upload__:")) {
      activePipelineId = v;
      activePipeline = pipelineCache.get(v);
      els.vocabStatus.textContent = `using ${activePipeline?._label || v}`;
      els.vocabStatus.className = "vocab-status ok";
      encodeAndRender();
      refreshCompare();
      return;
    }
    setActiveVocab(v);
  });

  els.customVocab.addEventListener("change", (e) => {
    const f = e.target.files?.[0];
    if (f) loadCustomFile(f);
  });

  wireHover();
  wireCompare();
  wireCost();
}

init();
