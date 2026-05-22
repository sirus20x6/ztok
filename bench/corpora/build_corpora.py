#!/usr/bin/env python3
"""Build the diverse corpus pack used by the 10K-line equivalence stress
gate. Idempotent: each corpus is regenerated only when its target file
is missing, so re-runs are cheap.

The pack vendors five files under bench/corpora/:

  english.txt        10K lines mixed Gutenberg public-domain prose
  code.txt           10K lines code (Python/JS/Go/Rust/Zig samples)
  multilingual.txt   10K lines across 8 scripts (es/fr/de/zh/ja/ru/ar/hi)
  chat.txt           10K lines conversational text (Q&A shaped)
  unicode_stress.txt 1K  lines adversarial Unicode

All sources are public-domain (Gutenberg) or generated synthetically
from permissively-licensed templates. Total vendored size stays under
20 MB; this script aims for ~2-3 MB per file.

Usage:
  bench/corpora/build_corpora.py            # build everything (default)
  bench/corpora/build_corpora.py --force    # rebuild even if present
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import random
import re
import sys
import textwrap
import unicodedata
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------- utils

def write(path: str, lines: list[str]) -> None:
    """Write one line per element, normalizing newlines / trimming
    runaway whitespace. Truncate any line longer than 8 KB so the file
    stays readable line-by-line."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            s = line.replace("\r\n", "\n").replace("\r", "\n").split("\n", 1)[0]
            s = s.rstrip()
            if len(s.encode("utf-8")) > 8192:
                # Keep things sane for the line-oriented harness.
                s = s.encode("utf-8")[:8192].decode("utf-8", errors="ignore")
            if s:
                f.write(s + "\n")


def fetch(url: str, cache_name: str) -> bytes:
    """Cache-aware fetch. Stores raw bytes under bench/corpora/_cache/
    so repeated runs don't hammer the network."""
    cache_dir = os.path.join(HERE, "_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, cache_name)
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        with open(cache_path, "rb") as f:
            return f.read()
    print(f"  fetching {url}", file=sys.stderr)
    req = urllib.request.Request(url, headers={"User-Agent": "ztok-bench/1.21"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    with open(cache_path, "wb") as f:
        f.write(data)
    return data


def strip_gutenberg_header(text: str) -> str:
    """Drop the standard Gutenberg legal header/footer so the corpus is
    actual prose only."""
    start_marker = "*** START OF "
    end_marker = "*** END OF "
    si = text.find(start_marker)
    if si != -1:
        # Skip past the start marker line.
        nl = text.find("\n", si)
        text = text[nl + 1:] if nl != -1 else text[si:]
    ei = text.find(end_marker)
    if ei != -1:
        text = text[:ei]
    return text


# --------------------------------------------------------------------- corpora

def build_english(target: str) -> None:
    """Pick three Gutenberg works of mixed register and interleave their
    lines. Sources:
      - Shakespeare collected works (#100): early modern English verse + prose
      - Twain "Adventures of Huckleberry Finn" (#76): 19c American vernacular
      - Darwin "On the Origin of Species" (#1228): technical 19c scientific
    """
    sources = [
        ("https://www.gutenberg.org/cache/epub/100/pg100.txt", "shakespeare.txt"),
        ("https://www.gutenberg.org/cache/epub/76/pg76.txt",   "huckfinn.txt"),
        ("https://www.gutenberg.org/cache/epub/1228/pg1228.txt", "darwin.txt"),
    ]
    pools = []
    for url, name in sources:
        raw = fetch(url, name).decode("utf-8", errors="replace")
        body = strip_gutenberg_header(raw)
        # Split on blank lines into paragraphs, then to lines, drop too-short.
        lines = [ln.strip() for ln in body.splitlines() if len(ln.strip()) >= 30]
        pools.append(lines)
    rng = random.Random(0xc01d)
    # Interleave a balanced 1/3 per source until 10K.
    out: list[str] = []
    cursors = [0, 0, 0]
    src_order = [0, 1, 2]
    while len(out) < 10000:
        rng.shuffle(src_order)
        for s in src_order:
            if cursors[s] < len(pools[s]):
                out.append(pools[s][cursors[s]])
                cursors[s] += 1
                if len(out) >= 10000:
                    break
            elif all(c >= len(p) for c, p in zip(cursors, pools)):
                break
        else:
            continue
        if all(c >= len(p) for c, p in zip(cursors, pools)):
            break
    write(target, out[:10000])


def build_code(target: str) -> None:
    """10K lines of code, mixed across Python / JS / Go / Rust / Zig.
    Vendored from CPython's stdlib (Python), Node's lib (JS) plus Go +
    Rust + Zig stdlib snippets. To keep dependency-free we synthesize
    a representative pool from the working ztok repo itself plus a
    small fetched sample, since CPython repos vary.

    Strategy: walk the ztok repo and adjacent code we know exists, then
    pad with synthetic patterns covering language idioms that the repo
    might not exercise heavily."""
    pools: list[str] = []

    # Pull from ztok's own src/ + bindings/ (MIT-licensed code we already own).
    repo_root = os.path.abspath(os.path.join(HERE, "..", ".."))
    code_exts = (".zig", ".py", ".js", ".mjs", ".ts", ".go", ".rs", ".c", ".h")
    skip_dirs = {"_cache", "zig-out", ".zig-cache", "node_modules", "__pycache__",
                 "refs", "corpora"}

    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
        for fn in files:
            if not fn.endswith(code_exts):
                continue
            path = os.path.join(root, fn)
            try:
                with open(path, encoding="utf-8") as f:
                    for ln in f:
                        s = ln.rstrip()
                        if 8 <= len(s) <= 200:
                            pools.append(s)
            except (OSError, UnicodeDecodeError):
                continue

    # Add synthetic patterns to ensure each language's distinctive
    # syntax is well-represented (closures, generics, lifetimes,
    # async, decorators, error handling).
    synthetic = [
        # Python
        "@functools.lru_cache(maxsize=1024)",
        "def fibonacci(n: int) -> int:",
        "    return fibonacci(n - 1) + fibonacci(n - 2)",
        "async def fetch_user(session: aiohttp.ClientSession, uid: int) -> dict:",
        "    async with session.get(f'/users/{uid}') as resp:",
        "        return await resp.json()",
        "with open(path, 'rb') as f: data = f.read()",
        "result = [x ** 2 for x in range(100) if x % 2 == 0]",
        "raise ValueError(f'expected int, got {type(x).__name__!r}')",
        # JavaScript
        "const fn = async (req, res) => { const data = await db.query(); res.json(data); };",
        "import { useState, useEffect } from 'react';",
        "export default function App() { return <div className='app'>Hello</div>; }",
        "const sum = arr.reduce((acc, x) => acc + x, 0);",
        "try { JSON.parse(input); } catch (e) { console.error(e.message); }",
        "for (const [key, value] of Object.entries(obj)) { /* ... */ }",
        # Go
        "func (s *Server) Handle(w http.ResponseWriter, r *http.Request) {",
        "    defer r.Body.Close()",
        "    if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {",
        "        http.Error(w, err.Error(), http.StatusBadRequest); return",
        "    }",
        "}",
        "go func() { defer wg.Done(); ch <- compute(x) }()",
        "ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)",
        # Rust
        "pub fn parse<'a>(input: &'a str) -> Result<Ast<'a>, ParseError> {",
        "    let mut iter = input.chars().peekable();",
        "    while let Some(&c) = iter.peek() { /* ... */ }",
        "}",
        "let v: Vec<u32> = (0..100).filter(|x| x % 3 == 0).collect();",
        "impl<T: Clone + Send + 'static> Worker<T> { /* ... */ }",
        "match opt { Some(v) => process(v), None => return Err(Error::Missing) }",
        # Zig
        "const std = @import(\"std\");",
        "pub fn main() !void {",
        "    var gpa = std.heap.GeneralPurposeAllocator(.{}){};",
        "    defer _ = gpa.deinit();",
        "    const allocator = gpa.allocator();",
        "    const args = try std.process.argsAlloc(allocator);",
        "    defer std.process.argsFree(allocator, args);",
        "}",
        "fn fmt(comptime T: type, value: T) []const u8 {",
        "    return std.fmt.bufPrint(&buf, \"{any}\", .{value}) catch unreachable;",
        "}",
        # C
        "static inline uint32_t rotl32(uint32_t x, int r) { return (x << r) | (x >> (32 - r)); }",
        "if (fd < 0) { perror(\"open\"); return -1; }",
        "memcpy(dst, src, n); dst[n] = '\\0';",
    ]
    # Repeat enough to dominate the rare patterns; interleave with repo.
    pools.extend(synthetic * 50)

    rng = random.Random(0xc0d3)
    rng.shuffle(pools)
    write(target, pools[:10000])


def build_multilingual(target: str) -> None:
    """1.25K lines per language across 8 languages = 10K total.

    Sources: Universal Declaration of Human Rights translations are
    public-domain via the UN; we fetch from unicode.org's UDHR archive
    which has one .txt per language. We also pad with prepared
    sentences covering scripts that the UDHR may not exercise heavily
    enough.
    """
    udhr_base = "https://www.unicode.org/udhr/d/"
    sources = [
        ("spa", "udhr_spa.txt"),
        ("fra", "udhr_fra.txt"),
        ("deu_1996", "udhr_deu.txt"),
        ("cmn_hans", "udhr_zh_hans.txt"),
        ("jpn", "udhr_jpn.txt"),
        ("rus", "udhr_rus.txt"),
        ("arb", "udhr_arb.txt"),
        ("hnd", "udhr_hnd.txt"),
    ]

    pools: list[list[str]] = []
    for code, name in sources:
        url = f"{udhr_base}udhr_{code}.txt"
        try:
            raw = fetch(url, name).decode("utf-8", errors="replace")
        except Exception:
            raw = ""
        # UDHR files start with a few header lines (title, attribution)
        # then articles. Treat any non-blank line as content.
        lines = [ln.strip() for ln in raw.splitlines() if len(ln.strip()) >= 10]
        pools.append(lines)

    # Pad each language to ~1.25K with sentence repeats + small templated
    # variations so the corpus has real script coverage even when UDHR
    # is short.
    padding_templates = {
        # Spanish
        0: [
            "El rápido zorro marrón salta sobre el perro perezoso.",
            "La inteligencia artificial transforma la forma en que trabajamos.",
            "Los niños jugaban en el parque mientras llovía suavemente.",
            "¿Cuántos años tienes? Tengo treinta y cuatro años.",
            "El precio del café aumentó un 15% durante el último trimestre.",
        ],
        # French
        1: [
            "Le développement durable est un enjeu majeur du XXIe siècle.",
            "L'écrivain a publié son nouveau roman la semaine dernière.",
            "Voulez-vous prendre un café avec moi cet après-midi ?",
            "La Tour Eiffel mesure 330 mètres de hauteur.",
            "Les œufs au plat sont accompagnés de bacon et de pommes de terre.",
        ],
        # German
        2: [
            "Die schnelle braune Füchsin springt über den faulen Hund.",
            "Künstliche Intelligenz verändert die Art und Weise, wie wir arbeiten.",
            "Größere Städte wie München, Köln und Düsseldorf wachsen weiter.",
            "Ich möchte ein Glas Bier mit Würstchen und Brot bestellen.",
            "Straßenbahn Nummer 15 fährt alle zehn Minuten vom Hauptbahnhof.",
        ],
        # Chinese
        3: [
            "今天天气很好,我们去公园散步吧。",
            "人工智能正在改变我们工作和生活的方式。",
            "他在北京大学学习计算机科学已经三年了。",
            "请给我一杯咖啡和两个包子,谢谢。",
            "中华人民共和国成立于一九四九年十月一日。",
        ],
        # Japanese
        4: [
            "今日は天気がとてもいいので、公園で散歩しましょう。",
            "人工知能は私たちの働き方を変えつつあります。",
            "コーヒーを一杯と、サンドイッチを二つお願いします。",
            "東京駅から京都駅まで新幹線で約二時間半かかります。",
            "桜の花が満開で、お花見に行く人がたくさんいます。",
        ],
        # Russian
        5: [
            "Быстрая коричневая лиса прыгает через ленивую собаку.",
            "Искусственный интеллект меняет способ нашей работы и жизни.",
            "Москва — столица Российской Федерации и крупнейший её город.",
            "Я хочу заказать чашку кофе и два бутерброда, пожалуйста.",
            "Поезд отправляется с третьей платформы через десять минут.",
        ],
        # Arabic
        6: [
            "الذكاء الاصطناعي يغير طريقة عملنا وحياتنا اليومية.",
            "أرجوك أعطني فنجاناً من القهوة وقطعتين من الخبز.",
            "تقع مدينة القاهرة على ضفاف نهر النيل في شمال مصر.",
            "اللغة العربية هي اللغة الرسمية في اثنتين وعشرين دولة.",
            "ذهبت إلى السوق لشراء الفواكه والخضراوات الطازجة.",
        ],
        # Hindi
        7: [
            "तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर से कूदती है।",
            "कृत्रिम बुद्धिमत्ता हमारे काम करने के तरीके को बदल रही है।",
            "कृपया मुझे एक कप कॉफ़ी और दो रोटी दीजिए।",
            "दिल्ली भारत की राजधानी है और इसकी आबादी दो करोड़ से अधिक है।",
            "रेलगाड़ी प्लेटफ़ॉर्म नंबर तीन से दस मिनट में रवाना होगी।",
        ],
    }
    out: list[str] = []
    target_per = 10000 // len(sources)
    for idx, lang_lines in enumerate(pools):
        bucket = list(lang_lines)
        templates = padding_templates[idx]
        i = 0
        while len(bucket) < target_per:
            # Append templates with rotating numeric suffix so adjacent
            # lines aren't bit-identical (helps stress tokenizer
            # diversity).
            bucket.append(f"{templates[i % len(templates)]} (n={i})")
            i += 1
        out.extend(bucket[:target_per])
    # Shuffle so languages interleave (avoids "all lines from one lang
    # in a row" sample bias).
    rng = random.Random(0xbabe)
    rng.shuffle(out)
    write(target, out[:10000])


def build_chat(target: str) -> None:
    """10K lines of conversational text. We don't have a public-domain
    StackOverflow dump readily available; we synthesize a Q&A pool from
    common technical question/answer patterns that exercise:
      - English contractions ("don't", "can't", "I'm")
      - Code fences within prose
      - URLs and emails
      - Quote characters (single, double, smart)
      - Mixed punctuation
    Backed by a hand-curated template list multiplied into 10K via
    parameter substitution.
    """
    templates = [
        # Questions
        "Why does my {lang} script throw '{err}' when I try to {action}?",
        "How do I {action} a {dtype} in {lang} without using {lib}?",
        "What's the difference between {a} and {b} in {lang}?",
        "Can someone explain why `{snippet}` returns `{val}` instead of `{expected}`?",
        "I'm trying to {action} but I keep getting `{err}`. Any ideas?",
        "Is it possible to {action} from within a {context}? I haven't found a clean way.",
        "Has anyone successfully deployed {lib} on {platform}? I'd love a writeup.",
        # Answers
        "You can't do that because {reason}. Use {alt} instead — it's idiomatic.",
        "I'd suggest using `{snippet}` here. It handles the edge case where {edge}.",
        "Don't forget to {action} before calling {fn}, otherwise you'll get '{err}'.",
        "@user_42 — that's a classic gotcha. The reason is {reason}. See {url} for the long version.",
        "Tried this on {platform} and it works fine for me. What version of {lib} are you on?",
        "Note: as of {lib} v{ver}, the {fn} signature changed. Old code looks like `{snippet}`.",
        "+1 to the above. I'd also add: always {action} when working with {dtype}.",
        "Hmm, that's odd — when I do `{snippet}` I get `{val}`. Are you sure {context}?",
        # Casual / chat
        "lol that's exactly what happened to me last week — I spent 3 hours debugging {err}",
        "Yeah, I've been doing it like this: `{snippet}` and it works ~99% of the time",
        "TIL you can do {action} with a single line of {lang}. Mind = blown.",
        "thanks!!! that worked perfectly. one small note: in newer versions you need {ver}+",
        "ok so I tried your suggestion but now I'm getting '{err}' instead. ideas?",
    ]
    langs = ["Python", "JavaScript", "TypeScript", "Go", "Rust", "Zig", "C++", "Ruby",
             "Java", "Kotlin", "Swift", "Bash"]
    libs = ["NumPy", "pandas", "TensorFlow", "React", "Next.js", "Express", "tokio",
            "serde", "asyncio", "gRPC", "FastAPI", "Flask", "Spring", "RxJS"]
    actions = ["serialize the response", "parse the JSON", "stream the file",
               "handle the timeout", "validate the input", "mock the database",
               "set up the test fixture", "configure the retry policy",
               "iterate over the rows lazily", "compose the middleware",
               "extract the substring without copying", "chunk the iterator",
               "throttle the requests"]
    errs = ["TypeError: object is not subscriptable", "ENOENT: no such file or directory",
            "panic: runtime error: index out of range [3] with length 3",
            "borrowck: cannot borrow `*self` as mutable",
            "AttributeError: 'NoneType' object has no attribute 'split'",
            "OperationalError: database is locked", "ECONNREFUSED 127.0.0.1:5432",
            "401 Unauthorized: bad token"]
    dtypes = ["bytes buffer", "lazy iterator", "Option<&str>", "Promise<Response>",
              "std::sync::Arc", "context.Context", "Vec<u8>", "[]const u8",
              "concurrent.futures.Future"]
    snippets = [
        "list[0:10:2]", "json.dumps(obj, indent=2)", "await fetch('/api/v1')",
        ".collect::<Result<Vec<_>, _>>()", "ctx, cancel := context.WithCancel(parent)",
        "let mut v = Vec::with_capacity(n);", "@std.builtin.SourceLocation",
        "Object.assign({}, defaults, opts)", "df.groupby('id').sum()",
        "rxjs.merge(...streams).pipe(debounceTime(50))", "Box::pin(async move { ... })",
        "std.fs.cwd().readFileAlloc(a, p, 1 << 30)"]
    vals = ["None", "undefined", "[1, 2, 3]", "Err(\"...\")", "Ok(())", "''", "0", "NaN",
            "true", "false", "{}", "[]"]
    expecteds = ["[2, 4, 6]", "Some(42)", "200 OK", "Ok(vec![1, 2, 3])", "the next item",
                 "an empty slice", "the parsed object"]
    reasons = ["the borrow checker enforces uniqueness", "the GIL serializes the access",
               "the event loop hasn't drained yet", "the cache invalidation fires late",
               "the type elaboration is strict", "the comptime evaluator can't see the value"]
    alts = ["a typed dict", "Arc<Mutex<T>>", "a channel-based broker", "an async generator",
            "a lazy stream", "a comptime-known constant", "a tagged union", "an enum"]
    contexts = ["nested closure", "async callback", "trait impl", "comptime branch",
                "background goroutine", "service worker", "build script"]
    platforms = ["Linux x86_64", "macOS ARM64", "Windows WSL2", "Docker / Alpine",
                 "k8s 1.30", "Vercel", "Cloudflare Workers", "Bun 1.2"]
    fns = ["init()", "spawn(...)", "deserialize_with(...)", "render_html(...)", "select!(...)",
           "@import(\"std\")", "encode_batch(...)"]
    edges = ["the input is empty", "the path crosses a Unicode boundary", "two writers race",
             "the token contains a NUL", "the file is exactly 4096 bytes"]
    urls = ["https://docs.example.com/v1/api", "http://localhost:9123/health",
            "https://github.com/foo/bar/blob/main/README.md#install"]
    vers = ["0.16", "1.21", "2.0.4-rc1", "3.12", "18.0", "24.04 LTS"]
    a_b = [("&str", "String"), ("list", "tuple"), ("Vec", "VecDeque"), ("await", "yield"),
           ("usize", "isize"), ("?", "??"), ("error.OutOfMemory", "anyerror"),
           ("u32", "i32")]

    rng = random.Random(0xc4a7)
    out: list[str] = []
    while len(out) < 10000:
        tpl = rng.choice(templates)
        a, b = rng.choice(a_b)
        line = tpl.format(
            lang=rng.choice(langs), lib=rng.choice(libs), action=rng.choice(actions),
            err=rng.choice(errs), dtype=rng.choice(dtypes), snippet=rng.choice(snippets),
            val=rng.choice(vals), expected=rng.choice(expecteds), reason=rng.choice(reasons),
            alt=rng.choice(alts), context=rng.choice(contexts), platform=rng.choice(platforms),
            fn=rng.choice(fns), edge=rng.choice(edges), url=rng.choice(urls),
            ver=rng.choice(vers), a=a, b=b,
        )
        # Inject occasional smart quotes / em-dashes (covers normalizer edges).
        if rng.random() < 0.06:
            line = line.replace("'", "’").replace('"', "“")
        if rng.random() < 0.03:
            line = line.replace(" — ", " — ")
        out.append(line)
    write(target, out[:10000])


def build_unicode_stress(target: str) -> None:
    """1K lines of adversarial Unicode covering:
      - Combining marks (decomposed forms)
      - ZWJ emoji sequences (family, profession)
      - Bidi (RTL embedded in LTR, isolates)
      - Variation selectors (text/emoji presentation)
      - Tag sequences (subdivision flags)
      - Half/fullwidth pairs
      - Hangul jamo (decomposed Korean)
      - Mathematical alphanumerics
      - Control / format characters mixed with text
      - Long strings of combining marks (Zalgo)
      - Various whitespace (NBSP, NNBSP, ZWSP, MMSP, OGHAM)
    """
    lines: list[str] = []

    # Combining marks: ASCII letters with stacked diacritics
    bases = ["a", "e", "i", "o", "u", "n", "c"]
    combiners = ["́", "̀", "̂", "̃", "̄", "̆", "̇",
                 "̈", "̊", "̋", "̌", "̧", "̨"]
    rng = random.Random(0xCAFE)
    for _ in range(120):
        n = rng.randint(1, 5)
        word = "".join(rng.choice(bases) + "".join(rng.sample(combiners, k=rng.randint(1, 3)))
                       for _ in range(n))
        lines.append(f"diacritic word: {word} canonical")

    # ZWJ emoji sequences
    zwj_emojis = [
        "\U0001F468‍\U0001F469‍\U0001F467‍\U0001F466",  # family
        "\U0001F469‍\U0001F4BB",                                   # woman technologist
        "\U0001F468\U0001F3FF‍\U0001F33E",                         # farmer dark skin
        "\U0001F3F3️‍\U0001F308",                              # rainbow flag
        "\U0001F3F4\U000E0067\U000E0062\U000E0073\U000E0063\U000E0074\U000E007F",  # Scotland subdivision
        "\U0001F469\U0001F3FD‍⚕️",                       # woman health worker
        "\U0001F469‍\U0001F469‍\U0001F466‍\U0001F466",   # two-mom family
        "\U0001F469\U0001F3FE‍\U0001F91D‍\U0001F468\U0001F3FB",  # handshake mixed skin
    ]
    for i in range(120):
        seq = rng.choice(zwj_emojis)
        lines.append(f"emoji {i}: {seq} after {seq*2}")

    # Bidi: Arabic embedded in English
    arabic_words = ["مرحبا", "العالم", "السلام", "كتاب", "مدرسة", "جامعة"]
    hebrew_words = ["שלום", "עולם", "ספר", "בית"]
    for _ in range(80):
        e = "Welcome to the project"
        a = rng.choice(arabic_words)
        h = rng.choice(hebrew_words)
        lines.append(f"{e} — {a} & {h} — end.")
        # With isolates
        lines.append(f"{e} ⁨{a}⁩ then ⁨{h}⁩ end.")

    # Variation selectors (text vs emoji presentation)
    vs_pairs = ["☃︎", "☃️", "☕️", "❤️",
                "⚠️", "▶️"]
    for _ in range(60):
        seq = " ".join(rng.choice(vs_pairs) for _ in range(rng.randint(2, 6)))
        lines.append(f"presentation: {seq}")

    # Half/full width
    hw = "Hello123"
    fw = "".join(chr(ord(c) + 0xFEE0) if "!" <= c <= "~" else c for c in hw)
    for _ in range(40):
        lines.append(f"{hw} vs {fw} mixed")

    # Hangul jamo (decomposed)
    jamo_pairs = [
        "각",  # 각
        "한",  # 한
        "낸",  # 낸
    ]
    composed = ["각", "한", "낸", "감사합니다", "안녕하세요"]
    for _ in range(60):
        lines.append(f"jamo: {''.join(jamo_pairs)} vs {''.join(composed)}")

    # Mathematical alphanumerics
    math_a = "\U0001D400"  # 𝐀
    math_z = "\U0001D419"  # 𝐙
    math_one = "\U0001D7CE"  # 𝟎
    for i in range(40):
        line = "".join(
            chr(0x1D400 + (ord(c) - ord("A")))
            if c.isupper() else c
            for c in f"Math BOLD ABC at i={i}"
        )
        lines.append(line)

    # Zalgo (extreme combining)
    base_word = "hello"
    zalgo_combiners = [chr(c) for c in range(0x300, 0x36F)]
    for _ in range(40):
        z = ""
        for ch in base_word:
            z += ch + "".join(rng.choice(zalgo_combiners) for _ in range(rng.randint(2, 8)))
        lines.append(f"zalgo: {z}")

    # Whitespace variants (NBSP, NNBSP, ZWSP, OGHAM, etc.)
    ws_chars = [" ", " ", " ", "​", " ", "　", " ", " "]
    for _ in range(60):
        ws = rng.choice(ws_chars)
        lines.append(f"word1{ws}word2{ws}word3 (ws={hex(ord(ws))})")

    # Surrogate-pair (4-byte UTF-8) round-trip targets
    astral = ["\U0001F600", "\U0001F4A9", "\U0001F914", "\U0001F4DA",
              "\U0001F680", "\U0001F4BB", "\U0001F4C8", "\U0001F381"]
    for _ in range(120):
        n = rng.randint(2, 5)
        seq = "".join(rng.choice(astral) for _ in range(n))
        # Mix astral with combining
        cm = rng.choice(combiners)
        lines.append(f"astral: {seq} + base{cm} + {seq}")

    # Mixed-script with explicit combining mark on emoji
    for _ in range(40):
        lines.append("emoji+combining: \U0001F600⃣ (keycap-ish) " +
                     "and \U0001F1FA\U0001F1F8 flag mix")

    # CJK + Latin word boundaries (pretokenizer stress)
    for _ in range(40):
        lines.append("混合 mixed text 测试 with English 中间 and 数字 12345 boundaries")

    # Various NFC/NFKC divergences
    nfkc_targets = [
        "ﬁ vs fi",                           # ligature → decomposed
        "Ⅻ vs XII",                         # roman numeral
        "㎏ vs kg",                          # squared abbreviation
        "ｶﾀｶﾅ vs カタカナ",                  # halfwidth Katakana → fullwidth
        "ℋ vs H",                           # math symbol → Latin
        "Ⓐ vs A",                          # circled
    ]
    for _ in range(40):
        lines.append("normalization: " + rng.choice(nfkc_targets))

    # Trim/pad to 1000 lines.
    rng.shuffle(lines)
    if len(lines) < 1000:
        # Pad with mixed-script chains.
        extras = [
            "Lorem ipsum مرحبا 世界 \U0001F44B Bonjour Привет שלום",
            "Tab\there\tand\there\t— with NBSP and ZWSP​ mixed.",
            "Composed: é (U+00E9) vs Decomposed: é (e + COMBINING ACUTE)",
            "Emoji ZWJ family: \U0001F469‍\U0001F468‍\U0001F467‍\U0001F466 end",
            "RTL embedded: hello ‮olleh‬ world",
            "Math: \U0001D400\U0001D401\U0001D402 ≠ ABC",
            "Korean composed/decomposed: 한 vs 한 end",
        ]
        i = 0
        while len(lines) < 1000:
            lines.append(extras[i % len(extras)] + f" (i={i})")
            i += 1
    write(target, lines[:1000])


# --------------------------------------------------------------------- driver

BUILDERS = {
    "english.txt": build_english,
    "code.txt": build_code,
    "multilingual.txt": build_multilingual,
    "chat.txt": build_chat,
    "unicode_stress.txt": build_unicode_stress,
}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="rebuild even if file exists")
    ap.add_argument("--only", choices=list(BUILDERS), help="rebuild a single corpus")
    args = ap.parse_args(argv)

    items = [args.only] if args.only else list(BUILDERS)
    for name in items:
        path = os.path.join(HERE, name)
        if os.path.exists(path) and not args.force:
            print(f"[skip] {name} already present ({os.path.getsize(path)} bytes)",
                  file=sys.stderr)
            continue
        print(f"[build] {name}", file=sys.stderr)
        BUILDERS[name](path)
        size = os.path.getsize(path)
        with open(path, encoding="utf-8") as f:
            n_lines = sum(1 for _ in f)
        print(f"        wrote {size} bytes ({n_lines} lines)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
