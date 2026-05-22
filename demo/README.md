# ztok demo site

A static, single-page demo of the ztok BPE tokenizer running in your
browser via WebAssembly. Built with vanilla HTML/CSS/JS — no
framework, no bundler, no runtime dependencies.

## What it shows

- **Live tokenization**: type into a textarea and watch token
  boundaries appear inline, with a sidebar of token IDs that
  cross-hover with the inline highlights.
- **Token count card**: bytes, tokens, and bytes-per-token ratio,
  recomputed on every keystroke.
- **Comparison mode**: pick two vocabs and see their tokenizations
  of the same text side-by-side.
- **Cost estimator**: pick a per-token price preset (GPT-4o, Claude
  Opus, etc.) and see the projected per-request / daily / monthly
  cost.
- **Custom vocab upload**: drop any `.tiktoken`-format vocab file
  in and tokenize against it locally.

Everything runs client-side. No network requests after the initial
page + WASM + vocab load.

## Local development

```sh
cd demo
python3 -m http.server 8742
# open http://localhost:8742
```

Any static file server will do. The WASM and vocab fetches are
relative URLs (no CDN).

## Rebuilding the WASM module

The WASM binary lives at `vendor/ztok-wasm.wasm` and `vendor/ztok-wasm-scalar.wasm`.
Both are checked in to make local dev frictionless. To rebuild from
the Zig source after editing `src/wasm_browser_root.zig`:

```sh
zig build ztok-wasm-browser           # SIMD128 build
zig build ztok-wasm-browser-scalar    # scalar fallback for older browsers

cp zig-out/bin/ztok_browser.wasm         demo/vendor/ztok-wasm.wasm
cp zig-out/bin/ztok_browser_scalar.wasm  demo/vendor/ztok-wasm-scalar.wasm
```

The GitHub Pages workflow (`.github/workflows/deploy-demo.yml`)
runs these steps on every push to `main`, so production deploys
always carry a freshly-built WASM.

## Bundled vocabs

- **cl100k_base.tiktoken** (~1.6 MB) — GPT-4 / GPT-3.5 tokenizer.
  The current WASM binding hard-wires the cl100k pre-tokenizer, so
  this is the only vocab format that produces meaningful encodings
  out of the box.
- **gpt2.json** (~1.3 MB) — HuggingFace `tokenizer.json` format,
  shipped as a preview. Not loadable by the current WASM binding
  (different format); shown in the dropdown for visibility and
  marked unavailable.
- **llama2** — SentencePiece, listed but disabled. SentencePiece
  support has not landed in the WASM binding yet.

Custom uploads expect a `.tiktoken` file (base64-id-per-line format,
the same shape as `cl100k_base.tiktoken` from the openai/tiktoken
repo).

## Deploying to GitHub Pages

The workflow at `.github/workflows/deploy-demo.yml` deploys this
directory to GitHub Pages on every push to `main` that touches
`demo/` or `src/wasm_browser_root.zig`.

To enable it on a fresh fork:

1. Go to **Settings → Pages** on the GitHub repo.
2. Under **Build and deployment**, set **Source** to **GitHub Actions**.
3. Push to `main`; the workflow will pick it up and deploy.

The site will be available at
`https://<your-user>.github.io/<repo>/` (or your custom domain).

## Files

```
demo/
├── README.md                 you are here
├── index.html                page markup, ~6 KB
├── style.css                 layout + theme, ~13 KB
├── app.js                    UI wiring + WASM driver, ~14 KB
├── vendor/
│   ├── ztok-wasm.js          ES-module loader, ~5 KB
│   ├── ztok-wasm.wasm        SIMD128 WASM build, ~384 KB
│   └── ztok-wasm-scalar.wasm scalar fallback, ~367 KB
└── vocabs/
    ├── cl100k_base.tiktoken  GPT-4/3.5 vocab, ~1.6 MB
    └── gpt2.json             gpt2 HF vocab (preview), ~1.3 MB
```

Total page weight (excluding WASM + vocabs):
**~38 KB** HTML + CSS + JS, uncompressed.
