# vscode-ztok

Live, in-editor token highlighting and analysis powered by the
[ztok](https://github.com/sirus20x6/ztok) tokenizer library.

## Features

- **Live token highlighting** — alternating-color spans over every token in
  the visible viewport. Updates 300 ms after the last edit.
- **Inline token count** in the status bar:
  `N tokens (M bytes, bytes/token=B)`. Switches to a selection-scoped count
  whenever a non-empty selection is active.
- **Hover info** — hovering over a character shows
  `Token <id> "<piece text>" (rank <r>, score <s>)` (rank/score included when
  the underlying tokenizer exposes them).
- **`ZTok: Choose tokenizer...`** — pick any `.tiktoken`, HuggingFace
  `tokenizer.json`, SentencePiece `.model`, or ztok `.ztm` file. Persists per
  workspace in `.vscode/settings.json` under `ztok.vocabPath`.
- **`ZTok: Toggle token highlighting`** — flip decorations on/off without
  unloading the tokenizer.

## Default vocab

When `ztok.vocabPath` is empty the extension uses `cl100k_base`. It looks
in the parent ztok repo (`bench/vocabs/cl100k_base.tiktoken`) first and
then under `~/.cache/vscode-ztok/cl100k_base.tiktoken`. If neither is
present the extension prompts you to choose a vocab.

## Performance

The viewport is re-tokenized incrementally — only the visible range is
decorated, capped at `ztok.maxTokensRendered` (default 5000) per render.
Edits coalesce through a 300 ms debounce, configurable via
`ztok.debounceMs`.

## Development

```sh
cd tools/vscode-ztok
npm install
npm run compile
```

To launch a development host inside VS Code, open this folder and press F5.

The Node binding (`ztok` npm package) is wired as a local file dependency at
`../../bindings/nodejs`; the native `libztok.{so,dylib,dll}` must already be
discoverable on `LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH`, or its location set
via `ZTOK_LIB_PATH`. Build it first with `zig build` in the repo root.

## License

AGPL-3.0-only — see `LICENSE`.
