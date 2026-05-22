# Changelog

## 0.1.0 - 2026-05-19

Initial release.

- Live alternating-color token highlighting for the active editor (300 ms debounce,
  re-renders only the visible viewport, capped at `ztok.maxTokensRendered` tokens
  per render).
- Status bar item `N tokens (M bytes, bytes/token=B)` updated for the active
  editor or selection.
- Hover provider showing token id, piece text, and (when available) rank/score
  for the token under the cursor.
- `ZTok: Choose tokenizer...` command with a file picker; selection persists
  per-workspace via `ztok.vocabPath`.
- `ZTok: Toggle token highlighting` command.
- Falls back to a bundled cl100k_base vocab when `ztok.vocabPath` is empty
  (loaded from `bench/vocabs/cl100k_base.tiktoken` if accessible, otherwise
  cached under `~/.cache/vscode-ztok/`).
