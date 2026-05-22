// Entry point. Wires together:
//   - Tokenizer (loaded lazily on first activation)
//   - TokenDecorator (one decoration set, reused across editors)
//   - TokenStatusBar (one item, shows count for active editor / selection)
//   - TokenHoverProvider (looks up active-editor span snapshot)
//   - Commands: chooseVocab, toggleHighlight, refresh
//
// The active "session" (most recent successful tokenization for the
// active editor) is held in `currentSnapshot` so both the hover provider
// and the status bar can read it without re-tokenizing. Edits are
// debounced through `scheduleRefresh`; scroll events trigger a cheap
// re-render of the visible viewport but reuse the existing spans.

import * as path from 'path';
import * as vscode from 'vscode';

import { readConfig, resolveUserPath, setHighlightEnabled, setVocabPath } from './config';
import { TokenDecorator } from './decorations';
import {
  buildLineByteOffsets,
  byteToPosition,
  positionToByte,
  TokenHoverProvider,
  type SessionSnapshot,
} from './hover';
import { TokenStatusBar } from './statusbar';
import { Tokenizer, type TokenSpan } from './tokenizer';

let tokenizer: Tokenizer | null = null;
let currentSnapshot: SessionSnapshot | null = null;
let decorator: TokenDecorator | null = null;
let statusBar: TokenStatusBar | null = null;
let debounceTimer: NodeJS.Timeout | null = null;
let extensionContext: vscode.ExtensionContext | null = null;
const outputChannel = vscode.window.createOutputChannel('ZTok');

export async function activate(ctx: vscode.ExtensionContext): Promise<void> {
  extensionContext = ctx;
  decorator = new TokenDecorator();
  statusBar = new TokenStatusBar();
  ctx.subscriptions.push(decorator, statusBar, outputChannel);

  // Hover provider: registered for ALL schemes/languages — token boundaries
  // are universal across text content.
  ctx.subscriptions.push(
    vscode.languages.registerHoverProvider(
      { scheme: 'file' },
      new TokenHoverProvider(() => currentSnapshot)
    ),
    vscode.languages.registerHoverProvider(
      { scheme: 'untitled' },
      new TokenHoverProvider(() => currentSnapshot)
    )
  );

  // Commands.
  ctx.subscriptions.push(
    vscode.commands.registerCommand('ztok.chooseVocab', chooseVocabCommand),
    vscode.commands.registerCommand('ztok.toggleHighlight', toggleHighlightCommand),
    vscode.commands.registerCommand('ztok.refresh', () => scheduleRefresh(0))
  );

  // Editor lifecycle.
  ctx.subscriptions.push(
    vscode.window.onDidChangeActiveTextEditor(() => scheduleRefresh()),
    vscode.workspace.onDidChangeTextDocument((e) => {
      if (vscode.window.activeTextEditor?.document === e.document) {
        scheduleRefresh();
      }
    }),
    vscode.window.onDidChangeTextEditorSelection((e) => {
      if (e.textEditor === vscode.window.activeTextEditor) updateStatusBar();
    }),
    vscode.window.onDidChangeTextEditorVisibleRanges((e) => {
      if (e.textEditor === vscode.window.activeTextEditor) rerenderVisible();
    }),
    vscode.workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration('ztok')) onConfigChanged();
    })
  );

  // Eager-load the tokenizer + run the first pass against whatever editor
  // is already active.
  await ensureTokenizerLoaded();
  scheduleRefresh(0);
}

export function deactivate(): void {
  if (debounceTimer) {
    clearTimeout(debounceTimer);
    debounceTimer = null;
  }
  if (tokenizer) {
    tokenizer.dispose();
    tokenizer = null;
  }
  currentSnapshot = null;
}

// --- commands ---

async function chooseVocabCommand(): Promise<void> {
  const picked = await vscode.window.showOpenDialog({
    canSelectFiles: true,
    canSelectFolders: false,
    canSelectMany: false,
    openLabel: 'Use as ZTok tokenizer',
    title: 'Choose a tokenizer vocab',
    filters: {
      'Tokenizer files': ['tiktoken', 'json', 'model', 'ztm'],
      'All files': ['*'],
    },
  });
  if (!picked || picked.length === 0) return;
  const chosen = picked[0].fsPath;
  try {
    // Load it once to validate before persisting.
    const next = Tokenizer.fromPath(chosen);
    if (tokenizer) tokenizer.dispose();
    tokenizer = next;
    currentSnapshot = null;
    await setVocabPath(chosen);
    vscode.window.setStatusBarMessage(`ZTok: loaded ${path.basename(chosen)}`, 3000);
    scheduleRefresh(0);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    vscode.window.showErrorMessage(`ZTok: failed to load ${chosen}: ${msg}`);
  }
}

async function toggleHighlightCommand(): Promise<void> {
  const cfg = readConfig();
  await setHighlightEnabled(!cfg.highlightEnabled);
  // onConfigChanged will fire the re-render.
}

// --- refresh pipeline ---

function scheduleRefresh(delayMsOverride?: number): void {
  if (debounceTimer) {
    clearTimeout(debounceTimer);
    debounceTimer = null;
  }
  const delay = delayMsOverride ?? readConfig().debounceMs;
  debounceTimer = setTimeout(() => {
    debounceTimer = null;
    void runRefresh();
  }, Math.max(0, delay));
}

async function runRefresh(): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    if (statusBar) statusBar.hide();
    currentSnapshot = null;
    return;
  }
  const ok = await ensureTokenizerLoaded();
  if (!ok || !tokenizer) return;
  try {
    const doc = editor.document;
    const text = doc.getText();
    const { bytes, lineByteOffsets } = buildLineByteOffsets(text);
    const spans = tokenizer.encodeWithOffsets(text);
    currentSnapshot = {
      uri: doc.uri.toString(),
      version: doc.version,
      spans,
      bytes,
      lineByteOffsets,
      tokenizer,
      vocabLabel: labelForVocab(tokenizer.vocabPath),
    };
    rerenderVisible();
    updateStatusBar();
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    outputChannel.appendLine(`encode failed: ${msg}`);
    if (statusBar) statusBar.showError(msg);
  }
}

function rerenderVisible(): void {
  const editor = vscode.window.activeTextEditor;
  if (!editor || !decorator || !currentSnapshot) return;
  if (currentSnapshot.uri !== editor.document.uri.toString()) return;
  const cfg = readConfig();
  if (!cfg.highlightEnabled) {
    decorator.clear(editor);
    return;
  }
  const visible = unionRanges(editor.visibleRanges) ?? new vscode.Range(0, 0, editor.document.lineCount, 0);
  const snap = currentSnapshot;
  decorator.render(
    editor,
    snap.spans,
    (byteOffset) => byteToPosition(snap, byteOffset),
    visible,
    cfg.maxTokensRendered
  );
}

function updateStatusBar(): void {
  const editor = vscode.window.activeTextEditor;
  if (!editor || !statusBar || !currentSnapshot || !tokenizer) {
    if (statusBar) statusBar.hide();
    return;
  }
  const snap = currentSnapshot;
  const selection = editor.selection;
  let tokens: number;
  let bytes: number;
  let scoped: boolean;
  if (!selection.isEmpty) {
    const startByte = positionToByte(snap, selection.start);
    const endByte = positionToByte(snap, selection.end);
    bytes = endByte - startByte;
    tokens = countSpansInRange(snap.spans, startByte, endByte);
    scoped = true;
  } else {
    bytes = snap.bytes.length;
    tokens = snap.spans.length;
    scoped = false;
  }
  statusBar.showCount({ tokens, bytes, scoped }, snap.vocabLabel);
}

function countSpansInRange(spans: readonly TokenSpan[], startByte: number, endByte: number): number {
  // Count tokens whose midpoint falls within [start, end) — a stable
  // heuristic when the selection lands mid-token.
  let n = 0;
  for (const s of spans) {
    const mid = (s.byteStart + s.byteEnd) / 2;
    if (mid >= startByte && mid < endByte) n++;
    if (s.byteStart >= endByte) break;
  }
  return n;
}

// --- tokenizer load ---

async function ensureTokenizerLoaded(): Promise<boolean> {
  if (tokenizer) return true;
  if (!extensionContext || !statusBar) return false;
  const cfg = readConfig();
  let pathToLoad = cfg.vocabPath ? resolveUserPath(cfg.vocabPath) : '';
  if (!pathToLoad) {
    statusBar.showLoading('cl100k_base (default)');
    try {
      pathToLoad = await Tokenizer.ensureDefaultVocab(extensionContext.extensionPath);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      statusBar.showError(`could not locate default vocab: ${msg}`);
      outputChannel.appendLine(`default vocab failed: ${msg}`);
      return false;
    }
  } else {
    statusBar.showLoading(path.basename(pathToLoad));
  }
  try {
    tokenizer = Tokenizer.fromPath(pathToLoad);
    outputChannel.appendLine(`loaded ${pathToLoad} (${tokenizer.format})`);
    return true;
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    statusBar.showError(msg);
    outputChannel.appendLine(`load ${pathToLoad} failed: ${msg}`);
    return false;
  }
}

function onConfigChanged(): void {
  // Vocab change -> force reload.
  const cfg = readConfig();
  const resolved = cfg.vocabPath ? resolveUserPath(cfg.vocabPath) : '';
  if (tokenizer && resolved && resolved !== tokenizer.vocabPath) {
    tokenizer.dispose();
    tokenizer = null;
    currentSnapshot = null;
  }
  scheduleRefresh(0);
}

// --- helpers ---

function unionRanges(ranges: readonly vscode.Range[]): vscode.Range | null {
  if (ranges.length === 0) return null;
  let start = ranges[0].start;
  let end = ranges[0].end;
  for (let i = 1; i < ranges.length; i++) {
    if (ranges[i].start.isBefore(start)) start = ranges[i].start;
    if (ranges[i].end.isAfter(end)) end = ranges[i].end;
  }
  return new vscode.Range(start, end);
}

function labelForVocab(p: string): string {
  if (!p) return 'unknown';
  return path.basename(p);
}

// --- test surface ---
//
// Exported for the test harness. Not part of the public extension API.
export const __testing = {
  getSnapshot: () => currentSnapshot,
  isTokenizerLoaded: () => tokenizer !== null,
  runRefreshNow: runRefresh,
};
