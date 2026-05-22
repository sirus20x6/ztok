// Right-aligned status bar item showing token / byte counts.
//
// The item is a single `StatusBarItem` reused for the lifetime of the
// extension. When the active editor changes we re-target it; when the
// selection changes we recompute the count over the selected range
// only (so users can highlight a paragraph and see its cost).

import * as vscode from 'vscode';

export interface CountSummary {
  readonly tokens: number;
  readonly bytes: number;
  /** True when the count was computed over a selection rather than the whole doc. */
  readonly scoped: boolean;
}

export class TokenStatusBar implements vscode.Disposable {
  private readonly item: vscode.StatusBarItem;

  constructor() {
    this.item = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    this.item.command = 'ztok.chooseVocab';
    this.item.tooltip = 'ZTok — click to choose a tokenizer vocab';
    this.item.hide();
  }

  /** True when the status bar is currently visible (used by tests). */
  get visible(): boolean {
    return this.item.text.length > 0;
  }

  /** Current text of the status item (for tests). */
  get text(): string {
    return this.item.text;
  }

  showCount(summary: CountSummary, vocabLabel: string): void {
    const bpt = summary.tokens > 0 ? summary.bytes / summary.tokens : 0;
    const prefix = summary.scoped ? '$(selection) ' : '$(symbol-string) ';
    const bptStr = bpt > 0 ? bpt.toFixed(2) : '-';
    this.item.text = `${prefix}${summary.tokens} tokens (${summary.bytes} bytes, bytes/token=${bptStr})`;
    this.item.tooltip = `ZTok: ${vocabLabel}\nClick to change tokenizer`;
    this.item.show();
  }

  showLoading(vocabLabel: string): void {
    this.item.text = '$(sync~spin) ZTok loading...';
    this.item.tooltip = `Loading ${vocabLabel}`;
    this.item.show();
  }

  showError(message: string): void {
    this.item.text = '$(error) ZTok';
    this.item.tooltip = `ZTok error: ${message}\nClick to choose a vocab`;
    this.item.show();
  }

  hide(): void {
    this.item.hide();
    this.item.text = '';
  }

  dispose(): void {
    this.item.dispose();
  }
}
