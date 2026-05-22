// Thin wrapper over `vscode.workspace.getConfiguration('ztok')`.
//
// One place to read/write the four settings we expose so we can swap the
// storage layer (workspace vs global) without spraying scope choices through
// the rest of the extension. Vocab persistence intentionally defaults to
// workspace scope: a developer's per-project tokenizer choice should not
// leak into unrelated workspaces.

import * as path from 'path';
import * as vscode from 'vscode';

const SECTION = 'ztok';

export interface ZtokConfig {
  readonly vocabPath: string;
  readonly highlightEnabled: boolean;
  readonly maxTokensRendered: number;
  readonly debounceMs: number;
}

export function readConfig(): ZtokConfig {
  const cfg = vscode.workspace.getConfiguration(SECTION);
  return {
    vocabPath: cfg.get<string>('vocabPath', '').trim(),
    highlightEnabled: cfg.get<boolean>('highlightEnabled', true),
    maxTokensRendered: clampPositive(cfg.get<number>('maxTokensRendered', 5000), 5000),
    debounceMs: clampPositive(cfg.get<number>('debounceMs', 300), 300),
  };
}

export async function setVocabPath(absPath: string): Promise<void> {
  const cfg = vscode.workspace.getConfiguration(SECTION);
  // Prefer workspace scope when a workspace is open; otherwise fall back
  // to global so the choice still persists for single-file sessions.
  const target = vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders.length > 0
    ? vscode.ConfigurationTarget.Workspace
    : vscode.ConfigurationTarget.Global;
  await cfg.update('vocabPath', absPath, target);
}

export async function setHighlightEnabled(enabled: boolean): Promise<void> {
  const cfg = vscode.workspace.getConfiguration(SECTION);
  const target = vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders.length > 0
    ? vscode.ConfigurationTarget.Workspace
    : vscode.ConfigurationTarget.Global;
  await cfg.update('highlightEnabled', enabled, target);
}

/**
 * Resolve a user-supplied vocab path: expand ~, resolve relative paths
 * against the first workspace folder, and return an absolute string.
 */
export function resolveUserPath(input: string): string {
  if (!input) return '';
  let p = input;
  if (p.startsWith('~')) {
    const home = process.env.HOME || process.env.USERPROFILE || '';
    p = path.join(home, p.slice(1));
  }
  if (path.isAbsolute(p)) return p;
  const ws = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
  return ws ? path.resolve(ws, p) : path.resolve(p);
}

function clampPositive(value: number | undefined, fallback: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) return fallback;
  return Math.floor(value);
}
