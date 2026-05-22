// Integration tests for vscode-ztok.
//
// These run inside the VS Code Extension Host (launched by ../runTest.ts).
// We use an in-repo tiny .tiktoken fixture so the harness is hermetic
// even without network access.

import * as assert from 'assert';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import * as vscode from 'vscode';

import { __testing } from '../../src/extension';

const EXT_ID = 'anthropic-ztok.vscode-ztok';

function tinyTiktokenPath(): string {
  // 256 single-byte tokens + a handful of multi-byte merges that cover
  // the test corpus we open below. Base64-encoded piece + rank, one per
  // line, matching the `.tiktoken` format ztok auto-detects.
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ztok-vscode-test-'));
  const p = path.join(dir, 'tiny.tiktoken');
  const lines: string[] = [];
  let rank = 0;
  for (let b = 0; b < 256; b++) {
    lines.push(`${Buffer.from([b]).toString('base64')} ${rank++}`);
  }
  for (const piece of ['he', 'hel', 'hell', 'hello', ' wo', ' wor', ' worl', ' world']) {
    lines.push(`${Buffer.from(piece, 'utf-8').toString('base64')} ${rank++}`);
  }
  fs.writeFileSync(p, lines.join('\n') + '\n');
  return p;
}

async function settleDebounce(): Promise<void> {
  // Force an immediate refresh + wait one tick for VS Code to apply decorations.
  await __testing.runRefreshNow();
  await new Promise((r) => setTimeout(r, 50));
}

suite('vscode-ztok', () => {
  let vocabPath: string;
  let openedDoc: vscode.TextDocument;

  suiteSetup(async function () {
    this.timeout(30_000);
    vocabPath = tinyTiktokenPath();
    // Pin the vocab so activation doesn't try to fetch cl100k.
    const cfg = vscode.workspace.getConfiguration('ztok');
    await cfg.update('vocabPath', vocabPath, vscode.ConfigurationTarget.Global);
    await cfg.update('highlightEnabled', true, vscode.ConfigurationTarget.Global);
    await cfg.update('debounceMs', 1, vscode.ConfigurationTarget.Global);
    const ext = vscode.extensions.getExtension(EXT_ID);
    assert.ok(ext, `extension ${EXT_ID} should be installed`);
    await ext!.activate();

    openedDoc = await vscode.workspace.openTextDocument({
      language: 'plaintext',
      content: 'hello world\nhello world hello world\n',
    });
    await vscode.window.showTextDocument(openedDoc);
    await settleDebounce();
  });

  suiteTeardown(async () => {
    const cfg = vscode.workspace.getConfiguration('ztok');
    await cfg.update('vocabPath', '', vscode.ConfigurationTarget.Global);
    await cfg.update('highlightEnabled', undefined, vscode.ConfigurationTarget.Global);
    await cfg.update('debounceMs', undefined, vscode.ConfigurationTarget.Global);
  });

  test('tokenizer loads from the configured vocab', () => {
    assert.ok(__testing.isTokenizerLoaded(), 'tokenizer should be loaded after activation');
    const snap = __testing.getSnapshot();
    assert.ok(snap, 'snapshot should be populated for the active editor');
    assert.ok(snap!.spans.length > 0, 'snapshot should contain at least one token span');
  });

  test('snapshot tokens cover the buffer bytes contiguously', () => {
    const snap = __testing.getSnapshot()!;
    let lastEnd = 0;
    for (const s of snap.spans) {
      assert.ok(s.byteStart >= lastEnd, `span starts must be monotonic (got ${s.byteStart} < ${lastEnd})`);
      assert.ok(s.byteEnd >= s.byteStart, 'span end must be >= start');
      lastEnd = s.byteEnd;
    }
    assert.strictEqual(lastEnd, snap.bytes.length, 'tokens should cover the full buffer');
  });

  test('toggle highlight command flips the configuration flag', async () => {
    const before = vscode.workspace.getConfiguration('ztok').get<boolean>('highlightEnabled');
    await vscode.commands.executeCommand('ztok.toggleHighlight');
    const after = vscode.workspace.getConfiguration('ztok').get<boolean>('highlightEnabled');
    assert.notStrictEqual(before, after, 'toggle should change highlightEnabled');
    // Restore.
    await vscode.commands.executeCommand('ztok.toggleHighlight');
  });

  test('hover provider returns a Token <id> hover for an in-range position', async () => {
    const hovers = (await vscode.commands.executeCommand<vscode.Hover[]>(
      'vscode.executeHoverProvider',
      openedDoc.uri,
      new vscode.Position(0, 0)
    )) || [];
    assert.ok(hovers.length > 0, 'should produce at least one hover');
    const md = hovers[0].contents
      .map((c) => (typeof c === 'string' ? c : c.value))
      .join('\n');
    assert.match(md, /Token `\d+`/, `hover markdown should mention a token id (got: ${md})`);
  });

  test('refresh command repopulates the snapshot', async () => {
    await vscode.commands.executeCommand('ztok.refresh');
    await new Promise((r) => setTimeout(r, 50));
    const snap = __testing.getSnapshot();
    assert.ok(snap, 'snapshot should exist after explicit refresh');
  });
});
