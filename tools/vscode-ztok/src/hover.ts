// HoverProvider that resolves a cursor position to the token covering it
// and renders a one-line summary in markdown.
//
// We don't re-tokenize on every hover — the active TokenSession already
// keeps the most recent span array for the active editor. The hover
// provider does a binary search over those spans by byte offset to find
// the covering token.

import * as vscode from 'vscode';
import type { TokenSpan, Tokenizer } from './tokenizer';

export interface SessionSnapshot {
  /** Document URI (string) the spans belong to. */
  uri: string;
  /** Document version the spans were computed at. */
  version: number;
  spans: readonly TokenSpan[];
  /** Original UTF-8 buffer the encoder saw (used for position -> byte). */
  bytes: Buffer;
  /** Per-line cumulative byte offsets (length = lineCount + 1). */
  lineByteOffsets: Int32Array;
  tokenizer: Tokenizer;
  vocabLabel: string;
}

export class TokenHoverProvider implements vscode.HoverProvider {
  constructor(private readonly getSnapshot: () => SessionSnapshot | null) {}

  provideHover(
    document: vscode.TextDocument,
    position: vscode.Position,
    _token: vscode.CancellationToken
  ): vscode.ProviderResult<vscode.Hover> {
    const snap = this.getSnapshot();
    if (!snap) return null;
    if (snap.uri !== document.uri.toString()) return null;
    // Stale snapshot from a prior edit — better to say nothing than to
    // point at the wrong token.
    if (snap.version !== document.version) return null;

    const byteOffset = positionToByte(snap, position);
    const idx = findSpanIndex(snap.spans, byteOffset);
    if (idx < 0) return null;
    const span = snap.spans[idx];
    const info = snap.tokenizer.info(span.id);

    const pieceLabel = formatPiece(info.piece);
    const md = new vscode.MarkdownString(undefined, true);
    md.appendMarkdown(`**Token \`${span.id}\`** ${pieceLabel}`);
    md.appendMarkdown('\n\n');
    const meta: string[] = [];
    if (info.rank !== undefined) meta.push(`rank \`${info.rank}\``);
    if (info.score !== undefined) meta.push(`score \`${info.score.toFixed(4)}\``);
    meta.push(`bytes \`${span.byteEnd - span.byteStart}\``);
    meta.push(`position ${idx + 1} / ${snap.spans.length}`);
    md.appendMarkdown(meta.join(' · '));
    md.appendMarkdown(`\n\n_vocab: ${snap.vocabLabel}_`);
    md.isTrusted = false;

    // Highlight just the token's character range.
    const start = byteToPosition(snap, span.byteStart);
    const end = byteToPosition(snap, span.byteEnd);
    return new vscode.Hover(md, new vscode.Range(start, end));
  }
}

export function buildLineByteOffsets(text: string): {
  bytes: Buffer;
  lineByteOffsets: Int32Array;
} {
  const bytes = Buffer.from(text, 'utf-8');
  // Count lines: + 1 for the trailing offset past the last line.
  let lines = 1;
  for (let i = 0; i < bytes.length; i++) if (bytes[i] === 0x0a) lines++;
  const offs = new Int32Array(lines + 1);
  let li = 1;
  offs[0] = 0;
  for (let i = 0; i < bytes.length; i++) {
    if (bytes[i] === 0x0a) offs[li++] = i + 1;
  }
  offs[li] = bytes.length;
  return { bytes, lineByteOffsets: offs };
}

export function positionToByte(
  snap: { bytes: Buffer; lineByteOffsets: Int32Array },
  pos: vscode.Position
): number {
  const lineBase = snap.lineByteOffsets[Math.min(pos.line, snap.lineByteOffsets.length - 1)];
  // Re-walk the line up to the requested character to convert UTF-16
  // characters to UTF-8 bytes. Lines are usually short so a linear scan
  // is fine and avoids materializing a full position table.
  const nextLineBase =
    pos.line + 1 < snap.lineByteOffsets.length
      ? snap.lineByteOffsets[pos.line + 1]
      : snap.bytes.length;
  const lineSlice = snap.bytes.subarray(lineBase, Math.max(nextLineBase - 1, lineBase));
  const lineText = lineSlice.toString('utf-8');
  // Clamp to actual line length.
  const charsWanted = Math.min(pos.character, lineText.length);
  const prefix = lineText.substring(0, charsWanted);
  return lineBase + Buffer.byteLength(prefix, 'utf-8');
}

export function byteToPosition(
  snap: { bytes: Buffer; lineByteOffsets: Int32Array },
  byteOffset: number
): vscode.Position {
  const offs = snap.lineByteOffsets;
  // Binary search for the largest line whose offset <= byteOffset.
  let lo = 0;
  let hi = offs.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >>> 1;
    if (offs[mid] <= byteOffset) lo = mid;
    else hi = mid - 1;
  }
  const line = Math.min(lo, offs.length - 2);
  const lineStart = offs[line];
  const lineSlice = snap.bytes.subarray(lineStart, byteOffset);
  // utf-8 -> utf-16 (string) length gives the character column.
  const character = lineSlice.toString('utf-8').length;
  return new vscode.Position(line, character);
}

/**
 * Binary search for the span covering `byteOffset`. Spans are sorted by
 * byteStart and (assumed) non-overlapping. Returns the span index or -1
 * when no span covers the offset (e.g. when the cursor is on whitespace
 * the tokenizer treated as a separator boundary at exact length).
 */
function findSpanIndex(spans: readonly TokenSpan[], byteOffset: number): number {
  let lo = 0;
  let hi = spans.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >>> 1;
    const s = spans[mid];
    if (byteOffset < s.byteStart) hi = mid - 1;
    else if (byteOffset >= s.byteEnd) lo = mid + 1;
    else return mid;
  }
  // No covering span — return the closest preceding non-empty span so
  // the user still sees *something* meaningful at boundary positions.
  return Math.min(lo, spans.length - 1);
}

function formatPiece(piece: string): string {
  if (piece === '') return '`""` _(empty / special)_';
  // Escape backticks and show control chars as visible glyphs.
  const visible = piece.replace(/[\x00-\x1f\x7f]/g, (c) => {
    switch (c) {
      case '\n':
        return '⏎';
      case '\t':
        return '→';
      case '\r':
        return '⏎';
      default:
        return '●';
    }
  });
  const escaped = visible.replace(/`/g, '​`');
  return `\`"${escaped}"\``;
}
