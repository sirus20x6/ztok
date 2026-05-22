// Wraps the `ztok` Node binding for the extension's needs.
//
// The TypeScript declarations shipped with the binding expose `encode` /
// `decode` / `decodeBytes` but NOT a per-token offset API. To render
// highlights we need byte spans, so this module reconstructs them by
// decoding each token id and walking the result back over the input
// bytes. That gives us original-input offsets even when normalizers
// rewrite codepoints (because we always trust the encoder's id sequence
// to round-trip via decodeBytes; if a decoded piece can't be located we
// fall back to a zero-width span at the current cursor — defensive only).
//
// All file I/O is sync at load time (one Pipeline per vocab); encode is
// hot and stays on the main thread because koffi calls are sub-ms for
// realistic editor buffers (we cap render at 5000 tokens).

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import * as https from 'https';

// `ztok` is a CommonJS module; the typings on disk are accurate enough
// to use directly, but we don't want a hard build-time dep on having the
// native lib available at compile time. Use `require` so missing-binary
// failures surface at activation, not at compile.
type ZtokPipeline = {
  encode(text: string | Buffer): Uint32Array;
  decode(ids: Uint32Array | number[]): string;
  decodeBytes(ids: Uint32Array | number[]): Buffer;
  close(): void;
};

type ZtokModule = {
  version(): string;
  detectFormat(p: string): 'unknown' | 'tiktoken' | 'hf_json' | 'sentencepiece' | 'ztm';
  Pipeline: {
    fromPath(p: string, opts?: Record<string, unknown>): ZtokPipeline;
    fromTiktoken(p: string, opts?: Record<string, unknown>): ZtokPipeline;
  };
  ZtokLibraryNotFoundError: { new (...args: unknown[]): Error };
};

// Lazy require so test environments without the native lib can stub it.
let _ztok: ZtokModule | null = null;
function ztok(): ZtokModule {
  if (_ztok) return _ztok;
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  _ztok = require('ztok') as ZtokModule;
  return _ztok;
}

export interface TokenSpan {
  /** Token id in the vocabulary. */
  id: number;
  /** Byte offset (UTF-8) of the token's first byte within the original input. */
  byteStart: number;
  /** Byte offset (UTF-8) one past the token's last byte. */
  byteEnd: number;
  /** Decoded piece text, as UTF-8 (may be empty for control/special tokens). */
  piece: string;
}

export interface TokenInfo {
  id: number;
  piece: string;
  /** Rank when known (BPE merge order). Undefined for non-BPE pipelines. */
  rank?: number;
  /** Score when known (Unigram log-prob). Undefined for non-Unigram pipelines. */
  score?: number;
}

const CACHE_DIR = path.join(os.homedir(), '.cache', 'vscode-ztok');
const CL100K_FILENAME = 'cl100k_base.tiktoken';
const CL100K_URL =
  'https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken';

export class Tokenizer {
  private constructor(
    private readonly pipeline: ZtokPipeline,
    public readonly vocabPath: string,
    public readonly format: string
  ) {}

  /**
   * Load a pipeline from an on-disk vocab. Throws on failure (caller is
   * expected to surface a user-visible error).
   */
  static fromPath(vocabPath: string): Tokenizer {
    const z = ztok();
    const format = z.detectFormat(vocabPath);
    if (format === 'unknown') {
      throw new Error(
        `ztok could not auto-detect the format of ${vocabPath}. ` +
          'Pick a .tiktoken / tokenizer.json / .model / .ztm file.'
      );
    }
    const pipeline = z.Pipeline.fromPath(vocabPath);
    return new Tokenizer(pipeline, vocabPath, format);
  }

  /**
   * Locate the bundled / cached cl100k_base vocab. Returns the path on
   * success, or `null` if neither the repo copy nor the cache copy
   * exists yet. (Download happens via ensureDefaultVocab below.)
   */
  static findDefaultVocab(extensionRoot: string): string | null {
    const candidates = [
      // Side-by-side with the repo (tools/vscode-ztok -> ../../bench/vocabs/...).
      path.resolve(extensionRoot, '..', '..', 'bench', 'vocabs', CL100K_FILENAME),
      // Per-user cache.
      path.join(CACHE_DIR, CL100K_FILENAME),
    ];
    for (const c of candidates) {
      try {
        const st = fs.statSync(c);
        if (st.isFile() && st.size > 0) return c;
      } catch {
        /* fall through */
      }
    }
    return null;
  }

  /**
   * Ensure a default vocab is present on disk. If neither the repo copy
   * nor the cached copy is available, download cl100k_base into the
   * user cache directory. Returns the absolute path.
   */
  static async ensureDefaultVocab(extensionRoot: string): Promise<string> {
    const existing = Tokenizer.findDefaultVocab(extensionRoot);
    if (existing) return existing;
    fs.mkdirSync(CACHE_DIR, { recursive: true });
    const dest = path.join(CACHE_DIR, CL100K_FILENAME);
    await downloadFile(CL100K_URL, dest);
    return dest;
  }

  /** Encode bytes -> token ids. */
  encodeBytes(bytes: Buffer): Uint32Array {
    return this.pipeline.encode(bytes);
  }

  /**
   * Encode text and align every produced id to a byte span in the
   * original input. Spans are returned in token order.
   *
   * Algorithm: walk the id stream, decode each id to its raw bytes,
   * then advance a cursor over the source buffer matching that exact
   * byte sequence. If a piece doesn't match at the cursor (which can
   * happen for normalizer-introduced bytes, e.g. SentencePiece's leading
   * dummy ▁), we scan forward up to 16 bytes for the next match; if we
   * still can't find it the span collapses to the current cursor.
   */
  encodeWithOffsets(text: string): TokenSpan[] {
    const buf = Buffer.from(text, 'utf-8');
    const ids = this.pipeline.encode(buf);
    const spans: TokenSpan[] = new Array(ids.length);
    let cursor = 0;
    for (let i = 0; i < ids.length; i++) {
      const id = ids[i];
      const pieceBytes = this.pipeline.decodeBytes(Uint32Array.of(id));
      let start = cursor;
      let end = cursor;
      if (pieceBytes.length > 0) {
        const at = findSubarray(buf, pieceBytes, cursor, 16);
        if (at >= 0) {
          start = at;
          end = at + pieceBytes.length;
          cursor = end;
        } else {
          // Fallback: advance by 1 byte so we never get stuck in a loop
          // (covers normalizer drift; rare in practice).
          end = Math.min(cursor + 1, buf.length);
          cursor = end;
        }
      }
      spans[i] = {
        id,
        byteStart: start,
        byteEnd: end,
        piece: safeUtf8(pieceBytes),
      };
    }
    return spans;
  }

  /** Per-token info for the hover provider. */
  info(id: number): TokenInfo {
    const piece = safeUtf8(this.pipeline.decodeBytes(Uint32Array.of(id)));
    // The current Node binding doesn't expose per-token rank/score, so we
    // leave them undefined. Hover renders them only when present.
    return { id, piece };
  }

  dispose(): void {
    try {
      this.pipeline.close();
    } catch {
      /* ignore */
    }
  }
}

/**
 * Search `haystack` for `needle` starting at `from`, allowing the match
 * to slide forward by up to `slack` bytes. Returns absolute index or -1.
 */
function findSubarray(
  haystack: Buffer,
  needle: Buffer,
  from: number,
  slack: number
): number {
  if (needle.length === 0) return from;
  const limit = Math.min(haystack.length - needle.length, from + slack);
  for (let i = from; i <= limit; i++) {
    let ok = true;
    for (let j = 0; j < needle.length; j++) {
      if (haystack[i + j] !== needle[j]) {
        ok = false;
        break;
      }
    }
    if (ok) return i;
  }
  return -1;
}

/** UTF-8 decode that returns U+FFFD-laden text rather than throwing. */
function safeUtf8(b: Buffer): string {
  try {
    return b.toString('utf-8');
  } catch {
    return '';
  }
}

function downloadFile(url: string, dest: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const tmp = `${dest}.part`;
    const out = fs.createWriteStream(tmp);
    const req = https.get(url, (res) => {
      if (res.statusCode !== 200) {
        res.resume();
        reject(new Error(`download ${url} -> HTTP ${res.statusCode}`));
        return;
      }
      res.pipe(out);
      out.on('finish', () => {
        out.close(() => {
          try {
            fs.renameSync(tmp, dest);
            resolve();
          } catch (e) {
            reject(e instanceof Error ? e : new Error(String(e)));
          }
        });
      });
    });
    req.on('error', (e) => {
      try {
        fs.unlinkSync(tmp);
      } catch {
        /* ignore */
      }
      reject(e);
    });
  });
}
