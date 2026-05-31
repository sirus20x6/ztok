// TypeScript declarations for the `ztok` Node.js binding.
//
// Mirrors the C ABI declared in include/ztok.h. Every public method is
// covered; precise types throughout (Uint32Array for ids, Buffer for raw
// decoded bytes, string for UTF-8 text).

/// <reference types="node" />

/** Auto-detected tokenizer file format (return value of `detectFormat`). */
export type ZtokFormat = 'unknown' | 'tiktoken' | 'hf_json' | 'sentencepiece' | 'ztm' | 'rwkv';

/** Boundary mode for `Pipeline.chunk` (mirrors `ztok_chunk_boundary`). */
export enum ChunkBoundary {
    /** Pure token-count windows (default). */
    Token = 0,
    /** Snap to a UTF-8 codepoint boundary. */
    Codepoint = 1,
    /** Snap to a whitespace word boundary. */
    Word = 2,
    /** Snap to a dictionary word boundary (CJK/Thai/...). */
    WordDict = 3,
    /** Snap to a sentence boundary. */
    Sentence = 4,
    /** Snap to a paragraph break (\n\n). */
    Paragraph = 5,
}

/** One token window produced by `Pipeline.chunk`. */
export interface Chunk {
    /** The token ids in this chunk. */
    ids: Uint32Array;
    /** Half-open byte range this chunk covers in the ORIGINAL input. */
    byteStart: number;
    byteEnd: number;
    /** Half-open token-index range in the full encoding. */
    tokenStart: number;
    tokenEnd: number;
}

/** Options for `Pipeline.chunk`. */
export interface ChunkOptions {
    /** Tokens shared between neighboring windows. Default: 0. Must be < maxTokens. */
    overlap?: number;
    /** Boundary mode. Default: `ChunkBoundary.Token` (0). */
    boundary?: ChunkBoundary | number;
}

/** Common options for every file-loading constructor. */
export interface PipelineLoadOptions {
    /** Normalizer kind. Default: `NORMALIZER_IDENTITY`. */
    normalizer?: number;
    /** Pre-tokenizer kind. Default: constructor-specific. */
    preTokenizer?: number;
    /** Decoder kind. Default: constructor-specific. */
    decoder?: number;
}

/** Options for `Pipeline.fromTiktoken` and `Pipeline.fromHfJson`. */
export interface BpeLoadOptions extends PipelineLoadOptions {
    /** Enable the cl100k_base pre-tokenizer (true for OpenAI tiktoken vocabs). */
    cl100k?: boolean;
}

/** Options for `Pipeline.fromWordPiece`. */
export interface WordPieceLoadOptions extends PipelineLoadOptions {
    /** Unknown-token id. Required. */
    unkId: number;
}

/** Options for `Pipeline.fromSentencePiece`. */
export interface SentencePieceLoadOptions extends PipelineLoadOptions {
    /** Unknown-token id. Default: 0. */
    unkId?: number;
}

/** Options for `Pipeline.byteId`. */
export interface ByteIdOptions {
    normalizer?: number;
    preTokenizer?: number;
    decoder?: number;
}

/** Options for `Pipeline.encodeStream`. */
export interface StreamOptions {
    /** Bytes per feed. Default: 65536 (64 KiB). */
    chunkSize?: number;
}

/** Options for `BatchPool`. */
export interface BatchPoolOptions {
    /** Worker count. 0 (default) = auto-detect CPU count. */
    workers?: number;
}

/** Base class for every ztok error. */
export class ZtokError extends Error { }
export class ZtokInvalidInputError extends ZtokError { }
export class ZtokOutOfMemoryError extends ZtokError { }
export class ZtokBufferTooSmallError extends ZtokError { }
export class ZtokInternalError extends ZtokError { }
export class ZtokLibraryNotFoundError extends Error { }

/** Persistent multithreaded worker pool. Reuse across many encodeBatch calls. */
export class BatchPool {
    constructor(opts?: BatchPoolOptions);
    /** Actual worker count (resolves workers=0 to detected CPU count). */
    readonly workers: number;
    /** Release the pool's worker threads + arenas. Idempotent. */
    close(): void;
}

/**
 * A loaded tokenizer pipeline. Construct via one of the static `from*`
 * methods rather than calling `new Pipeline(...)` directly.
 */
export class Pipeline {
    /** The baseline byte_id pipeline (one id per input byte). */
    static byteId(opts?: ByteIdOptions): Pipeline;

    /** Load a .tiktoken vocab into a byte-level BPE pipeline. */
    static fromTiktoken(path: string, opts?: BpeLoadOptions): Pipeline;

    /** Load a HuggingFace tokenizer.json BPE model. */
    static fromHfJson(path: string, opts?: BpeLoadOptions): Pipeline;

    /** Load a HuggingFace WordPiece model from tokenizer.json. */
    static fromWordPiece(path: string, opts: WordPieceLoadOptions): Pipeline;

    /** Load a SentencePiece .model (Unigram) file. */
    static fromSentencePiece(path: string, opts?: SentencePieceLoadOptions): Pipeline;

    /** Load a ztok TokenMonster .ztm vocab file. */
    static fromMonster(path: string, opts?: PipelineLoadOptions): Pipeline;

    /**
     * Load an RWKV "World" vocab (rwkv_vocab_v20230424.txt). Greedy
     * longest-match byte trie with no pre-tokenizer.
     */
    static fromRWKV(path: string, opts?: PipelineLoadOptions): Pipeline;

    /**
     * Load a Mistral Tekken `tekken.json` vocab (Nemo / Pixtral /
     * Devstral / Magistral, etc.). Lowered into a BPE with the Tekken
     * pre-tokenizer (PRETOK_TEKKEN) and a concat decoder by default.
     */
    static fromTekken(path: string, opts?: PipelineLoadOptions): Pipeline;

    /**
     * Auto-detect the file format and dispatch to the right loader.
     *   - `.tiktoken`        -> BPE + cl100k pre-tokenizer
     *   - `tokenizer.json`   -> BPE (HF JSON)
     *   - `.model`           -> SentencePiece Unigram
     *   - `.ztm`             -> TokenMonster
     *   - RWKV World vocab   -> RWKV greedy byte trie
     *   - `tekken.json`      -> Mistral Tekken
     */
    static fromPath(
        path: string,
        opts?: BpeLoadOptions & SentencePieceLoadOptions
    ): Pipeline;

    /** Free the underlying pipeline. Idempotent. */
    close(): void;

    /** Encode a string (UTF-8) or bytes into a Uint32Array of token ids. */
    encode(text: string | Buffer | Uint8Array): Uint32Array;

    /**
     * Select which domain normalizer populates the domain overlay channels
     * (OPCODE/OPERAND/SYMBOL_REF/HUNK). Pass one of the `OVERLAY_DOMAIN_*`
     * constants. `OVERLAY_DOMAIN_NONE` (the default) leaves those channels
     * zero-filled. An unrecognized value throws.
     */
    setOverlayDomain(domain: number): void;

    /**
     * Encode `text` and return ids plus per-token overlay channels.
     *
     * `channels` is a list of overlay-kind codes (the `OVERLAY_*`
     * constants). Requesting overlays never changes tokenization — the id
     * stream is identical to {@link encode}. The returned `overlays` map is
     * keyed by overlay kind; each channel array has exactly `ids.length`
     * entries. Domain channels (OPCODE/OPERAND/SYMBOL_REF/HUNK) come back
     * zero-filled until a domain plugin populates them.
     */
    encodeWithOverlays(
        text: string | Buffer | Uint8Array,
        channels: number[]
    ): { ids: Uint32Array; overlays: Map<number, Uint32Array> };

    /** Decode token ids into a UTF-8 string. */
    decode(ids: Uint32Array | number[]): string;

    /** Decode token ids into raw bytes (no UTF-8 round-tripping). */
    decodeBytes(ids: Uint32Array | number[]): Buffer;

    /** Encode many inputs in parallel via a persistent BatchPool. */
    encodeBatch(
        pool: BatchPool,
        inputs: (string | Buffer | Uint8Array)[]
    ): Uint32Array[];

    /**
     * Stream-encode `text` and yield Uint32Array chunks as ids are
     * emitted. The encoder defers a trailing partial UTF-8 codepoint or
     * pre-tokenizer span up to a soft cap of 1 MiB.
     */
    encodeStream(
        text: string | Buffer | Uint8Array,
        opts?: StreamOptions
    ): Generator<Uint32Array, void, unknown>;

    /**
     * Split `text` into overlapping token windows (late chunking). Each
     * window holds at most `maxTokens` ids with `overlap` ids shared
     * between neighbors (stride = maxTokens - overlap). Returns [] for
     * empty input. Throws `ZtokInvalidInputError` if `maxTokens <= 0` or
     * `overlap >= maxTokens`.
     */
    chunk(
        text: string | Buffer | Uint8Array,
        maxTokens: number,
        opts?: ChunkOptions
    ): Chunk[];

    /**
     * Compute the tokenizer fingerprint: a deterministic 32-byte SHA-256
     * digest over the pipeline's encoding behavior on a fixed canonical
     * input set plus a model-kind tag and vocab size. Two pipelines that
     * return equal fingerprints produce bit-identical id streams for any
     * input — use it as a cache key, KV-store discriminator, or
     * training-pipeline guard.
     */
    fingerprint(): Fingerprint;
}

/**
 * 32-byte deterministic tokenizer fingerprint. Two pipelines that return
 * equal fingerprints produce bit-identical id streams for any input.
 */
export class Fingerprint {
    /** Fixed digest size in bytes (32). */
    static readonly SIZE: 32;
    /** Raw 32 fingerprint bytes (a fresh copy). */
    readonly bytes: Buffer;
    /** Lowercase 64-char hexadecimal form, no separators. */
    hex(): string;
    /** Structural equality against another Fingerprint. */
    equals(other: Fingerprint): boolean;
    toString(): string;
}

/**
 * Hash every length-`n` window of `ids` under `heads` hash functions,
 * returning the row-major [position][head] uint64 hashes as a
 * BigUint64Array (positions = ids.length - n + 1, or 0 if shorter than
 * one window). Deterministic; operates on raw ids — no Pipeline needed.
 */
export function hashNgrams(
    ids: Uint32Array | number[],
    n: number,
    heads: number
): BigUint64Array;

/**
 * Hash many id streams in parallel across `pool`. `results[i]` holds the
 * row-major hashes for `streams[i]` (empty for a stream shorter than one
 * window). Equivalent to calling `hashNgrams` on each stream.
 */
export function hashNgramsBatch(
    pool: BatchPool,
    streams: (Uint32Array | number[])[],
    n: number,
    heads: number
): BigUint64Array[];

/** Returns the libztok version string (e.g. "1.19.0"). */
export function version(): string;

/** Auto-detect the on-disk tokenizer format of `path`. Best-effort. */
export function detectFormat(path: string): ZtokFormat;

// --- enum constants (mirror ztok_*_kind enums in include/ztok.h) ---

export const NORMALIZER_IDENTITY: 0;
export const NORMALIZER_NFC: 1;
export const NORMALIZER_NFD: 2;
export const NORMALIZER_NFKC: 3;
export const NORMALIZER_NFKD: 4;
export const NORMALIZER_BYTE_LEVEL: 5;

export const PRETOK_IDENTITY: 0;
export const PRETOK_CL100K: 1;
export const PRETOK_TEKKEN: 2;

export const DECODER_CONCAT: 0;
export const DECODER_WORDPIECE: 1;
export const DECODER_BYTE_LEVEL: 2;

// --- overlay channel kinds (mirror ztok_overlay_kind in include/ztok.h) ---

export const OVERLAY_BYTE_START: 0;
export const OVERLAY_BYTE_END: 1;
export const OVERLAY_BOUNDARY: 2;
export const OVERLAY_OPCODE: 3;
export const OVERLAY_OPERAND: 4;
export const OVERLAY_SYMBOL_REF: 5;
export const OVERLAY_HUNK: 6;
export const OVERLAY_PROVENANCE: 7;
export const OVERLAY_USER_BASE: 0x8000;

/** Overlay domains (mirror `ztok_overlay_domain` in include/ztok.h). */
export const OVERLAY_DOMAIN_NONE: 0;
export const OVERLAY_DOMAIN_X86_64: 1;

// --- chunk boundary modes (mirror ztok_chunk_boundary in include/ztok.h) ---

export const CHUNK_BOUNDARY_TOKEN: 0;
export const CHUNK_BOUNDARY_CODEPOINT: 1;
export const CHUNK_BOUNDARY_WORD: 2;
export const CHUNK_BOUNDARY_WORD_DICT: 3;
export const CHUNK_BOUNDARY_SENTENCE: 4;
export const CHUNK_BOUNDARY_PARAGRAPH: 5;
