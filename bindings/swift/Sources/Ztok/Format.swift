// Format — Swift mirror of `ztok_format`.
//
// Returned by `Pipeline.detectFormat(_:)` (best-effort sniffer). The C
// ABI guarantees that any I/O error or unrecognized magic maps to
// `.unknown` rather than surfacing as an error, so the Swift API does
// the same.

import CZtok
import Foundation

/// Auto-detected on-disk vocab format. Values mirror `ztok_format`
/// integers exactly so future additions stay non-breaking — unknown
/// codes fall through to `.unknown`.
public enum Format: UInt32, Sendable, CaseIterable {
    /// Unrecognized or unreadable file.
    case unknown = 0
    /// OpenAI-style `.tiktoken` byte-level BPE vocab.
    case tiktoken = 1
    /// HuggingFace `tokenizer.json` (BPE or WordPiece).
    case hfJson = 2
    /// SentencePiece `.model` (Unigram).
    case sentencePiece = 3
    /// ztok native `.ztm` TokenMonster format.
    case ztm = 4
    /// Mistral `tekken.json` (tiktoken-style with extra config).
    case tekken = 5
    /// RWKV "World" vocab (`rwkv_vocab_v20230424.txt`) — greedy
    /// longest-match byte trie.
    case rwkv = 6

    /// Build a `Format` from the raw `ztok_format` integer. Unknown
    /// codes map to `.unknown` — matches the C ABI's "best-effort,
    /// never errors" contract.
    internal static func fromCode(_ code: UInt32) -> Format {
        return Format(rawValue: code) ?? .unknown
    }
}

/// Unicode / byte-level normalizer kind (mirrors `ztok_normalizer_kind`).
public enum Normalizer: UInt32, Sendable {
    case identity = 0
    case nfc = 1
    case nfd = 2
    case nfkc = 3
    case nfkd = 4
    case byteLevel = 5
}

/// Pre-tokenizer kind (mirrors `ztok_pretok_kind`).
public enum PreTokenizer: UInt32, Sendable {
    case identity = 0
    case cl100k = 1
    /// Mistral Tekken pattern (used by `Pipeline.fromTekken`).
    case tekken = 2
}

/// Decoder kind (mirrors `ztok_decoder_kind`).
public enum Decoder: UInt32, Sendable {
    case concat = 0
    case wordPiece = 1
    case byteLevel = 2
}

/// Overlay channel kind (mirrors `ztok_overlay_kind`).
///
/// Pass a list of these to `Pipeline.encodeWithOverlays(_:channels:)` to
/// request per-token annotation channels aligned 1:1 with the id stream.
/// Cheap channels (`.byteStart` / `.byteEnd` / `.boundary` /
/// `.provenance`) carry encoder-derived values; domain channels
/// (`.opcode` / `.operand` / `.symbolRef` / `.hunk`) come back
/// zero-filled until a domain plugin populates them.
public enum OverlayKind: UInt32, Sendable, Hashable, CaseIterable {
    /// Original-input byte offset where the token's span starts.
    case byteStart = 0
    /// Original-input byte offset where the token's span ends (exclusive).
    case byteEnd = 1
    /// Boundary bitset: 0x1 chunk-start, 0x2 codepoint-start.
    case boundary = 2
    /// Domain: normalized opcode class (zero-filled without a plugin).
    case opcode = 3
    /// Domain: normalized operand class (zero-filled without a plugin).
    case operand = 4
    /// Domain: symbol-table index, 0 = none (zero-filled without a plugin).
    case symbolRef = 5
    /// Domain: diff-hunk id (zero-filled without a plugin).
    case hunk = 6
    /// Provenance: 0 = model text, 1 = special token.
    case provenance = 7
}

/// Overlay domain (mirrors `ztok_overlay_domain`).
///
/// Selects which domain normalizer populates the domain overlay channels
/// (`.opcode` / `.operand` / `.symbolRef` / `.hunk`). Pass to
/// `Pipeline.setOverlayDomain(_:)`. `.none` (the default) leaves those
/// channels zero-filled; `.x86_64` decodes the input as x86-64 machine code.
public enum OverlayDomain: UInt32, Sendable {
    /// No domain normalizer; domain channels stay zero-filled (default).
    case none = 0
    /// Decode the input as x86-64 machine code.
    case x86_64 = 1
}

/// Boundary mode for `Pipeline.chunk(_:maxTokens:overlap:boundary:)`
/// (mirrors `ztok_chunk_boundary`). Selects where chunk edges are
/// allowed to fall.
public enum ChunkBoundary: UInt32, Sendable {
    /// Pure token-count windows (default).
    case token = 0
    /// Snap to a UTF-8 codepoint boundary.
    case codepoint = 1
    /// Snap to a whitespace word boundary.
    case word = 2
    /// Snap to a dictionary word boundary (CJK/Thai/...).
    case wordDict = 3
    /// Snap to a sentence boundary.
    case sentence = 4
    /// Snap to a paragraph break (`\n\n`).
    case paragraph = 5
}

/// Optional configuration for pipeline constructors.
///
/// The model kind is implicit (set by the constructor used: BPE,
/// WordPiece, Unigram, Monster, or byte_id). Pre-tokenizer defaults
/// differ between loaders — `Pipeline.fromTiktoken` overrides to
/// `.cl100k` unless the caller passes a custom config.
public struct PipelineConfig: Sendable {
    public var normalizer: Normalizer
    public var preTokenizer: PreTokenizer
    public var decoder: Decoder

    public init(
        normalizer: Normalizer = .identity,
        preTokenizer: PreTokenizer = .identity,
        decoder: Decoder = .concat
    ) {
        self.normalizer = normalizer
        self.preTokenizer = preTokenizer
        self.decoder = decoder
    }

    internal func toC() -> ztok_pipeline_config {
        // ztok_model_kind only exposes BYTE_ID from ztok_pipeline_new;
        // the file constructors ignore the model field anyway because
        // they hard-wire BPE / WordPiece / Unigram / Monster.
        return ztok_pipeline_config(
            normalizer: ztok_normalizer_kind(rawValue: normalizer.rawValue),
            pre_tokenizer: ztok_pretok_kind(rawValue: preTokenizer.rawValue),
            model: ZTOK_MODEL_BYTE_ID,
            decoder: ztok_decoder_kind(rawValue: decoder.rawValue)
        )
    }
}
