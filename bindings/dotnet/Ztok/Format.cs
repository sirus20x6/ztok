// Format.cs — vocab-format enum + auto-detect helper.

using System;
using System.IO;
using System.Text;

namespace Ztok;

/// <summary>
/// Auto-detected on-disk vocab format. Mirrors <c>ztok_format</c> in
/// <c>include/ztok.h</c>; values are stable and additive.
/// </summary>
public enum Format : uint
{
    /// <summary>Unrecognized or unreadable file.</summary>
    Unknown = 0,
    /// <summary>OpenAI-style <c>.tiktoken</c> byte-level BPE vocab.</summary>
    Tiktoken = 1,
    /// <summary>HuggingFace <c>tokenizer.json</c> (BPE or WordPiece).</summary>
    HfJson = 2,
    /// <summary>SentencePiece <c>.model</c> (Unigram).</summary>
    SentencePiece = 3,
    /// <summary>ztok native <c>.ztm</c> TokenMonster format.</summary>
    Ztm = 4,
    /// <summary>Mistral <c>tekken.json</c> (tiktoken-style with extra config).</summary>
    Tekken = 5,
}

/// <summary>Normalizer kind (mirrors <c>ztok_normalizer_kind</c>).</summary>
public enum Normalizer : uint
{
    /// <summary>No normalization.</summary>
    Identity = 0,
    /// <summary>Unicode NFC normalization.</summary>
    Nfc = 1,
    /// <summary>Unicode NFD normalization.</summary>
    Nfd = 2,
    /// <summary>Unicode NFKC normalization.</summary>
    Nfkc = 3,
    /// <summary>Unicode NFKD normalization.</summary>
    Nfkd = 4,
    /// <summary>Byte-level normalization (GPT-2 / cl100k style).</summary>
    ByteLevel = 5,
}

/// <summary>Pre-tokenizer kind (mirrors <c>ztok_pretok_kind</c>).</summary>
public enum PreTokenizer : uint
{
    /// <summary>No pre-tokenization (treat input as a single span).</summary>
    Identity = 0,
    /// <summary>OpenAI cl100k_base regex split.</summary>
    Cl100k = 1,
}

/// <summary>Decoder kind (mirrors <c>ztok_decoder_kind</c>).</summary>
public enum Decoder : uint
{
    /// <summary>Concatenate token bytes verbatim.</summary>
    Concat = 0,
    /// <summary>WordPiece-style decoding (handles "##" continuation markers).</summary>
    WordPiece = 1,
    /// <summary>Byte-level decoding (inverse of byte-level normalization).</summary>
    ByteLevel = 2,
}

/// <summary>
/// Overlay channel kind (mirrors <c>ztok_overlay_kind</c>). Pass a set of
/// these to <see cref="Pipeline.EncodeWithOverlays"/> to request per-token
/// annotation channels aligned 1:1 with the id stream. Cheap channels
/// (<see cref="ByteStart"/> / <see cref="ByteEnd"/> / <see cref="Boundary"/>
/// / <see cref="Provenance"/>) carry encoder-derived values; domain
/// channels (<see cref="Opcode"/> / <see cref="Operand"/> /
/// <see cref="SymbolRef"/> / <see cref="Hunk"/>) come back zero-filled
/// until a domain plugin populates them.
/// </summary>
public enum OverlayKind : uint
{
    /// <summary>Original-input byte offset where the token's span starts.</summary>
    ByteStart = 0,
    /// <summary>Original-input byte offset where the token's span ends (exclusive).</summary>
    ByteEnd = 1,
    /// <summary>Boundary bitset: 0x1 chunk-start, 0x2 codepoint-start.</summary>
    Boundary = 2,
    /// <summary>Domain: normalized opcode class (zero-filled without a plugin).</summary>
    Opcode = 3,
    /// <summary>Domain: normalized operand class (zero-filled without a plugin).</summary>
    Operand = 4,
    /// <summary>Domain: symbol-table index, 0 = none (zero-filled without a plugin).</summary>
    SymbolRef = 5,
    /// <summary>Domain: diff-hunk id (zero-filled without a plugin).</summary>
    Hunk = 6,
    /// <summary>Provenance: 0 = model text, 1 = special token.</summary>
    Provenance = 7,
}

/// <summary>Format-detection helpers and libztok version probe.</summary>
public static class ZtokLibrary
{
    /// <summary>
    /// libztok version string (e.g. <c>"1.23.0"</c>).
    /// </summary>
    public static string Version
    {
        get
        {
            var raw = Native.Version();
            return Native.PtrToStringUtf8(raw) ?? string.Empty;
        }
    }

    /// <summary>
    /// Sniff <paramref name="path"/> for a known tokenizer format.
    /// Best-effort: any I/O error or unrecognized magic returns
    /// <see cref="Format.Unknown"/>.
    /// </summary>
    public static Format DetectFormat(string path)
    {
        ArgumentNullException.ThrowIfNull(path);
        return (Format)Native.AutoDetect(path);
    }
}
