//! Safe `Pipeline` wrapper around `ztok_pipeline*`.
//!
//! Construction goes through `Pipeline::open` (auto-detect) or one of
//! the format-specific `from_*` constructors. `Drop` calls
//! `ztok_pipeline_free`, so callers never have to remember to release a
//! handle.
//!
//! Thread-safety: the underlying libztok pipeline is read-only after
//! load (encode/decode never mutate it; the worker pool lives on the
//! `BatchPool` side, not the `Pipeline`). Per `src/c_api.zig`, encode
//! and decode are safe to call from multiple threads concurrently
//! against the same pipeline, so we mark `Pipeline` as both `Send` and
//! `Sync`.

use core::ffi::c_int;
use std::collections::BTreeMap;
use std::ffi::CString;
use std::path::Path;

use crate::error::{check_status, Error, Result};
use crate::sys;

/// Auto-detected on-disk vocab format (mirrors `ztok_format`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Format {
    /// Unrecognized or unreadable file.
    Unknown,
    /// OpenAI-style `.tiktoken` byte-level BPE vocab.
    Tiktoken,
    /// HuggingFace `tokenizer.json` (BPE or WordPiece).
    HfJson,
    /// SentencePiece `.model` (Unigram).
    SentencePiece,
    /// ztok native `.ztm` TokenMonster format.
    Ztm,
    /// Mistral `tekken.json` (tiktoken-style with an extra config block).
    Tekken,
    /// RWKV "World" vocab (`rwkv_vocab_v20230424.txt`): line-oriented
    /// `<id> <python-repr> <byte-len>` greedy longest-match byte trie.
    Rwkv,
}

impl Format {
    fn from_code(code: u32) -> Self {
        match code {
            sys::ZTOK_FORMAT_TIKTOKEN => Format::Tiktoken,
            sys::ZTOK_FORMAT_HF_JSON => Format::HfJson,
            sys::ZTOK_FORMAT_SP_MODEL => Format::SentencePiece,
            sys::ZTOK_FORMAT_ZTM => Format::Ztm,
            sys::ZTOK_FORMAT_TEKKEN => Format::Tekken,
            sys::ZTOK_FORMAT_RWKV => Format::Rwkv,
            _ => Format::Unknown,
        }
    }
}

/// Normalizer kind (mirrors `ztok_normalizer_kind`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[repr(u32)]
pub enum Normalizer {
    /// No normalization.
    Identity = sys::ZTOK_NORMALIZER_IDENTITY,
    /// Unicode NFC normalization.
    Nfc = sys::ZTOK_NORMALIZER_NFC,
    /// Unicode NFD normalization.
    Nfd = sys::ZTOK_NORMALIZER_NFD,
    /// Unicode NFKC normalization.
    Nfkc = sys::ZTOK_NORMALIZER_NFKC,
    /// Unicode NFKD normalization.
    Nfkd = sys::ZTOK_NORMALIZER_NFKD,
    /// Byte-level normalization (GPT-2 / cl100k style).
    ByteLevel = sys::ZTOK_NORMALIZER_BYTE_LEVEL,
}

/// Pre-tokenizer kind (mirrors `ztok_pretok_kind`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[repr(u32)]
pub enum PreTokenizer {
    /// No pre-tokenization (treat input as a single span).
    Identity = sys::ZTOK_PRETOK_IDENTITY,
    /// OpenAI cl100k_base regex split.
    Cl100k = sys::ZTOK_PRETOK_CL100K,
    /// Mistral Tekken pre-tokenization pattern.
    Tekken = sys::ZTOK_PRETOK_TEKKEN,
}

/// Decoder kind (mirrors `ztok_decoder_kind`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[repr(u32)]
pub enum Decoder {
    /// Concatenate token bytes verbatim.
    Concat = sys::ZTOK_DECODER_CONCAT,
    /// WordPiece-style decoding (handles `##` continuation markers).
    WordPiece = sys::ZTOK_DECODER_WORDPIECE,
    /// Byte-level decoding (inverse of byte-level normalization).
    ByteLevel = sys::ZTOK_DECODER_BYTE_LEVEL,
}

/// Overlay channel kind (mirrors `ztok_overlay_kind`).
///
/// Pass a slice of these to [`Pipeline::encode_with_overlays`] to request
/// per-token annotation channels aligned 1:1 with the id stream. Cheap
/// channels ([`ByteStart`](OverlayKind::ByteStart) / [`ByteEnd`](OverlayKind::ByteEnd)
/// / [`Boundary`](OverlayKind::Boundary) / [`Provenance`](OverlayKind::Provenance))
/// carry encoder-derived values; domain channels
/// ([`Opcode`](OverlayKind::Opcode) / [`Operand`](OverlayKind::Operand) /
/// [`SymbolRef`](OverlayKind::SymbolRef) / [`Hunk`](OverlayKind::Hunk)) come
/// back zero-filled until a domain plugin populates them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
#[repr(u32)]
pub enum OverlayKind {
    /// Original-input byte offset where the token's span starts.
    ByteStart = sys::ZTOK_OVERLAY_BYTE_START,
    /// Original-input byte offset where the token's span ends (exclusive).
    ByteEnd = sys::ZTOK_OVERLAY_BYTE_END,
    /// Boundary bitset: `0x1` chunk-start, `0x2` codepoint-start.
    Boundary = sys::ZTOK_OVERLAY_BOUNDARY,
    /// Domain: normalized opcode class (zero-filled without a plugin).
    Opcode = sys::ZTOK_OVERLAY_OPCODE,
    /// Domain: normalized operand class (zero-filled without a plugin).
    Operand = sys::ZTOK_OVERLAY_OPERAND,
    /// Domain: symbol-table index, `0` = none (zero-filled without a plugin).
    SymbolRef = sys::ZTOK_OVERLAY_SYMBOL_REF,
    /// Domain: diff-hunk id (zero-filled without a plugin).
    Hunk = sys::ZTOK_OVERLAY_HUNK,
    /// Provenance: `0` = model text, `1` = special token.
    Provenance = sys::ZTOK_OVERLAY_PROVENANCE,
}

/// Overlay domain (mirrors `ztok_overlay_domain`).
///
/// Selects which domain normalizer populates the domain overlay channels
/// ([`Opcode`](OverlayKind::Opcode) / [`Operand`](OverlayKind::Operand) /
/// [`SymbolRef`](OverlayKind::SymbolRef) / [`Hunk`](OverlayKind::Hunk)).
/// Pass to [`Pipeline::set_overlay_domain`]. [`None`](OverlayDomain::None)
/// (the default) leaves those channels zero-filled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[non_exhaustive]
#[repr(u32)]
pub enum OverlayDomain {
    /// No domain normalizer; domain channels stay zero-filled (default).
    #[default]
    None = sys::ZTOK_OVERLAY_DOMAIN_NONE,
    /// Decode the input as x86-64 machine code.
    X86_64 = sys::ZTOK_OVERLAY_DOMAIN_X86_64,
}

/// Where chunk edges are allowed to fall (mirrors `ztok_chunk_boundary`).
///
/// [`Token`](ChunkBoundary::Token) produces pure token-count windows;
/// the others snap window edges to the nearest boundary of the named
/// kind so a downstream embedder's spans line up with natural text
/// units. The default is [`Token`](ChunkBoundary::Token).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
#[repr(u32)]
pub enum ChunkBoundary {
    /// Pure token-count windows (default).
    #[default]
    Token = sys::ZTOK_CHUNK_BOUNDARY_TOKEN,
    /// Snap to a UTF-8 codepoint boundary.
    Codepoint = sys::ZTOK_CHUNK_BOUNDARY_CODEPOINT,
    /// Snap to a whitespace word boundary.
    Word = sys::ZTOK_CHUNK_BOUNDARY_WORD,
    /// Snap to a dictionary word boundary (CJK / Thai / ...).
    WordDict = sys::ZTOK_CHUNK_BOUNDARY_WORD_DICT,
    /// Snap to a sentence boundary.
    Sentence = sys::ZTOK_CHUNK_BOUNDARY_SENTENCE,
    /// Snap to a paragraph break (`\n\n`).
    Paragraph = sys::ZTOK_CHUNK_BOUNDARY_PARAGRAPH,
}

/// One token-window chunk produced by [`Pipeline::chunk`].
///
/// `ids` are the token ids in this window (copied out of C memory and
/// fully owned by Rust). `byte_start`/`byte_end` is the half-open byte
/// range this chunk covers in the ORIGINAL input; `token_start`/
/// `token_end` is the half-open token-index range in the full encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    /// Token ids in this window.
    pub ids: Vec<u32>,
    /// Half-open start byte offset in the original input.
    pub byte_start: u32,
    /// Half-open end byte offset (exclusive) in the original input.
    pub byte_end: u32,
    /// Half-open start token index in the full encoding.
    pub token_start: u32,
    /// Half-open end token index (exclusive) in the full encoding.
    pub token_end: u32,
}

/// Optional configuration for pipeline constructors.
///
/// The model kind is implicit (set by the constructor used: BPE,
/// WordPiece, Unigram, Monster, or byte_id). Pre-tokenizer defaults
/// differ between loaders (tiktoken defaults to CL100K; everything else
/// defaults to Identity) — `Default` reflects the byte_id constructor's
/// defaults; per-loader `from_*` methods override pre-tokenizer.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    /// Unicode / byte-level normalization applied before pre-tokenization.
    pub normalizer: Normalizer,
    /// Pre-tokenizer regex/strategy applied before the BPE merge loop.
    pub pre_tokenizer: PreTokenizer,
    /// Decoder strategy applied when joining decoded token bytes.
    pub decoder: Decoder,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            normalizer: Normalizer::Identity,
            pre_tokenizer: PreTokenizer::Identity,
            decoder: Decoder::Concat,
        }
    }
}

impl Config {
    fn to_c(self) -> sys::ZtokPipelineConfig {
        sys::ZtokPipelineConfig {
            normalizer: self.normalizer as u32,
            pre_tokenizer: self.pre_tokenizer as u32,
            model: sys::ZTOK_MODEL_BYTE_ID,
            decoder: self.decoder as u32,
        }
    }
}

/// A loaded tokenizer pipeline.
///
/// Construct via [`Pipeline::open`] (auto-detect) or one of the
/// `from_*` constructors. The handle is freed on `Drop`.
pub struct Pipeline {
    handle: *mut sys::ZtokPipeline,
}

// libztok's encode/decode paths take a `const ztok_pipeline*` and the
// underlying state is read-only after load. The persistent worker pool
// lives on `BatchPool`, not here. See `src/c_api.zig` for the
// thread-safety contract.
unsafe impl Send for Pipeline {}
unsafe impl Sync for Pipeline {}

impl Pipeline {
    /// Build the byte_id baseline pipeline (each input byte maps to its
    /// own id). Useful for tests / fuzzing because the round-trip is
    /// guaranteed exact for any byte sequence.
    pub fn byte_id(cfg: Option<Config>) -> Result<Self> {
        let cfg_c = cfg.unwrap_or_default().to_c();
        let mut status: c_int = sys::ZTOK_OK;
        let h = unsafe { sys::ztok_pipeline_new(&cfg_c, &mut status) };
        check_status(status, "ztok_pipeline_new")?;
        Self::wrap(h, "ztok_pipeline_new")
    }

    /// Auto-detect `path`'s vocab format and dispatch to the right
    /// loader. WordPiece (which lives inside `tokenizer.json` but needs
    /// a specific `unk_id`) is *not* covered — call
    /// [`Pipeline::from_wordpiece`] directly for that.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        match detect_format(path)? {
            Format::Tiktoken => Self::from_tiktoken(path, None),
            Format::HfJson => Self::from_hf_json(path, None),
            Format::SentencePiece => Self::from_sentencepiece(path, 0, None),
            Format::Ztm => Self::from_monster(path, None),
            Format::Rwkv => Self::from_rwkv(path, None),
            Format::Tekken => Self::from_tekken(path, None),
            Format::Unknown => Err(Error::UnknownFormat),
        }
    }

    /// Load a `.tiktoken` vocab into a byte-level BPE pipeline. The
    /// CL100K pre-tokenizer is the default for tiktoken files; override
    /// via `cfg.pre_tokenizer`.
    pub fn from_tiktoken<P: AsRef<Path>>(path: P, cfg: Option<Config>) -> Result<Self> {
        let mut cfg = cfg.unwrap_or(Config {
            pre_tokenizer: PreTokenizer::Cl100k,
            ..Config::default()
        });
        // Even when caller passes Some(cfg), tiktoken defaults to CL100K
        // unless the caller explicitly chose a different pre-tokenizer.
        // We can't tell "user passed Identity explicitly" from "user
        // left the default" without a richer Config type, so we honor
        // exactly what the caller passed (mirrors the Go binding's
        // behavior when CL100K=false is explicit).
        if cfg.pre_tokenizer == PreTokenizer::Identity {
            // Match the Python binding's default: from_tiktoken uses
            // CL100K unless the caller opted out. Here we treat a
            // None-cfg path through the branch above; an explicit
            // Identity stays Identity.
            let _ = &mut cfg;
        }
        let cpath = c_path(path.as_ref())?;
        let cfg_c = cfg.to_c();
        let mut status: c_int = sys::ZTOK_OK;
        let h = unsafe {
            sys::ztok_pipeline_new_bpe_from_tiktoken(cpath.as_ptr(), &cfg_c, &mut status)
        };
        check_status(status, "ztok_pipeline_new_bpe_from_tiktoken")?;
        Self::wrap(h, "ztok_pipeline_new_bpe_from_tiktoken")
    }

    /// Load a HuggingFace `tokenizer.json` BPE model.
    pub fn from_hf_json<P: AsRef<Path>>(path: P, cfg: Option<Config>) -> Result<Self> {
        let cpath = c_path(path.as_ref())?;
        let cfg_c = cfg.unwrap_or_default().to_c();
        let mut status: c_int = sys::ZTOK_OK;
        let h =
            unsafe { sys::ztok_pipeline_new_bpe_from_hf_json(cpath.as_ptr(), &cfg_c, &mut status) };
        check_status(status, "ztok_pipeline_new_bpe_from_hf_json")?;
        Self::wrap(h, "ztok_pipeline_new_bpe_from_hf_json")
    }

    /// Load a HuggingFace WordPiece model from `tokenizer.json`.
    /// `unk_id` is required (no sensible default for an unknown-token id).
    pub fn from_wordpiece<P: AsRef<Path>>(
        path: P,
        unk_id: u32,
        cfg: Option<Config>,
    ) -> Result<Self> {
        let cpath = c_path(path.as_ref())?;
        let cfg = cfg.unwrap_or(Config {
            decoder: Decoder::WordPiece,
            ..Config::default()
        });
        let cfg_c = cfg.to_c();
        let mut status: c_int = sys::ZTOK_OK;
        let h = unsafe {
            sys::ztok_pipeline_new_wordpiece_from_hf_json(
                cpath.as_ptr(),
                unk_id,
                &cfg_c,
                &mut status,
            )
        };
        check_status(status, "ztok_pipeline_new_wordpiece_from_hf_json")?;
        Self::wrap(h, "ztok_pipeline_new_wordpiece_from_hf_json")
    }

    /// Load a SentencePiece `.model` (Unigram) file. `unk_id` defaults
    /// to 0 if you pass `0`, matching the Python binding.
    pub fn from_sentencepiece<P: AsRef<Path>>(
        path: P,
        unk_id: u32,
        cfg: Option<Config>,
    ) -> Result<Self> {
        let cpath = c_path(path.as_ref())?;
        let cfg_c = cfg.unwrap_or_default().to_c();
        let mut status: c_int = sys::ZTOK_OK;
        let h = unsafe {
            sys::ztok_pipeline_new_unigram_from_sp_model(
                cpath.as_ptr(),
                unk_id,
                &cfg_c,
                &mut status,
            )
        };
        check_status(status, "ztok_pipeline_new_unigram_from_sp_model")?;
        Self::wrap(h, "ztok_pipeline_new_unigram_from_sp_model")
    }

    /// Load a ztok TokenMonster `.ztm` vocab file.
    pub fn from_monster<P: AsRef<Path>>(path: P, cfg: Option<Config>) -> Result<Self> {
        let cpath = c_path(path.as_ref())?;
        let cfg_c = cfg.unwrap_or_default().to_c();
        let mut status: c_int = sys::ZTOK_OK;
        let h = unsafe {
            sys::ztok_pipeline_new_monster_from_file(cpath.as_ptr(), &cfg_c, &mut status)
        };
        check_status(status, "ztok_pipeline_new_monster_from_file")?;
        Self::wrap(h, "ztok_pipeline_new_monster_from_file")
    }

    /// Load an RWKV "World" vocab (`rwkv_vocab_v20230424.txt`) into a
    /// greedy longest-match byte-trie pipeline. The World scheme is
    /// byte-lossless (every byte `0..=255` is a token), so it runs with
    /// an identity pre-tokenizer and concat decoder and `encode` never
    /// fails on unknown bytes.
    pub fn from_rwkv<P: AsRef<Path>>(path: P, cfg: Option<Config>) -> Result<Self> {
        let cpath = c_path(path.as_ref())?;
        let cfg_c = cfg.unwrap_or_default().to_c();
        let mut status: c_int = sys::ZTOK_OK;
        let h = unsafe {
            sys::ztok_pipeline_new_rwkv_from_file(cpath.as_ptr(), &cfg_c, &mut status)
        };
        check_status(status, "ztok_pipeline_new_rwkv_from_file")?;
        Self::wrap(h, "ztok_pipeline_new_rwkv_from_file")
    }

    /// Load a Mistral Tekken `tekken.json` vocab (Nemo / Pixtral /
    /// Devstral / Magistral, etc.) into a BPE pipeline. The loader lowers
    /// Tekken's base64 byte vocab into a `Bpe` with the special tokens
    /// packed into the bottom of the id space. The defaults are an
    /// identity normalizer, the Tekken pre-tokenizer (NOT cl100k), and a
    /// concat decoder (pieces are raw bytes); override via `cfg`.
    pub fn from_tekken<P: AsRef<Path>>(path: P, cfg: Option<Config>) -> Result<Self> {
        let cpath = c_path(path.as_ref())?;
        // Tekken defaults to its own pre-tokenizer pattern (NOT cl100k);
        // a None-cfg caller gets it. An explicit cfg is honored verbatim.
        let cfg = cfg.unwrap_or(Config {
            pre_tokenizer: PreTokenizer::Tekken,
            ..Config::default()
        });
        let cfg_c = cfg.to_c();
        let mut status: c_int = sys::ZTOK_OK;
        let h = unsafe {
            sys::ztok_pipeline_new_tekken_from_file(cpath.as_ptr(), &cfg_c, &mut status)
        };
        check_status(status, "ztok_pipeline_new_tekken_from_file")?;
        Self::wrap(h, "ztok_pipeline_new_tekken_from_file")
    }

    fn wrap(h: *mut sys::ZtokPipeline, op: &'static str) -> Result<Self> {
        if h.is_null() {
            Err(Error::NullHandle { op })
        } else {
            Ok(Self { handle: h })
        }
    }

    /// Encode a UTF-8 string into a vector of token ids.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.encode_bytes(text.as_bytes())
    }

    /// Encode raw bytes into token ids. Use this for non-UTF-8 inputs;
    /// the underlying tokenizer treats input as a byte stream.
    pub fn encode_bytes(&self, data: &[u8]) -> Result<Vec<u32>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }
        // libztok's per-span maxTokensFor upper bound is conservative,
        // so even a buffer sized to the true encoded length can trip
        // BUFFER_TOO_SMALL mid-stream. Start generous, double on retry,
        // hard-cap at 8 attempts (matches the Python/Go bindings).
        let mut cap = (data.len() + 16).max(64);
        for _ in 0..8 {
            let mut buf: Vec<u32> = vec![0; cap];
            let mut out_len: usize = 0;
            let rc = unsafe {
                sys::ztok_encode(
                    self.handle,
                    data.as_ptr() as *const _,
                    data.len(),
                    buf.as_mut_ptr(),
                    cap,
                    &mut out_len,
                )
            };
            match rc {
                sys::ZTOK_OK => {
                    buf.truncate(out_len);
                    return Ok(buf);
                }
                sys::ZTOK_ERR_BUFFER_TOO_SMALL => {
                    // out_len now carries the (conservative) required size.
                    cap = (cap * 2).max(out_len + 16);
                    continue;
                }
                _ => return Err(check_status(rc, "ztok_encode").unwrap_err()),
            }
        }
        Err(Error::Internal {
            status: sys::ZTOK_ERR_BUFFER_TOO_SMALL,
            op: "ztok_encode: BUFFER_TOO_SMALL after 8 grow attempts",
        })
    }

    /// Decode token ids back to a UTF-8 string. Invalid UTF-8 returns
    /// [`Error::InvalidUtf8`]; use [`Pipeline::decode_bytes`] to
    /// preserve raw bytes.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let bytes = self.decode_bytes(ids)?;
        String::from_utf8(bytes).map_err(|_| Error::InvalidUtf8)
    }

    /// Decode token ids to raw bytes (no UTF-8 round-tripping).
    pub fn decode_bytes(&self, ids: &[u32]) -> Result<Vec<u8>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        // Sizing pass: pass out_cap=0 so the call writes the required
        // size into out_len and returns BUFFER_TOO_SMALL.
        let mut sized: usize = 0;
        let rc = unsafe {
            sys::ztok_decode(
                self.handle,
                ids.as_ptr(),
                ids.len(),
                core::ptr::null_mut(),
                0,
                &mut sized,
            )
        };
        if rc != sys::ZTOK_OK && rc != sys::ZTOK_ERR_BUFFER_TOO_SMALL {
            return Err(check_status(rc, "ztok_decode (sizing)").unwrap_err());
        }
        if sized == 0 {
            return Ok(Vec::new());
        }
        let mut buf: Vec<u8> = vec![0; sized];
        let mut written: usize = 0;
        let rc = unsafe {
            sys::ztok_decode(
                self.handle,
                ids.as_ptr(),
                ids.len(),
                buf.as_mut_ptr() as *mut _,
                sized,
                &mut written,
            )
        };
        check_status(rc, "ztok_decode")?;
        buf.truncate(written);
        Ok(buf)
    }

    /// Select which domain normalizer populates the domain overlay channels
    /// ([`Opcode`](OverlayKind::Opcode) / [`Operand`](OverlayKind::Operand) /
    /// [`SymbolRef`](OverlayKind::SymbolRef) / [`Hunk`](OverlayKind::Hunk)).
    ///
    /// [`OverlayDomain::None`] (the default) leaves those channels
    /// zero-filled; [`OverlayDomain::X86_64`] decodes the input as x86-64
    /// machine code. An unrecognized value leaves the pipeline unchanged and
    /// returns [`Error::InvalidInput`].
    pub fn set_overlay_domain(&mut self, domain: OverlayDomain) -> Result<()> {
        let rc = unsafe { sys::ztok_pipeline_set_overlay_domain(self.handle, domain as u32) };
        check_status(rc, "ztok_pipeline_set_overlay_domain")
    }

    /// Encode `text` and return the ids plus a map of per-token overlay
    /// channels aligned 1:1 with the id stream.
    ///
    /// Requesting overlays never changes tokenization — the ids are
    /// identical to [`Pipeline::encode`]. Each requested [`OverlayKind`]
    /// maps to a `Vec<u32>` of `ids.len()` values. Cheap channels carry
    /// encoder-derived values; domain channels come back zero-filled
    /// until a domain plugin populates them.
    ///
    /// Duplicate kinds in `channels` are rejected with
    /// [`Error::InvalidInput`].
    pub fn encode_with_overlays(
        &self,
        text: &str,
        channels: &[OverlayKind],
    ) -> Result<(Vec<u32>, BTreeMap<OverlayKind, Vec<u32>>)> {
        self.encode_bytes_with_overlays(text.as_bytes(), channels)
    }

    /// Raw-bytes form of [`Pipeline::encode_with_overlays`]. Use this for
    /// non-UTF-8 inputs.
    pub fn encode_bytes_with_overlays(
        &self,
        data: &[u8],
        channels: &[OverlayKind],
    ) -> Result<(Vec<u32>, BTreeMap<OverlayKind, Vec<u32>>)> {
        // Reject duplicate kinds — the result map would silently collapse
        // them, which is almost certainly a caller bug.
        for (i, k) in channels.iter().enumerate() {
            if channels[..i].contains(k) {
                return Err(Error::InvalidInput);
            }
        }

        let empty_overlays = || -> BTreeMap<OverlayKind, Vec<u32>> {
            channels.iter().map(|&k| (k, Vec::new())).collect()
        };

        if data.is_empty() {
            return Ok((Vec::new(), empty_overlays()));
        }

        let n_ch = channels.len();

        // Sizing pass: out_ids = NULL queries the token count. Build the
        // channels[] array with NULL out pointers so the C side just
        // reports the count.
        let mut size_chans: Vec<sys::ZtokOverlayChannel> = channels
            .iter()
            .map(|&k| sys::ZtokOverlayChannel {
                kind: k as u32,
                out: core::ptr::null_mut(),
                out_cap: 0,
            })
            .collect();
        let size_chan_ptr = if n_ch == 0 {
            core::ptr::null_mut()
        } else {
            size_chans.as_mut_ptr()
        };

        let mut out_len: usize = 0;
        let rc = unsafe {
            sys::ztok_encode_with_overlays(
                self.handle,
                data.as_ptr() as *const _,
                data.len(),
                core::ptr::null_mut(),
                0,
                size_chan_ptr,
                n_ch,
                &mut out_len,
            )
        };
        if rc != sys::ZTOK_OK && rc != sys::ZTOK_ERR_BUFFER_TOO_SMALL {
            return Err(check_status(rc, "ztok_encode_with_overlays (sizing)").unwrap_err());
        }

        let count = out_len;
        if count == 0 {
            return Ok((Vec::new(), empty_overlays()));
        }

        // Fill pass: allocate the id buffer + one Vec<u32> per channel,
        // each sized to the exact count. The Vecs own their storage; we
        // hand the C side raw pointers into them and read back after.
        let mut ids: Vec<u32> = vec![0; count];
        let mut chan_bufs: Vec<Vec<u32>> = (0..n_ch).map(|_| vec![0u32; count]).collect();
        let mut fill_chans: Vec<sys::ZtokOverlayChannel> = channels
            .iter()
            .zip(chan_bufs.iter_mut())
            .map(|(&k, buf)| sys::ZtokOverlayChannel {
                kind: k as u32,
                out: buf.as_mut_ptr(),
                out_cap: count,
            })
            .collect();
        let fill_chan_ptr = if n_ch == 0 {
            core::ptr::null_mut()
        } else {
            fill_chans.as_mut_ptr()
        };

        let mut out_len2: usize = 0;
        let rc = unsafe {
            sys::ztok_encode_with_overlays(
                self.handle,
                data.as_ptr() as *const _,
                data.len(),
                ids.as_mut_ptr(),
                count,
                fill_chan_ptr,
                n_ch,
                &mut out_len2,
            )
        };
        check_status(rc, "ztok_encode_with_overlays")?;

        let n = out_len2;
        ids.truncate(n);
        let mut overlays = BTreeMap::new();
        for (&k, mut buf) in channels.iter().zip(chan_bufs.into_iter()) {
            buf.truncate(n);
            overlays.insert(k, buf);
        }
        Ok((ids, overlays))
    }

    /// Split `text` into overlapping token windows for embedding /
    /// late-chunking pipelines.
    ///
    /// Each returned [`Chunk`] holds at most `max_tokens` ids with
    /// `overlap` ids shared between neighbors (stride =
    /// `max_tokens - overlap`), plus the byte- and token-index range it
    /// covers in the original input. `boundary` selects where window
    /// edges may fall (see [`ChunkBoundary`]).
    ///
    /// Empty input returns an empty `Vec` with no error. Returns
    /// [`Error::InvalidInput`] if `max_tokens == 0` or
    /// `overlap >= max_tokens`. The C-owned id buffers are copied into
    /// Rust `Vec`s and freed before returning, so the result is fully
    /// owned by Rust.
    pub fn chunk(
        &self,
        text: &str,
        max_tokens: u32,
        overlap: u32,
        boundary: ChunkBoundary,
    ) -> Result<Vec<Chunk>> {
        self.chunk_bytes(text.as_bytes(), max_tokens, overlap, boundary)
    }

    /// Raw-bytes form of [`Pipeline::chunk`]. Use this for non-UTF-8
    /// inputs; the byte ranges in each [`Chunk`] index into `data`.
    pub fn chunk_bytes(
        &self,
        data: &[u8],
        max_tokens: u32,
        overlap: u32,
        boundary: ChunkBoundary,
    ) -> Result<Vec<Chunk>> {
        if max_tokens == 0 || overlap >= max_tokens {
            return Err(Error::InvalidInput);
        }
        if data.is_empty() {
            return Ok(Vec::new());
        }

        // Sizing pass: out_chunks = NULL -> *out_len = chunk count.
        let mut need: usize = 0;
        let rc = unsafe {
            sys::ztok_chunk(
                self.handle,
                data.as_ptr() as *const _,
                data.len(),
                max_tokens,
                overlap,
                boundary as u32,
                core::ptr::null_mut(),
                0,
                &mut need,
            )
        };
        if rc != sys::ZTOK_OK && rc != sys::ZTOK_ERR_BUFFER_TOO_SMALL {
            return Err(check_status(rc, "ztok_chunk (sizing)").unwrap_err());
        }
        let count = need;
        if count == 0 {
            return Ok(Vec::new());
        }

        // Fill pass: hand the C side a caller-owned record array. Each
        // record's `ids` is a ztok-allocated buffer; we copy the ids out
        // and release them via `ztok_chunks_free` before returning.
        let mut recs: Vec<sys::ZtokChunkRec> = (0..count)
            .map(|_| sys::ZtokChunkRec {
                ids: core::ptr::null_mut(),
                ids_len: 0,
                byte_start: 0,
                byte_end: 0,
                token_start: 0,
                token_end: 0,
            })
            .collect();
        let mut got: usize = 0;
        let rc = unsafe {
            sys::ztok_chunk(
                self.handle,
                data.as_ptr() as *const _,
                data.len(),
                max_tokens,
                overlap,
                boundary as u32,
                recs.as_mut_ptr(),
                count,
                &mut got,
            )
        };
        // On failure no records were written, so there are no ztok id
        // buffers to free; the `recs` Vec frees its own storage on drop.
        check_status(rc, "ztok_chunk")?;

        let out: Vec<Chunk> = recs[..got]
            .iter()
            .map(|r| {
                // SAFETY: on success the C side set `r.ids` to a
                // ztok-allocated buffer of `r.ids_len` u32s (NULL iff
                // ids_len == 0). Copy into Rust-owned memory; the C
                // buffer is freed by `ztok_chunks_free` below.
                let ids = if r.ids.is_null() || r.ids_len == 0 {
                    Vec::new()
                } else {
                    let slice = unsafe { core::slice::from_raw_parts(r.ids, r.ids_len) };
                    slice.to_vec()
                };
                Chunk {
                    ids,
                    byte_start: r.byte_start,
                    byte_end: r.byte_end,
                    token_start: r.token_start,
                    token_end: r.token_end,
                }
            })
            .collect();

        // SAFETY: `recs[..got]` are the records ztok_chunk populated;
        // ztok_chunks_free releases each record's `ids` buffer. The
        // record array itself stays caller-owned (the Vec).
        unsafe { sys::ztok_chunks_free(recs.as_mut_ptr(), got) };

        Ok(out)
    }

    /// Compute the tokenizer fingerprint — a deterministic 32-byte
    /// SHA-256 digest over the pipeline's encoding behavior on a fixed
    /// canonical input set, plus model-kind tag and vocab size.
    ///
    /// Two pipelines that return the same 32 bytes will produce
    /// bit-identical id streams for any input. Use it as a cache key,
    /// KV-store discriminator, or training-pipeline guard. New in
    /// libztok 1.22.
    pub fn fingerprint(&self) -> Result<[u8; 32]> {
        let mut out = [0u8; 32];
        let rc = unsafe { sys::ztok_fingerprint(self.handle, out.as_mut_ptr()) };
        check_status(rc, "ztok_fingerprint")?;
        Ok(out)
    }

    /// Borrow the raw FFI handle for use with [`crate::batch::BatchPool`]
    /// and [`crate::stream::StreamEncoder`]. Internal API.
    pub(crate) fn raw(&self) -> *mut sys::ZtokPipeline {
        self.handle
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { sys::ztok_pipeline_free(self.handle) };
            self.handle = core::ptr::null_mut();
        }
    }
}

/// Return libztok's version string (e.g. `"1.22.0"`).
pub fn version() -> Result<String> {
    let raw = unsafe { sys::ztok_version() };
    if raw.is_null() {
        return Err(Error::NullHandle { op: "ztok_version" });
    }
    let cstr = unsafe { core::ffi::CStr::from_ptr(raw) };
    cstr.to_str()
        .map(|s| s.to_owned())
        .map_err(|_| Error::InvalidUtf8)
}

/// Sniff `path` for a known tokenizer format. Best-effort: any I/O
/// error or unrecognized magic returns `Format::Unknown` (matches the C
/// contract — never raises).
pub fn detect_format<P: AsRef<Path>>(path: P) -> Result<Format> {
    let cpath = c_path(path.as_ref())?;
    let code = unsafe { sys::ztok_auto_detect(cpath.as_ptr()) };
    Ok(Format::from_code(code))
}

/// Convert a `Path` to a NUL-terminated C string. Returns
/// `Error::InvalidPath` if the path contains an interior NUL.
pub(crate) fn c_path(path: &Path) -> Result<CString> {
    // On Unix, OsStr -> &[u8] is free; on Windows, fall back to the
    // lossy str representation (the C ABI takes UTF-8 paths anyway).
    #[cfg(unix)]
    let bytes = {
        use std::os::unix::ffi::OsStrExt;
        path.as_os_str().as_bytes().to_vec()
    };
    #[cfg(not(unix))]
    let bytes = path.to_str().ok_or(Error::InvalidPath)?.as_bytes().to_vec();
    CString::new(bytes).map_err(|_| Error::InvalidPath)
}
