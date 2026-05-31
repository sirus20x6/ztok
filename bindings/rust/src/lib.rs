//! # ztok — Rust bindings for libztok
//!
//! Idiomatic safe Rust wrapper around libztok, a fast multithreaded
//! tokenizer library written in Zig. Mirrors the C ABI declared in
//! `include/ztok.h` and the surface of the existing Python / Node /
//! Ruby / Go bindings.
//!
//! ## Quickstart
//!
//! ```no_run
//! use ztok::Pipeline;
//!
//! // Auto-detect format (.tiktoken / tokenizer.json / .model / .ztm).
//! let pipe = Pipeline::open("cl100k_base.tiktoken")?;
//! let ids = pipe.encode("hello world")?;
//! let text = pipe.decode(&ids)?;
//! assert_eq!(text, "hello world");
//! # Ok::<(), ztok::Error>(())
//! ```
//!
//! ## Multi-threaded batching
//!
//! ```no_run
//! use ztok::{BatchPool, Pipeline};
//!
//! let pipe = Pipeline::open("cl100k_base.tiktoken")?;
//! let pool = BatchPool::new(8)?;
//! let results = pool.encode_batch(&pipe, &["foo", "bar", "baz"])?;
//! assert_eq!(results.len(), 3);
//! # Ok::<(), ztok::Error>(())
//! ```
//!
//! ## Streaming
//!
//! ```no_run
//! use ztok::Pipeline;
//!
//! let pipe = Pipeline::open("cl100k_base.tiktoken")?;
//! for batch in pipe.encode_stream_default(b"hello world")? {
//!     let ids = batch?;
//!     println!("{} ids", ids.len());
//! }
//! # Ok::<(), ztok::Error>(())
//! ```
//!
//! ## Fingerprint (new in 1.22)
//!
//! ```no_run
//! use ztok::Pipeline;
//!
//! let pipe = Pipeline::open("cl100k_base.tiktoken")?;
//! let fp: [u8; 32] = pipe.fingerprint()?;
//! // Use as a cache key, KV-store discriminator, etc.
//! # Ok::<(), ztok::Error>(())
//! ```
//!
//! ## Linking libztok
//!
//! By default the crate resolves libztok via `pkg-config`. After
//! `zig build -p prefix`, drop `prefix/lib/pkgconfig` on
//! `PKG_CONFIG_PATH` and `prefix/lib` on `LD_LIBRARY_PATH` so both the
//! build and the runtime loader can find it. Alternatives:
//!
//! - `ZTOK_LIB_DIR=/path/to/lib` overrides the pkg-config lookup.
//! - Build with `--features link-static` to link `libztok.a` instead of
//!   the shared object. The crate emits the typical Zig-stdlib C deps
//!   (`m`, `pthread`, `dl`) on Linux automatically.
//!
//! ## Wrapped C ABI surface
//!
//! Every function declared in `include/ztok.h` for libztok 1.22 is
//! wrapped:
//!
//! - lifecycle: `byte_id`, `from_tiktoken`, `from_hf_json`,
//!   `from_wordpiece`, `from_sentencepiece`, `from_monster`, `open`
//!   (auto-detect)
//! - encode / decode: `encode`, `encode_bytes`, `decode`, `decode_bytes`
//! - persistent batch: [`BatchPool`] (`encode_batch`,
//!   `encode_batch_bytes`)
//! - streaming: [`StreamEncoder`] sink-style API + [`Pipeline::encode_stream`]
//!   iterator adapter
//! - fingerprint: [`Pipeline::fingerprint`] (new in 1.22)
//! - utility: [`version`], [`detect_format`]
//!
//! Not yet exposed: `ztok_encode_batch` (the per-call worker pool
//! variant). Use [`BatchPool`] instead — it reuses arenas + worker
//! threads across calls and is what the other four bindings expose as
//! the primary batch surface.
//!
//! ## Safety
//!
//! All `unsafe` in this crate is confined to the [`sys`] module (raw
//! `extern "C"` declarations) and small unsafe blocks inside the
//! `pipeline`, `batch`, `stream`, and `ngram` modules that translate the
//! FFI surface to safe Rust. The public API contains no `unsafe`.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod sys;

mod batch;
mod error;
mod ngram;
mod pipeline;
mod stream;

pub use batch::BatchPool;
pub use error::{Error, Result};
pub use ngram::{hash_ngrams, hash_ngrams_batch};
pub use pipeline::{
    detect_format, version, Chunk, ChunkBoundary, Config, Decoder, Format, Normalizer,
    OverlayDomain, OverlayKind, Pipeline, PreTokenizer,
};
pub use stream::{StreamEncoder, StreamIter, DEFAULT_CHUNK_SIZE};
