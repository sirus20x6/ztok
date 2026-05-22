"""ztok — Python bindings for the ztok tokenizer library.

Thin ctypes wrapper around libztok. Mirrors the C ABI declared in
include/ztok.h. No Python-side allocation of id arrays — all id buffers
remain owned by the C library and are freed via `ztok_ids_free` when the
wrapper object is garbage-collected (via `weakref.finalize`).

Quickstart::

    import ztok

    pipe = ztok.Pipeline.from_tiktoken("cl100k_base.tiktoken", cl100k=True)
    ids = pipe.encode("hello world")
    text = pipe.decode(ids)
    pipe.close()

    with ztok.BatchPool(workers=8) as pool:
        results = pipe.encode_batch(pool, ["foo", "bar", "baz"])

    pipe_auto = ztok.Pipeline.from_path("tokenizer.json")
"""

from __future__ import annotations

import ctypes
import os
import weakref
from ctypes import POINTER, c_char, c_char_p, c_int, c_size_t, c_uint32, c_void_p
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Union

from . import _ffi
from ._ffi import (
    DECODER_BYTE_LEVEL,
    DECODER_CONCAT,
    DECODER_WORDPIECE,
    MODEL_BYTE_ID,
    NORMALIZER_BYTE_LEVEL,
    NORMALIZER_IDENTITY,
    NORMALIZER_NFC,
    NORMALIZER_NFD,
    NORMALIZER_NFKC,
    NORMALIZER_NFKD,
    OVERLAY_BOUNDARY,
    OVERLAY_BYTE_END,
    OVERLAY_BYTE_START,
    OVERLAY_HUNK,
    OVERLAY_OPCODE,
    OVERLAY_OPERAND,
    OVERLAY_PROVENANCE,
    OVERLAY_SYMBOL_REF,
    OVERLAY_USER_BASE,
    PRETOK_CL100K,
    PRETOK_IDENTITY,
    TokenId,
    TokenIdPtr,
    ZTOK_ERR_BUFFER_TOO_SMALL,
    ZTOK_ERR_INTERNAL,
    ZTOK_ERR_INVALID_INPUT,
    ZTOK_ERR_OUT_OF_MEMORY,
    ZTOK_OK,
    ZtokOverlayChannel,
    ZtokPipelineConfig,
)
from ._lib import ZtokLibraryNotFoundError, load_libztok


# --- exceptions ----------------------------------------------------------


class ZtokError(Exception):
    """Base class for all ztok errors."""


class ZtokInvalidInputError(ZtokError):
    """Maps ZTOK_ERR_INVALID_INPUT (status 2)."""


class ZtokOutOfMemoryError(ZtokError, MemoryError):
    """Maps ZTOK_ERR_OUT_OF_MEMORY (status 1)."""


class ZtokBufferTooSmallError(ZtokError):
    """Maps ZTOK_ERR_BUFFER_TOO_SMALL (status 3)."""


class ZtokInternalError(ZtokError):
    """Maps ZTOK_ERR_INTERNAL (status 99) and any unknown status code."""


_ERROR_MAP = {
    ZTOK_ERR_OUT_OF_MEMORY: ZtokOutOfMemoryError,
    ZTOK_ERR_INVALID_INPUT: ZtokInvalidInputError,
    ZTOK_ERR_BUFFER_TOO_SMALL: ZtokBufferTooSmallError,
    ZTOK_ERR_INTERNAL: ZtokInternalError,
}


def _raise_for_status(status: int, ctx: str) -> None:
    if status == ZTOK_OK:
        return
    cls = _ERROR_MAP.get(status, ZtokInternalError)
    raise cls(f"{ctx}: ztok status {status}")


# --- library handle (lazy singleton) -------------------------------------


_lib: Optional[ctypes.CDLL] = None


def _get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = _ffi.bind(load_libztok())
    return _lib


def version() -> str:
    """Return the libztok version string (e.g. ``"1.16.0"``)."""

    lib = _get_lib()
    raw = lib.ztok_version()
    if raw is None:
        raise ZtokInternalError("ztok_version returned NULL")
    return raw.decode("ascii")


# --- BatchPool wrapper ---------------------------------------------------


class BatchPool:
    """Persistent multithreaded worker pool.

    Reuse one pool across many ``encode_batch`` calls — each pool owns its
    own arenas and worker threads, so creating one per batch wastes work.

    Use as a context manager or call :meth:`close` explicitly::

        with ztok.BatchPool(workers=8) as pool:
            ids = pipe.encode_batch(pool, ["foo", "bar"])
    """

    def __init__(self, workers: int = 0) -> None:
        if workers < 0:
            raise ValueError("workers must be >= 0 (0 = auto)")
        lib = _get_lib()
        status = c_int(0)
        handle = lib.ztok_batch_pool_new(c_uint32(workers), ctypes.byref(status))
        _raise_for_status(status.value, "ztok_batch_pool_new")
        if not handle:
            raise ZtokInternalError("ztok_batch_pool_new returned NULL")
        self._handle: Optional[int] = handle
        self._finalizer = weakref.finalize(self, _BatchPool_free, lib, handle)

    @property
    def workers(self) -> int:
        """Actual worker count (resolves ``workers=0`` to the detected cpu count)."""

        if self._handle is None:
            raise ZtokError("BatchPool is closed")
        return int(_get_lib().ztok_batch_pool_worker_count(self._handle))

    def close(self) -> None:
        """Release the pool's worker threads and arenas. Safe to call twice."""

        if self._finalizer.alive:
            self._finalizer()
        self._handle = None

    def __enter__(self) -> "BatchPool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # Internal accessor; the Pipeline reaches in for the raw handle.
    def _raw(self) -> int:
        if self._handle is None:
            raise ZtokError("BatchPool is closed")
        return self._handle


def _BatchPool_free(lib: ctypes.CDLL, handle: int) -> None:
    lib.ztok_batch_pool_free(handle)


def _materialize_and_free_ids(
    lib: ctypes.CDLL, ptr: Optional[int], n: int
) -> List[int]:
    """Copy a libztok-owned id buffer into a Python list, then free it.

    Used by both `Pipeline.encode_batch` (per-input buffers) and the
    streaming path. The C side returns each buffer with a length-prefix
    header so the ONLY safe free is `ztok_ids_free` — see
    src/c_api.zig's allocIdBuf for details.
    """

    if not ptr or n <= 0:
        return []
    arr = ctypes.cast(ptr, POINTER(TokenId * n)).contents
    out = list(arr)
    lib.ztok_ids_free(ptr)
    return out


# --- Pipeline wrapper ----------------------------------------------------


def _make_config(
    normalizer: int,
    pre_tokenizer: int,
    model: int = MODEL_BYTE_ID,
    decoder: int = DECODER_CONCAT,
) -> ZtokPipelineConfig:
    return ZtokPipelineConfig(
        normalizer=normalizer,
        pre_tokenizer=pre_tokenizer,
        model=model,
        decoder=decoder,
    )


def _detect_format(path: Union[str, os.PathLike]) -> str:
    """Auto-detect a tokenizer file's format.

    Routes through the C ABI's `ztok_auto_detect` (post-1.18 agent C).
    Returns one of ``"tiktoken" | "hf_json" | "sentencepiece" | "ztm" |
    "unknown"`` — same string vocabulary the previous Python-side
    detector used, so callers don't need to know they're now hitting
    libztok directly. Best-effort: any I/O error collapses to
    ``"unknown"`` (matches the C contract).
    """

    return _ffi.ztok_auto_detect(_get_lib(), os.fspath(path))


class Pipeline:
    """A loaded tokenizer pipeline.

    Construct via one of the ``from_*`` class methods rather than calling
    ``__init__`` directly. Always close (via context manager or
    :meth:`close`) when done; the underlying library owns vocab tables
    that can be tens of MB.
    """

    def __init__(self, handle: int) -> None:
        if not handle:
            raise ZtokInternalError("Pipeline constructed with NULL handle")
        lib = _get_lib()
        self._handle: Optional[int] = handle
        self._finalizer = weakref.finalize(self, _Pipeline_free, lib, handle)

    # --- constructors ---------------------------------------------------

    @classmethod
    def byte_id(
        cls,
        *,
        normalizer: int = NORMALIZER_IDENTITY,
        pre_tokenizer: int = PRETOK_IDENTITY,
        decoder: int = DECODER_CONCAT,
    ) -> "Pipeline":
        """The byte_id baseline pipeline (each input byte maps to its own id)."""

        lib = _get_lib()
        cfg = _make_config(normalizer, pre_tokenizer, MODEL_BYTE_ID, decoder)
        status = c_int(0)
        handle = lib.ztok_pipeline_new(ctypes.byref(cfg), ctypes.byref(status))
        _raise_for_status(status.value, "ztok_pipeline_new")
        return cls(handle)

    @classmethod
    def from_tiktoken(
        cls,
        path: Union[str, os.PathLike],
        *,
        cl100k: bool = True,
        normalizer: int = NORMALIZER_IDENTITY,
        decoder: int = DECODER_CONCAT,
    ) -> "Pipeline":
        """Load a .tiktoken vocab into a byte-level BPE pipeline.

        ``cl100k=True`` is the right default for OpenAI cl100k_base.
        """

        return cls._from_file_with_cfg(
            "ztok_pipeline_new_bpe_from_tiktoken",
            path,
            normalizer=normalizer,
            pre_tokenizer=PRETOK_CL100K if cl100k else PRETOK_IDENTITY,
            decoder=decoder,
        )

    @classmethod
    def from_hf_json(
        cls,
        path: Union[str, os.PathLike],
        *,
        cl100k: bool = False,
        normalizer: int = NORMALIZER_IDENTITY,
        decoder: int = DECODER_CONCAT,
    ) -> "Pipeline":
        """Load a HuggingFace ``tokenizer.json`` BPE model."""

        return cls._from_file_with_cfg(
            "ztok_pipeline_new_bpe_from_hf_json",
            path,
            normalizer=normalizer,
            pre_tokenizer=PRETOK_CL100K if cl100k else PRETOK_IDENTITY,
            decoder=decoder,
        )

    @classmethod
    def from_wordpiece(
        cls,
        path: Union[str, os.PathLike],
        *,
        unk_id: int,
        normalizer: int = NORMALIZER_IDENTITY,
        pre_tokenizer: int = PRETOK_IDENTITY,
        decoder: int = DECODER_WORDPIECE,
    ) -> "Pipeline":
        """Load a HuggingFace WordPiece model from ``tokenizer.json``."""

        lib = _get_lib()
        cfg = _make_config(normalizer, pre_tokenizer, MODEL_BYTE_ID, decoder)
        status = c_int(0)
        path_b = os.fsencode(path)
        fn = getattr(lib, "ztok_pipeline_new_wordpiece_from_hf_json")
        handle = fn(path_b, c_uint32(unk_id), ctypes.byref(cfg), ctypes.byref(status))
        _raise_for_status(status.value, "ztok_pipeline_new_wordpiece_from_hf_json")
        return cls(handle)

    @classmethod
    def from_sentencepiece(
        cls,
        path: Union[str, os.PathLike],
        *,
        unk_id: int = 0,
        normalizer: int = NORMALIZER_IDENTITY,
        pre_tokenizer: int = PRETOK_IDENTITY,
        decoder: int = DECODER_CONCAT,
    ) -> "Pipeline":
        """Load a SentencePiece ``.model`` (Unigram) file."""

        lib = _get_lib()
        cfg = _make_config(normalizer, pre_tokenizer, MODEL_BYTE_ID, decoder)
        status = c_int(0)
        path_b = os.fsencode(path)
        fn = getattr(lib, "ztok_pipeline_new_unigram_from_sp_model")
        handle = fn(path_b, c_uint32(unk_id), ctypes.byref(cfg), ctypes.byref(status))
        _raise_for_status(status.value, "ztok_pipeline_new_unigram_from_sp_model")
        return cls(handle)

    @classmethod
    def from_monster(
        cls,
        path: Union[str, os.PathLike],
        *,
        normalizer: int = NORMALIZER_IDENTITY,
        pre_tokenizer: int = PRETOK_IDENTITY,
        decoder: int = DECODER_CONCAT,
    ) -> "Pipeline":
        """Load a ztok TokenMonster ``.ztm`` vocab file."""

        return cls._from_file_with_cfg(
            "ztok_pipeline_new_monster_from_file",
            path,
            normalizer=normalizer,
            pre_tokenizer=pre_tokenizer,
            decoder=decoder,
        )

    @classmethod
    def from_path(
        cls,
        path: Union[str, os.PathLike],
        *,
        unk_id: int = 0,
        normalizer: int = NORMALIZER_IDENTITY,
        decoder: Optional[int] = None,
    ) -> "Pipeline":
        """Auto-detect the file format and dispatch to the right loader.

        - ``.tiktoken``    → BPE + cl100k pre-tokenizer
        - ``tokenizer.json`` → BPE (HF JSON)
        - ``.model``       → SentencePiece Unigram (``unk_id`` defaults to 0)
        - ``.ztm``         → TokenMonster
        """

        fmt = _detect_format(path)
        if fmt == "tiktoken":
            return cls.from_tiktoken(path, normalizer=normalizer)
        if fmt == "hf_json":
            return cls.from_hf_json(path, normalizer=normalizer)
        if fmt == "sentencepiece":
            return cls.from_sentencepiece(path, unk_id=unk_id, normalizer=normalizer)
        if fmt == "ztm":
            return cls.from_monster(path, normalizer=normalizer)
        raise ZtokInvalidInputError(
            f"could not auto-detect tokenizer format for {path!r}; "
            "use a specific from_* constructor instead"
        )

    @classmethod
    def _from_file_with_cfg(
        cls,
        fn_name: str,
        path: Union[str, os.PathLike],
        *,
        normalizer: int,
        pre_tokenizer: int,
        decoder: int,
    ) -> "Pipeline":
        lib = _get_lib()
        cfg = _make_config(normalizer, pre_tokenizer, MODEL_BYTE_ID, decoder)
        status = c_int(0)
        path_b = os.fsencode(path)
        fn = getattr(lib, fn_name)
        handle = fn(path_b, ctypes.byref(cfg), ctypes.byref(status))
        _raise_for_status(status.value, fn_name)
        return cls(handle)

    # --- lifecycle ------------------------------------------------------

    def close(self) -> None:
        """Free the underlying pipeline. Safe to call multiple times."""

        if self._finalizer.alive:
            self._finalizer()
        self._handle = None

    def __enter__(self) -> "Pipeline":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _raw(self) -> int:
        if self._handle is None:
            raise ZtokError("Pipeline is closed")
        return self._handle

    # --- encode / decode -----------------------------------------------

    def encode(self, text: Union[str, bytes]) -> List[int]:
        """Encode a string (UTF-8) or bytes into a list of token ids."""

        lib = _get_lib()
        handle = self._raw()
        data = text.encode("utf-8") if isinstance(text, str) else bytes(text)
        if not data:
            return []

        # The C ABI's per-span `maxTokensFor` upper bound is conservative,
        # so even a buffer sized to the true encoded length can still
        # trip `BUFFER_TOO_SMALL` mid-stream. Start at a generous capacity
        # (input byte count + headroom) and grow on demand.
        cap = max(len(data) + 16, 64)
        for _ in range(8):
            buf = (TokenId * cap)()
            out_len = c_size_t(0)
            rc = lib.ztok_encode(
                handle,
                data,
                c_size_t(len(data)),
                ctypes.cast(buf, TokenIdPtr),
                c_size_t(cap),
                ctypes.byref(out_len),
            )
            if rc == ZTOK_OK:
                return list(buf[: out_len.value])
            if rc == ZTOK_ERR_BUFFER_TOO_SMALL:
                # `out_len` carries the (possibly conservative) required size.
                cap = max(cap * 2, out_len.value + 16)
                continue
            _raise_for_status(rc, "ztok_encode")
        raise ZtokInternalError(
            "ztok_encode kept reporting BUFFER_TOO_SMALL after 8 grow attempts"
        )

    # --- encode with overlays ------------------------------------------

    def encode_with_overlays(
        self,
        text: Union[str, bytes],
        channels: Sequence[int],
    ) -> "tuple[List[int], dict[int, List[int]]]":
        """Encode ``text`` and return ids plus aligned overlay channels.

        ``channels`` is a sequence of overlay-kind codes (the
        ``OVERLAY_*`` constants). The id stream is identical to
        :meth:`encode` — requesting overlays never changes tokenization.

        Returns ``(ids, overlays)`` where ``overlays`` maps each requested
        kind to a list of per-token uint32 values, one per id (so every
        list has ``len(ids)`` entries). Cheap channels
        (BYTE_START/BYTE_END/BOUNDARY/PROVENANCE) carry encoder-derived
        values; domain channels (OPCODE/OPERAND/SYMBOL_REF/HUNK) come back
        zero-filled until a domain plugin populates them.

        Mirrors the C ABI sizing protocol: a first call with
        ``out_ids=NULL`` queries the token count, then buffers are
        allocated and a second call fills them.
        """

        lib = _get_lib()
        handle = self._raw()
        data = text.encode("utf-8") if isinstance(text, str) else bytes(text)

        kinds = list(channels)
        if len(set(kinds)) != len(kinds):
            raise ZtokInvalidInputError(
                "encode_with_overlays: duplicate overlay kinds requested"
            )

        if not data:
            return [], {k: [] for k in kinds}

        n_channels = len(kinds)

        def _make_channels(bufs: Optional[List["ctypes.Array"]]):
            arr = (ZtokOverlayChannel * n_channels)()
            for i, kind in enumerate(kinds):
                arr[i].kind = kind
                if bufs is None:
                    arr[i].out = ctypes.cast(None, TokenIdPtr)
                    arr[i].out_cap = 0
                else:
                    arr[i].out = ctypes.cast(bufs[i], TokenIdPtr)
                    arr[i].out_cap = c_size_t(len(bufs[i]))
            return arr

        # Sizing pass: out_ids = NULL queries the token count.
        out_len = c_size_t(0)
        chan_arr = _make_channels(None)
        rc = lib.ztok_encode_with_overlays(
            handle,
            data,
            c_size_t(len(data)),
            ctypes.cast(None, TokenIdPtr),
            c_size_t(0),
            chan_arr if n_channels else None,
            c_size_t(n_channels),
            ctypes.byref(out_len),
        )
        if rc not in (ZTOK_OK, ZTOK_ERR_BUFFER_TOO_SMALL):
            _raise_for_status(rc, "ztok_encode_with_overlays (sizing)")

        count = out_len.value
        if count == 0:
            return [], {k: [] for k in kinds}

        # Fill pass: allocate ids buffer + one uint32 buffer per channel,
        # each sized to the token count. The conservative upper-bound that
        # plain encode can trip does not apply here — the count returned by
        # the sizing pass is exact for this overlay-aware path.
        id_buf = (TokenId * count)()
        chan_bufs = [(TokenId * count)() for _ in range(n_channels)]
        chan_arr = _make_channels(chan_bufs)
        out_len = c_size_t(0)
        rc = lib.ztok_encode_with_overlays(
            handle,
            data,
            c_size_t(len(data)),
            ctypes.cast(id_buf, TokenIdPtr),
            c_size_t(count),
            chan_arr if n_channels else None,
            c_size_t(n_channels),
            ctypes.byref(out_len),
        )
        _raise_for_status(rc, "ztok_encode_with_overlays")

        n = out_len.value
        ids = list(id_buf[:n])
        overlays = {
            kinds[i]: list(chan_bufs[i][:n]) for i in range(n_channels)
        }
        return ids, overlays

    # --- streaming ------------------------------------------------------

    # Default feed-chunk size matches the `ztok serve /encode_stream`
    # cadence (4 KiB is a typical socket-write granularity; we go a bit
    # bigger here so a 1 MB input fans out to ~16 yields instead of 256).
    _STREAM_FEED_SIZE: int = 64 * 1024

    def encode_stream(
        self,
        text: Union[str, bytes],
        chunk_size: int = _STREAM_FEED_SIZE,
    ) -> Iterator[List[int]]:
        """Stream-encode ``text`` and yield lists of ids as they're emitted.

        Internally wraps the C ABI ``ztok_stream_*`` family. The text (or
        bytes) is chopped into ``chunk_size``-byte pieces; each piece is
        fed to the encoder and any newly-emitted ids are yielded as a
        Python list. A final flush via ``ztok_stream_finish`` is run
        after the last chunk to drain the encoder's carry. Empty batches
        between yields are skipped so callers see only non-empty lists.

        The encoder defers a trailing partial UTF-8 codepoint or
        pre-tokenizer span up to a soft cap of 1 MiB (the same bound
        documented in src/stream.zig); past that it force-cuts at the
        nearest codepoint boundary.

        Example::

            with ztok.Pipeline.from_path("cl100k.tiktoken") as pipe:
                for batch in pipe.encode_stream("hello world"):
                    print(batch)
        """

        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

        lib = _get_lib()
        handle = self._raw()
        data = text.encode("utf-8") if isinstance(text, str) else bytes(text)

        status = c_int(0)
        stream_handle = lib.ztok_stream_new(handle, ctypes.byref(status))
        _raise_for_status(status.value, "ztok_stream_new")
        if not stream_handle:
            raise ZtokInternalError("ztok_stream_new returned NULL")

        try:
            for start in range(0, len(data), chunk_size):
                chunk = data[start : start + chunk_size]
                out_ids = c_void_p(0)
                out_n = c_size_t(0)
                rc = lib.ztok_stream_feed(
                    stream_handle,
                    chunk,
                    c_size_t(len(chunk)),
                    ctypes.byref(out_ids),
                    ctypes.byref(out_n),
                )
                _raise_for_status(rc, "ztok_stream_feed")
                ids = _materialize_and_free_ids(lib, out_ids.value, out_n.value)
                if ids:
                    yield ids

            # Final flush.
            out_ids = c_void_p(0)
            out_n = c_size_t(0)
            rc = lib.ztok_stream_finish(
                stream_handle,
                ctypes.byref(out_ids),
                ctypes.byref(out_n),
            )
            _raise_for_status(rc, "ztok_stream_finish")
            ids = _materialize_and_free_ids(lib, out_ids.value, out_n.value)
            if ids:
                yield ids
        finally:
            lib.ztok_stream_free(stream_handle)

    def decode(self, ids: Sequence[int]) -> str:
        """Decode a sequence of token ids back to a UTF-8 string."""

        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    def decode_bytes(self, ids: Sequence[int]) -> bytes:
        """Decode token ids without UTF-8 round-tripping (raw bytes)."""

        lib = _get_lib()
        handle = self._raw()
        n = len(ids)
        if n == 0:
            return b""

        id_buf = (TokenId * n)(*ids)
        # Sizing pass.
        out_len = c_size_t(0)
        rc = lib.ztok_decode(
            handle,
            ctypes.cast(id_buf, TokenIdPtr),
            c_size_t(n),
            None,
            c_size_t(0),
            ctypes.byref(out_len),
        )
        if rc != ZTOK_ERR_BUFFER_TOO_SMALL and rc != ZTOK_OK:
            _raise_for_status(rc, "ztok_decode (sizing)")
        nbytes = out_len.value
        if nbytes == 0:
            return b""

        # Allocate and decode.
        out_buf = ctypes.create_string_buffer(nbytes)
        rc = lib.ztok_decode(
            handle,
            ctypes.cast(id_buf, TokenIdPtr),
            c_size_t(n),
            ctypes.cast(out_buf, POINTER(c_char)),
            c_size_t(nbytes),
            ctypes.byref(out_len),
        )
        _raise_for_status(rc, "ztok_decode")
        return out_buf.raw[: out_len.value]

    # --- batch ----------------------------------------------------------

    def encode_batch(
        self,
        pool: BatchPool,
        inputs: Iterable[Union[str, bytes]],
    ) -> List[List[int]]:
        """Encode many strings in parallel via a persistent BatchPool.

        Each output id buffer is allocated and freed inside libztok; we
        materialize it into a Python list and immediately call
        ``ztok_ids_free`` (no lingering C pointers leak into Python).
        """

        lib = _get_lib()
        handle = self._raw()
        pool_handle = pool._raw()

        # Materialize inputs as bytes objects. We keep the bytes objects
        # alive in `byte_inputs` so the c_char_p pointers in
        # `inputs_arr` stay valid for the duration of the C call.
        byte_inputs: List[bytes] = [
            i.encode("utf-8") if isinstance(i, str) else bytes(i) for i in inputs
        ]
        n = len(byte_inputs)
        if n == 0:
            return []

        inputs_arr = (c_char_p * n)(*byte_inputs)
        lens_arr = (c_size_t * n)(*(len(b) for b in byte_inputs))
        out_ids = (c_void_p * n)()
        out_lens = (c_size_t * n)()

        rc = lib.ztok_encode_batch_pooled(
            handle,
            pool_handle,
            inputs_arr,
            lens_arr,
            c_size_t(n),
            out_ids,
            out_lens,
        )
        try:
            _raise_for_status(rc, "ztok_encode_batch_pooled")
            results: List[List[int]] = []
            for i in range(n):
                length = out_lens[i]
                ptr = out_ids[i]
                if length == 0 or not ptr:
                    results.append([])
                    continue
                # Cast c_void_p back to a TokenId array of the known length
                # and convert to a plain Python list. The C buffer stays
                # alive until we free it below.
                arr = ctypes.cast(ptr, POINTER(TokenId * length)).contents
                results.append(list(arr))
            return results
        finally:
            # Free every per-input id buffer through the C ABI. This is
            # the ONLY safe free path — the buffers carry a length-header
            # before the pointer (see src/c_api.zig).
            for i in range(n):
                if out_ids[i]:
                    lib.ztok_ids_free(out_ids[i])


def _Pipeline_free(lib: ctypes.CDLL, handle: int) -> None:
    lib.ztok_pipeline_free(handle)


__all__ = [
    "BatchPool",
    "Pipeline",
    "ZtokBufferTooSmallError",
    "ZtokError",
    "ZtokInternalError",
    "ZtokInvalidInputError",
    "ZtokLibraryNotFoundError",
    "ZtokOutOfMemoryError",
    # Enum constants (exposed so callers can pass non-default normalizers).
    "DECODER_BYTE_LEVEL",
    "DECODER_CONCAT",
    "DECODER_WORDPIECE",
    "NORMALIZER_BYTE_LEVEL",
    "NORMALIZER_IDENTITY",
    "NORMALIZER_NFC",
    "NORMALIZER_NFD",
    "NORMALIZER_NFKC",
    "NORMALIZER_NFKD",
    "OVERLAY_BOUNDARY",
    "OVERLAY_BYTE_END",
    "OVERLAY_BYTE_START",
    "OVERLAY_HUNK",
    "OVERLAY_OPCODE",
    "OVERLAY_OPERAND",
    "OVERLAY_PROVENANCE",
    "OVERLAY_SYMBOL_REF",
    "OVERLAY_USER_BASE",
    "PRETOK_CL100K",
    "PRETOK_IDENTITY",
    "version",
]
