"""libztok loader.

Resolution order:
    1. ZTOK_LIB_PATH environment variable (explicit override).
    2. Path next to this file (wheel install layout).
    3. ctypes.util.find_library("ztok") (system search).
    4. /usr/local/lib/libztok.{so,dylib}, /usr/lib/libztok.{so,dylib}.

Raises ZtokLibraryNotFoundError with a helpful message if nothing works.

Note on memory: the C ABI returns id arrays from `ztok_encode_batch_pooled`
that are NOT plain libc malloc — every buffer is prefixed by a small
length-header (see src/c_api.zig allocIdBuf / freeIdBuf), and the ONLY
correct way to free them is via `ztok_ids_free`. Never pass these
pointers to ctypes.cdll.libc.free or Python will eventually segfault.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from pathlib import Path
from typing import Optional


class ZtokLibraryNotFoundError(RuntimeError):
    """Raised when libztok cannot be located on the system."""


_PKG_DIR = Path(__file__).resolve().parent


def _shared_lib_basename() -> str:
    if sys.platform == "darwin":
        return "libztok.dylib"
    if sys.platform.startswith("win"):
        # No Windows shared lib is built today; included for completeness.
        return "ztok.dll"
    return "libztok.so"


def _candidate_paths() -> list[Path]:
    base = _shared_lib_basename()
    paths: list[Path] = []
    paths.append(_PKG_DIR / base)
    paths.append(_PKG_DIR.parent / base)
    # Walk up to the repo root for in-tree development (bindings/python/ztok/
    # → repo root → zig-out/lib).
    repo_zigout = _PKG_DIR.parent.parent.parent / "zig-out" / "lib" / base
    paths.append(repo_zigout)
    if sys.platform.startswith("linux"):
        paths.append(Path("/usr/local/lib") / base)
        paths.append(Path("/usr/lib") / base)
        paths.append(Path("/usr/lib64") / base)
    elif sys.platform == "darwin":
        paths.append(Path("/usr/local/lib") / base)
        paths.append(Path("/opt/homebrew/lib") / base)
    return paths


def _try_find_via_loader() -> Optional[str]:
    name = "ztok"
    found = ctypes.util.find_library(name)
    return found


def load_libztok() -> ctypes.CDLL:
    """Locate and dlopen libztok. See module docstring for the search order."""

    # 1. ZTOK_LIB_PATH explicit override.
    override = os.environ.get("ZTOK_LIB_PATH")
    if override:
        p = Path(override)
        if not p.exists():
            raise ZtokLibraryNotFoundError(
                f"ZTOK_LIB_PATH={override!r} does not point to an existing file."
            )
        return ctypes.CDLL(str(p))

    # 2 + 4. Standard candidate paths.
    candidates = _candidate_paths()
    tried: list[str] = []
    for path in candidates:
        tried.append(str(path))
        if path.exists():
            try:
                return ctypes.CDLL(str(path))
            except OSError as e:  # pragma: no cover - rare loader failure
                tried[-1] = f"{path} (dlopen failed: {e})"

    # 3. ctypes.util.find_library fallback.
    discovered = _try_find_via_loader()
    if discovered:
        tried.append(f"find_library -> {discovered}")
        try:
            return ctypes.CDLL(discovered)
        except OSError as e:  # pragma: no cover
            tried[-1] = f"{discovered} (dlopen failed: {e})"

    raise ZtokLibraryNotFoundError(
        "Could not locate libztok. Build it with `zig build` and either:\n"
        "  - install it system-wide (e.g. cp zig-out/lib/libztok.so /usr/local/lib/),\n"
        "  - place it next to the ztok Python package, or\n"
        "  - set ZTOK_LIB_PATH=/absolute/path/to/libztok.so.\n"
        "Tried: " + "; ".join(tried)
    )
