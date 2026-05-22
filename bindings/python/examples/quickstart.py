"""3-line quickstart: load a tokenizer, encode, decode.

Run from the repo root after `zig build` succeeds::

    ZTOK_LIB_PATH=zig-out/lib/libztok.so \\
        python3 bindings/python/examples/quickstart.py path/to/tokenizer.model
"""

from __future__ import annotations

import sys

import ztok


def main(path: str) -> None:
    with ztok.Pipeline.from_path(path) as pipe:
        ids = pipe.encode("hello world")
        print(f"ztok {ztok.version()}: {len(ids)} ids -> {ids[:10]}...")
        print(f"decoded: {pipe.decode(ids)!r}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: quickstart.py <tokenizer-file>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
