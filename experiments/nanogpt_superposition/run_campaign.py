#!/usr/bin/env python3
"""Run an ordered config campaign and expose durable status to trainboard users."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("configs", type=Path, nargs="+")
    args = parser.parse_args()
    started = time.time()
    completed: list[str] = []
    for index, config in enumerate(args.configs):
        payload = {
            "schema": "ztok.nanogpt_superposition.campaign.v1",
            "started_unix": started,
            "updated_unix": time.time(),
            "state": "running",
            "current_index": index,
            "current_config": str(config.resolve()),
            "completed": completed,
            "configs": [str(path.resolve()) for path in args.configs],
        }
        write_status(args.status, payload)
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("train.py")),
                "--config",
                str(config),
            ],
            check=False,
        )
        if result.returncode:
            payload.update(
                {
                    "updated_unix": time.time(),
                    "state": "failed",
                    "returncode": result.returncode,
                }
            )
            write_status(args.status, payload)
            raise SystemExit(result.returncode)
        completed.append(str(config.resolve()))
    write_status(
        args.status,
        {
            "schema": "ztok.nanogpt_superposition.campaign.v1",
            "started_unix": started,
            "updated_unix": time.time(),
            "state": "completed",
            "completed": completed,
            "configs": [str(path.resolve()) for path in args.configs],
        },
    )


if __name__ == "__main__":
    main()
