"""Shared helper for concise public entry points."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_entry(relative_entry: str, description: str) -> None:
    parser = argparse.ArgumentParser(
        description=description,
        epilog="Arguments after '--' are forwarded unchanged to the validated stage driver.",
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    parsed = parser.parse_args()
    workspace = parsed.workspace.expanduser().resolve()
    entry = workspace / relative_entry
    if not entry.is_file():
        raise SystemExit(
            f"Missing materialized entry: {entry}\n"
            "Run tools/materialize_workspace.py first."
        )
    forwarded = parsed.args[1:] if parsed.args[:1] == ["--"] else parsed.args
    command = [sys.executable, str(entry), *forwarded]
    print("PUBLIC_ENTRY", " ".join(command), flush=True)
    raise SystemExit(subprocess.call(command, cwd=entry.parent.parent))
