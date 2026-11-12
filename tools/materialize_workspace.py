#!/usr/bin/env python3
"""Create a runnable copy of the path-tokenized research code."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


TEXT_SUFFIXES = {".py", ".json", ".sh", ".md", ".txt", ".yaml", ".yml"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="Fresh project directory")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--change-agent-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--storage-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = Path(__file__).resolve().parents[1]
    stage_sources = (
        repository / "experiments" / "legacy_stages",
        repository / "baselines" / "wrappers",
    )
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to write into non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    ignore = shutil.ignore_patterns(
        "__pycache__", "*.pyc", "*.pt", "*.pth", "*.log", "runs", "results", "logs"
    )
    copied_names: set[str] = set()
    for source in stage_sources:
        for stage in source.iterdir():
            if not stage.is_dir():
                continue
            if stage.name in copied_names:
                raise SystemExit(f"Duplicate materialized stage name: {stage.name}")
            copied_names.add(stage.name)
            shutil.copytree(stage, output / stage.name, ignore=ignore)
    shutil.copytree(repository / "partial_relkd", output / "partial_relkd", ignore=ignore)
    shutil.copytree(repository / "configs", output / "configs", ignore=ignore)

    work_root = (args.work_root or output.parent).expanduser().resolve()
    storage_root = (args.storage_root or work_root.parent).expanduser().resolve()
    replacements = {
        "__PROJECT_ROOT__": output.as_posix(),
        "__DATA_ROOT__": args.data_root.expanduser().resolve().as_posix(),
        "__CHANGE_AGENT_ROOT__": args.change_agent_root.expanduser().resolve().as_posix(),
        "__WORK_ROOT__": work_root.as_posix(),
        "__STORAGE_ROOT__": storage_root.as_posix(),
    }

    changed = 0
    unresolved: list[str] = []
    for path in output.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        updated = text
        for token, value in replacements.items():
            updated = updated.replace(token, value)
        if updated != text:
            path.write_text(updated, encoding="utf-8")
            changed += 1
        if "__PROJECT_ROOT__" in updated or "__DATA_ROOT__" in updated or "__CHANGE_AGENT_ROOT__" in updated:
            unresolved.append(str(path))

    if unresolved:
        raise SystemExit("Unresolved path tokens:\n" + "\n".join(unresolved))
    print(f"MATERIALIZED_OK files_changed={changed} output={output}")


if __name__ == "__main__":
    main()
