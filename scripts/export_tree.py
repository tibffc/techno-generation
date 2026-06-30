# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXCLUDES = {
    ".cache",
    ".git",
    ".venv",
    "__pycache__",
    "data/dirty_data",
    "data/processed",
    "data/raw_clean",
    "external",
    "outputs",
}
SKIP_SUFFIXES = {".aac", ".aif", ".aiff", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a compact project source tree.")
    parser.add_argument("--out", type=Path, default=Path("tree.txt"))
    parser.add_argument("--max-depth", type=int, default=4)
    return parser.parse_args()


def _is_excluded(path: Path) -> bool:
    rel = path.relative_to(PROJECT_ROOT).as_posix()
    parts = set(path.relative_to(PROJECT_ROOT).parts)
    return (
        rel in DEFAULT_EXCLUDES
        or bool(parts & {".git", ".venv", "__pycache__", ".cache", "external", "outputs"})
        or path.suffix.lower() in SKIP_SUFFIXES
    )


def build_tree(max_depth: int) -> str:
    lines = [PROJECT_ROOT.name + "/"]

    def walk(directory: Path, depth: int, prefix: str) -> None:
        if depth >= max_depth:
            return
        entries = sorted(
            [item for item in directory.iterdir() if not _is_excluded(item)],
            key=lambda item: (item.is_file(), item.name.lower()),
        )
        for index, item in enumerate(entries):
            connector = "`-- " if index == len(entries) - 1 else "|-- "
            suffix = "/" if item.is_dir() else ""
            lines.append(prefix + connector + item.name + suffix)
            if item.is_dir():
                extension = "    " if index == len(entries) - 1 else "|   "
                walk(item, depth + 1, prefix + extension)

    walk(PROJECT_ROOT, 0, "")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    target = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    target.write_text(build_tree(args.max_depth), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
