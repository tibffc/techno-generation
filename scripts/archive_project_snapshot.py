# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


LOGGER = logging.getLogger("archive_project_snapshot")
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".aiff", ".opus"}
MODEL_SUFFIXES = {".safetensors", ".pt", ".bin", ".ckpt"}
SKIP_DIR_NAMES = {".venv", "__pycache__", ".git", ".ipynb_checkpoints"}
INCLUDE_DIRECTORIES = [
    Path("scripts"),
    Path("configs"),
    Path("notebooks"),
    Path("app"),
    Path("outputs/logs"),
    Path("outputs/manifests"),
]
INCLUDE_FILES = [
    Path("README.md"),
    Path("requirements.txt"),
    Path("data/metadata.csv"),
    Path("data/metadata_features.csv"),
    Path("data/metadata_quality_report.json"),
]
OPTIONAL_EVAL_SUFFIXES = {".csv", ".md", ".json"}
EXCLUDED_ROOTS = [
    Path("external/ACE-Step-1.5"),
    Path("data/dirty_data"),
    Path("data/raw_clean"),
    Path("data/processed"),
    Path("outputs/tensors"),
    Path("outputs/checkpoints"),
    Path("outputs/samples"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive a lightweight snapshot of the project state.")
    parser.add_argument("--max_file_mb", type=float, default=20.0)
    parser.add_argument("--output_dir", type=Path, default=Path("archives"))
    parser.add_argument("--log_file", type=Path, default=Path("outputs/logs/archive_project_snapshot.log"))
    return parser.parse_args()


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )


def is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def should_skip_by_dir(path: Path) -> bool:
    return any(part in SKIP_DIR_NAMES for part in path.parts)


def should_skip_by_extension(path: Path) -> bool:
    suffix = path.suffix.lower()
    return suffix in AUDIO_SUFFIXES or suffix in MODEL_SUFFIXES


def should_skip_by_root(path: Path) -> bool:
    return any(is_under(path, WORKSPACE_ROOT / root) for root in EXCLUDED_ROOTS)


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    index = 0
    while size >= 1024.0 and index < len(units) - 1:
        size /= 1024.0
        index += 1
    return f"{size:.2f} {units[index]}"


def collect_candidates() -> tuple[list[Path], list[str], list[str]]:
    included_files: list[Path] = []
    included_roots: list[str] = []
    excluded_roots: list[str] = [str(root) for root in EXCLUDED_ROOTS]

    for rel_path in INCLUDE_DIRECTORIES:
        abs_path = WORKSPACE_ROOT / rel_path
        included_roots.append(str(rel_path))
        if not abs_path.exists():
            continue
        for file_path in abs_path.rglob("*"):
            if file_path.is_file():
                included_files.append(file_path)

    for rel_path in INCLUDE_FILES:
        abs_path = WORKSPACE_ROOT / rel_path
        if abs_path.exists():
            included_files.append(abs_path)

    eval_dir = WORKSPACE_ROOT / "outputs/evaluation"
    if eval_dir.exists():
        included_roots.append("outputs/evaluation (small csv/md/json only)")
        for file_path in eval_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in OPTIONAL_EVAL_SUFFIXES:
                included_files.append(file_path)

    deduped = sorted(set(included_files))
    return deduped, included_roots, excluded_roots


def filter_files(candidates: list[Path], max_file_bytes: int) -> tuple[list[Path], list[dict[str, str]]]:
    kept: list[Path] = []
    skipped: list[dict[str, str]] = []

    for path in candidates:
        rel = path.relative_to(WORKSPACE_ROOT)
        if should_skip_by_root(path):
            skipped.append({"path": rel.as_posix(), "reason": "excluded_root"})
            continue
        if should_skip_by_dir(rel):
            skipped.append({"path": rel.as_posix(), "reason": "excluded_directory_name"})
            continue
        if should_skip_by_extension(path):
            skipped.append({"path": rel.as_posix(), "reason": f"excluded_extension:{path.suffix.lower()}"})
            continue
        try:
            size_bytes = path.stat().st_size
        except OSError as error:
            skipped.append({"path": rel.as_posix(), "reason": f"stat_failed:{error}"})
            continue
        if size_bytes > max_file_bytes:
            skipped.append({"path": rel.as_posix(), "reason": f"file_too_large:{human_size(size_bytes)}"})
            continue
        kept.append(path)
    return kept, skipped


def build_manifest(
    archive_path: Path,
    files: list[Path],
    skipped: list[dict[str, str]],
    included_roots: list[str],
    excluded_roots: list[str],
    max_file_mb: float,
) -> dict:
    file_entries = []
    total_uncompressed = 0
    for path in files:
        rel = path.relative_to(WORKSPACE_ROOT)
        size_bytes = path.stat().st_size
        total_uncompressed += size_bytes
        file_entries.append(
            {
                "path": rel.as_posix(),
                "size_bytes": size_bytes,
                "size_human": human_size(size_bytes),
            }
        )

    return {
        "archive_path": str(archive_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "workspace_root": str(WORKSPACE_ROOT),
        "max_file_mb": max_file_mb,
        "included_roots": included_roots,
        "excluded_roots": excluded_roots,
        "included_file_count": len(file_entries),
        "skipped_file_count": len(skipped),
        "total_uncompressed_bytes": total_uncompressed,
        "total_uncompressed_human": human_size(total_uncompressed),
        "included_files": file_entries,
        "skipped_files": skipped,
    }


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = (args.output_dir / f"dl_project_snapshot_{timestamp}.zip").resolve()
    manifest_path = (args.output_dir / f"dl_project_snapshot_{timestamp}_manifest.json").resolve()
    max_file_bytes = int(args.max_file_mb * 1024 * 1024)

    candidates, included_roots, excluded_roots = collect_candidates()
    kept_files, skipped_files = filter_files(candidates, max_file_bytes=max_file_bytes)

    LOGGER.info("Included directories/files:")
    for item in included_roots + [str(path) for path in INCLUDE_FILES]:
        LOGGER.info("  + %s", item)

    LOGGER.info("Excluded roots:")
    for item in excluded_roots:
        LOGGER.info("  - %s", item)

    LOGGER.info("Creating archive: %s", archive_path)
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as zip_handle:
        for path in kept_files:
            arcname = path.relative_to(WORKSPACE_ROOT).as_posix()
            zip_handle.write(path, arcname=arcname)

    manifest = build_manifest(
        archive_path=archive_path,
        files=kept_files,
        skipped=skipped_files,
        included_roots=included_roots,
        excluded_roots=excluded_roots,
        max_file_mb=args.max_file_mb,
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    archive_size = archive_path.stat().st_size
    LOGGER.info("Archive created: %s", archive_path)
    LOGGER.info("Manifest created: %s", manifest_path)
    LOGGER.info("Included files: %d", len(kept_files))
    LOGGER.info("Skipped files: %d", len(skipped_files))
    LOGGER.info("Archive size: %s", human_size(archive_size))


if __name__ == "__main__":
    main()
