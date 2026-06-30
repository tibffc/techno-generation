# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pydub import AudioSegment
from tqdm import tqdm


LOGGER = logging.getLogger("clean_dataset")
SUPPORTED_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".flac",
    ".m4a",
    ".aac",
    ".ogg",
    ".aiff",
    ".aif",
    ".opus",
}


@dataclass(slots=True)
class CleaningReport:
    total_files_found: int = 0
    candidate_audio_files: int = 0
    valid_audio_files: int = 0
    duplicates_removed: int = 0
    corrupted_files: int = 0
    skipped_small_files: int = 0
    copied_files: int = 0
    elapsed_seconds: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast dataset ingestion and validation for audio collections.")
    parser.add_argument("--input_dir", type=Path, default=Path("dirty_data"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/raw_clean"))
    parser.add_argument("--report_path", type=Path, default=Path("data/cleaning_report.json"))
    parser.add_argument("--copy_mode", choices=["copy", "hardlink"], default="copy")
    parser.add_argument("--verify_mode", choices=["none", "ffprobe", "decode"], default="ffprobe")
    parser.add_argument("--reuse_existing_output", action="store_true")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--min_size_kb", type=int, default=100)
    parser.add_argument("--hash_algo", choices=["sha1", "md5"], default="sha1")
    parser.add_argument("--log_file", type=Path, default=Path("outputs/logs/clean_dataset.log"))
    return parser.parse_args()


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )


def iter_files(input_dir: Path) -> Iterable[Path]:
    for path in input_dir.rglob("*"):
        if path.is_file():
            yield path


def safe_stem(name: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name).strip("_") or "track"


def read_chunked_hash(path: Path, algo: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.new(algo)
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_with_ffprobe(path: Path) -> bool:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        path.as_posix(),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        return False
    try:
        payload = json.loads(result.stdout or "{}")
        duration = float(payload.get("format", {}).get("duration", 0.0))
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return duration > 0.0


def verify_with_decode(path: Path) -> bool:
    try:
        AudioSegment.from_file(path)
    except Exception:  # noqa: BLE001
        return False
    return True


def validate_audio(path: Path, verify_mode: str) -> bool:
    if verify_mode == "none":
        return True
    if verify_mode == "ffprobe":
        return verify_with_ffprobe(path)
    if verify_mode == "decode":
        return verify_with_decode(path)
    raise ValueError(f"Unsupported verify_mode: {verify_mode}")


def ensure_unique_target(output_dir: Path, source_path: Path) -> Path:
    base_name = safe_stem(source_path.stem)
    suffix = source_path.suffix.lower()
    candidate = output_dir / f"{base_name}{suffix}"
    index = 1
    while candidate.exists():
        candidate = output_dir / f"{base_name}_{index:04d}{suffix}"
        index += 1
    return candidate


def materialize_file(source_path: Path, target_path: Path, copy_mode: str) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if copy_mode == "copy":
        shutil.copy2(source_path, target_path)
        return
    if copy_mode == "hardlink":
        os.link(source_path, target_path)
        return
    raise ValueError(f"Unsupported copy_mode: {copy_mode}")


def write_report(
    report_path: Path,
    input_dir: Path,
    output_dir: Path,
    report: CleaningReport,
    copy_mode: str,
    verify_mode: str,
    reuse_existing_output: bool,
    min_size_kb: int,
    max_files: int | None,
    hash_algo: str,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_dir": input_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "copy_mode": copy_mode,
        "verify_mode": verify_mode,
        "reuse_existing_output": reuse_existing_output,
        "min_size_kb": min_size_kb,
        "max_files": max_files,
        "hash_algo": hash_algo,
        **asdict(report),
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file)
    start_time = time.perf_counter()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = CleaningReport()
    min_size_bytes = args.min_size_kb * 1024
    existing_output_files = sorted(
        [path for path in args.output_dir.glob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS]
    )

    if args.reuse_existing_output and existing_output_files:
        report.copied_files = len(existing_output_files)
        report.valid_audio_files = len(existing_output_files)
        report.elapsed_seconds = round(time.perf_counter() - start_time, 3)
        write_report(
            report_path=args.report_path,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            report=report,
            copy_mode=args.copy_mode,
            verify_mode=args.verify_mode,
            reuse_existing_output=args.reuse_existing_output,
            min_size_kb=args.min_size_kb,
            max_files=args.max_files,
            hash_algo=args.hash_algo,
        )
        LOGGER.info(
            "Reused %d existing files in %s without rescanning or recopying source audio.",
            len(existing_output_files),
            args.output_dir,
        )
        return

    LOGGER.info("Scanning files in %s", args.input_dir)
    all_files = list(tqdm(iter_files(args.input_dir), desc="Scanning files", unit="file"))
    report.total_files_found = len(all_files)

    candidate_files: list[Path] = []
    for path in tqdm(all_files, desc="Filtering candidates", unit="file"):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            size_bytes = path.stat().st_size
        except OSError as error:
            LOGGER.warning("Failed to stat %s: %s", path, error)
            report.corrupted_files += 1
            continue
        if size_bytes < min_size_bytes:
            report.skipped_small_files += 1
            continue
        candidate_files.append(path)
        if args.max_files is not None and len(candidate_files) >= args.max_files:
            break

    report.candidate_audio_files = len(candidate_files)
    LOGGER.info("Candidate audio files after extension/size filtering: %d", report.candidate_audio_files)

    valid_files: list[Path] = []
    for path in tqdm(candidate_files, desc="Validating audio", unit="file"):
        if validate_audio(path, args.verify_mode):
            valid_files.append(path)
        else:
            report.corrupted_files += 1
            LOGGER.warning("Validation failed for %s", path)

    report.valid_audio_files = len(valid_files)
    LOGGER.info("Valid audio files after %s verification: %d", args.verify_mode, report.valid_audio_files)

    size_groups: dict[int, list[Path]] = {}
    for path in valid_files:
        try:
            size_groups.setdefault(path.stat().st_size, []).append(path)
        except OSError as error:
            LOGGER.warning("Failed to stat during dedupe %s: %s", path, error)
            report.corrupted_files += 1

    unique_files: list[Path] = []
    duplicates_removed = 0
    for group in tqdm(size_groups.values(), desc="Deduplicating", unit="group"):
        if len(group) == 1:
            unique_files.append(group[0])
            continue
        seen_hashes: set[str] = set()
        for path in group:
            try:
                file_hash = read_chunked_hash(path, args.hash_algo)
            except OSError as error:
                LOGGER.warning("Hashing failed for %s: %s", path, error)
                report.corrupted_files += 1
                continue
            if file_hash in seen_hashes:
                duplicates_removed += 1
                continue
            seen_hashes.add(file_hash)
            unique_files.append(path)

    report.duplicates_removed = duplicates_removed
    LOGGER.info("Unique files after deduplication: %d", len(unique_files))

    copied_files = 0
    for path in tqdm(unique_files, desc="Copying to raw_clean", unit="file"):
        target_path = ensure_unique_target(args.output_dir, path)
        try:
            materialize_file(path, target_path, args.copy_mode)
            copied_files += 1
        except OSError as error:
            LOGGER.warning("Failed to materialize %s -> %s: %s", path, target_path, error)
            report.corrupted_files += 1

    report.copied_files = copied_files
    report.elapsed_seconds = round(time.perf_counter() - start_time, 3)

    write_report(
        report_path=args.report_path,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        report=report,
        copy_mode=args.copy_mode,
        verify_mode=args.verify_mode,
        reuse_existing_output=args.reuse_existing_output,
        min_size_kb=args.min_size_kb,
        max_files=args.max_files,
        hash_algo=args.hash_algo,
    )

    LOGGER.info("Cleaning report saved to %s", args.report_path)
    LOGGER.info("Cleaning finished: %s", report)


if __name__ == "__main__":
    main()
