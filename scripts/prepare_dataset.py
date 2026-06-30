# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from tqdm import tqdm


LOGGER = logging.getLogger("prepare_dataset")
SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".aiff", ".aif", ".opus"}


@dataclass(slots=True)
class ChunkRecord:
    filepath: str
    source_file: str
    chunk_index: int
    start_seconds: float
    end_seconds: float
    duration: float
    sample_rate: int
    channels: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare chunked dataset for music model training.")
    parser.add_argument("--input_dir", type=Path, default=Path("data/raw_mp3"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--chunk_seconds", type=float, default=30.0)
    parser.add_argument("--overlap_seconds", type=float, default=5.0)
    parser.add_argument("--sample_rate", type=int, default=44100)
    parser.add_argument("--min_duration_seconds", type=float, default=30.0)
    parser.add_argument("--target_lufs", type=float, default=-14.0)
    parser.add_argument("--audio_format", type=str, default="flac", choices=["flac", "wav"])
    parser.add_argument("--metadata_path", type=Path, default=None)
    parser.add_argument("--reuse_existing_chunks", action="store_true")
    parser.add_argument("--log_file", type=Path, default=Path("outputs/logs/prepare_dataset.log"))
    return parser.parse_args()


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )


def iter_audio_files(input_dir: Path) -> Iterable[Path]:
    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    audio, _ = librosa.load(path.as_posix(), sr=sample_rate, mono=False)
    if audio.ndim == 1:
        audio = np.vstack([audio, audio])
    return audio.astype(np.float32)


def normalize_loudness(audio: np.ndarray, sample_rate: int, target_lufs: float) -> np.ndarray:
    meter = pyln.Meter(sample_rate)
    transposed = audio.T
    loudness = meter.integrated_loudness(transposed)
    normalized = pyln.normalize.loudness(transposed, loudness, target_lufs)
    normalized = np.clip(normalized, -1.0, 1.0)
    return normalized.T.astype(np.float32)


def chunk_audio(audio: np.ndarray, sample_rate: int, chunk_seconds: float, overlap_seconds: float) -> list[tuple[int, int]]:
    chunk_size = int(chunk_seconds * sample_rate)
    overlap_size = int(overlap_seconds * sample_rate)
    hop_size = chunk_size - overlap_size
    if hop_size <= 0:
        raise ValueError("overlap_seconds must be smaller than chunk_seconds")
    total_samples = audio.shape[1]
    if total_samples < chunk_size:
        return []
    chunks: list[tuple[int, int]] = []
    start = 0
    while start + chunk_size <= total_samples:
        chunks.append((start, start + chunk_size))
        start += hop_size
    return chunks


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value).strip("_") or "chunk"


def write_metadata(records: list[ChunkRecord], metadata_path: Path) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "filepath",
                "source_file",
                "chunk_index",
                "start_seconds",
                "end_seconds",
                "duration",
                "sample_rate",
                "channels",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def parse_chunk_filename(path: Path) -> tuple[str, int] | None:
    match = re.match(r"(?P<stem>.+)_chunk_(?P<index>\d+)\.(wav|flac)$", path.name, re.IGNORECASE)
    if match is None:
        return None
    return match.group("stem"), int(match.group("index"))


def rebuild_metadata_from_existing_chunks(
    output_dir: Path,
    metadata_path: Path,
    sample_rate: int,
    chunk_seconds: float,
    overlap_seconds: float,
) -> int:
    chunk_files = sorted([path for path in output_dir.glob("*") if path.suffix.lower() in {".wav", ".flac"}])
    if not chunk_files:
        return 0

    hop_seconds = chunk_seconds - overlap_seconds
    records: list[ChunkRecord] = []
    for chunk_path in chunk_files:
        parsed = parse_chunk_filename(chunk_path)
        if parsed is None:
            LOGGER.warning("Skipping chunk with unexpected filename format: %s", chunk_path)
            continue

        source_stem, chunk_index = parsed
        start_seconds = chunk_index * hop_seconds
        end_seconds = start_seconds + chunk_seconds
        records.append(
            ChunkRecord(
                filepath=chunk_path.as_posix(),
                source_file=(Path("data/raw_clean") / f"{source_stem}.mp3").as_posix(),
                chunk_index=chunk_index,
                start_seconds=round(start_seconds, 4),
                end_seconds=round(end_seconds, 4),
                duration=round(chunk_seconds, 4),
                sample_rate=sample_rate,
                channels=2,
            )
        )

    write_metadata(records, metadata_path)
    return len(records)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.metadata_path or args.output_dir / "metadata.csv"
    existing_chunks = sorted([path for path in args.output_dir.glob("*") if path.suffix.lower() in {".wav", ".flac"}])

    if args.reuse_existing_chunks and existing_chunks:
        restored_count = rebuild_metadata_from_existing_chunks(
            output_dir=args.output_dir,
            metadata_path=metadata_path,
            sample_rate=args.sample_rate,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=args.overlap_seconds,
        )
        LOGGER.info(
            "Reused %d existing chunks in %s and rebuilt metadata at %s without reprocessing audio.",
            restored_count,
            args.output_dir,
            metadata_path,
        )
        return

    records: list[ChunkRecord] = []
    audio_files = list(iter_audio_files(args.input_dir))
    LOGGER.info("Found %d candidate audio files in %s", len(audio_files), args.input_dir)

    for source_path in tqdm(audio_files, desc="Preparing dataset"):
        try:
            audio = load_audio(source_path, args.sample_rate)
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("Failed to decode %s: %s", source_path, error)
            continue

        duration = audio.shape[1] / args.sample_rate
        if duration < args.min_duration_seconds:
            LOGGER.info("Skipping short file %s (%.2fs)", source_path, duration)
            continue

        try:
            audio = normalize_loudness(audio, args.sample_rate, args.target_lufs)
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("Loudness normalization failed for %s: %s", source_path, error)
            continue

        chunk_ranges = chunk_audio(audio, args.sample_rate, args.chunk_seconds, args.overlap_seconds)
        if not chunk_ranges:
            LOGGER.info("No valid chunks for %s", source_path)
            continue

        for chunk_index, (start, end) in enumerate(chunk_ranges):
            chunk = audio[:, start:end].T
            start_seconds = start / args.sample_rate
            end_seconds = end / args.sample_rate
            output_name = f"{safe_name(source_path.stem)}_chunk_{chunk_index:04d}.{args.audio_format}"
            output_path = args.output_dir / output_name

            try:
                sf.write(output_path, chunk, args.sample_rate, format=args.audio_format.upper())
            except Exception as error:  # noqa: BLE001
                LOGGER.warning("Failed to save chunk %s: %s", output_path, error)
                continue

            records.append(
                ChunkRecord(
                    filepath=output_path.as_posix(),
                    source_file=source_path.as_posix(),
                    chunk_index=chunk_index,
                    start_seconds=round(start_seconds, 4),
                    end_seconds=round(end_seconds, 4),
                    duration=round(end_seconds - start_seconds, 4),
                    sample_rate=args.sample_rate,
                    channels=2,
                )
            )

    write_metadata(records, metadata_path)
    LOGGER.info("Prepared %d chunks. Metadata saved to %s", len(records), metadata_path)


if __name__ == "__main__":
    main()
