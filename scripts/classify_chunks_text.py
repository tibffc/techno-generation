# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import logging
import math
import re
import shutil
import sys
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from collections.abc import Iterable
from typing import Any


LOGGER = logging.getLogger("classify_chunks_text")
SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus"}
WORD_RE = re.compile(r"[\w']+", flags=re.UNICODE)


@dataclass(slots=True)
class SegmentStats:
    word_count: int
    text_chars: int
    duration: float
    no_speech_prob: float | None
    avg_logprob: float | None
    compression_ratio: float | None
    is_valid_text: bool


@dataclass(slots=True)
class ClassificationRecord:
    filepath: str
    duration: float | None
    text_detected: bool
    decision: str
    word_count: int
    text_chars: int
    valid_text_segments: int
    total_segments: int
    text_duration: float
    text_coverage: float
    speech_score: float
    language: str
    avg_logprob_mean: float | None
    no_speech_prob_min: float | None
    compression_ratio_max: float | None
    transcript: str
    error: str


class OpenAIWhisperBackend:
    def __init__(self, model_name: str, device: str) -> None:
        try:
            import whisper  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError(
                "Missing optional dependency: openai-whisper. "
                "Install it with: pip install openai-whisper"
            ) from error

        self.device = device
        self.model = whisper.load_model(model_name, device=device)

    def transcribe(self, path: Path, language: str | None, temperature: float) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "verbose": False,
            "temperature": temperature,
            "fp16": self.device == "cuda",
            "condition_on_previous_text": False,
        }
        if language:
            kwargs["language"] = language
        return self.model.transcribe(path.as_posix(), **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify prepared audio chunks by text/vocal presence with local Whisper ASR. "
            "Default mode only writes reports and filtered metadata; it does not move or delete audio."
        )
    )
    parser.add_argument("--input_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--metadata_in", type=Path, default=Path("data/processed/metadata.csv"))
    parser.add_argument("--output_csv", type=Path, default=Path("data/chunk_text_classification.csv"))
    parser.add_argument("--clean_metadata_out", type=Path, default=Path("data/processed/metadata_no_text.csv"))
    parser.add_argument("--rejected_dir", type=Path, default=Path("data/processed_with_text"))
    parser.add_argument("--action", choices=["report", "copy_text", "move_text"], default="report")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--extensions", nargs="+", default=sorted(SUPPORTED_EXTENSIONS))
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--include_errors", choices=["keep", "reject"], default="keep")
    parser.add_argument("--model", type=str, default="base")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--language", type=str, default=None, help="Optional Whisper language code, for example en or ru.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--min_words", type=int, default=4)
    parser.add_argument("--min_chars", type=int, default=12)
    parser.add_argument("--min_segment_words", type=int, default=2)
    parser.add_argument("--min_text_segments", type=int, default=1)
    parser.add_argument("--max_no_speech_prob", type=float, default=0.60)
    parser.add_argument("--min_avg_logprob", type=float, default=-1.20)
    parser.add_argument("--max_compression_ratio", type=float, default=2.40)
    parser.add_argument("--log_file", type=Path, default=Path("outputs/logs/classify_chunks_text.log"))
    return parser.parse_args()


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )


def resolve_device(device_arg: str) -> str:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "Missing dependency: torch. Install project requirements with: pip install -r requirements.txt"
        ) from error

    if device_arg == "cpu":
        return "cpu"
    if device_arg == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        LOGGER.warning("Requested CUDA, but torch.cuda.is_available() is False. Falling back to CPU.")
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def normalize_path_text(value: str) -> str:
    return value.replace("\\", "/")


def windows_path_to_wsl(path_text: str) -> Path | None:
    normalized = normalize_path_text(path_text)
    match = re.match(r"^(?P<drive>[A-Za-z]):/(?P<rest>.*)$", normalized)
    if match is None:
        return None
    return Path("/mnt") / match.group("drive").lower() / match.group("rest")


def resolve_audio_path(raw_path: str, input_dir: Path) -> Path:
    direct = Path(raw_path)
    if direct.exists():
        return direct

    wsl_path = windows_path_to_wsl(raw_path)
    if wsl_path is not None and wsl_path.exists():
        return wsl_path

    relative = Path(normalize_path_text(raw_path))
    if not relative.is_absolute():
        cwd_relative = Path.cwd() / relative
        if cwd_relative.exists():
            return cwd_relative

    basename_match = input_dir / Path(normalize_path_text(raw_path)).name
    if basename_match.exists():
        return basename_match

    return direct


def iter_audio_files(input_dir: Path, extensions: set[str], recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in extensions
    )


def read_metadata_rows(metadata_path: Path, input_dir: Path) -> tuple[list[dict[str, str]], list[Path]]:
    if not metadata_path.exists():
        return [], []

    with metadata_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        return rows, []
    if "filepath" not in rows[0]:
        raise ValueError(f"Metadata file has no filepath column: {metadata_path}")

    paths: list[Path] = []
    seen: set[str] = set()
    for row in rows:
        raw_filepath = row.get("filepath", "")
        if not raw_filepath:
            continue
        path = resolve_audio_path(raw_filepath, input_dir)
        key = path.resolve().as_posix() if path.exists() else path.as_posix()
        if key not in seen:
            paths.append(path)
            seen.add(key)
    return rows, paths


def count_words(text: str) -> int:
    return len([match.group(0) for match in WORD_RE.finditer(text) if any(char.isalnum() for char in match.group(0))])


def count_text_chars(text: str) -> int:
    return sum(1 for char in text if char.isalnum())


def clean_transcript(text: str) -> str:
    return " ".join(text.strip().split())


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def get_audio_duration(path: Path) -> float | None:
    try:
        import soundfile as sf
    except ImportError:
        return None

    try:
        info = sf.info(path)
    except Exception:  # noqa: BLE001
        return None
    if info.samplerate <= 0:
        return None
    return float(info.frames / info.samplerate)


def segment_duration(segment: dict[str, Any]) -> float:
    start = safe_float(segment.get("start"))
    end = safe_float(segment.get("end"))
    if start is None or end is None:
        return 0.0
    return max(0.0, end - start)


def classify_segment(segment: dict[str, Any], args: argparse.Namespace) -> SegmentStats:
    text = clean_transcript(str(segment.get("text", "")))
    word_count = count_words(text)
    text_chars = count_text_chars(text)
    no_speech_prob = safe_float(segment.get("no_speech_prob"))
    avg_logprob = safe_float(segment.get("avg_logprob"))
    compression_ratio = safe_float(segment.get("compression_ratio"))

    passes_no_speech = no_speech_prob is None or no_speech_prob <= args.max_no_speech_prob
    passes_logprob = avg_logprob is None or avg_logprob >= args.min_avg_logprob
    passes_compression = compression_ratio is None or compression_ratio <= args.max_compression_ratio
    is_valid_text = (
        word_count >= args.min_segment_words
        and passes_no_speech
        and passes_logprob
        and passes_compression
    )

    return SegmentStats(
        word_count=word_count,
        text_chars=text_chars,
        duration=segment_duration(segment),
        no_speech_prob=no_speech_prob,
        avg_logprob=avg_logprob,
        compression_ratio=compression_ratio,
        is_valid_text=is_valid_text,
    )


def mean_or_none(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return float(mean(clean))


def min_or_none(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return float(min(clean))


def max_or_none(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return float(max(clean))


def classify_file(path: Path, backend: OpenAIWhisperBackend, args: argparse.Namespace) -> ClassificationRecord:
    duration = get_audio_duration(path)

    if not path.exists():
        return ClassificationRecord(
            filepath=path.as_posix(),
            duration=duration,
            text_detected=False,
            decision="error",
            word_count=0,
            text_chars=0,
            valid_text_segments=0,
            total_segments=0,
            text_duration=0.0,
            text_coverage=0.0,
            speech_score=0.0,
            language="",
            avg_logprob_mean=None,
            no_speech_prob_min=None,
            compression_ratio_max=None,
            transcript="",
            error="file does not exist",
        )

    try:
        result = backend.transcribe(path, language=args.language, temperature=args.temperature)
    except Exception as error:  # noqa: BLE001
        LOGGER.warning("Failed to classify %s: %s", path, error)
        return ClassificationRecord(
            filepath=path.as_posix(),
            duration=duration,
            text_detected=False,
            decision="error",
            word_count=0,
            text_chars=0,
            valid_text_segments=0,
            total_segments=0,
            text_duration=0.0,
            text_coverage=0.0,
            speech_score=0.0,
            language="",
            avg_logprob_mean=None,
            no_speech_prob_min=None,
            compression_ratio_max=None,
            transcript="",
            error=str(error),
        )

    transcript = clean_transcript(str(result.get("text", "")))
    segments = [segment for segment in result.get("segments", []) if isinstance(segment, dict)]
    segment_stats = [classify_segment(segment, args) for segment in segments]
    valid_segments = [stats for stats in segment_stats if stats.is_valid_text]

    word_count = count_words(transcript)
    text_chars = count_text_chars(transcript)
    text_duration = float(sum(stats.duration for stats in valid_segments))
    total_duration = duration if duration is not None and duration > 0 else text_duration
    text_coverage = float(text_duration / total_duration) if total_duration > 0 else 0.0
    valid_no_speech = [stats.no_speech_prob for stats in valid_segments]
    speech_prob = [1.0 - value for value in valid_no_speech if value is not None]
    speech_score = float(max([text_coverage, *speech_prob], default=0.0))

    text_detected = (
        word_count >= args.min_words
        and text_chars >= args.min_chars
        and len(valid_segments) >= args.min_text_segments
    )

    return ClassificationRecord(
        filepath=path.as_posix(),
        duration=duration,
        text_detected=text_detected,
        decision="reject" if text_detected else "keep",
        word_count=word_count,
        text_chars=text_chars,
        valid_text_segments=len(valid_segments),
        total_segments=len(segment_stats),
        text_duration=round(text_duration, 4),
        text_coverage=round(text_coverage, 6),
        speech_score=round(speech_score, 6),
        language=str(result.get("language") or ""),
        avg_logprob_mean=mean_or_none([stats.avg_logprob for stats in segment_stats]),
        no_speech_prob_min=min_or_none([stats.no_speech_prob for stats in segment_stats]),
        compression_ratio_max=max_or_none([stats.compression_ratio for stats in segment_stats]),
        transcript=transcript,
        error="",
    )


def write_classification_csv(records: list[ClassificationRecord], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()) if records else [field for field in ClassificationRecord.__slots__])
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def progress(values: Iterable[Path], desc: str, unit: str) -> Iterable[Path]:
    try:
        from tqdm import tqdm
    except ImportError:
        return values
    return tqdm(values, desc=desc, unit=unit)


def result_key(path: Path) -> str:
    if path.exists():
        return path.resolve().as_posix()
    return path.as_posix()


def write_clean_metadata(
    metadata_rows: list[dict[str, str]],
    metadata_in: Path,
    clean_metadata_out: Path,
    input_dir: Path,
    records_by_path: dict[str, ClassificationRecord],
    include_errors: str,
) -> tuple[int, int]:
    if not metadata_rows:
        return 0, 0

    kept_rows: list[dict[str, str]] = []
    removed_rows = 0

    for row in metadata_rows:
        path = resolve_audio_path(row.get("filepath", ""), input_dir)
        record = records_by_path.get(result_key(path))

        should_remove = False
        if record is not None and record.text_detected:
            should_remove = True
        if record is not None and record.decision == "error" and include_errors == "reject":
            should_remove = True

        if should_remove:
            removed_rows += 1
        else:
            kept_rows.append(row)

    clean_metadata_out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(metadata_rows[0].keys())
    with clean_metadata_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    LOGGER.info(
        "Filtered metadata %s -> %s | kept=%d removed=%d",
        metadata_in,
        clean_metadata_out,
        len(kept_rows),
        removed_rows,
    )
    return len(kept_rows), removed_rows


def unique_destination(path: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    candidate = destination_dir / path.name
    if not candidate.exists():
        return candidate

    for index in range(1, 10_000):
        candidate = destination_dir / f"{path.stem}_{index:04d}{path.suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create unique destination for {path}")


def apply_audio_action(records: list[ClassificationRecord], action: str, rejected_dir: Path) -> int:
    if action == "report":
        return 0

    changed = 0
    for record in records:
        if not record.text_detected:
            continue
        source = Path(record.filepath)
        if not source.exists():
            LOGGER.warning("Cannot %s missing rejected chunk: %s", action, source)
            continue
        destination = unique_destination(source, rejected_dir)
        if action == "copy_text":
            shutil.copy2(source, destination)
        elif action == "move_text":
            shutil.move(source.as_posix(), destination.as_posix())
        else:
            raise ValueError(f"Unsupported action: {action}")
        changed += 1
    return changed


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file)
    start_time = time.perf_counter()

    extensions = {extension.lower() if extension.startswith(".") else f".{extension.lower()}" for extension in args.extensions}
    metadata_rows, metadata_paths = read_metadata_rows(args.metadata_in, args.input_dir)
    if metadata_paths:
        audio_files = metadata_paths
        LOGGER.info("Loaded %d chunk paths from metadata: %s", len(audio_files), args.metadata_in)
    else:
        audio_files = iter_audio_files(args.input_dir, extensions=extensions, recursive=args.recursive)
        LOGGER.info("Found %d chunk files in %s", len(audio_files), args.input_dir)

    if args.max_files is not None:
        audio_files = audio_files[: args.max_files]

    if not audio_files:
        raise FileNotFoundError(f"No audio chunks found in {args.input_dir} or {args.metadata_in}")

    device = resolve_device(args.device)
    LOGGER.info("Loading Whisper model '%s' on %s", args.model, device)
    backend = OpenAIWhisperBackend(args.model, device=device)

    records: list[ClassificationRecord] = []
    for path in progress(audio_files, desc="Classifying chunks", unit="file"):
        record = classify_file(path, backend, args)
        records.append(record)
        LOGGER.info(
            "%s | decision=%s words=%d valid_segments=%d transcript=%r",
            path,
            record.decision,
            record.word_count,
            record.valid_text_segments,
            record.transcript[:120],
        )

    write_classification_csv(records, args.output_csv)

    records_by_path = {result_key(Path(record.filepath)): record for record in records}
    kept_rows, removed_rows = write_clean_metadata(
        metadata_rows=metadata_rows,
        metadata_in=args.metadata_in,
        clean_metadata_out=args.clean_metadata_out,
        input_dir=args.input_dir,
        records_by_path=records_by_path,
        include_errors=args.include_errors,
    )

    changed_files = apply_audio_action(records, action=args.action, rejected_dir=args.rejected_dir)

    rejected = sum(1 for record in records if record.text_detected)
    errors = sum(1 for record in records if record.decision == "error")
    elapsed = max(time.perf_counter() - start_time, 1e-6)
    LOGGER.info("Saved classification report to %s", args.output_csv)
    if metadata_rows:
        LOGGER.info("Saved clean metadata to %s | kept=%d removed=%d", args.clean_metadata_out, kept_rows, removed_rows)
    LOGGER.info(
        "Finished text classification in %.2fs | files=%d rejected=%d errors=%d changed_files=%d",
        elapsed,
        len(records),
        rejected,
        errors,
        changed_files,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001
        LOGGER.error("%s", error)
        sys.exit(1)
