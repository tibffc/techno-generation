# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import ast
import csv
import difflib
import json
import logging
import math
import re
import shutil
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("build_final_prompt_tags")
KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
AMBIGUOUS_BPM_SOURCE_INDICES = {685, 1834, 4084, 4106, 4140, 4172, 6195, 6858, 6890, 6954}

SUPPORTED_AUDIO_EXTENSIONS = {
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

SPOTIFY_NUMERIC_COLUMNS = [
    "trackFeatureAcousticness",
    "trackFeatureDanceability",
    "trackFeatureEnergy",
    "trackFeatureInstrumentalness",
    "trackFeatureLiveness",
    "trackFeatureLoudness",
    "trackFeatureSpeechiness",
    "trackFeatureTempo",
    "trackFeatureValence",
    "trackFeatureKey",
    "trackFeatureMode",
    "trackFeatureTimeSignature",
    "artistPopularity",
    "artistFollowers",
    "trackPopularity",
    "albumPopularity",
    "trackDuration",
]

LIBROSA_BASE_FEATURES = [
    "librosa_spectral_centroid_mean",
    "librosa_spectral_centroid_std",
    "librosa_spectral_rolloff_mean",
    "librosa_spectral_rolloff_std",
    "librosa_spectral_bandwidth_mean",
    "librosa_spectral_bandwidth_std",
    "librosa_spectral_contrast_mean",
    "librosa_spectral_contrast_std",
    "librosa_spectral_flatness_mean",
    "librosa_spectral_flatness_std",
    "librosa_zero_crossing_rate_mean",
    "librosa_zero_crossing_rate_std",
    "librosa_onset_strength_mean",
    "librosa_onset_strength_std",
    "librosa_onset_density",
    "librosa_onset_rate",
]

MFCC_FEATURES = [
    *(f"librosa_mfcc_{index}_mean" for index in range(1, 14)),
    *(f"librosa_mfcc_{index}_std" for index in range(1, 14)),
]

CHROMA_FEATURES = [
    *(f"librosa_chroma_{index}_mean" for index in range(1, 13)),
    *(f"librosa_chroma_{index}_std" for index in range(1, 13)),
]

GENRE_PRIORITY = [
    ("industrial techno", "industrial techno"),
    ("hard techno", "hard techno"),
    ("minimal techno", "minimal techno"),
    ("dub techno", "dub techno"),
    ("acid techno", "acid techno"),
    ("melodic techno", "melodic techno"),
    ("deep techno", "deep techno"),
]

CLAP_CANDIDATE_TAGS = [
    "acid bassline",
    "acid techno",
    "aggressive electronic music",
    "ambient techno",
    "atmospheric electronic music",
    "bright electronic music",
    "bright melodic techno",
    "calm electronic track",
    "clean electronic texture",
    "dark club track",
    "dark melodic techno",
    "dark techno",
    "deep bass groove",
    "deep club track",
    "deep house influenced techno",
    "deep melodic techno",
    "deep techno",
    "dense drum groove",
    "distorted electronic texture",
    "downtempo electronic music",
    "driving bass groove",
    "driving club track",
    "driving techno",
    "dub techno",
    "electronica",
    "energetic electronic music",
    "grainy electronic texture",
    "groovy techno",
    "hard techno",
    "high-energy electronic track",
    "hypnotic techno",
    "industrial techno",
    "leftfield electronic music",
    "melodic techno",
    "melodic house influenced techno",
    "minimal techno",
    "noisy electronic texture",
    "peak-time techno",
    "progressive techno",
    "punchy drum groove",
    "rave techno",
    "raw electronic music",
    "rolling drum groove",
    "smooth techno",
    "soft drum groove",
    "soft techno groove",
    "sparse minimal rhythm",
    "steady club groove",
    "steady drum groove",
    "steady electronic track",
    "sub bass groove",
    "textured electronic music",
    "warehouse techno",
    "warm bass groove",
    "warm electronic music",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build final ACE-Step/LoRA prompt tags for a techno dataset. "
            "Supports Spotify speechiness prefiltering before audio preprocessing and "
            "final metadata/tagging after chunking and vocal chunk removal."
        )
    )
    parser.add_argument("--mode", choices=["prefilter", "build_metadata", "all"], default="all")
    parser.add_argument("--spotify_csv", type=Path, default=Path("[Skiley] techno - Skiley Export.csv"))
    parser.add_argument("--audio_dir", type=Path, default=Path("dirty_data"))
    parser.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--processed_metadata_csv", type=Path, default=None)
    parser.add_argument("--vocal_classification_csv", type=Path, default=None)
    parser.add_argument("--filtered_spotify_csv", type=Path, default=Path("data/spotify_filtered_speechiness.csv"))
    parser.add_argument("--filtered_audio_manifest", type=Path, default=Path("data/filtered_audio_manifest.csv"))
    parser.add_argument("--matched_tracks_csv", type=Path, default=Path("data/matched_tracks.csv"))
    parser.add_argument("--copy_filtered_audio_dir", type=Path, default=None)
    parser.add_argument("--output_csv", type=Path, default=Path("data/metadata_final_prompt_tags.csv"))
    parser.add_argument("--features_csv", type=Path, default=Path("data/metadata_final_features.csv"))
    parser.add_argument("--report_json", type=Path, default=Path("data/final_prompt_tags_report.json"))
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--speechiness_threshold", type=float, default=0.1)
    parser.add_argument("--keep_missing_speechiness", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--match_threshold", type=float, default=0.74)
    parser.add_argument("--clap_top_k", type=int, default=3)
    parser.add_argument("--clap_model", type=str, default="laion/clap-htsat-unfused")
    parser.add_argument("--disable_clap", action="store_true")
    parser.add_argument("--ollama_model", type=str, default="qwen3:8b")
    parser.add_argument("--ollama_url", type=str, default="http://localhost:11434/api/generate")
    parser.add_argument("--disable_ollama", action="store_true")
    parser.add_argument("--split_train", type=float, default=0.90)
    parser.add_argument("--split_val", type=float, default=0.05)
    parser.add_argument("--log_file", type=Path, default=Path("outputs/logs/build_final_prompt_tags.log"))
    return parser.parse_args()


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )


def require_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError("Missing dependency: pandas. Install project requirements with: pip install -r requirements.txt") from error
    return pd


def progress(values: Iterable[Any], desc: str, unit: str) -> Iterable[Any]:
    try:
        from tqdm import tqdm
    except ImportError:
        return values
    return tqdm(values, desc=desc, unit=unit)


def read_spotify_csv(path: Path) -> Any:
    pd = require_pandas()
    if not path.exists():
        raise FileNotFoundError(f"Spotify CSV does not exist: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "source_index" not in frame.columns:
        frame.insert(0, "source_index", range(len(frame)))
    frame["source_index"] = pd.to_numeric(frame["source_index"], errors="coerce").astype("Int64")
    return frame


def write_csv(frame: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def safe_stem(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value).strip("_") or "track"


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text or text == "nan":
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[_\-./]+", " ", text)
    text = re.sub(r"[^0-9a-zа-яё]+", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def strip_leading_number(value: str) -> str:
    return re.sub(r"^\s*\d+\s+", "", value).strip()


def normalize_path_key(path: Any) -> str:
    if path is None:
        return ""
    return str(path).replace("\\", "/").strip().lower()


def path_keys(path: Path) -> set[str]:
    stem = path.stem
    normalized_stem = normalize_text(stem)
    stripped_stem = strip_leading_number(normalized_stem)
    safe = safe_stem(stem)
    return {
        normalize_path_key(path.as_posix()),
        normalize_path_key(path.name),
        normalize_path_key(path.stem),
        normalize_text(path.name),
        normalized_stem,
        stripped_stem,
        safe.lower(),
        normalize_text(safe),
    }


def source_chunk_stem(path: Path) -> str:
    match = re.match(r"(?P<stem>.+)_chunk_\d+$", path.stem, flags=re.IGNORECASE)
    if match:
        return match.group("stem")
    return path.stem


def iter_audio_files(audio_dir: Path) -> list[Path]:
    if not audio_dir.exists():
        LOGGER.warning("Audio directory does not exist: %s", audio_dir)
        return []
    return sorted(
        path
        for path in audio_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )


def filter_bpm_issues(df: Any) -> tuple[Any, dict[str, Any]]:
    pd = require_pandas()
    working = df.copy()
    if "source_index" not in working.columns:
        working.insert(0, "source_index", range(len(working)))
    working["source_index"] = pd.to_numeric(working["source_index"], errors="coerce").astype("Int64")
    if "trackFeatureTempo" not in working.columns:
        working["trackFeatureTempo"] = pd.NA
    tempo = pd.to_numeric(working["trackFeatureTempo"], errors="coerce")
    working["trackFeatureTempo"] = tempo

    ambiguous_mask = working["source_index"].isin(AMBIGUOUS_BPM_SOURCE_INDICES)
    bpm_outlier_mask = tempo > 200
    remove_mask = ambiguous_mask | bpm_outlier_mask
    filtered = working.loc[~remove_mask].copy()
    report = {
        "removed_ambiguous_bpm_tracks_count": int(ambiguous_mask.sum()),
        "removed_ambiguous_bpm_source_indices": [int(value) for value in working.loc[ambiguous_mask, "source_index"].dropna().tolist()],
        "removed_bpm_outliers_count": int(bpm_outlier_mask.sum()),
        "removed_bpm_outlier_source_indices": [int(value) for value in working.loc[bpm_outlier_mask, "source_index"].dropna().tolist()],
        "rows_after_bpm_filter": int(len(filtered)),
    }
    return filtered, report


def filter_by_speechiness(df: Any, threshold: float = 0.1, keep_missing: bool = True) -> tuple[Any, dict[str, Any]]:
    pd = require_pandas()
    working = df.copy()
    if "trackFeatureSpeechiness" not in working.columns:
        working["trackFeatureSpeechiness"] = pd.NA
        missing_column = True
    else:
        missing_column = False

    speechiness = pd.to_numeric(working["trackFeatureSpeechiness"], errors="coerce")
    working["trackFeatureSpeechiness"] = speechiness
    missing_mask = speechiness.isna()
    pass_mask = speechiness < threshold
    if keep_missing:
        keep_mask = pass_mask | missing_mask
    else:
        keep_mask = pass_mask

    filtered = working.loc[keep_mask].copy()
    filtered["speechiness_filter_passed"] = filtered["trackFeatureSpeechiness"].lt(threshold)
    if keep_missing:
        filtered.loc[filtered["trackFeatureSpeechiness"].isna(), "speechiness_filter_passed"] = True

    report = {
        "input_rows": int(len(working)),
        "rows_after_speechiness_filter": int(len(filtered)),
        "removed_by_speechiness_count": int((speechiness >= threshold).sum()),
        "missing_speechiness_count": int(missing_mask.sum()),
        "speechiness_threshold": float(threshold),
        "keep_missing_speechiness": bool(keep_missing),
        "trackFeatureSpeechiness_column_missing": bool(missing_column),
    }
    return filtered, report


def make_audio_index(audio_files: list[Path]) -> list[dict[str, Any]]:
    index: list[dict[str, Any]] = []
    for path in audio_files:
        stem_norm = normalize_text(path.stem)
        stripped_stem = strip_leading_number(stem_norm)
        index.append(
            {
                "path": path,
                "stem_norm": stem_norm,
                "stripped_stem": stripped_stem,
                "safe_stem": safe_stem(path.stem).lower(),
                "keys": path_keys(path),
            }
        )
    return index


def compact_uri(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text == "nan":
        return ""
    return text.rsplit(":", 1)[-1].rsplit("/", 1)[-1].strip()


def ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return difflib.SequenceMatcher(None, left, right).ratio()


def best_audio_match(row: Any, audio_index: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    track_name = row.get("trackName", "")
    artist_name = row.get("artistName", "")
    track_norm = normalize_text(track_name)
    artist_norm = normalize_text(artist_name)
    artist_track = normalize_text(f"{artist_name} {track_name}")
    track_artist = normalize_text(f"{track_name} {artist_name}")
    isrc = normalize_text(row.get("trackIsrc", ""))
    uri = normalize_text(compact_uri(row.get("trackUri", "")))

    best: dict[str, Any] = {
        "matched_audio_filepath": "",
        "match_score": 0.0,
        "match_method": "unmatched",
        "match_query": "",
    }

    for item in audio_index:
        path = item["path"]
        stem_norm = item["stem_norm"]
        stripped_stem = item["stripped_stem"]
        keys = item["keys"]

        candidates: list[tuple[float, str, str]] = []
        if uri and any(uri in key for key in keys):
            candidates.append((1.0, "trackUri", uri))
        if isrc and any(isrc in key for key in keys):
            candidates.append((1.0, "trackIsrc", isrc))
        if track_norm:
            if track_norm == stripped_stem or track_norm == stem_norm:
                candidates.append((0.98, "trackName_exact", track_norm))
            if track_norm in stripped_stem or track_norm in stem_norm:
                candidates.append((0.94, "trackName_contains", track_norm))
            candidates.append((ratio(track_norm, stripped_stem), "trackName_fuzzy", track_norm))
        if artist_track:
            candidates.append((ratio(artist_track, stripped_stem), "artist_track_fuzzy", artist_track))
        if track_artist:
            candidates.append((ratio(track_artist, stripped_stem), "track_artist_fuzzy", track_artist))
        if artist_norm and track_norm and artist_norm in stem_norm and track_norm in stem_norm:
            candidates.append((0.96, "artist_and_track_contains", track_artist))

        if not candidates:
            continue
        score, method, query = max(candidates, key=lambda value: value[0])
        if score > best["match_score"]:
            best = {
                "matched_audio_filepath": path.as_posix(),
                "match_score": round(float(score), 6),
                "match_method": method,
                "match_query": query,
            }

    if best["match_score"] < threshold:
        best["matched_audio_filepath"] = ""
        best["match_method"] = "unmatched"
    return best


def match_spotify_to_audio(filtered_df: Any, audio_dir: Path, threshold: float, max_files: int | None) -> tuple[Any, dict[str, Any]]:
    pd = require_pandas()
    audio_files = iter_audio_files(audio_dir)
    audio_index = make_audio_index(audio_files)

    rows: list[dict[str, Any]] = []
    missing_audio: list[dict[str, Any]] = []
    iterable = filtered_df.head(max_files).iterrows() if max_files is not None else filtered_df.iterrows()
    for _, row in progress(iterable, desc="Matching Spotify rows to audio", unit="row"):
        payload = row.to_dict()
        match = best_audio_match(payload, audio_index, threshold=threshold)
        payload.update(match)
        matched_path = match["matched_audio_filepath"]
        if matched_path:
            path = Path(matched_path)
            payload["matched_audio_filename"] = path.name
            payload["matched_audio_stem"] = path.stem
            payload["matched_audio_safe_stem"] = safe_stem(path.stem)
        else:
            payload["matched_audio_filename"] = ""
            payload["matched_audio_stem"] = ""
            payload["matched_audio_safe_stem"] = ""
            missing_audio.append(
                {
                    "trackName": str(payload.get("trackName", "")),
                    "artistName": str(payload.get("artistName", "")),
                    "trackUri": str(payload.get("trackUri", "")),
                    "best_score": match["match_score"],
                }
            )
            LOGGER.warning("No local audio match for %s - %s", payload.get("artistName", ""), payload.get("trackName", ""))
        rows.append(payload)

    matched = pd.DataFrame(rows)
    report = {
        "audio_dir": audio_dir.as_posix(),
        "audio_files_scanned": int(len(audio_files)),
        "matched_audio_files": int((matched["matched_audio_filepath"].fillna("") != "").sum()) if len(matched) else 0,
        "missing_audio_files": int(len(missing_audio)),
        "missing_audio": missing_audio[:200],
    }
    return matched, report


def write_filtered_audio_manifest(matched_df: Any, path: Path) -> None:
    pd = require_pandas()
    columns = [
        "matched_audio_filepath",
        "source_index",
        "trackName",
        "artistName",
        "trackUri",
        "trackIsrc",
        "trackFeatureSpeechiness",
        "match_score",
        "match_method",
    ]
    if "matched_audio_filepath" not in matched_df.columns:
        manifest = pd.DataFrame(columns=["filepath", *[column for column in columns if column != "matched_audio_filepath"]])
        write_csv(manifest, path)
        return
    manifest = matched_df.loc[matched_df["matched_audio_filepath"].fillna("") != "", [col for col in columns if col in matched_df.columns]].copy()
    manifest.rename(columns={"matched_audio_filepath": "filepath"}, inplace=True)
    write_csv(manifest, path)


def copy_filtered_audio(matched_df: Any, destination_dir: Path) -> int:
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for raw_path in matched_df.get("matched_audio_filepath", []):
        if not raw_path:
            continue
        source = Path(raw_path)
        if not source.exists():
            LOGGER.warning("Cannot copy missing filtered audio: %s", source)
            continue
        destination = destination_dir / source.name
        if destination.exists():
            stem = destination.stem
            for index in range(1, 10_000):
                candidate = destination_dir / f"{stem}_{index:04d}{destination.suffix}"
                if not candidate.exists():
                    destination = candidate
                    break
        shutil.copy2(source, destination)
        copied += 1
    return copied


def run_prefilter(args: argparse.Namespace, report: dict[str, Any]) -> Any:
    spotify_df = read_spotify_csv(args.spotify_csv)
    filtered_df, speech_report = filter_by_speechiness(
        spotify_df,
        threshold=args.speechiness_threshold,
        keep_missing=args.keep_missing_speechiness,
    )
    filtered_df, bpm_report = filter_bpm_issues(filtered_df)
    write_csv(filtered_df, args.filtered_spotify_csv)
    matched_df, match_report = match_spotify_to_audio(
        filtered_df=filtered_df,
        audio_dir=args.audio_dir,
        threshold=args.match_threshold,
        max_files=args.max_files,
    )
    write_csv(matched_df, args.matched_tracks_csv)
    write_filtered_audio_manifest(matched_df, args.filtered_audio_manifest)

    copied_count = 0
    if args.copy_filtered_audio_dir is not None:
        copied_count = copy_filtered_audio(matched_df, args.copy_filtered_audio_dir)

    report.update(speech_report)
    report.update(bpm_report)
    report.update(match_report)
    report["filtered_spotify_csv"] = args.filtered_spotify_csv.as_posix()
    report["filtered_audio_manifest"] = args.filtered_audio_manifest.as_posix()
    report["matched_tracks_csv"] = args.matched_tracks_csv.as_posix()
    report["copied_filtered_audio_files"] = int(copied_count)
    LOGGER.info("Saved filtered Spotify CSV to %s", args.filtered_spotify_csv)
    LOGGER.info("Saved filtered audio manifest to %s", args.filtered_audio_manifest)
    LOGGER.info("Saved matched tracks CSV to %s", args.matched_tracks_csv)
    return matched_df


def read_processed_metadata(processed_dir: Path, processed_metadata_csv: Path | None, max_files: int | None) -> Any:
    pd = require_pandas()
    metadata_path = processed_metadata_csv or processed_dir / "metadata.csv"
    if metadata_path.exists():
        frame = pd.read_csv(metadata_path, encoding="utf-8-sig")
        if max_files is not None:
            frame = frame.head(max_files).copy()
        return frame

    audio_files = sorted(
        path for path in processed_dir.glob("*") if path.is_file() and path.suffix.lower() in {".wav", ".flac", ".mp3"}
    )
    if max_files is not None:
        audio_files = audio_files[:max_files]
    return pd.DataFrame(
        {
            "filepath": [path.as_posix() for path in audio_files],
            "source_file": ["" for _ in audio_files],
        }
    )


def read_vocal_classification(path: Path | None) -> tuple[dict[str, bool], dict[str, Any]]:
    if path is None or not path.exists():
        return {}, {"vocal_classification_csv": None, "chunks_removed_by_vocal_classifier": 0}

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    rejected: dict[str, bool] = {}
    for row in rows:
        filepath = row.get("filepath", "")
        if not filepath:
            continue
        text_detected = str(row.get("text_detected", "")).strip().lower() in {"true", "1", "yes"}
        decision = str(row.get("decision", "")).strip().lower()
        should_reject = text_detected or decision == "reject"
        for key in path_keys(Path(filepath)):
            rejected[key] = should_reject
    return rejected, {"vocal_classification_csv": path.as_posix(), "vocal_classification_rows": len(rows)}


def build_matched_lookup(matched_df: Any) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    if "matched_audio_filepath" not in matched_df.columns:
        return lookup
    for _, row in matched_df.iterrows():
        payload = row.to_dict()
        matched_path = str(payload.get("matched_audio_filepath", "") or "")
        if not matched_path:
            continue
        raw_path = Path(matched_path)
        keys = set(path_keys(raw_path))
        safe = str(payload.get("matched_audio_safe_stem", "") or "")
        if safe:
            keys.add(safe.lower())
            keys.add(normalize_text(safe))
        for key in keys:
            if key:
                lookup[key] = payload
    return lookup


def find_spotify_for_chunk(chunk_row: dict[str, Any], matched_lookup: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    candidates: set[str] = set()
    filepath = str(chunk_row.get("filepath", "") or "")
    source_file = str(chunk_row.get("source_file", "") or "")

    if source_file:
        source_path = Path(source_file)
        candidates.update(path_keys(source_path))
        candidates.add(source_path.stem.lower())
        candidates.add(normalize_text(source_path.stem))
    if filepath:
        chunk_base = source_chunk_stem(Path(filepath))
        candidates.add(chunk_base.lower())
        candidates.add(normalize_text(chunk_base))

    for key in candidates:
        if key in matched_lookup:
            return matched_lookup[key]
    return None


def is_chunk_rejected_by_vocal(filepath: str, rejected_lookup: dict[str, bool]) -> bool:
    for key in path_keys(Path(filepath)):
        if rejected_lookup.get(key, False):
            return True
    return False


def coerce_spotify_numeric(frame: Any) -> Any:
    pd = require_pandas()
    output = frame.copy()
    for column in SPOTIFY_NUMERIC_COLUMNS:
        if column not in output.columns:
            output[column] = pd.NA
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def nan_to_none(value: Any) -> Any:
    try:
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
    except TypeError:
        return value
    return value


def finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def compute_librosa_features(path: Path, sample_rate: int) -> dict[str, float]:
    import librosa
    import numpy as np

    y, sr = librosa.load(path.as_posix(), sr=sample_rate, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    if duration <= 0:
        raise ValueError("audio duration is zero")

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_strength, sr=sr, units="frames")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    features: dict[str, float] = {
        "librosa_duration": float(duration),
        "librosa_spectral_centroid_mean": float(np.mean(spectral_centroid)),
        "librosa_spectral_centroid_std": float(np.std(spectral_centroid)),
        "librosa_spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
        "librosa_spectral_rolloff_std": float(np.std(spectral_rolloff)),
        "librosa_spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
        "librosa_spectral_bandwidth_std": float(np.std(spectral_bandwidth)),
        "librosa_spectral_contrast_mean": float(np.mean(spectral_contrast)),
        "librosa_spectral_contrast_std": float(np.std(spectral_contrast)),
        "librosa_spectral_flatness_mean": float(np.mean(spectral_flatness)),
        "librosa_spectral_flatness_std": float(np.std(spectral_flatness)),
        "librosa_zero_crossing_rate_mean": float(np.mean(zero_crossing_rate)),
        "librosa_zero_crossing_rate_std": float(np.std(zero_crossing_rate)),
        "librosa_onset_strength_mean": float(np.mean(onset_strength)),
        "librosa_onset_strength_std": float(np.std(onset_strength)),
        "librosa_onset_density": float(len(onset_frames) / duration),
        "librosa_onset_rate": float(len(onset_frames) / duration * 60.0),
    }
    for index in range(13):
        features[f"librosa_mfcc_{index + 1}_mean"] = float(np.mean(mfcc[index]))
        features[f"librosa_mfcc_{index + 1}_std"] = float(np.std(mfcc[index]))
    for index in range(12):
        features[f"librosa_chroma_{index + 1}_mean"] = float(np.mean(chroma[index]))
        features[f"librosa_chroma_{index + 1}_std"] = float(np.std(chroma[index]))
    return features


def build_chunk_feature_rows(
    processed_df: Any,
    matched_df: Any,
    vocal_rejected_lookup: dict[str, bool],
    sample_rate: int,
    speechiness_threshold: float,
) -> tuple[Any, dict[str, Any]]:
    pd = require_pandas()
    matched_lookup = build_matched_lookup(matched_df)
    rows: list[dict[str, Any]] = []
    failed_files: list[dict[str, str]] = []
    unmatched_chunks: list[str] = []
    vocal_removed = 0

    for _, chunk_row in progress(processed_df.iterrows(), desc="Extracting librosa features", unit="chunk"):
        chunk = chunk_row.to_dict()
        filepath = str(chunk.get("filepath", "") or "")
        if not filepath:
            failed_files.append({"filepath": "", "error": "missing filepath"})
            continue
        if is_chunk_rejected_by_vocal(filepath, vocal_rejected_lookup):
            vocal_removed += 1
            continue

        spotify_row = find_spotify_for_chunk(chunk, matched_lookup)
        if spotify_row is None:
            unmatched_chunks.append(filepath)
            LOGGER.warning("No matched Spotify metadata for processed chunk: %s", filepath)
            continue

        path = Path(filepath)
        if not path.exists():
            failed_files.append({"filepath": filepath, "error": "chunk file does not exist"})
            LOGGER.warning("Chunk file does not exist: %s", filepath)
            continue

        try:
            librosa_features = compute_librosa_features(path, sample_rate=sample_rate)
        except Exception as error:  # noqa: BLE001
            failed_files.append({"filepath": filepath, "error": str(error)})
            LOGGER.warning("Librosa feature extraction failed for %s: %s", filepath, error)
            continue

        row: dict[str, Any] = {}
        row.update({f"spotify_{key}": value for key, value in spotify_row.items()})
        for key, value in spotify_row.items():
            if key not in row:
                row[key] = value
        row.update(chunk)
        row.update(librosa_features)
        speechiness = finite_float(row.get("trackFeatureSpeechiness"), default=None)
        row["speechiness_filter_passed"] = bool(
            speechiness is None
            or speechiness < speechiness_threshold
        )
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame = coerce_spotify_numeric(frame) if len(frame) else frame
    report = {
        "processed_chunks": int(len(processed_df)),
        "chunks_removed_by_vocal_classifier": int(vocal_removed),
        "unmatched_processed_chunks": int(len(unmatched_chunks)),
        "unmatched_processed_chunk_examples": unmatched_chunks[:200],
        "failed_librosa_extractions": int(len(failed_files)),
        "failed_files": failed_files[:200],
        "final_chunks_after_feature_extraction": int(len(frame)),
    }
    return frame, report


def fill_numeric_for_tags(frame: Any, columns: list[str]) -> Any:
    output = frame.copy()
    for column in columns:
        if column not in output.columns:
            output[column] = math.nan
        median = output[column].median(skipna=True)
        median_value = finite_float(median, default=None)
        if median_value is None:
            median = 0.0
        else:
            median = median_value
        output[f"{column}_filled"] = output[column].fillna(median)
    return output



def energy_tag(value: float) -> str:
    if value < 0.45:
        return "low energy"
    if value < 0.70:
        return "medium energy"
    return "high energy"


def mood_tag(value: float) -> str:
    if value < 0.35:
        return "dark mood"
    if value < 0.65:
        return "neutral mood"
    return "bright mood"


def groove_tag(value: float) -> str:
    if value >= 0.75:
        return "club dance groove"
    if value >= 0.55:
        return "steady groove"
    return "less danceable groove"


def instrumentalness_tag(value: float) -> str:
    if value >= 0.70:
        return "instrumental track"
    if value >= 0.30:
        return "mostly instrumental texture"
    return "sampled vocal texture"


def quantile_tag(value: float, q33: float, q66: float, low: str, mid: str, high: str) -> str:
    if value <= q33:
        return low
    if value <= q66:
        return mid
    return high


def parse_artist_genres(value: Any) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []

    parsed: Any = None
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
    if isinstance(parsed, list):
        values = [str(item).strip() for item in parsed]
    else:
        values = re.split(r"[,;|]", text)
    return [normalize_text(item) for item in values if normalize_text(item)]


def spotify_genre_tags(value: Any) -> list[str]:
    genres = parse_artist_genres(value)
    joined = " | ".join(genres)
    tags: list[str] = []
    for needle, tag in GENRE_PRIORITY:
        if needle in joined and tag not in tags:
            tags.append(tag)
    if ("raw techno" in joined or "underground techno" in joined or "raw" in joined) and "raw underground techno" not in tags:
        tags.append("raw underground techno")
    if not tags and any("techno" == genre or genre.endswith(" techno") or "techno" in genre for genre in genres):
        tags.append("techno")
    return tags[:2]


def assign_rule_tags(frame: Any) -> Any:
    output = fill_numeric_for_tags(
        frame,
        [
            "trackFeatureEnergy",
            "trackFeatureValence",
            "trackFeatureDanceability",
            "trackFeatureInstrumentalness",
            "librosa_spectral_centroid_mean",
            "librosa_spectral_rolloff_mean",
            "librosa_spectral_flatness_mean",
            "librosa_zero_crossing_rate_mean",
            "librosa_spectral_contrast_mean",
            "librosa_onset_density",
            "librosa_onset_strength_mean",
        ],
    )

    output["energy_tag"] = output["trackFeatureEnergy_filled"].apply(energy_tag)
    output["mood_tag"] = output["trackFeatureValence_filled"].apply(mood_tag)
    output["groove_tag"] = output["trackFeatureDanceability_filled"].apply(groove_tag)
    output["instrumentalness_tag"] = output["trackFeatureInstrumentalness_filled"].apply(instrumentalness_tag)

    centroid_q33 = output["librosa_spectral_centroid_mean_filled"].quantile(0.33)
    centroid_q66 = output["librosa_spectral_centroid_mean_filled"].quantile(0.66)
    onset_q33 = output["librosa_onset_density_filled"].quantile(0.33)
    onset_q66 = output["librosa_onset_density_filled"].quantile(0.66)
    punch_q33 = output["librosa_onset_strength_mean_filled"].quantile(0.33)
    punch_q66 = output["librosa_onset_strength_mean_filled"].quantile(0.66)
    flatness_q33 = output["librosa_spectral_flatness_mean_filled"].quantile(0.33)
    flatness_q66 = output["librosa_spectral_flatness_mean_filled"].quantile(0.66)
    zcr_q33 = output["librosa_zero_crossing_rate_mean_filled"].quantile(0.33)
    zcr_q66 = output["librosa_zero_crossing_rate_mean_filled"].quantile(0.66)

    output["brightness_tag"] = output["librosa_spectral_centroid_mean_filled"].apply(
        lambda value: quantile_tag(value, centroid_q33, centroid_q66, "dark sound", "neutral brightness", "bright sound")
    )
    output["rhythm_density_tag"] = output["librosa_onset_density_filled"].apply(
        lambda value: quantile_tag(value, onset_q33, onset_q66, "sparse rhythm", "medium density rhythm", "dense rhythm")
    )
    output["punch_tag"] = output["librosa_onset_strength_mean_filled"].apply(
        lambda value: quantile_tag(value, punch_q33, punch_q66, "soft drum attack", "balanced drum attack", "punchy drums")
    )

    def texture(row: Any) -> str:
        flatness = row["librosa_spectral_flatness_mean_filled"]
        zcr = row["librosa_zero_crossing_rate_mean_filled"]
        if flatness >= flatness_q66 or zcr >= zcr_q66:
            return "noisy texture"
        if flatness <= flatness_q33 and zcr <= zcr_q33:
            return "clean texture"
        return "balanced texture"

    output["texture_tag"] = output.apply(texture, axis=1)
    output["spotify_genre_tags"] = output["artistGenres"].apply(lambda value: "|".join(spotify_genre_tags(value))) if "artistGenres" in output.columns else ""
    return output



def extract_clap_embedding(output: Any) -> Any:
    """Convert different HuggingFace CLAP outputs to a plain tensor."""
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    if hasattr(output, "text_embeds"):
        return output.text_embeds
    if hasattr(output, "audio_embeds"):
        return output.audio_embeds
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state.mean(dim=1)
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    return output


class ClapTagger:
    def __init__(self, model_name: str, top_k: int, disabled: bool = False) -> None:
        self.top_k = max(0, int(top_k))
        self.model_name = model_name
        self.available = False
        self.error = ""
        self.device = "cpu"
        self.model = None
        self.processor = None
        self.text_features = None
        self.torch = None
        self.librosa = None
        if disabled or self.top_k <= 0:
            self.error = "disabled"
            return
        try:
            import torch
            import librosa
            from transformers import ClapModel, ClapProcessor
        except ImportError as error:
            self.error = f"missing CLAP dependencies: {error}"
            LOGGER.warning("CLAP semantic tagging disabled: %s", self.error)
            return
        try:
            self.torch = torch
            self.librosa = librosa
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.processor = ClapProcessor.from_pretrained(model_name)
            self.model = ClapModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            text_inputs = self.processor(text=CLAP_CANDIDATE_TAGS, padding=True, return_tensors="pt")
            text_inputs = {key: value.to(self.device) for key, value in text_inputs.items()}
            with torch.no_grad():
                text_features = self.model.get_text_features(**text_inputs)
            text_features = extract_clap_embedding(text_features)
            self.text_features = torch.nn.functional.normalize(text_features, dim=-1)
            self.available = True
            LOGGER.info("Loaded CLAP model %s on %s", model_name, self.device)
        except Exception as error:  # noqa: BLE001
            self.error = str(error)
            LOGGER.warning("CLAP semantic tagging disabled: %s", self.error)

    def top_tags(self, path: Path) -> list[str]:
        if not self.available or self.top_k <= 0:
            return []
        assert self.torch is not None
        assert self.librosa is not None
        assert self.processor is not None
        assert self.model is not None
        assert self.text_features is not None
        try:
            audio, sample_rate = self.librosa.load(path.as_posix(), sr=48000, mono=True)
            inputs = self.processor(audio=audio, sampling_rate=sample_rate, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self.torch.no_grad():
                audio_features = self.model.get_audio_features(**inputs)
            audio_features = extract_clap_embedding(audio_features)
            audio_features = self.torch.nn.functional.normalize(audio_features, dim=-1)
            scores = (audio_features @ self.text_features.T).squeeze(0)
            top_count = min(self.top_k, len(CLAP_CANDIDATE_TAGS))
            indices = self.torch.topk(scores, k=top_count).indices.detach().cpu().tolist()
            return [CLAP_CANDIDATE_TAGS[int(index)] for index in indices]
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("CLAP tagging failed for %s: %s", path, error)
            return []


def assign_clap_tags(frame: Any, args: argparse.Namespace, report: dict[str, Any]) -> Any:
    output = frame.copy()
    tagger = ClapTagger(model_name=args.clap_model, top_k=args.clap_top_k, disabled=args.disable_clap)
    report["clap_model"] = args.clap_model
    report["clap_top_k"] = int(args.clap_top_k)
    report["clap_available"] = bool(tagger.available)
    report["clap_error"] = tagger.error
    tags_by_row: list[str] = []
    for filepath in progress(output["filepath"].fillna("").tolist(), desc="Selecting CLAP semantic tags", unit="chunk"):
        tags = tagger.top_tags(Path(str(filepath))) if filepath else []
        tags_by_row.append("|".join(tags[: args.clap_top_k]))
    output["clap_tags"] = tags_by_row
    return output


def format_keyscale(key_value: Any, mode_value: Any) -> str:
    key = finite_float(key_value, default=None)
    mode = finite_float(mode_value, default=None)
    if key is None or int(key) < 0 or int(key) >= len(KEY_NAMES):
        return ""
    scale = "major" if int(mode or 0) == 1 else "minor"
    return f"{KEY_NAMES[int(key)]} {scale}"


def format_timesignature(value: Any) -> str:
    time_signature = finite_float(value, default=None)
    if time_signature is None:
        return ""
    return str(int(time_signature))


def clean_caption(text: str) -> str:
    cleaned = " ".join(str(text).strip().strip('"').strip("'").split())
    if not cleaned:
        return ""
    cleaned = re.sub(r"\b\d+(?:\.\d+)?\s*bpm\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(key|keyscale|time signature|timesignature)\b[^,.]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+([,.])", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,")
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def fallback_caption(tags: list[str]) -> str:
    descriptors = [tag for tag in tags if tag not in {"techno music", "electronic track"}]
    descriptors = descriptors[:8]
    if not descriptors:
        return "A techno track with a steady electronic groove."
    if len(descriptors) == 1:
        detail = descriptors[0]
    else:
        detail = ", ".join(descriptors[:-1]) + f", and {descriptors[-1]}"
    return f"A techno track with {detail}."


def ollama_caption(tags: list[str], args: argparse.Namespace) -> str | None:
    if args.disable_ollama:
        return None
    prompt = (
        "Write one concise English sentence for a music training caption. "
        "Use only the supplied tags as evidence. Do not mention BPM, tempo numbers, key, scale, "
        "time signature, lyrics, vocals, language, metadata, or JSON. Avoid comma-separated tag lists. "
        "Return only the caption sentence.\n\n"
        f"Tags: {', '.join(tags)}"
    )
    payload = json.dumps({"model": args.ollama_model, "prompt": prompt, "stream": False}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        args.ollama_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as error:
        LOGGER.warning("Ollama caption generation failed: %s", error)
        return None
    caption = clean_caption(str(data.get("response", "")))
    return caption or None


def build_caption_from_tags(tags: list[str], args: argparse.Namespace, cache: dict[str, str]) -> str:
    key = "|".join(tags)
    if key in cache:
        return cache[key]
    caption = ollama_caption(tags, args) or fallback_caption(tags)
    caption = clean_caption(caption) or fallback_caption(tags)
    cache[key] = caption
    return caption


def build_chunk_json(row: Any) -> str:
    payload = {
        "caption": str(row.get("caption", "")),
        "bpm": nan_to_none(finite_float(row.get("bpm"), default=None)),
        "keyscale": str(row.get("keyscale", "")),
        "timesignature": str(row.get("timesignature", "")),
    }
    return json.dumps(payload, ensure_ascii=False)


def dedupe_tags(tags: list[str], max_tags: int = 14) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        clean = " ".join(str(tag).strip().split())
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        result.append(clean)
        seen.add(key)
        if len(result) >= max_tags:
            break
    return result


def build_tags_for_row(row: Any) -> list[str]:
    genre_tags = [tag for tag in str(row.get("spotify_genre_tags", "") or "").split("|") if tag]
    clap_tags = [tag for tag in str(row.get("clap_tags", "") or "").split("|") if tag]
    return dedupe_tags(
        [
            "techno music",
            "electronic track",
            row.get("energy_tag", ""),
            row.get("mood_tag", ""),
            row.get("groove_tag", ""),
            row.get("instrumentalness_tag", ""),
            row.get("brightness_tag", ""),
            row.get("rhythm_density_tag", ""),
            row.get("texture_tag", ""),
            row.get("punch_tag", ""),
            *genre_tags,
            *clap_tags[:3],
        ],
        max_tags=14,
    )


def assign_splits(frame: Any, seed: int, train_ratio: float, val_ratio: float) -> Any:
    import numpy as np

    split = ["test"] * len(frame)
    indices = np.arange(len(frame))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    train_end = int(len(indices) * train_ratio)
    val_end = train_end + int(len(indices) * val_ratio)
    for index in indices[:train_end]:
        split[int(index)] = "train"
    for index in indices[train_end:val_end]:
        split[int(index)] = "val"
    return split


def finalize_prompts(frame: Any, args: argparse.Namespace, report: dict[str, Any]) -> Any:
    output = assign_rule_tags(frame)
    output = assign_clap_tags(output, args=args, report=report)
    output["bpm"] = output["trackFeatureTempo"]
    output["keyscale"] = output.apply(lambda row: format_keyscale(row.get("trackFeatureKey"), row.get("trackFeatureMode")), axis=1)
    output["timesignature"] = output["trackFeatureTimeSignature"].apply(format_timesignature) if "trackFeatureTimeSignature" in output.columns else ""
    output["prompt_generic"] = "techno music, electronic music"

    caption_cache: dict[str, str] = {}
    final_tags: list[str] = []
    captions: list[str] = []
    for _, row in progress(output.iterrows(), desc="Generating final captions", unit="chunk"):
        tags = build_tags_for_row(row)
        final_tags.append("|".join(tags))
        captions.append(build_caption_from_tags(tags, args=args, cache=caption_cache))

    output["caption"] = captions
    output["prompt_features"] = output["caption"]
    output["final_tags"] = final_tags
    output["chunk_json"] = output.apply(build_chunk_json, axis=1)
    output["split"] = assign_splits(output, seed=args.seed, train_ratio=args.split_train, val_ratio=args.split_val)
    report["ollama_model"] = args.ollama_model
    report["ollama_disabled"] = bool(args.disable_ollama)
    report["caption_cache_entries"] = int(len(caption_cache))
    return output


def value_counts_dict(frame: Any, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    return {str(key): int(value) for key, value in frame[column].value_counts(dropna=False).items()}


def build_report_counts(frame: Any) -> dict[str, Any]:
    tag_columns = [
        "energy_tag",
        "mood_tag",
        "groove_tag",
        "instrumentalness_tag",
        "brightness_tag",
        "texture_tag",
        "rhythm_density_tag",
        "punch_tag",
        "spotify_genre_tags",
        "clap_tags",
    ]
    top_prompts = {}
    if "prompt_features" in frame.columns:
        top_prompts = {str(key): int(value) for key, value in frame["prompt_features"].value_counts().head(30).items()}
    return {
        "tag_counts": {column: value_counts_dict(frame, column) for column in tag_columns},
        "unique_prompt_features_count": int(frame["prompt_features"].nunique()) if "prompt_features" in frame.columns else 0,
        "clap_tag_counts": value_counts_dict(frame, "clap_tags"),
        "top_30_prompt_features": top_prompts,
    }


def output_metadata_columns(frame: Any) -> list[str]:
    minimal = [
        "filepath",
        "source_file",
        "source_index",
        "duration",
        "bpm",
        "keyscale",
        "timesignature",
        "caption",
        "chunk_json",
        "trackName",
        "artistName",
        "trackUri",
        "trackFeatureTempo",
        "trackFeatureEnergy",
        "trackFeatureDanceability",
        "trackFeatureValence",
        "trackFeatureInstrumentalness",
        "trackFeatureSpeechiness",
        "trackFeatureKey",
        "trackFeatureMode",
        "trackFeatureTimeSignature",
        "speechiness_filter_passed",
        "energy_tag",
        "mood_tag",
        "groove_tag",
        "instrumentalness_tag",
        "brightness_tag",
        "texture_tag",
        "rhythm_density_tag",
        "punch_tag",
        "spotify_genre_tags",
        "clap_tags",
        "prompt_generic",
        "prompt_features",
        "final_tags",
        "split",
    ]
    return [column for column in minimal if column in frame.columns]


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def load_or_create_matched_tracks(args: argparse.Namespace, report: dict[str, Any]) -> Any:
    if args.matched_tracks_csv.exists():
        pd = require_pandas()
        matched_df = pd.read_csv(args.matched_tracks_csv, encoding="utf-8-sig")
        report["matched_tracks_csv_reused"] = args.matched_tracks_csv.as_posix()
        if "rows_after_speechiness_filter" not in report:
            report["rows_after_speechiness_filter"] = int(len(matched_df))
        if "matched_audio_files" not in report and "matched_audio_filepath" in matched_df.columns:
            matched_mask = matched_df["matched_audio_filepath"].fillna("") != ""
            report["matched_audio_files"] = int(matched_mask.sum())
            report["missing_audio_files"] = int((~matched_mask).sum())
        if "input_rows" not in report and args.spotify_csv.exists():
            try:
                report["input_rows"] = int(len(read_spotify_csv(args.spotify_csv)))
            except Exception as error:  # noqa: BLE001
                LOGGER.warning("Could not read Spotify CSV for report counts: %s", error)
        return matched_df
    return run_prefilter(args, report)


def run_build_metadata(args: argparse.Namespace, report: dict[str, Any]) -> Any:
    matched_df = load_or_create_matched_tracks(args, report)
    processed_df = read_processed_metadata(args.processed_dir, args.processed_metadata_csv, max_files=args.max_files)
    vocal_lookup, vocal_report = read_vocal_classification(args.vocal_classification_csv)
    report.update(vocal_report)

    feature_df, feature_report = build_chunk_feature_rows(
        processed_df=processed_df,
        matched_df=matched_df,
        vocal_rejected_lookup=vocal_lookup,
        sample_rate=args.sample_rate,
        speechiness_threshold=args.speechiness_threshold,
    )
    report.update(feature_report)
    if feature_df.empty:
        raise RuntimeError("No final chunk rows were built. Check audio matching, processed metadata, and vocal filtering.")

    final_df = finalize_prompts(feature_df, args, report=report)
    write_csv(final_df, args.features_csv)
    metadata_df = final_df[output_metadata_columns(final_df)].copy()
    write_csv(metadata_df, args.output_csv)
    report.update(build_report_counts(final_df))
    report["features_csv"] = args.features_csv.as_posix()
    report["output_csv"] = args.output_csv.as_posix()
    return final_df


def print_quality_summary(report: dict[str, Any], final_df: Any | None) -> None:
    print("\nFinal prompt tags quality summary")
    print(f"input Spotify rows: {report.get('input_rows', 0)}")
    print(f"rows after speechiness filter: {report.get('rows_after_speechiness_filter', 0)}")
    print(f"matched audio files: {report.get('matched_audio_files', 0)}")
    print(f"processed chunks: {report.get('processed_chunks', 0)}")
    print(f"chunks removed by vocal classifier: {report.get('chunks_removed_by_vocal_classifier', 0)}")
    print(f"final chunks: {report.get('final_chunks_after_feature_extraction', 0)}")
    print(f"unique prompt_features: {report.get('unique_prompt_features_count', 0)}")
    if final_df is None or final_df.empty:
        return
    for column in ["energy_tag", "mood_tag", "brightness_tag", "rhythm_density_tag", "texture_tag", "clap_tags"]:
        if column in final_df.columns:
            print(f"\n{column}:")
            for key, value in final_df[column].value_counts().head(10).items():
                print(f"  {key}: {value}")
    print("\nprompt_features examples:")
    for prompt in final_df["prompt_features"].drop_duplicates().head(5):
        print(f"  - {prompt}")


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file)
    started = time.perf_counter()
    report: dict[str, Any] = {
        "mode": args.mode,
        "spotify_csv": args.spotify_csv.as_posix(),
        "sample_rate": args.sample_rate,
        "clap_top_k": int(args.clap_top_k),
        "clap_model": args.clap_model,
        "ollama_model": args.ollama_model,
        "seed": int(args.seed),
    }

    final_df = None
    if args.mode in {"prefilter", "all"}:
        run_prefilter(args, report)
    if args.mode in {"build_metadata", "all"}:
        final_df = run_build_metadata(args, report)

    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    write_report(args.report_json, report)
    LOGGER.info("Saved report to %s", args.report_json)
    print_quality_summary(report, final_df)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001
        LOGGER.error("%s", error)
        sys.exit(1)
