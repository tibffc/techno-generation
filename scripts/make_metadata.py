# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


LOGGER = logging.getLogger("make_metadata")
KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build self-labeled metadata from processed audio chunks.")
    parser.add_argument("--input_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--metadata_out", type=Path, default=Path("data/metadata.csv"))
    parser.add_argument("--features_out", type=Path, default=Path("data/metadata_features.csv"))
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generic_prompt", type=str, default="techno music, electronic music")
    parser.add_argument("--split_train", type=float, default=0.90)
    parser.add_argument("--split_val", type=float, default=0.05)
    parser.add_argument("--split_test", type=float, default=0.05)
    parser.add_argument("--tempo_min", type=float, default=110.0)
    parser.add_argument("--tempo_max", type=float, default=150.0)
    parser.add_argument("--allow_half_double_fix", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--quality_report_out", type=Path, default=Path("data/metadata_quality_report.json"))
    parser.add_argument("--log_file", type=Path, default=Path("outputs/logs/make_metadata.log"))
    return parser.parse_args()


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        LOGGER.warning("Requested CUDA, but torch.cuda.is_available() is False. Falling back to CPU.")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def estimate_key(chroma: np.ndarray) -> str:
    chroma_mean = chroma.mean(axis=1)
    key_index = int(np.argmax(chroma_mean))
    return KEY_NAMES[key_index]


def parse_bool_flag(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Unsupported boolean flag value: {value}")


def normalize_bpm_for_techno(bpm_raw: float, tempo_min: float = 110.0, tempo_max: float = 150.0) -> float:
    if pd.isna(bpm_raw) or bpm_raw <= 0:
        return float("nan")

    center = (tempo_min + tempo_max) / 2.0
    candidates = [float(bpm_raw), float(bpm_raw) * 2.0, float(bpm_raw) / 2.0]
    valid_candidates = [candidate for candidate in candidates if tempo_min <= candidate <= tempo_max]
    if valid_candidates:
        return min(valid_candidates, key=lambda candidate: abs(candidate - center))
    return float(bpm_raw)


def hz_to_torch_bins(sample_rate: int, n_fft: int, device: torch.device) -> torch.Tensor:
    return torch.linspace(0.0, sample_rate / 2.0, steps=(n_fft // 2) + 1, device=device)


def build_chroma_filter(freqs_hz: torch.Tensor) -> torch.Tensor:
    chroma_filter = torch.zeros((12, freqs_hz.numel()), device=freqs_hz.device, dtype=torch.float32)
    valid = freqs_hz > 0
    valid_freqs = freqs_hz[valid]
    midi = 69.0 + 12.0 * torch.log2(valid_freqs / 440.0)
    pitch_class = torch.remainder(torch.round(midi).long(), 12)
    valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
    chroma_filter[pitch_class, valid_indices] = 1.0
    return chroma_filter


def compute_spectral_features(
    batch_waveforms: torch.Tensor,
    sample_rate: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    n_fft = 2048
    hop_length = 512
    win_length = 2048
    eps = 1e-8

    window = torch.hann_window(win_length, device=device)
    stft = torch.stft(
        batch_waveforms,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    magnitude = stft.abs()
    power = magnitude.square()
    num_bins = power.shape[1]
    num_frames = power.shape[2]

    freqs = hz_to_torch_bins(sample_rate, n_fft, device).view(1, -1, 1)
    power_sum = power.sum(dim=1, keepdim=True).clamp_min(eps)
    centroid_frames = (power * freqs).sum(dim=1) / power_sum.squeeze(1)
    bandwidth_frames = torch.sqrt(
        ((power * (freqs - centroid_frames.unsqueeze(1)).square()).sum(dim=1) / power_sum.squeeze(1)).clamp_min(eps)
    )

    rms_frames = torch.sqrt(power.mean(dim=1).clamp_min(eps))
    rms_mean = rms_frames.mean(dim=1)
    loudness_proxy = 20.0 * torch.log10(batch_waveforms.square().mean(dim=1).sqrt().clamp_min(eps))

    rolloff_threshold = power.sum(dim=1) * 0.85
    cumulative_power = power.cumsum(dim=1)
    rolloff_indices = (cumulative_power >= rolloff_threshold.unsqueeze(1)).float().argmax(dim=1)
    rolloff_frames = freqs.squeeze(0).squeeze(-1)[rolloff_indices]

    centered = batch_waveforms - batch_waveforms.mean(dim=1, keepdim=True)
    zero_cross = centered[:, 1:] * centered[:, :-1] < 0
    zero_cross_rate = zero_cross.float().mean(dim=1)

    flux = torch.relu(magnitude[:, :, 1:] - magnitude[:, :, :-1]).mean(dim=1)
    onset_strength = flux.mean(dim=1)
    onset_threshold = flux.mean(dim=1, keepdim=True) + flux.std(dim=1, keepdim=True)
    onset_events = (flux > onset_threshold).float()
    durations = torch.full(
        (batch_waveforms.shape[0],),
        batch_waveforms.shape[1] / sample_rate,
        dtype=batch_waveforms.dtype,
        device=device,
    )
    onset_density = onset_events.sum(dim=1) / durations.clamp_min(1e-6)

    frame_rate = sample_rate / hop_length
    bpm_candidates = torch.arange(60, 201, device=device, dtype=batch_waveforms.dtype)
    lag_candidates = torch.clamp((60.0 * frame_rate / bpm_candidates).round().long(), min=1)
    lag_scores: list[torch.Tensor] = []
    for lag in lag_candidates.tolist():
        if lag >= flux.shape[1]:
            lag_scores.append(torch.zeros(batch_waveforms.shape[0], device=device, dtype=batch_waveforms.dtype))
            continue
        score = (flux[:, :-lag] * flux[:, lag:]).mean(dim=1)
        lag_scores.append(score)
    lag_scores_tensor = torch.stack(lag_scores, dim=1)
    bpm_estimate = bpm_candidates[lag_scores_tensor.argmax(dim=1)]

    chroma_filter_tensor = build_chroma_filter(freqs.squeeze(0).squeeze(-1)).to(dtype=power.dtype)
    chroma_energy = torch.einsum("cf,bft->bct", chroma_filter_tensor, power)
    chroma_mean = chroma_energy.mean(dim=2)
    key_index = chroma_mean.argmax(dim=1)

    return {
        "duration": durations,
        "bpm": bpm_estimate,
        "rms_energy": rms_mean,
        "loudness_proxy": loudness_proxy,
        "spectral_centroid": centroid_frames.mean(dim=1),
        "spectral_bandwidth": bandwidth_frames.mean(dim=1),
        "spectral_rolloff": rolloff_frames.float().mean(dim=1),
        "zero_crossing_rate": zero_cross_rate,
        "onset_strength": onset_strength,
        "onset_density": onset_density,
        "chroma_mean": chroma_mean,
        "key_index": key_index,
    }


def safe_load_audio(path: Path, sample_rate: int) -> tuple[torch.Tensor, int]:
    audio_np, sr = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio_np.T.copy())

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        sr = sample_rate
    return waveform.squeeze(0), sr


def inspect_num_frames(path: Path, sample_rate: int) -> int:
    info_fn = getattr(torchaudio, "info", None)
    if callable(info_fn):
        info = info_fn(path)
        num_frames = int(info.num_frames)
        src_rate = int(info.sample_rate)
        if src_rate != sample_rate and src_rate > 0:
            duration_seconds = num_frames / src_rate
            return int(round(duration_seconds * sample_rate))
        return num_frames

    # Fallback for environments where torchaudio.info is unavailable.
    sf_info = sf.info(path)
    num_frames = int(sf_info.frames)
    src_rate = int(sf_info.samplerate)
    if src_rate != sample_rate and src_rate > 0:
        duration_seconds = num_frames / src_rate
        return int(round(duration_seconds * sample_rate))
    return num_frames


def process_batch_on_device(
    file_batch: list[Path],
    sample_rate: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    waveforms: list[torch.Tensor] = []

    for path in file_batch:
        waveform, _ = safe_load_audio(path, sample_rate)
        waveforms.append(waveform)

    batch = torch.stack(waveforms, dim=0).to(device)
    gpu_features = compute_spectral_features(batch, sample_rate, device)
    rows: list[dict[str, Any]] = []

    for index, path in enumerate(file_batch):
        chroma_values = gpu_features["chroma_mean"][index].detach().cpu().numpy()
        row = {
            "filepath": path.as_posix(),
            "duration": float(gpu_features["duration"][index].detach().cpu().item()),
            "bpm": float(gpu_features["bpm"][index].detach().cpu().item()),
            "key_estimate": KEY_NAMES[int(gpu_features["key_index"][index].detach().cpu().item())],
            "rms_energy": float(gpu_features["rms_energy"][index].detach().cpu().item()),
            "loudness_proxy": float(gpu_features["loudness_proxy"][index].detach().cpu().item()),
            "spectral_centroid": float(gpu_features["spectral_centroid"][index].detach().cpu().item()),
            "spectral_bandwidth": float(gpu_features["spectral_bandwidth"][index].detach().cpu().item()),
            "spectral_rolloff": float(gpu_features["spectral_rolloff"][index].detach().cpu().item()),
            "zero_crossing_rate": float(gpu_features["zero_crossing_rate"][index].detach().cpu().item()),
            "onset_strength": float(gpu_features["onset_strength"][index].detach().cpu().item()),
            "onset_density": float(gpu_features["onset_density"][index].detach().cpu().item()),
        }
        for chroma_index, chroma_name in enumerate(KEY_NAMES):
            row[f"chroma_{chroma_name.replace('#', 'sharp')}"] = float(chroma_values[chroma_index])
        rows.append(row)
    return rows


def process_files_batched(
    audio_files: list[Path],
    sample_rate: int,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    grouped_paths: dict[int, list[Path]] = defaultdict(list)

    LOGGER.info("Preloading audio and grouping by length for batched extraction.")
    for path in tqdm(audio_files, desc="Grouping files", unit="file"):
        try:
            num_frames = inspect_num_frames(path, sample_rate)
            grouped_paths[num_frames].append(path)
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("Failed to inspect %s for batching metadata: %s", path, error)

    if not grouped_paths:
        raise RuntimeError("Failed to group any audio files for metadata extraction.")

    all_rows: list[dict[str, Any]] = []
    total_files = sum(len(paths) for paths in grouped_paths.values())
    processed_files = 0
    start_time = time.perf_counter()
    active_device = device

    for num_frames, paths in grouped_paths.items():
        LOGGER.info("Processing group with %d samples per file (%d files).", num_frames, len(paths))
        for start in range(0, len(paths), batch_size):
            file_batch = paths[start : start + batch_size]
            try:
                rows = process_batch_on_device(file_batch, sample_rate, active_device)
            except torch.cuda.OutOfMemoryError:
                LOGGER.warning("CUDA OOM while processing batch of %d files. Falling back to CPU.", len(file_batch))
                if active_device.type == "cuda":
                    torch.cuda.empty_cache()
                active_device = torch.device("cpu")
                rows = process_batch_on_device(file_batch, sample_rate, active_device)
            except RuntimeError as error:
                if active_device.type == "cuda" and "out of memory" in str(error).lower():
                    LOGGER.warning("CUDA runtime OOM while processing batch of %d files. Falling back to CPU.", len(file_batch))
                    torch.cuda.empty_cache()
                    active_device = torch.device("cpu")
                    rows = process_batch_on_device(file_batch, sample_rate, active_device)
                else:
                    raise
            all_rows.extend(rows)
            processed_files += len(file_batch)
            elapsed = max(time.perf_counter() - start_time, 1e-6)
            LOGGER.info(
                "Processed %d/%d files on %s | batch_size=%d | elapsed=%.2fs | files/sec=%.2f",
                processed_files,
                total_files,
                active_device,
                len(file_batch),
                elapsed,
                processed_files / elapsed,
            )

    return all_rows


def quantile_tag(series: pd.Series, low_label: str, mid_label: str, high_label: str) -> pd.Series:
    q1 = series.quantile(0.33)
    q2 = series.quantile(0.66)
    return pd.Series(
        np.where(series <= q1, low_label, np.where(series <= q2, mid_label, high_label)),
        index=series.index,
    )


def human_to_snake(text: str) -> str:
    return text.replace(" ", "_")


def describe_cluster(row: pd.Series) -> str:
    return (
        f"cluster_{int(row['cluster_id'])}_"
        f"{human_to_snake(row['tempo_tag'])}_"
        f"{human_to_snake(row['energy_tag'])}_"
        f"{human_to_snake(row['density_tag'])}"
    )


def assign_splits(frame: pd.DataFrame, seed: int, train_ratio: float, val_ratio: float) -> pd.Series:
    rng = np.random.default_rng(seed)
    shuffled = frame.index.to_numpy().copy()
    rng.shuffle(shuffled)
    train_end = int(len(shuffled) * train_ratio)
    val_end = train_end + int(len(shuffled) * val_ratio)
    split = pd.Series(index=frame.index, dtype="object")
    split.loc[shuffled[:train_end]] = "train"
    split.loc[shuffled[train_end:val_end]] = "val"
    split.loc[shuffled[val_end:]] = "test"
    return split


def summarize_numeric(series: pd.Series) -> dict[str, float | int | None]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "25%": None,
            "50%": None,
            "75%": None,
            "max": None,
        }
    return {
        "count": int(clean.count()),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=1)) if clean.count() > 1 else 0.0,
        "min": float(clean.min()),
        "25%": float(clean.quantile(0.25)),
        "50%": float(clean.quantile(0.50)),
        "75%": float(clean.quantile(0.75)),
        "max": float(clean.max()),
    }


def write_quality_report(path: Path, features_df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    corrected_mask = features_df["bpm_was_corrected"].fillna(False).astype(bool)
    payload = {
        "n_rows": int(len(features_df)),
        "bpm_raw_summary": summarize_numeric(features_df["bpm_raw"]),
        "bpm_corrected_summary": summarize_numeric(features_df["bpm"]),
        "bpm_corrected_count": int(corrected_mask.sum()),
        "bpm_corrected_ratio": float(corrected_mask.mean()) if len(features_df) else 0.0,
        "tempo_tag_counts": {str(k): int(v) for k, v in features_df["tempo_tag"].value_counts().sort_index().items()},
        "energy_tag_counts": {str(k): int(v) for k, v in features_df["energy_tag"].value_counts().sort_index().items()},
        "brightness_tag_counts": {str(k): int(v) for k, v in features_df["brightness_tag"].value_counts().sort_index().items()},
        "density_tag_counts": {str(k): int(v) for k, v in features_df["density_tag"].value_counts().sort_index().items()},
        "cluster_counts": {str(k): int(v) for k, v in features_df["cluster_tag"].value_counts().sort_index().items()},
        "prompt_features_unique_count": int(features_df["prompt_features"].nunique()),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file)
    allow_half_double_fix = parse_bool_flag(args.allow_half_double_fix)
    requested_device = resolve_device(args.device)
    start_time = time.perf_counter()

    if requested_device.type == "cuda":
        LOGGER.info("Selected device: cuda")
        LOGGER.info("GPU name: %s", torch.cuda.get_device_name(0))
    else:
        LOGGER.info("Selected device: cpu")
    LOGGER.info("Configured batch size: %d", args.batch_size)
    LOGGER.info("Analysis sample rate: %d", args.sample_rate)

    audio_files = sorted([path for path in args.input_dir.glob("*") if path.suffix.lower() in {".wav", ".flac"}])
    if args.max_files is not None:
        audio_files = audio_files[: args.max_files]
    if not audio_files:
        raise FileNotFoundError(f"No processed audio chunks found in {args.input_dir}")

    try:
        feature_rows = process_files_batched(audio_files, args.sample_rate, requested_device, args.batch_size)
    except Exception as error:  # noqa: BLE001
        if requested_device.type == "cuda":
            LOGGER.warning("GPU pipeline failed (%s). Retrying entire extraction on CPU.", error)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            feature_rows = process_files_batched(audio_files, args.sample_rate, torch.device("cpu"), args.batch_size)
        else:
            raise

    features_df = pd.DataFrame(feature_rows)
    numeric_columns = [column for column in features_df.columns if column not in {"filepath", "key_estimate"}]

    features_df.rename(columns={"bpm": "bpm_raw"}, inplace=True)
    if allow_half_double_fix:
        features_df["bpm"] = features_df["bpm_raw"].apply(
            lambda value: normalize_bpm_for_techno(value, tempo_min=args.tempo_min, tempo_max=args.tempo_max)
        )
    else:
        features_df["bpm"] = features_df["bpm_raw"]
    features_df["bpm_was_corrected"] = (
        features_df["bpm_raw"].notna()
        & features_df["bpm"].notna()
        & (features_df["bpm_raw"].round(6) != features_df["bpm"].round(6))
    )

    numeric_columns = [column for column in features_df.columns if column not in {"filepath", "key_estimate"}]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features_df[numeric_columns])
    scaled_df = pd.DataFrame(scaled, columns=[f"{column}_z" for column in numeric_columns])
    features_df = pd.concat([features_df.reset_index(drop=True), scaled_df], axis=1)

    features_df["tempo_tag"] = quantile_tag(features_df["bpm"], "low tempo", "medium tempo", "high tempo")
    features_df["energy_tag"] = quantile_tag(features_df["rms_energy"], "low energy", "medium energy", "high energy")
    features_df["brightness_tag"] = quantile_tag(
        features_df["spectral_centroid"],
        "dark sound",
        "neutral brightness",
        "bright sound",
    )
    features_df["density_tag"] = quantile_tag(
        features_df["onset_density"],
        "sparse rhythm",
        "medium density rhythm",
        "dense rhythm",
    )

    cluster_features = scaled_df[
        [
            "bpm_z",
            "rms_energy_z",
            "spectral_centroid_z",
            "spectral_bandwidth_z",
            "spectral_rolloff_z",
            "onset_density_z",
            "onset_strength_z",
        ]
    ]
    n_clusters = max(1, min(args.clusters, len(features_df)))
    kmeans = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=10)
    features_df["cluster_id"] = kmeans.fit_predict(cluster_features)

    cluster_means = (
        features_df.groupby("cluster_id")[
            ["bpm_z", "rms_energy_z", "spectral_centroid_z", "onset_density_z"]
        ]
        .mean()
        .reset_index()
    )
    cluster_tag_source = (
        features_df.groupby("cluster_id")[["tempo_tag", "energy_tag", "density_tag"]]
        .agg(lambda series: series.mode().iat[0] if not series.mode().empty else series.iloc[0])
        .reset_index()
    )
    cluster_means = cluster_means.merge(cluster_tag_source, on="cluster_id", how="left")
    cluster_means["cluster_tag"] = cluster_means.apply(describe_cluster, axis=1)
    features_df = features_df.merge(cluster_means[["cluster_id", "cluster_tag"]], on="cluster_id", how="left")

    features_df["prompt_generic"] = args.generic_prompt
    features_df["prompt_features"] = (
        "techno music, electronic track, "
        + features_df["tempo_tag"]
        + ", "
        + features_df["energy_tag"]
        + ", "
        + features_df["brightness_tag"]
        + ", "
        + features_df["density_tag"]
    )

    features_df["split"] = assign_splits(
        features_df,
        seed=args.seed,
        train_ratio=args.split_train,
        val_ratio=args.split_val,
    )

    args.features_out.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(args.features_out, index=False)
    write_quality_report(args.quality_report_out, features_df)

    metadata_columns = [
        "filepath",
        "duration",
        "bpm",
        "tempo_tag",
        "energy_tag",
        "brightness_tag",
        "density_tag",
        "cluster_id",
        "cluster_tag",
        "prompt_generic",
        "prompt_features",
        "split",
    ]
    metadata_df = features_df[metadata_columns].copy()
    metadata_df.to_csv(args.metadata_out, index=False)

    elapsed = max(time.perf_counter() - start_time, 1e-6)
    LOGGER.info("Saved features to %s", args.features_out)
    LOGGER.info("Saved final metadata to %s", args.metadata_out)
    LOGGER.info("Saved metadata quality report to %s", args.quality_report_out)
    LOGGER.info(
        "Metadata extraction finished in %.2fs | files=%d | files/sec=%.2f",
        elapsed,
        len(feature_rows),
        len(feature_rows) / elapsed,
    )


if __name__ == "__main__":
    main()
