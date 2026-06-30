# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

try:
    from scripts.make_metadata import compute_spectral_features, normalize_bpm_for_techno, safe_load_audio
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from make_metadata import compute_spectral_features, normalize_bpm_for_techno, safe_load_audio


LOGGER = logging.getLogger("evaluate")
FEATURE_COLUMNS = ["bpm", "duration", "spectral_centroid", "rms_energy", "onset_density"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated samples against dataset statistics.")
    parser.add_argument("--dataset_metadata", type=Path, default=Path("data/metadata.csv"))
    parser.add_argument("--dataset_features", type=Path, default=Path("data/metadata_features.csv"))
    parser.add_argument("--base_dir", type=Path, default=Path("outputs/samples/base"))
    parser.add_argument("--generic_dir", type=Path, default=Path("outputs/samples/lora_generic"))
    parser.add_argument("--feature_dir", type=Path, default=Path("outputs/samples/lora_features"))
    parser.add_argument(
        "--sample_dir",
        action="append",
        default=[],
        help="Optional explicit sample variant in the form label=path. May be passed multiple times.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--log_file", type=Path, default=Path("outputs/logs/evaluate.log"))
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--tempo_min", type=float, default=110.0)
    parser.add_argument("--tempo_max", type=float, default=150.0)
    parser.add_argument("--plot_min_samples", type=int, default=10)
    return parser.parse_args()


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )


def extract_audio_metrics(path: Path, sample_rate: int, tempo_min: float, tempo_max: float) -> dict[str, float | str]:
    waveform, _ = safe_load_audio(path, sample_rate)
    device = torch.device("cpu")
    batch = waveform.unsqueeze(0).to(device)
    features = compute_spectral_features(batch, sample_rate, device)

    bpm_raw = float(features["bpm"][0].detach().cpu().item())
    bpm = float(normalize_bpm_for_techno(bpm_raw, tempo_min=tempo_min, tempo_max=tempo_max))
    duration = float(features["duration"][0].detach().cpu().item())
    spectral_centroid = float(features["spectral_centroid"][0].detach().cpu().item())
    rms_energy = float(features["rms_energy"][0].detach().cpu().item())
    onset_density = float(features["onset_density"][0].detach().cpu().item())

    return {
        "filepath": path.as_posix(),
        "duration": duration,
        "bpm_raw": bpm_raw,
        "bpm": bpm,
        "spectral_centroid": spectral_centroid,
        "rms_energy": rms_energy,
        "onset_density": onset_density,
    }


def collect_metrics(directory: Path, label: str, sample_rate: int, tempo_min: float, tempo_max: float) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    audio_files = sorted([path for path in directory.glob("*") if path.suffix.lower() in {".wav", ".flac", ".mp3"}])
    LOGGER.info("Collecting metrics for %s from %s (%d files)", label, directory, len(audio_files))
    for path in tqdm(audio_files, desc=f"Evaluating {label}"):
        try:
            row = extract_audio_metrics(path, sample_rate=sample_rate, tempo_min=tempo_min, tempo_max=tempo_max)
            row["variant"] = label
            rows.append(row)
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("Evaluation failed for %s: %s", path, error)
    return pd.DataFrame(rows)


def resolve_sample_dirs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.sample_dir:
        resolved: list[tuple[str, Path]] = []
        for item in args.sample_dir:
            if "=" not in item:
                raise ValueError(f"Invalid --sample_dir value `{item}`. Expected label=path.")
            label, raw_path = item.split("=", 1)
            label = label.strip()
            path = Path(raw_path.strip()).resolve()
            if not label:
                raise ValueError(f"Invalid --sample_dir value `{item}`. Label must not be empty.")
            resolved.append((label, path))
        return resolved

    return [
        ("base", args.base_dir.resolve()),
        ("lora_generic", args.generic_dir.resolve()),
        ("lora_features", args.feature_dir.resolve()),
    ]


def compute_feature_distance(dataset_frame: pd.DataFrame, sample_frame: pd.DataFrame) -> pd.Series:
    scaler = StandardScaler()
    dataset_values = dataset_frame[FEATURE_COLUMNS].astype(float)
    scaler.fit(dataset_values)
    sample_scaled = scaler.transform(sample_frame[FEATURE_COLUMNS].astype(float))
    dataset_center = scaler.transform(dataset_values).mean(axis=0, keepdims=True)
    distances = np.linalg.norm(sample_scaled - dataset_center, axis=1)
    return pd.Series(distances, index=sample_frame.index)


def summarize_against_dataset(dataset_frame: pd.DataFrame, sample_frame: pd.DataFrame) -> pd.DataFrame:
    dataset_means = dataset_frame[FEATURE_COLUMNS].mean()
    summary = (
        sample_frame.groupby("variant")[FEATURE_COLUMNS + ["feature_l2_to_dataset"]]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary.columns = [
        "_".join(column).strip("_") if isinstance(column, tuple) else str(column)
        for column in summary.columns.to_flat_index()
    ]
    for metric in FEATURE_COLUMNS:
        summary[f"{metric}_dataset_mean"] = float(dataset_means[metric])
        summary[f"{metric}_delta_to_dataset_mean"] = summary[f"{metric}_mean"] - float(dataset_means[metric])
    return summary


def write_report(summary: pd.DataFrame, dataset_frame: pd.DataFrame, output_path: Path, plot_paths: list[Path]) -> None:
    dataset_means = dataset_frame[FEATURE_COLUMNS].mean()
    lines = [
        "# Evaluation report",
        "",
        "## Dataset feature means",
        "",
        dataset_means.to_frame(name="dataset_mean").to_markdown(),
        "",
        "## Summary metrics",
        "",
        summary.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- `feature_l2_to_dataset` is computed in the same raw feature space as `make_metadata.py`, then z-normalized using the dataset only.",
        "- Lower distance means the generated audio is closer to the dataset centroid.",
        "- Large positive `spectral_centroid_delta_to_dataset_mean` usually means the samples are too bright or noisy compared with the dataset.",
        "- Large negative `onset_density_delta_to_dataset_mean` usually means the rhythm is sparser than the dataset.",
        "",
        "## Saved plots",
        "",
    ]
    if plot_paths:
        lines.extend(f"- {path.name}" for path in plot_paths)
    else:
        lines.append("- Plot generation skipped because there were too few generated samples per variant.")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_plots(dataset_frame: pd.DataFrame, samples_frame: pd.DataFrame, output_dir: Path, min_samples: int) -> list[Path]:
    plot_paths: list[Path] = []
    variant_counts = samples_frame["variant"].value_counts()
    if variant_counts.empty or int(variant_counts.min()) < min_samples:
        LOGGER.info(
            "Skipping plots because the minimum sample count per variant is %s, below threshold %s.",
            int(variant_counts.min()) if not variant_counts.empty else 0,
            min_samples,
        )
        return plot_paths

    for metric in FEATURE_COLUMNS:
        plt.figure(figsize=(10, 6))
        plt.axvline(dataset_frame[metric].mean(), linestyle="--", label="dataset mean")
        for variant in samples_frame["variant"].unique():
            subset = samples_frame.loc[samples_frame["variant"] == variant, metric]
            subset.plot(kind="density", label=variant)
        plt.title(f"{metric} density")
        plt.xlabel(metric)
        plt.ylabel("density")
        plt.legend()
        plt.tight_layout()
        plot_path = output_dir / f"{metric}_density.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        plot_paths.append(plot_path)
    return plot_paths


def write_summary_json(path: Path, summary: pd.DataFrame, sample_frame: pd.DataFrame) -> None:
    payload = {
        "variants": sorted(sample_frame["variant"].unique().tolist()),
        "sample_count": int(len(sample_frame)),
        "variant_counts": {str(k): int(v) for k, v in sample_frame["variant"].value_counts().sort_index().items()},
        "feature_l2_to_dataset_mean_by_variant": {
            str(k): float(v)
            for k, v in sample_frame.groupby("variant")["feature_l2_to_dataset"].mean().items()
        },
        "summary_rows": summary.to_dict(orient="records"),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_features = pd.read_csv(args.dataset_features)
    required_columns = ["filepath"] + FEATURE_COLUMNS
    missing = [column for column in required_columns if column not in dataset_features.columns]
    if missing:
        raise ValueError(f"Dataset features are missing required columns: {missing}")

    dataset_compare = dataset_features[required_columns].copy()
    sample_frames = [
        collect_metrics(directory, label, sample_rate=args.sample_rate, tempo_min=args.tempo_min, tempo_max=args.tempo_max)
        for label, directory in resolve_sample_dirs(args)
        if directory.exists()
    ]
    sample_frames = [frame for frame in sample_frames if not frame.empty]
    if not sample_frames:
        raise FileNotFoundError("No generated sample files found for evaluation.")

    all_samples = pd.concat(sample_frames, ignore_index=True)
    all_samples["feature_l2_to_dataset"] = compute_feature_distance(dataset_compare, all_samples)
    all_samples.to_csv(args.output_dir / "evaluation.csv", index=False)

    summary = summarize_against_dataset(dataset_compare, all_samples)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    plot_paths = save_plots(dataset_compare, all_samples, args.output_dir, min_samples=args.plot_min_samples)
    write_report(summary, dataset_compare, args.output_dir / "report.md", plot_paths)
    write_summary_json(args.output_dir / "summary.json", summary, all_samples)

    LOGGER.info("Saved evaluation to %s", args.output_dir / "evaluation.csv")
    LOGGER.info("Saved summary to %s", args.output_dir / "summary.csv")
    LOGGER.info("Saved report to %s", args.output_dir / "report.md")


if __name__ == "__main__":
    main()
