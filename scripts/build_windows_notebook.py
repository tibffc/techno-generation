# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def add_markdown(nb: nbf.NotebookNode, text: str) -> None:
    nb.cells.append(nbf.v4.new_markdown_cell(text))


def add_code(nb: nbf.NotebookNode, text: str) -> None:
    nb.cells.append(nbf.v4.new_code_cell(text))


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()

    add_markdown(
        nb,
        """# ACE-Step 1.5 Techno Pipeline on Windows 10

Этот notebook — orchestration-only workflow для локального проекта:

`dirty_data -> raw_clean -> processed -> metadata -> train -> validate -> generate -> evaluate`

Он не дублирует логику `scripts/*.py`, а только:

- запускает существующие скрипты;
- проверяет артефакты после каждого этапа;
- сохраняет состояние параметров датасета;
- не даёт случайно использовать несогласованные `processed / metadata / tensors`.

## Что лежит в данных

- `data/dirty_data` — исходный грязный датасет.
- `data/raw_clean` — быстрый ingestion после `clean_dataset.py`.
- `data/processed` — чанки для обучения.
- `data/metadata.csv` и `data/metadata_features.csv` — self-labeling metadata.

## Что создаётся

- `outputs/tensors` — ACE-Step tensors для training.
- `outputs/checkpoints` — LoRA checkpoints.
- `outputs/samples/post_training` — реальные сэмплы после обучения.
- `outputs/evaluation` — CSV/summary/plots для сравнения с датасетом.
""",
    )

    add_code(
        nb,
        """from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from IPython.display import Audio, display

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

DIRTY_DATA_DIR = PROJECT_ROOT / "data" / "dirty_data"
RAW_CLEAN_DIR = PROJECT_ROOT / "data" / "raw_clean"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

CHUNK_SECONDS = 60
OVERLAP_SECONDS = 10
SAMPLE_RATE = 44100
GENERATION_DURATION = 60

PROMPT_MODE = "features"
TRAINING_PROMPT_MODE = "features"
TRAINING_ENABLED = True

REPROCESS_DATASET = False
RECOMPUTE_METADATA = False
LAUNCH_GRADIO = False

CONFIG_PATH = PROJECT_ROOT / "configs" / "train_lora.yaml"
PIPELINE_STATE_PATH = OUTPUTS_DIR / "pipeline_state.json"

DATASET_METADATA_PATH = PROJECT_ROOT / "data" / "metadata.csv"
FEATURES_METADATA_PATH = PROJECT_ROOT / "data" / "metadata_features.csv"
QUALITY_REPORT_PATH = PROJECT_ROOT / "data" / "metadata_quality_report.json"
CLEANING_REPORT_PATH = PROJECT_ROOT / "data" / "cleaning_report.json"

ACESTEP_REPO_PATH = PROJECT_ROOT / "external" / "ACE-Step-1.5"
ACESTEP_WEIGHTS_PATH = ACESTEP_REPO_PATH / "checkpoints"
ACESTEP_PYTHON = ACESTEP_REPO_PATH / ".venv" / "Scripts" / "python.exe"

TRAIN_LOG_PATH = OUTPUTS_DIR / "logs" / "train_lora.log"
TRAIN_SUMMARY_PATH = OUTPUTS_DIR / "checkpoints" / "features" / "training_summary.json"
POST_TRAINING_SAMPLES_DIR = OUTPUTS_DIR / "samples" / "post_training"
EVALUATION_DIR = OUTPUTS_DIR / "evaluation"

plt.rcParams["figure.figsize"] = (10, 5)

for path in [RAW_CLEAN_DIR, PROCESSED_DIR, OUTPUTS_DIR, OUTPUTS_DIR / "logs"]:
    path.mkdir(parents=True, exist_ok=True)

print(f"PROJECT_ROOT = {PROJECT_ROOT}")
print(f"CHUNK_SECONDS = {CHUNK_SECONDS}")
print(f"OVERLAP_SECONDS = {OVERLAP_SECONDS}")
print(f"SAMPLE_RATE = {SAMPLE_RATE}")
print(f"GENERATION_DURATION = {GENERATION_DURATION}")
print(f"REPROCESS_DATASET = {REPROCESS_DATASET}")
print(f"RECOMPUTE_METADATA = {RECOMPUTE_METADATA}")
print(f"TRAINING_ENABLED = {TRAINING_ENABLED}")
""",
    )

    add_code(
        nb,
        """def run_command(cmd: list[str], cwd: Path = PROJECT_ROOT) -> int:
    print(f"\\n[run_command] cwd={cwd}")
    print("[run_command] cmd=", " ".join(cmd))
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
    return_code = process.wait()
    print(f"\\n[run_command] return_code={return_code}")
    return return_code


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def list_files(path: Path, suffixes: tuple[str, ...] | None = None) -> list[Path]:
    if not path.exists():
        return []
    files = [item for item in path.rglob("*") if item.is_file()]
    if suffixes is not None:
        suffixes = tuple(item.lower() for item in suffixes)
        files = [item for item in files if item.suffix.lower() in suffixes]
    return sorted(files)


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    return f"{size:.2f} {units[unit_index]}"


def dir_size(path: Path) -> int:
    return sum(file.stat().st_size for file in list_files(path))


def show_paths(paths: list[Path], limit: int = 10) -> None:
    for path in paths[:limit]:
        print(path)


def current_dataset_params() -> dict:
    return {
        "chunk_seconds": CHUNK_SECONDS,
        "overlap_seconds": OVERLAP_SECONDS,
        "sample_rate": SAMPLE_RATE,
    }


def load_pipeline_state() -> dict:
    if PIPELINE_STATE_PATH.exists():
        return load_json(PIPELINE_STATE_PATH)
    return {}


def update_pipeline_state(**fields: dict) -> dict:
    state = load_pipeline_state()
    state.update(fields)
    save_json(PIPELINE_STATE_PATH, state)
    return state


def dataset_state_mismatch() -> tuple[bool, list[str]]:
    state = load_pipeline_state()
    recorded = state.get("dataset_params")
    current = current_dataset_params()
    reasons: list[str] = []
    if recorded is None:
        reasons.append("No recorded dataset_params in outputs/pipeline_state.json")
        return False, reasons
    for key, value in current.items():
        if recorded.get(key) != value:
            reasons.append(f"{key}: recorded={recorded.get(key)} current={value}")
    return len(reasons) > 0, reasons


def require_dataset_consistency() -> bool:
    mismatch, reasons = dataset_state_mismatch()
    if mismatch:
        print("Dataset parameter mismatch detected.")
        for reason in reasons:
            print(" -", reason)
        if not REPROCESS_DATASET:
            print("Refusing to continue because REPROCESS_DATASET = False.")
            print("Set REPROCESS_DATASET = True and rerun processed/metadata/tensors from scratch.")
            return False
    return True


def metadata_artifacts_valid() -> tuple[bool, list[str]]:
    reasons: list[str] = []
    required_paths = [DATASET_METADATA_PATH, FEATURES_METADATA_PATH, QUALITY_REPORT_PATH]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        return False, [f"Missing metadata artifacts: {missing}"]
    try:
        metadata_df = pd.read_csv(DATASET_METADATA_PATH)
        features_df = pd.read_csv(FEATURES_METADATA_PATH)
        quality = load_json(QUALITY_REPORT_PATH)
    except Exception as error:
        return False, [f"Failed to read metadata artifacts: {error}"]

    if metadata_df.empty:
        reasons.append("metadata.csv is empty")
    if features_df.empty:
        reasons.append("metadata_features.csv is empty")

    metadata_columns = {
        "filepath", "duration", "bpm", "prompt_generic", "prompt_features", "split",
        "tempo_tag", "energy_tag", "brightness_tag", "density_tag", "cluster_id", "cluster_tag",
    }
    feature_columns = {
        "filepath", "duration", "bpm_raw", "bpm", "bpm_was_corrected",
        "rms_energy", "spectral_centroid", "onset_density",
    }
    quality_keys = {
        "n_rows", "tempo_tag_counts", "energy_tag_counts",
        "brightness_tag_counts", "density_tag_counts", "prompt_features_unique_count",
    }

    missing_metadata = sorted(metadata_columns.difference(metadata_df.columns))
    missing_features = sorted(feature_columns.difference(features_df.columns))
    missing_quality = sorted(quality_keys.difference(quality.keys()))
    if missing_metadata:
        reasons.append(f"metadata.csv missing columns: {missing_metadata}")
    if missing_features:
        reasons.append(f"metadata_features.csv missing columns: {missing_features}")
    if missing_quality:
        reasons.append(f"metadata_quality_report.json missing keys: {missing_quality}")
    return len(reasons) == 0, reasons


def latest_checkpoint_dir(root: Path) -> Path | None:
    final_dir = root / "final"
    checkpoints_dir = root / "checkpoints"
    candidates: list[Path] = []
    if final_dir.exists() and final_dir.is_dir():
        candidates.append(final_dir)
    if checkpoints_dir.exists():
        candidates.extend(path for path in checkpoints_dir.iterdir() if path.is_dir())
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def show_audio_previews(audio_files: list[Path], limit: int = 3) -> None:
    for path in audio_files[:limit]:
        print(path)
        display(Audio(filename=str(path)))
""",
    )

    add_markdown(nb, "## Environment Checks")
    add_code(
        nb,
        """print("Python:", sys.version)
print("Working directory:", PROJECT_ROOT)
print("dirty_data exists:", DIRTY_DATA_DIR.exists())
print("dirty_data files:", len(list_files(DIRTY_DATA_DIR)))

try:
    subprocess.run(["ffmpeg", "-version"], check=False, text=True, encoding="utf-8", errors="replace")
except Exception as error:
    print("Warning: ffmpeg check failed:", error)

try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
        print("total VRAM:", human_size(torch.cuda.get_device_properties(0).total_memory))
except Exception as error:
    print("Warning: torch check failed:", error)
""",
    )

    add_markdown(nb, "## Full Cleanup")
    add_code(
        nb,
        """cleanup_targets = [
    OUTPUTS_DIR / "tensors",
    OUTPUTS_DIR / "checkpoints",
    OUTPUTS_DIR / "samples",
    OUTPUTS_DIR / "evaluation",
]

removed_bytes = 0
for target in cleanup_targets:
    if target.exists():
        removed_bytes += dir_size(target)
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

print("Removed:", human_size(removed_bytes))
print("Cleanup targets reset:")
for target in cleanup_targets:
    print(" -", target)
""",
    )

    add_markdown(nb, "## Dataset Cleaning")
    add_code(
        nb,
        """clean_returncode = run_command(
    [
        sys.executable,
        "scripts/clean_dataset.py",
        "--input_dir", str(DIRTY_DATA_DIR),
        "--output_dir", str(RAW_CLEAN_DIR),
        "--copy_mode", "copy",
        "--verify_mode", "ffprobe",
        "--min_size_kb", "100",
    ]
)

if CLEANING_REPORT_PATH.exists():
    report = load_json(CLEANING_REPORT_PATH)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report.get("elapsed_seconds"):
        total = report.get("candidate_audio_files") or report.get("total_files_found") or 0
        speed = total / max(report["elapsed_seconds"], 1e-9)
        print(f"Files per second: {speed:.2f}")

raw_clean_files = list_files(RAW_CLEAN_DIR)
print("raw_clean files:", len(raw_clean_files))
show_paths(raw_clean_files, limit=10)
""",
    )

    add_markdown(
        nb,
        """## Chunk Preparation

Этот этап режет аудио на `60s` чанки с overlap `10s`.

Если параметры датасета изменились, notebook требует полный пересчёт `processed + metadata + tensors`.
Без `REPROCESS_DATASET = True` пересоздание не запускается.
""",
    )
    add_code(
        nb,
        """processed_audio_files = list_files(PROCESSED_DIR, suffixes=(".wav", ".flac"))
mismatch, mismatch_reasons = dataset_state_mismatch()

if processed_audio_files and mismatch and not REPROCESS_DATASET:
    print("Processed dataset already exists, but chunk settings changed.")
    for reason in mismatch_reasons:
        print(" -", reason)
    print("Set REPROCESS_DATASET = True to rebuild processed/metadata/tensors.")
elif not processed_audio_files or REPROCESS_DATASET:
    prepare_returncode = run_command(
        [
            sys.executable,
            "scripts/prepare_dataset.py",
            "--input_dir", str(RAW_CLEAN_DIR),
            "--output_dir", str(PROCESSED_DIR),
            "--chunk_seconds", str(CHUNK_SECONDS),
            "--overlap_seconds", str(OVERLAP_SECONDS),
            "--sample_rate", str(SAMPLE_RATE),
        ]
    )
    if prepare_returncode == 0:
        update_pipeline_state(dataset_params=current_dataset_params())
else:
    print("Reusing existing processed dataset because parameters match and REPROCESS_DATASET = False.")

processed_audio_files = list_files(PROCESSED_DIR, suffixes=(".wav", ".flac"))
print("processed chunks:", len(processed_audio_files))
show_paths(processed_audio_files, limit=10)
""",
    )

    add_markdown(nb, "## Metadata / Self-Labeling")
    add_code(
        nb,
        """if not require_dataset_consistency():
    metadata_returncode = -1
else:
    valid, reasons = metadata_artifacts_valid()
    should_recompute = RECOMPUTE_METADATA or not valid
    if not valid:
        print("Metadata recompute required:")
        for reason in reasons:
            print(" -", reason)

    if should_recompute:
        metadata_returncode = run_command(
            [
                sys.executable,
                "scripts/make_metadata.py",
                "--input_dir", str(PROCESSED_DIR),
                "--metadata_out", str(DATASET_METADATA_PATH),
                "--features_out", str(FEATURES_METADATA_PATH),
                "--device", "auto",
                "--sample_rate", str(SAMPLE_RATE),
                "--tempo_min", "110",
                "--tempo_max", "150",
                "--allow_half_double_fix", "true",
            ]
        )
        if metadata_returncode == 0:
            update_pipeline_state(dataset_params=current_dataset_params(), metadata_ready=True)
    else:
        metadata_returncode = 0
        print("Reusing existing metadata artifacts.")

metadata_df = pd.read_csv(DATASET_METADATA_PATH)
features_df = pd.read_csv(FEATURES_METADATA_PATH)
quality_report = load_json(QUALITY_REPORT_PATH)

display(metadata_df.head())
metadata_df.info()
print("\\nSplit distribution:")
print(metadata_df["split"].value_counts(dropna=False))
print("\\nPrompt examples:")
print(metadata_df["prompt_generic"].dropna().head(3).tolist())
print(metadata_df["prompt_features"].dropna().head(5).tolist())
print("\\nQuality report:")
print(json.dumps(quality_report, ensure_ascii=False, indent=2))
""",
    )

    add_code(
        nb,
        """metadata_df = pd.read_csv(DATASET_METADATA_PATH)
features_df = pd.read_csv(FEATURES_METADATA_PATH)

for column, title in [
    ("bpm_raw", "BPM Raw Histogram"),
    ("bpm", "BPM Corrected Histogram"),
    ("rms_energy", "RMS Energy Histogram"),
    ("spectral_centroid", "Spectral Centroid Histogram"),
]:
    if column in features_df.columns:
        plt.figure()
        plt.hist(features_df[column].dropna(), bins=30)
        plt.title(title)
        plt.xlabel(column)
        plt.ylabel("count")
        plt.show()

for column, title in [
    ("tempo_tag", "Tempo Tag Counts"),
    ("energy_tag", "Energy Tag Counts"),
    ("brightness_tag", "Brightness Tag Counts"),
    ("density_tag", "Density Tag Counts"),
]:
    if column in metadata_df.columns:
        plt.figure()
        metadata_df[column].value_counts().plot(kind="bar")
        plt.title(title)
        plt.xlabel(column)
        plt.ylabel("count")
        plt.show()

if "bpm_was_corrected" in features_df.columns:
    plt.figure()
    features_df["bpm_was_corrected"].value_counts().plot(kind="bar")
    plt.title("BPM Was Corrected")
    plt.xlabel("bpm_was_corrected")
    plt.ylabel("count")
    plt.show()
""",
    )

    add_markdown(nb, "## LoRA Training")
    add_code(
        nb,
        """if not require_dataset_consistency():
    print("Training blocked by dataset mismatch.")
else:
    backend_checks = {
        "processed_dir": PROCESSED_DIR.exists() and any(PROCESSED_DIR.rglob("*.flac")) or any(PROCESSED_DIR.rglob("*.wav")),
        "metadata_csv": DATASET_METADATA_PATH.exists(),
        "repo_path": ACESTEP_REPO_PATH.exists(),
        "weights_path": ACESTEP_WEIGHTS_PATH.exists() and any(ACESTEP_WEIGHTS_PATH.iterdir()),
        "backend_python": ACESTEP_PYTHON.exists(),
    }

    print(json.dumps(backend_checks, ensure_ascii=False, indent=2))
    if not all(backend_checks.values()):
        print("Training preflight failed. Fix missing dependencies first.")
    elif TRAINING_ENABLED:
        run_command(
            [
                sys.executable,
                "scripts/train_lora.py",
                "--config", str(CONFIG_PATH),
                "--prompt_mode", TRAINING_PROMPT_MODE,
                "--run",
            ]
        )
    else:
        print("Training skipped because TRAINING_ENABLED = False.")
""",
    )

    add_markdown(nb, "## Post-Training Validation")
    add_code(
        nb,
        """checkpoint_root = OUTPUTS_DIR / "checkpoints" / TRAINING_PROMPT_MODE
latest_dir = latest_checkpoint_dir(checkpoint_root)
print("checkpoint_root:", checkpoint_root)
print("latest_checkpoint_dir:", latest_dir)

if TRAIN_SUMMARY_PATH.exists():
    summary = load_json(TRAIN_SUMMARY_PATH)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
else:
    print("training_summary.json not found.")

checkpoint_files = list_files(checkpoint_root)
print("\\nCheckpoint files:", len(checkpoint_files))
for path in checkpoint_files[:30]:
    print(f"{path} | {human_size(path.stat().st_size)}")

if TRAIN_LOG_PATH.exists():
    log_lines = TRAIN_LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
    print("\\nLast lines from train_lora.log:")
    print("\\n".join(log_lines[-40:]))
else:
    print("train_lora.log not found.")
""",
    )

    add_markdown(
        nb,
        """## Inference (LoRA Samples)

Этот этап использует только `scripts/generate_samples.py`.

`generate_samples.py` не делает искусственный fade-out. Если конец трека затухает, это поведение модели/сэмплинга, а не notebook post-processing.
""",
    )
    add_code(
        nb,
        """checkpoint_root = OUTPUTS_DIR / "checkpoints" / TRAINING_PROMPT_MODE
latest_dir = latest_checkpoint_dir(checkpoint_root)
print("latest_checkpoint_dir:", latest_dir)

if latest_dir is None:
    print("No checkpoint directory found. Run training first.")
else:
    generation_returncode = run_command(
        [
            sys.executable,
            "scripts/generate_samples.py",
            "--config", str(CONFIG_PATH),
            "--checkpoint_dir", str(latest_dir),
            "--prompt_mode", TRAINING_PROMPT_MODE,
            "--num_samples", "3",
            "--prompt_strategy", "diverse",
            "--duration", str(GENERATION_DURATION),
            "--output_dir", str(POST_TRAINING_SAMPLES_DIR),
            "--device", "cuda",
        ]
    )

sample_files = list_files(POST_TRAINING_SAMPLES_DIR, suffixes=(".wav", ".mp3", ".flac"))
print("Generated files:", len(sample_files))
show_paths(sample_files, limit=10)
show_audio_previews(sample_files, limit=3)
""",
    )

    add_markdown(nb, "## Evaluation / Metrics")
    add_code(
        nb,
        """if not POST_TRAINING_SAMPLES_DIR.exists() or not list_files(POST_TRAINING_SAMPLES_DIR, suffixes=(".wav", ".mp3", ".flac")):
    print("No post-training samples found. Run inference first.")
else:
    run_command(
        [
            sys.executable,
            "scripts/evaluate.py",
            "--dataset_metadata", str(DATASET_METADATA_PATH),
            "--dataset_features", str(FEATURES_METADATA_PATH),
            "--sample_dir", f"lora_features={POST_TRAINING_SAMPLES_DIR}",
            "--output_dir", str(EVALUATION_DIR),
        ]
    )

evaluation_csv = EVALUATION_DIR / "evaluation.csv"
summary_csv = EVALUATION_DIR / "summary.csv"
summary_json = EVALUATION_DIR / "summary.json"
report_md = EVALUATION_DIR / "report.md"

if evaluation_csv.exists():
    evaluation_df = pd.read_csv(evaluation_csv)
    display(evaluation_df.head())
else:
    print("evaluation.csv not found.")

if summary_csv.exists():
    display(pd.read_csv(summary_csv).head())
if summary_json.exists():
    print(summary_json.read_text(encoding="utf-8", errors="replace"))
if report_md.exists():
    print("\\n".join(report_md.read_text(encoding="utf-8", errors="replace").splitlines()[:30]))

plot_files = list_files(EVALUATION_DIR, suffixes=(".png",))
print("Saved evaluation plots:", len(plot_files))
show_paths(plot_files, limit=20)
""",
    )

    add_markdown(nb, "## Gradio")
    add_code(
        nb,
        """if LAUNCH_GRADIO:
    run_command([sys.executable, "app/gradio_app.py"])
else:
    print("Gradio launch skipped. Set LAUNCH_GRADIO = True to start the demo.")
""",
    )

    return nb


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    notebook_path = project_root / "notebooks" / "run_pipeline_windows10.ipynb"
    notebook_path.parent.mkdir(parents=True, exist_ok=True)

    notebook = build_notebook()
    for cell in notebook.cells:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    with open(notebook_path, "w", encoding="utf-8") as handle:
        nbf.write(notebook, handle)

    print(f"Notebook recreated at {notebook_path}")


if __name__ == "__main__":
    main()
