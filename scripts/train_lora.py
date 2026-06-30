# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any

import yaml

try:
    from scripts.export_acestep_manifest import export_manifests
except ModuleNotFoundError:  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parent))
    from export_acestep_manifest import export_manifests


LOGGER = logging.getLogger("train_lora")
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
PROMPT_COLUMN_MAP = {
    "generic": "prompt_generic",
    "features": "prompt_features",
}
STEP_RE = re.compile(r"\bStep\s+(\d+)\b", re.IGNORECASE)
LOSS_RE = re.compile(r"\bLoss\s+([0-9]+(?:\.[0-9]+)?)\b", re.IGNORECASE)
LR_RE = re.compile(r"\bLR\s+([0-9.eE+-]+)\b")
ETA_RE = re.compile(r"\bETA\s+([0-9:]+|--)\b", re.IGNORECASE)
ELAPSED_RE = re.compile(r"\bElapsed\s+([0-9smhd:]+)\b", re.IGNORECASE)
PROCESSED_RE = re.compile(r"Preprocessing complete:\s+(\d+)/(\d+)\s+processed", re.IGNORECASE)
ERROR_RE = re.compile(r"\b(?:ERROR|FAIL|Traceback|ValueError:|RuntimeError:|FileNotFoundError:)\b")
WARNING_RE = re.compile(r"\b(?:WARNING|SyntaxWarning|UserWarning)\b")
MOJIBAKE_MARKERS = "ÐÑâ€™€œ”–‘’"


@dataclass
class BackendMetrics:
    step: int | None = None
    loss: float | None = None
    lr: float | None = None
    eta: str | None = None
    elapsed: str | None = None
    preprocess_done: int | None = None
    preprocess_total: int | None = None
    last_line: str | None = None


@dataclass
class CommandResult:
    elapsed_seconds: float
    metrics: BackendMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ACE-Step / Side-Step LoRA training wrapper.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_lora.yaml"))
    parser.add_argument("--prompt_mode", choices=["generic", "features"], required=True)
    parser.add_argument("--dry_run", action="store_true", help="Print commands only. Default behavior.")
    parser.add_argument("--run", action="store_true", help="Actually run preprocessing/training.")
    parser.add_argument("--log_file", type=Path, default=Path("outputs/logs/train_lora.log"))
    args = parser.parse_args()
    if args.dry_run and args.run:
        raise ValueError("Use either --dry_run or --run, not both.")
    return args


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("", encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler()],
    )


def append_raw_line(log_file: Path, line: str) -> None:
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(line)


def append_wrapper_section(log_file: Path, title: str, lines: list[str]) -> None:
    append_raw_line(log_file, f"=== {title} ===\n")
    for line in lines:
        append_raw_line(log_file, f"{line}\n")
    append_raw_line(log_file, "\n")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_exists(path: Path, message: str) -> None:
    if not path.exists():
        raise FileNotFoundError(message)


def ensure_non_empty_dir(path: Path, message: str) -> None:
    ensure_exists(path, message)
    if not any(path.iterdir()):
        raise FileNotFoundError(message)


def resolve_path(path_like: str | Path) -> Path:
    return Path(path_like).resolve()


def detect_backend_bf16_support(python_executable: Path) -> bool:
    command = [
        str(python_executable),
        "-c",
        "import torch; print('1' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else '0')",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0 and result.stdout.strip().endswith("1")


def render_template(template: str, context: dict[str, Any]) -> str:
    placeholder_names = {
        field_name
        for _, field_name, _, _ in Formatter().parse(template)
        if field_name is not None
    }
    missing = sorted(name for name in placeholder_names if name not in context)
    if missing:
        raise KeyError(f"Template placeholders are missing from context: {missing}")
    return template.format(**context)


def split_command(command: str) -> list[str]:
    parts = shlex.split(command, posix=os.name != "nt")
    if os.name == "nt":
        return [part[1:-1] if len(part) >= 2 and part[0] == '"' and part[-1] == '"' else part for part in parts]
    return parts


def should_suppress_backend_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if "matplotlib:" in stripped or "torio._extension.utils" in stripped:
        return True
    if "_readerthread" in stripped or "UnauthorizedAccess" in stripped or "Execution_Policies" in stripped:
        return True
    mojibake_hits = sum(1 for char in stripped if char in MOJIBAKE_MARKERS)
    ascii_alnum = sum(1 for char in stripped if char.isascii() and char.isalnum())
    return mojibake_hits >= 16 and ascii_alnum < 24


def sanitize_backend_line(line: str) -> str:
    stripped = line.rstrip("\r\n")
    if not stripped:
        return line
    mojibake_hits = sum(1 for char in stripped if char in MOJIBAKE_MARKERS)
    ascii_alnum = sum(1 for char in stripped if char.isascii() and char.isalnum())
    if mojibake_hits >= 8 and ascii_alnum >= 4:
        cleaned = "".join(char for char in stripped if char == "\t" or 32 <= ord(char) <= 126)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned:
            return cleaned + "\n"
    return line


def parse_backend_line(metrics: BackendMetrics, line: str) -> None:
    clean = line.strip()
    if not clean:
        return
    metrics.last_line = clean

    match = STEP_RE.search(clean)
    if match:
        metrics.step = int(match.group(1))
    match = LOSS_RE.search(clean)
    if match:
        metrics.loss = float(match.group(1))
    match = LR_RE.search(clean)
    if match:
        try:
            metrics.lr = float(match.group(1))
        except ValueError:
            pass
    match = ETA_RE.search(clean)
    if match:
        metrics.eta = match.group(1)
    match = ELAPSED_RE.search(clean)
    if match:
        metrics.elapsed = match.group(1)
    match = PROCESSED_RE.search(clean)
    if match:
        metrics.preprocess_done = int(match.group(1))
        metrics.preprocess_total = int(match.group(2))


def list_completed_tensor_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.rglob("*.pt") if not p.name.endswith(".tmp.pt"))


def format_size_bytes(size_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    unit_index = 0
    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    return f"{size:.2f} {units[unit_index]}"


def format_duration(seconds: float) -> str:
    total_seconds = int(max(seconds, 0.0))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def monitor_tensor_progress(tensor_dir: Path, total_expected: int, stop_event: threading.Event, interval_seconds: float = 30.0) -> None:
    started_at = time.perf_counter()
    while not stop_event.wait(interval_seconds):
        current_count = len(list_completed_tensor_files(tensor_dir))
        elapsed = max(time.perf_counter() - started_at, 1e-6)
        speed = current_count / elapsed
        if total_expected > 0 and speed > 0.0:
            remaining = max(total_expected - current_count, 0)
            eta_seconds = remaining / speed
            LOGGER.info(
                "Preprocessing progress: %s/%s tensors (%.1f%%) | %.2f files/s | ETA %s",
                current_count,
                total_expected,
                100.0 * current_count / total_expected,
                speed,
                format_duration(eta_seconds),
            )
        else:
            LOGGER.info("Preprocessing progress: %s tensors ready", current_count)


def cleanup_incomplete_tmp_tensors(path: Path) -> int:
    if not path.exists():
        return 0
    patterns = [path.rglob("*.tmp.pt")] if path.name.startswith("acestep_") else [path.glob("acestep_*/**/*.tmp.pt")]
    removed = 0
    for iterator in patterns:
        for tmp_path in iterator:
            try:
                tmp_path.unlink()
                removed += 1
            except OSError as error:
                LOGGER.warning("Failed to remove incomplete tensor file %s: %s", tmp_path, error)
    return removed


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    candidates: list[Path] = []
    final_dir = output_dir / "final"
    checkpoints_dir = output_dir / "checkpoints"
    if final_dir.exists():
        candidates.append(final_dir)
    if checkpoints_dir.exists():
        candidates.extend(path for path in checkpoints_dir.iterdir() if path.is_dir())
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def summarize_checkpoint_outputs(output_dir: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "checkpoint_dir": str(output_dir),
        "checkpoint_file_count": 0,
        "checkpoint_files": [],
        "latest_checkpoint_dir": None,
    }
    if not output_dir.exists():
        return summary

    all_files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    summary["checkpoint_file_count"] = len(all_files)
    summary["checkpoint_files"] = [
        {
            "path": str(file_path),
            "size_bytes": file_path.stat().st_size,
            "size_human": format_size_bytes(file_path.stat().st_size),
        }
        for file_path in all_files
    ]
    latest_checkpoint = find_latest_checkpoint(output_dir)
    if latest_checkpoint is not None:
        summary["latest_checkpoint_dir"] = str(latest_checkpoint)
    return summary


def read_log_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def parse_training_metrics_from_lines(lines: list[str]) -> dict[str, Any]:
    metrics = BackendMetrics()
    error_lines: list[str] = []
    warning_lines: list[str] = []
    for line in lines:
        parse_backend_line(metrics, line)
        if ERROR_RE.search(line):
            error_lines.append(line.strip())
        elif WARNING_RE.search(line):
            warning_lines.append(line.strip())
    return {
        "latest_step": metrics.step,
        "latest_loss": metrics.loss,
        "latest_lr": metrics.lr,
        "latest_eta": metrics.eta,
        "latest_elapsed": metrics.elapsed,
        "preprocess_done": metrics.preprocess_done,
        "preprocess_total": metrics.preprocess_total,
        "error_lines": error_lines[-20:],
        "warning_lines": warning_lines[-20:],
    }


def save_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def reset_backend_runtime_logs() -> None:
    for path in (WORKSPACE_ROOT / "sidestep.log", WORKSPACE_ROOT / "external" / "ACE-Step-1.5" / "sidestep.log"):
        if path.exists():
            try:
                path.unlink()
            except OSError as error:
                LOGGER.warning("Could not remove stale backend runtime log %s: %s", path, error)


def run_command(
    command: str,
    cwd: Path,
    log_file: Path,
    *,
    preprocess_tensor_dir: Path | None = None,
    preprocess_total: int | None = None,
) -> CommandResult:
    argv = split_command(command)
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONIOENCODING"] = "utf-8:replace"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    LOGGER.info("Running command in %s", cwd)
    LOGGER.info("Resolved absolute command: %s", command)

    metrics = BackendMetrics()
    started_at = time.perf_counter()
    stop_event = threading.Event()
    monitor_thread: threading.Thread | None = None
    if preprocess_tensor_dir is not None and preprocess_total is not None and preprocess_total > 0:
        monitor_thread = threading.Thread(
            target=monitor_tensor_progress,
            args=(preprocess_tensor_dir, preprocess_total, stop_event),
            daemon=True,
        )
        monitor_thread.start()

    process = subprocess.Popen(
        argv,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        bufsize=0,
    )
    assert process.stdout is not None

    try:
        while True:
            raw_line = process.stdout.readline()
            if not raw_line:
                if process.poll() is not None:
                    break
                continue
            decoded = sanitize_backend_line(raw_line.decode("utf-8", errors="replace"))
            if should_suppress_backend_line(decoded):
                continue
            sys.stdout.write(decoded)
            sys.stdout.flush()
            append_raw_line(log_file, decoded)
            parse_backend_line(metrics, decoded)
    finally:
        stop_event.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=2.0)

    return_code = process.wait()
    elapsed_seconds = time.perf_counter() - started_at
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, argv)
    return CommandResult(elapsed_seconds=elapsed_seconds, metrics=metrics)


def load_detection_report(config: dict[str, Any]) -> dict[str, Any]:
    detected_path = Path(config["backend"].get("detected_config_path", "configs/acestep_detected.json"))
    ensure_exists(detected_path, f"ACE-Step detection report not found: {detected_path}. Run `python scripts/setup_acestep.py` first.")
    return load_json(detected_path)


def resolve_command_templates(config: dict[str, Any], detection: dict[str, Any]) -> tuple[str | None, str]:
    backend_cfg = config["backend"]
    preprocess_template = backend_cfg.get("preprocess_command_template")
    train_template = backend_cfg.get("train_command_template")
    if train_template:
        return preprocess_template, train_template
    detected_train = detection["entrypoints"].get("recommended_train_command_template")
    detected_preprocess = detection["entrypoints"].get("recommended_preprocess_command_template")
    if detected_train:
        return preprocess_template or detected_preprocess, detected_train
    raise RuntimeError(
        "ACE-Step LoRA training entrypoint was not detected. "
        "Please open external/ACE-Step-1.5 docs and set backend.train_command_template in configs/train_lora.yaml."
    )


def export_backend_manifests(config: dict[str, Any], prompt_mode: str) -> tuple[Path, Path]:
    metadata_path = Path(config["training"]["metadata_path"])
    ensure_exists(metadata_path, f"Metadata file does not exist: {metadata_path}. Run scripts/make_metadata.py before training.")
    manifests_root = Path(config["backend"].get("manifests_root", "outputs/manifests"))
    export_manifests(
        metadata_csv=metadata_path,
        prompt_mode="generic",
        jsonl_out=manifests_root / "acestep_train_generic.jsonl",
        dataset_json_out=manifests_root / "acestep_dataset_generic.json",
    )
    export_manifests(
        metadata_csv=metadata_path,
        prompt_mode="features",
        jsonl_out=manifests_root / "acestep_train_features.jsonl",
        dataset_json_out=manifests_root / "acestep_dataset_features.json",
    )
    return manifests_root / f"acestep_train_{prompt_mode}.jsonl", manifests_root / f"acestep_dataset_{prompt_mode}.json"


def count_manifest_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def build_warning_messages(config: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if config["training"].get("max_train_steps") is not None:
        warnings.append("ACE-Step train.py fixed is epoch-based; training.max_train_steps is reference-only and is not passed to the backend.")
    if os.name == "nt" and config["training"].get("num_workers", 0) not in {0, None}:
        warnings.append("ACE-Step on Windows forces num_workers=0 internally. The configured num_workers value will be ignored by the backend.")
    return warnings


def build_context(config: dict[str, Any], detection: dict[str, Any], prompt_mode: str) -> dict[str, Any]:
    backend_cfg = config["backend"]
    training_cfg = config["training"]
    model_cfg = config["model"]

    repo_path = resolve_path(backend_cfg["repo_path"])
    base_model_path = resolve_path(backend_cfg["base_model_path"])
    metadata_path = resolve_path(training_cfg["metadata_path"])
    processed_dir = resolve_path(training_cfg.get("processed_dir", "data/processed"))
    output_dir = resolve_path(Path(training_cfg["output_root"]) / prompt_mode)
    tensor_output_dir = resolve_path(Path(backend_cfg.get("tensor_output_root", "outputs/tensors")) / f"acestep_{prompt_mode}")
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_output_dir.mkdir(parents=True, exist_ok=True)

    ensure_exists(repo_path, f"ACE-Step repository was not found at {repo_path}.")
    ensure_non_empty_dir(base_model_path, f"ACE-Step checkpoints directory is missing or empty: {base_model_path}.")
    ensure_exists(metadata_path, f"Metadata file does not exist: {metadata_path}.")
    ensure_exists(processed_dir, f"Processed audio directory does not exist: {processed_dir}.")
    if not any(processed_dir.rglob("*.wav")) and not any(processed_dir.rglob("*.flac")):
        raise FileNotFoundError(f"Processed audio directory contains no .wav/.flac files: {processed_dir}.")

    manifest_path, dataset_json_path = export_backend_manifests(config, prompt_mode)
    manifest_path = manifest_path.resolve()
    dataset_json_path = dataset_json_path.resolve()
    python_executable_value = detection["entrypoints"].get("recommended_python_executable") or "python"
    # Do not resolve symlinks for virtualenv python. In uv/venv, .venv/bin/python is often
    # a symlink to the base interpreter, and resolving it loses the virtualenv context.
    python_executable = Path(python_executable_value).absolute() if python_executable_value != "python" else Path(sys.executable).resolve()
    train_script_path = (repo_path / "train.py").resolve()
    logs_dir = resolve_path(training_cfg.get("logs_dir", "outputs/logs"))
    logs_dir.mkdir(parents=True, exist_ok=True)

    ensure_exists(python_executable, f"ACE-Step python executable was not found: {python_executable}.")
    ensure_exists(train_script_path, f"ACE-Step training entrypoint was not found: {train_script_path}.")
    ensure_exists(manifest_path, f"ACE-Step manifest JSONL does not exist: {manifest_path}.")
    ensure_exists(dataset_json_path, f"ACE-Step dataset JSON does not exist: {dataset_json_path}.")

    requested_precision = str(training_cfg["mixed_precision"]).lower()
    resolved_precision = requested_precision
    if requested_precision == "bf16" and backend_cfg.get("device", "cuda") == "cuda":
        if not detect_backend_bf16_support(python_executable):
            resolved_precision = "fp32"

    return {
        "python_executable": str(python_executable),
        "repo_path": str(repo_path),
        "workspace_root": str(WORKSPACE_ROOT),
        "train_script_path": str(train_script_path),
        "base_model_path": str(base_model_path),
        "metadata_path": str(metadata_path),
        "processed_dir": str(processed_dir),
        "output_dir": str(output_dir),
        "tensor_output_dir": str(tensor_output_dir),
        "manifest_path": str(manifest_path),
        "dataset_json_path": str(dataset_json_path),
        "prompt_mode": prompt_mode,
        "prompt_column": PROMPT_COLUMN_MAP[prompt_mode],
        "batch_size": training_cfg["batch_size"],
        "gradient_accumulation_steps": training_cfg["gradient_accumulation_steps"],
        "learning_rate": training_cfg["learning_rate"],
        "epochs": training_cfg["epochs"],
        "max_train_steps": training_cfg.get("max_train_steps", ""),
        "mixed_precision": resolved_precision,
        "requested_mixed_precision": requested_precision,
        "lora_rank": training_cfg["lora_rank"],
        "lora_alpha": training_cfg["lora_alpha"],
        "lora_dropout": training_cfg["lora_dropout"],
        "seed": training_cfg.get("seed", config.get("seed", 42)),
        "logs_dir": str(logs_dir),
        "summary_path": str(output_dir / "training_summary.json"),
        "save_every_n_epochs": training_cfg.get("save_every_n_epochs", training_cfg.get("save_every", 10)),
        "num_workers": training_cfg.get("num_workers", 0),
        "device": backend_cfg.get("device", "cuda"),
        "model_variant": backend_cfg.get("model_variant", "turbo"),
        "foundation_name": model_cfg.get("foundation_name", "ACE-Step 1.5"),
        "optimizer_type": training_cfg.get("optimizer_type", "adamw"),
        "max_duration_seconds": training_cfg.get("max_duration_seconds", 30),
        "offload_encoder_flag": "--offload-encoder" if training_cfg.get("offload_encoder", False) else "",
    }


def build_summary(
    *,
    context: dict[str, Any],
    phase: str,
    manifest_rows: int,
    total_elapsed_seconds: float,
    checkpoint_summary: dict[str, Any],
    preprocess_metrics: BackendMetrics | None,
    train_metrics: BackendMetrics | None,
    log_file: Path,
) -> dict[str, Any]:
    parsed_log_metrics = parse_training_metrics_from_lines(read_log_lines(log_file))
    sidestep_log_path = WORKSPACE_ROOT / "sidestep.log"
    return {
        "status": "success" if checkpoint_summary.get("latest_checkpoint_dir") else "incomplete",
        "phase": phase,
        "prompt_mode": context["prompt_mode"],
        "prompt_column": context["prompt_column"],
        "backend_repo": context["repo_path"],
        "backend_weights": context["base_model_path"],
        "manifest_path": context["manifest_path"],
        "dataset_json_path": context["dataset_json_path"],
        "tensor_output_dir": context["tensor_output_dir"],
        "checkpoint_output_dir": context["output_dir"],
        "train_examples": manifest_rows,
        "device": context["device"],
        "lora": {
            "rank": context["lora_rank"],
            "alpha": context["lora_alpha"],
            "dropout": context["lora_dropout"],
        },
        "training": {
            "batch_size": context["batch_size"],
            "gradient_accumulation_steps": context["gradient_accumulation_steps"],
            "learning_rate": context["learning_rate"],
            "epochs": context["epochs"],
            "precision": context["mixed_precision"],
        },
        "elapsed_seconds": total_elapsed_seconds,
        "elapsed_human": format_duration(total_elapsed_seconds),
        "preprocess_metrics": None if preprocess_metrics is None else preprocess_metrics.__dict__,
        "train_metrics_runtime": None if train_metrics is None else train_metrics.__dict__,
        "train_metrics_log": parsed_log_metrics,
        "checkpoint_summary": checkpoint_summary,
        "sidestep_log_path": str(sidestep_log_path) if sidestep_log_path.exists() else None,
        "train_log_path": str(log_file),
    }


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file)
    try:
        config = load_yaml(args.config)
        detection = load_detection_report(config)
        context = build_context(config, detection, args.prompt_mode)
        preprocess_template, train_template = resolve_command_templates(config, detection)
        preprocess_command = render_template(preprocess_template, context) if preprocess_template else None
        train_command = render_template(train_template, context)
        manifest_rows = count_manifest_rows(Path(context["manifest_path"]))
        warnings = build_warning_messages(config)
        if context["requested_mixed_precision"] != context["mixed_precision"]:
            warnings.append(
                f"Requested precision {context['requested_mixed_precision']} is not supported by the backend environment; falling back to {context['mixed_precision']}."
            )

        LOGGER.info("Prompt mode: %s", args.prompt_mode)
        LOGGER.info("Backend repo: %s", context["repo_path"])
        LOGGER.info("Backend python: %s", context["python_executable"])
        LOGGER.info("Device: %s", context["device"])
        LOGGER.info("Precision: requested=%s resolved=%s", context["requested_mixed_precision"], context["mixed_precision"])
        LOGGER.info("Train examples: %s", manifest_rows)

        append_wrapper_section(
            args.log_file,
            "wrapper",
            [
                f"Prompt mode: {args.prompt_mode}",
                f"Foundation: {context['foundation_name']}",
                f"Backend repo: {context['repo_path']}",
                f"Backend python: {context['python_executable']}",
                f"Backend weights: {context['base_model_path']}",
                f"Processed dir: {context['processed_dir']}",
                f"Manifest: {context['manifest_path']}",
                f"Dataset JSON: {context['dataset_json_path']}",
                f"Tensor dir: {context['tensor_output_dir']}",
                f"Checkpoint output dir: {context['output_dir']}",
                f"Device: {context['device']}",
                f"LoRA params: rank={context['lora_rank']} alpha={context['lora_alpha']} dropout={context['lora_dropout']}",
                f"Training params: batch_size={context['batch_size']} grad_accum={context['gradient_accumulation_steps']} precision={context['mixed_precision']} lr={context['learning_rate']} epochs={context['epochs']}",
                f"Requested precision: {context['requested_mixed_precision']}",
                *(f"Warning: {warning}" for warning in warnings),
            ],
        )

        if preprocess_command:
            append_wrapper_section(args.log_file, "preprocess command", [preprocess_command])
        append_wrapper_section(args.log_file, "train command", [train_command])

        if not args.run:
            return

        reset_backend_runtime_logs()
        tensor_output_dir = Path(context["tensor_output_dir"]).resolve()
        output_dir = Path(context["output_dir"]).resolve()
        total_started_at = time.perf_counter()
        preprocess_metrics: BackendMetrics | None = None
        train_metrics: BackendMetrics | None = None

        tensor_files = list_completed_tensor_files(tensor_output_dir)
        if preprocess_command and not tensor_files:
            removed_tmp_count = cleanup_incomplete_tmp_tensors(tensor_output_dir.parent)
            if removed_tmp_count:
                LOGGER.warning("Removed %s incomplete temporary tensor files.", removed_tmp_count)
            preprocess_result = run_command(
                preprocess_command,
                cwd=WORKSPACE_ROOT,
                log_file=args.log_file,
                preprocess_tensor_dir=tensor_output_dir,
                preprocess_total=manifest_rows,
            )
            preprocess_metrics = preprocess_result.metrics
            tensor_files = list_completed_tensor_files(tensor_output_dir)
        elif tensor_files:
            LOGGER.info("Reusing existing preprocessed tensors: %s", len(tensor_files))
            append_wrapper_section(args.log_file, "preprocess", [f"Reusing {len(tensor_files)} existing tensor files from {tensor_output_dir}"])

        if preprocess_command and not tensor_files:
            raise RuntimeError(f"No completed .pt tensors were found after preprocessing in {tensor_output_dir}.")

        train_result = run_command(
            train_command,
            cwd=WORKSPACE_ROOT,
            log_file=args.log_file,
        )
        train_metrics = train_result.metrics
        checkpoint_summary = summarize_checkpoint_outputs(output_dir)
        summary = build_summary(
            context=context,
            phase="training",
            manifest_rows=manifest_rows,
            total_elapsed_seconds=time.perf_counter() - total_started_at,
            checkpoint_summary=checkpoint_summary,
            preprocess_metrics=preprocess_metrics,
            train_metrics=train_metrics,
            log_file=args.log_file,
        )
        save_summary(Path(context["summary_path"]), summary)
        append_wrapper_section(
            args.log_file,
            "summary",
            [
                f"Status: {summary['status']}",
                f"Elapsed: {summary['elapsed_human']}",
                f"Latest checkpoint dir: {checkpoint_summary.get('latest_checkpoint_dir')}",
                f"Checkpoint files: {checkpoint_summary.get('checkpoint_file_count')}",
                f"Latest step: {summary['train_metrics_log'].get('latest_step')}",
                f"Latest loss: {summary['train_metrics_log'].get('latest_loss')}",
                f"Summary JSON: {context['summary_path']}",
            ],
        )
        LOGGER.info("Saved training summary to %s", context["summary_path"])
    except (FileNotFoundError, RuntimeError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        LOGGER.error("%s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
