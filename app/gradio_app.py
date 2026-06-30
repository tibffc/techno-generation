# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr

try:
    from app.gradio_helpers import (
        StageArtifacts,
        StageConfig,
        StageRunner,
        StageSummary,
        count_audio_files,
        count_csv_rows_and_columns,
        diagnostics_snapshot,
        execute_hard_reset,
        ffmpeg_status,
        format_hard_reset_plan,
        hard_reset_protected_paths,
        latest_summary_text,
        list_logs,
        list_run_ids,
        list_audio_files,
        read_stage_run_report,
        read_full_log,
        read_json_file,
        scan_path_stats,
    )
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from app.gradio_helpers import (
        StageArtifacts,
        StageConfig,
        StageRunner,
        StageSummary,
        count_audio_files,
        count_csv_rows_and_columns,
        diagnostics_snapshot,
        execute_hard_reset,
        ffmpeg_status,
        format_hard_reset_plan,
        hard_reset_protected_paths,
        latest_summary_text,
        list_logs,
        list_run_ids,
        list_audio_files,
        read_stage_run_report,
        read_full_log,
        read_json_file,
        scan_path_stats,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
LATEST_ROOT = REPO_ROOT / "outputs" / "latest"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "train_lora.yaml"
OUTPUT_MODES = ["Use existing outputs", "Recreate outputs"]
PREPROCESS_OUTPUT_MODES = ["Use existing tensors", "Recreate tensors"]
TRAIN_OUTPUT_MODES = ["Continue from existing checkpoint", "Recreate training run"]
PIPELINE_MENU = [
    "Dataset cleaning",
    "Dataset prepare",
    "Metadata",
    "Manifests",
    "Training",
    "Generation",
    "Evaluation",
    "Logs",
    "Hard reset",
    "Diagnostics",
]
CSS = """
.app-shell {gap: 14px;}
.sidebar {border-right: 1px solid #d4d4d8; padding-right: 12px;}
.sidebar .gradio-accordion {margin-top: 12px;}
.stage-panel textarea {font-family: Consolas, ui-monospace, monospace;}
.stage-title-row {display:flex; flex-wrap:wrap; align-items:center; gap:10px; margin:0 0 8px 0;}
.stage-title-row h2 {margin:0 10px 0 0;}
.status-pill {border-radius:999px; padding:3px 10px; font-size:12px; font-weight:700; text-transform:uppercase;}
.status-idle {background:#e5e7eb; color:#374151;}
.status-running, .status-stopping {background:#dbeafe; color:#1d4ed8;}
.status-success {background:#dcfce7; color:#166534;}
.status-failed {background:#fee2e2; color:#991b1b;}
.status-stopped {background:#fef3c7; color:#92400e;}
.status-meta {font-size:13px; color:#52525b;}
"""

AUTOSCROLL_HTML = """
<script>
(() => {
  const stageMap = {
    "Dataset cleaning": ["sidebar-clean", "section-clean"],
    "Dataset prepare": ["sidebar-prepare", "section-prepare"],
    "Metadata": ["sidebar-metadata", "section-metadata"],
    "Manifests": ["sidebar-manifests", "section-manifests"],
    "Training": ["sidebar-training", "section-training"],
    "Generation": ["sidebar-generation", "section-generation"],
    "Evaluation": ["sidebar-evaluation", "section-evaluation"],
    "Logs": ["sidebar-logs", "section-logs"],
    "Hard reset": ["sidebar-hard-reset", "section-hard-reset"],
    "Diagnostics": ["sidebar-diagnostics", "section-diagnostics"]
  };
  window.__pipelineActiveStage = window.__pipelineActiveStage || "Dataset cleaning";
  const applyStageVisibility = () => {
    const active = stageMap[window.__pipelineActiveStage] || stageMap["Dataset cleaning"];
    Object.values(stageMap).flat().forEach((id) => {
      const node = document.getElementById(id);
      if (!node) return;
      const show = active.includes(id);
      node.style.display = show ? "block" : "none";
      node.hidden = !show;
    });
    const section = document.getElementById(active[1]);
    if (!section) return;
    const text = section.textContent.toLowerCase();
    const running = text.includes("running") || text.includes("stopping");
    section.querySelectorAll('button').forEach((button) => {
      const label = (button.textContent || "").trim();
      if (label === "Run stage") button.style.display = running ? "none" : "";
      if (label === "Stop stage") button.style.display = running ? "" : "none";
    });
  };
  window.applyPipelineStage = (label) => {
    window.__pipelineActiveStage = label;
    applyStageVisibility();
    setTimeout(applyStageVisibility, 100);
    setTimeout(applyStageVisibility, 600);
  };
  const bindMenu = () => {
    document.querySelectorAll('button').forEach((button) => {
      const label = (button.textContent || "").trim();
      if (!stageMap[label] || button.dataset.pipelineBound === "1") return;
      button.dataset.pipelineBound = "1";
      button.addEventListener("click", () => {
        window.applyPipelineStage(label);
      });
    });
  };
  const refreshVisibleStage = (force = false) => {
    const active = stageMap[window.__pipelineActiveStage] || stageMap["Dataset cleaning"];
    const section = document.getElementById(active[1]);
    if (!section) return;
    const button = Array.from(section.querySelectorAll('button')).find((item) => (item.textContent || '').trim() === 'Refresh stage state');
    if (!button) return;
    const token = `${window.__pipelineActiveStage}:${force ? 'force' : 'soft'}`;
    const now = Date.now();
    const lastToken = button.dataset.pipelineRefreshToken || "";
    const lastAt = Number(button.dataset.pipelineRefreshAt || "0");
    if (!force && lastToken === token && now - lastAt < 3000) return;
    button.dataset.pipelineRefreshToken = token;
    button.dataset.pipelineRefreshAt = String(now);
    button.click();
  };
  const observer2 = new MutationObserver(() => { bindMenu(); applyStageVisibility(); });
  observer2.observe(document.body, { childList: true, subtree: true, attributes: true });
  setInterval(() => { bindMenu(); applyStageVisibility(); }, 500);
  setInterval(() => { refreshVisibleStage(false); }, 2000);
  window.addEventListener('load', () => {
    bindMenu();
    applyStageVisibility();
    setTimeout(() => refreshVisibleStage(true), 300);
    setTimeout(() => refreshVisibleStage(true), 1200);
  });
  const originalApply = window.applyPipelineStage;
  window.applyPipelineStage = (label) => {
    originalApply(label);
    setTimeout(() => refreshVisibleStage(true), 150);
    setTimeout(() => refreshVisibleStage(true), 900);
  };
})();
</script>
"""


def _script(name: str) -> Path:
    return (REPO_ROOT / "scripts" / name).resolve()


def _resolve_path(value: str | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _append_value(command: list[str], flag: str, value: object | None) -> None:
    if value is None:
        return
    text = str(value).strip()
    if text:
        command.extend([flag, text])


def _append_optional_int(command: list[str], flag: str, value: object | None) -> None:
    if value is None or str(value).strip() == "":
        return
    command.extend([flag, str(int(value))])


def _append_optional_float(command: list[str], flag: str, value: object | None) -> None:
    if value is None or str(value).strip() == "":
        return
    command.extend([flag, str(float(value))])


def _rows_columns(path: Path | None) -> dict[str, Any]:
    info = count_csv_rows_and_columns(path)
    return {"path": str(path.resolve()) if path else None, **info}


def _with_examples(payload: dict[str, Any], limit: int = 20) -> dict[str, Any]:
    examples = payload.get("examples") or {}
    trimmed: dict[str, Any] = {}
    for key, value in examples.items():
        if isinstance(value, list):
            trimmed[key] = value[:limit]
        else:
            trimmed[key] = value
    payload["examples"] = trimmed
    return payload


def _copy_file_if_needed(source: Path | None, target: Path | None) -> None:
    if source is None or target is None or not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == target.resolve():
        return
    shutil.copy2(source, target)


def _latest_valid_checkpoint(prompt_mode: str = "features") -> Path | None:
    root = (LATEST_ROOT / "checkpoints" / prompt_mode).resolve()
    candidates: list[Path] = []
    for base in [root / "final", root / "latest"]:
        if (base / "adapter_config.json").exists() and (base / "adapter_model.safetensors").exists():
            candidates.append(base)
    checkpoints_dir = root / "checkpoints"
    if checkpoints_dir.exists():
        candidates.extend(
            path
            for path in checkpoints_dir.iterdir()
            if path.is_dir() and (path / "adapter_config.json").exists() and (path / "adapter_model.safetensors").exists()
        )
    if not candidates:
        train_stage_root = (REPO_ROOT / "outputs" / "runs" / "train_lora_train").resolve()
        if train_stage_root.exists():
            run_candidates = [
                item
                for item in train_stage_root.rglob("summary.json")
                if item.is_file()
            ]
            for summary_path in sorted(run_candidates, key=lambda item: item.stat().st_mtime, reverse=True):
                payload = read_json_file(summary_path) or {}
                checkpoint_dir = (
                    payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
                )
                latest_dir = checkpoint_dir.get("latest_checkpoint_dir")
                if latest_dir:
                    resolved = Path(str(latest_dir)).resolve()
                    if resolved.exists():
                        return resolved
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def _generation_checkpoint_value(values: dict[str, Any]) -> str:
    explicit = str(values.get("checkpoint_dir") or "").strip()
    if explicit:
        return explicit
    latest = _latest_valid_checkpoint(str(values.get("prompt_mode") or "features"))
    return str(latest) if latest else ""


def _checkpoint_value_for_prompt_mode(prompt_mode: str) -> str:
    latest = _latest_valid_checkpoint(prompt_mode)
    return str(latest) if latest else ""


def _ffmpeg_install_note() -> str:
    return "Install ffmpeg/ffprobe: `winget install Gyan.FFmpeg` or download FFmpeg and add its `bin` directory to `PATH`."


def _generation_metadata_value(values: dict[str, Any]) -> str:
    explicit = str(values.get("metadata_path") or "").strip()
    if explicit:
        return explicit
    cfg = _resolve_path(values.get("config")) or DEFAULT_CONFIG
    payload = read_json_file(cfg) if cfg.suffix.lower() == ".json" else None
    if payload:
        return str(payload.get("training", {}).get("metadata_path") or "")
    try:
        import yaml

        with cfg.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return str(data.get("training", {}).get("metadata_path") or "data/metadata.csv")
    except Exception:
        return "data/metadata.csv"


def _generation_audio_choices(output_dir_value: str | None) -> tuple[list[str], str | None, str]:
    output_dir = _resolve_path(output_dir_value)
    choices = list_audio_files(output_dir, {".wav", ".mp3", ".flac", ".ogg"})
    selected = choices[0] if choices else None
    message = "No generated audio available." if not choices else f"{len(choices)} generated audio file(s)."
    return choices, selected, message


def _path_output(path: Path | None) -> dict[str, Any]:
    return scan_path_stats(path)


def _backend_warnings() -> list[str]:
    backend_root = REPO_ROOT / "external" / "ACE-Step-1.5"
    backend_python = backend_root / ".venv" / "Scripts" / "python.exe"
    checkpoints = backend_root / "checkpoints"
    train_script = backend_root / "train.py"
    warnings: list[str] = []
    if not backend_root.exists():
        warnings.append(f"missing backend: {backend_root.resolve()}")
    if not backend_python.exists():
        warnings.append(f"missing backend python: {backend_python.resolve()}")
    if not checkpoints.exists() or not any(checkpoints.iterdir()):
        warnings.append(f"missing backend checkpoints/config: {checkpoints.resolve()}")
    if not train_script.exists():
        warnings.append(f"missing ACE-Step train script: {train_script.resolve()}")
    return warnings


def clean_command(values: dict[str, Any], artifacts: StageArtifacts) -> list[str]:
    command = [sys.executable, str(_script("clean_dataset.py"))]
    _append_value(command, "--input_dir", values["input_dir"])
    _append_value(command, "--output_dir", values["output_dir"])
    _append_value(command, "--report_path", str(artifacts.summary_json))
    _append_value(command, "--copy_mode", values["copy_mode"])
    _append_value(command, "--verify_mode", values["verify_mode"])
    if values["output_mode"] == "Use existing outputs":
        command.append("--reuse_existing_output")
    _append_optional_int(command, "--max_files", values["max_files"])
    _append_optional_int(command, "--min_size_kb", values["min_size_kb"])
    _append_value(command, "--hash_algo", values["hash_algo"])
    _append_value(command, "--log_file", str(artifacts.log_file))
    return command


def clean_expected(values: dict[str, Any]) -> list[Path]:
    return [_resolve_path(values["output_dir"]) or REPO_ROOT / "data" / "raw_clean"]


def clean_cleanup(values: dict[str, Any]) -> list[Path]:
    return [_resolve_path(values["output_dir"]) or REPO_ROOT / "data" / "raw_clean"]


def clean_preflight(values: dict[str, Any]) -> dict[str, Any]:
    input_dir = _resolve_path(values["input_dir"])
    output_dir = _resolve_path(values["output_dir"])
    warnings: list[str] = []
    if input_dir is None or not input_dir.exists():
        warnings.append(f"input_dir does not exist: {input_dir}")
    return {
        "inputs": {"input_dir": scan_path_stats(input_dir)},
        "outputs": {"output_dir": scan_path_stats(output_dir)},
        "notes": [],
        "warnings": warnings,
    }


def clean_summary(context: dict[str, Any]) -> StageSummary:
    state = context["state"]
    artifacts = context["artifacts"]
    values = context["values"]
    payload = read_json_file(artifacts.summary_json) or {}
    input_dir = _resolve_path(values["input_dir"])
    output_dir = _resolve_path(values["output_dir"])
    return StageSummary(
        stage="clean_dataset",
        title="Dataset cleaning",
        run_id=state.run_id,
        status=state.status,
        started_at=state.started_at,
        finished_at=state.finished_at,
        elapsed_seconds=state.elapsed_seconds,
        output_mode=state.output_mode,
        inputs={"input_dir": scan_path_stats(input_dir)},
        outputs={"output_dir": scan_path_stats(output_dir)},
        metrics={
            "total_files_found": payload.get("total_files_found"),
            "candidate_audio_files": payload.get("candidate_audio_files"),
            "valid_audio_files": payload.get("valid_audio_files"),
            "copied_files": payload.get("copied_files"),
            "reused_files": payload.get("reused_files"),
            "duplicate_files": payload.get("duplicate_files"),
            "skipped_unsupported": payload.get("skipped_unsupported"),
            "skipped_small": payload.get("skipped_small_files"),
            "corrupted_files": payload.get("corrupted_files"),
            "error_files": payload.get("error_files"),
        },
        artifacts={},
        warnings=[],
        errors=[state.error_message] if state.error_message else [],
        notes=[],
        diagnostics={},
    )


def prepare_command(values: dict[str, Any], artifacts: StageArtifacts) -> list[str]:
    command = [sys.executable, str(_script("prepare_dataset.py"))]
    _append_value(command, "--input_dir", values["input_dir"])
    _append_value(command, "--output_dir", values["output_dir"])
    _append_optional_float(command, "--chunk_seconds", values["chunk_seconds"])
    _append_optional_float(command, "--overlap_seconds", values["overlap_seconds"])
    _append_optional_int(command, "--sample_rate", values["sample_rate"])
    _append_optional_float(command, "--min_duration_seconds", values["min_duration_seconds"])
    _append_optional_float(command, "--target_lufs", values["target_lufs"])
    _append_value(command, "--audio_format", values["audio_format"])
    _append_value(command, "--metadata_path", values["metadata_path"])
    if values["output_mode"] == "Use existing outputs":
        command.append("--reuse_existing_chunks")
    _append_value(command, "--log_file", str(artifacts.log_file))
    return command


def prepare_expected(values: dict[str, Any]) -> list[Path]:
    return [
        _resolve_path(values["output_dir"]) or REPO_ROOT / "data" / "processed",
        _resolve_path(values["metadata_path"]) or REPO_ROOT / "data" / "processed" / "metadata.csv",
    ]


def prepare_cleanup(values: dict[str, Any]) -> list[Path]:
    return prepare_expected(values)


def prepare_preflight(values: dict[str, Any]) -> dict[str, Any]:
    input_dir = _resolve_path(values["input_dir"])
    output_dir = _resolve_path(values["output_dir"])
    metadata_path = _resolve_path(values["metadata_path"])
    warnings: list[str] = []
    if input_dir is None or not input_dir.exists():
        warnings.append(f"input_dir does not exist: {input_dir}")
    return {
        "inputs": {"input_dir": scan_path_stats(input_dir)},
        "outputs": {
            "output_dir": scan_path_stats(output_dir),
            "metadata_path": scan_path_stats(metadata_path),
        },
        "notes": [f"candidate_audio_files={count_audio_files(input_dir)}"],
        "warnings": warnings,
    }


def prepare_summary(context: dict[str, Any]) -> StageSummary:
    state = context["state"]
    values = context["values"]
    input_dir = _resolve_path(values["input_dir"])
    output_dir = _resolve_path(values["output_dir"])
    metadata_path = _resolve_path(values["metadata_path"])
    return StageSummary(
        stage="prepare_dataset",
        title="Dataset prepare",
        run_id=state.run_id,
        status=state.status,
        started_at=state.started_at,
        finished_at=state.finished_at,
        elapsed_seconds=state.elapsed_seconds,
        output_mode=state.output_mode,
        inputs={"input_dir": scan_path_stats(input_dir)},
        outputs={
            "output_dir": scan_path_stats(output_dir),
            "metadata_path": _rows_columns(metadata_path),
        },
        metrics={
            "chunks_created": count_audio_files(output_dir, {".wav", ".flac"}),
            "metadata_rows": count_csv_rows_and_columns(metadata_path).get("rows"),
            "warnings": 0,
        },
        artifacts={},
        warnings=[],
        errors=[state.error_message] if state.error_message else [],
        notes=[],
    )


def metadata_command(values: dict[str, Any], artifacts: StageArtifacts) -> list[str]:
    command = [sys.executable, str(_script("make_metadata.py"))]
    _append_value(command, "--input_dir", values["input_dir"])
    _append_value(command, "--metadata_out", values["metadata_out"])
    _append_value(command, "--features_out", values["features_out"])
    _append_optional_int(command, "--clusters", values["clusters"])
    _append_optional_int(command, "--seed", values["seed"])
    _append_value(command, "--generic_prompt", values["generic_prompt"])
    _append_value(command, "--device", values["device"])
    _append_optional_int(command, "--batch_size", values["batch_size"])
    _append_optional_int(command, "--sample_rate", values["sample_rate"])
    _append_optional_int(command, "--max_files", values["max_files"])
    _append_value(command, "--quality_report_out", str(artifacts.summary_json))
    _append_value(command, "--log_file", str(artifacts.log_file))
    return command


def metadata_expected(values: dict[str, Any]) -> list[Path]:
    return [
        _resolve_path(values["metadata_out"]) or REPO_ROOT / "data" / "metadata.csv",
        _resolve_path(values["features_out"]) or REPO_ROOT / "data" / "metadata_features.csv",
    ]


def metadata_cleanup(values: dict[str, Any]) -> list[Path]:
    return metadata_expected(values)


def metadata_preflight(values: dict[str, Any]) -> dict[str, Any]:
    input_dir = _resolve_path(values["input_dir"])
    warnings: list[str] = []
    if input_dir is None or not input_dir.exists():
        warnings.append(f"input_dir does not exist: {input_dir}")
    if input_dir and input_dir.exists() and count_audio_files(input_dir, {".wav", ".flac"}) == 0:
        warnings.append(f"input_dir has no .wav/.flac chunks: {input_dir}")
    return {
        "inputs": {"input_dir": scan_path_stats(input_dir)},
        "outputs": {
            "metadata_out": scan_path_stats(_resolve_path(values["metadata_out"])),
            "features_out": scan_path_stats(_resolve_path(values["features_out"])),
        },
        "notes": [f"device={values['device']}", f"sample_rate={values['sample_rate']}"],
        "warnings": warnings,
    }


def metadata_summary(context: dict[str, Any]) -> StageSummary:
    state = context["state"]
    artifacts = context["artifacts"]
    values = context["values"]
    quality_payload = read_json_file(artifacts.summary_json) or {}
    metadata_path = _resolve_path(values["metadata_out"])
    features_path = _resolve_path(values["features_out"])
    return StageSummary(
        stage="make_metadata",
        title="Metadata",
        run_id=state.run_id,
        status=state.status,
        started_at=state.started_at,
        finished_at=state.finished_at,
        elapsed_seconds=state.elapsed_seconds,
        output_mode=state.output_mode,
        inputs={"input_dir": scan_path_stats(_resolve_path(values["input_dir"]))},
        outputs={
            "metadata.csv": _rows_columns(metadata_path),
            "metadata_features.csv": _rows_columns(features_path),
        },
        metrics={
            "audio_files_found": count_audio_files(_resolve_path(values["input_dir"]), {".wav", ".flac"}),
            "clusters": int(values["clusters"]),
            "device": values["device"],
            "sample_rate": int(values["sample_rate"]),
            "bpm_corrected_count": quality_payload.get("bpm_corrected_count"),
        },
        artifacts={},
        warnings=[],
        errors=[state.error_message] if state.error_message else [],
        notes=[],
    )


def manifest_command(values: dict[str, Any], artifacts: StageArtifacts) -> list[str]:
    command = [sys.executable, str(_script("export_acestep_manifest.py"))]
    _append_value(command, "--metadata_csv", values["metadata_csv"])
    _append_value(command, "--prompt_mode", values["prompt_mode"])
    _append_value(command, "--jsonl_out", values["jsonl_out"])
    _append_value(command, "--dataset_json_out", values["dataset_json_out"])
    _append_value(command, "--log_file", str(artifacts.log_file))
    return command


def manifest_expected(values: dict[str, Any]) -> list[Path]:
    return [
        _resolve_path(values["jsonl_out"]) or LATEST_ROOT / "manifests" / "features" / "manifest.jsonl",
        _resolve_path(values["dataset_json_out"]) or LATEST_ROOT / "manifests" / "features" / "dataset.json",
    ]


def manifest_cleanup(values: dict[str, Any]) -> list[Path]:
    return manifest_expected(values)


def manifest_preflight(values: dict[str, Any]) -> dict[str, Any]:
    metadata_csv = _resolve_path(values["metadata_csv"])
    warnings = [] if metadata_csv and metadata_csv.exists() else [f"metadata_csv does not exist: {metadata_csv}"]
    return {
        "inputs": {"metadata_csv": scan_path_stats(metadata_csv)},
        "outputs": {
            "jsonl_out": scan_path_stats(_resolve_path(values["jsonl_out"])),
            "dataset_json_out": scan_path_stats(_resolve_path(values["dataset_json_out"])),
        },
        "notes": [f"prompt_mode={values['prompt_mode']}"],
        "warnings": warnings,
    }


def manifest_summary(context: dict[str, Any]) -> StageSummary:
    state = context["state"]
    values = context["values"]
    jsonl_out = _resolve_path(values["jsonl_out"])
    dataset_json = _resolve_path(values["dataset_json_out"])
    jsonl_rows = 0
    if jsonl_out and jsonl_out.exists():
        with jsonl_out.open("r", encoding="utf-8", errors="replace") as handle:
            jsonl_rows = sum(1 for line in handle if line.strip())
    dataset_payload = read_json_file(dataset_json) or {}
    return StageSummary(
        stage="export_manifest",
        title="Manifests",
        run_id=state.run_id,
        status=state.status,
        started_at=state.started_at,
        finished_at=state.finished_at,
        elapsed_seconds=state.elapsed_seconds,
        output_mode=state.output_mode,
        inputs={"metadata_csv": scan_path_stats(_resolve_path(values["metadata_csv"]))},
        outputs={"jsonl_out": scan_path_stats(jsonl_out), "dataset_json_out": scan_path_stats(dataset_json)},
        metrics={"train_rows": jsonl_rows, "dataset_json_samples": len(dataset_payload.get("samples", []))},
        artifacts={},
        warnings=[],
        errors=[state.error_message] if state.error_message else [],
        notes=[],
    )


def training_command(values: dict[str, Any], artifacts: StageArtifacts, preprocess_only: bool) -> list[str]:
    command = [sys.executable, "-u", str(_script("train_lora.py")), "--config", str(values["config"]), "--prompt_mode", values["prompt_mode"]]
    if preprocess_only:
        command.append("--preprocess_only")
    else:
        command.append("--run")
    return command


def training_preprocess_expected(values: dict[str, Any]) -> list[Path]:
    return [LATEST_ROOT / "tensors" / values["prompt_mode"]]


def training_preprocess_cleanup(values: dict[str, Any]) -> list[Path]:
    return training_preprocess_expected(values)


def training_train_expected(values: dict[str, Any]) -> list[Path]:
    return [LATEST_ROOT / "checkpoints" / values["prompt_mode"]]


def training_train_cleanup(values: dict[str, Any]) -> list[Path]:
    return training_train_expected(values)


def training_preflight(values: dict[str, Any], preprocess_only: bool) -> dict[str, Any]:
    cfg = _resolve_path(values["config"])
    warnings = [] if cfg and cfg.exists() else [f"config does not exist: {cfg}"]
    kind = "preprocess" if preprocess_only else "train"
    return {
        "inputs": {"config": scan_path_stats(cfg)},
        "outputs": {
            "tensors": scan_path_stats(LATEST_ROOT / "tensors" / values["prompt_mode"]),
            "checkpoints": scan_path_stats(LATEST_ROOT / "checkpoints" / values["prompt_mode"]),
        },
        "notes": [f"prompt_mode={values['prompt_mode']}", f"mode={kind}"],
        "warnings": warnings,
    }


def training_summary(context: dict[str, Any], title: str, stage_name: str, extra_output: Path) -> StageSummary:
    state = context["state"]
    summary_payload = read_json_file(context["artifacts"].summary_json) or {}
    metrics: dict[str, Any] = {"exit_code": state.exit_code}
    outputs: dict[str, Any] = {"output_dir": scan_path_stats(extra_output)}
    if stage_name == "train_lora_train":
        metrics["final_loss"] = summary_payload.get("final_loss")
        metrics["best_loss"] = summary_payload.get("best_loss")
        checkpoint_summary = summary_payload.get("checkpoint_summary") or {}
        checkpoint_metadata = summary_payload.get("checkpoint_metadata") or {}
        outputs["latest_checkpoint_dir"] = checkpoint_summary.get("latest_checkpoint_dir")
        outputs["checkpoint_metadata"] = checkpoint_summary.get("metadata_path") or checkpoint_metadata.get("checkpoint_root")
    return StageSummary(
        stage=stage_name,
        title=title,
        run_id=state.run_id,
        status=state.status,
        started_at=state.started_at,
        finished_at=state.finished_at,
        elapsed_seconds=state.elapsed_seconds,
        output_mode=state.output_mode,
        inputs={"config": scan_path_stats(_resolve_path(context["values"]["config"]))},
        outputs=outputs,
        metrics=metrics,
        artifacts={},
        warnings=[],
        errors=[state.error_message] if state.error_message else [],
        notes=[],
    )


def validation_command(values: dict[str, Any], artifacts: StageArtifacts) -> list[str]:
    command = [sys.executable, "-u", str(_script("validate_training.py"))]
    _append_value(command, "--config", values["config"])
    _append_value(command, "--prompt_mode", values["prompt_mode"])
    _append_value(command, "--checkpoint_dir", values["checkpoint_dir"])
    _append_value(command, "--device", values["device"])
    return command


def validation_expected(values: dict[str, Any]) -> list[Path]:
    return [_resolve_path(values["checkpoint_dir"]) or LATEST_ROOT / "checkpoints" / values["prompt_mode"]]


def validation_preflight(values: dict[str, Any]) -> dict[str, Any]:
    checkpoint_dir = _resolve_path(values["checkpoint_dir"]) or _latest_valid_checkpoint(values["prompt_mode"])
    checkpoint_root = LATEST_ROOT / "checkpoints" / values["prompt_mode"]
    resolved_target = checkpoint_dir or checkpoint_root
    warnings = _backend_warnings()
    ffmpeg = ffmpeg_status()
    if not ffmpeg["ready"]:
        warnings.append(_ffmpeg_install_note())
    if not resolved_target.exists():
        warnings.append(f"missing checkpoint: {resolved_target.resolve()}")
    return {
        "inputs": {
            "config": scan_path_stats(_resolve_path(values["config"])),
            "checkpoint_dir": scan_path_stats(resolved_target),
        },
        "outputs": {"validation_target": scan_path_stats(resolved_target)},
        "notes": [f"device={values['device']}", f"checkpoint_selected={resolved_target.resolve() if resolved_target else '--'}"],
        "warnings": warnings,
    }


def validation_summary(context: dict[str, Any]) -> StageSummary:
    return training_summary(
        context,
        title="Pipeline health check",
        stage_name="validate_training",
        extra_output=_resolve_path(context["values"]["checkpoint_dir"]) or (LATEST_ROOT / "checkpoints" / context["values"]["prompt_mode"]),
    )


def generation_command(values: dict[str, Any], artifacts: StageArtifacts) -> list[str]:
    command = [sys.executable, "-u", str(_script("generate_samples.py"))]
    _append_value(command, "--config", values["config"])
    _append_value(command, "--checkpoint_dir", _generation_checkpoint_value(values))
    _append_value(command, "--prompt_mode", values["prompt_mode"])
    _append_value(command, "--prompt", values["prompt"])
    _append_value(command, "--prompt_strategy", values["prompt_strategy"])
    _append_optional_int(command, "--num_samples", values["num_samples"])
    _append_optional_float(command, "--duration", values["duration"])
    _append_value(command, "--output_dir", values["output_dir"])
    _append_value(command, "--device", values["device"])
    _append_optional_int(command, "--seed", values["seed"])
    _append_value(command, "--metadata_path", _generation_metadata_value(values))
    _append_value(command, "--model_variant", values["model_variant"])
    return command


def generation_expected(values: dict[str, Any]) -> list[Path]:
    return [_resolve_path(values["output_dir"]) or LATEST_ROOT / "generated"]


def generation_cleanup(values: dict[str, Any]) -> list[Path]:
    return generation_expected(values)


def generation_preflight(values: dict[str, Any]) -> dict[str, Any]:
    output_dir = _resolve_path(values["output_dir"])
    checkpoint_value = _generation_checkpoint_value(values)
    checkpoint_dir = _resolve_path(checkpoint_value)
    warnings = _backend_warnings()
    ffmpeg = ffmpeg_status()
    if not ffmpeg["ready"]:
        warnings.append(_ffmpeg_install_note())
    if checkpoint_dir is None:
        warnings.append(f"missing checkpoint: no valid checkpoint found for prompt_mode={values['prompt_mode']}")
    elif not checkpoint_dir.exists():
        warnings.append(f"missing checkpoint: {checkpoint_dir.resolve()}")
    config_path = _resolve_path(values["config"])
    metadata_path = _resolve_path(_generation_metadata_value(values))
    if config_path is None or not config_path.exists():
        warnings.append(f"missing config: {config_path}")
    if metadata_path is None or not metadata_path.exists():
        warnings.append(f"missing metadata_path: {metadata_path}")
    if output_dir is not None:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            probe = output_dir / ".pipeline_write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
        except OSError as error:
            warnings.append(f"output_dir is not writable: {output_dir.resolve()} ({error})")
    return {
        "inputs": {
            "config": scan_path_stats(config_path),
            "checkpoint_dir": scan_path_stats(checkpoint_dir),
            "metadata_path": scan_path_stats(metadata_path),
        },
        "outputs": {"output_dir": scan_path_stats(output_dir)},
        "notes": [
            f"num_samples={int(values['num_samples'])}",
            f"duration={float(values['duration'])}",
            f"checkpoint_selected={checkpoint_value or '--'}",
            f"checkpoint_auto_selected={'yes' if not str(values.get('checkpoint_dir') or '').strip() and checkpoint_value else 'no'}",
        ],
        "warnings": warnings,
    }


def generation_summary(context: dict[str, Any]) -> StageSummary:
    state = context["state"]
    values = context["values"]
    payload = read_json_file(context["artifacts"].summary_json) or {}
    output_dir = _resolve_path(payload.get("output_dir") or values["output_dir"])
    generated_files = [str(item) for item in payload.get("generated_files", [])]
    if not generated_files and output_dir and output_dir.exists():
        generated_files = list_audio_files(output_dir, {".wav", ".mp3", ".flac", ".ogg"})
    return StageSummary(
        stage="generate_samples",
        title="Generation",
        run_id=state.run_id,
        status=state.status,
        started_at=state.started_at,
        finished_at=state.finished_at,
        elapsed_seconds=state.elapsed_seconds,
        output_mode=state.output_mode,
        inputs={
            "checkpoint_dir": scan_path_stats(_resolve_path(_generation_checkpoint_value(values))),
            "selected_checkpoint": _generation_checkpoint_value(values) or "--",
        },
        outputs={"output_dir": scan_path_stats(output_dir), "generated_audio": generated_files},
        metrics={
            "generated_files": payload.get("generated_files_count", len(generated_files)),
            "generation_run_name": payload.get("generation_run_name"),
            "num_samples_requested": int(values["num_samples"]),
        },
        artifacts={},
        warnings=[],
        errors=[state.error_message] if state.error_message else [],
        notes=[],
    )


def evaluation_command(values: dict[str, Any], artifacts: StageArtifacts) -> list[str]:
    command = [sys.executable, "-u", str(_script("evaluate.py"))]
    _append_value(command, "--dataset_metadata", values["dataset_metadata"])
    _append_value(command, "--dataset_features", values["dataset_features"])
    _append_value(command, "--base_dir", values["base_dir"])
    _append_value(command, "--generic_dir", values["generic_dir"])
    _append_value(command, "--feature_dir", values["feature_dir"])
    for line in str(values["sample_dirs"]).splitlines():
        line = line.strip()
        if line:
            command.extend(["--sample_dir", line])
    _append_value(command, "--output_dir", values["output_dir"])
    _append_optional_int(command, "--sample_rate", values["sample_rate"])
    _append_optional_float(command, "--tempo_min", values["tempo_min"])
    _append_optional_float(command, "--tempo_max", values["tempo_max"])
    _append_optional_int(command, "--plot_min_samples", values["plot_min_samples"])
    _append_value(command, "--log_file", str(artifacts.log_file))
    return command


def evaluation_expected(values: dict[str, Any]) -> list[Path]:
    return [_resolve_path(values["output_dir"]) or REPO_ROOT / "outputs" / "evaluation"]


def evaluation_cleanup(values: dict[str, Any]) -> list[Path]:
    return evaluation_expected(values)


def evaluation_preflight(values: dict[str, Any]) -> dict[str, Any]:
    warnings: list[str] = []
    dataset_metadata = _resolve_path(values["dataset_metadata"])
    dataset_features = _resolve_path(values["dataset_features"])
    if dataset_metadata is None or not dataset_metadata.exists():
        warnings.append(f"dataset_metadata does not exist: {dataset_metadata}")
    if dataset_features is None or not dataset_features.exists():
        warnings.append(f"dataset_features does not exist: {dataset_features}")
    return {
        "inputs": {
            "dataset_metadata": scan_path_stats(dataset_metadata),
            "dataset_features": scan_path_stats(dataset_features),
        },
        "outputs": {"output_dir": scan_path_stats(_resolve_path(values["output_dir"]))},
        "notes": [],
        "warnings": warnings,
    }


def evaluation_summary(context: dict[str, Any]) -> StageSummary:
    state = context["state"]
    values = context["values"]
    output_dir = _resolve_path(values["output_dir"])
    summary_payload = read_json_file(context["artifacts"].summary_json)
    return StageSummary(
        stage="evaluate",
        title="Evaluation",
        run_id=state.run_id,
        status=state.status,
        started_at=state.started_at,
        finished_at=state.finished_at,
        elapsed_seconds=state.elapsed_seconds,
        output_mode=state.output_mode,
        inputs={
            "dataset_metadata": scan_path_stats(_resolve_path(values["dataset_metadata"])),
            "dataset_features": scan_path_stats(_resolve_path(values["dataset_features"])),
        },
        outputs={"output_dir": scan_path_stats(output_dir)},
        metrics={
            "generated_files_count": summary_payload.get("overall_metrics", {}).get("generated_files_count") if summary_payload else None,
            "duration_seconds_mean": summary_payload.get("overall_metrics", {}).get("duration_seconds_mean") if summary_payload else None,
            "sample_rate_hz": summary_payload.get("overall_metrics", {}).get("sample_rate_hz") if summary_payload else None,
            "peak_level_max": summary_payload.get("overall_metrics", {}).get("peak_level_max") if summary_payload else None,
            "clipping_count_total": summary_payload.get("overall_metrics", {}).get("clipping_count_total") if summary_payload else None,
            "loudness_rms_dbfs_mean": summary_payload.get("overall_metrics", {}).get("loudness_rms_dbfs_mean") if summary_payload else None,
            "spectral_centroid_mean": summary_payload.get("overall_metrics", {}).get("spectral_centroid_mean") if summary_payload else None,
            "tempo_bpm_mean": summary_payload.get("overall_metrics", {}).get("tempo_bpm_mean") if summary_payload else None,
            "warnings_count": summary_payload.get("warnings_count") if summary_payload else None,
            "errors_count": summary_payload.get("errors_count") if summary_payload else None,
            "training_loss_note": summary_payload.get("training_loss_note") if summary_payload else None,
        },
        artifacts={},
        warnings=[],
        errors=[state.error_message] if state.error_message else [],
        notes=[],
    )


RUNNER = StageRunner(REPO_ROOT)
STAGE_CONFIGS = {
    "clean_dataset": StageConfig(
        stage="clean_dataset",
        title="Dataset cleaning",
        script_path=_script("clean_dataset.py"),
        output_mode_choices=OUTPUT_MODES,
        default_output_mode=OUTPUT_MODES[0],
        command_builder=clean_command,
        expected_outputs_builder=clean_expected,
        cleanup_targets_builder=clean_cleanup,
        preflight_builder=clean_preflight,
        summary_builder=clean_summary,
    ),
    "prepare_dataset": StageConfig(
        stage="prepare_dataset",
        title="Dataset prepare",
        script_path=_script("prepare_dataset.py"),
        output_mode_choices=OUTPUT_MODES,
        default_output_mode=OUTPUT_MODES[0],
        command_builder=prepare_command,
        expected_outputs_builder=prepare_expected,
        cleanup_targets_builder=prepare_cleanup,
        preflight_builder=prepare_preflight,
        summary_builder=prepare_summary,
    ),
    "make_metadata": StageConfig(
        stage="make_metadata",
        title="Metadata",
        script_path=_script("make_metadata.py"),
        output_mode_choices=OUTPUT_MODES,
        default_output_mode=OUTPUT_MODES[0],
        command_builder=metadata_command,
        expected_outputs_builder=metadata_expected,
        cleanup_targets_builder=metadata_cleanup,
        preflight_builder=metadata_preflight,
        summary_builder=metadata_summary,
    ),
    "export_manifest": StageConfig(
        stage="export_manifest",
        title="Manifests",
        script_path=_script("export_acestep_manifest.py"),
        output_mode_choices=OUTPUT_MODES,
        default_output_mode=OUTPUT_MODES[0],
        command_builder=manifest_command,
        expected_outputs_builder=manifest_expected,
        cleanup_targets_builder=manifest_cleanup,
        preflight_builder=manifest_preflight,
        summary_builder=manifest_summary,
    ),
    "train_lora_preprocess": StageConfig(
        stage="train_lora_preprocess",
        title="Training preprocess",
        script_path=_script("train_lora.py"),
        output_mode_choices=PREPROCESS_OUTPUT_MODES,
        default_output_mode=PREPROCESS_OUTPUT_MODES[0],
        command_builder=lambda values, artifacts: training_command(values, artifacts, preprocess_only=True),
        expected_outputs_builder=training_preprocess_expected,
        cleanup_targets_builder=training_preprocess_cleanup,
        preflight_builder=lambda values: training_preflight(values, preprocess_only=True),
        summary_builder=lambda context: training_summary(
            context,
            title="Training preprocess",
            stage_name="train_lora_preprocess",
            extra_output=LATEST_ROOT / "tensors" / context["values"]["prompt_mode"],
        ),
    ),
    "train_lora_train": StageConfig(
        stage="train_lora_train",
        title="Training",
        script_path=_script("train_lora.py"),
        output_mode_choices=TRAIN_OUTPUT_MODES,
        default_output_mode=TRAIN_OUTPUT_MODES[0],
        command_builder=lambda values, artifacts: training_command(values, artifacts, preprocess_only=False),
        expected_outputs_builder=training_train_expected,
        cleanup_targets_builder=training_train_cleanup,
        preflight_builder=lambda values: training_preflight(values, preprocess_only=False),
        summary_builder=lambda context: training_summary(
            context,
            title="Training",
            stage_name="train_lora_train",
            extra_output=LATEST_ROOT / "checkpoints" / context["values"]["prompt_mode"],
        ),
    ),
    "validate_training": StageConfig(
        stage="validate_training",
        title="Pipeline health check",
        script_path=_script("validate_training.py"),
        output_mode_choices=OUTPUT_MODES,
        default_output_mode=OUTPUT_MODES[0],
        command_builder=validation_command,
        expected_outputs_builder=validation_expected,
        cleanup_targets_builder=lambda values: [],
        preflight_builder=validation_preflight,
        summary_builder=validation_summary,
    ),
    "generate_samples": StageConfig(
        stage="generate_samples",
        title="Generation",
        script_path=_script("generate_samples.py"),
        output_mode_choices=OUTPUT_MODES,
        default_output_mode=OUTPUT_MODES[0],
        command_builder=generation_command,
        expected_outputs_builder=generation_expected,
        cleanup_targets_builder=generation_cleanup,
        preflight_builder=generation_preflight,
        summary_builder=generation_summary,
    ),
    "evaluate": StageConfig(
        stage="evaluate",
        title="Evaluation",
        script_path=_script("evaluate.py"),
        output_mode_choices=OUTPUT_MODES,
        default_output_mode=OUTPUT_MODES[0],
        command_builder=evaluation_command,
        expected_outputs_builder=evaluation_expected,
        cleanup_targets_builder=evaluation_cleanup,
        preflight_builder=evaluation_preflight,
        summary_builder=evaluation_summary,
    ),
}


@dataclass
class StageWidgets:
    names: list[str]
    inputs: list[gr.components.Component]
    preview: gr.Textbox
    status: gr.HTML
    logs: gr.Textbox
    summary: gr.Textbox


PARAMETER_GROUP_ORDER = ["Input / Output", "Processing", "Constraints", "Advanced"]

STAGE_PARAMETER_GROUPS = {
    "clean_dataset": {
        "Input / Output": ["input_dir", "output_dir", "output_mode"],
        "Processing": ["copy_mode", "verify_mode", "hash_algo"],
        "Constraints": ["max_files", "min_size_kb"],
    },
    "prepare_dataset": {
        "Input / Output": ["input_dir", "output_dir", "output_mode", "metadata_path"],
        "Processing": ["chunk_seconds", "overlap_seconds", "sample_rate", "min_duration_seconds", "target_lufs", "audio_format"],
    },
    "make_metadata": {
        "Input / Output": ["input_dir", "metadata_out", "features_out", "output_mode"],
        "Processing": ["clusters", "generic_prompt", "sample_rate"],
        "Constraints": ["max_files"],
        "Advanced": ["seed", "device", "batch_size"],
    },
    "export_manifest": {
        "Input / Output": ["metadata_csv", "output_mode", "jsonl_out", "dataset_json_out"],
        "Processing": ["prompt_mode"],
    },
    "train_lora_preprocess": {
        "Input / Output": ["config", "output_mode"],
        "Processing": ["prompt_mode"],
    },
    "train_lora_train": {
        "Input / Output": ["config", "output_mode"],
        "Processing": ["prompt_mode"],
    },
    "generate_samples": {
        "Input / Output": ["config", "checkpoint_dir", "output_dir", "output_mode", "metadata_path"],
        "Processing": ["prompt_mode", "prompt", "prompt_strategy", "num_samples", "duration", "model_variant"],
        "Advanced": ["device", "seed"],
    },
    "validate_training": {
        "Input / Output": ["config", "checkpoint_dir", "output_mode"],
        "Processing": ["prompt_mode"],
        "Advanced": ["device"],
    },
    "evaluate": {
        "Input / Output": ["dataset_metadata", "dataset_features", "output_dir", "output_mode", "base_dir", "generic_dir", "feature_dir", "sample_dirs"],
        "Processing": ["sample_rate"],
        "Constraints": ["tempo_min", "tempo_max", "plot_min_samples"],
    },
}

PARAMETER_COMPONENTS: dict[str, dict[str, Any]] = {
    "audio_format": {"type": "dropdown", "choices": ["flac", "wav"]},
    "batch_size": {"type": "number", "precision": 0},
    "base_dir": {"type": "textbox"},
    "checkpoint_dir": {"type": "textbox"},
    "chunk_seconds": {"type": "number"},
    "clusters": {"type": "number", "precision": 0},
    "config": {"type": "textbox"},
    "copy_mode": {"type": "dropdown", "choices": ["copy", "hardlink"]},
    "dataset_features": {"type": "textbox"},
    "dataset_json_out": {"type": "textbox"},
    "dataset_metadata": {"type": "textbox"},
    "device": {"type": "dropdown", "choices": ["auto", "cpu", "cuda"]},
    "duration": {"type": "number"},
    "feature_dir": {"type": "textbox"},
    "features_out": {"type": "textbox"},
    "generic_dir": {"type": "textbox"},
    "generic_prompt": {"type": "textbox"},
    "hash_algo": {"type": "dropdown", "choices": ["sha1", "md5"]},
    "input_dir": {"type": "textbox"},
    "jsonl_out": {"type": "textbox"},
    "max_files": {"type": "number", "precision": 0},
    "metadata_csv": {"type": "textbox"},
    "metadata_out": {"type": "textbox"},
    "metadata_path": {"type": "textbox"},
    "min_duration_seconds": {"type": "number"},
    "min_size_kb": {"type": "number", "precision": 0},
    "model_variant": {"type": "textbox"},
    "num_samples": {"type": "number", "precision": 0},
    "output_dir": {"type": "textbox"},
    "output_mode": {"type": "radio"},
    "overlap_seconds": {"type": "number"},
    "plot_min_samples": {"type": "number", "precision": 0},
    "prompt": {"type": "textbox", "lines": 3},
    "prompt_mode": {"type": "dropdown", "choices": ["generic", "features"]},
    "prompt_strategy": {"type": "dropdown", "choices": ["metadata", "diverse"]},
    "sample_dirs": {"type": "textbox", "label": "sample_dir entries (label=path, one per line)", "lines": 4},
    "sample_rate": {"type": "number", "precision": 0},
    "seed": {"type": "number", "precision": 0},
    "target_lufs": {"type": "number"},
    "tempo_max": {"type": "number"},
    "tempo_min": {"type": "number"},
    "verify_mode": {"type": "dropdown", "choices": ["none", "ffprobe", "decode"]},
}


def _parameter_component(stage_key: str, name: str, defaults: dict[str, Any]) -> gr.components.Component:
    spec = PARAMETER_COMPONENTS[name]
    label = spec.get("label", name)
    value = defaults.get(name)
    component_type = spec["type"]
    if component_type == "textbox":
        return gr.Textbox(label=label, value=value, lines=spec.get("lines", 1))
    if component_type == "number":
        return gr.Number(label=label, value=value, precision=spec.get("precision"))
    if component_type == "dropdown":
        return gr.Dropdown(label=label, choices=spec["choices"], value=value)
    if component_type == "radio":
        return gr.Radio(STAGE_CONFIGS[stage_key].output_mode_choices, value=value, label=label)
    raise ValueError(f"Unsupported parameter component type: {component_type}")


def _render_parameters(stage_key: str, defaults: dict[str, Any]) -> tuple[list[str], list[gr.components.Component]]:
    gr.Markdown("### Parameters")
    names: list[str] = []
    inputs: list[gr.components.Component] = []
    stage_groups = STAGE_PARAMETER_GROUPS[stage_key]
    for group_name in PARAMETER_GROUP_ORDER:
        parameter_names = stage_groups.get(group_name)
        if not parameter_names:
            continue
        container = gr.Accordion(group_name, open=False) if group_name == "Advanced" else gr.Group()
        with container:
            gr.Markdown(f"#### {group_name}")
            for parameter_name in parameter_names:
                component = _parameter_component(stage_key, parameter_name, defaults)
                names.append(parameter_name)
                inputs.append(component)
    return names, inputs


def _switch_sidebar(selected: str):
    return [gr.update(visible=(selected == name)) for name in PIPELINE_MENU]


def _switch_content(selected: str):
    return [gr.update(visible=(selected == name)) for name in PIPELINE_MENU]


def _switch_menu_buttons(selected: str):
    return [gr.update(variant="primary" if selected == name else "secondary") for name in PIPELINE_MENU]


def _refresh_navigation_payload(_selected: str):
    return [
        _default_summary("clean_dataset"),
        _default_summary("prepare_dataset"),
        _default_summary("make_metadata"),
        _default_summary("export_manifest"),
        _default_summary("train_lora_preprocess"),
        _default_summary("train_lora_train"),
        _default_summary("generate_samples"),
        _default_summary("evaluate"),
        _diagnostics_text(),
    ]


def _menu_payload(selected: str):
    return (
        _switch_menu_buttons(selected)
        + _switch_sidebar(selected)
        + _switch_content(selected)
        + _refresh_navigation_payload(selected)
    )


def _menu_handler(selected: str):
    def _fn(*_args: Any):
        return _menu_payload(selected)

    return _fn


def _stage_defaults(names: list[str], values: list[Any]) -> dict[str, Any]:
    return dict(zip(names, values, strict=False))


def _extract_line_value(text: str, prefix: str) -> str | None:
    for line in text.splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return None


def _status_class(status: str) -> str:
    base = status.split(" ", 1)[0].strip().lower() or "idle"
    if base in {"running", "stopping", "success", "failed", "stopped", "idle"}:
        return base
    if base in {"warning", "ready"}:
        return "idle"
    return "idle"


def _status_row_html(title: str, status: str, preview: str, summary: str) -> str:
    status = status.split("(", 1)[0].strip()
    run_id = _extract_line_value(summary, "Run ID") or _extract_line_value(preview, "Run ID") or "--"
    output_mode = _extract_line_value(summary, "Output mode") or _extract_line_value(preview, "Output mode") or "--"
    elapsed = _extract_line_value(summary, "Elapsed")
    elapsed_text = f'<span class="status-meta">Elapsed: {elapsed}</span>' if elapsed and elapsed != "--" else ""
    status_key = _status_class(status)
    return (
        '<div class="stage-title-row">'
        f"<h2>{title}</h2>"
        f'<span class="status-pill status-{status_key}">Status: {status}</span>'
        f'<span class="status-meta">Output mode: {output_mode}</span>'
        f'<span class="status-meta">Run ID: {run_id}</span>'
        f"{elapsed_text}"
        "</div>"
    )


def _static_status_row_html(title: str, status: str = "idle") -> str:
    return (
        '<div class="stage-title-row">'
        f"<h2>{title}</h2>"
        f'<span class="status-pill status-{_status_class(status)}">Status: {status}</span>'
        "</div>"
    )


def _preview_fn(stage_key: str, names: list[str]):
    config = STAGE_CONFIGS[stage_key]

    def _fn(*args):
        return RUNNER.build_preview(config, _stage_defaults(names, list(args)))

    return _fn


def _check_fn(stage_key: str, names: list[str]):
    config = STAGE_CONFIGS[stage_key]

    def _fn(*args):
        result = RUNNER.check(config, _stage_defaults(names, list(args)))
        return result["preview"], _status_row_html(config.title, result["status"], result["preview"], result["summary"]), (result["logs"] or "No active run."), result["summary"]

    return _fn


def _stop_fn(stage_key: str, names: list[str]):
    config = STAGE_CONFIGS[stage_key]

    def _fn(*args):
        result = RUNNER.stop(config, _stage_defaults(names, list(args)))
        return result["preview"], _status_row_html(config.title, result["status"], result["preview"], result["summary"]), (result["logs"] or "No active run."), result["summary"]

    return _fn


def _run_fn(stage_key: str, names: list[str]):
    config = STAGE_CONFIGS[stage_key]

    def _fn(*args):
        values = _stage_defaults(names, list(args))
        for item in RUNNER.run(config, values):
            yield item["preview"], _status_row_html(config.title, item["status"], item["preview"], item["summary"]), (item["logs"] or "No active run."), item["summary"]

    return _fn


def _generation_check_fn(names: list[str]):
    base = _check_fn("generate_samples", names)

    def _fn(*args):
        preview, status, logs, summary = base(*args)
        values = _stage_defaults(names, list(args))
        choices, selected, message = _generation_audio_choices(values.get("output_dir"))
        return preview, status, logs, summary, gr.update(choices=choices, value=selected), selected, message

    return _fn


def _generation_run_fn(names: list[str]):
    config = STAGE_CONFIGS["generate_samples"]

    def _fn(*args):
        values = _stage_defaults(names, list(args))
        for item in RUNNER.run(config, values):
            choices, selected, message = _generation_audio_choices(values.get("output_dir"))
            yield (
                item["preview"],
                _status_row_html(config.title, item["status"], item["preview"], item["summary"]),
                (item["logs"] or "No active run."),
                item["summary"],
                gr.update(choices=choices, value=selected),
                selected,
                message,
            )

    return _fn


def _generation_stop_fn(names: list[str]):
    base = _stop_fn("generate_samples", names)

    def _fn(*args):
        preview, status, logs, summary = base(*args)
        values = _stage_defaults(names, list(args))
        choices, selected, message = _generation_audio_choices(values.get("output_dir"))
        return preview, status, logs, summary, gr.update(choices=choices, value=selected), selected, message

    return _fn


def _select_generated_audio(path: str | None) -> str | None:
    return path if path else None


def _stage_panel(
    title: str,
    initial_status: str,
    default_preview: str,
    default_logs: str,
    default_summary: str,
) -> tuple[gr.Textbox, gr.HTML, gr.Textbox, gr.Textbox]:
    status = gr.HTML(value=_status_row_html(title, initial_status, default_preview, default_summary))
    gr.Markdown("### Command preview")
    preview = gr.Textbox(label=" ", show_label=False, value=default_preview, lines=12, interactive=False, elem_classes=["stage-panel"])
    gr.Markdown("### Live logs")
    logs = gr.Textbox(label=" ", show_label=False, value=default_logs or "No active run.", lines=18, interactive=False, elem_classes=["stage-panel"])
    gr.Markdown("### Result summary")
    summary = gr.Textbox(label=" ", show_label=False, value=default_summary, lines=18, interactive=False, elem_classes=["stage-panel"])
    return preview, status, logs, summary


def _bind_preview(config_key: str, widgets: StageWidgets) -> None:
    fn = _preview_fn(config_key, widgets.names)
    for component in widgets.inputs:
        if hasattr(component, "change"):
            component.change(fn, inputs=widgets.inputs, outputs=widgets.preview)


def _bind_stage_load(demo: gr.Blocks, config_key: str, widgets: StageWidgets) -> None:
    demo.load(
        _check_fn(config_key, widgets.names),
        inputs=widgets.inputs,
        outputs=[widgets.preview, widgets.status, widgets.logs, widgets.summary],
    )


def _bind_generation_load(
    demo: gr.Blocks,
    widgets: StageWidgets,
    dropdown: gr.Dropdown,
    audio: gr.Audio,
    message: gr.Markdown,
) -> None:
    demo.load(
        _generation_check_fn(widgets.names),
        inputs=widgets.inputs,
        outputs=[widgets.preview, widgets.status, widgets.logs, widgets.summary, dropdown, audio, message],
    )


def _initial_preview(config_key: str, defaults: dict[str, Any]) -> str:
    return RUNNER.build_preview(STAGE_CONFIGS[config_key], defaults)


def _initial_stage_state(config_key: str, defaults: dict[str, Any]) -> dict[str, str]:
    return RUNNER.stage_state(STAGE_CONFIGS[config_key], defaults)


def _default_summary(stage_key: str) -> str:
    return latest_summary_text(REPO_ROOT, STAGE_CONFIGS[stage_key].stage)


def _diagnostics_text() -> str:
    ffmpeg = ffmpeg_status()
    lines = [
        "Project diagnostics",
        diagnostics_snapshot(REPO_ROOT, sys.executable, STAGE_CONFIGS.values()),
        "",
        "Backend diagnostics",
        "\n".join(f"- {item}" for item in _backend_warnings()) if _backend_warnings() else "- backend paths look ready",
        "",
        "Checkpoint check",
    ]
    for prompt_mode in ("generic", "features"):
        checkpoint = _latest_valid_checkpoint(prompt_mode)
        lines.append(f"- {prompt_mode}: {checkpoint or 'missing'}")
    lines.extend(
        [
            "",
            "FFmpeg/ffprobe check",
            f"- ffmpeg: {ffmpeg['ffmpeg'] or 'missing'}",
            f"- ffprobe: {ffmpeg['ffprobe'] or 'missing'}",
        ]
    )
    if not ffmpeg["ready"]:
        lines.append(f"- install: {_ffmpeg_install_note()}")
    lines.extend(
        [
            "",
            "Pipeline health check",
            latest_summary_text(REPO_ROOT, "validate_training"),
        ]
    )
    return "\n".join(lines)


def _hard_reset_protected_text() -> str:
    return "\n".join(f"- {path}" for path in hard_reset_protected_paths(REPO_ROOT))


def _hard_reset_check() -> tuple[str, str]:
    plan = execute_hard_reset(REPO_ROOT, confirm=False)
    status = f"dry-run | targets={len(plan.targets)}"
    return _static_status_row_html("Hard reset", status), format_hard_reset_plan(plan, confirm=False)


def _hard_reset_run(confirmed: bool) -> tuple[str, str]:
    if not confirmed:
        return _static_status_row_html("Hard reset", "blocked"), "Confirmation checkbox is required before HARD RESET PROJECT can run."
    plan = execute_hard_reset(REPO_ROOT, confirm=True)
    status = f"deleted={len(plan.deleted)} skipped={len(plan.skipped)} protected={len(plan.protected)} errors={len(plan.errors)}"
    return _static_status_row_html("Hard reset", "failed" if plan.errors else "success"), format_hard_reset_plan(plan, confirm=True)


def _log_stage_choices() -> list[str]:
    return [stage for stage in STAGE_CONFIGS]


def _refresh_log_runs(stage: str) -> gr.Dropdown:
    choices = list_run_ids(REPO_ROOT, stage)
    return gr.update(choices=choices, value=choices[0] if choices else None)


def _select_log_stage(stage: str) -> tuple[gr.Dropdown, str, str]:
    choices = list_run_ids(REPO_ROOT, stage)
    selected = choices[0] if choices else None
    log_text, summary_text = _show_stage_run(stage, selected)
    return gr.update(choices=choices, value=selected), log_text, summary_text


def _show_stage_run(stage: str, run_id: str | None) -> tuple[str, str]:
    return read_stage_run_report(REPO_ROOT, stage, run_id)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Project Pipeline Control Panel") as demo:
        gr.HTML(AUTOSCROLL_HTML)
        gr.Markdown("# Project Pipeline Control Panel")
        with gr.Row(elem_classes=["app-shell"]):
            with gr.Column(scale=1, min_width=240, elem_classes=["sidebar"]):
                gr.Markdown("### Pipeline menu")
                menu_buttons: list[gr.Button] = []
                for index, item in enumerate(PIPELINE_MENU):
                    menu_buttons.append(gr.Button(item, variant="primary" if index == 0 else "secondary"))
                with gr.Column(visible=True, elem_id="sidebar-clean") as clean_sidebar:
                    clean_defaults = {
                        "input_dir": "data/dirty_data",
                        "output_dir": "data/raw_clean",
                        "copy_mode": "copy",
                        "verify_mode": "ffprobe",
                        "output_mode": OUTPUT_MODES[0],
                        "max_files": None,
                        "min_size_kb": 100,
                        "hash_algo": "sha1",
                    }
                    clean_names, clean_inputs = _render_parameters("clean_dataset", clean_defaults)

                with gr.Column(visible=True, elem_id="sidebar-prepare") as prepare_sidebar:
                    prepare_defaults = {
                        "input_dir": "data/raw_clean",
                        "output_dir": "data/processed",
                        "metadata_path": "data/processed/metadata.csv",
                        "output_mode": OUTPUT_MODES[0],
                        "chunk_seconds": 30,
                        "overlap_seconds": 5,
                        "sample_rate": 44100,
                        "min_duration_seconds": 30,
                        "target_lufs": -14,
                        "audio_format": "flac",
                    }
                    prepare_names, prepare_inputs = _render_parameters("prepare_dataset", prepare_defaults)

                with gr.Column(visible=True, elem_id="sidebar-metadata") as metadata_sidebar:
                    metadata_defaults = {
                        "input_dir": "data/processed",
                        "metadata_out": "data/metadata.csv",
                        "features_out": "data/metadata_features.csv",
                        "output_mode": OUTPUT_MODES[0],
                        "clusters": 4,
                        "seed": 42,
                        "generic_prompt": "techno music, electronic music",
                        "device": "auto",
                        "batch_size": 1,
                        "sample_rate": 16000,
                        "max_files": None,
                    }
                    metadata_names, metadata_inputs = _render_parameters("make_metadata", metadata_defaults)

                with gr.Column(visible=True, elem_id="sidebar-manifests") as manifest_sidebar:
                    manifest_defaults = {
                        "metadata_csv": "data/metadata.csv",
                        "prompt_mode": "features",
                        "jsonl_out": "outputs/latest/manifests/features/manifest.jsonl",
                        "dataset_json_out": "outputs/latest/manifests/features/dataset.json",
                        "output_mode": OUTPUT_MODES[0],
                    }
                    manifest_names, manifest_inputs = _render_parameters("export_manifest", manifest_defaults)

                with gr.Column(visible=True, elem_id="sidebar-training") as training_sidebar:
                    train_defaults = {"config": str(DEFAULT_CONFIG), "prompt_mode": "features"}
                    gr.Markdown("### Parameters")
                    with gr.Group():
                        gr.Markdown("#### Input / Output")
                        train_config = gr.Textbox(label="config", value=train_defaults["config"])
                        preprocess_mode = gr.Radio(PREPROCESS_OUTPUT_MODES, value=PREPROCESS_OUTPUT_MODES[0], label="preprocess_output_mode")
                        train_mode = gr.Radio(TRAIN_OUTPUT_MODES, value=TRAIN_OUTPUT_MODES[0], label="train_output_mode")
                    with gr.Group():
                        gr.Markdown("#### Processing")
                        train_prompt_mode = gr.Dropdown(label="prompt_mode", choices=["generic", "features"], value=train_defaults["prompt_mode"])

                with gr.Column(visible=True, elem_id="sidebar-generation") as generation_sidebar:
                    generation_defaults = {
                        "config": str(DEFAULT_CONFIG),
                        "checkpoint_dir": _checkpoint_value_for_prompt_mode("features"),
                        "prompt_mode": "features",
                        "prompt": "techno music, electronic track, high tempo, high energy, dark sound, dense rhythm",
                        "prompt_strategy": "diverse",
                        "num_samples": 5,
                        "duration": 30,
                        "output_dir": "outputs/latest/generated",
                        "device": "auto",
                        "seed": 42,
                        "metadata_path": "data/metadata.csv",
                        "model_variant": "turbo",
                        "output_mode": OUTPUT_MODES[0],
                    }
                    generation_names, generation_inputs = _render_parameters("generate_samples", generation_defaults)

                with gr.Column(visible=True, elem_id="sidebar-evaluation") as evaluation_sidebar:
                    evaluation_defaults = {
                        "dataset_metadata": "data/metadata.csv",
                        "dataset_features": "data/metadata_features.csv",
                        "base_dir": "outputs/latest/generated",
                        "generic_dir": "outputs/latest/generated",
                        "feature_dir": "outputs/latest/generated",
                        "sample_dirs": "generated=outputs/latest/generated",
                        "output_dir": "outputs/evaluation",
                        "sample_rate": 16000,
                        "tempo_min": 110,
                        "tempo_max": 150,
                        "plot_min_samples": 10,
                        "output_mode": OUTPUT_MODES[0],
                    }
                    evaluation_names, evaluation_inputs = _render_parameters("evaluate", evaluation_defaults)

                with gr.Column(visible=True, elem_id="sidebar-logs") as logs_sidebar:
                    gr.Markdown("### Parameters")
                    log_stage_choices = _log_stage_choices()
                    log_stage_dropdown = gr.Dropdown(label="Stage", choices=log_stage_choices, value=log_stage_choices[0] if log_stage_choices else None)
                    initial_log_runs = list_run_ids(REPO_ROOT, log_stage_choices[0]) if log_stage_choices else []
                    log_run_dropdown = gr.Dropdown(label="Run", choices=initial_log_runs, value=initial_log_runs[0] if initial_log_runs else None)

                with gr.Column(visible=True, elem_id="sidebar-hard-reset") as hard_reset_sidebar:
                    gr.Markdown("### Parameters")
                    reset_confirm = gr.Checkbox(
                        label="I understand that all generated artifacts except data/dirty_data will be deleted",
                        value=False,
                    )

                with gr.Column(visible=True, elem_id="sidebar-diagnostics") as diagnostics_sidebar:
                    gr.Markdown("### Pipeline health check")
                    validation_defaults = {
                        "config": str(DEFAULT_CONFIG),
                        "prompt_mode": "features",
                        "checkpoint_dir": _checkpoint_value_for_prompt_mode("features"),
                        "device": "auto",
                        "output_mode": OUTPUT_MODES[0],
                    }
                    validation_names, validation_inputs = _render_parameters("validate_training", validation_defaults)

            with gr.Column(scale=5):
                with gr.Column(visible=True, elem_id="section-clean") as clean_section:
                    clean_state = _initial_stage_state("clean_dataset", clean_defaults)
                    clean_preview, clean_status, clean_logs, clean_summary = _stage_panel(
                        "Dataset cleaning",
                        clean_state["status"],
                        clean_state["preview"],
                        clean_state["logs"],
                        clean_state["summary"],
                    )
                    clean_widgets = StageWidgets(
                        names=clean_names,
                        inputs=clean_inputs,
                        preview=clean_preview,
                        status=clean_status,
                        logs=clean_logs,
                        summary=clean_summary,
                    )
                    _bind_preview("clean_dataset", clean_widgets)
                    _bind_stage_load(demo, "clean_dataset", clean_widgets)
                    with gr.Row():
                        gr.Button("Refresh stage state").click(_check_fn("clean_dataset", clean_widgets.names), inputs=clean_widgets.inputs, outputs=[clean_preview, clean_status, clean_logs, clean_summary])
                        gr.Button("Run stage", variant="primary").click(_run_fn("clean_dataset", clean_widgets.names), inputs=clean_widgets.inputs, outputs=[clean_preview, clean_status, clean_logs, clean_summary])
                        gr.Button("Stop stage", variant="stop").click(_stop_fn("clean_dataset", clean_widgets.names), inputs=clean_widgets.inputs, outputs=[clean_preview, clean_status, clean_logs, clean_summary])

                with gr.Column(visible=True, elem_id="section-prepare") as prepare_section:
                    prepare_state = _initial_stage_state("prepare_dataset", prepare_defaults)
                    prepare_preview, prepare_status, prepare_logs, prepare_summary_box = _stage_panel(
                        "Dataset prepare",
                        prepare_state["status"],
                        prepare_state["preview"],
                        prepare_state["logs"],
                        prepare_state["summary"],
                    )
                    prepare_widgets = StageWidgets(
                        names=prepare_names,
                        inputs=prepare_inputs,
                        preview=prepare_preview,
                        status=prepare_status,
                        logs=prepare_logs,
                        summary=prepare_summary_box,
                    )
                    _bind_preview("prepare_dataset", prepare_widgets)
                    _bind_stage_load(demo, "prepare_dataset", prepare_widgets)
                    with gr.Row():
                        gr.Button("Refresh stage state").click(_check_fn("prepare_dataset", prepare_widgets.names), inputs=prepare_widgets.inputs, outputs=[prepare_preview, prepare_status, prepare_logs, prepare_summary_box])
                        gr.Button("Run stage", variant="primary").click(_run_fn("prepare_dataset", prepare_widgets.names), inputs=prepare_widgets.inputs, outputs=[prepare_preview, prepare_status, prepare_logs, prepare_summary_box])
                        gr.Button("Stop stage", variant="stop").click(_stop_fn("prepare_dataset", prepare_widgets.names), inputs=prepare_widgets.inputs, outputs=[prepare_preview, prepare_status, prepare_logs, prepare_summary_box])

                with gr.Column(visible=True, elem_id="section-metadata") as metadata_section:
                    metadata_state = _initial_stage_state("make_metadata", metadata_defaults)
                    metadata_preview, metadata_status, metadata_logs, metadata_summary_box = _stage_panel(
                        "Metadata",
                        metadata_state["status"],
                        metadata_state["preview"],
                        metadata_state["logs"],
                        metadata_state["summary"],
                    )
                    metadata_widgets = StageWidgets(
                        names=metadata_names,
                        inputs=metadata_inputs,
                        preview=metadata_preview,
                        status=metadata_status,
                        logs=metadata_logs,
                        summary=metadata_summary_box,
                    )
                    _bind_preview("make_metadata", metadata_widgets)
                    _bind_stage_load(demo, "make_metadata", metadata_widgets)
                    with gr.Row():
                        gr.Button("Refresh stage state").click(_check_fn("make_metadata", metadata_widgets.names), inputs=metadata_widgets.inputs, outputs=[metadata_preview, metadata_status, metadata_logs, metadata_summary_box])
                        gr.Button("Run stage", variant="primary").click(_run_fn("make_metadata", metadata_widgets.names), inputs=metadata_widgets.inputs, outputs=[metadata_preview, metadata_status, metadata_logs, metadata_summary_box])
                        gr.Button("Stop stage", variant="stop").click(_stop_fn("make_metadata", metadata_widgets.names), inputs=metadata_widgets.inputs, outputs=[metadata_preview, metadata_status, metadata_logs, metadata_summary_box])

                with gr.Column(visible=True, elem_id="section-manifests") as manifest_section:
                    manifest_state = _initial_stage_state("export_manifest", manifest_defaults)
                    manifest_preview, manifest_status, manifest_logs, manifest_summary_box = _stage_panel(
                        "Manifests",
                        manifest_state["status"],
                        manifest_state["preview"],
                        manifest_state["logs"],
                        manifest_state["summary"],
                    )
                    manifest_widgets = StageWidgets(
                        names=manifest_names,
                        inputs=manifest_inputs,
                        preview=manifest_preview,
                        status=manifest_status,
                        logs=manifest_logs,
                        summary=manifest_summary_box,
                    )
                    _bind_preview("export_manifest", manifest_widgets)
                    _bind_stage_load(demo, "export_manifest", manifest_widgets)
                    with gr.Row():
                        gr.Button("Refresh stage state").click(_check_fn("export_manifest", manifest_widgets.names), inputs=manifest_widgets.inputs, outputs=[manifest_preview, manifest_status, manifest_logs, manifest_summary_box])
                        gr.Button("Run stage", variant="primary").click(_run_fn("export_manifest", manifest_widgets.names), inputs=manifest_widgets.inputs, outputs=[manifest_preview, manifest_status, manifest_logs, manifest_summary_box])
                        gr.Button("Stop stage", variant="stop").click(_stop_fn("export_manifest", manifest_widgets.names), inputs=manifest_widgets.inputs, outputs=[manifest_preview, manifest_status, manifest_logs, manifest_summary_box])

                with gr.Column(visible=True, elem_id="section-training") as training_section:
                    preprocess_defaults = {**train_defaults, "output_mode": PREPROCESS_OUTPUT_MODES[0]}
                    preprocess_state = _initial_stage_state("train_lora_preprocess", preprocess_defaults)
                    preprocess_preview, preprocess_status, preprocess_logs, preprocess_summary = _stage_panel(
                        "Training preprocess",
                        preprocess_state["status"],
                        preprocess_state["preview"],
                        preprocess_state["logs"],
                        preprocess_state["summary"],
                    )
                    preprocess_widgets = StageWidgets(
                        names=list(preprocess_defaults.keys()),
                        inputs=[train_config, train_prompt_mode, preprocess_mode],
                        preview=preprocess_preview,
                        status=preprocess_status,
                        logs=preprocess_logs,
                        summary=preprocess_summary,
                    )
                    _bind_preview("train_lora_preprocess", preprocess_widgets)
                    _bind_stage_load(demo, "train_lora_preprocess", preprocess_widgets)
                    with gr.Row():
                        gr.Button("Refresh stage state").click(_check_fn("train_lora_preprocess", preprocess_widgets.names), inputs=preprocess_widgets.inputs, outputs=[preprocess_preview, preprocess_status, preprocess_logs, preprocess_summary])
                        gr.Button("Run stage", variant="primary").click(_run_fn("train_lora_preprocess", preprocess_widgets.names), inputs=preprocess_widgets.inputs, outputs=[preprocess_preview, preprocess_status, preprocess_logs, preprocess_summary])
                        gr.Button("Stop stage", variant="stop").click(_stop_fn("train_lora_preprocess", preprocess_widgets.names), inputs=preprocess_widgets.inputs, outputs=[preprocess_preview, preprocess_status, preprocess_logs, preprocess_summary])

                    train_stage_defaults = {**train_defaults, "output_mode": TRAIN_OUTPUT_MODES[0]}
                    train_state = _initial_stage_state("train_lora_train", train_stage_defaults)
                    train_preview, train_status, train_logs, train_summary_box = _stage_panel(
                        "Training",
                        train_state["status"],
                        train_state["preview"],
                        train_state["logs"],
                        train_state["summary"],
                    )
                    train_widgets = StageWidgets(
                        names=list(train_stage_defaults.keys()),
                        inputs=[train_config, train_prompt_mode, train_mode],
                        preview=train_preview,
                        status=train_status,
                        logs=train_logs,
                        summary=train_summary_box,
                    )
                    _bind_preview("train_lora_train", train_widgets)
                    _bind_stage_load(demo, "train_lora_train", train_widgets)
                    with gr.Row():
                        gr.Button("Refresh stage state").click(_check_fn("train_lora_train", train_widgets.names), inputs=train_widgets.inputs, outputs=[train_preview, train_status, train_logs, train_summary_box])
                        gr.Button("Run stage", variant="primary").click(_run_fn("train_lora_train", train_widgets.names), inputs=train_widgets.inputs, outputs=[train_preview, train_status, train_logs, train_summary_box])
                        gr.Button("Stop stage", variant="stop").click(_stop_fn("train_lora_train", train_widgets.names), inputs=train_widgets.inputs, outputs=[train_preview, train_status, train_logs, train_summary_box])

                with gr.Column(visible=True, elem_id="section-generation") as generation_section:
                    generation_state = _initial_stage_state("generate_samples", generation_defaults)
                    gen_preview, gen_status, gen_logs, gen_summary = _stage_panel(
                        "Generation",
                        generation_state["status"],
                        generation_state["preview"],
                        generation_state["logs"],
                        generation_state["summary"],
                    )
                    gen_widgets = StageWidgets(
                        names=generation_names,
                        inputs=generation_inputs,
                        preview=gen_preview,
                        status=gen_status,
                        logs=gen_logs,
                        summary=gen_summary,
                    )
                    _bind_preview("generate_samples", gen_widgets)
                    initial_audio_choices, initial_audio_value, initial_audio_message = _generation_audio_choices(generation_defaults["output_dir"])
                    gr.Markdown("### Generated audio")
                    generated_audio_message = gr.Markdown(initial_audio_message)
                    generated_audio_dropdown = gr.Dropdown(
                        label="generated files",
                        choices=initial_audio_choices,
                        value=initial_audio_value,
                        allow_custom_value=True,
                    )
                    generated_audio_player = gr.Audio(label="Preview", value=initial_audio_value, interactive=False)
                    generated_audio_dropdown.change(_select_generated_audio, inputs=generated_audio_dropdown, outputs=generated_audio_player)
                    _bind_generation_load(demo, gen_widgets, generated_audio_dropdown, generated_audio_player, generated_audio_message)
                    generation_outputs = [
                        gen_preview,
                        gen_status,
                        gen_logs,
                        gen_summary,
                        generated_audio_dropdown,
                        generated_audio_player,
                        generated_audio_message,
                    ]
                    generation_prompt_mode = generation_inputs[generation_names.index("prompt_mode")]
                    generation_checkpoint = generation_inputs[generation_names.index("checkpoint_dir")]
                    generation_prompt_mode.change(
                        lambda prompt_mode: gr.update(value=_checkpoint_value_for_prompt_mode(str(prompt_mode))),
                        inputs=generation_prompt_mode,
                        outputs=generation_checkpoint,
                    )
                    with gr.Row():
                        gr.Button("Refresh checkpoint").click(
                            lambda prompt_mode: gr.update(value=_checkpoint_value_for_prompt_mode(str(prompt_mode))),
                            inputs=generation_prompt_mode,
                            outputs=generation_checkpoint,
                        )
                        gr.Button("Refresh stage state").click(_generation_check_fn(gen_widgets.names), inputs=gen_widgets.inputs, outputs=generation_outputs)
                        gr.Button("Run stage", variant="primary").click(_generation_run_fn(gen_widgets.names), inputs=gen_widgets.inputs, outputs=generation_outputs)
                        gr.Button("Stop stage", variant="stop").click(_generation_stop_fn(gen_widgets.names), inputs=gen_widgets.inputs, outputs=generation_outputs)

                with gr.Column(visible=True, elem_id="section-evaluation") as evaluation_section:
                    evaluation_state = _initial_stage_state("evaluate", evaluation_defaults)
                    eval_preview, eval_status, eval_logs, eval_summary = _stage_panel(
                        "Evaluation",
                        evaluation_state["status"],
                        evaluation_state["preview"],
                        evaluation_state["logs"],
                        evaluation_state["summary"],
                    )
                    eval_widgets = StageWidgets(
                        names=evaluation_names,
                        inputs=evaluation_inputs,
                        preview=eval_preview,
                        status=eval_status,
                        logs=eval_logs,
                        summary=eval_summary,
                    )
                    _bind_preview("evaluate", eval_widgets)
                    _bind_stage_load(demo, "evaluate", eval_widgets)
                    with gr.Row():
                        gr.Button("Refresh stage state").click(_check_fn("evaluate", eval_widgets.names), inputs=eval_widgets.inputs, outputs=[eval_preview, eval_status, eval_logs, eval_summary])
                        gr.Button("Run stage", variant="primary").click(_run_fn("evaluate", eval_widgets.names), inputs=eval_widgets.inputs, outputs=[eval_preview, eval_status, eval_logs, eval_summary])
                        gr.Button("Stop stage", variant="stop").click(_stop_fn("evaluate", eval_widgets.names), inputs=eval_widgets.inputs, outputs=[eval_preview, eval_status, eval_logs, eval_summary])

                with gr.Column(visible=True, elem_id="section-logs") as logs_section:
                    gr.HTML(_static_status_row_html("Logs", "idle"))
                    initial_log_text, initial_summary_text = _show_stage_run(
                        log_stage_choices[0] if log_stage_choices else "clean_dataset",
                        initial_log_runs[0] if initial_log_runs else None,
                    )
                    gr.Markdown("### Live log")
                    log_view = gr.Textbox(label=" ", show_label=False, value=initial_log_text, lines=22, interactive=False, elem_classes=["stage-panel"])
                    gr.Markdown("### Result summary")
                    log_summary_view = gr.Textbox(label=" ", show_label=False, value=initial_summary_text, lines=18, interactive=False, elem_classes=["stage-panel"])
                    log_stage_dropdown.change(_select_log_stage, inputs=log_stage_dropdown, outputs=[log_run_dropdown, log_view, log_summary_view])
                    log_run_dropdown.change(_show_stage_run, inputs=[log_stage_dropdown, log_run_dropdown], outputs=[log_view, log_summary_view])
                    gr.Button("Refresh runs").click(_select_log_stage, inputs=log_stage_dropdown, outputs=[log_run_dropdown, log_view, log_summary_view])

                with gr.Column(visible=True, elem_id="section-hard-reset") as hard_reset_section:
                    reset_status = gr.HTML(_static_status_row_html("Hard reset", "idle"))
                    gr.Markdown("This will delete all generated artifacts and keep only source assets, especially `data/dirty_data`.")
                    gr.Markdown("### Protected paths")
                    protected_paths = gr.Textbox(label=" ", show_label=False, value=_hard_reset_protected_text(), lines=14, interactive=False, elem_classes=["stage-panel"])
                    reset_output = gr.Textbox(label=" ", show_label=False, value="No hard reset has been run.", lines=22, interactive=False, elem_classes=["stage-panel"])
                    with gr.Row():
                        gr.Button("Check hard reset plan").click(_hard_reset_check, outputs=[reset_status, reset_output])
                        gr.Button("HARD RESET PROJECT", variant="stop").click(_hard_reset_run, inputs=reset_confirm, outputs=[reset_status, reset_output])

                with gr.Column(visible=True, elem_id="section-diagnostics") as diagnostics_section:
                    validation_state = _initial_stage_state("validate_training", validation_defaults)
                    val_preview, val_status, val_logs, val_summary_box = _stage_panel(
                        "Pipeline health check",
                        validation_state["status"],
                        validation_state["preview"],
                        validation_state["logs"],
                        validation_state["summary"],
                    )
                    val_widgets = StageWidgets(
                        names=validation_names,
                        inputs=validation_inputs,
                        preview=val_preview,
                        status=val_status,
                        logs=val_logs,
                        summary=val_summary_box,
                    )
                    _bind_preview("validate_training", val_widgets)
                    _bind_stage_load(demo, "validate_training", val_widgets)
                    validation_prompt_mode = validation_inputs[validation_names.index("prompt_mode")]
                    validation_checkpoint = validation_inputs[validation_names.index("checkpoint_dir")]
                    validation_prompt_mode.change(
                        lambda prompt_mode: gr.update(value=_checkpoint_value_for_prompt_mode(str(prompt_mode))),
                        inputs=validation_prompt_mode,
                        outputs=validation_checkpoint,
                    )
                    diagnostics_box = gr.Textbox(label=" ", show_label=False, value=_diagnostics_text(), lines=20, interactive=False, elem_classes=["stage-panel"])
                    with gr.Row():
                        gr.Button("Refresh checkpoint").click(
                            lambda prompt_mode: gr.update(value=_checkpoint_value_for_prompt_mode(str(prompt_mode))),
                            inputs=validation_prompt_mode,
                            outputs=validation_checkpoint,
                        )
                        gr.Button("Refresh stage state").click(_check_fn("validate_training", val_widgets.names), inputs=val_widgets.inputs, outputs=[val_preview, val_status, val_logs, val_summary_box])
                        gr.Button("Run stage", variant="primary").click(_run_fn("validate_training", val_widgets.names), inputs=val_widgets.inputs, outputs=[val_preview, val_status, val_logs, val_summary_box])
                        gr.Button("Stop stage", variant="stop").click(_stop_fn("validate_training", val_widgets.names), inputs=val_widgets.inputs, outputs=[val_preview, val_status, val_logs, val_summary_box])
                    gr.Markdown("### Diagnostics")
                    gr.Button("Refresh diagnostics").click(_diagnostics_text, outputs=diagnostics_box)

        menu_outputs = [
            *menu_buttons,
            clean_sidebar,
            prepare_sidebar,
            metadata_sidebar,
            manifest_sidebar,
            training_sidebar,
            generation_sidebar,
            evaluation_sidebar,
            logs_sidebar,
            hard_reset_sidebar,
            diagnostics_sidebar,
            clean_section,
            prepare_section,
            metadata_section,
            manifest_section,
            training_section,
            generation_section,
            evaluation_section,
            logs_section,
            hard_reset_section,
            diagnostics_section,
            clean_summary,
            prepare_summary_box,
            metadata_summary_box,
            manifest_summary_box,
            preprocess_summary,
            train_summary_box,
            gen_summary,
            eval_summary,
            diagnostics_box,
        ]
        for label, button in zip(PIPELINE_MENU, menu_buttons, strict=False):
            button.click(
                _menu_handler(label),
                outputs=menu_outputs,
                queue=False,
                js=f"() => {{ if (window.applyPipelineStage) window.applyPipelineStage({json.dumps(label)}); }}",
            )
        demo.load(lambda: _menu_payload(PIPELINE_MENU[0]), outputs=menu_outputs, queue=False)
    return demo


if __name__ == "__main__":
    build_app().queue().launch(css=CSS)

