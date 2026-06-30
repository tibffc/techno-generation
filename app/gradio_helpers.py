# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import json
import os
import queue
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable


PROTECTED_PATH_NAMES = {".git", ".venv", "app", "configs", "external", "models", "notebooks", "scripts"}
AUDIO_EXTENSIONS = {
    ".aac",
    ".aif",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".wma",
}


@dataclass(slots=True)
class StageArtifacts:
    run_id: str
    run_dir: Path
    log_file: Path
    run_json: Path
    summary_json: Path
    summary_md: Path


@dataclass(slots=True)
class StageSummary:
    stage: str
    title: str
    run_id: str
    status: str
    started_at: str
    finished_at: str | None
    elapsed_seconds: float | None
    output_mode: str
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=lambda: {"examples": {}})


@dataclass(slots=True)
class ResetPlan:
    project_root: Path
    targets: list[Path]
    deleted: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    protected: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    elapsed_seconds: float | None = None


@dataclass(slots=True)
class StageRunState:
    run_id: str
    stage: str
    title: str
    status: str
    started_at: str
    cwd: str
    python: str
    command: list[str]
    output_mode: str
    parameters: dict[str, Any]
    expected_outputs: list[str]
    cleanup_plan: list[str]
    deleted_outputs: list[str]
    log_file: str
    summary_file: str
    run_dir: str
    finished_at: str | None = None
    elapsed_seconds: float | None = None
    exit_code: int | None = None
    error_message: str | None = None
    process: Any | None = None
    pid: int | None = None
    stop_requested: bool = False


@dataclass(slots=True)
class StageConfig:
    stage: str
    title: str
    script_path: Path
    output_mode_choices: list[str]
    default_output_mode: str
    progress_hint: str = "auto"
    command_builder: Callable[[dict[str, Any], StageArtifacts], list[str]] | None = None
    expected_outputs_builder: Callable[[dict[str, Any]], list[Path]] | None = None
    cleanup_targets_builder: Callable[[dict[str, Any]], list[Path]] | None = None
    preflight_builder: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    summary_builder: Callable[[dict[str, Any]], StageSummary] | None = None
    env_builder: Callable[[dict[str, Any], StageArtifacts], dict[str, str]] | None = None


def build_run_id(stage: str) -> str:
    return datetime.now().strftime("%Y-%m-%d__%H-%M-%S")


def get_run_dir(root: Path, stage: str, run_id: str) -> Path:
    return (root / "outputs" / "runs" / stage / run_id).resolve()


def get_run_file(root: Path, stage: str, run_id: str, filename: str) -> Path:
    return get_run_dir(root, stage, run_id) / filename


def get_run_summary_md(root: Path, stage: str, run_id: str) -> Path:
    return get_run_file(root, stage, run_id, "summary.md")


def get_run_summary_json(root: Path, stage: str, run_id: str) -> Path:
    return get_run_file(root, stage, run_id, "summary.json")


def get_run_json(root: Path, stage: str, run_id: str) -> Path:
    return get_run_file(root, stage, run_id, "run.json")


def get_latest_run(root: Path, stage: str) -> str | None:
    stage_root = (root / "outputs" / "runs" / stage).resolve()
    if not stage_root.exists():
        return None
    candidates = [item for item in stage_root.iterdir() if item.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime).name


def create_run_dir(root: Path, stage: str, persist: bool = True) -> StageArtifacts:
    run_id = build_run_id(stage)
    run_dir = get_run_dir(root, stage, run_id)
    while persist and run_dir.exists():
        time.sleep(1.0)
        run_id = build_run_id(stage)
        run_dir = get_run_dir(root, stage, run_id)
    if persist:
        run_dir.mkdir(parents=True, exist_ok=True)
    return StageArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        log_file=get_run_file(root, stage, run_id, "live.log"),
        run_json=get_run_json(root, stage, run_id),
        summary_json=get_run_summary_json(root, stage, run_id),
        summary_md=get_run_summary_md(root, stage, run_id),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _display_command(command: list[str]) -> str:
    compact: list[str] = []
    skip_next = False
    for index, item in enumerate(command):
        text = str(item)
        if skip_next:
            skip_next = False
            continue
        if text == "-c" and index + 1 < len(command):
            compact.extend(["-c", "<inline_python_script>"])
            skip_next = True
            continue
        if index >= 4 and len(text) > 240:
            compact.append(f"<payload:{len(text)} chars>")
            continue
        compact.append(text)
    return subprocess.list2cmdline(compact)


def scan_path_stats(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": "no", "kind": "missing"}
    resolved = path.resolve()
    info: dict[str, Any] = {"path": str(resolved), "exists": "yes" if resolved.exists() else "no"}
    if not resolved.exists():
        info["kind"] = "missing"
        return info
    if resolved.is_dir():
        file_count = 0
        dir_count = 0
        for item in resolved.rglob("*"):
            if item.is_file():
                file_count += 1
            elif item.is_dir():
                dir_count += 1
        info["kind"] = "dir"
        info["files"] = file_count
        info["dirs"] = dir_count
        return info
    info["kind"] = "file"
    info["size_bytes"] = resolved.stat().st_size
    return info


def count_audio_files(path: Path | None, suffixes: set[str] | None = None) -> int:
    active = suffixes or AUDIO_EXTENSIONS
    if path is None or not path.exists() or not path.is_dir():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in active)


def list_audio_files(path: Path | None, suffixes: set[str] | None = None) -> list[str]:
    active = suffixes or AUDIO_EXTENSIONS
    if path is None or not path.exists() or not path.is_dir():
        return []
    items = [item.resolve() for item in path.rglob("*") if item.is_file() and item.suffix.lower() in active]
    return [str(item) for item in sorted(items, key=lambda value: value.stat().st_mtime, reverse=True)]


def count_csv_rows_and_columns(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists() or not path.is_file():
        return {"exists": "no", "rows": None, "columns": []}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return {"exists": "yes", "rows": 0, "columns": []}
    return {"exists": "yes", "rows": max(0, len(rows) - 1), "columns": rows[0]}


def read_json_file(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_full_log(path: Path | None) -> str:
    if path is None or not path.exists():
        return "No log available."
    return path.read_text(encoding="utf-8", errors="replace")


def _protected_paths(root: Path) -> set[Path]:
    protected = {root, root / "data" / "dirty_data", root / "dirty_data"}
    for name in PROTECTED_PATH_NAMES:
        protected.add(root / name)
    return protected


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def safe_cleanup_stage_outputs(
    project_root: Path,
    targets: Iterable[Path],
) -> tuple[list[str], list[str], list[str]]:
    root = project_root.resolve()
    protected = _protected_paths(root)
    deleted: list[str] = []
    skipped: list[str] = []
    forbidden: list[str] = []
    seen: set[Path] = set()

    for target in targets:
        resolved = (target if target.is_absolute() else root / target).resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not _is_under(resolved, root):
            forbidden.append(f"outside project root: {resolved}")
            continue
        if any(resolved == item or item in resolved.parents for item in protected if item != root) or resolved == root:
            forbidden.append(f"protected path: {resolved}")
            continue
        if not resolved.exists():
            skipped.append(f"not found: {resolved}")
            continue
        if resolved.is_dir():
            shutil.rmtree(resolved)
            deleted.append(str(resolved))
        else:
            resolved.unlink()
            deleted.append(str(resolved))
    return deleted, skipped, forbidden


def build_log_header(
    *,
    stage: str,
    run_id: str,
    started_at: str,
    cwd: Path,
    python_executable: str,
    command: list[str],
    output_mode: str,
    preflight: dict[str, Any],
    cleanup_plan: list[str],
) -> str:
    lines = [
        "[HEADER]",
        f"stage: {stage}",
        f"run_id: {run_id}",
        f"cwd: {cwd.resolve()}",
        f"python: {Path(python_executable).resolve()}",
        f"command: {_display_command(command)}",
        "",
        "[PREFLIGHT]",
    ]
    for label in ("inputs", "outputs"):
        values = preflight.get(label, {})
        for key, value in values.items():
            lines.append(f"{key}: {format_inline_value(value)}")
    warnings = preflight.get("warnings", [])
    if warnings:
        lines.extend(["", "[WARNING]"])
        lines.extend(warnings)
    return "\n".join(lines) + "\n"


def build_log_footer(
    *,
    status: str,
    exit_code: int | None,
    elapsed_seconds: float | None,
    summary_file: Path,
    error_message: str | None = None,
) -> str:
    lines = [
        "",
        "[FOOTER]",
        f"status: {status}",
        f"elapsed_seconds: {elapsed_seconds if elapsed_seconds is not None else '--'}",
        f"summary_path: {summary_file.resolve()}",
    ]
    if error_message:
        lines.extend(["", "[ERROR]", error_message])
    return "\n".join(lines) + "\n"


def format_inline_value(value: Any) -> str:
    if isinstance(value, dict):
        ordered = []
        for key in ("path", "exists", "files", "dirs", "rows", "columns", "size_bytes", "kind"):
            if key in value and value[key] not in (None, [], {}):
                ordered.append(f"{key}={value[key]}")
        extra = [f"{key}={item}" for key, item in value.items() if key not in {"path", "exists", "files", "dirs", "rows", "columns", "size_bytes", "kind"} and item not in (None, [], {})]
        return " ".join(ordered + extra) or str(value)
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "--"
    return str(value)


def write_run_json(path: Path, state: StageRunState) -> None:
    summary_json_path = (Path(state.run_dir) / "summary.json").resolve()
    payload = {
        "run_id": state.run_id,
        "stage": state.stage,
        "title": state.title,
        "status": state.status,
        "started_at": state.started_at,
        "finished_at": state.finished_at,
        "elapsed_seconds": state.elapsed_seconds,
        "command": state.command,
        "output_mode": state.output_mode,
        "parameters": _json_safe(state.parameters),
        "expected_outputs": state.expected_outputs,
        "cleanup_plan": state.cleanup_plan,
        "deleted_outputs": state.deleted_outputs,
        "exit_code": state.exit_code,
        "pid": state.pid,
        "log_file": state.log_file,
        "summary_md": state.summary_file,
        "summary_json": str(summary_json_path),
        "error_message": state.error_message,
        "run_dir": state.run_dir,
    }
    summary_payload = read_json_file(summary_json_path) or {}
    for key in ("final_loss", "best_loss", "checkpoint_metadata", "checkpoint_summary"):
        if key in summary_payload:
            payload[key] = summary_payload[key]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _summary_payload(summary: StageSummary) -> dict[str, Any]:
    return {
        "run_id": summary.run_id,
        "stage": summary.stage,
        "status": summary.status,
        "started_at": summary.started_at,
        "finished_at": summary.finished_at,
        "elapsed_seconds": summary.elapsed_seconds,
        "output_mode": summary.output_mode,
        "inputs": _json_safe(summary.inputs),
        "outputs": _json_safe(summary.outputs),
        "metrics": _json_safe(summary.metrics),
        "warnings": _json_safe(summary.warnings),
        "errors": _json_safe(summary.errors),
        "notes": _json_safe(summary.notes),
    }


def write_summary_json(path: Path, summary: StageSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _summary_payload(summary)
    existing = read_json_file(path)
    if existing:
        merged = dict(existing)
        merged.update(payload)
        payload = merged
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def format_result_summary(summary: StageSummary) -> str:
    def format_label(key: str) -> str:
        if key == "final_loss":
            return "Final loss"
        if key == "best_loss":
            return "Best loss"
        return key

    lines = [
        f"Stage: {summary.title}",
        f"Status: {summary.status}",
        f"Run ID: {summary.run_id}",
    ]
    if summary.elapsed_seconds is not None:
        lines.append(f"Elapsed: {summary.elapsed_seconds}s")
    for key, value in summary.metrics.items():
        if value in (None, "", [], {}):
            continue
        lines.append(f"{format_label(key)}: {format_inline_value(value)}")
    for key, value in summary.outputs.items():
        if value in (None, "", [], {}):
            continue
        lines.append(f"{format_label(key)}: {format_inline_value(value)}")
    for item in summary.warnings + summary.errors + summary.notes:
        if item:
            lines.append(f"Note: {item}")
    return "\n".join(lines)


def write_summary_md(path: Path, summary: StageSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_result_summary(summary), encoding="utf-8")


def latest_stage_report(root: Path, stage: str) -> StageArtifacts | None:
    run_id = get_latest_run(root, stage)
    if run_id is None:
        return None
    run_dir = get_run_dir(root, stage, run_id)
    return StageArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        log_file=get_run_file(root, stage, run_id, "live.log"),
        run_json=get_run_json(root, stage, run_id),
        summary_json=get_run_summary_json(root, stage, run_id),
        summary_md=get_run_summary_md(root, stage, run_id),
    )


def list_logs(root: Path) -> list[str]:
    runs_root = (root / "outputs" / "runs").resolve()
    if not runs_root.exists():
        return []
    candidates = []
    for path in runs_root.rglob("*.log"):
        if path.is_file():
            candidates.append(path)
    return [str(item.resolve()) for item in sorted(candidates, key=lambda value: value.stat().st_mtime, reverse=True)]


def list_run_ids(root: Path, stage: str) -> list[str]:
    stage_root = (root / "outputs" / "runs" / stage).resolve()
    if not stage_root.exists():
        return []
    candidates = [item for item in stage_root.iterdir() if item.is_dir()]
    return [str(item.name) for item in sorted(candidates, key=lambda value: value.stat().st_mtime, reverse=True)]


def read_stage_run_report(root: Path, stage: str, run_id: str | None) -> tuple[str, str]:
    if not run_id:
        return "No log available.", "No summary available."
    run_dir = get_run_dir(root, stage, run_id)
    log_text = read_full_log(run_dir / "live.log")
    summary_path = run_dir / "summary.md"
    if not summary_path.exists():
        return log_text, "No summary available for this run."
    return log_text, summary_path.read_text(encoding="utf-8", errors="replace")


def list_latest_runs_by_stage(root: Path) -> dict[str, str]:
    logs_root = (root / "outputs" / "runs").resolve()
    latest: dict[str, tuple[float, str]] = {}
    if not logs_root.exists():
        return {}
    for run_json in logs_root.rglob("run.json"):
        payload = read_json_file(run_json) or {}
        stage = str(payload.get("stage") or "")
        run_id = str(payload.get("run_id") or run_json.parent.name)
        if not stage:
            continue
        mtime = run_json.stat().st_mtime
        if stage not in latest or latest[stage][0] < mtime:
            latest[stage] = (mtime, run_id)
    return {stage: run_id for stage, (_mtime, run_id) in sorted(latest.items())}


def list_existing_reports(root: Path) -> list[str]:
    reports_root = (root / "outputs" / "runs").resolve()
    if not reports_root.exists():
        return []
    items = []
    for path in reports_root.rglob("summary.md"):
        items.append(path)
    return [str(item.resolve()) for item in sorted(items, key=lambda value: value.stat().st_mtime, reverse=True)[:20]]


class StageRunner:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self._lock = threading.RLock()
        self._active: dict[str, StageRunState] = {}

    def is_running(self, stage: str) -> bool:
        with self._lock:
            state = self._active.get(stage)
            return bool(state and state.status in {"running", "stopping"})

    def latest_run_artifacts(self, stage: str) -> StageArtifacts | None:
        run_id = get_latest_run(self.project_root, stage)
        if run_id is None:
            return None
        latest = get_run_dir(self.project_root, stage, run_id)
        return StageArtifacts(
            run_id=run_id,
            run_dir=latest,
            log_file=get_run_file(self.project_root, stage, run_id, "live.log"),
            run_json=get_run_json(self.project_root, stage, run_id),
            summary_json=get_run_summary_json(self.project_root, stage, run_id),
            summary_md=get_run_summary_md(self.project_root, stage, run_id),
        )

    @staticmethod
    def _pid_is_running(pid: int | None) -> bool:
        if pid is None:
            return False
        try:
            import psutil  # type: ignore

            process = psutil.Process(pid)
            if not process.is_running():
                return False
            status = process.status()
            return status != psutil.STATUS_ZOMBIE
        except Exception:
            return False

    def _latest_running_artifacts(self, stage: str) -> StageArtifacts | None:
        artifacts = self.latest_run_artifacts(stage)
        if artifacts is None or not artifacts.run_json.exists():
            return None
        payload = read_json_file(artifacts.run_json) or {}
        if str(payload.get("status") or "") not in {"running", "stopping"}:
            return None
        if not self._pid_is_running(payload.get("pid")):
            return None
        return artifacts

    def _latest_state_payload(self, stage: str) -> tuple[StageArtifacts | None, dict[str, Any] | None]:
        artifacts = self.latest_run_artifacts(stage)
        if artifacts is None or not artifacts.run_json.exists():
            return artifacts, None
        return artifacts, read_json_file(artifacts.run_json) or {}

    def stage_state(self, config: StageConfig, values: dict[str, Any]) -> dict[str, str]:
        preview = self.build_preview(config, values)
        with self._lock:
            active = self._active.get(config.stage)
        if active and active.status in {"running", "stopping"}:
            logs = read_full_log(Path(active.log_file))
            summary = "Stage is running. Summary will be available after completion."
            status = active.status
        elif recovered := self._latest_running_artifacts(config.stage):
            payload = read_json_file(recovered.run_json) or {}
            logs = read_full_log(recovered.log_file)
            summary = "Stage is running. Summary will be available after completion."
            status = str(payload.get("status") or "running")
        else:
            artifacts, payload = self._latest_state_payload(config.stage)
            logs = read_full_log(artifacts.log_file) if artifacts and artifacts.log_file.exists() else f"No run yet for this stage."
            summary = latest_summary_text(self.project_root, config.stage)
            status = str((payload or {}).get("status") or "idle")
            if status in {"running", "stopping"} and not self._pid_is_running((payload or {}).get("pid")):
                status = "idle" if summary == "No summary available." else "failed"
        preflight = self._preflight(config, values)
        if preflight.get("warnings") and status == "idle":
            status = "warning"
        return {"preview": preview, "status": status, "logs": logs, "summary": summary}

    def stop(self, config: StageConfig, values: dict[str, Any]) -> dict[str, str]:
        preview = self.build_preview(config, values)
        with self._lock:
            state = self._active.get(config.stage)
            process = state.process if state else None
            pid = state.pid if state else None
            log_file = Path(state.log_file) if state else None
            run_json = log_file.parent / "run.json" if log_file else None
            if state and state.status in {"running", "stopping"}:
                state.stop_requested = True
                state.status = "stopping"
                write_run_json(run_json, state)
            else:
                recovered = self._latest_running_artifacts(config.stage)
                if recovered is None:
                    return {
                        "preview": preview,
                        "status": "idle",
                        "logs": "No running process for this stage.",
                        "summary": latest_summary_text(self.project_root, config.stage),
                    }
                payload = read_json_file(recovered.run_json) or {}
                pid = payload.get("pid")
                log_file = recovered.log_file
                run_json = recovered.run_json
                recovered_state = StageRunState(
                    run_id=str(payload.get("run_id") or recovered.run_id),
                    stage=str(payload.get("stage") or config.stage),
                    title=str(payload.get("title") or config.title),
                    status="stopping",
                    started_at=str(payload.get("started_at") or ""),
                    cwd=str(payload.get("cwd") or self.project_root),
                    python=str(payload.get("python") or sys.executable),
                    command=[str(item) for item in payload.get("command", [])],
                    output_mode=str(payload.get("output_mode") or config.default_output_mode),
                    parameters=dict(payload.get("parameters") or {}),
                    expected_outputs=[str(item) for item in payload.get("expected_outputs", [])],
                    cleanup_plan=[str(item) for item in payload.get("cleanup_plan", [])],
                    deleted_outputs=[str(item) for item in payload.get("deleted_outputs", [])],
                    log_file=str(recovered.log_file.resolve()),
                    summary_file=str(payload.get("summary_md") or recovered.summary_md.resolve()),
                    run_dir=str(payload.get("run_dir") or recovered.run_dir.resolve()),
                    finished_at=payload.get("finished_at"),
                    elapsed_seconds=payload.get("elapsed_seconds"),
                    exit_code=payload.get("exit_code"),
                    error_message=payload.get("error_message"),
                    process=None,
                    pid=pid,
                    stop_requested=True,
                )
                write_run_json(run_json, recovered_state)
        assert log_file is not None
        self._append_text(log_file, "\n[STOP] user requested stop\n")
        exit_code, forced = self._terminate_process_tree(process=process, pid=pid)
        self._append_text(log_file, "[STOP] process terminated\n")
        if forced:
            self._append_text(log_file, "[STOP] fallback kill used\n")
        self._append_text(log_file, f"[STOP] exit_code: {exit_code if exit_code is not None else '--'}\n")
        return {
            "preview": preview,
            "status": "stopping",
            "logs": read_full_log(log_file),
            "summary": "Stage is stopping. Summary will be available after completion.",
        }

    @staticmethod
    def _terminate_process_tree(
        process: subprocess.Popen[Any] | None = None,
        pid: int | None = None,
        timeout: float = 5.0,
    ) -> tuple[int | None, bool]:
        target_pid = process.pid if process is not None else pid
        if target_pid is None:
            return None, False
        try:
            import psutil  # type: ignore

            parent = psutil.Process(target_pid)
            children = parent.children(recursive=True)
            for child in children:
                if child.is_running():
                    child.terminate()
            parent.terminate()
            _gone, alive = psutil.wait_procs([parent, *children], timeout=timeout)
            forced = False
            for proc in alive:
                forced = True
                if proc.is_running():
                    proc.kill()
            psutil.wait_procs(alive, timeout=timeout)
            if process is not None:
                try:
                    return process.wait(timeout=timeout), forced
                except Exception:
                    return process.poll(), forced
            try:
                return parent.wait(timeout=timeout), forced
            except Exception:
                return None, forced
        except Exception:
            forced = False
            try:
                if process is None:
                    return None, False
                process.terminate()
                return process.wait(timeout=timeout), False
            except Exception:
                try:
                    forced = True
                    process.kill()
                    return process.wait(timeout=timeout), True
                except Exception:
                    return process.poll() if process is not None else None, forced

    def build_preview(self, config: StageConfig, values: dict[str, Any], artifacts: StageArtifacts | None = None) -> str:
        stage_artifacts = artifacts or create_run_dir(self.project_root, config.stage, persist=False)
        command = self._build_command(config, values, stage_artifacts)
        expected_outputs = self._expected_outputs(config, values)
        cleanup_targets = self._cleanup_targets(config, values)
        output_mode = str(values.get("output_mode", config.default_output_mode))
        cleanup_plan = [str(item.resolve()) for item in cleanup_targets] if output_mode.startswith("Recreate") else []
        preflight = self._preflight(config, values)
        lines = [
            f"Stage: {config.title}",
            f"Run ID: {stage_artifacts.run_id}",
            f"Output mode: {output_mode}",
            f"CWD: {self.project_root}",
            f"Python: {Path(command[0]).resolve()}",
            "",
            "Command:",
            _display_command(command),
            "",
            "Expected outputs:",
        ]
        if expected_outputs:
            lines.extend(f"- {item.resolve()}" for item in expected_outputs)
        else:
            lines.append("- none")
        lines.extend(
            [
                "",
                "Cleanup plan:",
                f"- mode: {output_mode}",
            ]
        )
        if cleanup_plan:
            lines.append("- would delete:")
            lines.extend(f"  - {item}" for item in cleanup_plan)
        else:
            lines.append("- would delete:")
            lines.append("  - none")
        protected = [str((self.project_root / name).resolve()) for name in sorted(PROTECTED_PATH_NAMES)]
        protected.insert(0, str((self.project_root / "data" / "dirty_data").resolve()))
        lines.append("- protected:")
        lines.extend(f"  - {item}" for item in protected)
        lines.append("")
        lines.append("Reuse behavior:")
        lines.append(self._reuse_behavior(output_mode))
        warnings = preflight.get("warnings", [])
        if warnings:
            lines.extend(["", "Warnings:"])
            lines.extend(f"- {item}" for item in warnings)
        return "\n".join(lines)

    def run(self, config: StageConfig, values: dict[str, Any]) -> Iterable[dict[str, str]]:
        with self._lock:
            active = self._active.get(config.stage)
            if active and active.status in {"running", "stopping"}:
                log_text = read_full_log(Path(active.log_file))
                yield {
                    "preview": self.build_preview(config, values),
                    "status": active.status,
                    "logs": log_text,
                    "summary": "Stage is already running. Stop it before starting another run.",
                }
                return
            recovered = self._latest_running_artifacts(config.stage)
            if recovered is not None:
                payload = read_json_file(recovered.run_json) or {}
                yield {
                    "preview": self.build_preview(config, values),
                    "status": str(payload.get("status") or "running"),
                    "logs": read_full_log(recovered.log_file),
                    "summary": "Stage is already running. Stop it before starting another run.",
                }
                return
        artifacts = create_run_dir(self.project_root, config.stage)
        output_mode = str(values.get("output_mode", config.default_output_mode))
        command = self._build_command(config, values, artifacts)
        expected_outputs = self._expected_outputs(config, values)
        cleanup_targets = self._cleanup_targets(config, values)
        preflight = self._preflight(config, values)
        cleanup_plan = [str(item.resolve()) for item in cleanup_targets] if output_mode.startswith("Recreate") else []
        preview = self.build_preview(config, values, artifacts)
        started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        state = StageRunState(
            run_id=artifacts.run_id,
            stage=config.stage,
            title=config.title,
            status="running",
            started_at=started_at,
            cwd=str(self.project_root),
            python=str(Path(command[0]).resolve()),
            command=[str(item) for item in command],
            output_mode=output_mode,
            parameters=_json_safe(values),
            expected_outputs=[str(item.resolve()) for item in expected_outputs],
            cleanup_plan=cleanup_plan,
            deleted_outputs=[],
            log_file=str(artifacts.log_file.resolve()),
            summary_file=str(artifacts.summary_md.resolve()),
            run_dir=str(artifacts.run_dir.resolve()),
        )
        with self._lock:
            self._active[config.stage] = state
        write_run_json(artifacts.run_json, state)
        header = build_log_header(
            stage=config.stage,
            run_id=artifacts.run_id,
            started_at=started_at,
            cwd=self.project_root,
            python_executable=command[0],
            command=command,
            output_mode=output_mode,
            preflight=preflight,
            cleanup_plan=cleanup_plan,
        )
        artifacts.log_file.write_text(header, encoding="utf-8")
        blocked = preflight.get("warnings", [])
        if output_mode.startswith("Recreate"):
            deleted, skipped, forbidden = safe_cleanup_stage_outputs(self.project_root, cleanup_targets)
            state.deleted_outputs = deleted
            write_run_json(artifacts.run_json, state)
            if deleted:
                self._append_text(artifacts.log_file, "\n[PROCESS]\n")
                self._append_text(artifacts.log_file, f"cleanup_deleted={len(deleted)}\n")
            if skipped:
                self._append_text(artifacts.log_file, "\n[WARNING]\n")
                self._append_text(artifacts.log_file, "\n".join(skipped) + "\n")
            if forbidden:
                self._append_text(artifacts.log_file, "\n[ERROR]\n")
                self._append_text(artifacts.log_file, "\n".join(forbidden) + "\n")
            if forbidden:
                blocked = blocked + forbidden
        self._append_text(artifacts.log_file, "\n[PROCESS]\n")
        if blocked:
            state.status = "failed"
            state.finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            state.elapsed_seconds = 0.0
            state.exit_code = None
            state.error_message = "; ".join(blocked)
            footer = build_log_footer(
                status="failed",
                exit_code=None,
                elapsed_seconds=0.0,
                summary_file=artifacts.summary_md,
                error_message=state.error_message,
            )
            self._append_text(artifacts.log_file, "[ERROR]\n" + "\n".join(blocked) + "\n" + footer)
            summary = self._build_summary(config, values, artifacts, state, preflight)
            write_summary_json(artifacts.summary_json, summary)
            write_summary_md(artifacts.summary_md, summary)
            state.summary_file = str(artifacts.summary_md.resolve())
            write_run_json(artifacts.run_json, state)
            final_log = read_full_log(artifacts.log_file)
            yield {"preview": preview, "status": "failed", "logs": final_log, "summary": format_result_summary(summary)}
            with self._lock:
                self._active.pop(config.stage, None)
            return

        env = os.environ.copy()
        env.update(
            {
                "PIPELINE_RUN_ID": artifacts.run_id,
                "PIPELINE_RUN_DIR": str(artifacts.run_dir.resolve()),
                "PIPELINE_STAGE_LOG_FILE": str(artifacts.log_file.resolve()),
                "PIPELINE_SUMMARY_JSON": str(artifacts.summary_json.resolve()),
                "PIPELINE_STAGE_NAME": config.stage,
            }
        )
        if config.env_builder is not None:
            env.update(config.env_builder(values, artifacts))

        try:
            launch_command = [str(item) for item in command]
            if Path(launch_command[0]).name.lower().startswith("python") and "-u" not in launch_command[1:3]:
                launch_command.insert(1, "-u")
            process = subprocess.Popen(
                launch_command,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0,
            )
            state.process = process
            state.pid = process.pid
            write_run_json(artifacts.run_json, state)
        except OSError as error:
            state.status = "failed"
            state.finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            state.elapsed_seconds = 0.0
            state.error_message = str(error)
            write_run_json(artifacts.run_json, state)
            self._append_text(
                artifacts.log_file,
                f"Launch failed: {error}\n" + build_log_footer(status="failed", exit_code=None, elapsed_seconds=0.0, summary_file=artifacts.summary_md, error_message=str(error)),
            )
            summary = self._build_summary(config, values, artifacts, state, preflight)
            write_summary_json(artifacts.summary_json, summary)
            write_summary_md(artifacts.summary_md, summary)
            yield {"preview": preview, "status": "failed", "logs": read_full_log(artifacts.log_file), "summary": format_result_summary(summary)}
            with self._lock:
                self._active.pop(config.stage, None)
            return

        assert process.stdout is not None
        output_queue: queue.Queue[str | None] = queue.Queue()

        def _enqueue_output(pipe: Any) -> None:
            try:
                for line in iter(pipe.readline, ""):
                    output_queue.put(line)
            finally:
                output_queue.put(None)

        reader = threading.Thread(target=_enqueue_output, args=(process.stdout,), daemon=True)
        reader.start()
        start_perf = time.perf_counter()
        last_emit = 0.0
        reader_finished = False
        while True:
            drained = False
            while True:
                try:
                    item = output_queue.get_nowait()
                except queue.Empty:
                    break
                drained = True
                if item is None:
                    reader_finished = True
                    continue
                self._append_text(artifacts.log_file, item)
            code = process.poll()
            now = time.perf_counter()
            if now - last_emit >= 0.8:
                current_log = read_full_log(artifacts.log_file)
                yield {
                    "preview": preview,
                    "status": "stopping" if state.stop_requested else "running",
                    "logs": current_log,
                    "summary": latest_summary_text(self.project_root, config.stage),
                }
                last_emit = now
            if code is not None and reader_finished and output_queue.empty():
                elapsed = round(time.perf_counter() - start_perf, 3)
                state.status = "stopped" if state.stop_requested else "success" if code == 0 else "failed"
                state.finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                state.elapsed_seconds = elapsed
                state.exit_code = code
                if code != 0 and not state.stop_requested:
                    state.error_message = f"Process exited with code {code}"
                footer = build_log_footer(
                    status=state.status,
                    exit_code=code,
                    elapsed_seconds=elapsed,
                    summary_file=artifacts.summary_md,
                    error_message=state.error_message,
                )
                self._append_text(artifacts.log_file, footer)
                summary = self._build_summary(config, values, artifacts, state, preflight)
                write_summary_json(artifacts.summary_json, summary)
                write_summary_md(artifacts.summary_md, summary)
                state.summary_file = str(artifacts.summary_md.resolve())
                write_run_json(artifacts.run_json, state)
                with self._lock:
                    self._active.pop(config.stage, None)
                yield {
                    "preview": preview,
                    "status": state.status,
                    "logs": read_full_log(artifacts.log_file),
                    "summary": format_result_summary(summary),
                }
                return
            if not drained:
                time.sleep(0.1)

    def check(self, config: StageConfig, values: dict[str, Any]) -> dict[str, str]:
        current = self.stage_state(config, values)
        preflight = self._preflight(config, values)
        if current["status"] in {"running", "stopping"}:
            return {
                "preview": current["preview"],
                "status": current["status"],
                "logs": current["logs"],
                "summary": current["summary"],
            }
        preview = current["preview"]
        lines = [
            f"Stage: {config.title}",
            "Check result:",
            f"- status: {'warning' if preflight.get('warnings') else 'ready'}",
            "",
            "Inputs:",
        ]
        for key, value in preflight.get("inputs", {}).items():
            lines.append(f"- {key}: {format_inline_value(value)}")
        lines.append("")
        lines.append("Expected outputs:")
        for item in preflight.get("expected_outputs", []):
            lines.append(f"- {item}")
        notes = preflight.get("notes", [])
        if notes:
            lines.extend(["", "Notes:"])
            lines.extend(f"- {item}" for item in notes)
        warnings = preflight.get("warnings", [])
        if warnings:
            lines.extend(["", "Warnings:"])
            lines.extend(f"- {item}" for item in warnings)
        return {
            "preview": preview,
            "status": "warning" if warnings else "ready",
            "logs": current["logs"],
            "summary": "\n".join(lines),
        }

    def _build_command(self, config: StageConfig, values: dict[str, Any], artifacts: StageArtifacts) -> list[str]:
        if config.command_builder is None:
            raise ValueError(f"Stage {config.stage} has no command builder")
        return config.command_builder(values, artifacts)

    def _expected_outputs(self, config: StageConfig, values: dict[str, Any]) -> list[Path]:
        if config.expected_outputs_builder is None:
            return []
        return [item.resolve() for item in config.expected_outputs_builder(values)]

    def _cleanup_targets(self, config: StageConfig, values: dict[str, Any]) -> list[Path]:
        if config.cleanup_targets_builder is None:
            return []
        return [item.resolve() for item in config.cleanup_targets_builder(values)]

    def _preflight(self, config: StageConfig, values: dict[str, Any]) -> dict[str, Any]:
        base = {
            "inputs": {},
            "outputs": {},
            "expected_outputs": [str(item.resolve()) for item in self._expected_outputs(config, values)],
            "notes": [],
            "warnings": [],
        }
        if config.preflight_builder is None:
            return base
        custom = config.preflight_builder(values)
        for key, value in custom.items():
            base[key] = value
        return base

    def _build_summary(
        self,
        config: StageConfig,
        values: dict[str, Any],
        artifacts: StageArtifacts,
        state: StageRunState,
        preflight: dict[str, Any],
    ) -> StageSummary:
        context = {
            "project_root": self.project_root,
            "config": config,
            "values": values,
            "artifacts": artifacts,
            "state": state,
            "preflight": preflight,
        }
        if config.summary_builder is None:
            summary = default_summary_builder(context)
        else:
            summary = config.summary_builder(context)
        return summary

    @staticmethod
    def _append_text(path: Path, text: str) -> None:
        with path.open("a", encoding="utf-8", errors="replace") as handle:
            handle.write(text)
            if text and not text.endswith("\n"):
                handle.write("\n")

    @staticmethod
    def _reuse_behavior(output_mode: str) -> str:
        if output_mode == "Use existing outputs":
            return "Existing outputs are preserved and the script may reuse them when supported."
        if output_mode == "Recreate outputs":
            return "Only the current stage outputs are deleted before the run."
        if output_mode == "Use existing tensors":
            return "Existing tensors are preserved and reused."
        if output_mode == "Recreate tensors":
            return "Only outputs/tensors for this prompt mode are deleted before preprocessing."
        if output_mode == "Continue from existing checkpoint":
            return "Existing checkpoints are preserved; training continues from the current run directory."
        if output_mode == "Recreate training run":
            return "Only checkpoint outputs for this prompt mode are deleted; tensors remain untouched."
        return output_mode

def default_summary_builder(context: dict[str, Any]) -> StageSummary:
    config: StageConfig = context["config"]
    state: StageRunState = context["state"]
    preflight: dict[str, Any] = context["preflight"]
    return StageSummary(
        stage=config.stage,
        title=config.title,
        run_id=state.run_id,
        status=state.status,
        started_at=state.started_at,
        finished_at=state.finished_at,
        elapsed_seconds=state.elapsed_seconds,
        output_mode=state.output_mode,
        inputs=preflight.get("inputs", {}),
        outputs=preflight.get("outputs", {}),
        metrics={"exit_code": state.exit_code},
        artifacts={},
        warnings=preflight.get("warnings", []),
        errors=[state.error_message] if state.error_message else [],
        notes=preflight.get("notes", []),
    )


def latest_summary_text(root: Path, stage: str) -> str:
    artifacts = latest_stage_report(root, stage)
    if artifacts is None or not artifacts.summary_md.exists():
        return "No summary available."
    return artifacts.summary_md.read_text(encoding="utf-8", errors="replace")


def ffmpeg_status() -> dict[str, Any]:
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    return {
        "ffmpeg": ffmpeg_path,
        "ffprobe": ffprobe_path,
        "ready": bool(ffmpeg_path and ffprobe_path),
        "instructions": [
            "winget install Gyan.FFmpeg",
            "or download FFmpeg and add its bin directory to PATH",
        ],
    }


def diagnostics_snapshot(root: Path, python_executable: str, stage_configs: Iterable[StageConfig]) -> str:
    ffmpeg = ffmpeg_status()
    lines = [
        f"project root: {root.resolve()}",
        f"python executable: {Path(python_executable).resolve()}",
        f"runs root: {(root / 'outputs' / 'runs').resolve()}",
        f"cache root: {(root / 'outputs' / 'cache').resolve()}",
        f"environment info: os={os.name} pid={os.getpid()}",
        "",
        "ffmpeg / ffprobe:",
        f"- ffmpeg: {ffmpeg['ffmpeg'] or 'missing'}",
        f"- ffprobe: {ffmpeg['ffprobe'] or 'missing'}",
    ]
    if not ffmpeg["ready"]:
        lines.extend(
            [
                "- install: winget install Gyan.FFmpeg",
                "- install: download FFmpeg and add its bin directory to PATH",
            ]
        )
    lines.extend(
        [
            "",
        "available scripts:",
        ]
    )
    for config in stage_configs:
        lines.append(f"- {config.stage}: {config.script_path.resolve()} exists={'yes' if config.script_path.exists() else 'no'}")
    lines.extend(["", "existing reports:"])
    reports = list_existing_reports(root)
    lines.extend(f"- {item}" for item in reports) if reports else lines.append("- none")
    lines.extend(["", "latest runs by stage:"])
    latest_runs = list_latest_runs_by_stage(root)
    if latest_runs:
        for stage, run_id in latest_runs.items():
            lines.append(f"- {stage}: {get_run_dir(root, stage, run_id)}")
    else:
        lines.append("- none")
    for deprecated_root in [(root / "outputs" / "logs").resolve(), (root / "outputs" / "reports").resolve(), (root / ".cache").resolve()]:
        if deprecated_root.exists():
            lines.extend(["", f"deprecated path present: {deprecated_root}"])
    return "\n".join(lines)


def hard_reset_protected_paths(root: Path) -> list[Path]:
    protected = sorted(_protected_paths(root), key=lambda item: str(item))
    extra = [root / "requirements.txt", root / "README.md", root / ".gitignore"]
    seen: set[Path] = set()
    result: list[Path] = []
    for item in protected + [path.resolve() for path in extra if path.exists()]:
        resolved = item.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(resolved)
    return result


def hard_reset_targets(root: Path) -> list[Path]:
    candidates: list[Path] = []
    protected = {path for path in hard_reset_protected_paths(root) if path != root.resolve()}
    direct_targets = [
        root / "outputs",
        root / "logs",
        root / ".cache",
        root / "reset_project.log",
        root / "sidestep.log",
        root / "data" / "raw_clean",
        root / "data" / "processed",
    ]
    candidates.extend(direct_targets)
    data_root = root / "data"
    if data_root.exists():
        for path in data_root.iterdir():
            if path.is_dir():
                continue
            if path.suffix.lower() in {".csv", ".json"}:
                candidates.append(path)
    for base in [root / "app", root / "scripts", root / "notebooks"]:
        if base.exists():
            candidates.extend(path for path in base.rglob("__pycache__") if path.is_dir())
    if (root / "__pycache__").exists():
        candidates.append((root / "__pycache__").resolve())
    candidates.extend(path for path in root.glob("*.log"))
    candidates.extend(path for path in root.glob("*.tmp"))
    candidates.extend(path for path in root.glob("*.temp"))
    candidates.extend(path for path in root.glob("*.bak"))
    unique: list[Path] = []
    seen: set[Path] = set()
    for item in candidates:
        resolved = item.resolve()
        if resolved.name != "__pycache__" and any(
            resolved == protected_path or protected_path in resolved.parents for protected_path in protected
        ):
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def build_hard_reset_plan(root: Path) -> ResetPlan:
    return ResetPlan(project_root=root.resolve(), targets=hard_reset_targets(root.resolve()))


def execute_hard_reset(root: Path, confirm: bool, log_file: Path | None = None) -> ResetPlan:
    project_root = root.resolve()
    plan = build_hard_reset_plan(project_root)
    protected = hard_reset_protected_paths(project_root)
    start = time.perf_counter()
    log_lines: list[str] = []

    def _log(message: str) -> None:
        log_lines.append(message)

    _log(f"hard_reset_start root={project_root} confirm={confirm}")
    if not confirm:
        plan.skipped.append("dry-run only; nothing deleted")
        plan.elapsed_seconds = round(time.perf_counter() - start, 3)
        _log("hard_reset_dry_run")
        logger_path = (log_file.resolve() if log_file else create_run_dir(project_root, "reset_project").log_file)
        logger_path.parent.mkdir(parents=True, exist_ok=True)
        logger_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        return plan

    protected_set = {path for path in protected if path != project_root}
    for target in plan.targets:
        resolved = target.resolve()
        if resolved == project_root:
            plan.protected.append(str(resolved))
            _log(f"protected {resolved}")
            continue
        if not _is_under(resolved, project_root):
            plan.protected.append(f"outside project root: {resolved}")
            _log(f"protected outside {resolved}")
            continue
        if resolved.name != "__pycache__" and any(resolved == item or item in resolved.parents for item in protected_set):
            plan.protected.append(str(resolved))
            _log(f"protected {resolved}")
            continue
        if not resolved.exists():
            plan.skipped.append(str(resolved))
            _log(f"skipped missing {resolved}")
            continue
        try:
            if resolved.is_dir():
                shutil.rmtree(resolved)
            else:
                resolved.unlink()
            plan.deleted.append(str(resolved))
            _log(f"deleted {resolved}")
        except OSError as error:
            message = f"{resolved}: {error}"
            plan.errors.append(message)
            _log(f"error {message}")

    for path in [project_root / "outputs", project_root / "outputs" / "runs", project_root / "outputs" / "cache"]:
        path.mkdir(parents=True, exist_ok=True)
        _log(f"recreated {path.resolve()}")

    plan.elapsed_seconds = round(time.perf_counter() - start, 3)
    _log(f"hard_reset_end elapsed={plan.elapsed_seconds}")
    logger_path = (log_file.resolve() if log_file else create_run_dir(project_root, "reset_project").log_file)
    logger_path.parent.mkdir(parents=True, exist_ok=True)
    logger_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return plan


def format_hard_reset_plan(plan: ResetPlan, confirm: bool) -> str:
    lines = [
        f"Project root: {plan.project_root}",
        f"Mode: {'confirm' if confirm else 'dry-run'}",
        f"Targets: {len(plan.targets)}",
        "",
        "Would delete:" if not confirm else "Deleted:",
    ]
    if confirm:
        if plan.deleted:
            lines.extend(f"- {item}" for item in plan.deleted)
        else:
            lines.append("- none")
    else:
        lines.extend(f"- {item}" for item in plan.targets[:200])
        if len(plan.targets) > 200:
            lines.append(f"- ... {len(plan.targets) - 200} more")
    lines.extend(["", "Protected paths:"])
    protected = hard_reset_protected_paths(plan.project_root)
    lines.extend(f"- {item}" for item in protected)
    if plan.skipped:
        lines.extend(["", "Skipped:"])
        lines.extend(f"- {item}" for item in plan.skipped)
    if plan.protected:
        lines.extend(["", "Protected during execution:"])
        lines.extend(f"- {item}" for item in plan.protected)
    if plan.errors:
        lines.extend(["", "Errors:"])
        lines.extend(f"- {item}" for item in plan.errors)
    if plan.elapsed_seconds is not None:
        lines.extend(["", f"Elapsed: {plan.elapsed_seconds}s"])
    return "\n".join(lines)
