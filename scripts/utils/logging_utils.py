from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOGS_DIR = WORKSPACE_ROOT / "outputs" / "runs"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


class _ConsoleVisibilityFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return bool(getattr(record, "console_visible", True))


@dataclass(frozen=True)
class StageLogger:
    logger: logging.Logger
    log_file: Path
    run_id: str


def build_run_id() -> str:
    return os.environ.get("PIPELINE_RUN_ID", datetime.now().strftime("%Y-%m-%d__%H-%M-%S"))


def ensure_logs_dir(logs_dir: Path | None = None) -> Path:
    explicit_run_dir = os.environ.get("PIPELINE_RUN_DIR")
    if explicit_run_dir:
        resolved = Path(explicit_run_dir).resolve()
    else:
        resolved = (logs_dir or WORKSPACE_ROOT / "outputs" / "runs" / "script" / build_run_id()).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def configure_stage_logger(stage: str, logs_dir: Path | None = None) -> StageLogger:
    run_id = build_run_id()
    explicit_log_file = os.environ.get("PIPELINE_STAGE_LOG_FILE")
    if explicit_log_file:
        log_file = Path(explicit_log_file).resolve()
        log_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        resolved_logs_dir = ensure_logs_dir(logs_dir or (WORKSPACE_ROOT / "outputs" / "runs" / stage / run_id))
        log_file = resolved_logs_dir / "live.log"

    logger_name = f"project.{stage}.{run_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(_ConsoleVisibilityFilter())

    logger.addHandler(console_handler)
    if not explicit_log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return StageLogger(logger=logger, log_file=log_file, run_id=run_id)


def log_file_only(logger: logging.Logger, level: int, message: str, *args: object) -> None:
    logger.log(level, message, *args, extra={"console_visible": False})
