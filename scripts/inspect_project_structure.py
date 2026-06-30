# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OLD_RUN_ID_RE = re.compile(r"^\d{8}_\d{6}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report misplaced project logs and deprecated output structure.")
    parser.add_argument("--fix", action="store_true", help="Deprecated. Use scripts/reset_project.py --confirm instead.")
    parser.add_argument("--confirm", action="store_true", help="Deprecated. Use scripts/reset_project.py --confirm instead.")
    return parser.parse_args()


def _existing(paths: list[Path]) -> list[Path]:
    return [path.resolve() for path in paths if path.exists()]


def inspect() -> dict[str, list[Path]]:
    root_logs = sorted(PROJECT_ROOT.glob("*.log"))
    data_logs = sorted((PROJECT_ROOT / "data").rglob("*.log")) if (PROJECT_ROOT / "data").exists() else []
    app_logs = sorted((PROJECT_ROOT / "app").rglob("*.log")) if (PROJECT_ROOT / "app").exists() else []
    script_logs = sorted((PROJECT_ROOT / "scripts").rglob("*.log")) if (PROJECT_ROOT / "scripts").exists() else []
    old_timestamp_dirs = [
        path.resolve()
        for base in [PROJECT_ROOT / "outputs" / "logs", PROJECT_ROOT / "outputs" / "reports"]
        if base.exists()
        for path in base.rglob("*")
        if path.is_dir() and OLD_RUN_ID_RE.match(path.name)
    ]
    stray_stdout = [
        path.resolve()
        for path in PROJECT_ROOT.rglob("*.log")
        if "outputs\\runs" not in str(path.resolve()).lower()
        and any(token in path.name.lower() for token in ["stdout", "stderr", "gradio_ui", "acestep_"])
    ]
    return {
        "root_level_logs": [path.resolve() for path in root_logs],
        "root_logs_dir": _existing([PROJECT_ROOT / "logs"]),
        "root_cache": _existing([PROJECT_ROOT / ".cache"]),
        "old_outputs_logs": _existing([PROJECT_ROOT / "outputs" / "logs"]),
        "old_outputs_reports": _existing([PROJECT_ROOT / "outputs" / "reports"]),
        "logs_under_data": [path.resolve() for path in data_logs],
        "logs_under_app": [path.resolve() for path in app_logs],
        "logs_under_scripts": [path.resolve() for path in script_logs],
        "stdout_stderr_outside_runs": stray_stdout,
        "old_timestamp_dirs": old_timestamp_dirs,
    }


def main() -> None:
    args = parse_args()
    if args.fix or args.confirm:
        raise SystemExit("--fix is intentionally not implemented. Run: python scripts/reset_project.py --confirm")
    payload = inspect()
    for label, paths in payload.items():
        print(f"{label}: {len(paths)}")
        for path in paths:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
