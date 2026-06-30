# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from app.gradio_helpers import execute_hard_reset, format_hard_reset_plan
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from app.gradio_helpers import execute_hard_reset, format_hard_reset_plan


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hard reset project generated artifacts while preserving source inputs.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting anything.")
    parser.add_argument("--confirm", action="store_true", help="Actually delete generated artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    confirm = bool(args.confirm)
    plan = execute_hard_reset(PROJECT_ROOT, confirm=confirm)
    print(format_hard_reset_plan(plan, confirm=confirm))


if __name__ == "__main__":
    main()
