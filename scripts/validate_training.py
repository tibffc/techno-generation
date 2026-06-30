# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

try:
    from scripts.backends import acestep_backend
    from scripts.utils.logging_utils import configure_stage_logger
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from backends import acestep_backend
    from utils.logging_utils import configure_stage_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate that ACE-Step can load the selected weights/checkpoints.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_lora.yaml"))
    parser.add_argument("--prompt_mode", choices=["generic", "features"], default="features")
    parser.add_argument("--checkpoint_dir", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_exists(path: Path, message: str) -> None:
    if not path.exists():
        raise FileNotFoundError(message)


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    candidates: list[Path] = []
    final_dir = output_dir / "final"
    latest_dir = output_dir / "latest"
    checkpoints_dir = output_dir / "checkpoints"
    for path in (final_dir, latest_dir):
        if (path / "adapter_config.json").exists() and (path / "adapter_model.safetensors").exists():
            candidates.append(path)
    if checkpoints_dir.exists():
        candidates.extend(
            path
            for path in checkpoints_dir.iterdir()
            if path.is_dir() and (path / "adapter_config.json").exists() and (path / "adapter_model.safetensors").exists()
        )
    if not candidates and final_dir.exists():
        candidates.append(final_dir)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def save_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    stage_logger = configure_stage_logger("validate")
    logger = stage_logger.logger

    try:
        config = load_yaml(args.config)
        backend_cfg = config["backend"]
        training_cfg = config["training"]
        paths = acestep_backend.get_backend_paths()

        checkpoint_root = (Path(training_cfg["output_root"]) / args.prompt_mode).resolve()
        checkpoint_dir = args.checkpoint_dir.resolve() if args.checkpoint_dir else find_latest_checkpoint(checkpoint_root)
        if checkpoint_dir is None:
            raise FileNotFoundError(
                f"No valid checkpoint was found under {checkpoint_root}. "
                "Run Training first or set checkpoint_dir explicitly."
            )
        ensure_exists(checkpoint_dir, f"Checkpoint directory not found: {checkpoint_dir}")

        device = backend_cfg.get("device", "cuda") if args.device == "auto" else args.device
        model_variant = backend_cfg.get("model_variant", "turbo")

        logger.info("[PROCESS] technical_check=backend_checkpoint_load")
        logger.info("[PROCESS] checkpoint_dir=%s", checkpoint_dir.resolve())
        logger.info("[PROCESS] device=%s model_variant=%s", device, model_variant)

        request = acestep_backend.ValidateRequest(
            checkpoint_dir=checkpoint_dir,
            device=str(device),
            model_variant=str(model_variant),
        )
        backend_result = acestep_backend.validate(request, logger, stage_logger.run_id)
        summary_json = os.environ.get("PIPELINE_SUMMARY_JSON")
        summary_path = Path(summary_json).resolve() if summary_json else None
        summary = {
            "status": "success",
            "config_path": str(args.config.resolve()),
            "log_file": str(stage_logger.log_file.resolve()),
            "technical_check": "backend_checkpoint_load",
            "checkpoint_root": str(checkpoint_root.resolve()),
            "checkpoint_dir": str(checkpoint_dir.resolve()),
            "backend_run": acestep_backend.result_to_dict(backend_result),
        }
        if summary_path is not None:
            save_summary(summary_path, summary)
        logger.info("[PROCESS] technical_check=ok")
    except Exception as error:  # noqa: BLE001
        logger.exception("Validation orchestration failed: %s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
