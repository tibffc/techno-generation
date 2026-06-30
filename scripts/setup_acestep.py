# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


REPO_CLONE_COMMAND = "git clone https://github.com/ace-step/ACE-Step-1.5 external/ACE-Step-1.5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect ACE-Step 1.5 backend files and training entrypoints.")
    parser.add_argument("--repo_path", type=Path, default=Path("external/ACE-Step-1.5"))
    parser.add_argument("--weights_path", type=Path, default=Path("external/ACE-Step-1.5/checkpoints"))
    parser.add_argument("--output_json", type=Path, default=Path("configs/acestep_detected.json"))
    return parser.parse_args()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_project_scripts(pyproject_path: Path) -> dict[str, str]:
    if not pyproject_path.exists():
        return {}
    text = _read_text(pyproject_path)
    match = re.search(r"\[project\.scripts\](.*?)(?:\n\[|\Z)", text, re.DOTALL)
    if not match:
        return {}
    block = match.group(1)
    scripts: dict[str, str] = {}
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        scripts[key.strip()] = value.strip().strip('"').strip("'")
    return scripts


def _detect_repo_python(repo_path: Path) -> str:
    candidates = [
        repo_path / ".venv" / "Scripts" / "python.exe",
        repo_path / ".venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.as_posix()
    return "python"


def detect_backend(repo_path: Path, weights_path: Path) -> dict[str, object]:
    external_root = repo_path.parent
    external_root.mkdir(parents=True, exist_ok=True)
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    repo_exists = repo_path.exists()
    detection: dict[str, object] = {
        "backend_name": "ace-step-1.5",
        "repo_path": repo_path.as_posix(),
        "repo_exists": repo_exists,
        "clone_instruction": REPO_CLONE_COMMAND,
        "weights_path": weights_path.as_posix(),
        "weights_exist": weights_path.exists(),
        "weights_instruction": (
            "Run `cd external/ACE-Step-1.5` and `uv run acestep-download` to populate external/ACE-Step-1.5/checkpoints"
        ),
        "install_files": {},
        "docs": {},
        "entrypoints": {
            "training_candidates": [],
            "cli_scripts": {},
            "recommended_preprocess_command_template": None,
            "recommended_train_command_template": None,
            "recommended_inference_command_template": None,
            "recommended_python_executable": None,
            "uv_available_in_path": False,
        },
        "manual_steps": [],
    }

    if not repo_exists:
        detection["manual_steps"] = [
            "Clone ACE-Step 1.5 into external/ACE-Step-1.5.",
            REPO_CLONE_COMMAND,
            "Install ACE-Step dependencies inside external/ACE-Step-1.5, for example `uv sync` or `pip install -r requirements.txt`.",
            "Populate external/ACE-Step-1.5/checkpoints with `uv run acestep-download`.",
        ]
        return detection

    pyproject_path = repo_path / "pyproject.toml"
    requirements_path = repo_path / "requirements.txt"
    setup_path = repo_path / "setup.py"
    train_py_path = repo_path / "train.py"
    lora_tutorial_path = repo_path / "docs" / "en" / "LoRA_Training_Tutorial.md"
    sidestep_training_guide = repo_path / "docs" / "sidestep" / "Training Guide.md"
    sidestep_dataset_guide = repo_path / "docs" / "sidestep" / "Dataset Preparation.md"

    detection["install_files"] = {
        "pyproject_toml": pyproject_path.exists(),
        "requirements_txt": requirements_path.exists(),
        "setup_py": setup_path.exists(),
    }
    detection["docs"] = {
        "lora_training_tutorial": lora_tutorial_path.exists(),
        "sidestep_training_guide": sidestep_training_guide.exists(),
        "sidestep_dataset_preparation": sidestep_dataset_guide.exists(),
    }

    cli_scripts = _extract_project_scripts(pyproject_path)
    detection["entrypoints"]["cli_scripts"] = cli_scripts

    candidates: list[dict[str, str]] = []
    for path in sorted(repo_path.rglob("train*.py")):
        candidates.append({"path": path.relative_to(repo_path).as_posix(), "reason": "filename matches train*.py"})
    for path in sorted(repo_path.rglob("*lora*.py")):
        rel_path = path.relative_to(repo_path).as_posix()
        if not any(candidate["path"] == rel_path for candidate in candidates):
            candidates.append({"path": rel_path, "reason": "filename contains lora"})
    detection["entrypoints"]["training_candidates"] = candidates

    repo_python = _detect_repo_python(repo_path)
    detection["entrypoints"]["recommended_python_executable"] = repo_python
    detection["entrypoints"]["uv_available_in_path"] = shutil.which("uv") is not None

    if train_py_path.exists():
        detection["entrypoints"]["recommended_preprocess_command_template"] = (
            "\"{python_executable}\" \"{repo_path}/train.py\" fixed "
            "--checkpoint-dir \"{base_model_path}\" "
            "--model-variant {model_variant} "
            "--preprocess "
            "--audio-dir \"{processed_dir}\" "
            "--dataset-json \"{dataset_json_path}\" "
            "--tensor-output \"{tensor_output_dir}\" "
            "--device {device} "
            "--precision {mixed_precision} "
            "--max-duration {max_duration_seconds}"
        )
        detection["entrypoints"]["recommended_train_command_template"] = (
            "\"{python_executable}\" \"{repo_path}/train.py\" fixed "
            "--checkpoint-dir \"{base_model_path}\" "
            "--model-variant {model_variant} "
            "--dataset-dir \"{tensor_output_dir}\" "
            "--output-dir \"{output_dir}\" "
            "--batch-size {batch_size} "
            "--gradient-accumulation {gradient_accumulation_steps} "
            "--lr {learning_rate} "
            "--epochs {epochs} "
            "--precision {mixed_precision} "
            "--adapter-type lora "
            "--rank {lora_rank} "
            "--alpha {lora_alpha} "
            "--dropout {lora_dropout} "
            "--save-every {save_every_n_epochs} "
            "--seed {seed} "
            "--device {device} "
            "--num-workers {num_workers} "
            "--optimizer-type {optimizer_type} "
            "{offload_encoder_flag} "
            "--yes"
        )

    if "acestep" in cli_scripts:
        detection["entrypoints"]["recommended_inference_command_template"] = (
            "{python_executable} -m acestep.acestep_v15_pipeline"
        )

    manual_steps = []
    if not weights_path.exists():
        manual_steps.append(
            "Weights are missing. Run `cd external/ACE-Step-1.5` and `uv run acestep-download`."
        )
    if not detection["entrypoints"]["recommended_train_command_template"]:
        manual_steps.append(
            "Training entrypoint was not auto-detected. Open external/ACE-Step-1.5 docs and set "
            "backend.train_command_template manually in configs/train_lora.yaml."
        )
    detection["manual_steps"] = manual_steps
    return detection


def main() -> None:
    args = parse_args()
    detection = detect_backend(args.repo_path, args.weights_path)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(detection, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Detection report written to {args.output_json}")
    print(f"Repo found: {detection['repo_exists']}")
    print(f"Weights found: {detection['weights_exist']}")

    preprocess_template = detection["entrypoints"]["recommended_preprocess_command_template"]
    train_template = detection["entrypoints"]["recommended_train_command_template"]
    print(f"Detected preprocess entrypoint: {bool(preprocess_template)}")
    print(f"Detected train entrypoint: {bool(train_template)}")
    if not detection["repo_exists"]:
        print(f"Clone ACE-Step with:\n  {REPO_CLONE_COMMAND}")
    if detection["manual_steps"]:
        print("Manual steps:")
        for step in detection["manual_steps"]:
            print(f"- {step}")


if __name__ == "__main__":
    main()
