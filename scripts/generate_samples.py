# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


LOGGER = logging.getLogger("generate_samples")
PROMPT_COLUMN_MAP = {
    "generic": "prompt_generic",
    "features": "prompt_features",
}
STYLE_PRESETS = [
    "hypnotic techno",
    "industrial techno",
    "dub techno",
    "minimal techno",
    "melodic techno",
    "electro techno",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate real ACE-Step samples with or without a trained LoRA adapter.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_lora.yaml"))
    parser.add_argument("--checkpoint_dir", type=Path, default=None, help="Path to a LoRA adapter directory such as final/ or checkpoints/epoch_N.")
    parser.add_argument("--prompt_mode", choices=["generic", "features"], default="features")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/samples/post_training"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--metadata_path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_variant", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None, help="Optional single prompt override.")
    parser.add_argument("--prompt_strategy", choices=["metadata", "diverse"], default="diverse")
    parser.add_argument("--log_file", type=Path, default=Path("outputs/logs/generate_samples.log"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repo_path", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--weights_path", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--backend_python", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--prompts_json", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_exists(path: Path, message: str) -> None:
    if not path.exists():
        raise FileNotFoundError(message)


def ensure_non_empty_dir(path: Path, message: str) -> None:
    ensure_exists(path, message)
    if not any(path.iterdir()):
        raise FileNotFoundError(message)


def resolve_backend_python(config: dict[str, Any]) -> Path:
    detected_path = Path(config["backend"].get("detected_config_path", "configs/acestep_detected.json"))
    ensure_exists(
        detected_path,
        f"ACE-Step detection report not found: {detected_path}. Run `python scripts/setup_acestep.py` first.",
    )
    with detected_path.open("r", encoding="utf-8") as handle:
        detected = json.load(handle)
    backend_python = detected.get("entrypoints", {}).get("recommended_python_executable")
    if not backend_python:
        raise FileNotFoundError(
            "ACE-Step backend python was not detected. Run `python scripts/setup_acestep.py` and verify the backend environment."
        )
    # Do not resolve symlinks for virtualenv python. In uv/venv, .venv/bin/python
    # is often a symlink to the base interpreter, and resolving it loses the venv context.
    return Path(backend_python).absolute()


def choose_prompts(metadata_path: Path, prompt_mode: str, num_samples: int, seed: int) -> list[str]:
    df = pd.read_csv(metadata_path)
    prompt_column = PROMPT_COLUMN_MAP[prompt_mode]
    if prompt_column not in df.columns:
        raise KeyError(f"Prompt column `{prompt_column}` is missing from {metadata_path}.")

    prompt_series = df[prompt_column].dropna().astype(str).str.strip()
    prompt_series = prompt_series[prompt_series != ""]
    unique_prompts = list(dict.fromkeys(prompt_series.tolist()))
    if not unique_prompts:
        raise ValueError(f"No usable prompts were found in column `{prompt_column}` of {metadata_path}.")

    rng = random.Random(seed)
    rng.shuffle(unique_prompts)
    return unique_prompts[: max(1, num_samples)]


def diversify_prompts(base_prompts: list[str], prompt_mode: str, num_samples: int) -> list[str]:
    prompts: list[str] = []
    for index in range(max(1, num_samples)):
        style = STYLE_PRESETS[index % len(STYLE_PRESETS)]
        source_prompt = base_prompts[index % len(base_prompts)]
        if prompt_mode == "features":
            suffix = re.sub(r"^\s*techno music,\s*electronic track,\s*", "", source_prompt, flags=re.IGNORECASE).strip()
            prompt = f"{style}, electronic music, {suffix}" if suffix else f"{style}, electronic music"
        else:
            prompt = f"{style}, electronic music"
        prompts.append(prompt)
    return prompts


def validate_checkpoint_dir(checkpoint_dir: Path) -> None:
    ensure_non_empty_dir(checkpoint_dir, f"Checkpoint directory is missing or empty: {checkpoint_dir}")
    adapter_config = checkpoint_dir / "adapter_config.json"
    adapter_weights = checkpoint_dir / "adapter_model.safetensors"
    if not adapter_config.exists():
        raise FileNotFoundError(
            f"LoRA adapter_config.json was not found in {checkpoint_dir}. "
            "Use a final/ or checkpoints/epoch_N directory produced by Side-Step LoRA training."
        )
    if not adapter_weights.exists():
        raise FileNotFoundError(
            f"LoRA adapter_model.safetensors was not found in {checkpoint_dir}. "
            "Use a final/ or checkpoints/epoch_N directory produced by Side-Step LoRA training."
        )


def run_backend_worker(args: argparse.Namespace, prompts: list[str]) -> None:
    config = load_yaml(args.config)
    backend = config["backend"]
    model = config["model"]
    training = config["training"]

    repo_path = Path(args.repo_path or backend["repo_path"]).resolve()
    weights_path = Path(args.weights_path or backend["base_model_path"]).resolve()
    backend_python = Path(args.backend_python or resolve_backend_python(config)).absolute()
    output_dir = args.output_dir.resolve()
    metadata_path = Path(args.metadata_path or training["metadata_path"]).resolve()
    checkpoint_dir = args.checkpoint_dir.resolve() if args.checkpoint_dir else None
    model_variant = args.model_variant or backend.get("model_variant", "turbo")

    ensure_exists(repo_path, f"ACE-Step repository not found: {repo_path}")
    ensure_non_empty_dir(weights_path, f"ACE-Step checkpoints directory is missing or empty: {weights_path}")
    ensure_exists(backend_python, f"ACE-Step backend python not found: {backend_python}")
    ensure_exists(metadata_path, f"Metadata file not found: {metadata_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint_dir is not None:
        validate_checkpoint_dir(checkpoint_dir)

    worker_payload_path = output_dir / "generation_prompts.json"
    worker_payload_path.write_text(json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8")

    command = [
        str(backend_python),
        str(Path(__file__).resolve()),
        "--worker",
        "--repo_path",
        str(repo_path),
        "--weights_path",
        str(weights_path),
        "--output_dir",
        str(output_dir),
        "--duration",
        str(args.duration),
        "--device",
        args.device,
        "--prompt_mode",
        args.prompt_mode,
        "--num_samples",
        str(len(prompts)),
        "--seed",
        str(args.seed),
        "--metadata_path",
        str(metadata_path),
        "--prompts_json",
        str(worker_payload_path),
        "--model_variant",
        model_variant,
    ]
    if checkpoint_dir is not None:
        command.extend(["--checkpoint_dir", str(checkpoint_dir)])
    if args.prompt:
        command.extend(["--prompt", args.prompt])

    LOGGER.info("Backend repo: %s", repo_path)
    LOGGER.info("Backend python: %s", backend_python)
    LOGGER.info("Backend weights path: %s", weights_path)
    LOGGER.info("Metadata path: %s", metadata_path)
    LOGGER.info("Output dir: %s", output_dir)
    LOGGER.info("Prompt mode: %s", args.prompt_mode)
    LOGGER.info("Prompts selected: %s", prompts)
    LOGGER.info("Checkpoint dir: %s", checkpoint_dir)
    LOGGER.info("No artificial fade-out is applied in this script. If tails fade, that comes from the model/decoder output.")
    LOGGER.info("Resolved backend worker command: %s", " ".join(f'"{part}"' if " " in part else part for part in command))

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.Popen(
        command,
        cwd=str(Path.cwd()),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    assert process.stdout is not None
    for line in process.stdout:
        clean = line.rstrip()
        if not clean:
            continue
        if clean.startswith('{"success":'):
            LOGGER.info("Backend result: %s", clean)
            continue
        if "INFO | generate_samples | Generated audio:" in clean or "INFO | generate_samples | Generated " in clean:
            continue
        LOGGER.info("[backend] %s", clean)

    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def worker_main(args: argparse.Namespace) -> None:
    repo_path = args.repo_path.resolve()
    weights_path = args.weights_path.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.prompt:
        prompts = [args.prompt]
    elif args.prompts_json is not None and args.prompts_json.exists():
        prompts = json.loads(args.prompts_json.read_text(encoding="utf-8"))
    else:
        prompts = choose_prompts(
            metadata_path=args.metadata_path.resolve(),
            prompt_mode=args.prompt_mode,
            num_samples=args.num_samples,
            seed=args.seed,
        )
    if not prompts:
        raise ValueError("No prompts were selected for generation.")

    ensure_exists(repo_path, f"ACE-Step repository not found: {repo_path}")
    ensure_non_empty_dir(weights_path, f"ACE-Step checkpoints directory is missing or empty: {weights_path}")

    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    os.environ["MPLBACKEND"] = "Agg"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(weights_path)

    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music
    from acestep.llm_inference import LLMHandler

    handler = AceStepHandler()
    status, ok = handler.initialize_service(
        project_root=str(repo_path),
        config_path=f"acestep-v15-{args.model_variant}",
        device=args.device,
        use_flash_attention=True,
        offload_to_cpu=False,
        offload_dit_to_cpu=False,
    )
    if not ok:
        raise RuntimeError(f"ACE-Step initialize_service failed: {status}")

    if args.checkpoint_dir is not None:
        checkpoint_dir = args.checkpoint_dir.resolve()
        validate_checkpoint_dir(checkpoint_dir)
        load_message = handler.load_lora(str(checkpoint_dir))
        LOGGER.info("LoRA load message: %s", load_message)
        toggle_message = handler.set_use_lora(True)
        LOGGER.info("LoRA toggle message: %s", toggle_message)

    llm_handler = LLMHandler()
    generated_items: list[dict[str, Any]] = []
    model_sample_rate = 48000 if str(args.model_variant).lower() == "turbo" else 48000
    for index, prompt in enumerate(prompts):
        sample_seed = args.seed + index
        params = GenerationParams(
            task_type="text2music",
            caption=prompt,
            lyrics="[Instrumental]",
            instrumental=True,
            duration=args.duration,
            bpm=128,
            thinking=False,
            use_cot_metas=False,
            use_cot_caption=False,
            use_cot_lyrics=False,
            use_cot_language=False,
            inference_steps=8 if str(args.model_variant).lower() == "turbo" else 50,
            shift=3.0 if str(args.model_variant).lower() == "turbo" else 1.0,
            seed=sample_seed,
        )
        config = GenerationConfig(
            batch_size=1,
            allow_lm_batch=False,
            use_random_seed=False,
            seeds=[sample_seed],
            audio_format="wav",
        )
        result = generate_music(handler, llm_handler, params, config, save_dir=str(output_dir))
        if not result.success:
            raise RuntimeError(f"ACE-Step generate_music failed for prompt `{prompt}`: {result.error}")
        for audio in result.audios:
            audio_path = audio.get("path")
            if audio_path:
                generated_items.append(
                    {
                        "path": audio_path,
                        "seed": sample_seed,
                        "prompt": prompt,
                        "sample_rate": model_sample_rate,
                    }
                )

    LOGGER.info("Generated %s audio files.", len(generated_items))
    print(json.dumps({"success": True, "generated_items": generated_items}, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file)

    try:
        if args.worker:
            worker_main(args)
            return

        config = load_yaml(args.config)
        metadata_path = Path(args.metadata_path or config["training"]["metadata_path"]).resolve()
        prompts = [args.prompt] if args.prompt else choose_prompts(
            metadata_path=metadata_path,
            prompt_mode=args.prompt_mode,
            num_samples=args.num_samples,
            seed=args.seed,
        )
        if not args.prompt and args.prompt_strategy == "diverse":
            prompts = diversify_prompts(prompts, prompt_mode=args.prompt_mode, num_samples=args.num_samples)
        if args.checkpoint_dir is not None:
            validate_checkpoint_dir(args.checkpoint_dir.resolve())
        run_backend_worker(args, prompts)
    except (FileNotFoundError, KeyError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        LOGGER.error("%s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
