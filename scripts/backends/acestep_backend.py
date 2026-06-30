from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    from scripts.utils.logging_utils import WORKSPACE_ROOT
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.logging_utils import WORKSPACE_ROOT


ACESTEP_REPO_PATH = WORKSPACE_ROOT / "external" / "ACE-Step-1.5"
ACESTEP_PYTHON_PATH = ACESTEP_REPO_PATH / ".venv" / "Scripts" / "python.exe"
ACESTEP_CHECKPOINTS_PATH = ACESTEP_REPO_PATH / "checkpoints"
RESULT_PREFIX = "__PROJECT_RESULT__="
STEP_RE = re.compile(r"(?:\bStep\b|\bstep[=/:\s]+)(\d+)(?:\s*/\s*(\d+))?", re.IGNORECASE)
EPOCH_RE = re.compile(r"(?:\bEpoch\b|\bepoch[=/:\s]+)(\d+)(?:\s*/\s*(\d+))?", re.IGNORECASE)
LOSS_RE = re.compile(r"\bloss[=:\s]+([0-9]+(?:\.[0-9]+)?)\b", re.IGNORECASE)
PROCESSED_RE = re.compile(r"(?:Preprocessing complete:|Processed:)\s+(\d+)/(\d+)", re.IGNORECASE)
TQDM_RE = re.compile(r"(\d+)%\|.*?\|\s*(\d+)/(\d+)", re.IGNORECASE)

GENERATE_INLINE_SCRIPT = r"""
from __future__ import annotations
import json
import os
import re
import shutil
import sys
from pathlib import Path

payload = json.loads(sys.argv[1])
repo_path = Path(payload["repo_path"]).resolve()
weights_path = Path(payload["weights_path"]).resolve()
output_dir = Path(payload["output_dir"]).resolve()
checkpoint_dir = Path(payload["checkpoint_dir"]).resolve() if payload.get("checkpoint_dir") else None
prompts = payload["prompts"]
prompt_mode = str(payload.get("prompt_mode") or "generated").strip() or "generated"

def _safe_slug(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return cleaned or "prompt"

if str(repo_path) not in sys.path:
    sys.path.insert(0, str(repo_path))

os.environ["MPLBACKEND"] = "Agg"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(weights_path)

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music
from acestep.llm_inference import LLMHandler

handler = AceStepHandler()
status, ok = handler.initialize_service(
    project_root=str(repo_path),
    config_path=f"acestep-v15-{payload['model_variant']}",
    device=payload["device"],
    use_flash_attention=True,
    offload_to_cpu=False,
    offload_dit_to_cpu=False,
)
if not ok:
    raise RuntimeError(f"ACE-Step initialize_service failed: {status}")

if checkpoint_dir is not None:
    adapter_config = checkpoint_dir / "adapter_config.json"
    adapter_weights = checkpoint_dir / "adapter_model.safetensors"
    if not adapter_config.exists():
        raise FileNotFoundError(f"LoRA adapter_config.json was not found in {checkpoint_dir}")
    if not adapter_weights.exists():
        raise FileNotFoundError(f"LoRA adapter_model.safetensors was not found in {checkpoint_dir}")
    print(handler.load_lora(str(checkpoint_dir)), flush=True)
    print(handler.set_use_lora(True), flush=True)

llm_handler = LLMHandler()
generated_items = []
for index, prompt in enumerate(prompts):
    sample_seed = int(payload["seed"]) + index
    params = GenerationParams(
        task_type="text2music",
        caption=prompt,
        lyrics="[Instrumental]",
        instrumental=True,
        duration=float(payload["duration"]),
        bpm=128,
        thinking=False,
        use_cot_metas=False,
        use_cot_caption=False,
        use_cot_lyrics=False,
        use_cot_language=False,
        inference_steps=8 if str(payload["model_variant"]).lower() == "turbo" else 50,
        shift=3.0 if str(payload["model_variant"]).lower() == "turbo" else 1.0,
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
    prompt_slug = _safe_slug(prompt_mode)
    for audio_index, audio in enumerate(result.audios, start=1):
        audio_path = audio.get("path")
        if audio_path:
            source_path = Path(audio_path).resolve()
            extension = source_path.suffix or ".wav"
            target_name = f"gen_{index + 1:03d}__{prompt_slug}__seed_{sample_seed}{extension}"
            target_path = output_dir / target_name
            if source_path != target_path:
                counter = 1
                while target_path.exists():
                    counter += 1
                    target_name = f"gen_{index + 1:03d}__{prompt_slug}__seed_{sample_seed}__alt_{counter}{extension}"
                    target_path = output_dir / target_name
                shutil.move(str(source_path), str(target_path))
            generated_items.append({"path": str(target_path), "seed": sample_seed, "prompt": prompt, "index": index + 1})

print("__PROJECT_RESULT__=" + json.dumps({"generated_items": generated_items}, ensure_ascii=False), flush=True)
"""

VALIDATE_INLINE_SCRIPT = r"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

payload = json.loads(sys.argv[1])
repo_path = Path(payload["repo_path"]).resolve()
weights_path = Path(payload["weights_path"]).resolve()
checkpoint_dir = Path(payload["checkpoint_dir"]).resolve() if payload.get("checkpoint_dir") else None

if str(repo_path) not in sys.path:
    sys.path.insert(0, str(repo_path))

os.environ["MPLBACKEND"] = "Agg"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(weights_path)

if not weights_path.exists():
    raise FileNotFoundError(f"ACE-Step checkpoints directory not found: {weights_path}")

from acestep.handler import AceStepHandler

handler = AceStepHandler()
status, ok = handler.initialize_service(
    project_root=str(repo_path),
    config_path=f"acestep-v15-{payload['model_variant']}",
    device=payload["device"],
    use_flash_attention=True,
    offload_to_cpu=False,
    offload_dit_to_cpu=False,
)
if not ok:
    raise RuntimeError(f"ACE-Step initialize_service failed: {status}")

load_message = None
toggle_message = None
if checkpoint_dir is not None:
    adapter_config = checkpoint_dir / "adapter_config.json"
    adapter_weights = checkpoint_dir / "adapter_model.safetensors"
    if not adapter_config.exists():
        raise FileNotFoundError(f"LoRA adapter_config.json was not found in {checkpoint_dir}")
    if not adapter_weights.exists():
        raise FileNotFoundError(f"LoRA adapter_model.safetensors was not found in {checkpoint_dir}")
    load_message = handler.load_lora(str(checkpoint_dir))
    toggle_message = handler.set_use_lora(True)

print("__PROJECT_RESULT__=" + json.dumps({
    "service_status": status,
    "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
    "load_message": load_message,
    "toggle_message": toggle_message,
}, ensure_ascii=False), flush=True)
"""


class BackendSubprocessError(RuntimeError):
    def __init__(
        self,
        *,
        stage: str,
        command: list[str],
        return_code: int,
        stderr_tail: list[str],
    ) -> None:
        self.stage = stage
        self.command = command
        self.return_code = return_code
        self.stderr_tail = stderr_tail
        message = (
            f"ACE-Step {stage} failed with return code {return_code}. "
            "See live.log for full merged output."
        )
        super().__init__(message)


@dataclass
class BackendPaths:
    repo_path: Path = ACESTEP_REPO_PATH
    python_executable: Path = ACESTEP_PYTHON_PATH
    weights_path: Path = ACESTEP_CHECKPOINTS_PATH


@dataclass
class BackendMetrics:
    latest_step: int | None = None
    total_steps: int | None = None
    latest_epoch: int | None = None
    total_epochs: int | None = None
    latest_loss: float | None = None
    preprocess_done: int | None = None
    preprocess_total: int | None = None
    last_progress_line: str | None = None
    last_progress_at: float | None = None
    last_process_line: str | None = None
    reported_ffmpeg_extension_warning: bool = False
    suppress_torio_traceback: bool = False


@dataclass
class BackendRunResult:
    stage: str
    command: list[str]
    command_display: str
    return_code: int
    duration_seconds: float
    metrics: BackendMetrics = field(default_factory=BackendMetrics)
    result_payload: dict[str, Any] | None = None
    stderr_tail: list[str] = field(default_factory=list)


@dataclass
class TrainRequest:
    prompt_mode: str
    model_variant: str
    device: str
    precision: str
    processed_dir: Path
    dataset_json_path: Path
    tensor_output_dir: Path
    output_dir: Path
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    epochs: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    save_every_n_epochs: int
    seed: int
    num_workers: int
    optimizer_type: str
    max_duration_seconds: int
    offload_encoder: bool
    manifest_rows: int
    reuse_existing_tensors: bool


@dataclass
class GenerateRequest:
    prompts: list[str]
    prompt_mode: str
    checkpoint_dir: Path | None
    output_dir: Path
    duration: float
    device: str
    seed: int
    model_variant: str


@dataclass
class ValidateRequest:
    checkpoint_dir: Path | None
    device: str
    model_variant: str


def get_backend_paths() -> BackendPaths:
    return BackendPaths()


def ensure_backend_environment(paths: BackendPaths) -> None:
    if not paths.repo_path.exists():
        raise FileNotFoundError(f"ACE-Step repository was not found: {paths.repo_path.resolve()}")
    if not paths.python_executable.exists():
        raise FileNotFoundError(f"ACE-Step backend python was not found: {paths.python_executable.resolve()}")
    if not paths.weights_path.exists() or not any(paths.weights_path.iterdir()):
        raise FileNotFoundError(f"ACE-Step checkpoints directory is missing or empty: {paths.weights_path.resolve()}")


def detect_bf16_support(paths: BackendPaths) -> bool:
    command = [
        str(paths.python_executable),
        "-c",
        "import torch; print('1' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else '0')",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            cwd=str(WORKSPACE_ROOT),
        )
    except OSError:
        return False
    return result.returncode == 0 and result.stdout.strip().endswith("1")


def sanitize_command(command: list[str]) -> str:
    redacted_tokens = ("TOKEN", "SECRET", "PASSWORD", "KEY")
    rendered: list[str] = []
    skip_next = False
    for index, part in enumerate(command):
        if skip_next:
            skip_next = False
            continue
        value = part
        upper = value.upper()
        if value == "-c" and index + 1 < len(command):
            rendered.extend(["-c", "<inline_python_script>"])
            skip_next = True
            continue
        if any(token in upper for token in redacted_tokens) and "=" in value:
            key, _, _ = value.partition("=")
            value = f"{key}=***"
        if index >= 4 and len(value) > 240:
            value = f"<payload:{len(value)} chars>"
        rendered.append(f'"{value}"' if " " in value else value)
    return " ".join(rendered)


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or math.isinf(seconds):
        return "unknown"
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, sec = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _eta_from_progress(done: int | None, total: int | None, elapsed: float) -> str:
    if not done or not total or done <= 0 or total <= done:
        return "unknown" if not total or done != total else "00:00"
    rate = elapsed / done
    return _format_duration(rate * (total - done))


def _update_metrics(metrics: BackendMetrics, line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    changed = False
    step_match = STEP_RE.search(stripped)
    if step_match:
        metrics.latest_step = int(step_match.group(1))
        if step_match.group(2):
            metrics.total_steps = int(step_match.group(2))
        changed = True
    epoch_match = EPOCH_RE.search(stripped)
    if epoch_match:
        metrics.latest_epoch = int(epoch_match.group(1))
        if epoch_match.group(2):
            metrics.total_epochs = int(epoch_match.group(2))
        changed = True
    loss_match = LOSS_RE.search(stripped)
    if loss_match:
        metrics.latest_loss = float(loss_match.group(1))
        changed = True
    processed_match = PROCESSED_RE.search(stripped)
    if processed_match:
        metrics.preprocess_done = int(processed_match.group(1))
        metrics.preprocess_total = int(processed_match.group(2))
        changed = True
    tqdm_match = TQDM_RE.search(stripped)
    if tqdm_match:
        metrics.latest_step = int(tqdm_match.group(2))
        metrics.total_steps = int(tqdm_match.group(3))
        changed = True
    return changed


def _line_level(line: str) -> int:
    lowered = line.lower()
    if any(token in lowered for token in ("error", "traceback", "failed", "exception")):
        return logging.ERROR
    if "warning" in lowered:
        return logging.WARNING
    return logging.INFO


def _should_log_process_line(line: str, changed: bool) -> bool:
    if changed:
        return False
    lowered = line.lower()
    noisy_tokens = (
        "skipping import of cpp extensions",
        "redirects are currently not supported in windows or macos",
        "[progress]",
        "step ",
        "epoch ",
        "loss ",
    )
    if any(token in lowered for token in noisy_tokens):
        return False
    important_tokens = (
        "starting generation",
        "preparing inputs",
        "vram pre-flight",
        "loaded",
        "loading lora",
        "saved",
        "complete",
        "adapter enabled",
        "falling back",
    )
    return any(token in lowered for token in important_tokens) or _line_level(line) >= logging.WARNING


def _is_torio_ffmpeg_noise(line: str) -> bool:
    lowered = line.lower()
    return any(
        token in lowered
        for token in (
            "torio._extension.utils",
            "libtorio_ffmpeg",
            "ffmpeg extension is not available",
            "could not find module",
        )
    )


def _progress_message(metrics: BackendMetrics, elapsed: float) -> str:
    done = metrics.latest_step or metrics.preprocess_done
    total = metrics.total_steps or metrics.preprocess_total
    eta = _eta_from_progress(done, total, elapsed)
    parts = []
    if metrics.latest_epoch is not None:
        epoch = str(metrics.latest_epoch)
        if metrics.total_epochs is not None:
            epoch = f"{epoch}/{metrics.total_epochs}"
        parts.append(f"epoch={epoch}")
    if done is not None:
        parts.append(f"step={done}")
    if total is not None:
        parts.append(f"total={total}")
    parts.append(f"elapsed={_format_duration(elapsed)}")
    parts.append(f"eta={eta}")
    if metrics.latest_loss is not None:
        parts.append(f"loss={metrics.latest_loss:.6g}")
    return "[PROCESS] " + " ".join(parts)


def _directory_progress_message(kind: str, path: Path | None, elapsed: float) -> str:
    if path is None or not path.exists():
        return f"[PROCESS] {kind}=0 size=0MB elapsed={_format_duration(elapsed)}"
    files = [item for item in path.rglob("*") if item.is_file()]
    total_size = sum(item.stat().st_size for item in files)
    size_mb = total_size / (1024 * 1024)
    return f"[PROCESS] {kind}={len(files)} size={size_mb:.2f}MB elapsed={_format_duration(elapsed)}"


def _stream_pipe(
    *,
    pipe: Any,
    source: str,
    logger: logging.Logger,
    metrics: BackendMetrics,
    result_holder: dict[str, Any],
    stderr_tail: deque[str],
    started_at: float,
) -> None:
    buffer = bytearray()

    def emit(decoded: str) -> None:
        stripped = decoded.strip()
        if not stripped:
            return
        if metrics.suppress_torio_traceback:
            traceback_line = stripped.startswith("Traceback") or stripped.startswith("File ") or stripped.startswith("RuntimeError:") or stripped.startswith("FileNotFoundError:")
            if traceback_line:
                if source == "stderr":
                    stderr_tail.append(stripped)
                return
            metrics.suppress_torio_traceback = False
        if _is_torio_ffmpeg_noise(stripped):
            metrics.suppress_torio_traceback = True
            if not metrics.reported_ffmpeg_extension_warning:
                metrics.reported_ffmpeg_extension_warning = True
                logger.warning(
                    "[WARNING] ACE-Step torio FFmpeg bindings are unavailable in the backend environment. "
                    "If audio decoding fails, verify FFmpeg installation and backend dependencies."
                )
            if source == "stderr":
                stderr_tail.append(stripped)
            return
        if stripped.startswith(RESULT_PREFIX):
            payload = stripped[len(RESULT_PREFIX) :]
            try:
                result_holder["payload"] = json.loads(payload)
            except json.JSONDecodeError:
                result_holder["payload_error"] = payload
            return
        changed = _update_metrics(metrics, stripped)
        if _should_log_process_line(stripped, changed) and stripped != metrics.last_process_line:
            metrics.last_process_line = stripped
            logger.log(_line_level(stripped), "[PROCESS] %s", stripped)
        if changed:
            progress_line = _progress_message(metrics, time.perf_counter() - started_at)
            now = time.perf_counter()
            should_emit = progress_line != metrics.last_progress_line and (
                metrics.last_progress_at is None or now - metrics.last_progress_at >= 5.0 or "eta=00:00" in progress_line
            )
            if should_emit:
                metrics.last_progress_line = progress_line
                metrics.last_progress_at = now
                logger.info(progress_line)
        if source == "stderr":
            stderr_tail.append(stripped)

    while True:
        chunk = pipe.read(1)
        if not chunk:
            break
        decoded_chunk = chunk.decode("utf-8", errors="replace")
        if decoded_chunk in {"\n", "\r"}:
            decoded = buffer.decode("utf-8", errors="replace")
            buffer.clear()
            emit(decoded)
        else:
            buffer.extend(chunk)
    if buffer:
        emit(buffer.decode("utf-8", errors="replace"))


def run_backend_command(
    stage: str,
    command: list[str],
    logger: logging.Logger,
    run_id: str,
    *,
    progress_dir: Path | None = None,
    progress_kind: str = "files",
    total_steps: int | None = None,
    total_epochs: int | None = None,
) -> BackendRunResult:
    metrics = BackendMetrics(total_steps=total_steps, total_epochs=total_epochs)
    stderr_tail: deque[str] = deque(maxlen=40)
    result_holder: dict[str, Any] = {}

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["ACESTEP_CACHE_DIR"] = str((WORKSPACE_ROOT / ".cache" / "acestep").resolve())

    started_at = time.perf_counter()
    stop_monitor = threading.Event()
    process = subprocess.Popen(
        command,
        cwd=str(WORKSPACE_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        bufsize=0,
    )

    assert process.stdout is not None
    assert process.stderr is not None

    stdout_thread = threading.Thread(
        target=_stream_pipe,
        kwargs={
            "pipe": process.stdout,
            "source": "stdout",
            "logger": logger,
            "metrics": metrics,
            "result_holder": result_holder,
            "stderr_tail": stderr_tail,
            "started_at": started_at,
        },
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_stream_pipe,
        kwargs={
            "pipe": process.stderr,
            "source": "stderr",
            "logger": logger,
            "metrics": metrics,
            "result_holder": result_holder,
            "stderr_tail": stderr_tail,
            "started_at": started_at,
        },
        daemon=True,
    )

    def _monitor_progress_dir() -> None:
        last_line = ""
        while not stop_monitor.wait(10.0):
            line = _directory_progress_message(progress_kind, progress_dir, time.perf_counter() - started_at)
            if line != last_line:
                logger.info(line)
                last_line = line

    stdout_thread.start()
    stderr_thread.start()
    monitor_thread = threading.Thread(target=_monitor_progress_dir, daemon=True) if progress_dir is not None else None
    if monitor_thread is not None:
        monitor_thread.start()
    stdout_thread.join()
    stderr_thread.join()
    stop_monitor.set()
    if monitor_thread is not None:
        monitor_thread.join(timeout=1.0)

    return_code = process.wait()
    duration_seconds = time.perf_counter() - started_at

    result = BackendRunResult(
        stage=stage,
        command=command,
        command_display=sanitize_command(command),
        return_code=return_code,
        duration_seconds=duration_seconds,
        metrics=metrics,
        result_payload=result_holder.get("payload"),
        stderr_tail=list(stderr_tail),
    )
    if return_code != 0:
        logger.error("[ERROR] ACE-Step %s failed", stage)
        for stderr_line in result.stderr_tail[-40:] or ["<empty>"]:
            logger.error("[ERROR] %s", stderr_line)
        raise BackendSubprocessError(
            stage=stage,
            command=command,
            return_code=return_code,
            stderr_tail=result.stderr_tail[-40:],
        )
    return result


def train(request: TrainRequest, logger: logging.Logger, run_id: str) -> list[BackendRunResult]:
    paths = get_backend_paths()
    ensure_backend_environment(paths)

    results: list[BackendRunResult] = []
    if not request.reuse_existing_tensors:
        results.extend(preprocess(request, logger, run_id))

    train_command = [
        str(paths.python_executable),
        "-u",
        str((paths.repo_path / "train.py").resolve()),
        "--yes",
        "fixed",
        "--checkpoint-dir",
        str(paths.weights_path.resolve()),
        "--model-variant",
        request.model_variant,
        "--dataset-dir",
        str(request.tensor_output_dir.resolve()),
        "--output-dir",
        str(request.output_dir.resolve()),
        "--batch-size",
        str(request.batch_size),
        "--gradient-accumulation",
        str(request.gradient_accumulation_steps),
        "--lr",
        str(request.learning_rate),
        "--epochs",
        str(request.epochs),
        "--precision",
        request.precision,
        "--adapter-type",
        "lora",
        "--rank",
        str(request.lora_rank),
        "--alpha",
        str(request.lora_alpha),
        "--dropout",
        str(request.lora_dropout),
        "--save-every",
        str(request.save_every_n_epochs),
        "--seed",
        str(request.seed),
        "--device",
        request.device,
        "--num-workers",
        str(request.num_workers),
        "--optimizer-type",
        request.optimizer_type,
    ]
    if request.offload_encoder:
        train_command.append("--offload-encoder")
    steps_per_epoch = max(1, math.ceil(request.manifest_rows / max(1, request.batch_size * request.gradient_accumulation_steps)))
    total_steps = max(1, steps_per_epoch * max(1, request.epochs))
    results.append(
        run_backend_command(
            "train",
            train_command,
            logger,
            run_id,
            progress_dir=request.output_dir,
            progress_kind="checkpoint_files",
            total_steps=total_steps,
            total_epochs=request.epochs,
        )
    )
    return results


def preprocess(request: TrainRequest, logger: logging.Logger, run_id: str) -> list[BackendRunResult]:
    paths = get_backend_paths()
    ensure_backend_environment(paths)
    preprocess_command = [
        str(paths.python_executable),
        "-u",
        str((paths.repo_path / "train.py").resolve()),
        "--yes",
        "fixed",
        "--checkpoint-dir",
        str(paths.weights_path.resolve()),
        "--model-variant",
        request.model_variant,
        "--dataset-dir",
        str(request.tensor_output_dir.resolve()),
        "--output-dir",
        str(request.output_dir.resolve()),
        "--preprocess",
        "--audio-dir",
        str(request.processed_dir.resolve()),
        "--dataset-json",
        str(request.dataset_json_path.resolve()),
        "--tensor-output",
        str(request.tensor_output_dir.resolve()),
        "--device",
        request.device,
        "--precision",
        request.precision,
        "--max-duration",
        str(request.max_duration_seconds),
    ]
    return [
        run_backend_command(
            "train_preprocess",
            preprocess_command,
            logger,
            run_id,
            progress_dir=request.tensor_output_dir,
            progress_kind="tensors",
            total_steps=request.manifest_rows,
        )
    ]


def generate(request: GenerateRequest, logger: logging.Logger, run_id: str) -> BackendRunResult:
    paths = get_backend_paths()
    ensure_backend_environment(paths)
    request.output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "repo_path": str(paths.repo_path.resolve()),
        "weights_path": str(paths.weights_path.resolve()),
        "prompt_mode": request.prompt_mode,
        "checkpoint_dir": str(request.checkpoint_dir.resolve()) if request.checkpoint_dir else None,
        "output_dir": str(request.output_dir.resolve()),
        "duration": request.duration,
        "device": request.device,
        "seed": request.seed,
        "model_variant": request.model_variant,
        "prompts": request.prompts,
    }
    command = [str(paths.python_executable), "-u", "-c", GENERATE_INLINE_SCRIPT, json.dumps(payload, ensure_ascii=False)]
    return run_backend_command("generate", command, logger, run_id)


def validate(request: ValidateRequest, logger: logging.Logger, run_id: str) -> BackendRunResult:
    paths = get_backend_paths()
    ensure_backend_environment(paths)

    payload = {
        "repo_path": str(paths.repo_path.resolve()),
        "weights_path": str(paths.weights_path.resolve()),
        "checkpoint_dir": str(request.checkpoint_dir.resolve()) if request.checkpoint_dir else None,
        "device": request.device,
        "model_variant": request.model_variant,
    }
    command = [str(paths.python_executable), "-u", "-c", VALIDATE_INLINE_SCRIPT, json.dumps(payload, ensure_ascii=False)]
    return run_backend_command("validate", command, logger, run_id)


def result_to_dict(result: BackendRunResult) -> dict[str, Any]:
    return asdict(result)
