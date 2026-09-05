from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.workloads import WORKLOADS, WORKLOAD_NAMES


def run_command(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip()
    except FileNotFoundError:
        return "unavailable"


def get_driver_version() -> str:
    output = run_command(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ]
    )

    if output == "unavailable":
        return output

    return output.splitlines()[0].strip()


def get_gpu_name() -> str:
    if not torch.cuda.is_available():
        return "cpu"

    return torch.cuda.get_device_name(0)


def sha256_file(path: str | Path) -> str:
    file_path = Path(path)

    if not file_path.exists():
        return "unavailable"

    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def get_git_commit() -> str:
    return run_command(["git", "rev-parse", "HEAD"])


def get_tensorrt_version() -> str:
    try:
        import tensorrt as trt

        return trt.__version__
    except Exception:
        return "unavailable"


def environment_fingerprint(environment: dict) -> str:
    comparable_keys = (
        "workloads", "gpu_name", "gpu_count", "driver",
        "cuda_runtime", "tensorrt", "torch", "python", "container_image",
    )
    comparable = {key: environment.get(key, "unavailable") for key in comparable_keys}
    return hashlib.sha256(json.dumps(comparable, sort_keys=True).encode()).hexdigest()[:16]


def collect_environment(
    model_path: str | None = None,
    engine_path: str | None = None,
) -> dict:
    environment = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "gpu_name": get_gpu_name(),
        "gpu_count": torch.cuda.device_count(),
        "driver": get_driver_version(),
        "cuda_runtime": torch.version.cuda or "unavailable",
        "torch": torch.__version__,
        "tensorrt": get_tensorrt_version(),
        "workloads": [
            {
                "name": name,
                "model_name": WORKLOADS[name].model_name,
                "revision": WORKLOADS[name].revision,
            }
            for name in WORKLOAD_NAMES
        ],
        "python": os.sys.version.split()[0],
        "git_commit": get_git_commit(),
        "model_sha256": sha256_file(model_path) if model_path else "unavailable",
        "engine_sha256": sha256_file(engine_path) if engine_path else "unavailable",
    }
    environment["fingerprint"] = environment_fingerprint(environment)
    return environment


def write_environment(
    output_path: str | Path,
    model_path: str | None = None,
    engine_path: str | None = None,
) -> dict:
    environment = collect_environment(model_path, engine_path)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_file.write_text(
        json.dumps(environment, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return environment
