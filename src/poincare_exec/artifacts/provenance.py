"""Machine-readable environment and per-row provenance capture."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import jax
import jaxlib

from poincare_exec.config.constants import REPOSITORY_ROOT
from poincare_exec.config.loader import canonical_json


def timestamp_utc() -> str:
    """Return a timezone-explicit ISO 8601 timestamp."""

    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _command(command: list[str], *, cwd: Path | None = None) -> dict[str, Any]:
    """Capture one diagnostic command without invoking a shell."""

    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": type(error).__name__}
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


@lru_cache(maxsize=4)
def git_provenance(repository: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    """Capture commit and dirty status without mutating the repository."""

    revision = _command(["git", "rev-parse", "HEAD"], cwd=repository)
    status = _command(["git", "status", "--porcelain"], cwd=repository)
    if status.get("returncode") != 0:
        raise RuntimeError(f"could not capture git status: {status.get('stderr')}")
    return {
        "commit": revision.get("stdout") or None,
        "dirty": bool(status.get("stdout")),
        "status_porcelain": status.get("stdout", "").splitlines(),
    }


def device_identity() -> str:
    """Return a stable descriptive identity for the selected JAX device."""

    device = jax.devices()[0]
    identifier = getattr(device, "id", "unknown")
    kind = getattr(device, "device_kind", "unknown")
    return f"{platform.node()}::{device.platform}:{identifier}::{kind}"


def _installed_packages() -> dict[str, str]:
    selected_prefixes = (
        "jax",
        "numpy",
        "scipy",
        "nvidia-",
        "matplotlib",
        "pillow",
        "pyyaml",
    )
    packages = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name", "").lower()
        if name.startswith(selected_prefixes):
            packages[name] = distribution.version
    return dict(sorted(packages.items()))


def capture_environment() -> dict[str, Any]:
    """Capture the complete environment fields needed to interpret a result."""

    devices = []
    for device in jax.devices():
        memory = device.memory_stats()
        devices.append(
            {
                "id": getattr(device, "id", None),
                "platform": device.platform,
                "device_kind": getattr(device, "device_kind", None),
                "memory_stats": memory,
            }
        )
    relevant_environment = {
        name: os.environ.get(name)
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "JAX_PLATFORMS",
            "JAX_ENABLE_X64",
            "LD_LIBRARY_PATH",
            "NVIDIA_TF32_OVERRIDE",
            "XLA_FLAGS",
            "XLA_PYTHON_CLIENT_ALLOCATOR",
            "XLA_PYTHON_CLIENT_MEM_FRACTION",
            "XLA_PYTHON_CLIENT_PREALLOCATE",
        )
    }
    return {
        "captured_at_utc": timestamp_utc(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "jax": {
            "version": jax.__version__,
            "jaxlib_version": jaxlib.__version__,
            "backend": jax.default_backend(),
            "x64_enabled": bool(jax.config.x64_enabled),
            "devices": devices,
        },
        "packages": _installed_packages(),
        "environment": relevant_environment,
        "git": git_provenance(),
        "nvidia_smi": _command(["nvidia-smi", "-q"]),
        "nvcc": _command(["nvcc", "--version"]),
        "lscpu": _command(["lscpu"]),
    }


def precision_policy(dtype: str) -> str:
    """Serialize the fixed arithmetic policy for result rows."""

    return canonical_json(
        {
            "storage_dtype": dtype,
            "matmul_precision": "highest",
            "reduced_precision_reduction": False,
            "tf32": False,
        }
    )


def base_result_row(
    protocol: dict[str, Any],
    *,
    experiment_id: str,
    variant: str,
    configuration_hash: str,
    dtype: str,
    seed_or_session: int | str,
    status: str = "ok",
) -> dict[str, Any]:
    """Construct the mandatory provenance prefix for any raw result row."""

    git = git_provenance()
    references = protocol["reference_commits"]
    return {
        "experiment_id": experiment_id,
        "protocol_version": protocol["protocol_version"],
        "variant": variant,
        "config_hash": configuration_hash,
        "code_commit": git["commit"],
        "code_dirty": git["dirty"],
        "reference_commit": canonical_json(references),
        "device_id": device_identity(),
        "precision_policy": precision_policy(dtype),
        "seed_or_session": seed_or_session,
        "status": status,
        "timestamp_utc": timestamp_utc(),
    }


def dumps_pretty(value: Any) -> str:
    """Serialize a deterministic human-readable JSON document."""

    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
