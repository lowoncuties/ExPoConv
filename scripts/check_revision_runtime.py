#!/usr/bin/env python3
"""Fail closed when a reviewer-requested benchmark uses the wrong runtime."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any


class RuntimeValidationError(ValueError):
    """Raised when the active JAX runtime does not match the planned cohort."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-jax", required=True)
    parser.add_argument("--expected-jaxlib", required=True)
    parser.add_argument("--expected-device-substring", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _validate_runtime(
    *,
    jax_version: str,
    jaxlib_version: str,
    backend: str,
    devices: Sequence[dict[str, Any]],
    expected_jax: str,
    expected_jaxlib: str,
    expected_device_substring: str,
) -> None:
    if jax_version != expected_jax:
        raise RuntimeValidationError(f"expected JAX {expected_jax}, found {jax_version}")
    if jaxlib_version != expected_jaxlib:
        raise RuntimeValidationError(
            f"expected jaxlib {expected_jaxlib}, found {jaxlib_version}"
        )
    if backend != "gpu":
        raise RuntimeValidationError(f"expected JAX GPU backend, found {backend}")
    if len(devices) != 1:
        raise RuntimeValidationError(
            f"expected exactly one JAX-visible device, found {len(devices)}"
        )
    device = devices[0]
    if device.get("platform") != "gpu":
        raise RuntimeValidationError(
            f"expected a GPU device, found {device.get('platform')!r}"
        )
    device_kind = str(device.get("device_kind", ""))
    if expected_device_substring.casefold() not in device_kind.casefold():
        raise RuntimeValidationError(
            f"expected device containing {expected_device_substring!r}, "
            f"found {device_kind!r}"
        )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> None:
    arguments = _parser().parse_args()

    import jax
    import jaxlib

    devices = [
        {
            "id": device.id,
            "platform": device.platform,
            "device_kind": device.device_kind,
        }
        for device in jax.devices()
    ]
    payload = {
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "default_backend": jax.default_backend(),
        "devices": devices,
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "JAX_PLATFORMS",
                "LD_LIBRARY_PATH",
                "NVIDIA_TF32_OVERRIDE",
                "XLA_FLAGS",
                "XLA_PYTHON_CLIENT_ALLOCATOR",
                "XLA_PYTHON_CLIENT_MEM_FRACTION",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
            )
        },
    }
    _validate_runtime(
        jax_version=payload["jax_version"],
        jaxlib_version=payload["jaxlib_version"],
        backend=payload["default_backend"],
        devices=devices,
        expected_jax=arguments.expected_jax,
        expected_jaxlib=arguments.expected_jaxlib,
        expected_device_substring=arguments.expected_device_substring,
    )

    value = (
        jax.numpy.ones((1024, 1024), dtype=jax.numpy.float32)
        @ jax.numpy.ones((1024, 1024), dtype=jax.numpy.float32)
    ).block_until_ready()
    if float(value[0, 0]) != 1024.0:
        raise RuntimeValidationError("compiled GPU smoke result was incorrect")
    payload["compiled_gpu_smoke"] = "passed"
    _atomic_json(arguments.output, payload)


if __name__ == "__main__":
    main()
