#!/usr/bin/env python3
"""Capture XLA ``memory_analysis()`` for the E31 batch-8 U-Net update (R, Rg, F).

Each variant is lowered and compiled in its own fresh process with the benchmark
worker's JAX settings, parameters, inputs, and optimizer state. The capture
records the compiled executable's argument, output, alias, and temporary buffer
sizes, the cuDNN convolution scratch buffers in its optimized HLO, and the JAX
allocator statistics after state creation and after compilation. Nothing is
timed, and the executable is not run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "benchmarks" / "e31_gather_reference.yaml"
DEFAULT_WORKLOAD = "unet_b8_update"
DEFAULT_VARIANTS = ("R", "Rg", "F")
ENVIRONMENT_FIELDS = (
    "CUDA_VISIBLE_DEVICES",
    "JAX_PLATFORMS",
    "LD_LIBRARY_PATH",
    "NVIDIA_TF32_OVERRIDE",
    "XLA_FLAGS",
    "XLA_PYTHON_CLIENT_ALLOCATOR",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
)
# A cuDNN convolution custom call returns (result, u8[scratch_bytes]).
_CUDNN_CONVOLUTION = re.compile(
    r"=\s*\((?P<shapes>[^)]*)\)\s*custom-call\(.*custom_call_target=\"__cudnn\$conv"
)
_SCRATCH = re.compile(r"u8\[(\d+)\]")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=REPOSITORY_ROOT / "protocol.yaml")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workload-id", default=DEFAULT_WORKLOAD)
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--session", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--single-variant", help=argparse.SUPPRESS)
    return parser


def _cudnn_scratch(hlo_text: str) -> dict[str, int]:
    sizes = []
    for line in hlo_text.splitlines():
        match = _CUDNN_CONVOLUTION.search(line)
        if match:
            scratch = _SCRATCH.findall(match.group("shapes"))
            sizes.append(int(scratch[-1]) if scratch else 0)
    return {
        "cudnn_convolution_calls": len(sizes),
        "cudnn_scratch_bytes_sum": sum(sizes),
        "cudnn_scratch_bytes_max": max(sizes, default=0),
    }


def _allocator_stats(device: Any) -> dict[str, int | None]:
    stats = device.memory_stats() or {}
    return {
        field: stats.get(field)
        for field in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")
    }


def _capture_variant(arguments: argparse.Namespace, variant: str) -> dict[str, Any]:
    """Compile one variant in the current process, mirroring benchmark_worker."""

    import jax

    from poincare_exec.config.loader import config_hash, load_yaml
    from poincare_exec.experiments.benchmark import _operator_state

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    protocol = load_yaml(arguments.protocol)
    config = load_yaml(arguments.config)
    workload = next(
        item for item in config["workloads"] if item["id"] == arguments.workload_id
    )
    seed = int(protocol["seeds"]["measurement_order"]) + arguments.session
    device = jax.devices()[0]
    function, state, inputs, curvature = _operator_state(workload, variant, seed, protocol)
    jax.block_until_ready((state, inputs, curvature))
    after_state = _allocator_stats(device)
    compiled = jax.jit(function).lower(state, inputs, curvature).compile()
    after_compile = _allocator_stats(device)
    analysis = compiled.memory_analysis()
    compiled_bytes = {
        name: getattr(analysis, f"{name}_size_in_bytes", None)
        for name in ("argument", "output", "alias", "temp", "generated_code")
    }
    return {
        "variant": variant,
        "workload_id": workload["id"],
        "config_hash": config_hash(protocol, config, workload),
        "seed": seed,
        "device_kind": device.device_kind,
        "compiled_bytes": compiled_bytes,
        "optimized_hlo": _cudnn_scratch(compiled.as_text()),
        "allocator_after_state": after_state,
        "allocator_after_compile": after_compile,
    }


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.single_variant is not None:
        payload = _capture_variant(arguments, arguments.single_variant)
        arguments.output.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return

    import jax
    import jaxlib

    from poincare_exec.artifacts.provenance import git_provenance, timestamp_utc

    captures = []
    with tempfile.TemporaryDirectory(prefix="e31-memory-") as directory:
        for variant in arguments.variants:
            output = Path(directory) / f"{variant}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--protocol",
                str(arguments.protocol),
                "--config",
                str(arguments.config),
                "--workload-id",
                arguments.workload_id,
                "--session",
                str(arguments.session),
                "--single-variant",
                variant,
                "--output",
                str(output),
            ]
            subprocess.run(command, check=True)
            captures.append(json.loads(output.read_text(encoding="utf-8")))
    payload = {
        "analysis_id": "e31_compiled_memory_analysis",
        "scope": (
            "compile-time XLA buffer assignment of the jitted benchmark function; "
            "temp bytes include saved activations and cuDNN scratch; allocator "
            "statistics after compilation include autotuning allocations; the "
            "executable is not run"
        ),
        "captured_at_utc": timestamp_utc(),
        "git": git_provenance(REPOSITORY_ROOT),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "environment": {name: os.environ.get(name) for name in ENVIRONMENT_FIELDS},
        "workload_id": arguments.workload_id,
        "session": arguments.session,
        "captures": captures,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
